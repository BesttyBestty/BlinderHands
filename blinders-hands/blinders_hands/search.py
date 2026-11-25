"""主动视觉搜索：机械臂持物翻转，看到结果再决定下一步怎么动。

两层决策：
1. 视角内微调（refine_joints）：这一面有字但偏了、歪了 → 按偏移和倾角微调腕部，直到能读
2. 视角间规划（ViewPlanner）：这一面没有要找的信息 → 根据已经看到的情况，选出最有希望的下一个视角
   - 文字出现在画面边缘：信息面在那个方向，优先往那边转
   - 这一面完全没有字：附近的视角大概率也是空白面，优先去离已看过视角最远的地方
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2

from .arm import Arm, clamp_joints
from .camera import Camera
from .dates import DateInfo, parse_dates
from .poses import PoseLibrary, RefineConfig, View
from .vision.recognizer import Recognition, Recognizer
from .vision.text_detector import TextDetector, ViewAssessment, assess_view

log = logging.getLogger(__name__)

Progress = Callable[[str], None]


@dataclass
class SearchStep:
    view: str
    refine: int
    joints: list[float] | None
    assessment: str
    image: str = ""
    recognized: dict | None = None


@dataclass
class SearchResult:
    success: bool
    task: str
    recognition: Recognition | None = None
    dates: DateInfo | None = None
    steps: list[SearchStep] = field(default_factory=list)
    cloud_calls: int = 0
    reason: str = ""


def task_satisfied(task: str, rec: Recognition) -> tuple[bool, DateInfo | None]:
    f = rec.fields or {}
    if task == "check_expiry":
        info = DateInfo()
        if f:
            labeled = (f"生产日期:{f.get('production_date', '')} 有效期至:{f.get('expiry', '')} "
                       f"保质期:{f.get('shelf_life', '')} {f.get('raw_text', '')}")
            info = parse_dates(labeled)
        if not info.found:
            info = parse_dates(rec.raw)
        return info.found, info
    if task == "read_usage":
        return bool(str(f.get("usage", "")).strip()), None
    if task == "ask":
        return bool(f.get("answered")) and bool(str(f.get("answer", "")).strip()), None
    return bool(str(f.get("name", "")).strip()), None


def refine_joints(current: list[float], a: ViewAssessment, cfg: RefineConfig) -> list[float] | None:
    """根据文字在画面中的偏移和倾斜，算出下一步的关节角。返回 None 表示不需要动。"""
    def step(x: float) -> float:
        return max(-cfg.max_step, min(cfg.max_step, x))

    new = list(current)
    moved = False
    dx, dy = a.center_offset
    if abs(a.skew_deg) > 8:
        new[cfg.roll_joint] += step(-cfg.roll_sign * a.skew_deg * 3.14159 / 180.0)
        moved = True
    if abs(dx) > 0.25:
        new[cfg.yaw_joint] += step(cfg.yaw_gain * dx)
        moved = True
    if abs(dy) > 0.25:
        new[cfg.pitch_joint] += step(cfg.pitch_gain * dy)
        moved = True
    if not moved:
        return None
    clamped = clamp_joints(new)
    return clamped if any(abs(a - b) > 1e-4 for a, b in zip(clamped, current)) else None


class ViewPlanner:
    """在观察姿态库上做“下一个最佳视角”选择。

    候选视角 = 姿态库里的视角 + 每个视角沿扫描关节（默认腕部偏航）展开的若干偏移。
    每个候选用扫描关节的角度表示“它朝向物体的哪一面”。
    """

    def __init__(self, poses: PoseLibrary):
        self.cfg = poses.refine
        self.sweep_joint = poses.sweep_joint if poses.sweep_joint is not None else self.cfg.yaw_joint
        self.candidates: list[View] = []
        seen: set[tuple] = set()
        for v in poses.views:
            for off in [0.0] + list(poses.sweep_offsets):
                j = list(v.joints)
                j[self.sweep_joint] += off
                j = clamp_joints(j)
                key = tuple(round(x, 2) for x in j)
                if key in seen:
                    continue
                seen.add(key)
                name = v.name if off == 0 else f"{v.name}{'+' if off > 0 else ''}{off:.2f}"
                self.candidates.append(View(name, j))
        self.prior = {id(c): 1.0 - i * 0.01 for i, c in enumerate(self.candidates)}  # 姿态库里的顺序作为先验
        self.bonus: dict[int, float] = {id(c): 0.0 for c in self.candidates}
        self.visited: list[View] = []

    def _angle(self, v: View) -> float:
        return v.joints[self.sweep_joint]

    def remaining(self) -> int:
        return len(self.candidates) - len(self.visited)

    def next(self) -> View | None:
        left = [c for c in self.candidates if all(c is not v for v in self.visited)]
        if not left:
            return None

        def score(c: View) -> float:
            coverage = min((abs(self._angle(c) - self._angle(v)) for v in self.visited), default=0.0)
            return self.prior[id(c)] + self.bonus[id(c)] + 0.5 * coverage

        return max(left, key=score)

    def observe(self, view: View, a: ViewAssessment | None) -> None:
        if all(view is not v for v in self.visited):
            self.visited.append(view)
        base = self._angle(view)
        for c in self.candidates:
            d = self._angle(c) - base
            if a is None or not a.has_text:
                # 空白面：离得近的视角也多半是空白面
                if abs(d) < 0.5:
                    self.bonus[id(c)] -= 0.3
            else:
                # 文字在画面一侧：往能把那一侧转过来的方向找
                want = self.cfg.yaw_gain * a.center_offset[0]
                if want * d > 0:
                    self.bonus[id(c)] += 0.6 * min(1.0, abs(a.center_offset[0]) * 1.5) / (1.0 + abs(d))


class ActiveSearcher:
    def __init__(self, arm: Arm, camera: Camera, detector: TextDetector, recognizer: Recognizer,
                 poses: PoseLibrary, capture_dir: Path | None = None, max_refine_steps: int = 2,
                 max_cloud_calls: int = 4, min_sharpness: float = 80.0, progress: Progress | None = None,
                 max_views: int = 8, question: str = "", target: str = ""):
        self.arm = arm
        self.camera = camera
        self.detector = detector
        self.recognizer = recognizer
        self.poses = poses
        self.capture_dir = capture_dir
        self.max_refine_steps = max_refine_steps
        self.max_cloud_calls = max_cloud_calls
        self.min_sharpness = min_sharpness
        self.progress = progress or (lambda msg: None)
        self.max_views = max_views
        self.question = question
        self.target = target
        if capture_dir:
            capture_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, frame, view_idx: int, refine: int) -> str:
        if not self.capture_dir:
            return ""
        path = self.capture_dir / f"{time.strftime('%H%M%S')}_v{view_idx:02d}_r{refine}.jpg"
        cv2.imwrite(str(path), frame)
        return str(path)

    def _recognize(self, frame, task: str) -> Recognition:
        return self.recognizer.recognize(frame, task, question=self.question, target=self.target)

    def run(self, task: str) -> SearchResult:
        result = SearchResult(False, task)
        if not self.poses.views:
            result.reason = "姿态库是空的，请先用 tools/calibrate.py 标定观察姿态"
            return result

        planner = ViewPlanner(self.poses)
        total = min(self.max_views, len(planner.candidates))
        vi = 0
        while vi < total:
            view = planner.next()
            if view is None:
                break
            vi += 1
            self.progress(f"转到第{vi}个角度")
            joints = self.arm.move_joints(view.joints)
            blur_retry = False
            last: ViewAssessment | None = None

            for r in range(self.max_refine_steps + 1):
                frame = self.camera.capture()
                a = assess_view(frame, self.detector, min_sharpness=self.min_sharpness)
                last = a
                step = SearchStep(view.name, r, list(joints), a.reason, self._save(frame, vi, r))
                result.steps.append(step)
                log.info("[%s #%d] %s（文字占比 %.3f，偏移 %.2f/%.2f，倾斜 %.1f°，清晰度 %.0f）",
                         view.name, r, a.reason, a.text_area_ratio, *a.center_offset, a.skew_deg, a.sharpness)

                if not a.has_text:
                    break

                new_joints = None if a.readable else refine_joints(joints, a, self.poses.refine)
                if a.readable or r == self.max_refine_steps or (new_joints is None and a.reason != "文字模糊"):
                    if result.cloud_calls >= self.max_cloud_calls:
                        result.reason = f"已识别{result.cloud_calls}次，达到上限"
                        return result
                    self.progress("这一面有字，正在读")
                    rec = self._recognize(frame, task)
                    result.cloud_calls += 1
                    step.recognized = rec.fields
                    ok, info = task_satisfied(task, rec)
                    if ok:
                        result.success, result.recognition, result.dates = True, rec, info
                        return result
                    result.recognition = rec  # 保留最后一次读到的内容（比如药名）
                    break

                # 只是模糊、位置和角度都还行：可能是刚停下来的抖动，原地重拍一次
                if a.reason == "文字模糊" and not blur_retry:
                    blur_retry = True
                    time.sleep(0.5)
                    continue

                if new_joints is None:
                    continue
                self.progress("看到字了，调整一下角度")
                joints = self.arm.move_joints(new_joints)

            planner.observe(view, last)

        result.reason = "所有角度都没有找到需要的信息"
        return result
