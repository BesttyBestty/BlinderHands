"""完整任务编排：找到物体 → 抓起来 → 主动搜索关键信息 → 判断 → 放回或递给使用者 → 一句人话。

支持的任务：
- check_expiry 查有效期       - read_usage 读用法用量     - identify 认出是什么
- ask 回答关于物品的任意问题  - find 告诉使用者东西在哪   - scan_box 把药箱逐一检查并记录
- expiring 说出哪些药快过期
另外：guide_step() 一步步引导使用者的手摸到目标物；stop() 急停；release() 递物时松开夹爪。
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .arm import Arm, ArmError, EmergencyStop, MockArm, NeroCanArm, NeroHttpArm
from .camera import Camera, CameraError, ImageSequenceCamera, OpenCVCamera
from .config import Settings
from .dates import judge_expiry
from .grasp import (GraspConfig, GraspPlan, RealSenseCamera, StaticRGBDCamera, estimate_object, execute_grasp,
                    load_handeye, place_back, plan_grasp, synthetic_scene)
from .guidance import GuideStep, guide_instruction
from .intent import Intent
from .inventory import Inventory
from .poses import ObjectConfig, PoseLibrary, load_objects, load_poses
from .search import ActiveSearcher, SearchResult
from .vision.recognizer import QwenVLRecognizer, Recognizer
from .vision.text_detector import TextDetector, build_detector

log = logging.getLogger(__name__)

Progress = Callable[[str], None]
BBox = tuple[float, float, float, float]

HAND = "使用者的手"


def direction_phrase(bbox: BBox) -> str:
    """全局相机画面 → 使用者能听懂的方位。约定：相机从使用者身后俯视，画面下方离使用者近。"""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    lr = "左边" if cx < 0.36 else "右边" if cx > 0.64 else "正前方"
    fb = "靠里" if cy < 0.4 else "靠近您" if cy > 0.62 else ""
    return f"在您{lr}{('、' + fb) if fb else ''}"


def pick_slot(objects: ObjectConfig, bbox: BBox | None) -> str:
    if bbox:
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        for s in objects.slots:
            x1, y1, x2, y2 = s.region
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return s.teach
    return objects.default_teach


@dataclass
class TaskOutcome:
    success: bool
    message: str
    intent: Intent
    location: str = ""
    search: SearchResult | None = None
    log: list[str] = field(default_factory=list)


def compose_answer(intent: Intent, result: SearchResult) -> tuple[bool, str]:
    name = ""
    if result.recognition:
        name = str(result.recognition.fields.get("name", "")).strip()
    label = name or intent.target
    if not result.success:
        wanted = {"check_expiry": "有效期", "read_usage": "用法用量", "ask": "能回答您问题的信息"}.get(intent.task, "名称")
        return False, f"抱歉，我转了几个角度，没有找到{label}的{wanted}。可以换个方向放进来，我再试一次。"
    f = result.recognition.fields
    if intent.task == "check_expiry":
        verdict = judge_expiry(result.dates)
        return verdict.status in {"valid", "expiring_soon"}, f"{label}{verdict.message}"
    if intent.task == "read_usage":
        msg = f"{label}的用法用量是：{f.get('usage', '')}。"
        if str(f.get("warnings", "")).strip():
            msg += f"注意：{f['warnings']}。"
        return True, msg
    if intent.task == "ask":
        answer = str(f.get("answer", "")).strip().rstrip("。")
        evidence = str(f.get("evidence", "")).strip()
        quote = f"包装上写着：{evidence}。" if evidence and evidence not in answer else ""
        return True, f"{answer}。{quote}"
    category = str(f.get("category", "")).strip()
    return True, f"这是{name}{('，' + category) if category else ''}。"


class _Held:
    """当前手里拿着的东西是怎么拿起来的，决定之后怎么放回去。"""

    def __init__(self, plan: GraspPlan | None = None, teach: str = ""):
        self.plan = plan
        self.teach = teach


class Assistant:
    def __init__(self, settings: Settings, arm: Arm, hand_camera: Camera, scene_camera, detector: TextDetector,
                 recognizer: Recognizer, poses: PoseLibrary, objects: ObjectConfig, grasp_cfg: GraspConfig | None = None,
                 T_base_cam: np.ndarray | None = None, inventory: Inventory | None = None):
        self.settings = settings
        self.arm = arm
        self.hand_camera = hand_camera
        self.scene_camera = scene_camera
        self.detector = detector
        self.recognizer = recognizer
        self.poses = poses
        self.objects = objects
        self.grasp_cfg = grasp_cfg or GraspConfig()
        self.T_base_cam = T_base_cam
        self.inventory = inventory or Inventory(settings.inventory_file)
        self._release = threading.Event()
        self.state = "idle"  # idle | working | waiting_release | stopped

    # ---- 外部控制 ----------------------------------------------------------

    def stop(self) -> None:
        self.state = "stopped"
        self.arm.stop()
        self._release.set()

    def resume(self) -> None:
        if hasattr(self.arm, "resume"):
            self.arm.resume()
        self.state = "idle"

    def release(self) -> None:
        self._release.set()

    @property
    def can_vision_grasp(self) -> bool:
        return (self.T_base_cam is not None and hasattr(self.scene_camera, "capture_rgbd")
                and getattr(self.arm, "supports_pose", False))

    def speak(self, text: str) -> None:
        if self.settings.speak_on_server and sys.platform == "darwin":
            subprocess.run(["say", text], check=False)

    # ---- 基本动作 ----------------------------------------------------------

    def _scene(self):
        if self.scene_camera is None:
            return None
        if hasattr(self.scene_camera, "capture_rgbd"):
            return self.scene_camera.capture_rgbd()
        return self.scene_camera.capture()

    @staticmethod
    def _color(scene) -> np.ndarray:
        return scene.color if hasattr(scene, "color") else scene

    def _pick(self, scene, bbox: BBox | None, report: Progress) -> _Held:
        if self.can_vision_grasp and bbox is not None and hasattr(scene, "depth"):
            est = estimate_object(scene, bbox, self.T_base_cam)
            plan = plan_grasp(est, self.grasp_cfg)
            report(f"我来把它拿起来，它大约宽{est.width * 100:.0f}厘米、高{est.height * 100:.0f}厘米")
            execute_grasp(self.arm, plan, self.grasp_cfg)
            return _Held(plan=plan)
        teach = pick_slot(self.objects, bbox)
        if teach:
            report("我来把它拿起来")
            self.arm.run_teach(teach)
            return _Held(teach=teach)
        report("请把它放到机械臂的夹爪前面")
        return _Held()

    def _finish(self, held: _Held, handover: bool, report: Progress, on_state: Callable[[str], None]) -> None:
        if handover and getattr(self.arm, "supports_pose", False):
            report("我把它递到您手边")
            self.arm.move_pose(self.grasp_cfg.handover_pose)
            self.state = "waiting_release"
            on_state("waiting_release")
            self._release.clear()
            if not self._release.wait(self.settings.handover_timeout):
                report("您没有接，我先把它放回原处")
            else:
                if self.state == "stopped":
                    return
                self.arm.gripper(self.grasp_cfg.gripper_max_width)
                self.state = "working"
                on_state("working")
                return
        if held.plan is not None:
            place_back(self.arm, held.plan)
        elif self.objects.place_back_teach:
            self.arm.run_teach(self.objects.place_back_teach)

    def _inspect(self, intent: Intent, report: Progress) -> SearchResult:
        searcher = ActiveSearcher(
            self.arm, self.hand_camera, self.detector, self.recognizer, self.poses,
            capture_dir=self.settings.capture_dir, max_refine_steps=self.settings.max_refine_steps,
            max_cloud_calls=self.settings.max_cloud_calls, min_sharpness=self.settings.min_sharpness,
            progress=report, max_views=self.settings.max_views, question=intent.question, target=intent.target)
        return searcher.run(intent.task)

    def _record(self, result: SearchResult, location: str = "") -> None:
        if result.success and result.dates and result.dates.expiry and result.recognition:
            name = str(result.recognition.fields.get("name", "")).strip()
            self.inventory.record(name, result.dates.expiry, result.dates.expiry_is_estimated, location)

    # ---- 任务 --------------------------------------------------------------

    def run(self, intent: Intent, progress: Progress | None = None,
            on_state: Callable[[str], None] | None = None) -> TaskOutcome:
        logs: list[str] = []
        on_state = on_state or (lambda s: None)

        def report(msg: str) -> None:
            logs.append(msg)
            log.info(msg)
            if progress:
                progress(msg)

        self.state = "working"
        try:
            outcome = self._run(intent, report, on_state)
        except EmergencyStop:
            outcome = TaskOutcome(False, "已经停下了。确认安全后，说“继续”我再接着用。", intent)
        except ArmError as exc:
            outcome = TaskOutcome(False, f"机械臂出了问题：{exc}", intent)
        except CameraError as exc:
            outcome = TaskOutcome(False, f"摄像头出了问题：{exc}", intent)
        finally:
            if self.state != "stopped":
                self.state = "idle"
        outcome.log = logs
        self.speak(outcome.message)
        return outcome

    def _run(self, intent: Intent, report: Progress, on_state: Callable[[str], None]) -> TaskOutcome:
        if intent.task == "expiring":
            return TaskOutcome(True, self.inventory.summary(), intent)
        if intent.task == "scan_box":
            return self._scan_box(intent, report)

        # 1. 找：全局相机看桌面，告诉使用者东西在哪
        location, bbox, scene = "", None, None
        if self.scene_camera is not None:
            report(f"正在桌上找{intent.target}")
            scene = self._scene()
            bbox = self.recognizer.locate(self._color(scene), intent.target)
            if bbox:
                location = direction_phrase(bbox)
                report(f"{intent.target}{location}")
            elif intent.task == "find":
                return TaskOutcome(False, f"桌上没有看到{intent.target}。", intent)

        if intent.task == "find":
            if not location:
                return TaskOutcome(False, "我需要连接桌面相机才能帮您找东西。", intent)
            return TaskOutcome(True, f"{intent.target}{location}。",
                               intent, location)

        # 2. 抓 → 3. 转着找 → 4. 放回或递给使用者
        held = self._pick(scene, bbox, report)
        result = self._inspect(intent, report)
        self._record(result, location)
        ok, message = compose_answer(intent, result)
        self._finish(held, intent.handover, report, on_state)
        return TaskOutcome(ok, message, intent, location, result)

    def _scan_box(self, intent: Intent, report: Progress) -> TaskOutcome:
        if self.scene_camera is None:
            return TaskOutcome(False, "整理药箱需要连接桌面相机。", intent)
        scene = self._scene()
        items = self.recognizer.list_objects(self._color(scene))
        if not items:
            return TaskOutcome(False, "桌上没有看到药盒或药瓶。", intent)
        report(f"桌上有{len(items)}样药，我一样一样看")
        checked, failed = 0, []
        for i, (name, bbox) in enumerate(items, start=1):
            report(f"第{i}样{('：' + name) if name else ''}")
            scene = self._scene()  # 上一样放回后画面可能变了
            held = self._pick(scene, bbox, report)
            sub = Intent(name or f"第{i}样药", "check_expiry", "")
            result = self._inspect(sub, report)
            self._record(result, direction_phrase(bbox))
            if result.success:
                checked += 1
                report(compose_answer(sub, result)[1])
            else:
                failed.append(name or f"第{i}样")
            self._finish(held, False, report, lambda s: None)
        msg = f"看完了，{checked}样记下了有效期。"
        if failed:
            msg += "、".join(failed) + "没有找到有效期。"
        return TaskOutcome(checked > 0, msg + self.inventory.summary(), intent)

    def guide_step(self, target: str) -> GuideStep:
        if self.scene_camera is None:
            return GuideStep("引导需要连接桌面相机。", True)
        found = self.recognizer.locate_many(self._color(self._scene()), [target, HAND])
        return guide_instruction(found.get(HAND), found.get(target))


def build_assistant(settings: Settings | None = None) -> Assistant:
    settings = settings or Settings()
    detector = build_detector(settings.detector_backend, settings.rdk_det_model)
    grasp_cfg = GraspConfig.load(settings.grasp_file)
    if settings.mock:
        from .demo import DemoRecognizer, make_demo_frames

        scene_frame, T, (bbox, bbox2) = synthetic_scene([(0.40, 0.10, 0.09, 0.045, 0.06, 0.5),
                                                         (0.48, -0.15, 0.10, 0.05, 0.04, -0.3)])
        arm: Arm = MockArm()
        hand: Camera = ImageSequenceCamera(make_demo_frames())
        scene = StaticRGBDCamera(scene_frame)
        recognizer: Recognizer = DemoRecognizer(bbox, [("布洛芬缓释胶囊", bbox), ("阿莫西林胶囊", bbox2)])
        poses = load_poses(settings.poses_file, allow_uncalibrated=True)
        inventory = Inventory(settings.inventory_file)
        return Assistant(settings, arm, hand, scene, detector, recognizer, poses,
                         load_objects(settings.objects_file), grasp_cfg, T, inventory)

    if settings.arm_backend == "can":
        arm = NeroCanArm(settings.can_channel, settings.can_interface, settings.nero_firmware,
                         settings.speed_percent, settings.crash_rating)
    else:
        arm = NeroHttpArm(settings.arm_base_url, settings.arm_token, settings.arm_move_wait, settings.arm_grasp_wait)
    hand = OpenCVCamera(settings.hand_camera_index)
    if settings.depth_camera == "realsense":
        scene = RealSenseCamera()
    elif settings.scene_camera_index >= 0:
        scene = OpenCVCamera(settings.scene_camera_index)
    else:
        scene = None
    recognizer = QwenVLRecognizer(settings.dashscope_api_key, settings.dashscope_base_url, settings.vl_model)
    return Assistant(settings, arm, hand, scene, detector, recognizer, load_poses(settings.poses_file),
                     load_objects(settings.objects_file), grasp_cfg, load_handeye(settings.handeye_file),
                     Inventory(settings.inventory_file))

