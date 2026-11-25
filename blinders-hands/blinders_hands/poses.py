"""观察姿态库与物体/抓取配置的读取。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import arm as arm_module


@dataclass
class View:
    name: str
    joints: list[float]
    calibrated: bool = True


@dataclass
class RefineConfig:
    """主动微调：文字在画面里偏了、歪了，机械臂该转哪个关节、转多少。

    关节编号从 0 开始。方向（sign/gain 的正负）取决于相机和机械臂的相对摆放，
    上真机时用 tools/calibrate.py 观察一次后填写。
    """
    roll_joint: int = 6     # 腕部自转：纠正文字倾斜
    roll_sign: float = 1.0
    yaw_joint: int = 4      # 左右偏：让文字面水平方向转向相机
    yaw_gain: float = 0.35
    pitch_joint: int = 5    # 上下偏：让文字面竖直方向转向相机
    pitch_gain: float = 0.35
    max_step: float = 0.35  # 单次微调最大角度（弧度），防止大幅甩动


@dataclass
class PoseLibrary:
    views: list[View]
    refine: RefineConfig = field(default_factory=RefineConfig)
    # 视角规划：每个观察姿态沿哪个关节展开、展开哪些偏移量（弧度）
    sweep_joint: int | None = None
    sweep_offsets: list[float] = field(default_factory=list)


@dataclass
class Slot:
    """桌面上的一个区域，以及从这个区域抓取物体的示教轨迹。"""
    name: str
    region: tuple[float, float, float, float]  # 全局相机画面中的 x1, y1, x2, y2，归一化到 0-1
    teach: str


@dataclass
class ObjectConfig:
    slots: list[Slot]
    default_teach: str = ""
    place_back_teach: str = ""


def load_poses(path: Path, allow_uncalibrated: bool = False) -> PoseLibrary:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("joint_limits"):
        arm_module.JOINT_LIMITS[:] = [tuple(x) for x in data["joint_limits"]]
    views = [View(v["name"], [float(j) for j in v["joints"]], v.get("calibrated", True)) for v in data.get("views", [])]
    if not allow_uncalibrated:
        views = [v for v in views if v.calibrated]
    refine = RefineConfig(**data.get("refine", {}))
    sweep = data.get("sweep", {})
    return PoseLibrary(views, refine, sweep.get("joint"), [float(x) for x in sweep.get("offsets", [])])


def load_objects(path: Path) -> ObjectConfig:
    path = Path(path)
    if not path.exists():
        return ObjectConfig([])
    data = json.loads(path.read_text(encoding="utf-8"))
    slots = [Slot(s["name"], tuple(s["region"]), s["teach"]) for s in data.get("slots", [])]
    return ObjectConfig(slots, data.get("default_teach", ""), data.get("place_back_teach", ""))
