#!/usr/bin/env python3
"""
盲手 Blinder's Hands · 核心功能单文件版（由 tools/build_single_file.py 从工程源码生成）
====================================================================================

为视障者设计的桌面主动视觉机械臂系统：一句话 → 找到物体 → 视觉抓取 →
一边翻转一边寻找关键信息面 → 识别 → 判断 → 放回或递到手边 → 用一句话告诉使用者。

模块：
  config         运行配置                    arm            NERO 机械臂（CAN 官方 SDK / HTTP）
  camera         摄像头                      dates          有效期解析与过期判断
  text_detector  端侧文字检测与视角评估      recognizer     云端多模态识别、定位、物体清单
  intent         意图理解                    poses          观察姿态库、微调参数、桌面区域
  search         主动视觉搜索与视角规划      grasp          深度相机位姿估计、抓取规划、手眼标定
  guidance       引导使用者的手              inventory      药箱记录与过期提醒
  demo           模拟画面与模拟识别          pipeline       完整任务编排
  server         Flask 后端 + 手机端无障碍语音网页

用法：
  python blinders_hands_core.py selftest                   内置自检
  python blinders_hands_core.py demo                       模拟运行全部任务（不需要硬件和网络）
  python blinders_hands_core.py run "布洛芬还能不能吃"      命令行执行一次任务
  python blinders_hands_core.py serve                      启动服务，手机打开 https://<IP>:5001

依赖：pip install flask openai requests opencv-python numpy cryptography
真机：pip install pyAgxArm python-can（NERO）、pyrealsense2（深度相机）
完整工程（标定工具、测试、RDK 部署说明）：https://github.com/SSShichuanyi/-Blinders-hands
"""

from __future__ import annotations

import argparse
import base64
import calendar
import itertools
import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

import cv2
import numpy as np
import requests
from flask import Flask, Response, jsonify, request

log = logging.getLogger("blinders_hands")

# 内置默认姿态库（同目录下没有 config/poses.json 时使用）
DEFAULT_POSES = {'refine': {'roll_joint': 6, 'roll_sign': 1.0, 'yaw_joint': 4, 'yaw_gain': 0.35, 'pitch_joint': 5, 'pitch_gain': 0.35, 'max_step': 0.35}, 'sweep': {'joint': 4, 'offsets': [-1.6, -0.8, 0.8, 1.6]}, 'views': [{'name': '示例-正面', 'joints': [-0.371598, 1.11352, 0.084282, -0.6, -1.354375, 0.003176, 1.164623], 'calibrated': False}, {'name': '示例-瓶底', 'joints': [-0.371598, 1.11352, 0.084282, -0.6, -1.354375, 0.9, 0.0], 'calibrated': False}]}

# ======================================================================
# config.py
# ======================================================================

ROOT = Path(__file__).resolve().parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # 机械臂（松灵 NERO）：can = 官方 SDK pyAgxArm（支持视觉抓取、夹爪、急停）；http = Web 后台接口（示教轨迹）
    arm_backend: str = field(default_factory=lambda: os.environ.get("ARM_BACKEND", "http"))
    can_channel: str = field(default_factory=lambda: os.environ.get("CAN_CHANNEL", "can0"))
    can_interface: str = field(default_factory=lambda: os.environ.get("CAN_INTERFACE", "socketcan"))
    nero_firmware: str = field(default_factory=lambda: os.environ.get("NERO_FIRMWARE", "default"))
    speed_percent: int = field(default_factory=lambda: int(os.environ.get("ARM_SPEED_PERCENT", "30")))
    crash_rating: int = field(default_factory=lambda: int(os.environ.get("ARM_CRASH_RATING", "3")))
    arm_base_url: str = field(default_factory=lambda: os.environ.get("ARM_BASE_URL", "http://10.90.0.150"))
    arm_token: str = field(default_factory=lambda: os.environ.get("ARM_TOKEN", ""))
    arm_move_wait: float = field(default_factory=lambda: float(os.environ.get("ARM_MOVE_WAIT", "2.0")))
    arm_grasp_wait: float = field(default_factory=lambda: float(os.environ.get("ARM_GRASP_WAIT", "3.0")))

    # 摄像头：手眼相机（看机械臂手里的物体）+ 可选的全局相机（看整个桌面）
    hand_camera_index: int = field(default_factory=lambda: int(os.environ.get("HAND_CAMERA_INDEX", os.environ.get("CAMERA_INDEX", "0"))))
    scene_camera_index: int = field(default_factory=lambda: int(os.environ.get("SCENE_CAMERA_INDEX", "-1")))

    # 深度相机（视觉抓取用，同时充当全局相机）：none / realsense
    depth_camera: str = field(default_factory=lambda: os.environ.get("DEPTH_CAMERA", "none"))

    # 云端多模态模型（阿里云 DashScope，OpenAI 兼容接口）
    dashscope_api_key: str = field(default_factory=lambda: os.environ.get("DASHSCOPE_API_KEY", ""))
    dashscope_base_url: str = field(default_factory=lambda: os.environ.get(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    vl_model: str = field(default_factory=lambda: os.environ.get("VL_MODEL", "qwen-vl-plus"))
    llm_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "qwen-turbo"))

    # 端侧文字检测：opencv（任何电脑可跑）或 rdk（地瓜 RDK BPU）
    detector_backend: str = field(default_factory=lambda: os.environ.get("DETECTOR_BACKEND", "opencv"))
    rdk_det_model: str = field(default_factory=lambda: os.environ.get("RDK_DET_MODEL", ""))

    # 主动搜索参数
    max_refine_steps: int = field(default_factory=lambda: int(os.environ.get("MAX_REFINE_STEPS", "2")))
    max_cloud_calls: int = field(default_factory=lambda: int(os.environ.get("MAX_CLOUD_CALLS", "4")))
    min_sharpness: float = field(default_factory=lambda: float(os.environ.get("MIN_SHARPNESS", "80")))

    # 数据文件
    poses_file: Path = field(default_factory=lambda: Path(os.environ.get("POSES_FILE", ROOT / "config" / "poses.json")))
    objects_file: Path = field(default_factory=lambda: Path(os.environ.get("OBJECTS_FILE", ROOT / "config" / "objects.json")))
    grasp_file: Path = field(default_factory=lambda: Path(os.environ.get("GRASP_FILE", ROOT / "config" / "grasp.json")))
    handeye_file: Path = field(default_factory=lambda: Path(os.environ.get("HANDEYE_FILE", ROOT / "config" / "handeye.json")))
    inventory_file: Path = field(default_factory=lambda: Path(os.environ.get("INVENTORY_FILE", ROOT / "data" / "inventory.json")))
    max_views: int = field(default_factory=lambda: int(os.environ.get("MAX_VIEWS", "8")))
    handover_timeout: float = field(default_factory=lambda: float(os.environ.get("HANDOVER_TIMEOUT", "30")))
    capture_dir: Path = field(default_factory=lambda: Path(os.environ.get("CAPTURE_DIR", ROOT / "captures")))

    # 模拟模式：没有机械臂、摄像头时也能把整条链路跑通
    mock: bool = field(default_factory=lambda: _env_bool("BH_MOCK", False))
    # 服务端是否也用 macOS `say` 播报（默认关，手机网页负责播报，避免说两遍）
    speak_on_server: bool = field(default_factory=lambda: _env_bool("SPEAK_ON_SERVER", False))


# ======================================================================
# arm.py
# ======================================================================

NUM_JOINTS = 7

# NERO 官方关节限位（弧度），来自 pyAgxArm 文档 create_agx_arm_config() 默认值。
# 可用 config/poses.json 的 joint_limits 覆盖。
JOINT_LIMITS: list[tuple[float, float]] = [
    (-2.705261, 2.705261),
    (-1.745330, 1.745330),
    (-2.757621, 2.757621),
    (-1.012291, 2.146755),
    (-2.757621, 2.757621),
    (-0.733039, 0.959932),
    (-1.570797, 1.570797),
]


class ArmError(RuntimeError):
    pass


class EmergencyStop(ArmError):
    pass


def clamp_joints(joints: Sequence[float]) -> list[float]:
    if len(joints) != NUM_JOINTS:
        raise ArmError(f"需要 {NUM_JOINTS} 个关节角，收到 {len(joints)} 个")
    return [max(lo, min(hi, float(j))) for j, (lo, hi) in zip(joints, JOINT_LIMITS)]


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Arm(Protocol):
    supports_pose: bool

    def move_joints(self, joints: Sequence[float]) -> list[float]: ...
    def move_pose(self, pose: Sequence[float], linear: bool = False) -> None: ...
    def get_joints(self) -> list[float] | None: ...
    def get_pose(self) -> list[float] | None: ...
    def gripper(self, width: float, force: float = 1.0) -> None: ...
    def run_teach(self, name: str) -> None: ...
    def set_speed(self, percent: int) -> None: ...
    def stop(self) -> None: ...


class NeroCanArm:
    """通过 pyAgxArm 控制 NERO。位姿单位：x, y, z 为米，roll, pitch, yaw 为弧度。"""

    supports_pose = True

    def __init__(self, channel: str = "can0", interface: str = "socketcan", firmware: str = "default",
                 speed_percent: int = 30, crash_rating: int = 3, gripper: bool = True, motion_timeout: float = 15.0):
        from pyAgxArm import AgxArmFactory, create_agx_arm_config  # 只在接 NERO 的机器上需要

        cfg = create_agx_arm_config(robot="nero", firmeware_version=firmware, channel=channel, interface=interface)
        self.robot = AgxArmFactory.create_arm(cfg)
        self.effector = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER) if gripper else None
        self.robot.connect()
        deadline = time.monotonic() + 5
        while not self.robot.enable():
            if time.monotonic() > deadline:
                raise ArmError("NERO 使能失败，检查 CAN 连接和急停按钮")
            time.sleep(0.01)
        self.motion_timeout = motion_timeout
        self._stopped = threading.Event()
        self.set_speed(speed_percent)
        if crash_rating:
            try:
                self.robot.set_crash_protection_rating(joint_index=255, rating=crash_rating)
            except Exception as exc:  # 旧固件可能不支持
                log.warning("设置碰撞保护等级失败：%s", exc)

    def _check(self) -> None:
        if self._stopped.is_set():
            raise EmergencyStop("已急停")

    def _wait(self) -> None:
        time.sleep(0.3)
        start = time.monotonic()
        while time.monotonic() - start < self.motion_timeout:
            self._check()
            status = self.robot.get_arm_status()
            if status is not None and status.msg.motion_status == 0:
                return
            time.sleep(0.05)
        raise ArmError("机械臂运动超时")

    def set_speed(self, percent: int) -> None:
        self.robot.set_speed_percent(int(max(1, min(100, percent))))

    def move_joints(self, joints: Sequence[float]) -> list[float]:
        self._check()
        target = clamp_joints(joints)
        self.robot.move_j(target)
        self._wait()
        return target

    def move_pose(self, pose: Sequence[float], linear: bool = False) -> None:
        self._check()
        x, y, z, r, p, yw = (float(v) for v in pose)
        target = [x, y, z, wrap_angle(r), max(-math.pi / 2, min(math.pi / 2, p)), wrap_angle(yw)]
        (self.robot.move_l if linear else self.robot.move_p)(target)
        self._wait()

    def get_joints(self) -> list[float] | None:
        msg = self.robot.get_joint_angles()
        return list(msg.msg) if msg is not None else None

    def get_pose(self) -> list[float] | None:
        msg = self.robot.get_flange_pose()
        return list(msg.msg) if msg is not None else None

    def gripper(self, width: float, force: float = 1.0) -> None:
        if self.effector is None:
            raise ArmError("没有初始化夹爪")
        self._check()
        self.effector.move_gripper_m(value=max(0.0, width), force=force)
        time.sleep(0.8)

    def run_teach(self, name: str) -> None:
        raise ArmError("CAN 连接方式不支持 Web 后台的示教轨迹，请改用视觉抓取或 HTTP 连接")

    def stop(self) -> None:
        self._stopped.set()
        self.robot.electronic_emergency_stop()

    def resume(self) -> None:
        """急停后恢复：复位并重新使能。复位时抬起的关节会落下，调用前确认安全。"""
        self.robot.reset()
        time.sleep(0.5)
        while not self.robot.enable():
            time.sleep(0.01)
        self._stopped.clear()


class NeroHttpArm:
    supports_pose = False

    def __init__(self, base_url: str, token: str, move_wait: float = 2.0,
                 grasp_wait: float = 3.0, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.move_wait = move_wait
        self.grasp_wait = grasp_wait
        self.timeout = timeout
        self.current: list[float] | None = None
        self._stopped = threading.Event()
        if not token:
            log.warning("ARM_TOKEN 未设置，请求将不带鉴权头")

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def move_joints(self, joints: Sequence[float]) -> list[float]:
        if self._stopped.is_set():
            raise EmergencyStop("已急停")
        target = clamp_joints(joints)
        try:
            resp = requests.post(f"{self.base_url}/api/move/joint", json=target,
                                 headers=self._headers(), timeout=self.timeout)
        except requests.RequestException as exc:
            raise ArmError(f"无法连接机械臂：{exc}") from exc
        if resp.status_code == 401:
            raise ArmError("机械臂 Token 失效，请重新登录 Web 后台获取")
        if resp.status_code != 200:
            raise ArmError(f"机械臂返回 HTTP {resp.status_code}：{resp.text[:200]}")
        time.sleep(self.move_wait)
        self.current = target
        return target

    def run_teach(self, name: str) -> None:
        if self._stopped.is_set():
            raise EmergencyStop("已急停")
        try:
            resp = requests.get(f"{self.base_url}/api/teach", params={"action": "3", "name": name},
                                headers=self._headers(), timeout=self.timeout)
        except requests.RequestException as exc:
            raise ArmError(f"无法连接机械臂：{exc}") from exc
        if resp.status_code != 200:
            raise ArmError(f"示教轨迹「{name}」执行失败：HTTP {resp.status_code}")
        time.sleep(self.grasp_wait)
        self.current = None

    def move_pose(self, pose: Sequence[float], linear: bool = False) -> None:
        raise ArmError("HTTP 连接方式不支持笛卡尔运动，视觉抓取请使用 ARM_BACKEND=can")

    def get_joints(self) -> list[float] | None:
        return self.current

    def get_pose(self) -> list[float] | None:
        return None

    def gripper(self, width: float, force: float = 1.0) -> None:
        raise ArmError("HTTP 连接方式不支持单独控制夹爪")

    def set_speed(self, percent: int) -> None:
        pass

    def stop(self) -> None:
        # HTTP 接口没有急停命令：停止发送新指令，并提示使用实体急停按钮
        self._stopped.set()
        log.warning("已停止发送指令。HTTP 方式无法中断正在执行的动作，必要时请按实体急停按钮")

    def resume(self) -> None:
        self._stopped.clear()


# 兼容旧名字
NeroArm = NeroHttpArm


class MockArm:
    """模拟机械臂：记录所有指令，供测试和无硬件演示使用。"""

    supports_pose = True

    def __init__(self):
        self.current: list[float] | None = None
        self.pose: list[float] | None = [0.3, 0.0, 0.3, math.pi, 0.0, 0.0]
        self.gripper_width = 0.0
        self.speed = 100
        self.history: list[tuple[str, object]] = []
        self._stopped = threading.Event()

    def _check(self) -> None:
        if self._stopped.is_set():
            raise EmergencyStop("已急停")

    def move_joints(self, joints: Sequence[float]) -> list[float]:
        self._check()
        target = clamp_joints(joints)
        self.current = target
        self.history.append(("move", target))
        return target

    def move_pose(self, pose: Sequence[float], linear: bool = False) -> None:
        self._check()
        self.pose = [float(v) for v in pose]
        self.history.append(("move_l" if linear else "move_p", list(self.pose)))

    def get_joints(self) -> list[float] | None:
        return self.current

    def get_pose(self) -> list[float] | None:
        return self.pose

    def gripper(self, width: float, force: float = 1.0) -> None:
        self._check()
        self.gripper_width = width
        self.history.append(("gripper", round(width, 4)))

    def run_teach(self, name: str) -> None:
        self._check()
        self.history.append(("teach", name))

    def set_speed(self, percent: int) -> None:
        self.speed = percent
        self.history.append(("speed", percent))

    def stop(self) -> None:
        self._stopped.set()
        self.history.append(("stop", None))

    def resume(self) -> None:
        self._stopped.clear()


# ======================================================================
# camera.py
# ======================================================================

class CameraError(RuntimeError):
    pass


class Camera(Protocol):
    def capture(self) -> np.ndarray: ...


class OpenCVCamera:
    def __init__(self, index: int, warmup_frames: int = 10, width: int = 1920, height: int = 1080):
        self.index = index
        self.warmup_frames = warmup_frames
        self.width = width
        self.height = height

    def capture(self) -> np.ndarray:
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise CameraError(f"无法打开摄像头（索引 {self.index}），先运行 tools/check_cameras.py")
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            for _ in range(self.warmup_frames):  # 等自动曝光、自动对焦稳定
                cap.read()
                time.sleep(0.03)
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            raise CameraError("摄像头读取帧失败")
        return frame


class ImageSequenceCamera:
    """依次返回给定的图片（数组或文件路径），用完后循环。"""

    def __init__(self, frames: Iterable[np.ndarray | str | Path]):
        loaded = []
        for f in frames:
            if isinstance(f, (str, Path)):
                img = cv2.imread(str(f))
                if img is None:
                    raise CameraError(f"读不到图片：{f}")
                loaded.append(img)
            else:
                loaded.append(f)
        if not loaded:
            raise CameraError("模拟相机至少需要一张图片")
        self._cycle = itertools.cycle(loaded)

    def capture(self) -> np.ndarray:
        return next(self._cycle).copy()


# ======================================================================
# dates.py
# ======================================================================

_YMD = r"(?P<y>20\d{2})\s*[-./年]?\s*(?P<m>1[0-2]|0?[1-9])(?!\d)\s*(?:[-./月]\s*(?P<d>3[01]|[12]\d|0?[1-9])(?!\d)\s*日?)?"
_YMD_COMPACT = r"(?P<y>20\d{2})(?P<m>0[1-9]|1[0-2])(?P<d>0[1-9]|[12]\d|3[01])?"
_MY = r"(?P<m>1[0-2]|0?[1-9])\s*[/.]\s*(?P<y>20\d{2})"

_EXPIRY_KEYS = r"(?:有效期至|有效期到|有效日期|失效日期|有效期|保质期至|保质期到|EXP(?:IRY)?(?:\s*DATE)?|USE\s*BY|BEST\s*BEFORE)"
_PRODUCTION_KEYS = r"(?:生产日期|生产批号日期|制造日期|MFG(?:\s*DATE)?|MFD|PROD(?:UCTION)?\s*DATE)"
_SEP = r"\s*[:：]?\s*"


@dataclass
class DateInfo:
    production: date | None = None
    expiry: date | None = None
    expiry_is_estimated: bool = False  # 由生产日期 + 保质期推算得出

    @property
    def found(self) -> bool:
        return self.expiry is not None


def _end_of_month(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])


def _to_date(y: str, m: str, d: str | None, month_end: bool) -> date | None:
    try:
        yi, mi = int(y), int(m)
        if d:
            return date(yi, mi, int(d))
        return _end_of_month(yi, mi) if month_end else date(yi, mi, 1)
    except ValueError:
        return None


def _find_date_after(keys: str, text: str, month_end: bool) -> date | None:
    for pattern in (_YMD_COMPACT + r"(?!\d)", _YMD, _MY):
        m = re.search(keys + _SEP + pattern, text, flags=re.IGNORECASE)
        if m:
            got = _to_date(m.group("y"), m.group("m"), m.group("d") if "d" in m.groupdict() else None, month_end)
            if got:
                return got
    return None


def _shelf_life_months(text: str) -> int | None:
    m = re.search(r"(?:保质期|有效期)\s*[:：]?\s*(\d{1,3})\s*(个月|月|年)", text)
    if not m:
        return None
    n = int(m.group(1))
    return n * 12 if m.group(2) == "年" else n


def _add_months(d: date, months: int) -> date:
    total = d.month - 1 + months
    y, mo = d.year + total // 12, total % 12 + 1
    return date(y, mo, min(d.day, calendar.monthrange(y, mo)[1]))


def parse_dates(text: str) -> DateInfo:
    info = DateInfo()
    if not text:
        return info
    normalized = text.replace("　", " ").replace("：", ":")
    info.production = _find_date_after(_PRODUCTION_KEYS, normalized, month_end=False)
    info.expiry = _find_date_after(_EXPIRY_KEYS, normalized, month_end=True)
    if info.expiry is None and info.production is not None:
        months = _shelf_life_months(normalized)
        if months:
            # 惯例：有效期 N 个月，到期日为生产日期加 N 个月的前一天
            info.expiry = _add_months(info.production, months) - timedelta(days=1)
            info.expiry_is_estimated = True
    return info


@dataclass
class ExpiryVerdict:
    status: str  # "valid" | "expiring_soon" | "expired" | "unknown"
    message: str
    expiry: date | None = None


def judge_expiry(info: DateInfo, today: date | None = None, soon_days: int = 30) -> ExpiryVerdict:
    today = today or date.today()
    if not info.found:
        return ExpiryVerdict("unknown", "没有找到有效期。")
    exp = info.expiry
    when = f"{exp.year}年{exp.month}月{exp.day}日"
    note = "（根据生产日期和保质期推算）" if info.expiry_is_estimated else ""
    left = (exp - today).days
    if left < 0:
        return ExpiryVerdict("expired", f"已经过期了，有效期到{when}{note}，不建议使用。", exp)
    if left <= soon_days:
        return ExpiryVerdict("expiring_soon", f"还能用，但快过期了，有效期到{when}{note}，还剩{left}天。", exp)
    return ExpiryVerdict("valid", f"在有效期内，可以使用，有效期到{when}{note}。", exp)


# ======================================================================
# vision/text_detector.py
# ======================================================================

@dataclass
class TextBox:
    x: int
    y: int
    w: int
    h: int
    angle: float  # 文字行相对水平方向的倾角，度，[-45, 45]

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass
class ViewAssessment:
    has_text: bool
    readable: bool
    boxes: list[TextBox] = field(default_factory=list)
    text_area_ratio: float = 0.0  # 文字区域占画面的比例
    center_offset: tuple[float, float] = (0.0, 0.0)  # 文字中心相对画面中心，归一化到 [-1, 1]
    skew_deg: float = 0.0  # 文字整体倾角
    sharpness: float = 0.0  # 文字区域的拉普拉斯方差，越大越清楚
    reason: str = ""


class TextDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[TextBox]: ...


def _box_from_contour(contour: np.ndarray) -> TextBox:
    x, y, w, h = cv2.boundingRect(contour)
    (_, _), (rw, rh), angle = cv2.minAreaRect(contour)
    # 统一成“文字行方向”的倾角：长边为行方向
    if rw < rh:
        angle = angle - 90
    while angle > 45:
        angle -= 90
    while angle < -45:
        angle += 90
    return TextBox(x, y, w, h, float(angle))


class OpenCVTextDetector:
    """经典的印刷体文字区域检测：梯度 → 二值化 → 横向闭运算连成文字行 → 筛选。"""

    def __init__(self, work_width: int = 960, min_height: int = 8, max_height_ratio: float = 0.25):
        self.work_width = work_width
        self.min_height = min_height
        self.max_height_ratio = max_height_ratio

    def detect(self, frame: np.ndarray) -> list[TextBox]:
        h0, w0 = frame.shape[:2]
        scale = self.work_width / w0 if w0 > self.work_width else 1.0
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small

        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        H = small.shape[0]
        boxes: list[TextBox] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h < self.min_height or h > H * self.max_height_ratio:
                continue
            if w < h * 1.5:  # 文字行是横长的
                continue
            fill = cv2.countNonZero(bw[y:y + h, x:x + w]) / float(w * h)
            if not 0.15 < fill < 0.9:  # 太空是噪声，太满是色块
                continue
            box = _box_from_contour(c)
            if scale != 1.0:
                box = TextBox(int(box.x / scale), int(box.y / scale), int(box.w / scale), int(box.h / scale), box.angle)
            boxes.append(box)
        return boxes


class RDKTextDetector:
    """在地瓜 RDK（X5 / S100）BPU 上运行 DBNet 类文字检测模型。

    需要先用地瓜的工具链把检测模型（例如 PaddleOCR 的检测模型）转换成 .bin，
    通过 RDK_DET_MODEL 指定路径。模型输入为 NV12，输出为 1×1×H×W 的文字概率图。
    """

    def __init__(self, model_path: str, prob_threshold: float = 0.3, min_area: int = 40):
        from hobot_dnn import pyeasy_dnn as dnn  # 只在 RDK 上存在

        self.model = dnn.load(model_path)[0]
        shape = self.model.inputs[0].properties.shape  # (1, 3, H, W)
        self.in_h, self.in_w = int(shape[2]), int(shape[3])
        self.prob_threshold = prob_threshold
        self.min_area = min_area

    @staticmethod
    def _bgr_to_nv12(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420).reshape(-1)
        y = yuv[: h * w]
        uv_planar = yuv[h * w:].reshape(2, -1)
        uv_packed = uv_planar.transpose(1, 0).reshape(-1)
        return np.concatenate([y, uv_packed])

    def detect(self, frame: np.ndarray) -> list[TextBox]:
        h0, w0 = frame.shape[:2]
        resized = cv2.resize(frame, (self.in_w, self.in_h))
        outputs = self.model.forward(self._bgr_to_nv12(resized))
        prob = np.squeeze(outputs[0].buffer).astype(np.float32)
        mask = (prob > self.prob_threshold).astype(np.uint8) * 255
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sx, sy = w0 / self.in_w, h0 / self.in_h
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            b = _box_from_contour(c)
            boxes.append(TextBox(int(b.x * sx), int(b.y * sy), int(b.w * sx), int(b.h * sy), b.angle))
        return boxes


def build_detector(backend: str, rdk_model: str = "") -> TextDetector:
    if backend == "rdk":
        if not rdk_model:
            raise ValueError("DETECTOR_BACKEND=rdk 时必须设置 RDK_DET_MODEL")
        return RDKTextDetector(rdk_model)
    return OpenCVTextDetector()


def sharpness_of(frame: np.ndarray, boxes: list[TextBox]) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if not boxes:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    values = []
    for b in boxes:
        roi = gray[max(b.y, 0): b.y + b.h, max(b.x, 0): b.x + b.w]
        if roi.size:
            values.append(cv2.Laplacian(roi, cv2.CV_64F).var())
    return float(np.median(values)) if values else 0.0


def assess_view(frame: np.ndarray, detector: TextDetector, min_sharpness: float = 80.0,
                min_area_ratio: float = 0.004, max_offset: float = 0.6, max_skew: float = 20.0) -> ViewAssessment:
    """把检测结果变成“这一面能不能读”的判断，以及调整方向。"""
    boxes = detector.detect(frame)
    if not boxes:
        return ViewAssessment(False, False, reason="这一面没有文字")

    H, W = frame.shape[:2]
    total = float(sum(b.area for b in boxes))
    ratio = total / (H * W)
    cx = sum((b.x + b.w / 2) * b.area for b in boxes) / total
    cy = sum((b.y + b.h / 2) * b.area for b in boxes) / total
    offset = ((cx - W / 2) / (W / 2), (cy - H / 2) / (H / 2))
    skew = float(sum(b.angle * b.area for b in boxes) / total)
    sharp = sharpness_of(frame, boxes)

    problems = []
    if ratio < min_area_ratio:
        problems.append("文字太小")
    if sharp < min_sharpness:
        problems.append("文字模糊")
    if max(abs(offset[0]), abs(offset[1])) > max_offset:
        problems.append("文字偏在边上")
    if abs(skew) > max_skew:
        problems.append("文字歪了")
    readable = not problems
    return ViewAssessment(True, readable, boxes, ratio, offset, skew, sharp,
                          "可以读" if readable else "、".join(problems))


# ======================================================================
# vision/recognizer.py
# ======================================================================

TASK_PROMPTS = {
    "check_expiry": (
        "这是一个药品或食品包装的照片。请读出包装上的信息，只输出 JSON，不要有其他文字：\n"
        '{"name": "商品或药品名称，看不到写空字符串", '
        '"production_date": "生产日期原文，看不到写空字符串", '
        '"expiry": "有效期至/保质期至/EXP 的原文，看不到写空字符串", '
        '"shelf_life": "保质期或有效期时长原文（如 24个月），看不到写空字符串", '
        '"raw_text": "图中与日期相关的所有原文"}'
    ),
    "identify": (
        "请识别照片里这件物品是什么，只输出 JSON：\n"
        '{"name": "名称", "category": "类别，如 药品/洗发水/调料/饮料", '
        '"key_text": "最能区分它的包装文字", "raw_text": "看到的主要文字"}'
    ),
    "read_usage": (
        "这是一个药品包装或说明书的照片。只输出 JSON：\n"
        '{"name": "药品名称", "usage": "用法用量原文", "warnings": "注意事项或禁忌原文，没有写空字符串", '
        '"raw_text": "看到的相关原文"}'
    ),
}

ASK_PROMPT = (
    "一位视障者拿着这件物品问：「{question}」\n"
    "请只根据照片里能看到的文字和外观回答，看不到就如实说看不到，不要猜。只输出 JSON：\n"
    '{{"name": "物品名称，看不到写空字符串", "answered": true 或 false（照片里有没有能回答这个问题的信息）, '
    '"answer": "一两句口语化的回答", "evidence": "作为依据的包装原文"}}'
)

LIST_PROMPT = (
    "这是桌面的俯视照片。请列出照片里所有的{kind}，只输出 JSON：\n"
    '{{"items": [{{"name": "名称，看不清写空字符串", "bbox": [x1, y1, x2, y2]}}]}}\n'
    "bbox 是外框，坐标归一化到 0-1000。没有就输出空列表。"
)

LOCATE_MANY_PROMPT = (
    "这是桌面的俯视照片。请分别找到下面这些东西：{targets}。只输出 JSON：\n"
    '{{"results": [{{"target": "名称", "found": true 或 false, "bbox": [x1, y1, x2, y2]}}]}}\n'
    "bbox 坐标归一化到 0-1000，找不到时 bbox 为 []。"
)

LOCATE_PROMPT = (
    "这是桌面的俯视照片。请找到「{target}」，只输出 JSON：\n"
    '{{"found": true 或 false, "bbox": [x1, y1, x2, y2]}}\n'
    "bbox 是物体的外框，坐标归一化到 0-1000。找不到时 found 为 false、bbox 为 []。"
)


@dataclass
class Recognition:
    task: str
    fields: dict
    raw: str


BBox = tuple[float, float, float, float]


class Recognizer(Protocol):
    def recognize(self, frame: np.ndarray, task: str, question: str = "", target: str = "") -> Recognition: ...
    def locate(self, frame: np.ndarray, target: str) -> BBox | None: ...
    def locate_many(self, frame: np.ndarray, targets: list[str]) -> dict[str, BBox | None]: ...
    def list_objects(self, frame: np.ndarray, kind: str = "药盒和药瓶") -> list[tuple[str, BBox]]: ...


def _norm_bbox(bbox) -> BBox | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (max(0.0, min(1.0, float(v) / 1000.0)) for v in bbox)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def encode_jpeg(frame: np.ndarray, max_side: int = 1600, quality: int = 90) -> str:
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("图片编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def extract_json(text: str) -> dict:
    """模型有时会包一层 ```json```，或在前后加话，这里尽量把 JSON 抠出来。"""
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


class QwenVLRecognizer:
    def __init__(self, api_key: str, base_url: str, model: str = "qwen-vl-plus"):
        from openai import OpenAI

        if not api_key:
            raise ValueError("缺少 DASHSCOPE_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def _ask(self, frame: np.ndarray, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_jpeg(frame)}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return (resp.choices[0].message.content or "").strip()

    def recognize(self, frame: np.ndarray, task: str, question: str = "", target: str = "") -> Recognition:
        if task == "ask":
            prompt = ASK_PROMPT.format(question=question or "这是什么？")
        else:
            prompt = TASK_PROMPTS.get(task, TASK_PROMPTS["identify"])
        if target:
            prompt = f"（使用者要找的是「{target}」，照片里如果有多件物品，以它为准。）\n" + prompt
        raw = self._ask(frame, prompt)
        log.info("识别结果：%s", raw)
        return Recognition(task, extract_json(raw), raw)

    def locate(self, frame: np.ndarray, target: str) -> BBox | None:
        data = extract_json(self._ask(frame, LOCATE_PROMPT.format(target=target)))
        return _norm_bbox(data.get("bbox")) if data.get("found") else None

    def locate_many(self, frame: np.ndarray, targets: list[str]) -> dict[str, BBox | None]:
        data = extract_json(self._ask(frame, LOCATE_MANY_PROMPT.format(targets="、".join(targets))))
        found: dict[str, BBox | None] = {t: None for t in targets}
        for item in data.get("results", []) or []:
            name = str(item.get("target", ""))
            key = next((t for t in targets if t in name or name in t), None)
            if key and item.get("found"):
                found[key] = _norm_bbox(item.get("bbox"))
        return found

    def list_objects(self, frame: np.ndarray, kind: str = "药盒和药瓶") -> list[tuple[str, BBox]]:
        data = extract_json(self._ask(frame, LIST_PROMPT.format(kind=kind)))
        items = []
        for item in data.get("items", []) or []:
            bbox = _norm_bbox(item.get("bbox"))
            if bbox:
                items.append((str(item.get("name", "")).strip(), bbox))
        return items


class ScriptedRecognizer:
    """模拟识别器：按顺序返回预设结果，供测试和无网络演示使用。"""

    def __init__(self, responses: list[dict] | None = None, locate_bbox=None,
                 objects: list[tuple[str, BBox]] | None = None, hand_track: list[BBox | None] | None = None):
        self.responses = list(responses or [])
        self.locate_bbox = locate_bbox
        self.objects = list(objects or [])
        self.hand_track = list(hand_track or [])
        self.calls = 0
        self.questions: list[str] = []

    def recognize(self, frame: np.ndarray, task: str, question: str = "", target: str = "") -> Recognition:
        self.calls += 1
        self.questions.append(question)
        fields = self.responses.pop(0) if self.responses else {}
        return Recognition(task, fields, json.dumps(fields, ensure_ascii=False))

    def locate(self, frame: np.ndarray, target: str):
        return self.locate_bbox

    def locate_many(self, frame: np.ndarray, targets: list[str]) -> dict[str, BBox | None]:
        out: dict[str, BBox | None] = {}
        for t in targets:
            if "手" in t:
                out[t] = self.hand_track.pop(0) if len(self.hand_track) > 1 else (self.hand_track[0] if self.hand_track else None)
            else:
                out[t] = self.locate_bbox
        return out

    def list_objects(self, frame: np.ndarray, kind: str = "药盒和药瓶") -> list[tuple[str, BBox]]:
        return list(self.objects)


# ======================================================================
# intent.py
# ======================================================================

TASKS = {
    "check_expiry": "看有没有过期",
    "identify": "认出是什么",
    "read_usage": "读用法用量",
    "find": "找到它在哪",
    "ask": "回答关于这件物品的任意问题",
    "scan_box": "把药箱里的药逐一检查一遍并记下有效期",
    "expiring": "说出哪些药快过期了",
}
NEEDS_TARGET = {"check_expiry", "identify", "read_usage", "find", "ask"}
NO_CONFIRM = {"expiring"}  # 只是查记录、不用动机械臂的任务，直接回答，不反问

PROMPT = (
    "你在帮一位视障者。他说：「{text}」\n"
    "判断他要找的物品，以及想做的事。task 只能是："
    "check_expiry（看是否过期）、identify（认出是哪一个/是什么）、read_usage（读用法用量）、find（只想知道在哪）、"
    "ask（关于这件物品的其他问题，如成分、过敏原、口味、能不能给孩子用）、"
    "scan_box（把药箱/桌上的药都检查一遍）、expiring（问哪些药快过期了）。\n"
    "只输出 JSON：{{\"target\": \"物品，尽量具体；scan_box 和 expiring 写 药箱\", \"task\": \"...\", "
    "\"question\": \"task 为 ask 时，把他的问题改写成一句完整的问题，否则写空字符串\", "
    "\"confirm\": \"一句简短口语的确认问句，不超过15个字，例如：要我看看布洛芬过没过期吗？\"}}\n"
    "如果完全听不出他要什么，target 写空字符串。"
)


@dataclass
class Intent:
    target: str
    task: str
    confirm: str
    question: str = ""
    handover: bool = False  # 做完后递到使用者手边，而不是放回原处

    @property
    def understood(self) -> bool:
        return bool(self.target)


def _confirm_sentence(target: str, task: str) -> str:
    return {
        "check_expiry": f"要我看看{target}过没过期吗？",
        "identify": f"要我帮您认出{target}吗？",
        "read_usage": f"要我读一下{target}怎么吃吗？",
        "find": f"要我找找{target}在哪吗？",
        "ask": f"要我拿起{target}，从包装上找答案吗？",
        "scan_box": "要我把药箱里的药都查一遍吗？",
        "expiring": "要我说说哪些药快过期了吗？",
    }[task]


def rule_based(text: str) -> Intent:
    """没有网络或没配置大模型时的兜底规则。"""
    t = text.strip()
    handover = bool(re.search(r"递给我|拿给我|给我拿|拿过来", t))
    if re.search(r"(整理|检查|扫描|看一遍|查一遍).{0,4}(药箱|所有|全部|都)|药箱.{0,4}(整理|检查|看看)", t):
        return Intent("药箱", "scan_box", _confirm_sentence("药箱", "scan_box"))
    if re.search(r"(哪些|什么|有没有).{0,6}(快过期|要过期|过期了)", t):
        return Intent("药箱", "expiring", _confirm_sentence("药箱", "expiring"))
    if re.search(r"过期|有效期|保质期|还能(不能)?(吃|用|喝)", t):
        task = "check_expiry"
    elif re.search(r"怎么吃|吃几|用法|用量|一次|一天", t):
        task = "read_usage"
    elif re.search(r"哪个是|哪瓶|哪盒|是什么|分不清", t):
        task = "identify"
    elif re.search(r"在哪|找不到|掉了", t):
        task = "find"
    elif re.search(r"含不含|有没有.{0,6}(成分|过敏|糖|酒精|麸质|小麦|花生)|能不能给|孕妇|小孩|儿童|辣不辣|什么味", t):
        task = "ask"
    elif handover:
        task = "identify"
    else:
        task = "identify"
    target = ""
    # 先找动词后面的物品：“孕妇能不能吃布洛芬”“把降压药递给我”
    m = re.search(r"(?:吃|用|喝|涂|找|拿|看看|看一下)(?:一下)?([一-龥A-Za-z0-9]{2,8}?)(?:[吗呢吧呀？?。，,]|$)", t)
    if m and not re.search(r"几|多少|什么|一次|一天", m.group(1)):
        target = m.group(1)
    m = None if target else re.search(r"([一-龥A-Za-z0-9]{2,8}?)(?:还能|有没有|是不是|过期|在哪|怎么吃|一次|一天|每次|吃几|用法|的有效期|的保质期|含不含|能不能|递给我|拿给我|里有没有)", t)
    if m:
        target = re.sub(r"^(我的|这个|那个|这瓶|这盒|帮我看看|看看|帮我|把)+", "", m.group(1))
    # 截掉物品名后面黏着的问句成分：“布洛芬还能不能吃”→“布洛芬”，“眼药水掉了”→“眼药水”
    target = re.split(r"还能|能不能|有没有|是不是|掉了|在哪|过期|怎么|一次|一天|含不含|递给|拿给|的有效期|的保质期", target)[0]
    if not target:
        m = re.search(r"(?:药|片|胶囊|颗粒|洗发水|护发素|沐浴露|酱油|醋)", t)
        target = m.group(0) if m else ""
    confirm = _confirm_sentence(target, task) if target else "抱歉，我没听清楚，能再说一遍您要找什么吗？"
    if target and handover:
        confirm = confirm.rstrip("吗？") + "，再递给您吗？"
    return Intent(target, task, confirm, t if task == "ask" else "", handover)


class IntentParser:
    def __init__(self, api_key: str = "", base_url: str = "", model: str = "qwen-turbo"):
        self.client = None
        self.model = model
        if api_key:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def parse(self, text: str) -> Intent:
        if not text.strip():
            return Intent("", "identify", "抱歉，我没听到，请再说一遍。")
        if self.client is None:
            return rule_based(text)
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": PROMPT.format(text=text)}])
            data = extract_json(resp.choices[0].message.content or "")
        except Exception as exc:  # 网络断了也要能用
            log.warning("意图理解调用失败，改用规则：%s", exc)
            return rule_based(text)
        target = str(data.get("target", "")).strip()
        task = data.get("task") if data.get("task") in TASKS else "identify"
        confirm = str(data.get("confirm", "")).strip()
        question = str(data.get("question", "")).strip() or (text if task == "ask" else "")
        handover = bool(re.search(r"递给我|拿给我|给我拿|拿过来", text))
        if task in NEEDS_TARGET and not target:
            return Intent("", task, confirm or "抱歉，我没听清楚，能再说一遍您要找什么吗？")
        target = target or "药箱"
        return Intent(target, task, confirm or _confirm_sentence(target, task), question, handover)


# ======================================================================
# poses.py
# ======================================================================

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
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else DEFAULT_POSES
    if data.get("joint_limits"):
        JOINT_LIMITS[:] = [tuple(x) for x in data["joint_limits"]]
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


# ======================================================================
# search.py
# ======================================================================

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


# ======================================================================
# grasp.py
# ======================================================================

@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class RGBDFrame:
    color: np.ndarray          # H×W×3 BGR
    depth: np.ndarray          # H×W，单位米，0 表示无效
    intrinsics: Intrinsics


class DepthCamera(Protocol):
    def capture_rgbd(self) -> RGBDFrame: ...


class RealSenseCamera:
    """Intel RealSense（D4xx 系列）。深度对齐到彩色图。"""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(cfg)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = Intrinsics(intr.fx, intr.fy, intr.ppx, intr.ppy)
        for _ in range(15):  # 等自动曝光稳定
            self.pipeline.wait_for_frames()

    def capture_rgbd(self) -> RGBDFrame:
        frames = self.align.process(self.pipeline.wait_for_frames())
        color = np.asanyarray(frames.get_color_frame().get_data()).copy()
        depth = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * self.depth_scale
        return RGBDFrame(color, depth, self.intrinsics)

    def capture(self) -> np.ndarray:
        return self.capture_rgbd().color


class StaticRGBDCamera:
    """返回固定的 RGB-D 画面，用于测试和模拟演示。"""

    def __init__(self, frame: RGBDFrame):
        self.frame = frame

    def capture_rgbd(self) -> RGBDFrame:
        return RGBDFrame(self.frame.color.copy(), self.frame.depth.copy(), self.frame.intrinsics)

    def capture(self) -> np.ndarray:
        return self.frame.color.copy()


# ---------------------------------------------------------------------------
# 手眼标定（眼在手外：全局相机固定，看着整个桌面）
# ---------------------------------------------------------------------------

def solve_rigid_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Kabsch：求 4×4 刚体变换 T，使 dst ≈ T · src。src、dst 为 N×3 对应点（N ≥ 3，不共线）。"""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("至少需要 3 组对应点")
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cd - R @ cs
    return T


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(pts)
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def load_handeye(path: Path) -> np.ndarray | None:
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    T = np.array(data["T_base_cam"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("T_base_cam 必须是 4×4 矩阵")
    return T


def save_handeye(path: Path, T: np.ndarray, rmse: float) -> None:
    Path(path).write_text(json.dumps({"T_base_cam": T.round(6).tolist(), "rmse_m": round(rmse, 5)},
                                     ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 物体位姿估计
# ---------------------------------------------------------------------------

@dataclass
class ObjectEstimate:
    center_cam: np.ndarray       # 物体顶面中心，相机坐标系
    axis_cam: np.ndarray         # 物体长边方向（单位向量），相机坐标系
    length: float                # 长边，米
    width: float                 # 短边（夹爪要张开的方向），米
    height: float                # 物体高出桌面的高度，米
    points: int                  # 参与估计的点数
    center_base: np.ndarray      # 物体顶面中心，基座坐标系
    yaw_base: float              # 长边方向在基座坐标系下的角度，弧度


def backproject(depth: np.ndarray, intr: Intrinsics, mask: np.ndarray) -> np.ndarray:
    v, u = np.nonzero(mask)
    z = depth[v, u]
    keep = z > 0
    u, v, z = u[keep], v[keep], z[keep]
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    return np.stack([x, y, z], axis=1)


def estimate_object(frame: RGBDFrame, bbox_norm: tuple[float, float, float, float],
                    T_base_cam: np.ndarray, margin: float = 0.15) -> ObjectEstimate:
    """在检测框内分割出物体点云（高于桌面的点），估计顶面中心、朝向和尺寸。

    bbox_norm：彩色图中的物体框 (x1, y1, x2, y2)，归一化到 0-1。
    """
    H, W = frame.depth.shape
    x1, y1, x2, y2 = bbox_norm
    bw, bh = (x2 - x1), (y2 - y1)
    # 框向外扩一圈，用外圈的点估计桌面高度
    X1, Y1 = int(max(0, (x1 - bw * margin)) * W), int(max(0, (y1 - bh * margin)) * H)
    X2, Y2 = int(min(1, (x2 + bw * margin)) * W), int(min(1, (y2 + bh * margin)) * H)
    outer = np.zeros((H, W), bool)
    outer[Y1:Y2, X1:X2] = True
    inner = np.zeros((H, W), bool)
    inner[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)] = True

    ring_pts = transform_points(T_base_cam, backproject(frame.depth, frame.intrinsics, outer & ~inner))
    obj_pts = transform_points(T_base_cam, backproject(frame.depth, frame.intrinsics, inner))
    if len(ring_pts) < 20 or len(obj_pts) < 50:
        raise ArmError("深度数据太少，物体可能超出深度相机范围")

    table_z = float(np.median(ring_pts[:, 2]))
    above = obj_pts[obj_pts[:, 2] > table_z + 0.01]  # 高出桌面 1 厘米以上的才算物体
    if len(above) < 50:
        raise ArmError("在检测框里没有分割出高于桌面的物体")

    top_z = float(np.percentile(above[:, 2], 95))
    xy = above[:, :2]
    center_xy = np.median(xy, axis=0)
    cov = np.cov((xy - center_xy).T)
    eigval, eigvec = np.linalg.eigh(cov)
    major = eigvec[:, int(np.argmax(eigval))]
    minor = np.array([-major[1], major[0]])
    proj_major = (xy - center_xy) @ major
    proj_minor = (xy - center_xy) @ minor
    length = float(np.percentile(proj_major, 99) - np.percentile(proj_major, 1))
    width = float(np.percentile(proj_minor, 99) - np.percentile(proj_minor, 1))

    center_base = np.array([center_xy[0], center_xy[1], top_z])
    T_cam_base = np.linalg.inv(T_base_cam)
    center_cam = transform_points(T_cam_base, center_base)[0]
    axis_cam = T_cam_base[:3, :3] @ np.array([major[0], major[1], 0.0])
    return ObjectEstimate(center_cam, axis_cam / np.linalg.norm(axis_cam), length, width,
                          top_z - table_z, len(above), center_base, math.atan2(major[1], major[0]))


# ---------------------------------------------------------------------------
# 抓取规划与执行
# ---------------------------------------------------------------------------

@dataclass
class GraspConfig:
    gripper_max_width: float = 0.07      # 夹爪最大开口，米
    approach_height: float = 0.10        # 预抓取点在物体顶面上方的高度
    grasp_depth: float = 0.025           # 从顶面往下伸多少去夹
    lift_height: float = 0.15            # 抓起后抬高多少
    grip_force: float = 1.5              # 夹持力，牛
    tool_length: float = 0.12            # 法兰到夹爪指尖的距离
    down_roll: float = math.pi           # 夹爪朝下时的法兰姿态 roll / pitch
    down_pitch: float = 0.0
    yaw_offset: float = 0.0              # 夹爪闭合方向与法兰 yaw 的固定偏差（标定得到）
    workspace: list[list[float]] = field(default_factory=lambda: [[0.15, 0.65], [-0.45, 0.45], [0.0, 0.6]])
    inspect_pose: list[float] = field(default_factory=lambda: [0.35, 0.0, 0.35, math.pi, 0.0, 0.0])
    handover_pose: list[float] = field(default_factory=lambda: [0.30, -0.25, 0.25, math.pi, 0.0, 0.0])

    @classmethod
    def load(cls, path: Path) -> "GraspConfig":
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if not k.startswith("_")})


@dataclass
class GraspPlan:
    pregrasp: list[float]
    grasp: list[float]
    lift: list[float]
    open_width: float
    object_width: float


def _in_workspace(p: list[float], ws: list[list[float]]) -> bool:
    return all(lo <= v <= hi for v, (lo, hi) in zip(p[:3], ws))


def plan_grasp(est: ObjectEstimate, cfg: GraspConfig) -> GraspPlan:
    cx, cy, top = (float(v) for v in est.center_base)
    if est.width > cfg.gripper_max_width - 0.005:
        if est.length <= cfg.gripper_max_width - 0.005:
            # 短边也比夹爪宽、长边却能夹：转 90 度沿长边方向夹
            est.width, est.length = est.length, est.width
            est.yaw_base += math.pi / 2
        else:
            raise ArmError(f"物体太宽（{est.width * 100:.1f} 厘米），夹爪张不开")
    # 夹爪手指沿短边方向闭合 → 夹爪 yaw 与物体长边方向一致
    yaw = float(est.yaw_base + cfg.yaw_offset)
    if yaw > math.pi / 2:
        yaw -= math.pi  # 二指夹爪对称，转到最近的等价角度，减少手腕转动
    elif yaw < -math.pi / 2:
        yaw += math.pi
    grasp_tip_z = max(top - min(cfg.grasp_depth, est.height * 0.5), top - est.height + 0.005)
    flange_z = grasp_tip_z + cfg.tool_length
    grasp = [cx, cy, flange_z, cfg.down_roll, cfg.down_pitch, yaw]
    pregrasp = [cx, cy, flange_z + cfg.approach_height, cfg.down_roll, cfg.down_pitch, yaw]
    lift = [cx, cy, flange_z + cfg.lift_height, cfg.down_roll, cfg.down_pitch, yaw]
    for name, pose in (("预抓取点", pregrasp), ("抓取点", grasp), ("抬起点", lift)):
        if not _in_workspace(pose, cfg.workspace):
            raise ArmError(f"{name}超出安全工作空间：{[round(v, 3) for v in pose[:3]]}")
    open_width = min(cfg.gripper_max_width, est.width + 0.02)
    return GraspPlan(pregrasp, grasp, lift, open_width, est.width)


def execute_grasp(arm: Arm, plan: GraspPlan, cfg: GraspConfig) -> None:
    if not arm.supports_pose:
        raise ArmError("当前机械臂连接方式不支持笛卡尔运动，无法视觉抓取")
    arm.gripper(plan.open_width)
    arm.move_pose(plan.pregrasp)
    arm.move_pose(plan.grasp, linear=True)
    arm.gripper(0.0, cfg.grip_force)
    arm.move_pose(plan.lift, linear=True)


def place_back(arm: Arm, plan: GraspPlan) -> None:
    arm.move_pose(plan.lift)
    arm.move_pose(plan.grasp, linear=True)
    arm.gripper(plan.open_width)
    arm.move_pose(plan.pregrasp, linear=True)


# ---------------------------------------------------------------------------
# 合成测试场景
# ---------------------------------------------------------------------------

def synthetic_scene(objects=None, W: int = 640, H: int = 480):
    """生成俯视相机看桌面上若干长方体药盒的 RGB-D 画面。

    objects：[(中心 x, 中心 y, 长, 宽, 高, 朝向)]，基座坐标系，单位米/弧度。
    返回 (画面, T_base_cam, 每个物体在画面中的归一化外框)。只有一个物体时第三项直接是外框。
    相机在基座坐标系 (0.40, 0, 0.80) 处竖直向下看。
    """
    import cv2

    single = objects is None
    objects = objects or [(0.40, 0.10, 0.09, 0.045, 0.06, 0.5)]
    intr = Intrinsics(600.0, 600.0, W / 2, H / 2)
    # 相机朝下：相机 x 轴 = 基座 -y，相机 y 轴 = 基座 -x，相机 z 轴 = 基座 -z
    R = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [0.40, 0.0, 0.80]

    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(us - intr.cx) / intr.fx, (vs - intr.cy) / intr.fy, np.ones_like(us, dtype=float)], axis=-1)
    rays_base = rays @ R.T
    cam_o = T[:3, 3]
    depth = ((0.0 - cam_o[2]) / rays_base[..., 2]).astype(np.float32)  # 视线与桌面 z=0 相交
    color = np.full((H, W, 3), 200, np.uint8)

    bboxes = []
    for (ox, oy, L, Wd, Ht, yaw) in objects:
        c, s_ = math.cos(yaw), math.sin(yaw)
        t_top = (Ht - cam_o[2]) / rays_base[..., 2]
        top = cam_o + rays_base * t_top[..., None]
        tx, ty = top[..., 0] - ox, top[..., 1] - oy
        on_top = (np.abs(tx * c + ty * s_) <= L / 2) & (np.abs(-tx * s_ + ty * c) <= Wd / 2)
        depth = np.where(on_top, np.minimum(depth, t_top), depth).astype(np.float32)
        color[on_top] = (60, 90, 220)
        ys, xs = np.nonzero(on_top)
        bboxes.append((xs.min() / W, ys.min() / H, (xs.max() + 1) / W, (ys.max() + 1) / H))
        cv2.putText(color, "MED", (int(xs.min()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return RGBDFrame(color, depth, intr), T, (bboxes[0] if single else bboxes)


# ======================================================================
# guidance.py
# ======================================================================

BBox = tuple[float, float, float, float]


@dataclass
class GuideStep:
    say: str
    done: bool
    distance: float | None = None


def _center(b: BBox) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _overlap(a: BBox, b: BBox) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def guide_instruction(hand: BBox | None, target: BBox | None, reach: float = 0.06) -> GuideStep:
    if target is None:
        return GuideStep("我现在看不到它了，请把手先收回来一点。", False)
    if hand is None:
        return GuideStep("请把手伸到桌面上方，我来告诉您往哪边挪。", False)

    hx, hy = _center(hand)
    tx, ty = _center(target)
    dx, dy = tx - hx, ty - hy  # dx>0：目标在手的右边；dy<0：目标比手更靠里（远离使用者）
    dist = (dx ** 2 + dy ** 2) ** 0.5
    if _overlap(hand, target) or dist < reach:
        return GuideStep("到了，就在您手下面，可以拿了。", True, dist)

    def amount(d: float) -> str:
        return "一点点" if d < 0.08 else "一点" if d < 0.2 else ""

    parts = []
    # 先纠正偏得多的那个方向，一次最多说两个方向
    axes = sorted([("x", dx), ("y", dy)], key=lambda t: -abs(t[1]))
    for axis, d in axes:
        if abs(d) < reach * 0.6:
            continue
        if axis == "x":
            parts.append(("往右" if d > 0 else "往左") + amount(abs(d)))
        else:
            parts.append(("往前" if d < 0 else "往回收") + amount(abs(d)))
    return GuideStep("，".join(parts[:2]) + "。", False, dist)


# ======================================================================
# inventory.py
# ======================================================================

@dataclass
class Item:
    name: str
    expiry: str          # ISO 日期
    estimated: bool = False
    location: str = ""   # “在您左边、靠近您”
    last_checked: str = ""


class Inventory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.items: dict[str, Item] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.items = {k: Item(**v) for k, v in data.items()}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: asdict(v) for k, v in self.items.items()}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.path)

    def record(self, name: str, expiry: date, estimated: bool = False, location: str = "",
               today: date | None = None) -> None:
        if not name:
            return
        with self._lock:
            self.items[name] = Item(name, expiry.isoformat(), estimated, location, (today or date.today()).isoformat())
            self._save()

    def expiring(self, within_days: int = 30, today: date | None = None) -> tuple[list[Item], list[Item]]:
        """返回（已过期, 即将过期），各自按有效期排序。"""
        today = today or date.today()
        expired, soon = [], []
        for it in self.items.values():
            left = (date.fromisoformat(it.expiry) - today).days
            if left < 0:
                expired.append(it)
            elif left <= within_days:
                soon.append(it)
        key = lambda it: it.expiry  # noqa: E731
        return sorted(expired, key=key), sorted(soon, key=key)

    def summary(self, within_days: int = 30, today: date | None = None) -> str:
        if not self.items:
            return "我还没有检查过您的药。说一句“帮我整理一下药箱”，我会把每盒药都看一遍。"
        expired, soon = self.expiring(within_days, today)
        parts = []
        if expired:
            parts.append(f"有{len(expired)}样已经过期了：" + "、".join(it.name for it in expired) + "，建议不要再用")
        if soon:
            today = today or date.today()
            desc = [f"{it.name}还剩{(date.fromisoformat(it.expiry) - today).days}天" for it in soon]
            parts.append(f"有{len(soon)}样快过期了：" + "、".join(desc))
        if not parts:
            return f"我记录的{len(self.items)}样药都在有效期内，最近一个月没有要过期的。"
        return "。".join(parts) + "。"


# ======================================================================
# demo.py
# ======================================================================

def _blank(w: int = 1280, h: int = 720) -> np.ndarray:
    img = np.full((h, w, 3), 235, np.uint8)
    cv2.rectangle(img, (380, 120), (900, 600), (205, 215, 225), -1)  # 一个没有字的瓶身
    return img


def _label(lines: list[str], angle: float = 0.0, offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    img = _blank()
    ox, oy = offset
    for i, line in enumerate(lines):
        cv2.putText(img, line, (430 + ox, 250 + oy + i * 70), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (20, 20, 20), 3, cv2.LINE_AA)
    if angle:
        m = cv2.getRotationMatrix2D((640, 360), angle, 1.0)
        img = cv2.warpAffine(img, m, (1280, 720), borderValue=(235, 235, 235))
    return img


def make_demo_frames() -> list[np.ndarray]:
    lines = ["MFG 2025-03", "EXP 2027-02", "LOT 25A031"]
    return [
        _blank(),                                   # 第 1 个角度：瓶身背面，没有字
        _label(lines, angle=28, offset=(260, 140)),  # 第 2 个角度：有字，但歪了、偏了
        _label(lines),                              # 微调之后：可以读
    ]


class DemoRecognizer:
    """模拟模式用的识别器：按任务返回像样的结果，药箱扫描时轮流返回不同的药，引导时让“手”逐步靠近目标。"""

    PRODUCTS = [
        {"name": "布洛芬缓释胶囊", "production_date": "2025年03月", "expiry": "2027年02月",
         "raw_text": "生产日期 2025年03月 有效期至 2027年02月"},
        {"name": "阿莫西林胶囊", "production_date": "2024年12月", "expiry": "2026年12月",
         "raw_text": "生产日期 2024年12月 有效期至 2026年12月"},
    ]

    def __init__(self, target_bbox, objects):
        self.target_bbox = target_bbox
        self.objects = objects
        self._i = 0
        self._hand = (0.05, 0.85, 0.15, 0.98)

    def recognize(self, frame, task: str, question: str = "", target: str = ""):
        import json

        if task == "check_expiry":
            match = [p for p in self.PRODUCTS if target and (target in p["name"] or p["name"] in target)]
            fields = dict(match[0] if match else self.PRODUCTS[self._i % len(self.PRODUCTS)])
            self._i += 0 if match else 1
        elif task == "read_usage":
            fields = {"name": "布洛芬缓释胶囊", "usage": "口服，成人一次1粒，一日2次，早晚各一次",
                      "warnings": "对本品过敏者禁用"}
        elif task == "ask":
            fields = {"name": "布洛芬缓释胶囊", "answered": True,
                      "answer": "说明书上写着孕妇及哺乳期妇女慎用，建议先问医生",
                      "evidence": "孕妇及哺乳期妇女慎用"}
        else:
            fields = {"name": "布洛芬缓释胶囊", "category": "药品", "key_text": "布洛芬缓释胶囊 0.3g"}
        return Recognition(task, fields, json.dumps(fields, ensure_ascii=False))

    def locate(self, frame, target: str):
        return self.target_bbox

    def locate_many(self, frame, targets):
        out = {}
        for t in targets:
            if "手" in t:
                # 手每次朝目标挪一半
                hx1, hy1, hx2, hy2 = self._hand
                tx = (self.target_bbox[0] + self.target_bbox[2]) / 2
                ty = (self.target_bbox[1] + self.target_bbox[3]) / 2
                cx, cy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
                nx, ny = cx + (tx - cx) * 0.5, cy + (ty - cy) * 0.5
                self._hand = (nx - 0.05, ny - 0.06, nx + 0.05, ny + 0.06)
                out[t] = hx1, hy1, hx2, hy2
            else:
                out[t] = self.target_bbox
        return out

    def list_objects(self, frame, kind: str = "药盒和药瓶"):
        return list(self.objects)


# ======================================================================
# pipeline.py
# ======================================================================

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


# ======================================================================
# server.py
# ======================================================================



@dataclass
class Job:
    id: str
    intent: Intent
    status: str = "running"  # running | waiting_release | done | error
    progress: list[str] = field(default_factory=list)
    message: str = ""
    success: bool = False
    location: str = ""


def create_app(assistant: Assistant | None = None, parser: IntentParser | None = None,
               settings: Settings | None = None) -> Flask:
    settings = settings or Settings()
    assistant = assistant or build_assistant(settings)
    if parser is None:
        key = "" if settings.mock else settings.dashscope_api_key
        parser = IntentParser(key, settings.dashscope_base_url, settings.llm_model)

    app = Flask(__name__, static_folder=None)
    jobs: dict[str, Job] = {}
    arm_lock = threading.Lock()  # 只有一只手，一次只做一件事

    @app.get("/")
    def index():
        return Response(INDEX_HTML, mimetype="text/html")

    @app.get("/api/health")
    def health():
        return jsonify({
            "mock": settings.mock,
            "detector": settings.detector_backend,
            "views": len(assistant.poses.views),
            "scene_camera": assistant.scene_camera is not None,
            "vision_grasp": assistant.can_vision_grasp,
            "state": assistant.state,
            "busy": arm_lock.locked(),
        })

    @app.post("/api/stop")
    def stop():
        assistant.stop()
        return jsonify({"say": "已经停下了。确认安全后，说“继续”。"})

    @app.post("/api/resume")
    def resume():
        assistant.resume()
        return jsonify({"say": "好的，可以继续了。"})

    @app.post("/api/release")
    def release():
        assistant.release()
        return jsonify({"say": "好的，我松手了。"})

    @app.post("/api/guide")
    def guide():
        target = str((request.get_json(silent=True) or {}).get("target", "")).strip()
        if not target:
            return jsonify({"error": "缺少目标物"}), 400
        if arm_lock.locked():
            return jsonify({"say": "我正在忙，请稍等。", "done": False})
        step = assistant.guide_step(target)
        return jsonify({"say": step.say, "done": step.done})

    @app.get("/api/inventory")
    def inventory():
        inv = assistant.inventory
        expired, soon = inv.expiring()
        return jsonify({"summary": inv.summary(), "remind": bool(expired or soon),
                        "items": [vars(it) for it in inv.items.values()]})

    @app.post("/api/understand")
    def understand():
        text = (request.get_json(silent=True) or {}).get("text", "")
        intent = parser.parse(text)
        return jsonify({"understood": intent.understood, "target": intent.target, "task": intent.task,
                        "confirm": intent.confirm, "question": intent.question, "handover": intent.handover,
                        "needs_confirm": intent.task not in NO_CONFIRM})

    @app.post("/api/tasks")
    def start_task():
        body = request.get_json(silent=True) or {}
        target = str(body.get("target", "")).strip()
        task = body.get("task", "identify")
        if task not in TASKS or (task in NEEDS_TARGET and not target):
            return jsonify({"error": "缺少要找的物品或任务类型"}), 400
        if assistant.state == "stopped":
            return jsonify({"error": "已经急停，请先说“继续”"}), 409
        if not arm_lock.acquire(blocking=False):
            return jsonify({"error": "我正在忙上一件事，请稍等"}), 409

        job = Job(uuid.uuid4().hex[:8], Intent(target or "药箱", task, "", str(body.get("question", "")),
                                               bool(body.get("handover", False))))
        jobs[job.id] = job

        def work():
            try:
                def on_state(state: str) -> None:
                    job.status = "waiting_release" if state == "waiting_release" else "running"

                outcome = assistant.run(job.intent, progress=job.progress.append, on_state=on_state)
                job.message, job.success, job.location = outcome.message, outcome.success, outcome.location
                job.status = "done"
            except Exception as exc:  # 任何意外都要变成一句能说出来的话
                log.exception("任务失败")
                job.message, job.status = f"出了点问题：{exc}", "error"
            finally:
                arm_lock.release()

        threading.Thread(target=work, daemon=True).start()
        return jsonify({"id": job.id}), 202

    @app.get("/api/tasks/<job_id>")
    def task_status(job_id: str):
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "没有这个任务"}), 404
        return jsonify({"id": job.id, "status": job.status, "progress": job.progress,
                        "message": job.message, "success": job.success, "location": job.location})

    return app




# ======================================================================
# 手机端无障碍语音网页
# ======================================================================

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
<title>盲手</title>
<style>
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  html, body { margin: 0; height: 100%; background: #111; color: #fff;
               font-family: -apple-system, "PingFang SC", sans-serif; }
  main { display: flex; flex-direction: column; height: 100%; padding: 16px; gap: 16px; }
  #talk { flex: 1; width: 100%; border: none; border-radius: 28px; background: #ffd60a; color: #111;
          font-size: 40px; font-weight: 700; touch-action: none; user-select: none; -webkit-user-select: none; }
  #talk.listening { background: #34c759; }
  #talk.busy { background: #636366; color: #fff; }
  #status { min-height: 96px; font-size: 24px; line-height: 1.4; text-align: center; padding: 0 8px; }
  #confirm { display: none; gap: 16px; }
  #confirm button { flex: 1; height: 96px; border: none; border-radius: 20px; font-size: 32px; font-weight: 700; }
  #yes { background: #34c759; color: #fff; }
  #no { background: #ff453a; color: #fff; }
  #stop { height: 88px; border: none; border-radius: 20px; background: #ff453a; color: #fff;
          font-size: 34px; font-weight: 800; }
  #release { display: none; height: 120px; border: none; border-radius: 20px; background: #0a84ff;
             color: #fff; font-size: 34px; font-weight: 700; }
</style>
</head>
<body>
<main>
  <button id="stop" aria-label="急停，让机械臂马上停下">停</button>
  <div id="status" role="status" aria-live="assertive">按住黄色区域说话</div>
  <button id="talk" aria-label="按住说话，松开结束">按住说话</button>
  <button id="release" aria-label="我拿到了，松开夹爪">我拿到了</button>
  <div id="confirm" role="group" aria-label="确认">
    <button id="yes">是的</button>
    <button id="no">不是</button>
  </div>
</main>
<script>
const talk = document.getElementById('talk');
const statusEl = document.getElementById('status');
const confirmEl = document.getElementById('confirm');
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

let pending = null;      // 等待确认的意图 {target, task, question, handover}
let busy = false;
let heard = '';
let offerGuide = null;   // 找到东西后，等待“要不要引导”的回答
let guiding = null;      // 正在引导的目标
let waitingRelease = false;
let reminder = '';       // 药箱提醒，第一次说话时一起播报
const releaseBtn = document.getElementById('release');

function say(text, onEnd) {
  statusEl.textContent = text;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = 'zh-CN';
  u.rate = 1.05;
  if (onEnd) u.onend = onEnd;
  speechSynthesis.speak(u);
}

function buzz(pattern) { if (navigator.vibrate) navigator.vibrate(pattern); }

function beep(freq) {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    osc.frequency.value = freq;
    osc.connect(ctx.destination);
    osc.start(); osc.stop(ctx.currentTime + 0.12);
  } catch (e) {}
}

function showConfirm(show) { confirmEl.style.display = show ? 'flex' : 'none'; }

const YES = /^(是|是的|对|对的|嗯|好|好的|没错|可以|行|要)/;
const NO = /^(不是|不对|不|错了|算了|取消|不用)/;
const STOP = /^(停|停下|停止|别动|暂停)/;

function post(url, body) {
  return fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'},
                     body: JSON.stringify(body || {})}).then(r => r.json()).catch(() => null);
}

async function emergencyStop() {
  guiding = null;
  buzz([300, 100, 300]);
  const r = await post('/api/stop');
  say(r ? r.say : '已经停下了');
}

async function releaseObject() {
  waitingRelease = false;
  releaseBtn.style.display = 'none';
  const r = await post('/api/release');
  say(r ? r.say : '好的');
}

async function guideLoop() {
  if (!guiding) return;
  const r = await post('/api/guide', {target: guiding});
  if (!r || !guiding) return;
  say(r.say, () => {
    if (r.done) { guiding = null; buzz([100, 60, 100, 60, 100]); return; }
    setTimeout(guideLoop, 600);
  });
}

async function handleSpeech(text) {
  text = text.trim();
  if (!text) { say('没有听到，请再按住说一遍'); return; }
  if (STOP.test(text)) { emergencyStop(); return; }
  if (/^继续/.test(text)) { const r = await post('/api/resume'); say(r ? r.say : '好的'); return; }
  if (waitingRelease && /(拿到|接住|拿稳|松手|好了)/.test(text)) { releaseObject(); return; }
  if (offerGuide) {
    const target = offerGuide;
    offerGuide = null;
    if (YES.test(text) || /引导|带/.test(text)) { guiding = target; say('好的，请把手伸到桌面上方', guideLoop); return; }
    if (NO.test(text)) { say('好的'); return; }
  }

  if (pending) {
    if (NO.test(text)) { pending = null; showConfirm(false); say('好的，请再说一遍您要找什么'); return; }
    if (YES.test(text)) { confirmTask(); return; }
  }

  statusEl.textContent = '你说：' + text;
  const r = await fetch('/api/understand', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text})
  }).then(r => r.json()).catch(() => null);
  if (!r) { say('网络好像断了，请再试一次'); return; }
  if (!r.understood) { say(r.confirm); return; }
  pending = {target: r.target, task: r.task, question: r.question, handover: r.handover};
  if (!r.needs_confirm) { confirmTask(); return; }
  showConfirm(true);
  buzz(80);
  say((reminder ? reminder + '另外，' : '') + r.confirm);
  reminder = '';
}

async function confirmTask() {
  if (!pending || busy) return;
  const intent = pending;
  pendingTask = intent;
  pending = null;
  showConfirm(false);
  const r = await fetch('/api/tasks', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(intent)
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) { say(data.error || '暂时没法开始，请再试一次'); return; }
  setBusy(true);
  say('好的，我这就去看');
  poll(data.id, 0);
}

async function poll(id, spoken) {
  const r = await fetch('/api/tasks/' + id).then(r => r.json()).catch(() => null);
  if (!r) { setTimeout(() => poll(id, spoken), 1000); return; }
  // 进度只说最新的一条，避免排队念一长串
  if (r.progress.length > spoken && r.status === 'running') {
    const latest = r.progress[r.progress.length - 1];
    if (!speechSynthesis.speaking) say(latest);
    spoken = r.progress.length;
  }
  if (r.status === 'waiting_release' && !waitingRelease) {
    waitingRelease = true;
    releaseBtn.style.display = 'block';
    buzz([80, 80, 80]);
    say('放在您手边了。拿稳以后，按住说“拿到了”，或者点“我拿到了”。');
  }
  if (r.status === 'running' || r.status === 'waiting_release') { setTimeout(() => poll(id, spoken), 700); return; }
  waitingRelease = false;
  releaseBtn.style.display = 'none';
  setBusy(false);
  buzz(r.success ? [100, 80, 100] : [400]);
  if (r.success && pendingTask && pendingTask.task === 'find') {
    offerGuide = pendingTask.target;
    say(r.message + '要我带您的手过去吗？');
  } else {
    say(r.message);
  }
}
let pendingTask = null;

function setBusy(b) {
  busy = b;
  talk.classList.toggle('busy', b);
  talk.textContent = b ? '正在看…' : '按住说话';
}

let recognition = null;
if (!SR) {
  statusEl.textContent = '这个浏览器不支持语音识别，请用 Chrome 或 Safari 打开';
} else {
  recognition = new SR();
  recognition.lang = 'zh-CN';
  recognition.interimResults = false;
  recognition.continuous = true;
  recognition.onresult = (e) => {
    for (let i = e.resultIndex; i < e.results.length; i++) {
      if (e.results[i].isFinal) heard += e.results[i][0].transcript;
    }
  };
  recognition.onend = () => { talk.classList.remove('listening'); handleSpeech(heard); };
  recognition.onerror = (e) => {
    if (e.error === 'not-allowed') say('请允许这个网页使用麦克风');
  };
}

function startListening(e) {
  e.preventDefault();
  guiding = null;
  if (!recognition) return;
  if (busy && !waitingRelease) { say('我还在忙。要停下，请点最上面的红色按钮'); return; }
  speechSynthesis.cancel();
  heard = '';
  beep(880); buzz(40);
  talk.classList.add('listening');
  statusEl.textContent = '正在听…';
  try { recognition.start(); } catch (err) {}
}

function stopListening(e) {
  e.preventDefault();
  if (!recognition || !talk.classList.contains('listening')) return;
  beep(660);
  recognition.stop();
}

talk.addEventListener('pointerdown', startListening);
talk.addEventListener('pointerup', stopListening);
talk.addEventListener('pointercancel', stopListening);
talk.addEventListener('pointerleave', stopListening);
document.getElementById('yes').addEventListener('click', confirmTask);
document.getElementById('stop').addEventListener('click', emergencyStop);
releaseBtn.addEventListener('click', releaseObject);
fetch('/api/inventory').then(r => r.json()).then(d => {
  if (d.remind) { reminder = d.summary; statusEl.textContent = d.summary + ' 按住黄色区域说话。'; }
}).catch(() => {});
document.getElementById('no').addEventListener('click', () => {
  pending = null; showConfirm(false); say('好的，请再说一遍您要找什么');
});
</script>
</body>
</html>
"""


# ======================================================================
# 自检与命令行入口
# ======================================================================

def selftest() -> None:
    today = date(2026, 11, 26)
    assert parse_dates("生产日期：2025年03月 有效期至：2027年02月").expiry == date(2027, 2, 28)
    assert parse_dates("有效期至 2026.11.20").expiry == date(2026, 11, 20)
    assert parse_dates("生产日期 2025-03-15 保质期 24个月").expiry == date(2027, 3, 14)
    assert judge_expiry(parse_dates("有效期至 2026-12-10"), today).status == "expiring_soon"
    assert rule_based("孕妇能不能吃布洛芬").target == "布洛芬"

    blank, tilted, readable = make_demo_frames()
    d = OpenCVTextDetector()
    assert not assess_view(blank, d).has_text and assess_view(readable, d).readable

    frame, T, bbox = synthetic_scene()
    est = estimate_object(frame, bbox, T)
    assert np.allclose(est.center_base, [0.40, 0.10, 0.06], atol=0.005)
    plan = plan_grasp(est, GraspConfig())
    assert abs(plan.grasp[5] - 0.5) < 0.05

    arm = MockArm()
    base = [0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    good = {"name": "布洛芬", "production_date": "2025年03月", "expiry": "2027年02月"}
    s = ActiveSearcher(arm, ImageSequenceCamera([blank, tilted, readable]), d, ScriptedRecognizer([good]),
                       PoseLibrary([View("正面", base), View("侧面", base[:6] + [0.4])], RefineConfig()))
    r = s.run("check_expiry")
    assert r.success and r.cloud_calls == 1
    assert guide_instruction((0.46, 0.42, 0.54, 0.52), (0.45, 0.40, 0.55, 0.50)).done
    print("自检通过：日期解析、意图理解、端侧文字检测、物体位姿估计、抓取规划、主动搜索闭环、引导")


def demo() -> None:
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    os.environ.update({"BH_MOCK": "1", "INVENTORY_FILE": str(tmp / "inv.json"),
                       "CAPTURE_DIR": str(tmp / "caps"), "HANDOVER_TIMEOUT": "0.3"})
    os.environ.setdefault("ARM_MOVE_WAIT", "0")
    assistant = build_assistant(Settings())
    for text in ["我牙有点疼，布洛芬还能不能吃", "布洛芬一次吃几片", "孕妇能不能吃布洛芬",
                 "帮我整理一下药箱", "有哪些药快过期了", "我的布洛芬在哪"]:
        intent = rule_based(text)
        print(f"\n使用者：{text}")
        if intent.task not in NO_CONFIRM:
            print(f"盲手：{intent.confirm}\n使用者：是的")
        outcome = assistant.run(intent, progress=lambda m: print(f"   · {m}"))
        print(f"盲手：{outcome.message}")
    print("盲手：要我带您的手过去吗？\n使用者：要")
    for _ in range(6):
        step = assistant.guide_step("布洛芬")
        print(f"盲手：{step.say}")
        if step.done:
            break


def main() -> None:
    ap = argparse.ArgumentParser(description="盲手 Blinder's Hands 核心功能")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="模拟运行全部任务")
    p_run = sub.add_parser("run", help="执行一次任务")
    p_run.add_argument("text", nargs="+")
    p_run.add_argument("-y", "--yes", action="store_true", help="跳过确认")
    sub.add_parser("serve", help="启动手机端服务")
    sub.add_parser("selftest", help="运行内置自检")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "demo":
        demo()
    elif args.cmd == "serve":
        logging.getLogger().setLevel(logging.INFO)
        app = create_app()
        ssl = None if os.environ.get("BH_HTTPS", "1") in {"0", "false", "no"} else "adhoc"
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5001")), ssl_context=ssl, threaded=True)
    else:
        s = Settings()
        parser = IntentParser("" if s.mock else s.dashscope_api_key, s.dashscope_base_url, s.llm_model)
        intent = parser.parse(" ".join(args.text))
        print(f"盲手：{intent.confirm}")
        if not intent.understood or (not args.yes and input("(y/n) ").strip().lower() != "y"):
            return
        outcome = build_assistant(s).run(intent, progress=lambda m: print(f"   · {m}"))
        print(f"盲手：{outcome.message}")


if __name__ == "__main__":
    main()
