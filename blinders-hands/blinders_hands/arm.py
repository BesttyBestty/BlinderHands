"""松灵 NERO 七轴机械臂控制。

两种连接方式：
- NeroCanArm：官方 SDK pyAgxArm，经 CAN 总线控制。支持关节运动、笛卡尔运动（move_p / move_l）、
  夹爪、限速、碰撞保护、电子急停。视觉抓取需要这种方式。
- NeroHttpArm：NERO Web 后台的 HTTP 接口。支持关节运动和执行示教轨迹。
  POST /api/move/joint  请求体为纯数组 [J1..J7]（弧度）
  GET  /api/teach?action=3&name=<轨迹名>
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Protocol, Sequence

import requests

log = logging.getLogger(__name__)

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
