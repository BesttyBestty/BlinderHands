"""视觉抓取：深度相机 → 物体 3D 位置、朝向、尺寸 → 抓取位姿 → 机械臂执行。

坐标约定：
- 相机坐标系：OpenCV 约定，x 向右、y 向下、z 向前，单位米
- 机械臂基座坐标系：NERO 基座，单位米；T_base_cam 为相机在基座坐标系下的位姿（手眼标定得到）
- 抓取方式：自上而下抓取，夹爪沿物体短边方向闭合
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .arm import Arm, ArmError

log = logging.getLogger(__name__)


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
