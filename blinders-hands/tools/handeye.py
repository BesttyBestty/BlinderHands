#!/usr/bin/env python3
"""手眼标定（眼在手外）：求深度相机在机械臂基座坐标系下的位姿 T_base_cam，写入 config/handeye.json。

准备：
1. 打印一个 ArUco 标记（DICT_4X4_50，id 0，边长 3–5 厘米）：python tools/handeye.py marker
2. 让夹爪夹住标记，标记正面朝上、中心对准夹爪指尖中点
3. ARM_BACKEND=can DEPTH_CAMERA=realsense python tools/handeye.py calibrate

程序会让机械臂依次移动到工作空间里的若干个点，每个点用深度相机测出标记中心的 3D 坐标，
同时从机械臂读出指尖坐标，最后用 Kabsch 算法求出刚体变换，并报告误差。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blinders_hands.config import Settings  # noqa: E402
from blinders_hands.grasp import GraspConfig, save_handeye, solve_rigid_transform, transform_points  # noqa: E402


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def tip_position(flange_pose: list[float], tool_length: float) -> np.ndarray:
    x, y, z, r, p, yw = flange_pose
    return np.array([x, y, z]) + rpy_to_matrix(r, p, yw) @ np.array([0.0, 0.0, tool_length])


def marker_point(frame, dictionary) -> np.ndarray | None:
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary))
    corners, ids, _ = detector.detectMarkers(frame.color)
    if ids is None or len(ids) == 0:
        return None
    c = corners[0].reshape(4, 2).mean(axis=0)
    u, v = int(round(c[0])), int(round(c[1]))
    patch = frame.depth[max(0, v - 3):v + 4, max(0, u - 3):u + 4]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    z = float(np.median(valid))
    intr = frame.intrinsics
    return np.array([(c[0] - intr.cx) * z / intr.fx, (c[1] - intr.cy) * z / intr.fy, z])


def calibration_poses(cfg: GraspConfig, n: int = 9) -> list[list[float]]:
    (x0, x1), (y0, y1), _ = cfg.workspace
    xs = np.linspace(x0 + 0.08, x1 - 0.12, 3)
    ys = np.linspace(y0 + 0.12, y1 - 0.12, 3)
    zs = [0.12, 0.20, 0.28]
    poses = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            poses.append([float(x), float(y), zs[(i + j) % 3] + cfg.tool_length, cfg.down_roll, cfg.down_pitch, 0.0])
    return poses[:n]


def calibrate() -> None:
    from blinders_hands.arm import NeroCanArm
    from blinders_hands.grasp import RealSenseCamera

    s = Settings()
    cfg = GraspConfig.load(s.grasp_file)
    arm = NeroCanArm(s.can_channel, s.can_interface, s.nero_firmware, speed_percent=20, gripper=False)
    cam = RealSenseCamera()
    cam_pts, base_pts = [], []
    for i, pose in enumerate(calibration_poses(cfg), start=1):
        print(f"[{i}] 移动到 {[round(v, 3) for v in pose[:3]]}")
        arm.move_pose(pose)
        time.sleep(0.8)
        p_cam = marker_point(cam.capture_rgbd(), cv2.aruco.DICT_4X4_50)
        flange = arm.get_pose()
        if p_cam is None or flange is None:
            print("    看不到标记或读不到机械臂位姿，跳过")
            continue
        cam_pts.append(p_cam)
        base_pts.append(tip_position(flange, cfg.tool_length))
        print(f"    相机 {p_cam.round(3)}  基座 {base_pts[-1].round(3)}")
    if len(cam_pts) < 4:
        print("有效点少于 4 个，无法标定。检查标记是否在相机视野内。")
        return
    T = solve_rigid_transform(np.array(cam_pts), np.array(base_pts))
    err = np.linalg.norm(transform_points(T, np.array(cam_pts)) - np.array(base_pts), axis=1)
    rmse = float(np.sqrt((err ** 2).mean()))
    save_handeye(s.handeye_file, T, rmse)
    print(f"\n标定完成，{len(cam_pts)} 个点，均方根误差 {rmse * 1000:.1f} 毫米，已写入 {s.handeye_file}")
    if rmse > 0.01:
        print("误差超过 1 厘米：检查标记是否夹正、tool_length 是否准确、深度相机是否对齐。")


def make_marker(path: str = "aruco_4x4_id0.png") -> None:
    img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 600)
    img = cv2.copyMakeBorder(img, 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(path, img)
    print(f"已生成 {path}，按 4 厘米边长打印（不含白边）")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["marker", "calibrate"])
    args = ap.parse_args()
    make_marker() if args.mode == "marker" else calibrate()


if __name__ == "__main__":
    main()
