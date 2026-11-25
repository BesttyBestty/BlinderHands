#!/usr/bin/env python3
"""观察姿态标定 + 微调方向标定。

用法：
  python tools/calibrate.py views    逐个输入关节角 → 机械臂到位 → 拍照 → 实时评估文字 → 确认后写入 config/poses.json
  python tools/calibrate.py refine   在当前姿态下分别微动腕部关节，观察文字在画面里往哪边移动，自动算出 refine 的方向
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blinders_hands.arm import NeroArm  # noqa: E402
from blinders_hands.camera import OpenCVCamera  # noqa: E402
from blinders_hands.config import Settings  # noqa: E402
from blinders_hands.vision.text_detector import assess_view, build_detector  # noqa: E402


def parse_joints(text: str) -> list[float] | None:
    try:
        parts = [float(p) for p in text.replace("，", ",").split(",")]
    except ValueError:
        return None
    return parts if len(parts) == 7 else None


def describe(a) -> str:
    return (f"{a.reason}｜文字占比 {a.text_area_ratio:.3f}｜偏移 x={a.center_offset[0]:+.2f} y={a.center_offset[1]:+.2f}"
            f"｜倾斜 {a.skew_deg:+.1f}°｜清晰度 {a.sharpness:.0f}")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"views": []}


def save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def calibrate_views(s: Settings, arm: NeroArm, cam: OpenCVCamera, detector) -> None:
    data = load(s.poses_file)
    print("输入 7 个关节角（弧度，逗号分隔），直接回车复用上一次，输入 q 结束。")
    last = None
    n = 0
    while True:
        raw = input("\n关节角> ").strip()
        if raw.lower() == "q":
            break
        joints = last if (raw == "" and last) else parse_joints(raw)
        if not joints:
            print("需要恰好 7 个数字")
            continue
        arm.move_joints(joints)
        frame = cam.capture()
        a = assess_view(frame, detector, min_sharpness=s.min_sharpness)
        n += 1
        path = Path(f"calib_{n:03d}.jpg")
        cv2.imwrite(str(path), frame)
        print(f"已拍照 {path}｜{describe(a)}")
        last = joints
        if input("这张能看清关键文字吗？保存为观察姿态？(y/n) ").strip().lower() == "y":
            name = input("给这个姿态起个名字（如 正面/侧面/瓶底）：").strip() or f"姿态{n}"
            data.setdefault("views", []).append({"name": name, "joints": joints, "calibrated": True})
            save(s.poses_file, data)
            print(f"已写入 {s.poses_file}")


def calibrate_refine(s: Settings, arm: NeroArm, cam: OpenCVCamera, detector) -> None:
    data = load(s.poses_file)
    raw = input("先输入一个能看到文字的姿态（7 个关节角）：")
    base = parse_joints(raw)
    if not base:
        print("格式不对")
        return
    refine = data.setdefault("refine", {})
    arm.move_joints(base)
    a0 = assess_view(cam.capture(), detector)
    if not a0.has_text:
        print("这个姿态看不到文字，换一个")
        return
    print("基准：", describe(a0))
    delta = 0.15
    for key, joint_key, axis in (("yaw", "yaw_joint", 0), ("pitch", "pitch_joint", 1)):
        j = refine.get(joint_key, 4 if key == "yaw" else 5)
        moved = list(base)
        moved[j] += delta
        arm.move_joints(moved)
        a = assess_view(cam.capture(), detector)
        shift = a.center_offset[axis] - a0.center_offset[axis]
        print(f"关节 J{j + 1} +{delta} rad → 文字{'水平' if axis == 0 else '竖直'}偏移变化 {shift:+.3f}")
        if a.has_text and abs(shift) > 0.02:
            # 文字偏到 +方向时要让它回来：gain 的符号与 shift 相反
            refine[f"{key}_gain"] = round(-0.35 if shift > 0 else 0.35, 2)
        arm.move_joints(base)
    j = refine.get("roll_joint", 6)
    moved = list(base)
    moved[j] += delta
    arm.move_joints(moved)
    a = assess_view(cam.capture(), detector)
    dskew = a.skew_deg - a0.skew_deg
    print(f"关节 J{j + 1} +{delta} rad → 文字倾斜变化 {dskew:+.1f}°")
    if a.has_text and abs(dskew) > 1:
        refine["roll_sign"] = 1.0 if dskew > 0 else -1.0
    arm.move_joints(base)
    save(s.poses_file, data)
    print("微调方向已写入：", refine)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["views", "refine"])
    args = ap.parse_args()
    s = Settings()
    arm = NeroArm(s.arm_base_url, s.arm_token, s.arm_move_wait)
    cam = OpenCVCamera(s.hand_camera_index)
    detector = build_detector(s.detector_backend, s.rdk_det_model)
    try:
        (calibrate_views if args.mode == "views" else calibrate_refine)(s, arm, cam, detector)
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()
