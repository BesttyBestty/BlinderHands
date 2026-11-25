#!/usr/bin/env python3
"""
松灵 NERO 机械臂旋转角度交互式标定工具
用法：python calibrate_arm.py
"""

import os
import sys
import json
import time

import cv2
import requests


# ─────────────────────────────────────────────
# 配置：从环境变量读取，不写死
# ─────────────────────────────────────────────
ARM_URL = "http://10.90.0.150/api/move/joint"
ARM_TOKEN = os.environ.get("ARM_TOKEN", "")
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
CANDIDATES_FILE = "candidates.json"
HTTP_TIMEOUT = 5  # 秒


def build_headers() -> dict:
    if not ARM_TOKEN:
        print("⚠️  警告：未设置环境变量 ARM_TOKEN，请求将不带鉴权头。")
    return {"Authorization": f"Bearer {ARM_TOKEN}", "Content-Type": "application/json"}


# ─────────────────────────────────────────────
# 解析用户输入的 7 个关节角度
# ─────────────────────────────────────────────
def parse_joints(text: str) -> list[float] | None:
    """
    把形如 '-0.37,1.11,0.08,-1.01,-1.35,0.003,1.16' 的字符串
    解析成 7 个浮点数列表。解析失败返回 None。
    """
    try:
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 7:
            return None
        return [float(p) for p in parts]
    except ValueError:
        return None


# ─────────────────────────────────────────────
# 控制机械臂
# ─────────────────────────────────────────────
def move_arm(joints: list[float]) -> bool:
    """
    向机械臂发送关节角度指令。
    返回 True 表示成功，False 表示失败（已打印错误信息）。
    """
    try:
        resp = requests.post(
            ARM_URL,
            json=joints,
            headers=build_headers(),
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code == 200:
            print(f"✅ 机械臂指令发送成功（HTTP {resp.status_code}）")
            return True
        else:
            print(f"❌ 机械臂返回错误：HTTP {resp.status_code}，响应内容：{resp.text[:200]}")
            return False
    except requests.exceptions.Timeout:
        print(f"❌ 请求超时（超过 {HTTP_TIMEOUT} 秒），请检查机械臂网络连接。")
        return False
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接机械臂（{ARM_URL}），请检查网络或 IP 地址。")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ HTTP 请求异常：{e}")
        return False


# ─────────────────────────────────────────────
# 摄像头拍照
# ─────────────────────────────────────────────
def capture_photo(filename: str) -> float | None:
    """
    用摄像头拍一张照片并保存到 filename。
    返回图像平均亮度，失败返回 None。
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头（索引 {CAMERA_INDEX}），请检查设备连接。")
        return None

    # 稳定曝光：丢弃前几帧，等摄像头自动曝光稳定
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("❌ 摄像头读取帧失败，请重试。")
        return None

    cv2.imwrite(filename, frame)
    print(f"📸 照片已保存：{filename}")

    # 计算平均亮度（转成灰度图后求均值）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    return brightness


# ─────────────────────────────────────────────
# 候选记录 JSON 读写
# ─────────────────────────────────────────────
def load_candidates() -> list[dict]:
    if not os.path.exists(CANDIDATES_FILE):
        return []
    try:
        with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def append_candidate(joints: list[float], filename: str, brightness: float) -> None:
    candidates = load_candidates()
    candidates.append(
        {
            "joints": joints,
            "filename": filename,
            "brightness": round(brightness, 2),
        }
    )
    with open(CANDIDATES_FILE, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
    print(f"📝 已追加记录到 {CANDIDATES_FILE}（共 {len(candidates)} 条）")


# ─────────────────────────────────────────────
# 主循环
# ─────────────────────────────────────────────
def main() -> None:
    print("=" * 55)
    print("  松灵 NERO 机械臂旋转角度交互式标定工具")
    print("=" * 55)
    print(f"  机械臂地址：{ARM_URL}")
    print(f"  摄像头索引：{CAMERA_INDEX}")
    print(f"  候选记录文件：{CANDIDATES_FILE}")
    print("  输入 Ctrl+C 随时退出")
    print("=" * 55)
    print()

    photo_counter = 1  # 照片递增编号
    last_joints: list[float] | None = None  # 上一次成功使用的角度

    while True:
        # ── 1. 输入关节角度 ──────────────────────────────
        while True:
            if last_joints is not None:
                hint = f"（直接回车复用上次角度：{','.join(str(round(j, 5)) for j in last_joints)}）"
            else:
                hint = "（示例：-0.37,1.11,0.08,-1.01,-1.35,0.003,1.16）"

            raw = input(f"\n🔧 请输入 7 个关节角度（弧度，逗号分隔）{hint}：").strip()

            if raw == "" and last_joints is not None:
                joints = last_joints
                print(f"   ↩️  复用上次角度：{joints}")
                break

            joints = parse_joints(raw)
            if joints is None:
                print("⚠️  格式错误：请输入恰好 7 个用逗号分隔的数字，例如：0,0,0,0,0,0,0")
                continue
            break

        # ── 2. 发送指令给机械臂 ──────────────────────────
        success = move_arm(joints)
        if not success:
            retry = input("   是否跳过本次运动，直接拍照？(y/n，默认 n)：").strip().lower()
            if retry != "y":
                continue  # 重新输入角度

        # ── 3. 等待机械臂运动到位 ────────────────────────
        print("⏳ 等待 2 秒让机械臂运动到位……")
        time.sleep(2)

        # ── 4. 拍照 ──────────────────────────────────────
        filename = f"calib_{photo_counter:03d}.jpg"
        brightness = None

        while brightness is None:
            brightness = capture_photo(filename)
            if brightness is None:
                retry = input("   是否重试拍照？(y/n，默认 y)：").strip().lower()
                if retry == "n":
                    break

        if brightness is None:
            print("   跳过本次拍照，请重新输入角度。")
            continue

        photo_counter += 1
        last_joints = joints  # 记录成功使用的角度

        # ── 5. 打印平均亮度 ──────────────────────────────
        print(f"💡 图像平均亮度：{brightness:.1f}（范围 0–255，越大越亮）")

        # ── 6. 询问照片质量 ──────────────────────────────
        visible = input("🔍 这张照片能看清日期文字吗？(y/n)：").strip().lower()
        if visible == "y":
            append_candidate(joints, filename, brightness)
        else:
            print("   本次照片未记录，继续调整角度。")

        # ── 7. 询问是否继续 ──────────────────────────────
        go_on = input("\n➡️  是否继续标定下一组角度？(y/n，默认 y)：").strip().lower()
        if go_on == "n":
            break

    # ── 退出：打印所有候选记录 ──────────────────────────
    print("\n" + "=" * 55)
    print("  标定结束，以下是所有候选角度记录：")
    print("=" * 55)
    candidates = load_candidates()
    if not candidates:
        print("  （暂无记录）")
    else:
        for i, rec in enumerate(candidates, 1):
            joints_str = ", ".join(str(round(j, 5)) for j in rec["joints"])
            print(f"  [{i:02d}] 文件：{rec['filename']}  亮度：{rec['brightness']}")
            print(f"       角度：{joints_str}")
    print("=" * 55)
    print(f"\n完整记录已保存在 {CANDIDATES_FILE}，祝标定顺利！")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ 用户中断，退出标定工具。")
        sys.exit(0)
