#!/usr/bin/env python3
"""
pick_and_search.py
──────────────────
把"抓取"和"自动旋转搜索"两个动作串联起来：

  1. 执行示教轨迹"抓348ml百岁山矿泉水"
  2. 等待 3 秒让抓取完成
  3. 依次尝试 candidates.json 里的候选关节角度
     - 每次移动后调用 full_pipeline.capture_and_check() 识别
     - 只要识别成功（返回结果不含"未能识别"）立刻停止并返回结果
  4. 全程中文控制台输出，方便现场演示

环境变量：
  ARM_TOKEN      机械臂鉴权 Token（必须）
  ARM_HOST       机械臂 IP/主机，默认 10.90.0.150
  CAMERA_INDEX   摄像头索引，默认 0
"""

import os
import sys
import json
import time

import requests

# ─────────────────────────────────────────────────────────────────
# 配置（全部从环境变量读取）
# ─────────────────────────────────────────────────────────────────
ARM_HOST     = os.environ.get("ARM_HOST", "10.90.0.150")
ARM_TOKEN    = os.environ.get("ARM_TOKEN", "")
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))

TEACH_URL    = f"http://{ARM_HOST}/api/teach"
JOINT_URL    = f"http://{ARM_HOST}/api/move/joint"
TEACH_NAME   = "抓348ml百岁山矿泉水"

CANDIDATES_FILE = "candidates.json"
HTTP_TIMEOUT    = 10   # 秒（示教轨迹可能比较长，给宽松一点）
MOVE_TIMEOUT    = 5    # 秒（关节移动指令）
MOVE_WAIT       = 2.0  # 秒（移动后等机械臂到位）
TEACH_WAIT      = 3.0  # 秒（抓取完成后的等待时间）

# ─────────────────────────────────────────────────────────────────
# 动态 import full_pipeline.capture_and_check
# 如果 full_pipeline.py 不存在，退化为"只拍照、不识别"模式
# ─────────────────────────────────────────────────────────────────
try:
    from full_pipeline import capture_and_check as _capture_and_check   # type: ignore
    _HAS_PIPELINE = True
    print("✅ 已加载 full_pipeline.capture_and_check()")
except ImportError:
    _HAS_PIPELINE = False
    print("⚠️  未找到 full_pipeline.py，将退化为仅拍照模式（识别结果视为"未能识别"）")

    import cv2

    def _capture_and_check(camera_index: int, save_path: str) -> str:   # type: ignore
        """
        降级实现：只拍照保存，不做 OCR，始终返回"未能识别"占位字符串。
        当 full_pipeline.py 存在时，此函数不会被调用。
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return "未能识别（摄像头打开失败）"
        for _ in range(5):          # 丢弃前几帧，等待自动曝光稳定
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return "未能识别（帧读取失败）"
        cv2.imwrite(save_path, frame)
        print(f"   📸 照片已保存（降级模式）：{save_path}")
        return "未能识别（full_pipeline 不可用，请检查模块）"


def capture_and_check(camera_index: int, save_path: str) -> str:
    """统一入口，屏蔽内部实现差异。"""
    return _capture_and_check(camera_index, save_path)


# ─────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────
def _headers() -> dict:
    if not ARM_TOKEN:
        print("⚠️  环境变量 ARM_TOKEN 未设置，请求将不带鉴权头！")
    return {
        "Authorization": f"Bearer {ARM_TOKEN}",
        "Content-Type": "application/json",
    }


def load_candidates() -> list[dict]:
    """从 candidates.json 读取候选角度列表。"""
    if not os.path.exists(CANDIDATES_FILE):
        print(f"❌ 找不到候选角度文件：{CANDIDATES_FILE}")
        return []
    try:
        with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            print(f"⚠️  {CANDIDATES_FILE} 为空或格式不正确。")
            return []
        return data
    except (json.JSONDecodeError, OSError) as e:
        print(f"❌ 读取 {CANDIDATES_FILE} 失败：{e}")
        return []


# ─────────────────────────────────────────────────────────────────
# Step 1：执行示教轨迹（抓取）
# ─────────────────────────────────────────────────────────────────
def run_teach_trajectory() -> bool:
    """
    通过 GET 请求触发已录制的示教轨迹。
    action=3 代表执行已录制轨迹。
    返回 True 表示 HTTP 请求成功，False 表示失败。
    """
    print(f"\n{'─' * 50}")
    print(f"🦾 Step 1／发送抓取指令："{TEACH_NAME}"")
    print(f"   请求：GET {TEACH_URL}?action=3&name={TEACH_NAME}")

    try:
        resp = requests.get(
            TEACH_URL,
            params={"action": "3", "name": TEACH_NAME},
            headers=_headers(),
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code == 200:
            print(f"   ✅ 示教轨迹指令发送成功（HTTP 200）")
            return True
        else:
            print(f"   ❌ 机械臂返回错误：HTTP {resp.status_code}")
            print(f"      响应内容：{resp.text[:300]}")
            return False
    except requests.exceptions.Timeout:
        print(f"   ❌ 请求超时（>{HTTP_TIMEOUT}s），请检查机械臂网络连接。")
        return False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ 无法连接机械臂（{ARM_HOST}），请检查网络。")
        return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ HTTP 异常：{e}")
        return False


# ─────────────────────────────────────────────────────────────────
# Step 2：移动到候选角度
# ─────────────────────────────────────────────────────────────────
def move_to_joints(joints: list[float]) -> bool:
    """
    POST /api/move/joint，请求体为纯数组 [J1..J7]。
    返回 True 表示成功。
    """
    joints_str = ", ".join(f"{j:.5f}" for j in joints)
    print(f"   📡 发送关节角度：[{joints_str}]")

    try:
        resp = requests.post(
            JOINT_URL,
            json=joints,           # 纯数组，不包外层字段名
            headers=_headers(),
            timeout=MOVE_TIMEOUT,
        )
        if resp.status_code == 200:
            print(f"   ✅ 移动指令发送成功（HTTP 200）")
            return True
        else:
            print(f"   ❌ 移动失败：HTTP {resp.status_code}，{resp.text[:200]}")
            return False
    except requests.exceptions.Timeout:
        print(f"   ❌ 移动请求超时（>{MOVE_TIMEOUT}s）")
        return False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ 无法连接机械臂（{ARM_HOST}）")
        return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ HTTP 异常：{e}")
        return False


# ─────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 55)
    print("  百岁山矿泉水 · 抓取 + 日期识别自动搜索")
    print("=" * 55)
    print(f"  机械臂地址：{ARM_HOST}")
    print(f"  摄像头索引：{CAMERA_INDEX}")
    print(f"  候选角度文件：{CANDIDATES_FILE}")
    print(f"  full_pipeline 可用：{'是' if _HAS_PIPELINE else '否（降级模式）'}")
    print("=" * 55)

    # ── 1. 读取候选角度，提前验证 ────────────────────────────────
    candidates = load_candidates()
    if not candidates:
        print("\n❌ 候选角度为空，无法继续。请先用 calibrate_arm.py 完成标定。")
        sys.exit(1)
    print(f"\n📋 共加载 {len(candidates)} 组候选角度。")

    # ── 2. 执行抓取示教轨迹 ───────────────────────────────────────
    ok = run_teach_trajectory()
    if not ok:
        answer = input("\n   ⚠️  抓取指令失败，是否仍然继续识别流程？(y/n)：").strip().lower()
        if answer != "y":
            print("   已退出。")
            sys.exit(1)

    # ── 3. 等待抓取完成 ──────────────────────────────────────────
    print(f"\n⏳ 等待 {TEACH_WAIT:.0f} 秒，让抓取动作完成……")
    time.sleep(TEACH_WAIT)
    print("   抓取等待结束，开始旋转搜索。")

    # ── 4. 依次尝试候选角度 ──────────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"🔍 Step 2／开始自动旋转搜索（共 {len(candidates)} 组角度）")

    result: str | None = None
    success_idx: int | None = None

    for idx, rec in enumerate(candidates, start=1):
        joints: list[float] = rec.get("joints", [])
        calib_name: str = rec.get("filename", f"候选{idx}")

        print(f"\n  ┌─ 尝试第 {idx}/{len(candidates)} 组角度（来自 {calib_name}）")

        # 4a. 移动机械臂
        moved = move_to_joints(joints)
        if not moved:
            print(f"  │  ⚠️  移动失败，跳过本组角度，继续下一组……")
            continue

        # 4b. 等待到位
        print(f"  │  ⏳ 等待 {MOVE_WAIT:.0f} 秒让机械臂到位……")
        time.sleep(MOVE_WAIT)

        # 4c. 拍照 + 识别
        save_path = f"search_{idx:03d}.jpg"
        print(f"  │  🔎 调用 capture_and_check()，保存照片 → {save_path}")
        result = capture_and_check(CAMERA_INDEX, save_path)

        print(f"  │  识别结果：{result}")

        # 4d. 判断是否成功
        if "未能识别" not in result:
            success_idx = idx
            print(f"  └─ ✅ 识别成功！停止搜索。")
            break
        else:
            print(f"  └─ ❌ 本组未识别成功，继续下一组……")

    # ── 5. 汇报最终结果 ──────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print("  搜索结束，最终结果：")
    print(f"{'=' * 55}")

    if success_idx is not None:
        rec = candidates[success_idx - 1]
        joints_str = ", ".join(f"{j:.5f}" for j in rec["joints"])
        print(f"  🎉 在第 {success_idx} 组角度处识别成功！")
        print(f"     使用的角度（弧度）：{joints_str}")
        print(f"     照片文件：search_{success_idx:03d}.jpg")
        print(f"     识别内容：{result}")
    else:
        print("  😞 已遍历全部候选角度，均未识别成功。")
        print("     建议：")
        print("       1. 用 calibrate_arm.py 增加更多候选角度")
        print("       2. 检查摄像头画面光线是否充足")
        print("       3. 确认 full_pipeline.py 运行正常")

    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ 用户中断，退出搜索流程。")
        sys.exit(0)
