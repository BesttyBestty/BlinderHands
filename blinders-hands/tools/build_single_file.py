#!/usr/bin/env python3
"""把 blinders_hands/ 下的模块合并成一个可独立运行的单文件 submission/blinders_hands_core.py。

单文件和工程代码来自同一份源码：改了工程代码后重新运行本脚本即可。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "blinders_hands"
OUT = ROOT / "submission" / "blinders_hands_core.py"

ORDER = [
    "config.py", "arm.py", "camera.py", "dates.py", "vision/text_detector.py", "vision/recognizer.py",
    "intent.py", "poses.py", "search.py", "grasp.py", "guidance.py", "inventory.py", "demo.py",
    "pipeline.py", "server.py",
]

HEADER = '''#!/usr/bin/env python3
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
'''

FOOTER = '''

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
        print(f"\\n使用者：{text}")
        if intent.task not in NO_CONFIRM:
            print(f"盲手：{intent.confirm}\\n使用者：是的")
        outcome = assistant.run(intent, progress=lambda m: print(f"   · {m}"))
        print(f"盲手：{outcome.message}")
    print("盲手：要我带您的手过去吗？\\n使用者：要")
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
'''


def strip_module(text: str) -> str:
    text = re.sub(r'^\s*"""[\s\S]*?"""\n', "", text, count=1)  # 模块文档字符串
    out, skip = [], False
    for line in text.splitlines():
        if skip:
            skip = ")" not in line
            continue
        if re.match(r"\s*from \.", line):
            skip = "(" in line and ")" not in line
            continue
        if line.startswith(("import ", "from ")):  # 顶层导入统一放到文件头
            skip = "(" in line and ")" not in line
            continue
        if line.startswith("log = logging.getLogger("):
            continue
        out.append(line)
    return "\n".join(out).strip("\n")


def main() -> None:
    parts = []
    for name in ORDER:
        body = strip_module((PKG / name).read_text(encoding="utf-8"))
        parts.append(f"# {'=' * 70}\n# {name}\n# {'=' * 70}\n\n{body}\n")
    code = "\n\n".join(parts)

    def sub(old: str, new: str) -> None:
        nonlocal code
        if old not in code:
            sys.exit(f"合并失败，找不到：{old[:60]}")
        code = code.replace(old, new)

    sub("ROOT = Path(__file__).resolve().parent.parent", "ROOT = Path(__file__).resolve().parent")
    sub("        arm_module.JOINT_LIMITS[:] =", "        JOINT_LIMITS[:] =")
    sub('    data = json.loads(Path(path).read_text(encoding="utf-8"))\n    if data.get("joint_limits"):',
        '    path = Path(path)\n    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else DEFAULT_POSES\n'
        '    if data.get("joint_limits"):')
    sub('STATIC = Path(__file__).resolve().parent / "static"\n', "")
    sub('        return send_from_directory(STATIC, "index.html")', '        return Response(INDEX_HTML, mimetype="text/html")')
    # 去掉 server.py 自己的命令行入口，统一用文件末尾的 main()
    start = code.index("def main() -> None:\n    logging.basicConfig(")
    end = code.index('if __name__ == "__main__":\n    main()\n', start) + len('if __name__ == "__main__":\n    main()\n')
    code = code[:start] + code[end:]

    poses = {k: v for k, v in json.loads((ROOT / "config" / "poses.json").read_text()).items() if not k.startswith("_")}
    html = (PKG / "static" / "index.html").read_text(encoding="utf-8")
    if '"""' in html:
        sys.exit("index.html 里不能有三引号")
    defaults = ("\n# 内置默认姿态库（同目录下没有 config/poses.json 时使用）\nDEFAULT_POSES = "
                + repr(poses) + "\n\n")
    page = f'\n\n# {"=" * 70}\n# 手机端无障碍语音网页\n# {"=" * 70}\n\nINDEX_HTML = r"""{html}"""\n'

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(HEADER + defaults + code + page + FOOTER, encoding="utf-8")
    print(f"已生成 {OUT.relative_to(ROOT)}，{len(OUT.read_text().splitlines())} 行")


if __name__ == "__main__":
    main()
