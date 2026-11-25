#!/usr/bin/env python3
"""不用手机，直接在命令行跑一次任务。

  python tools/run_task.py "帮我看看布洛芬还能不能吃"
  BH_MOCK=1 python tools/run_task.py "布洛芬过期了吗"     # 没有硬件时的模拟运行
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blinders_hands.config import Settings  # noqa: E402
from blinders_hands.intent import IntentParser  # noqa: E402
from blinders_hands.pipeline import build_assistant  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    text = " ".join(sys.argv[1:]) or input("您想让我做什么？")
    s = Settings()
    parser = IntentParser("" if s.mock else s.dashscope_api_key, s.dashscope_base_url, s.llm_model)
    intent = parser.parse(text)
    print(f"\n理解：找「{intent.target}」，任务 {intent.task}\n确认：{intent.confirm}")
    if not intent.understood or input("(y/n) ").strip().lower() != "y":
        return
    outcome = build_assistant(s).run(intent)  # 进度已通过日志打印
    print(f"\n结果：{outcome.message}")


if __name__ == "__main__":
    main()
