"""意图理解：把一句模糊的口语，变成“找什么 + 做什么 + 怎么确认”。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .vision.recognizer import extract_json

log = logging.getLogger(__name__)

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
