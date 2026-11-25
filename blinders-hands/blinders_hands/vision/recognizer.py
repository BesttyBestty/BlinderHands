"""云端多模态识别（Qwen-VL，OpenAI 兼容接口）。

端侧检测确认“这一面可以读”之后，才把图片送到这里，控制云端调用次数。
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

log = logging.getLogger(__name__)

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
