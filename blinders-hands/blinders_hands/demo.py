"""生成模拟演示用的画面：空白的一面、歪着的文字、可以读的文字。

没有机械臂和摄像头时（BH_MOCK=1），用这些画面把整条链路跑通。
"""

from __future__ import annotations

import cv2
import numpy as np


def _blank(w: int = 1280, h: int = 720) -> np.ndarray:
    img = np.full((h, w, 3), 235, np.uint8)
    cv2.rectangle(img, (380, 120), (900, 600), (205, 215, 225), -1)  # 一个没有字的瓶身
    return img


def _label(lines: list[str], angle: float = 0.0, offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    img = _blank()
    ox, oy = offset
    for i, line in enumerate(lines):
        cv2.putText(img, line, (430 + ox, 250 + oy + i * 70), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (20, 20, 20), 3, cv2.LINE_AA)
    if angle:
        m = cv2.getRotationMatrix2D((640, 360), angle, 1.0)
        img = cv2.warpAffine(img, m, (1280, 720), borderValue=(235, 235, 235))
    return img


def make_demo_frames() -> list[np.ndarray]:
    lines = ["MFG 2025-03", "EXP 2027-02", "LOT 25A031"]
    return [
        _blank(),                                   # 第 1 个角度：瓶身背面，没有字
        _label(lines, angle=28, offset=(260, 140)),  # 第 2 个角度：有字，但歪了、偏了
        _label(lines),                              # 微调之后：可以读
    ]


class DemoRecognizer:
    """模拟模式用的识别器：按任务返回像样的结果，药箱扫描时轮流返回不同的药，引导时让“手”逐步靠近目标。"""

    PRODUCTS = [
        {"name": "布洛芬缓释胶囊", "production_date": "2025年03月", "expiry": "2027年02月",
         "raw_text": "生产日期 2025年03月 有效期至 2027年02月"},
        {"name": "阿莫西林胶囊", "production_date": "2024年12月", "expiry": "2026年12月",
         "raw_text": "生产日期 2024年12月 有效期至 2026年12月"},
    ]

    def __init__(self, target_bbox, objects):
        self.target_bbox = target_bbox
        self.objects = objects
        self._i = 0
        self._hand = (0.05, 0.85, 0.15, 0.98)

    def recognize(self, frame, task: str, question: str = "", target: str = ""):
        from .vision.recognizer import Recognition
        import json

        if task == "check_expiry":
            match = [p for p in self.PRODUCTS if target and (target in p["name"] or p["name"] in target)]
            fields = dict(match[0] if match else self.PRODUCTS[self._i % len(self.PRODUCTS)])
            self._i += 0 if match else 1
        elif task == "read_usage":
            fields = {"name": "布洛芬缓释胶囊", "usage": "口服，成人一次1粒，一日2次，早晚各一次",
                      "warnings": "对本品过敏者禁用"}
        elif task == "ask":
            fields = {"name": "布洛芬缓释胶囊", "answered": True,
                      "answer": "说明书上写着孕妇及哺乳期妇女慎用，建议先问医生",
                      "evidence": "孕妇及哺乳期妇女慎用"}
        else:
            fields = {"name": "布洛芬缓释胶囊", "category": "药品", "key_text": "布洛芬缓释胶囊 0.3g"}
        return Recognition(task, fields, json.dumps(fields, ensure_ascii=False))

    def locate(self, frame, target: str):
        return self.target_bbox

    def locate_many(self, frame, targets):
        out = {}
        for t in targets:
            if "手" in t:
                # 手每次朝目标挪一半
                hx1, hy1, hx2, hy2 = self._hand
                tx = (self.target_bbox[0] + self.target_bbox[2]) / 2
                ty = (self.target_bbox[1] + self.target_bbox[3]) / 2
                cx, cy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
                nx, ny = cx + (tx - cx) * 0.5, cy + (ty - cy) * 0.5
                self._hand = (nx - 0.05, ny - 0.06, nx + 0.05, ny + 0.06)
                out[t] = hx1, hy1, hx2, hy2
            else:
                out[t] = self.target_bbox
        return out

    def list_objects(self, frame, kind: str = "药盒和药瓶"):
        return list(self.objects)
