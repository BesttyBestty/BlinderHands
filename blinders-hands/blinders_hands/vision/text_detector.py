"""端侧文字检测：判断“这一面有没有文字、在画面哪里、歪了多少、清不清楚”。

这是主动视觉闭环的反馈信号。它必须足够快，才能在机械臂每动一下之后立即判断，
所以放在端侧跑，不调用云端：
- OpenCVTextDetector：形态学梯度 + 连通域，任何电脑都能跑，用于开发和兜底
- RDKTextDetector：在地瓜 RDK 的 BPU 上跑 DBNet 类文字检测模型
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class TextBox:
    x: int
    y: int
    w: int
    h: int
    angle: float  # 文字行相对水平方向的倾角，度，[-45, 45]

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass
class ViewAssessment:
    has_text: bool
    readable: bool
    boxes: list[TextBox] = field(default_factory=list)
    text_area_ratio: float = 0.0  # 文字区域占画面的比例
    center_offset: tuple[float, float] = (0.0, 0.0)  # 文字中心相对画面中心，归一化到 [-1, 1]
    skew_deg: float = 0.0  # 文字整体倾角
    sharpness: float = 0.0  # 文字区域的拉普拉斯方差，越大越清楚
    reason: str = ""


class TextDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[TextBox]: ...


def _box_from_contour(contour: np.ndarray) -> TextBox:
    x, y, w, h = cv2.boundingRect(contour)
    (_, _), (rw, rh), angle = cv2.minAreaRect(contour)
    # 统一成“文字行方向”的倾角：长边为行方向
    if rw < rh:
        angle = angle - 90
    while angle > 45:
        angle -= 90
    while angle < -45:
        angle += 90
    return TextBox(x, y, w, h, float(angle))


class OpenCVTextDetector:
    """经典的印刷体文字区域检测：梯度 → 二值化 → 横向闭运算连成文字行 → 筛选。"""

    def __init__(self, work_width: int = 960, min_height: int = 8, max_height_ratio: float = 0.25):
        self.work_width = work_width
        self.min_height = min_height
        self.max_height_ratio = max_height_ratio

    def detect(self, frame: np.ndarray) -> list[TextBox]:
        h0, w0 = frame.shape[:2]
        scale = self.work_width / w0 if w0 > self.work_width else 1.0
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small

        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        H = small.shape[0]
        boxes: list[TextBox] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h < self.min_height or h > H * self.max_height_ratio:
                continue
            if w < h * 1.5:  # 文字行是横长的
                continue
            fill = cv2.countNonZero(bw[y:y + h, x:x + w]) / float(w * h)
            if not 0.15 < fill < 0.9:  # 太空是噪声，太满是色块
                continue
            box = _box_from_contour(c)
            if scale != 1.0:
                box = TextBox(int(box.x / scale), int(box.y / scale), int(box.w / scale), int(box.h / scale), box.angle)
            boxes.append(box)
        return boxes


class RDKTextDetector:
    """在地瓜 RDK（X5 / S100）BPU 上运行 DBNet 类文字检测模型。

    需要先用地瓜的工具链把检测模型（例如 PaddleOCR 的检测模型）转换成 .bin，
    通过 RDK_DET_MODEL 指定路径。模型输入为 NV12，输出为 1×1×H×W 的文字概率图。
    """

    def __init__(self, model_path: str, prob_threshold: float = 0.3, min_area: int = 40):
        from hobot_dnn import pyeasy_dnn as dnn  # 只在 RDK 上存在

        self.model = dnn.load(model_path)[0]
        shape = self.model.inputs[0].properties.shape  # (1, 3, H, W)
        self.in_h, self.in_w = int(shape[2]), int(shape[3])
        self.prob_threshold = prob_threshold
        self.min_area = min_area

    @staticmethod
    def _bgr_to_nv12(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420).reshape(-1)
        y = yuv[: h * w]
        uv_planar = yuv[h * w:].reshape(2, -1)
        uv_packed = uv_planar.transpose(1, 0).reshape(-1)
        return np.concatenate([y, uv_packed])

    def detect(self, frame: np.ndarray) -> list[TextBox]:
        h0, w0 = frame.shape[:2]
        resized = cv2.resize(frame, (self.in_w, self.in_h))
        outputs = self.model.forward(self._bgr_to_nv12(resized))
        prob = np.squeeze(outputs[0].buffer).astype(np.float32)
        mask = (prob > self.prob_threshold).astype(np.uint8) * 255
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sx, sy = w0 / self.in_w, h0 / self.in_h
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            b = _box_from_contour(c)
            boxes.append(TextBox(int(b.x * sx), int(b.y * sy), int(b.w * sx), int(b.h * sy), b.angle))
        return boxes


def build_detector(backend: str, rdk_model: str = "") -> TextDetector:
    if backend == "rdk":
        if not rdk_model:
            raise ValueError("DETECTOR_BACKEND=rdk 时必须设置 RDK_DET_MODEL")
        return RDKTextDetector(rdk_model)
    return OpenCVTextDetector()


def sharpness_of(frame: np.ndarray, boxes: list[TextBox]) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if not boxes:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    values = []
    for b in boxes:
        roi = gray[max(b.y, 0): b.y + b.h, max(b.x, 0): b.x + b.w]
        if roi.size:
            values.append(cv2.Laplacian(roi, cv2.CV_64F).var())
    return float(np.median(values)) if values else 0.0


def assess_view(frame: np.ndarray, detector: TextDetector, min_sharpness: float = 80.0,
                min_area_ratio: float = 0.004, max_offset: float = 0.6, max_skew: float = 20.0) -> ViewAssessment:
    """把检测结果变成“这一面能不能读”的判断，以及调整方向。"""
    boxes = detector.detect(frame)
    if not boxes:
        return ViewAssessment(False, False, reason="这一面没有文字")

    H, W = frame.shape[:2]
    total = float(sum(b.area for b in boxes))
    ratio = total / (H * W)
    cx = sum((b.x + b.w / 2) * b.area for b in boxes) / total
    cy = sum((b.y + b.h / 2) * b.area for b in boxes) / total
    offset = ((cx - W / 2) / (W / 2), (cy - H / 2) / (H / 2))
    skew = float(sum(b.angle * b.area for b in boxes) / total)
    sharp = sharpness_of(frame, boxes)

    problems = []
    if ratio < min_area_ratio:
        problems.append("文字太小")
    if sharp < min_sharpness:
        problems.append("文字模糊")
    if max(abs(offset[0]), abs(offset[1])) > max_offset:
        problems.append("文字偏在边上")
    if abs(skew) > max_skew:
        problems.append("文字歪了")
    readable = not problems
    return ViewAssessment(True, readable, boxes, ratio, offset, skew, sharp,
                          "可以读" if readable else "、".join(problems))
