"""摄像头：OpenCV 真实相机，以及用图片序列代替相机的模拟相机。"""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Iterable, Protocol

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass


class Camera(Protocol):
    def capture(self) -> np.ndarray: ...


class OpenCVCamera:
    def __init__(self, index: int, warmup_frames: int = 10, width: int = 1920, height: int = 1080):
        self.index = index
        self.warmup_frames = warmup_frames
        self.width = width
        self.height = height

    def capture(self) -> np.ndarray:
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise CameraError(f"无法打开摄像头（索引 {self.index}），先运行 tools/check_cameras.py")
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            for _ in range(self.warmup_frames):  # 等自动曝光、自动对焦稳定
                cap.read()
                time.sleep(0.03)
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            raise CameraError("摄像头读取帧失败")
        return frame


class ImageSequenceCamera:
    """依次返回给定的图片（数组或文件路径），用完后循环。"""

    def __init__(self, frames: Iterable[np.ndarray | str | Path]):
        loaded = []
        for f in frames:
            if isinstance(f, (str, Path)):
                img = cv2.imread(str(f))
                if img is None:
                    raise CameraError(f"读不到图片：{f}")
                loaded.append(img)
            else:
                loaded.append(f)
        if not loaded:
            raise CameraError("模拟相机至少需要一张图片")
        self._cycle = itertools.cycle(loaded)

    def capture(self) -> np.ndarray:
        return next(self._cycle).copy()
