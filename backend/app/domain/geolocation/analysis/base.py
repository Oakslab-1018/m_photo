"""分析器基类与图像工具函数。"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Tuple

from PIL import Image

from ..clue import Clue


class BaseAnalyzer(ABC):
    """所有 AI 分析器的基类。子类只需实现 analyze()。"""

    name: str = "base"

    @abstractmethod
    def analyze(self, image_path: str) -> List[Clue]:
        """对单张图片进行分析，返回线索列表。"""
        raise NotImplementedError

    # ---- 图像工具 ----
    @staticmethod
    def load_image(image_path: str) -> Image.Image:
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def dominant_color(image_path: str) -> Tuple[int, int, int]:
        """返回图像主色 (r,g,b)。"""
        img = BaseAnalyzer.load_image(image_path)
        img.thumbnail((64, 64))
        pixels = list(img.getdata())
        r = sum(p[0] for p in pixels) // len(pixels)
        g = sum(p[1] for p in pixels) // len(pixels)
        b = sum(p[2] for p in pixels) // len(pixels)
        return (r, g, b)

    @staticmethod
    def brightness(image_path: str) -> float:
        r, g, b = BaseAnalyzer.dominant_color(image_path)
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0

    @staticmethod
    def green_ratio(image_path: str) -> float:
        """绿色像素占比，用于植被判断。"""
        img = BaseAnalyzer.load_image(image_path)
        img.thumbnail((64, 64))
        pixels = list(img.getdata())
        total = len(pixels)
        green = sum(1 for r, g, b in pixels if g > r + 15 and g > b + 15 and g > 60)
        return green / total if total else 0.0
