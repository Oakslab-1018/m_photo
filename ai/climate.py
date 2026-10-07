"""气候特征分析器。

综合亮度、蓝色占比（天空/水体）、绿色占比推断气候带。
"""
from __future__ import annotations

from typing import List

from .base import BaseAnalyzer
from reasoning.clue import Clue


class ClimateAnalyzer(BaseAnalyzer):
    name = "climate"

    def analyze(self, image_path: str) -> List[Clue]:
        r, g, b = self.dominant_color(image_path)
        brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        green = self.green_ratio(image_path)
        blue_ratio = sum(1 for _ in range(1))  # placeholder, computed below

        img = self.load_image(image_path)
        img.thumbnail((64, 64))
        pixels = list(img.getdata())
        total = len(pixels)
        blue = sum(1 for pr, pg, pb in pixels if pb > pr + 20 and pb > pg + 10 and pb > 80)
        blue_ratio = blue / total if total else 0.0

        if blue_ratio > 0.35 and brightness > 0.6:
            value = "mediterranean_sunny"
            confidence = 0.55
            evidence = "蓝天占比高且明亮，推测地中海/晴朗气候"
        elif green > 0.4 and blue_ratio > 0.2:
            value = "temperate_maritime"
            confidence = 0.5
            evidence = "植被丰富且多云，推测温带海洋性气候"
        elif brightness < 0.3 and green < 0.15:
            value = "cold_polar"
            confidence = 0.45
            evidence = "整体偏暗且植被少，可能为寒冷气候"
        elif r > 150 and b < 100 and green < 0.2:
            value = "arid"
            confidence = 0.5
            evidence = "暖干色调、植被少，推测干旱/半干旱气候"
        else:
            value = "temperate"
            confidence = 0.3
            evidence = "综合特征倾向温带气候"

        return [
            Clue(
                dimension="climate",
                value=value,
                confidence=round(confidence, 3),
                evidence=evidence,
                source=self.name,
                extra={"blue_ratio": round(blue_ratio, 3), "brightness": round(brightness, 3)},
            )
        ]
