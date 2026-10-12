"""建筑风格分类分析器。

基于主色调与边缘信息启发式推断建筑风格。
real 模式可接入建筑风格分类模型（如 Places365 + 风格分类头）。
"""
from __future__ import annotations

from typing import List

from .base import BaseAnalyzer
from ..clue import Clue


class BuildingAnalyzer(BaseAnalyzer):
    name = "building"

    def analyze(self, image_path: str) -> List[Clue]:
        r, g, b = self.dominant_color(image_path)
        brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0

        # 暖色调 -> 地中海/南欧；冷灰 -> 北欧/现代；红褐 -> 北美/殖民
        if r > 150 and g > 120 and b < 120:
            value = "mediterranean"
            confidence = 0.5
            evidence = "暖橙/黄白色调，类似地中海风格建筑"
        elif r > 130 and g < 100 and b < 100:
            value = "colonial"
            confidence = 0.45
            evidence = "红褐砖色调，可能为殖民风格/红砖建筑"
        elif brightness < 0.35 and b > r:
            value = "gothic"
            confidence = 0.4
            evidence = "冷暗色调，可能为哥特式建筑"
        elif brightness > 0.6 and abs(r - g) < 30 and abs(g - b) < 30:
            value = "modern_glass"
            confidence = 0.5
            evidence = "高明度中性灰，可能为现代玻璃幕墙建筑"
        else:
            value = "generic"
            confidence = 0.25
            evidence = "建筑风格不明显"

        return [
            Clue(
                dimension="building",
                value=value,
                confidence=round(confidence, 3),
                evidence=evidence,
                source=self.name,
                extra={"dominant_rgb": [r, g, b], "display_only": True, "affects_ranking": False, "reliable": False},
            )
        ]
