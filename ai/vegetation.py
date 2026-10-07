"""植被类型分析器。

基于图像绿色像素占比与主色调，推断植被带类型。
real 模式可接入 PlantNet / CLIP 零样本分类，此处用启发式实现可运行版本。
"""
from __future__ import annotations

from typing import List

from .base import BaseAnalyzer
from reasoning.clue import Clue


class VegetationAnalyzer(BaseAnalyzer):
    name = "vegetation"

    def analyze(self, image_path: str) -> List[Clue]:
        green = self.green_ratio(image_path)
        r, g, b = self.dominant_color(image_path)

        if green > 0.45:
            value = "temperate_forest"
            confidence = min(0.5 + green, 0.9)
            evidence = "绿色像素占比较高，推测为温带森林/阔叶林带"
        elif green > 0.25:
            value = "grassland"
            confidence = 0.4 + green
            evidence = "存在一定绿色植被，推测为草原/农田带"
        elif green > 0.1:
            value = "savanna"
            confidence = 0.35
            evidence = "稀疏绿色植被，可能为稀树草原"
        elif g < 80 and r > 120 and b < 100:
            value = "desert"
            confidence = 0.6
            evidence = "偏黄褐色调、绿色极少，推测为荒漠/干旱带"
        elif g > 100 and b > 100 and r < 100:
            value = "tropical_rainforest"
            confidence = 0.5
            evidence = "深绿湿润色调，可能为热带雨林"
        else:
            value = "urban_sparse"
            confidence = 0.3
            evidence = "植被稀少，可能为城市建成区"

        return [
            Clue(
                dimension="vegetation",
                value=value,
                confidence=round(confidence, 3),
                evidence=evidence,
                source=self.name,
                extra={"green_ratio": round(green, 3), "dominant_rgb": [r, g, b]},
            )
        ]
