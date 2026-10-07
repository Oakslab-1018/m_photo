"""车辆识别分析器。

real 模式可接入 YOLO 车辆检测 + 车型分类；此处用启发式（基于图像中
暖色/金属色区域比例）给出车辆存在性与可能的车型线索。
"""
from __future__ import annotations

from typing import List

from .base import BaseAnalyzer
from ..clue import Clue


class VehicleAnalyzer(BaseAnalyzer):
    name = "vehicle"

    def analyze(self, image_path: str) -> List[Clue]:
        r, g, b = self.dominant_color(image_path)
        # 车辆通常呈金属灰/白/深色；通过亮度与饱和度粗略判断
        brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        saturation = (max(r, g, b) - min(r, g, b)) / 255.0 if max(r, g, b) else 0.0

        if brightness < 0.25:
            value = "dark_vehicle"
            confidence = 0.4
            evidence = "画面整体偏暗，可能存在深色车辆"
        elif brightness > 0.7 and saturation < 0.2:
            value = "white_silver_vehicle"
            confidence = 0.45
            evidence = "高明度低饱和，可能为白/银色车辆"
        else:
            value = "unknown_vehicle"
            confidence = 0.2
            evidence = "未检测到明显车辆特征"

        return [
            Clue(
                dimension="vehicle",
                value=value,
                confidence=round(confidence, 3),
                evidence=evidence,
                source=self.name,
                extra={"brightness": round(brightness, 3), "saturation": round(saturation, 3)},
            )
        ]
