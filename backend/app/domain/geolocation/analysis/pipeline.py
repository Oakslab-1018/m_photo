"""AI 分析流水线：编排所有分析器并合并线索。"""
from __future__ import annotations

from typing import List

from config import AI_MODE
from ..clue import Clue, clues_to_dicts

from .base import BaseAnalyzer
from .ocr import OCRAnalyzer
from .vegetation import VegetationAnalyzer
from .vehicle import VehicleAnalyzer
from .building import BuildingAnalyzer
from .climate import ClimateAnalyzer
from .text_language import TextLanguageAnalyzer


class AnalysisPipeline:
    """统一调用各分析器，返回完整线索列表。"""

    def __init__(self, mode: str | None = None):
        self.mode = mode or AI_MODE
        self.analyzers: List[BaseAnalyzer] = [
            OCRAnalyzer(mode=self.mode),
            VegetationAnalyzer(),
            VehicleAnalyzer(),
            BuildingAnalyzer(),
            ClimateAnalyzer(),
        ]
        self.lang_analyzer = TextLanguageAnalyzer(mode=self.mode)

    def analyze(self, image_path: str) -> List[Clue]:
        all_clues: List[Clue] = []
        signage_text = ""

        for analyzer in self.analyzers:
            try:
                clues = analyzer.analyze(image_path)
                all_clues.extend(clues)
                # 收集 OCR 文字供语言检测使用
                for c in clues:
                    if c.dimension == "signage":
                        signage_text = c.value
            except Exception as e:
                print(f"[Pipeline] {analyzer.name} 分析失败: {e}")

        # 若有文字则做语言检测
        if signage_text:
            all_clues.extend(self.lang_analyzer.analyze_from_text(signage_text))

        return all_clues

    def analyze_as_dict(self, image_path: str) -> list[dict]:
        return clues_to_dicts(self.analyze(image_path))
