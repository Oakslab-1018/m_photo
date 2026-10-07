"""多模态 AI 分析模块包。

每个分析器实现 BaseAnalyzer.analyze(image_path) -> List[Clue]。
通过 pipeline.AnalysisPipeline 统一编排所有分析器。
"""
from .pipeline import AnalysisPipeline

__all__ = ["AnalysisPipeline"]
