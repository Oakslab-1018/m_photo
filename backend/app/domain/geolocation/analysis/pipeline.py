"""AI 分析流水线：编排所有分析器并合并线索。"""
from __future__ import annotations

import logging
from typing import Any, List

from config import AI_MODE
from ..clue import Clue, clues_to_dicts

from .base import BaseAnalyzer
from .ocr import (
    DEFAULT_MIN_TEXT_SCORE,
    DEFAULT_MODEL_TYPE,
    MOCK_PROVIDER,
    OCRAnalyzer,
    normalize_provider,
    sanitize_diagnostic,
)
from .vegetation import VegetationAnalyzer
from .vehicle import VehicleAnalyzer
from .building import BuildingAnalyzer
from .climate import ClimateAnalyzer
from .text_language import DEFAULT_OCR_LANGUAGE_MIN_CONFIDENCE, TextLanguageAnalyzer
from .semantic import origin_group_for_clue

logger = logging.getLogger("m_photo.pipeline")


class AnalysisPipeline:
    """统一调用各分析器，返回完整线索列表。

    OCR provider 独立于其它分析器：``mock``（默认，无模型/无网络）或
    ``rapidocr``（PP-OCRv5 det/cls/rec + ONNX Runtime，CPU）。模型档位由
    ``ocr_model_type`` 选择，默认 ``server``，``mobile`` 保持兼容。
    """

    def __init__(
        self,
        mode: str | None = None,
        *,
        ocr_provider: str | None = None,
        ocr_model_type: str | None = None,
        ocr_model_dir: str | None = None,
        ocr_det_model_path: str | None = None,
        ocr_rec_model_path: str | None = None,
        ocr_cls_model_path: str | None = None,
        ocr_min_text_score: float = DEFAULT_MIN_TEXT_SCORE,
        ocr_language_min_confidence: float = DEFAULT_OCR_LANGUAGE_MIN_CONFIDENCE,
        ocr_analyzer: OCRAnalyzer | None = None,
    ) -> None:
        self.mode = mode or AI_MODE
        provider = normalize_provider(ocr_provider or self.mode)
        self.ocr_provider = provider
        self.ocr_model_type = ocr_model_type or DEFAULT_MODEL_TYPE
        self.ocr_language_min_confidence = ocr_language_min_confidence
        self.analyzers: List[BaseAnalyzer] = [
            ocr_analyzer
            or OCRAnalyzer(
                provider=provider,
                model_type=self.ocr_model_type,
                model_dir=ocr_model_dir,
                det_model_path=ocr_det_model_path,
                rec_model_path=ocr_rec_model_path,
                cls_model_path=ocr_cls_model_path,
                min_text_score=ocr_min_text_score,
            ),
            VegetationAnalyzer(),
            VehicleAnalyzer(),
            BuildingAnalyzer(),
            ClimateAnalyzer(),
        ]
        self.lang_analyzer = TextLanguageAnalyzer(
            mode=provider, min_confidence=ocr_language_min_confidence
        )

    @property
    def ocr_status(self) -> dict[str, Any]:
        ocr = next((a for a in self.analyzers if isinstance(a, OCRAnalyzer)), None)
        return ocr.status() if ocr else {"provider": MOCK_PROVIDER, "state": "idle"}

    def analyze(self, image_path: str) -> List[Clue]:
        all_clues: List[Clue] = []

        for analyzer in self.analyzers:
            try:
                clues = analyzer.analyze(image_path)
                all_clues.extend(clues)
            except Exception as e:
                print(f"[Pipeline] {analyzer.name} 分析失败: {e}")
                continue

            # 语言证据来自每一条**可靠** OCR 行：置信度受 OCR 行置信度约束。
            if isinstance(analyzer, OCRAnalyzer):
                for clue in clues:
                    if clue.dimension != "signage":
                        continue
                    line_confidence = float(
                        (clue.extra or {}).get("line_confidence", clue.confidence)
                    )
                    derived_language = self.lang_analyzer.analyze_from_line(
                        clue.value,
                        line_confidence,
                        evidence=f"OCR 行（置信度 {line_confidence:.2f}）文字体系判断",
                    )
                    group = origin_group_for_clue(clue)
                    for language_clue in derived_language:
                        # Same OCR-line identity as the semantic descendants:
                        # the engine counts only the strongest contribution per
                        # origin group; raw signage stays display-only.
                        language_clue.extra = {
                            **(language_clue.extra or {}),
                            "origin_group": group,
                        }
                    all_clues.extend(derived_language)

        return self._apply_semantic(all_clues)

    @staticmethod
    def _apply_semantic(clues: List[Clue]) -> List[Clue]:
        """Append semantic OCR clues from the required semantic module."""
        try:
            from .semantic import SemanticClueAnalyzer
        except ImportError as exc:
            # Import failures are deployment bugs: expose them, but never log a
            # raw exception message that could contain paths or URLs.
            logger.error("semantic analyzer import failed (%s)", type(exc).__name__)
            raise

        try:
            semantic = SemanticClueAnalyzer().analyze(list(clues))
        except Exception as exc:  # optional post-processing must not break OCR
            detail = sanitize_diagnostic(f"{type(exc).__name__}: {exc}")
            logger.warning("semantic analysis failed: %s", detail)
            return clues
        return list(clues) + list(semantic or [])

    def analyze_as_dict(self, image_path: str) -> list[dict]:
        return clues_to_dicts(self.analyze(image_path))