"""AI 分析流水线：编排所有分析器并合并线索。"""
from __future__ import annotations

import logging
import math
from typing import Any, List, Sequence

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

    def analyze_region(
        self,
        image_path: str,
        *,
        image_size: tuple[int, int],
        crop_px: tuple[int, int, int, int],
        ocr_scale: float = 1.0,
        region_id: str | None = None,
        region_bbox_norm: dict[str, float] | None = None,
    ) -> List[Clue]:
        """Run the focused OCR/text pipeline for one local region.

        Unlike :meth:`analyze`, this never runs the color-only analyzers. Every
        valid provider quad is projected into image-global normalized space so
        the report and clue provenance share one coordinate system.
        """
        ocr = next((a for a in self.analyzers if isinstance(a, OCRAnalyzer)), None)
        if ocr is None:
            return []
        scale = float(ocr_scale)
        if not (0.0 < scale <= 3.0):
            scale = 1.0
        try:
            clues = list(ocr.analyze(image_path))
        except Exception as exc:  # defensive: OCR failure is reported by status
            logger.warning("regional OCR failed (%s)", type(exc).__name__)
            return []

        projected: List[Clue] = []
        for clue in clues:
            if clue.dimension == "signage":
                _project_region_clue(
                    clue,
                    image_size=image_size,
                    crop_px=crop_px,
                    ocr_scale=scale,
                    region_id=region_id,
                    region_bbox_norm=region_bbox_norm,
                )
            projected.append(clue)

        all_clues: List[Clue] = list(projected)
        for clue in projected:
            if clue.dimension != "signage":
                continue
            line_confidence = float((clue.extra or {}).get("line_confidence", clue.confidence))
            derived_language = self.lang_analyzer.analyze_from_line(
                clue.value,
                line_confidence,
                evidence=f"OCR 行（置信度 {line_confidence:.2f}）文字体系判断",
            )
            group = origin_group_for_clue(clue)
            for language_clue in derived_language:
                language_clue.extra = {
                    **(language_clue.extra or {}),
                    **_projection_fields(clue.extra or {}),
                    "origin_group": group,
                }
                all_clues.append(language_clue)
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

    def analyze_region_as_dict(
        self,
        image_path: str,
        *,
        image_size: tuple[int, int],
        crop_px: tuple[int, int, int, int],
        ocr_scale: float = 1.0,
        region_id: str | None = None,
        region_bbox_norm: dict[str, float] | None = None,
    ) -> list[dict]:
        return clues_to_dicts(self.analyze_region(
            image_path,
            image_size=image_size,
            crop_px=crop_px,
            ocr_scale=ocr_scale,
            region_id=region_id,
            region_bbox_norm=region_bbox_norm,
        ))


def _project_region_clue(
    clue: Clue,
    *,
    image_size: tuple[int, int],
    crop_px: tuple[int, int, int, int],
    ocr_scale: float,
    region_id: str | None,
    region_bbox_norm: dict[str, float] | None,
) -> None:
    extra = dict(clue.extra or {})
    local = extra.get("bbox") or extra.get("quad")
    projected = _project_quad(local, crop_px=crop_px, image_size=image_size, ocr_scale=ocr_scale)
    extra.pop("bbox", None)
    extra.pop("quad", None)
    extra["coordinate_space"] = "image_normalized"
    if projected is not None:
        extra["bbox_norm"] = _bbox_from_quad(projected)
        extra["quad_norm"] = projected
        # Keep the legacy keys for downstream semantic-origin grouping, but in
        # image-global normalized coordinates rather than crop pixels.
        extra["bbox"] = projected
        extra["quad"] = projected
    if isinstance(region_bbox_norm, dict):
        extra["region_bbox_norm"] = dict(region_bbox_norm)
    if region_id:
        extra["region_id"] = region_id
    clue.extra = extra


def _projection_fields(extra: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in ("bbox_norm", "quad_norm", "bbox", "quad", "region_bbox_norm", "region_id", "provider"):
        if extra.get(key) is not None:
            fields[key] = extra[key]
    return fields


def _project_quad(
    box: Any,
    *,
    crop_px: tuple[int, int, int, int],
    image_size: tuple[int, int],
    ocr_scale: float,
) -> list[list[float]] | None:
    if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
        return None
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        return None
    left, top, _right, _bottom = crop_px
    points: list[list[float]] = []
    for point in box:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) < 2:
            return None
        try:
            local_x, local_y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(local_x) and math.isfinite(local_y)):
            return None
        global_x = left + local_x / ocr_scale
        global_y = top + local_y / ocr_scale
        points.append([
            round(min(1.0, max(0.0, global_x / width)), 6),
            round(min(1.0, max(0.0, global_y / height)), 6),
        ])
    area = abs(sum(
        points[index][0] * points[(index + 1) % 4][1]
        - points[(index + 1) % 4][0] * points[index][1]
        for index in range(4)
    )) / 2.0
    return points if area > 1e-9 else None


def _bbox_from_quad(quad: list[list[float]]) -> dict[str, float]:
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    left = min(0.999999, round(min(xs), 6))
    top = min(0.999999, round(min(ys), 6))
    width = max(1e-6, round(max(xs) - min(xs), 6))
    height = max(1e-6, round(max(ys) - min(ys), 6))
    width = min(width, max(1e-6, round(1.0 - left, 6)))
    height = min(height, max(1e-6, round(1.0 - top, 6)))
    return {
        "x": left,
        "y": top,
        "w": width,
        "h": height,
    }
