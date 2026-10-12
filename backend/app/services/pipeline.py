"""Adapter around the geolocation domain analysis pipeline.

The algorithms live in ``backend.app.domain.geolocation.analysis``; this
service only injects the backend configuration (mode + OCR provider + model
paths) so the backend never copies or forks the pipeline.

``OCR_PROVIDER`` defaults to ``mock``; ``rapidocr`` is imported lazily by
``analysis.ocr`` only when requested, so the backend imports and runs with
neither RapidOCR nor model weights installed. The default real profile is
PP-OCRv5 server det/cls/rec; ``OCR_MODEL_TYPE=mobile`` remains supported.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from ..config import Settings

logger = logging.getLogger("m_photo.pipeline")


class PipelineService:
    """Thread-safe, lazily-constructed wrapper around ``AnalysisPipeline``."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._pipeline: Any | None = None

    @property
    def mode(self) -> str:
        return self._settings.geo_ai_mode

    @property
    def ocr_provider(self) -> str:
        value = getattr(self._settings, "ocr_provider", "mock")
        return value if isinstance(value, str) and value.strip() else "mock"

    @property
    def ocr_model_type(self) -> str:
        value = getattr(self._settings, "ocr_model_type", "server")
        return value if isinstance(value, str) and value.strip() else "server"

    def _get_pipeline(self):
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    from ..domain.geolocation.analysis import (  # lazy: keeps startup light
                        AnalysisPipeline,
                    )

                    self._pipeline = AnalysisPipeline(
                        mode=self._settings.geo_ai_mode,
                        ocr_provider=self.ocr_provider,
                        ocr_model_type=self.ocr_model_type,
                        ocr_model_dir=self._path("ocr_model_dir"),
                        ocr_det_model_path=self._path("ocr_det_model_path"),
                        ocr_rec_model_path=self._path("ocr_rec_model_path"),
                        ocr_cls_model_path=self._path("ocr_cls_model_path"),
                        ocr_min_text_score=self._float("ocr_min_text_score", 0.5),
                        ocr_language_min_confidence=self._float(
                            "ocr_language_min_confidence", 0.7
                        ),
                    )
        return self._pipeline

    def _path(self, attr: str) -> str | None:
        value = getattr(self._settings, attr, None)
        if isinstance(value, (str, Path)) and str(value):
            return str(value)
        return None

    def _float(self, attr: str, default: float) -> float:
        value = getattr(self._settings, attr, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return default
        return default

    def analyze(self, image_path: str) -> list[dict[str, Any]]:
        """Run all analyzers and return plain clue dictionaries."""
        return self._get_pipeline().analyze_as_dict(image_path)

    def analyze_region(
        self,
        image_path: str,
        *,
        image_size: tuple[int, int],
        crop_px: tuple[int, int, int, int],
        ocr_scale: float = 1.0,
        region_id: str | None = None,
        region_bbox_norm: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Run focused OCR/text analysis for local-region reports.

        The legacy :meth:`analyze` contract is unchanged. This helper avoids all
        color-only analyzers and returns the OCR status captured by the same
        pipeline instance that produced the clues.
        """
        try:
            pipeline = self._get_pipeline()
            clues = pipeline.analyze_region_as_dict(
                image_path,
                image_size=image_size,
                crop_px=crop_px,
                ocr_scale=ocr_scale,
                region_id=region_id,
                region_bbox_norm=region_bbox_norm,
            )
            return {"clues": clues, "ocr_status": pipeline.ocr_status}
        except Exception as exc:  # defensive: report status, never leak details
            logger.warning("regional analysis unavailable (%s)", type(exc).__name__)
            return {
                "clues": [],
                "ocr_status": {
                    "provider": self.ocr_provider,
                    "state": "unavailable",
                    "enabled": self.ocr_provider != "mock",
                    "available": False,
                    "line_count": 0,
                    "detail": type(exc).__name__,
                },
            }

    def detailed_analyze_region(self, *args, **kwargs) -> dict[str, Any]:
        """Backward-friendly alias for callers that need the richer result."""
        return self.analyze_region(*args, **kwargs)

    def ocr_status(self) -> dict[str, Any]:
        """Safe OCR diagnostic status (never raises into callers)."""
        try:
            return self._get_pipeline().ocr_status
        except Exception as exc:  # pragma: no cover - defensive
            # Log only the exception type: messages may embed local paths/URLs.
            logger.warning("OCR status unavailable (%s)", type(exc).__name__)
            return {
                "provider": self.ocr_provider,
                "state": "unavailable",
                "detail": type(exc).__name__,
            }


__all__ = ["PipelineService"]
