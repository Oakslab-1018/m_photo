"""Adapter around the geolocation domain analysis pipeline.

The algorithms live in ``backend.app.domain.geolocation.analysis``; this
service only injects the backend configuration (AI mode + analyzer
construction) so the backend never copies or forks the pipeline.

``GEO_AI_MODE`` defaults to ``mock`` and ``easyocr`` is imported lazily by
``analysis.ocr`` only when ``real`` mode is requested, so the backend imports
and runs without easyocr installed.
"""
from __future__ import annotations

import threading
from typing import Any

from ..config import Settings


class PipelineService:
    """Thread-safe, lazily-constructed wrapper around ``AnalysisPipeline``."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._pipeline: Any | None = None

    @property
    def mode(self) -> str:
        return self._settings.geo_ai_mode

    def _get_pipeline(self):
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    from ..domain.geolocation.analysis import (  # lazy: keeps startup light
                        AnalysisPipeline,
                    )

                    self._pipeline = AnalysisPipeline(mode=self._settings.geo_ai_mode)
        return self._pipeline

    def analyze(self, image_path: str) -> list[dict[str, Any]]:
        """Run all analyzers and return plain clue dictionaries."""
        return self._get_pipeline().analyze_as_dict(image_path)


__all__ = ["PipelineService"]
