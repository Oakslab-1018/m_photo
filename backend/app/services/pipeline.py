"""Adapter around the existing (legacy) AI analysis pipeline.

The heavy lifting stays in the repository-level ``ai`` package; this service
only injects the backend configuration (AI mode + analyzer construction) so we
never copy or fork the algorithms.

``GEO_AI_MODE`` defaults to ``mock`` and ``easyocr`` is imported lazily by
``ai.ocr`` only when ``real`` mode is requested, so the backend imports and
runs without easyocr installed.
"""
from __future__ import annotations

import threading
from typing import Any

from ..config import Settings


class PipelineService:
    """Thread-safe, lazily-constructed wrapper around ``ai.AnalysisPipeline``."""

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
                    from ai import AnalysisPipeline  # lazy: keeps startup light

                    self._pipeline = AnalysisPipeline(mode=self._settings.geo_ai_mode)
        return self._pipeline

    def analyze(self, image_path: str) -> list[dict[str, Any]]:
        """Run all analyzers and return plain clue dictionaries."""
        return self._get_pipeline().analyze_as_dict(image_path)


__all__ = ["PipelineService"]
