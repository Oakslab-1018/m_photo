"""Service orchestration: authentication-independent prediction and health."""

from __future__ import annotations

import threading
from typing import Any

from .catalog import CatalogError, Country, compute_catalog_revision
from .config import MODEL_ID, MODEL_REVISION, Settings
from .predictor import Predictor, PredictorUnavailableError, StreetClipPredictor
from .schema import (
    DecodeError,
    EmptyImageError,
    ExifGpsPresentError,
    ImageTooLargeError,
    ImageValidationError,
    decode_image_bytes,
    validate_image_bytes,
)


class StreetClipService:
    """Thread-safe service object owning one predictor and its readiness."""

    def __init__(self, settings: Settings, *, predictor: Predictor | None = None) -> None:
        self.settings = settings
        self._predictor = predictor
        self._countries: tuple[Country, ...] = ()
        self._catalog_revision = ""
        self._load_error: str | None = None
        self._loaded = False
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    @property
    def predictor(self) -> Predictor | None:
        return self._predictor

    @property
    def countries(self) -> tuple[Country, ...]:
        return self._countries

    @property
    def catalog_revision(self) -> str:
        return self._catalog_revision

    def load(self) -> None:
        """Resolve the fixed catalog and load the real predictor exactly once.

        The country catalog is always read so readiness metadata (candidate
        list and ``catalog_revision``) is correct even when a predictor was
        injected. An injected predictor is used as-is and is never silently
        loaded behind the caller's back, so a not-ready injected predictor
        keeps ``/predict`` at 503 instead of scoring with fake weights.
        """
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            try:
                from .catalog import _read_catalog

                countries = _read_catalog(self.settings.countries_path)
            except CatalogError as exc:
                self._load_error = str(exc)
                raise
            self._countries = tuple(countries)
            self._catalog_revision = compute_catalog_revision(countries)
            if self._predictor is None:
                self._predictor = StreetClipPredictor(
                    self.settings,
                    countries=countries,
                    catalog_revision=self._catalog_revision,
                )
                try:
                    self._predictor.load()
                except PredictorUnavailableError as exc:
                    self._load_error = str(exc)
                    raise
                except Exception as exc:  # pragma: no cover - defensive
                    self._load_error = str(exc)
                    raise PredictorUnavailableError(f"failed to load StreetCLIP: {exc}") from exc
            self._load_error = None
            self._loaded = True

    def health(self) -> dict[str, Any]:
        ready = bool(self._predictor is not None and self._predictor.ready)
        device = self._predictor.device if self._predictor is not None else "unloaded"
        payload: dict[str, Any] = {
            "ready": ready,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "device": device,
            "countries": len(self._countries),
        }
        if self._load_error:
            payload["detail"] = "model unavailable; inspect local server logs"
        return payload

    def predict(self, payload: bytes) -> dict[str, Any]:
        """Validate then predict under a single-inference semaphore.

        Validation happens before the semaphore is acquired so malformed or
        oversized uploads never occupy the GPU slot.
        """
        validate_image_bytes(payload, max_bytes=self.settings.max_upload_bytes)
        if self._predictor is None or not self._predictor.ready:
            raise PredictorUnavailableError("model is not ready")
        with self._predict_lock:
            result = self._predictor.predict(payload)
        return result


__all__ = [
    "DecodeError",
    "EmptyImageError",
    "ExifGpsPresentError",
    "ImageTooLargeError",
    "ImageValidationError",
    "StreetClipService",
]
