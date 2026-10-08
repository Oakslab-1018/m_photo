"""Bounded predictor abstraction for StreetCLIP.

The interface is torch-free. The real implementation loads the pinned
``geolocal/StreetCLIP`` revision exactly once and precomputes one normalized
text embedding per country (three normalized templates averaged and
renormalized). Tests inject a stub predictor instead of loading weights.
"""

from __future__ import annotations

import logging
import threading
from contextlib import nullcontext
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

from .catalog import Country, compute_catalog_revision, render_prompts
from .config import MODEL_ID, MODEL_REVISION, PROMPT_TEMPLATE_VERSION
from .schema import build_prediction

logger = logging.getLogger(__name__)


class PredictorUnavailableError(RuntimeError):
    """Raised when the model cannot be loaded or executed safely."""


@runtime_checkable
class Predictor(Protocol):
    """Minimal structural interface used by the FastAPI application."""

    @property
    def ready(self) -> bool:
        ...

    @property
    def device(self) -> str:
        ...

    @property
    def catalog_revision(self) -> str:
        ...

    def predict(self, payload: bytes) -> dict[str, Any]:
        ...


class StreetClipPredictor:
    """Real CUDA predictor; model and country text embeddings load once.

    Construction is cheap and torch-free. ``load()`` imports torch and
    transformers lazily, so importing this module never requires GPU packages.
    """

    def __init__(
        self,
        settings: Any,
        *,
        countries: list[Country],
        catalog_revision: str | None = None,
    ) -> None:
        self.settings = settings
        self.countries = list(countries)
        self._catalog_revision = catalog_revision or compute_catalog_revision(self.countries)
        self._ready = False
        self._device = "unloaded"
        self._model: Any = None
        self._processor: Any = None
        self._text_embeddings: Any = None
        self._torch: Any = None
        self._inference_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def device(self) -> str:
        return self._device

    @property
    def catalog_revision(self) -> str:
        return self._catalog_revision

    def _torch_device(self) -> Any:
        import torch  # noqa: PLC0415 - intentionally lazy

        requested = (self.settings.device or "cuda").strip().lower()
        if requested in {"auto", ""}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise PredictorUnavailableError("CUDA requested but torch.cuda.is_available() is false")
        return device

    def load(self) -> None:
        """Load model weights and precompute country text embeddings once."""
        if self._ready:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise PredictorUnavailableError(
                "torch/transformers are not installed in this interpreter"
            ) from exc

        device = self._torch_device()
        try:
            model = CLIPModel.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                cache_dir=str(self.settings.storage_dir / "huggingface"),
                use_safetensors=False,
            )
            processor = CLIPProcessor.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                cache_dir=str(self.settings.storage_dir / "huggingface"),
                use_fast=False,
            )
            model.to(device)
            model.eval()
        except Exception as exc:  # pragma: no cover - requires real weights
            raise PredictorUnavailableError(f"failed to load pinned StreetCLIP model: {exc}") from exc

        try:
            embeddings = self._precompute_text_embeddings(model, processor, device)
        except Exception as exc:  # pragma: no cover - requires real weights
            raise PredictorUnavailableError(f"failed to precompute country text embeddings: {exc}") from exc

        self._torch = torch
        self._model = model
        self._processor = processor
        self._text_embeddings = embeddings
        self._device = str(device)
        self._ready = True
        logger.info(
            "StreetCLIP ready model=%s revision=%s device=%s countries=%d catalog_revision=%s",
            MODEL_ID,
            MODEL_REVISION,
            self._device,
            len(self.countries),
            self._catalog_revision,
        )

    def _precompute_text_embeddings(self, model: Any, processor: Any, device: Any) -> Any:
        torch = self._torch_module()
        prompts: list[str] = []
        for country in self.countries:
            prompts.extend(render_prompts(country))
        inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.get_text_features(**inputs)
        normalized = torch.nn.functional.normalize(output, p=2, dim=-1)
        template_count = len(prompts) // len(self.countries)
        shaped = normalized.reshape(len(self.countries), template_count, -1)
        averaged = shaped.mean(dim=1)
        return torch.nn.functional.normalize(averaged, p=2, dim=-1).contiguous()

    def _torch_module(self) -> Any:
        if self._torch is not None:
            return self._torch
        import torch  # noqa: PLC0415 - intentionally lazy

        self._torch = torch
        return torch

    def predict(self, payload: bytes) -> dict[str, Any]:
        """Return the strict public ranking contract for one image."""
        if not self._ready or self._model is None:
            raise PredictorUnavailableError("model is not loaded")
        try:
            from .schema import decode_image_bytes
        except Exception as exc:  # pragma: no cover - defensive
            raise PredictorUnavailableError("image decoder is unavailable") from exc

        image = decode_image_bytes(payload)
        torch = self._torch_module()
        try:
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with self._inference_lock, torch.inference_mode():
                output = self._model.get_image_features(**inputs)
                normalized = torch.nn.functional.normalize(output, p=2, dim=-1)
                similarities = (normalized @ self._text_embeddings.T).squeeze(0)
            values = similarities.detach().to("cpu").tolist()
            rankings = [(country.iso, float(value)) for country, value in zip(self.countries, values)]
            prediction = build_prediction(
                catalog_revision=self._catalog_revision,
                rankings=rankings,
            )
            return prediction.as_dict()
        except PredictorUnavailableError:
            raise
        except Exception as exc:
            raise PredictorUnavailableError(f"StreetCLIP inference failed: {exc}") from exc

    def close(self) -> None:
        self._ready = False
        self._model = None
        self._processor = None
        self._text_embeddings = None
        self._device = "unloaded"
        torch = self._torch
        self._torch = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "Predictor",
    "PredictorUnavailableError",
    "StreetClipPredictor",
]
