"""Bounded predictor abstraction for StreetCLIP.

The interface is torch-free. The real implementation loads the pinned
``geolocal/StreetCLIP`` revision exactly once and precomputes one normalized
text embedding per country (three normalized templates averaged and
renormalized). Tests inject a stub predictor instead of loading weights.

``encode_image`` is the single shared, thread-safe image encoder used by the
country ranking, the city text-prototype ranking and the optional labeled
reference builder. It always returns a CPU ``float32`` L2-normalized
``numpy.ndarray`` so downstream pure-NumPy code never needs torch.

City text features are computed on demand for the selected country only, then
kept in a small bounded LRU (never embeddings for the ~30k global cities at
once). Concurrent requests for different countries do not serialize text
prototype computation beyond the single GPU inference lock.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .catalog import Country, compute_catalog_revision, render_prompts
from .cities import City, city_prompt_count, render_city_prompts
from .config import (
    CITY_TEXT_FEATURE_CACHE_COUNTRIES,
    DEFAULT_CITY_PROMPT_CHUNK,
    MODEL_ID,
    MODEL_REVISION,
    PROMPT_TEMPLATE_VERSION,
)
from .scene import (
    SCENE_GROUPS,
    SCENE_MAX_CANDIDATES,
    SCENE_UNCERTAIN_MARGIN_THRESHOLD,
    iter_scene_prompts,
)
from .schema import build_prediction, build_scene_analysis

logger = logging.getLogger(__name__)


class PredictorUnavailableError(RuntimeError):
    """Raised when the model cannot be loaded or executed safely."""


class SceneAnalysisUnavailableError(PredictorUnavailableError):
    """Raised when the real scene-analysis capability cannot run."""


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


def _l2_normalize_numpy(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


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
        self._city_cache_lock = threading.Lock()
        self._city_cache: "OrderedDict[str, tuple[tuple[int, ...], np.ndarray]]" = OrderedDict()
        self._city_locks: dict[str, threading.Lock] = {}
        self._city_cache_limit = max(1, int(CITY_TEXT_FEATURE_CACHE_COUNTRIES))
        self._scene_cache_lock = threading.Lock()
        self._scene_text_cache: tuple[np.ndarray, ...] | None = None

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def device(self) -> str:
        return self._device

    @property
    def catalog_revision(self) -> str:
        return self._catalog_revision

    @property
    def dimension(self) -> int | None:
        """Embedding dimension once loaded, else ``None``."""
        embeddings = self._text_embeddings
        if embeddings is None:
            return None
        try:
            return int(embeddings.shape[-1])
        except Exception:  # pragma: no cover - defensive
            return None

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

    def encode_image(self, image: Any) -> np.ndarray:
        """Thread-safe shared encoder: PIL image -> CPU normalized float32 vector.

        This is the exact hook used by the country ranking, the city ranking and
        the offline reference index builder. It performs one model forward pass
        per image under the single inference lock and always detaches to a
        contiguous CPU array before returning.
        """
        if not self._ready or self._model is None:
            raise PredictorUnavailableError("model is not loaded")
        torch = self._torch_module()
        try:
            with self._inference_lock, torch.inference_mode():
                inputs = self._processor(images=image, return_tensors="pt")
                inputs = {key: value.to(self._device) for key, value in inputs.items()}
                output = self._model.get_image_features(**inputs)
                normalized = torch.nn.functional.normalize(output, p=2, dim=-1)
                vector = (
                    normalized.squeeze(0)
                    .detach()
                    .to("cpu")
                    .to(torch.float32)
                    .numpy()
                )
        except PredictorUnavailableError:
            raise
        except Exception as exc:  # pragma: no cover - requires real weights
            raise PredictorUnavailableError(f"StreetCLIP image encoding failed: {exc}") from exc
        array = np.ascontiguousarray(vector, dtype=np.float32)
        if array.ndim != 1 or array.size == 0:
            raise PredictorUnavailableError("StreetCLIP returned an invalid image embedding")
        return array

    def _encode_text_prompts(self, prompts: list[str]) -> np.ndarray:
        """Encode text prompts in bounded chunks; CPU normalized float32 rows."""
        if not self._ready or self._model is None:
            raise PredictorUnavailableError("model is not loaded")
        torch = self._torch_module()
        chunk_size = max(1, int(getattr(self.settings, "city_prompt_chunk", DEFAULT_CITY_PROMPT_CHUNK)))
        blocks: list[np.ndarray] = []
        try:
            for start in range(0, len(prompts), chunk_size):
                chunk = prompts[start : start + chunk_size]
                with self._inference_lock, torch.inference_mode():
                    inputs = self._processor(
                        text=chunk, return_tensors="pt", padding=True, truncation=True
                    )
                    inputs = {key: value.to(self._device) for key, value in inputs.items()}
                    output = self._model.get_text_features(**inputs)
                    normalized = torch.nn.functional.normalize(output, p=2, dim=-1)
                    blocks.append(
                        normalized.detach().to("cpu").to(torch.float32).numpy().astype(
                            np.float32, copy=True
                        )
                    )
        except PredictorUnavailableError:
            raise
        except Exception as exc:  # pragma: no cover - requires real weights
            raise PredictorUnavailableError(f"StreetCLIP text encoding failed: {exc}") from exc
        if not blocks:
            raise PredictorUnavailableError("no text prompts were rendered")
        return np.concatenate(blocks, axis=0)

    def _city_text_features(
        self,
        country_iso: str,
        cities: tuple[City, ...],
        country_name: str,
    ) -> np.ndarray:
        """Return normalized per-city prototypes, cached per country (bounded)."""
        iso = country_iso.strip().upper()
        city_ids = tuple(city.city_id for city in cities)
        with self._city_cache_lock:
            cached = self._city_cache.get(iso)
            if cached is not None and cached[0] == city_ids:
                self._city_cache.move_to_end(iso)
                return cached[1]
            lock = self._city_locks.setdefault(iso, threading.Lock())
        with lock:
            with self._city_cache_lock:
                cached = self._city_cache.get(iso)
                if cached is not None and cached[0] == city_ids:
                    self._city_cache.move_to_end(iso)
                    return cached[1]
            prompts: list[str] = []
            for city in cities:
                prompts.extend(render_city_prompts(city, country_name))
            flat = self._encode_text_prompts(prompts)
            per_city = flat.reshape(len(cities), city_prompt_count(), -1)
            averaged = per_city.mean(axis=1)
            prototypes = _l2_normalize_numpy(averaged)
            with self._city_cache_lock:
                self._city_cache[iso] = (city_ids, prototypes)
                self._city_cache.move_to_end(iso)
                while len(self._city_cache) > self._city_cache_limit:
                    self._city_cache.popitem(last=False)
            return prototypes

    def _scene_text_features(self) -> tuple[np.ndarray, ...]:
        """Return normalized scene option prototypes, cached lazily once.

        Prompts are flattened in the fixed group/option order exposed by
        ``scene.py``. The first call performs one bounded text batch (or chunked
        batch) under the shared inference lock; later calls only reuse the
        normalized NumPy rows and perform the image/text dot product.
        """
        if not self._ready or self._model is None:
            raise SceneAnalysisUnavailableError("model is not loaded")
        with self._scene_cache_lock:
            cached = self._scene_text_cache
            if cached is not None:
                return cached

            prompts = list(iter_scene_prompts())
            if not prompts:
                raise SceneAnalysisUnavailableError("no curated scene prompts were defined")
            try:
                flat = self._encode_text_prompts(prompts)
            except PredictorUnavailableError as exc:
                raise SceneAnalysisUnavailableError(
                    f"StreetCLIP scene text encoding failed: {exc}"
                ) from exc
            if flat.ndim != 2 or flat.shape[0] != len(prompts):
                raise SceneAnalysisUnavailableError(
                    "StreetCLIP returned an invalid scene text embedding matrix"
                )

            group_prototypes: list[np.ndarray] = []
            cursor = 0
            for group in SCENE_GROUPS:
                option_prototypes: list[np.ndarray] = []
                for option in group.options:
                    count = len(option.prompts)
                    rows = flat[cursor : cursor + count]
                    cursor += count
                    if count <= 0 or rows.shape[0] != count:
                        raise SceneAnalysisUnavailableError(
                            f"scene prompt rows are incomplete for {option.value}"
                        )
                    averaged = rows.mean(axis=0, dtype=np.float32)
                    prototype = _l2_normalize_numpy(averaged.reshape(1, -1))[0]
                    option_prototypes.append(prototype)
                group_prototypes.append(
                    np.ascontiguousarray(np.stack(option_prototypes, axis=0), dtype=np.float32)
                )
            if cursor != len(prompts):
                raise SceneAnalysisUnavailableError(
                    "StreetCLIP scene text embeddings do not match the curated prompts"
                )
            resolved = tuple(group_prototypes)
            self._scene_text_cache = resolved
            return resolved

    def scene_text_scores(self, image_vector: np.ndarray) -> list[dict[str, Any]]:
        """Return bounded top-two scene hypotheses for all fixed dimensions.

        The image vector must be the shared :meth:`encode_image` output. Text
        prototypes are loaded lazily and cached; scoring itself is a single
        normalized image/text matrix product per group. Raw cosine values are
        semantic similarities, never calibrated probabilities.
        """
        if not self._ready or self._model is None:
            raise SceneAnalysisUnavailableError("model is not loaded")
        try:
            query = np.asarray(image_vector, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise SceneAnalysisUnavailableError(
                "scene scoring requires a numeric image vector"
            ) from exc
        if query.ndim != 1 or query.size == 0:
            raise SceneAnalysisUnavailableError("scene scoring requires a 1-D image vector")
        norm = float(np.linalg.norm(query.astype(np.float64)))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise SceneAnalysisUnavailableError(
                "scene scoring requires a non-zero finite image vector"
            )
        query = query / np.float32(norm)

        try:
            group_features = self._scene_text_features()
            if len(group_features) != len(SCENE_GROUPS):
                raise SceneAnalysisUnavailableError("scene text feature group count mismatch")
            groups: list[dict[str, Any]] = []
            for definition, prototypes in zip(SCENE_GROUPS, group_features):
                if prototypes.ndim != 2 or prototypes.shape[0] != len(definition.options):
                    raise SceneAnalysisUnavailableError(
                        f"scene text features do not match dimension {definition.dimension}"
                    )
                if prototypes.shape[1] != query.shape[0]:
                    raise SceneAnalysisUnavailableError(
                        "scene text feature dimension does not match the image encoder"
                    )
                raw_scores = prototypes @ query
                ranked: list[tuple[Any, float]] = []
                for option, raw_score in zip(definition.options, raw_scores):
                    score = float(raw_score)
                    if not math.isfinite(score):
                        raise SceneAnalysisUnavailableError(
                            f"scene cosine is non-finite for {option.value}"
                        )
                    ranked.append((option, min(1.0, max(-1.0, score))))
                ranked.sort(key=lambda item: (-item[1], item[0].value))
                top = ranked[:SCENE_MAX_CANDIDATES]
                margin = top[0][1] - top[1][1] if len(top) >= 2 else 0.0
                groups.append(
                    {
                        "dimension": definition.dimension,
                        "candidates": [
                            {
                                "value": option.value,
                                "label": option.label,
                                "raw_cosine": score,
                            }
                            for option, score in top
                        ],
                        "margin": max(0.0, float(margin)),
                        "uncertain": bool(margin < SCENE_UNCERTAIN_MARGIN_THRESHOLD),
                    }
                )
            return groups
        except SceneAnalysisUnavailableError:
            raise
        except PredictorUnavailableError as exc:
            raise SceneAnalysisUnavailableError(
                f"StreetCLIP scene analysis failed: {exc}"
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise SceneAnalysisUnavailableError(
                f"StreetCLIP scene analysis failed: {exc}"
            ) from exc

    def city_text_scores(
        self,
        image_vector: np.ndarray,
        country_iso: str,
        cities: tuple[City, ...],
        country_name: str,
    ) -> list[tuple[int, float]]:
        """Cosine similarity for each selected city's text prototype.

        Returns ``(city_id, raw_cosine)`` in the same order as ``cities``. The
        image vector must be the shared :meth:`encode_image` output.
        """
        if not cities:
            return []
        query = np.asarray(image_vector, dtype=np.float32)
        if query.ndim != 1:
            raise PredictorUnavailableError("city scoring requires a 1-D image vector")
        norm = float(np.linalg.norm(query.astype(np.float64)))
        if norm <= 1e-12:
            raise PredictorUnavailableError("city scoring requires a non-zero image vector")
        query = query / np.float32(norm)
        prototypes = self._city_text_features(country_iso, cities, country_name)
        if prototypes.shape[0] != len(cities):
            raise PredictorUnavailableError("city text feature count mismatch")
        if prototypes.shape[1] != query.shape[0]:
            raise PredictorUnavailableError(
                "city text feature dimension does not match the image encoder"
            )
        scores = prototypes @ query
        return [
            (city.city_id, float(np.clip(scores[index], -1.0, 1.0)))
            for index, city in enumerate(cities)
        ]

    def predict_from_vector(self, image_vector: np.ndarray) -> dict[str, Any]:
        """Country ranking from a shared :meth:`encode_image` vector.

        Used by the service so one image forward pass can serve both the
        country ranking and the optional reference-index search.
        """
        if not self._ready or self._model is None:
            raise PredictorUnavailableError("model is not loaded")
        if self._text_embeddings is None:
            raise PredictorUnavailableError("country text embeddings are not loaded")
        vector = np.asarray(image_vector, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise PredictorUnavailableError("country scoring requires a 1-D image vector")
        norm = float(np.linalg.norm(vector.astype(np.float64)))
        if norm <= 1e-12:
            raise PredictorUnavailableError("country scoring requires a non-zero image vector")
        vector = vector / np.float32(norm)
        torch = self._torch_module()
        try:
            with torch.inference_mode():
                query = torch.from_numpy(vector).to(self._device)
                similarities = (query @ self._text_embeddings.T).squeeze(0)
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

    def predict(self, payload: bytes) -> dict[str, Any]:
        """Return the strict public ranking contract for one image."""
        if not self._ready or self._model is None:
            raise PredictorUnavailableError("model is not loaded")
        from .schema import decode_image_bytes

        image = decode_image_bytes(payload)
        vector = self.encode_image(image)
        return self.predict_from_vector(vector)

    def analyze_region(self, payload: bytes) -> dict[str, Any]:
        """Return bounded scene hypotheses for one decoded region image.

        The service normally orchestrates the image decode and shared
        prediction lock. This convenience method exposes the same single-image
        path for direct callers without creating a nested lock.
        """
        if not self._ready or self._model is None:
            raise SceneAnalysisUnavailableError("model is not loaded")
        from .schema import compute_image_sha256, decode_image_bytes

        image = decode_image_bytes(payload)
        vector = self.encode_image(image)
        groups = self.scene_text_scores(vector)
        return build_scene_analysis(
            image_sha256=compute_image_sha256(payload),
            scene_groups=groups,
        ).as_dict()

    def close(self) -> None:
        self._ready = False
        self._model = None
        self._processor = None
        self._text_embeddings = None
        self._device = "unloaded"
        with self._scene_cache_lock:
            self._scene_text_cache = None
        with self._city_cache_lock:
            self._city_cache.clear()
            self._city_locks.clear()
        torch = self._torch
        self._torch = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "Predictor",
    "PredictorUnavailableError",
    "SceneAnalysisUnavailableError",
    "StreetClipPredictor",
]

