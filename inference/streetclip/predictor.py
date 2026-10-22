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
    DEFAULT_BATCH_PROCESSOR_CHUNK,
    DEFAULT_CITY_PROMPT_CHUNK,
    MAX_BATCH_PROCESSOR_CHUNK,
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
from services.city_scene_contract import (
    MAX_CITY_SCENE_CITIES,
    MAX_CITY_SCENE_CLUES,
    MAX_CITY_SCENE_PROMPTS,
    normalize_city_scene_clue,
    render_city_scene_prompts,
)

logger = logging.getLogger(__name__)

MAX_CITY_SCENE_PROTOTYPE_CACHE_ENTRIES = 32


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
        self._city_cache: "OrderedDict[tuple[str, tuple[int, ...]], np.ndarray]" = (
            OrderedDict()
        )
        self._city_locks: dict[str, threading.Lock] = {}
        self._city_cache_limit = max(1, int(CITY_TEXT_FEATURE_CACHE_COUNTRIES))
        self._scene_cache_lock = threading.Lock()
        self._scene_text_cache: tuple[np.ndarray, ...] | None = None
        self._city_scene_cache_lock = threading.Lock()
        self._city_scene_compute_lock = threading.Lock()
        self._city_scene_cache: "OrderedDict[tuple[Any, ...], np.ndarray]" = (
            OrderedDict()
        )
        self._city_scene_cache_limit = MAX_CITY_SCENE_PROTOTYPE_CACHE_ENTRIES

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

        Single-image callers use the same bounded batch path with a one-image
        list, so their exact embedding computation and OOM fallback remain
        consistent with ``encode_images``.
        """
        return self.encode_images([image])[0]

    @staticmethod
    def _is_out_of_memory(exc: BaseException, torch: Any) -> bool:
        oom_type = getattr(torch, "OutOfMemoryError", None)
        if isinstance(oom_type, type) and isinstance(exc, oom_type):
            return True
        message = str(exc).lower()
        return "out of memory" in message or "not enough memory" in message

    @staticmethod
    def _release_inference_memory(torch: Any) -> None:
        """Best-effort cleanup before retrying a smaller image microbatch."""
        import gc  # noqa: PLC0415 - only needed on the exceptional path

        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - defensive cleanup only
            return

    def _encode_image_batch_once(self, images: list[Any], torch: Any) -> np.ndarray:
        """Run one processor batch and exactly one image-feature model call."""
        inputs: dict[str, Any] | None = None
        output: Any = None
        normalized: Any = None
        values: Any = None
        try:
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = self._model.get_image_features(**inputs)
                normalized = torch.nn.functional.normalize(output, p=2, dim=-1)
                values = normalized.detach().to("cpu").to(torch.float32).numpy()
            array = np.ascontiguousarray(values, dtype=np.float32)
            if (
                array.ndim != 2
                or array.shape[0] != len(images)
                or array.shape[1] == 0
            ):
                raise PredictorUnavailableError(
                    "StreetCLIP returned an invalid image embedding batch"
                )
            return array
        finally:
            # Do not let a failed OOM attempt retain tensors through its
            # traceback while the caller retries a smaller microbatch.
            inputs = None
            output = None
            normalized = None
            values = None

    def _encode_image_batch_with_oom_fallback(
        self,
        images: list[Any],
        torch: Any,
    ) -> np.ndarray:
        """Encode one microbatch, recursively halving it only on OOM."""
        try:
            return self._encode_image_batch_once(images, torch)
        except PredictorUnavailableError:
            raise
        except Exception as exc:  # pragma: no cover - requires real weights
            if not self._is_out_of_memory(exc, torch):
                raise
            self._release_inference_memory(torch)
            if len(images) <= 1:
                raise PredictorUnavailableError(
                    "StreetCLIP image encoding ran out of memory for one image"
                ) from exc
            split_at = len(images) // 2
            logger.warning(
                "StreetCLIP image batch OOM size=%d; retrying as %d+%d",
                len(images),
                split_at,
                len(images) - split_at,
            )
            first = self._encode_image_batch_with_oom_fallback(
                images[:split_at], torch
            )
            second = self._encode_image_batch_with_oom_fallback(
                images[split_at:], torch
            )
            return np.ascontiguousarray(
                np.concatenate((first, second), axis=0), dtype=np.float32
            )

    def encode_images(self, images: list[Any]) -> np.ndarray:
        """PIL images -> contiguous CPU L2-normalized float32 embedding rows.

        The processor receives real image batches and each bounded microbatch
        invokes ``get_image_features`` once. The configured processor chunk
        bounds peak tensor size; only an actual OOM retries a split microbatch,
        recursively down to one image. A one-image OOM is reported as an
        unavailable predictor instead of being hidden or returning a fake row.
        """
        if not self._ready or self._model is None:
            raise PredictorUnavailableError("model is not loaded")
        if not isinstance(images, list) or not images:
            raise ValueError("images must be a non-empty list")
        torch = self._torch_module()
        try:
            configured_chunk = int(
                getattr(
                    self.settings,
                    "batch_processor_chunk",
                    DEFAULT_BATCH_PROCESSOR_CHUNK,
                )
            )
        except (TypeError, ValueError):
            configured_chunk = DEFAULT_BATCH_PROCESSOR_CHUNK
        chunk_size = min(
            MAX_BATCH_PROCESSOR_CHUNK,
            max(1, configured_chunk),
        )

        try:
            blocks: list[np.ndarray] = []
            with self._inference_lock:
                for start in range(0, len(images), chunk_size):
                    chunk = images[start : start + chunk_size]
                    block = self._encode_image_batch_with_oom_fallback(chunk, torch)
                    if block.shape[0] != len(chunk):
                        raise PredictorUnavailableError(
                            "StreetCLIP image batch row count mismatch"
                        )
                    blocks.append(block)
            combined = np.ascontiguousarray(
                np.concatenate(blocks, axis=0), dtype=np.float32
            )
        except PredictorUnavailableError:
            raise
        except Exception as exc:  # pragma: no cover - requires real weights
            raise PredictorUnavailableError(
                f"StreetCLIP image batch encoding failed: {exc}"
            ) from exc
        if (
            combined.ndim != 2
            or combined.shape[0] != len(images)
            or combined.shape[1] == 0
        ):
            raise PredictorUnavailableError(
                "StreetCLIP returned an invalid image embedding batch"
            )
        return combined

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
        """Return normalized per-city prototypes, cached per exact city set."""
        iso = country_iso.strip().upper()
        city_ids = tuple(city.city_id for city in cities)
        cache_key = (iso, city_ids)
        with self._city_cache_lock:
            cached = self._city_cache.get(cache_key)
            if cached is not None:
                self._city_cache.move_to_end(cache_key)
                return cached
            lock = self._city_locks.setdefault(iso, threading.Lock())
        with lock:
            with self._city_cache_lock:
                cached = self._city_cache.get(cache_key)
                if cached is not None:
                    self._city_cache.move_to_end(cache_key)
                    return cached
            prompts: list[str] = []
            for city in cities:
                prompts.extend(render_city_prompts(city, country_name))
            flat = self._encode_text_prompts(prompts)
            per_city = flat.reshape(len(cities), city_prompt_count(), -1)
            averaged = per_city.mean(axis=1)
            prototypes = _l2_normalize_numpy(averaged)
            with self._city_cache_lock:
                self._city_cache[cache_key] = prototypes
                self._city_cache.move_to_end(cache_key)
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

    def _city_scene_text_features(
        self,
        country_iso: str,
        cities: tuple[City, ...],
        country_name: str,
        clues: tuple[tuple[str, str], ...],
    ) -> np.ndarray:
        """Return bounded conditioned city-scene prototypes from a fixed LRU.

        This cache is independent from ``_city_cache``. Each entry stores one
        ``(clue, city, embedding_dimension)`` matrix for the exact ordered city
        and clue set; at most 32 entries are retained. Prompt text is always
        the fixed contract description plus the catalog's original city and
        country names, never caller-provided free text.
        """
        if not self._ready or self._model is None:
            raise PredictorUnavailableError("model is not loaded")
        if not cities or len(cities) > MAX_CITY_SCENE_CITIES:
            raise ValueError("invalid city-scene city count")
        if not clues or len(clues) > MAX_CITY_SCENE_CLUES:
            raise ValueError("invalid city-scene clue count")

        iso = country_iso.strip().upper()
        city_ids = tuple(city.city_id for city in cities)
        cache_key = (iso, country_name, city_ids, clues)
        with self._city_scene_cache_lock:
            cached = self._city_scene_cache.get(cache_key)
            if cached is not None:
                self._city_scene_cache.move_to_end(cache_key)
                return cached

        # Avoid duplicate text-forward work when direct predictor callers race.
        # The service already serializes this work with its prediction lock.
        with self._city_scene_compute_lock:
            with self._city_scene_cache_lock:
                cached = self._city_scene_cache.get(cache_key)
                if cached is not None:
                    self._city_scene_cache.move_to_end(cache_key)
                    return cached

            prompts: list[str] = []
            for city in cities:
                for dimension, value in clues:
                    prompts.extend(
                        render_city_scene_prompts(
                            city.name,
                            country_name,
                            (dimension, value),
                        )
                    )
            expected_prompt_count = len(cities) * len(clues) * 2
            if (
                expected_prompt_count > MAX_CITY_SCENE_PROMPTS
                or len(prompts) != expected_prompt_count
            ):
                raise PredictorUnavailableError(
                    "city-scene prompt count is outside the bounded contract"
                )

            flat = np.asarray(self._encode_text_prompts(prompts), dtype=np.float32)
            if (
                flat.ndim != 2
                or flat.shape[0] != expected_prompt_count
                or flat.shape[1] == 0
                or not bool(np.isfinite(flat).all())
            ):
                raise PredictorUnavailableError(
                    "StreetCLIP returned invalid city-scene text embeddings"
                )

            per_city_clue = flat.reshape(
                len(cities), len(clues), 2, flat.shape[1]
            )
            averaged = per_city_clue.mean(axis=2, dtype=np.float32)
            normalized = _l2_normalize_numpy(averaged)
            # Score in clue-major order: (clue, city, embedding_dimension).
            prototypes = np.ascontiguousarray(
                np.transpose(normalized, (1, 0, 2)), dtype=np.float32
            )
            with self._city_scene_cache_lock:
                self._city_scene_cache[cache_key] = prototypes
                self._city_scene_cache.move_to_end(cache_key)
                while len(self._city_scene_cache) > self._city_scene_cache_limit:
                    self._city_scene_cache.popitem(last=False)
            return prototypes

    def city_scene_scores(
        self,
        image_vector: np.ndarray,
        country_iso: str,
        cities: tuple[City, ...],
        countryname: str,
        clues: tuple[tuple[str, str], ...],
    ) -> list[dict[str, Any]]:
        """Return experimental city scene-condition cosine scores.

        Each clue carries a conditioned cosine from two fixed street-scene
        prompts that combine the fixed English condition description with the
        catalog's original city and country names. ``neutral_cosine`` is the
        existing city text-prototype cosine from ``city_text_scores``. Both are
        model similarities in ``[-1, 1]``, not city-attribute verification,
        probabilities or validated accuracy claims.
        """
        if not self._ready or self._model is None:
            raise PredictorUnavailableError("model is not loaded")
        if not isinstance(cities, (tuple, list)) or not 1 <= len(cities) <= MAX_CITY_SCENE_CITIES:
            raise ValueError("invalid city-scene cities")
        if not isinstance(clues, (tuple, list)) or not 1 <= len(clues) <= MAX_CITY_SCENE_CLUES:
            raise ValueError("invalid city-scene clues")
        cities = tuple(cities)

        iso = str(country_iso or "").strip().upper()
        if len(iso) != 2 or not iso.isalpha():
            raise ValueError("invalid city-scene country_iso")
        country_name = " ".join(str(countryname or "").split())
        if not country_name:
            raise ValueError("invalid city-scene country name")

        seen_city_ids: set[int] = set()
        for city in cities:
            if isinstance(city.city_id, bool) or city.city_id <= 0:
                raise ValueError("invalid city-scene city id")
            if city.city_id in seen_city_ids:
                raise ValueError("duplicate city-scene city id")
            seen_city_ids.add(city.city_id)
            if city.iso.strip().upper() != iso:
                raise ValueError("city-scene city is outside the requested country")

        normalized_clues: list[tuple[str, str]] = []
        for clue in clues:
            if not isinstance(clue, (tuple, list)) or len(clue) != 2:
                raise ValueError("invalid city-scene clue pair")
            pair = normalize_city_scene_clue(clue[0], clue[1])
            if pair is None:
                raise ValueError("unknown city-scene clue")
            normalized_clues.append(pair)
        canonical_clues = tuple(normalized_clues)

        try:
            query = np.asarray(image_vector, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise PredictorUnavailableError(
                "city-scene scoring requires a numeric image vector"
            ) from exc
        if query.ndim != 1 or query.size == 0:
            raise PredictorUnavailableError(
                "city-scene scoring requires a 1-D image vector"
            )
        norm = float(np.linalg.norm(query.astype(np.float64)))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise PredictorUnavailableError(
                "city-scene scoring requires a non-zero finite image vector"
            )
        query = query / np.float32(norm)

        neutral_rows = self.city_text_scores(
            query,
            iso,
            cities,
            country_name,
        )
        expected_ids = tuple(city.city_id for city in cities)
        if tuple(city_id for city_id, _ in neutral_rows) != expected_ids:
            raise PredictorUnavailableError(
                "neutral city-scene score IDs do not match the selected cities"
            )
        neutral_by_id: dict[int, float] = {}
        for city_id, neutral_cosine in neutral_rows:
            value = float(neutral_cosine)
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise PredictorUnavailableError(
                    "neutral city-scene cosine is non-finite or out of range"
                )
            neutral_by_id[city_id] = value

        prototypes = self._city_scene_text_features(
            iso,
            cities,
            country_name,
            canonical_clues,
        )
        if (
            prototypes.ndim != 3
            or prototypes.shape[0] != len(canonical_clues)
            or prototypes.shape[1] != len(cities)
            or prototypes.shape[2] != query.shape[0]
        ):
            raise PredictorUnavailableError(
                "city-scene text feature shape does not match the request"
            )
        conditioned = prototypes @ query
        if conditioned.shape != (len(canonical_clues), len(cities)):
            raise PredictorUnavailableError(
                "city-scene conditioned score shape mismatch"
            )
        if not bool(np.isfinite(conditioned).all()):
            raise PredictorUnavailableError("city-scene cosine is non-finite")

        results: list[dict[str, Any]] = []
        for clue_index, (dimension, value) in enumerate(canonical_clues):
            scores: list[dict[str, Any]] = []
            for city_index, city in enumerate(cities):
                raw_cosine = float(
                    np.clip(conditioned[clue_index, city_index], -1.0, 1.0)
                )
                neutral_cosine = neutral_by_id[city.city_id]
                if not -1.0 <= raw_cosine <= 1.0:
                    raise PredictorUnavailableError(
                        "conditioned city-scene cosine is out of range"
                    )
                scores.append(
                    {
                        "city_id": city.city_id,
                        "raw_cosine": raw_cosine,
                        "neutral_cosine": neutral_cosine,
                    }
                )
            results.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "scores": scores,
                }
            )
        return results

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
        with self._city_scene_cache_lock:
            self._city_scene_cache.clear()
        torch = self._torch
        self._torch = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "MAX_CITY_SCENE_PROTOTYPE_CACHE_ENTRIES",
    "Predictor",
    "PredictorUnavailableError",
    "SceneAnalysisUnavailableError",
    "StreetClipPredictor",
]

