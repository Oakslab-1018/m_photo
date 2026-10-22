"""Service orchestration: authentication-independent prediction and health.

The service owns one real predictor, the loaded country/city catalogs and the
optional labeled reference index. All GPU work is serialized by a single
prediction lock; catalog lookups and the pure-NumPy reference search are cheap
and stay outside that critical section. A missing or broken city catalog or
reference index never disables the fixed-country prediction path.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import math
import threading
from typing import Any

import numpy as np

from .catalog import Country, compute_catalog_revision
from .cities import (
    City,
    CityCatalogError,
    catalog_file_hash,
    load_cities,
    select_cities,
)
from .config import (
    DEFAULT_BATCH_MAX_AGGREGATE_BYTES,
    DEFAULT_BATCH_MAX_SIZE,
    MAX_BATCH_MAX_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    Settings,
)
from .predictor import (
    Predictor,
    PredictorUnavailableError,
    SceneAnalysisUnavailableError,
    StreetClipPredictor,
)
from services.city_scene_contract import (
    CITY_SCENE_VERSION,
    MAX_CITY_SCENE_CITIES,
    MAX_CITY_SCENE_CLUES,
    MAX_CITY_SCENE_PROMPTS,
    normalize_city_scene_clue,
)
from .schema import (
    DecodeError,
    EmptyImageError,
    ExifGpsPresentError,
    ImageTooLargeError,
    ImageValidationError,
    InvalidCountryError,
    build_city_prediction,
    build_scene_analysis,
    compute_image_sha256,
    decode_image_bytes,
    validate_image_bytes,
)

logger = logging.getLogger(__name__)

REFERENCE_TOP_K = 5
MAX_TARGET_CITY_IDS = 10


class InvalidCityIdsError(ValueError):
    """Raised when targeted scoring IDs are malformed or outside the catalog."""


class InvalidCitySceneCluesError(ValueError):
    """Raised when city-scene clues are outside the fixed bounded vocabulary."""


def _normalize_target_city_ids(city_ids: Any) -> tuple[int, ...]:
    """Validate unique positive target IDs in request order."""
    if isinstance(city_ids, (str, bytes, bytearray)):
        raise InvalidCityIdsError("city_ids must be an array of positive integers")
    try:
        values = tuple(city_ids)
    except TypeError as exc:
        raise InvalidCityIdsError("city_ids must be an array of positive integers") from exc
    if not 1 <= len(values) <= MAX_TARGET_CITY_IDS:
        raise InvalidCityIdsError(
            f"city_ids must contain 1 to {MAX_TARGET_CITY_IDS} unique positive integers"
        )
    normalized: list[int] = []
    seen: set[int] = set()
    for city_id in values:
        if isinstance(city_id, bool) or not isinstance(city_id, int) or city_id <= 0:
            raise InvalidCityIdsError("city_ids must contain positive integers")
        if city_id in seen:
            raise InvalidCityIdsError("city_ids must be unique")
        seen.add(city_id)
        normalized.append(city_id)
    return tuple(normalized)


def _normalize_city_scene_city_ids(city_ids: Any) -> tuple[int, ...]:
    """Validate 1..20 unique positive city IDs for scene-condition scoring."""
    if isinstance(city_ids, (str, bytes, bytearray)):
        raise InvalidCityIdsError("city_ids must be an array of positive integers")
    try:
        values = tuple(city_ids)
    except TypeError as exc:
        raise InvalidCityIdsError("city_ids must be an array of positive integers") from exc
    if not 1 <= len(values) <= MAX_CITY_SCENE_CITIES:
        raise InvalidCityIdsError(
            f"city_ids must contain 1 to {MAX_CITY_SCENE_CITIES} unique positive integers"
        )
    normalized: list[int] = []
    seen: set[int] = set()
    for city_id in values:
        if isinstance(city_id, bool) or not isinstance(city_id, int) or city_id <= 0:
            raise InvalidCityIdsError("city_ids must contain positive integers")
        if city_id in seen:
            raise InvalidCityIdsError("city_ids must be unique")
        seen.add(city_id)
        normalized.append(city_id)
    return tuple(normalized)


def _normalize_city_scene_clue_pairs(clues: Any) -> tuple[tuple[str, str], ...]:
    """Validate canonical ``(dimension, value)`` pairs without free text."""
    if not isinstance(clues, (list, tuple)):
        raise InvalidCitySceneCluesError("clues must be an array")
    if not 1 <= len(clues) <= MAX_CITY_SCENE_CLUES:
        raise InvalidCitySceneCluesError(
            f"clues must contain 1 to {MAX_CITY_SCENE_CLUES} entries"
        )
    normalized: list[tuple[str, str]] = []
    for clue in clues:
        if not isinstance(clue, (list, tuple)) or len(clue) != 2:
            raise InvalidCitySceneCluesError("each clue must be a dimension/value pair")
        pair = normalize_city_scene_clue(clue[0], clue[1])
        if pair is None:
            raise InvalidCitySceneCluesError("unknown clue dimension or value")
        normalized.append(pair)
    return tuple(normalized)


class StreetClipService:
    """Thread-safe service object owning one predictor and its readiness."""

    def __init__(
        self,
        settings: Settings,
        *,
        predictor: Predictor | None = None,
        reference_index: Any | None = None,
    ) -> None:
        self.settings = settings
        self._predictor = predictor
        self._countries: tuple[Country, ...] = ()
        self._country_by_iso: dict[str, Country] = {}
        self._catalog_revision = ""
        self._cities: dict[str, tuple[City, ...]] = {}
        self._cities_loaded = 0
        self._city_capability = False
        self._city_catalog_revision: str | None = None
        self._city_error: str | None = None
        self._reference_index: Any | None = reference_index
        self._reference_configured = bool(
            reference_index is not None or settings.reference_index_path is not None
        )
        self._reference_loaded = reference_index is not None
        self._reference_error: str | None = None
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

    @property
    def cities_loaded(self) -> int:
        return self._cities_loaded

    @property
    def city_capability(self) -> bool:
        return self._city_capability

    @property
    def city_catalog_revision(self) -> str | None:
        return self._city_catalog_revision

    @property
    def reference_loaded(self) -> bool:
        return self._reference_index is not None

    def _batch_max_size(self) -> int:
        try:
            configured = int(self.settings.batch_max_size)
        except (TypeError, ValueError):
            configured = DEFAULT_BATCH_MAX_SIZE
        if configured <= 0:
            configured = DEFAULT_BATCH_MAX_SIZE
        return min(MAX_BATCH_MAX_SIZE, configured)

    def _batch_aggregate_limit(self) -> int:
        try:
            configured = int(self.settings.batch_max_aggregate_bytes)
        except (TypeError, ValueError):
            configured = DEFAULT_BATCH_MAX_AGGREGATE_BYTES
        return configured if configured > 0 else DEFAULT_BATCH_MAX_AGGREGATE_BYTES

    @property
    def batch_max_size(self) -> int:
        return self._batch_max_size()

    @property
    def batch_max_aggregate_bytes(self) -> int:
        return self._batch_aggregate_limit()

    def load(self) -> None:
        """Resolve catalogs and load the real predictor exactly once.

        The country catalog is always read so readiness metadata (candidate
        list and ``catalog_revision``) is correct even when a predictor was
        injected. An injected predictor is used as-is and is never silently
        loaded behind the caller's back, so a not-ready injected predictor
        keeps ``/predict`` at 503 instead of scoring with fake weights. The
        city catalog and the optional reference index are best-effort: their
        failures degrade city/reference coverage without disabling the
        country service.
        """
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            from .catalog import _read_catalog

            countries = _read_catalog(self.settings.catalog_db_path)
            self._countries = tuple(countries)
            self._country_by_iso = {country.iso: country for country in countries}
            self._catalog_revision = compute_catalog_revision(countries)
            self._load_cities()
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
            self._load_reference_index()
            self._load_error = None
            self._loaded = True

    def _load_cities(self) -> None:
        path = self.settings.catalog_db_path
        try:
            grouped = load_cities(path)
            revision = catalog_file_hash(path)
        except CityCatalogError as exc:
            self._cities = {}
            self._cities_loaded = 0
            self._city_capability = False
            self._city_catalog_revision = None
            self._city_error = str(exc)
            logger.warning("city catalog unavailable; /predict-cities reports no_cities")
            return
        except Exception as exc:  # pragma: no cover - defensive
            self._cities = {}
            self._cities_loaded = 0
            self._city_capability = False
            self._city_catalog_revision = None
            self._city_error = type(exc).__name__
            logger.warning("city catalog failed to load: %s", type(exc).__name__)
            return
        self._cities = grouped
        self._cities_loaded = sum(len(values) for values in grouped.values())
        self._city_capability = self._cities_loaded > 0
        self._city_catalog_revision = revision
        self._city_error = None
        logger.info(
            "city catalog loaded cities=%d countries=%d revision=%s",
            self._cities_loaded,
            len(grouped),
            revision,
        )

    def _load_reference_index(self) -> None:
        """Load the configured ReferenceIndex via pure-library introspection.

        The inference service never re-implements index math; it only calls the
        separate ``reference.ReferenceIndex.load`` library, passing only the
        keyword arguments that library actually accepts. A configured but
        broken index is recorded as not loaded while the country service stays
        fully operational.
        """
        if self._reference_index is not None:
            self._reference_loaded = True
            self._reference_error = None
            return
        path = self.settings.reference_index_path
        if path is None:
            self._reference_configured = False
            self._reference_loaded = False
            return
        self._reference_configured = True
        try:
            module = importlib.import_module("inference.streetclip.reference")
            factory = getattr(module, "ReferenceIndex", None)
            loader = getattr(factory, "load", None)
            if factory is None or not callable(loader):
                raise RuntimeError("reference library has no ReferenceIndex.load")
            kwargs: dict[str, Any] = {}
            try:
                parameters = inspect.signature(loader).parameters
            except (TypeError, ValueError):  # pragma: no cover - builtins only
                parameters = {}
            if "model_revision" in parameters:
                kwargs["model_revision"] = MODEL_REVISION
            if "catalog_revision" in parameters:
                kwargs["catalog_revision"] = self._catalog_revision
            if "allowed_isos" in parameters:
                kwargs["allowed_isos"] = set(self._country_by_iso)
            dimension = getattr(self._predictor, "dimension", None)
            if "expected_dimension" in parameters and isinstance(dimension, int):
                kwargs["expected_dimension"] = dimension
            index = loader(str(path), **kwargs)
            if index is None:
                raise RuntimeError("reference library returned no index")
            self._reference_index = index
            self._reference_loaded = True
            self._reference_error = None
            logger.info("reference index loaded")
        except Exception as exc:
            self._reference_index = None
            self._reference_loaded = False
            self._reference_error = type(exc).__name__
            logger.warning("configured reference index unavailable: %s", type(exc).__name__)

    def _search_reference(self, vector: np.ndarray) -> dict[str, Any] | None:
        index = self._reference_index
        if index is None:
            return None
        search = getattr(index, "search", None)
        if not callable(search):
            return None
        try:
            try:
                sidecar = search(vector, top_k=REFERENCE_TOP_K)
            except TypeError:
                sidecar = search(vector, REFERENCE_TOP_K)
        except Exception:
            logger.warning("reference search failed; continuing without sidecar", exc_info=True)
            return None
        if isinstance(sidecar, dict):
            return sidecar
        as_dict = getattr(sidecar, "as_dict", None)
        if callable(as_dict):
            resolved = as_dict()
            return resolved if isinstance(resolved, dict) else None
        return None

    def health(self) -> dict[str, Any]:
        ready = bool(self._predictor is not None and self._predictor.ready)
        device = self._predictor.device if self._predictor is not None else "unloaded"
        country_batch = bool(
            self._predictor is not None
            and callable(getattr(self._predictor, "encode_images", None))
            and callable(getattr(self._predictor, "predict_from_vector", None))
        )
        payload: dict[str, Any] = {
            "ready": ready,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "device": device,
            "country_batch": country_batch,
            "batch_max_size": self.batch_max_size,
            "batch_max_aggregate_bytes": self.batch_max_aggregate_bytes,
            "countries": len(self._countries),
            "city_capability": self._city_capability,
            "cities_loaded": self._cities_loaded,
            "city_catalog_revision": self._city_catalog_revision,
            "city_limit_per_country": self.settings.city_limit_per_country,
            "reference_configured": self._reference_configured,
            "reference_loaded": self._reference_loaded,
        }
        if self._load_error:
            payload["detail"] = "model unavailable; inspect local server logs"
        if self._city_error:
            payload["city_detail"] = "city catalog unavailable; inspect local server logs"
        if self._reference_error and self._reference_configured:
            payload["reference_detail"] = "reference index unavailable; inspect local server logs"
        return payload

    def predict(self, payload: bytes) -> dict[str, Any]:
        """Validate then predict under a single-inference semaphore.

        Validation happens before the semaphore is acquired so malformed or
        oversized uploads never occupy the GPU slot. The ``reference`` sidecar
        is attached only when a reference index is loaded; otherwise the exact
        six-key country contract is preserved byte-for-byte.
        """
        validate_image_bytes(payload, max_bytes=self.settings.max_upload_bytes)
        if self._predictor is None or not self._predictor.ready:
            raise PredictorUnavailableError("model is not ready")

        vector: np.ndarray | None = None
        encode_image = getattr(self._predictor, "encode_image", None)
        predict_from_vector = getattr(self._predictor, "predict_from_vector", None)
        if self._reference_index is not None and callable(encode_image) and callable(predict_from_vector):
            image = decode_image_bytes(payload)
            with self._predict_lock:
                vector = encode_image(image)
                result = predict_from_vector(vector)
        else:
            with self._predict_lock:
                result = self._predictor.predict(payload)

        if self._reference_index is not None and vector is not None:
            sidecar = self._search_reference(vector)
            if sidecar is not None:
                result = dict(result)
                result["reference"] = sidecar
        return result

    def predict_batch(self, payloads: list[bytes]) -> dict[str, Any]:
        """Predict a bounded country batch under one inference critical section.

        Every item is size-validated, decoded, orientation-normalized and
        stripped of metadata before the GPU lock is acquired. The real
        predictor then performs bounded processor batches, and the existing
        per-vector country ranking preserves each single-prediction shape in
        input order. Optional reference search uses the same per-vector helper
        and remains outside the GPU lock.
        """
        if not isinstance(payloads, list) or not payloads:
            raise ImageValidationError("batch must contain at least one image")
        max_batch_size = self._batch_max_size()
        if len(payloads) > max_batch_size:
            raise ImageTooLargeError(
                f"batch contains {len(payloads)} images; maximum is {max_batch_size}"
            )
        for payload in payloads:
            if not isinstance(payload, bytes):
                raise ImageValidationError("each batch item must be image bytes")
        aggregate_bytes = sum(len(payload) for payload in payloads)
        aggregate_limit = self._batch_aggregate_limit()
        if aggregate_bytes > aggregate_limit:
            raise ImageTooLargeError(
                f"batch exceeds {aggregate_limit} aggregate byte limit"
            )

        for payload in payloads:
            validate_image_bytes(payload, max_bytes=self.settings.max_upload_bytes)
        if self._predictor is None or not self._predictor.ready:
            raise PredictorUnavailableError("model is not ready")

        images = [decode_image_bytes(payload) for payload in payloads]
        encode_images = getattr(self._predictor, "encode_images", None)
        predict_from_vector = getattr(self._predictor, "predict_from_vector", None)

        if callable(encode_images) and callable(predict_from_vector):
            # encode_images owns the predictor-level lock. predict_from_vector
            # is deliberately lock-free, so this path never nests that lock.
            with self._predict_lock:
                vectors = np.asarray(encode_images(images), dtype=np.float32)
                if (
                    vectors.ndim != 2
                    or vectors.shape[0] != len(payloads)
                    or vectors.shape[1] == 0
                ):
                    raise PredictorUnavailableError(
                        "StreetCLIP returned an invalid image embedding batch"
                    )
                results = [predict_from_vector(vector) for vector in vectors]

            if self._reference_index is not None:
                referenced_results: list[dict[str, Any]] = []
                for vector, result in zip(vectors, results):
                    sidecar = self._search_reference(vector)
                    if sidecar is not None:
                        result = dict(result)
                        result["reference"] = sidecar
                    referenced_results.append(result)
                results = referenced_results
            return {"results": results}

        # Compatibility path for injected predictors that only implement the
        # original single-image protocol. The real predictor always uses the
        # true batched path above.
        if self._reference_index is not None:
            raise PredictorUnavailableError(
                "batch reference sidecars require batched predictor support"
            )
        with self._predict_lock:
            results = [self._predictor.predict(payload) for payload in payloads]
        return {"results": results}

    def analyze_region(self, payload: bytes) -> dict[str, Any]:
        """Run bounded zero-shot scene hypotheses with the shared encoder.

        The uploaded image is validated and EXIF-orientation normalized once
        before the inference critical section. The same ``_predict_lock`` used
        by country and city requests then wraps the shared image embedding and
        cached text-prototype scoring. A predictor without the scene capability
        is reported as unavailable rather than scored with fabricated output.
        """
        validate_image_bytes(payload, max_bytes=self.settings.max_upload_bytes)
        if self._predictor is None or not self._predictor.ready:
            raise SceneAnalysisUnavailableError(
                "scene analysis requires a loaded StreetCLIP model"
            )

        encode_image = getattr(self._predictor, "encode_image", None)
        scene_text_scores = getattr(self._predictor, "scene_text_scores", None)
        if not callable(encode_image) or not callable(scene_text_scores):
            raise SceneAnalysisUnavailableError(
                "scene analysis capability is not available for this predictor"
            )

        image = decode_image_bytes(payload)
        try:
            with self._predict_lock:
                vector = encode_image(image)
                groups = scene_text_scores(vector)
        except SceneAnalysisUnavailableError:
            raise
        except PredictorUnavailableError as exc:
            raise SceneAnalysisUnavailableError(
                f"scene analysis failed: {exc}"
            ) from exc
        except Exception as exc:  # pragma: no cover - delegated capability boundary
            raise SceneAnalysisUnavailableError(
                f"scene analysis failed: {type(exc).__name__}"
            ) from exc

        try:
            return build_scene_analysis(
                image_sha256=compute_image_sha256(payload),
                scene_groups=groups,
            ).as_dict()
        except (TypeError, ValueError) as exc:
            raise SceneAnalysisUnavailableError(
                f"scene analysis returned invalid bounded results: {exc}"
            ) from exc

    def city_scene_scores(
        self,
        country_iso: str,
        payload: bytes,
        *,
        city_ids: tuple[int, ...],
        clues: tuple[tuple[str, str], ...],
    ) -> dict[str, Any]:
        """Score fixed scene conditions for exactly 1..20 catalog cities.

        This is experimental model matching, not city-attribute verification:
        it compares one image against the neutral city text prototypes and two
        fixed street-scene prompts per clue. The prompts contain only the
        catalog's original city/country names and a contract-owned English
        description. No LLM, caller prompt, city fact or coordinate is inferred,
        and the cosine values are not probabilities or accuracy claims.

        The image is encoded exactly once by the shared encoder while the
        existing ``_predict_lock`` is held. Every clue covers every requested
        city in request order.
        """
        iso = str(country_iso or "").strip().upper()
        if len(iso) != 2 or not iso.isalpha():
            raise InvalidCountryError("country_iso must be a two-letter ISO code")
        if iso not in self._country_by_iso:
            raise InvalidCountryError("unsupported country_iso")

        targeted_city_ids = _normalize_city_scene_city_ids(city_ids)
        canonical_clues = _normalize_city_scene_clue_pairs(clues)
        if (
            len(targeted_city_ids) * len(canonical_clues) * 2
            > MAX_CITY_SCENE_PROMPTS
        ):
            raise InvalidCitySceneCluesError(
                "city-scene prompt count exceeds the bounded contract"
            )

        validate_image_bytes(payload, max_bytes=self.settings.max_upload_bytes)

        catalog_cities = self._cities.get(iso, ())
        city_by_id = {city.city_id: city for city in catalog_cities}
        if any(city_id not in city_by_id for city_id in targeted_city_ids):
            raise InvalidCityIdsError(
                "city_ids must reference catalog cities in the requested country"
            )
        cities = tuple(city_by_id[city_id] for city_id in targeted_city_ids)
        if self._city_catalog_revision is None:
            raise PredictorUnavailableError("city catalog revision is unavailable")

        if self._predictor is None or not self._predictor.ready:
            raise PredictorUnavailableError("model is not ready")
        encode_image = getattr(self._predictor, "encode_image", None)
        city_scene_predictor = getattr(self._predictor, "city_scene_scores", None)
        if not callable(encode_image) or not callable(city_scene_predictor):
            raise PredictorUnavailableError(
                "city scene-condition scoring is unavailable"
            )

        image = decode_image_bytes(payload)
        try:
            with self._predict_lock:
                vector = encode_image(image)
                raw_results = city_scene_predictor(
                    vector,
                    iso,
                    cities,
                    self._country_by_iso[iso].name,
                    canonical_clues,
                )
        except PredictorUnavailableError:
            raise
        except Exception as exc:  # pragma: no cover - delegated capability boundary
            raise PredictorUnavailableError(
                f"city scene-condition scoring failed: {type(exc).__name__}"
            ) from exc

        if not isinstance(raw_results, (list, tuple)) or len(raw_results) != len(canonical_clues):
            raise PredictorUnavailableError(
                "city scene-condition clue count does not match the request"
            )

        expected_city_ids = tuple(city.city_id for city in cities)
        normalized_results: list[dict[str, Any]] = []
        for clue_index, (dimension, value) in enumerate(canonical_clues):
            row = raw_results[clue_index]
            if not isinstance(row, dict):
                raise PredictorUnavailableError(
                    "city scene-condition clue result is malformed"
                )
            if row.get("dimension") != dimension or row.get("value") != value:
                raise PredictorUnavailableError(
                    "city scene-condition clue identity does not match the request"
                )
            raw_scores = row.get("scores")
            if not isinstance(raw_scores, (list, tuple)) or len(raw_scores) != len(cities):
                raise PredictorUnavailableError(
                    "city scene-condition scores do not cover all requested cities"
                )
            scores: list[dict[str, Any]] = []
            for city_index, city in enumerate(cities):
                score_row = raw_scores[city_index]
                if not isinstance(score_row, dict):
                    raise PredictorUnavailableError(
                        "city scene-condition score row is malformed"
                    )
                if score_row.get("city_id") != expected_city_ids[city_index]:
                    raise PredictorUnavailableError(
                        "city scene-condition city order does not match the request"
                    )
                try:
                    raw_cosine = float(score_row.get("raw_cosine"))
                    neutral_cosine = float(score_row.get("neutral_cosine"))
                except (TypeError, ValueError) as exc:
                    raise PredictorUnavailableError(
                        "city scene-condition cosine is not numeric"
                    ) from exc
                if (
                    not math.isfinite(raw_cosine)
                    or not math.isfinite(neutral_cosine)
                    or not -1.0 <= raw_cosine <= 1.0
                    or not -1.0 <= neutral_cosine <= 1.0
                ):
                    raise PredictorUnavailableError(
                        "city scene-condition cosine is outside the bounded range"
                    )
                scores.append(
                    {
                        "city_id": city.city_id,
                        "raw_cosine": raw_cosine,
                        "neutral_cosine": neutral_cosine,
                    }
                )
            normalized_results.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "scores": scores,
                }
            )

        return {
            "country_iso": iso,
            "city_catalog_revision": self._city_catalog_revision,
            "model_revision": MODEL_REVISION,
            "prompt_version": CITY_SCENE_VERSION,
            "city_ids": list(targeted_city_ids),
            "clues": normalized_results,
        }

    def predict_cities(
        self,
        country_iso: str,
        payload: bytes,
        *,
        city_ids: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        """Rank the selected country's top cities by real image/text CLIP.

        Image bytes are validated first (same limits as ``/predict``), the
        country must be a supported ISO candidate, and only then is the single
        shared image encoder invoked. City identity, region, representative
        coordinates and the top-N selection all come from the GeoNames city
        catalog; the score is a bounded similarity, not a probability.

        When ``city_ids`` is supplied, the country-wide population limit is
        bypassed and only those unique catalog rows are scored. This supports
        verified cities outside the default top-N pool without inventing a
        score for an unscored candidate.
        """
        iso = str(country_iso or "").strip().upper()
        if len(iso) != 2 or not iso.isalpha():
            raise InvalidCountryError("country_iso must be a two-letter ISO code")
        if iso not in self._country_by_iso:
            raise InvalidCountryError("unsupported country_iso")
        targeted_city_ids = (
            _normalize_target_city_ids(city_ids) if city_ids is not None else None
        )
        validate_image_bytes(payload, max_bytes=self.settings.max_upload_bytes)

        limit = int(self.settings.city_limit_per_country)
        if targeted_city_ids is None:
            cities = select_cities(self._cities, iso, limit=limit)
        else:
            catalog_cities = self._cities.get(iso, ())
            city_by_id = {city.city_id: city for city in catalog_cities}
            if any(city_id not in city_by_id for city_id in targeted_city_ids):
                raise InvalidCityIdsError(
                    "city_ids must reference catalog cities in the requested country"
                )
            cities = tuple(city_by_id[city_id] for city_id in targeted_city_ids)

        if self._predictor is None or not self._predictor.ready:
            raise PredictorUnavailableError("model is not ready")
        if not cities:
            return build_city_prediction(
                country_iso=iso,
                city_catalog_revision=self._city_catalog_revision,
                city_limit_per_country=limit,
                candidates=[],
            ).as_dict()

        city_text_scores = getattr(self._predictor, "city_text_scores", None)
        encode_image = getattr(self._predictor, "encode_image", None)
        if not callable(city_text_scores) or not callable(encode_image):
            raise PredictorUnavailableError("city-level scoring is unavailable")

        image = decode_image_bytes(payload)
        with self._predict_lock:
            vector = encode_image(image)
            scores = city_text_scores(
                vector,
                iso,
                cities,
                self._country_by_iso[iso].name,
            )
        if len(scores) != len(cities):
            raise PredictorUnavailableError("city score count does not match the selected cities")
        expected_score_ids = tuple(city.city_id for city in cities)
        if tuple(city_id for city_id, _ in scores) != expected_score_ids:
            raise PredictorUnavailableError("city score IDs do not match the selected cities")

        reference_sidecar: dict[str, Any] | None = None
        if self._reference_index is not None:
            reference_sidecar = self._search_reference(vector)

        candidates = [
            {
                "city_id": city.city_id,
                "iso": city.iso,
                "name": city.name,
                "region": city.region,
                "lat": city.lat,
                "lng": city.lng,
                "raw_cosine": raw_cosine,
            }
            for city, (_, raw_cosine) in zip(cities, scores)
        ]
        return build_city_prediction(
            country_iso=iso,
            city_catalog_revision=self._city_catalog_revision,
            city_limit_per_country=limit,
            candidates=candidates,
            reference_matches=reference_sidecar,
        ).as_dict()


__all__ = [
    "DecodeError",
    "EmptyImageError",
    "ExifGpsPresentError",
    "ImageTooLargeError",
    "ImageValidationError",
    "InvalidCityIdsError",
    "InvalidCitySceneCluesError",
    "InvalidCountryError",
    "SceneAnalysisUnavailableError",
    "StreetClipService",
]

