"""Request validation and ranking schema helpers.

This module intentionally imports only the standard library plus Pillow.
Importing ``schema`` must never require torch, transformers or CUDA so unit
tests and contract reviews can run on a CPU-only checkout.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import (
    CITY_REFERENCE_FUSION_WEIGHT,
    MAX_UPLOAD_BYTES,
    MODEL_ID,
    MODEL_REVISION,
    UNCERTAIN_MARGIN_THRESHOLD,
)

CITY_TEXT_ONLY_SCORE_TYPE = "city_text_cosine"
CITY_HYBRID_SCORE_TYPE = "city_reference_hybrid"
CITY_STATUS_OK = "ok"
CITY_STATUS_NO_CITIES = "no_cities"
CITY_TOP_N = 10


class ImageValidationError(ValueError):
    """Base class for deterministic, user-safe image validation failures."""


class EmptyImageError(ImageValidationError):
    """The uploaded byte string is empty."""


class ImageTooLargeError(ImageValidationError):
    """The uploaded byte string exceeds the configured 16 MiB bound."""


class DecodeError(ImageValidationError):
    """The bytes are not a decodable image."""


class ExifGpsPresentError(ImageValidationError):
    """The upload carries EXIF GPS metadata and must not be processed."""


class InvalidCountryError(ValueError):
    """The requested ISO code is not a supported 2-letter candidate."""


@dataclass(frozen=True)
class Ranking:
    iso: str
    raw_cosine: float


@dataclass(frozen=True)
class Prediction:
    model: str
    model_revision: str
    catalog_revision: str
    rankings: tuple[Ranking, ...]
    margin: float
    uncertain: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "model_revision": self.model_revision,
            "catalog_revision": self.catalog_revision,
            "rankings": [
                {"iso": ranking.iso, "raw_cosine": ranking.raw_cosine}
                for ranking in self.rankings
            ],
            "margin": self.margin,
            "uncertain": self.uncertain,
        }


@dataclass(frozen=True)
class CityScore:
    city_id: int
    iso: str
    name: str
    region: str
    lat: float
    lng: float
    raw_cosine: float
    score: float
    score_type: str
    reference_cosine: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "city_id": self.city_id,
            "iso": self.iso,
            "name": self.name,
            "region": self.region,
            "lat": self.lat,
            "lng": self.lng,
            "raw_cosine": self.raw_cosine,
            "score": self.score,
            "score_type": self.score_type,
            "reference_cosine": self.reference_cosine,
        }


@dataclass(frozen=True)
class CityPrediction:
    country_iso: str
    city_limit_per_country: int
    city_catalog_revision: str | None
    cities_considered: int
    status: str
    city_scores: tuple[CityScore, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "country_iso": self.country_iso,
            "scope": {
                "city_limit_per_country": self.city_limit_per_country,
                "city_catalog_revision": self.city_catalog_revision,
                "cities_considered": self.cities_considered,
            },
            "status": self.status,
            "city_scores": [score.as_dict() for score in self.city_scores],
        }


def validate_image_bytes(payload: bytes, *, max_bytes: int = MAX_UPLOAD_BYTES) -> None:
    """Validate size and decodability without retaining EXIF/GPS metadata."""
    if not payload:
        raise EmptyImageError("empty upload")
    if len(payload) > max_bytes:
        raise ImageTooLargeError(f"image exceeds {max_bytes} byte limit")
    try:
        with Image.open(io.BytesIO(payload)) as probe:
            probe.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise DecodeError("upload is not a valid image") from exc


def decode_image_bytes(payload: bytes) -> Image.Image:
    """Decode a bounded upload, apply EXIF orientation, strip metadata.

    The returned image is converted to RGB and has no EXIF metadata attached.
    EXIF GPS tags are rejected explicitly because the service must never use
    or forward location metadata.
    """
    validate_image_bytes(payload)
    try:
        with Image.open(io.BytesIO(payload)) as source:
            exif = source.getexif()
            if exif:
                gps_ifd = exif.get_ifd(0x8825) if hasattr(exif, "get_ifd") else {}
                if gps_ifd:
                    raise ExifGpsPresentError("image contains EXIF GPS metadata")
            oriented = ImageOps.exif_transpose(source)
            rgb = oriented.convert("RGB")
            rgb.load()
            rgb.info.clear()
            return rgb
    except ExifGpsPresentError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise DecodeError("upload is not a valid image") from exc


def _finite_cosine(value: object, *, label: str) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    if parsed < -1.0000001 or parsed > 1.0000001:
        raise ValueError(f"{label} is outside [-1, 1]: {parsed}")
    return min(1.0, max(-1.0, parsed))


def _bounded_unit(value: float) -> float:
    """Clamp a cosine-like similarity into the [0, 1] fusion range."""
    return min(1.0, max(0.0, float(value)))


def build_prediction(
    *,
    catalog_revision: str,
    rankings: list[tuple[str, float]] | tuple[tuple[str, float], ...],
    margin_threshold: float = UNCERTAIN_MARGIN_THRESHOLD,
) -> Prediction:
    """Validate, sort descending and wrap a complete country ranking.

    Raw cosine values are finite floats in ``[-1, 1]`` (within floating point
    tolerance). Ties are broken deterministically by ISO code.
    """
    if not rankings:
        raise ValueError("rankings must not be empty")
    cleaned: list[Ranking] = []
    seen: set[str] = set()
    for iso, raw_cosine in rankings:
        if not isinstance(iso, str) or not iso:
            raise ValueError("ranking ISO codes must be non-empty strings")
        iso = iso.upper()
        if iso in seen:
            raise ValueError(f"duplicate ISO code in rankings: {iso}")
        seen.add(iso)
        value = _finite_cosine(raw_cosine, label=f"raw_cosine for {iso}")
        cleaned.append(Ranking(iso=iso, raw_cosine=value))
    cleaned.sort(key=lambda item: (-item.raw_cosine, item.iso))
    margin = cleaned[0].raw_cosine - cleaned[1].raw_cosine if len(cleaned) >= 2 else 0.0
    return Prediction(
        model=MODEL_ID,
        model_revision=MODEL_REVISION,
        catalog_revision=catalog_revision,
        rankings=tuple(cleaned),
        margin=float(margin),
        uncertain=bool(margin < margin_threshold),
    )


def build_city_prediction(
    *,
    country_iso: str,
    city_catalog_revision: str | None,
    city_limit_per_country: int,
    candidates: list[dict[str, object]] | tuple[dict[str, object], ...],
    reference_matches: object | None = None,
    reference_weight: float = CITY_REFERENCE_FUSION_WEIGHT,
) -> CityPrediction:
    """Build the bounded city-level response contract.

    ``candidates`` carry the real GeoNames identity/coordinates plus the text
    prototype cosine. When a loaded reference index returns matches, only
    matches whose ``city_id`` equals a scored candidate participate in the
    fusion; every other row stays text-only. The fused ``score`` is a bounded
    similarity in [0, 1], never a probability, while ``raw_cosine`` remains the
    unmodified city text-prototype cosine.
    """
    iso = country_iso.strip().upper()
    if len(iso) != 2:
        raise ValueError("country_iso must be a two-letter ISO code")
    if city_limit_per_country <= 0:
        raise ValueError("city_limit_per_country must be positive")

    reference_by_city: dict[int, float] = {}
    if isinstance(reference_matches, dict):
        for match in reference_matches.get("matches") or ():
            if not isinstance(match, dict):
                continue
            city_id = match.get("city_id")
            if isinstance(city_id, bool):
                continue
            try:
                city_id_int = int(city_id)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            try:
                raw = _finite_cosine(match.get("raw_cosine"), label="reference raw_cosine")
            except ValueError:
                continue
            if city_id_int > 0:
                reference_by_city[city_id_int] = max(
                    reference_by_city.get(city_id_int, float("-inf")), raw
                )

    scores: list[CityScore] = []
    seen_ids: set[int] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("city candidate must be a mapping")
        city_id = candidate.get("city_id")
        if isinstance(city_id, bool):
            raise ValueError("city_id must be an integer")
        try:
            city_id_int = int(city_id)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError("city_id must be an integer") from exc
        if city_id_int in seen_ids:
            raise ValueError(f"duplicate city_id in candidates: {city_id_int}")
        seen_ids.add(city_id_int)
        candidate_iso = str(candidate.get("iso", iso)).strip().upper()
        if candidate_iso != iso:
            raise ValueError(f"candidate {city_id_int} does not belong to {iso}")
        raw_cosine = _finite_cosine(candidate.get("raw_cosine"), label=f"raw_cosine for {city_id_int}")
        reference_cosine = reference_by_city.get(city_id_int)
        if reference_cosine is None:
            score_type = CITY_TEXT_ONLY_SCORE_TYPE
            score = _bounded_unit(raw_cosine)
        else:
            score_type = CITY_HYBRID_SCORE_TYPE
            bounded_reference = _bounded_unit(reference_cosine)
            bounded_text = _bounded_unit(raw_cosine)
            weight = min(1.0, max(0.0, float(reference_weight)))
            score = (1.0 - weight) * bounded_text + weight * bounded_reference
            reference_cosine = float(reference_cosine)
        lat = candidate.get("lat")
        lng = candidate.get("lng")
        name = str(candidate.get("name") or "").strip()
        region = str(candidate.get("region") or "").strip()
        if not name:
            raise ValueError(f"candidate {city_id_int} has no city name")
        try:
            lat_value = float(lat)  # type: ignore[arg-type]
            lng_value = float(lng)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"candidate {city_id_int} has invalid coordinates") from exc
        if not math.isfinite(lat_value) or not math.isfinite(lng_value):
            raise ValueError(f"candidate {city_id_int} has non-finite coordinates")
        if lat_value < -90.0 or lat_value > 90.0 or lng_value < -180.0 or lng_value > 180.0:
            raise ValueError(f"candidate {city_id_int} has out-of-range coordinates")
        scores.append(
            CityScore(
                city_id=city_id_int,
                iso=candidate_iso,
                name=name,
                region=region,
                lat=lat_value,
                lng=lng_value,
                raw_cosine=raw_cosine,
                score=min(1.0, max(0.0, score)),
                score_type=score_type,
                reference_cosine=reference_cosine,
            )
        )

    scores.sort(key=lambda item: (-item.score, item.city_id))
    top = tuple(scores[:CITY_TOP_N])
    status = CITY_STATUS_OK if top else CITY_STATUS_NO_CITIES
    return CityPrediction(
        country_iso=iso,
        city_limit_per_country=int(city_limit_per_country),
        city_catalog_revision=city_catalog_revision,
        cities_considered=len(scores),
        status=status,
        city_scores=top,
    )


__all__ = [
    "CITY_HYBRID_SCORE_TYPE",
    "CITY_STATUS_NO_CITIES",
    "CITY_STATUS_OK",
    "CITY_TEXT_ONLY_SCORE_TYPE",
    "CITY_TOP_N",
    "CityPrediction",
    "CityScore",
    "DecodeError",
    "EmptyImageError",
    "ExifGpsPresentError",
    "ImageTooLargeError",
    "ImageValidationError",
    "InvalidCountryError",
    "Prediction",
    "Ranking",
    "build_city_prediction",
    "build_prediction",
    "decode_image_bytes",
    "validate_image_bytes",
]
