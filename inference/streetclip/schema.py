"""Request validation and ranking schema helpers.

This module intentionally imports only the standard library plus Pillow.
Importing ``schema`` must never require torch, transformers or CUDA so unit
tests and contract reviews can run on a CPU-only checkout.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import MAX_UPLOAD_BYTES, MODEL_ID, MODEL_REVISION, UNCERTAIN_MARGIN_THRESHOLD


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


def build_prediction(
    *,
    catalog_revision: str,
    rankings: list[tuple[str, float]] | tuple[tuple[str, float], ...],
    margin_threshold: float = UNCERTAIN_MARGIN_THRESHOLD,
) -> Prediction:
    """Validate, sort descending and wrap a complete 20-country ranking.

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
        value = float(raw_cosine)
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"raw_cosine for {iso} must be finite")
        if value < -1.0000001 or value > 1.0000001:
            raise ValueError(f"raw_cosine for {iso} is outside [-1, 1]: {value}")
        cleaned.append(Ranking(iso=iso, raw_cosine=min(1.0, max(-1.0, value))))
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


__all__ = [
    "DecodeError",
    "EmptyImageError",
    "ExifGpsPresentError",
    "ImageTooLargeError",
    "ImageValidationError",
    "Prediction",
    "Ranking",
    "build_prediction",
    "decode_image_bytes",
    "validate_image_bytes",
]
