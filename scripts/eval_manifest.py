"""Manifest and environment-file parsing for offline geolocation evaluation.

This module is intentionally dependency-free (standard library only) so its
rules can be unit-tested without loading the backend, RapidOCR or torch.

Manifest format (JSON Lines, one photo per line)::

    {"id": "tokyo-001", "image": "images/tokyo-001.jpg",
     "country_iso": "JP", "city_id": "1850147", "city": "Tokyo",
     "lat": 35.6895, "lng": 139.6917, "license": "CC BY-SA 4.0",
     "source": "https://example.org/photo", "tags": ["urban", "signage"]}

``image`` is always relative to ``--image-root`` (which defaults to the
manifest's own directory). Absolute paths and ``..`` escapes are rejected so a
shared manifest cannot silently read arbitrary files.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

ALLOWED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})

_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_ISO_RE = re.compile(r"^[A-Za-z]{2}$")


class ManifestError(ValueError):
    """A manifest line is missing, malformed or points outside the image root."""


@dataclass(frozen=True)
class ManifestEntry:
    """One validated evaluation photo.

    ``country_iso`` is ``None`` only when the loader explicitly permits
    coordinate-only ground truth.
    """

    id: str
    image_path: Path
    image_ref: str
    country_iso: str | None
    lat: float
    lng: float
    city_id: str | None = None
    city: str | None = None
    admin1_code: str | None = None
    city_source_iso: str | None = None
    license: str | None = None
    source: str | None = None
    tags: tuple[str, ...] = ()
    notes: str | None = None


def load_env_file(path: str | Path) -> dict[str, str]:
    """Parse a minimal ``KEY=VALUE`` env file without importing dotenv.

    Existing process environment variables are *not* consulted here; callers
    merge with ``os.environ`` so real environment variables win. Values are
    never logged. Quotes around a value are removed; ``export `` prefixes are
    accepted. No variable expansion is performed.
    """

    candidate = Path(path)
    if not candidate.exists():
        return {}
    values: dict[str, str] = {}
    for line in candidate.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _require_text(raw: Mapping[str, object], key: str, *, lineno: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"line {lineno}: {key} must be a non-empty string")
    return value.strip()


def _optional_text(raw: Mapping[str, object], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ManifestError(f"{key} must be a string when present")
    text = value.strip()
    return text or None


def _optional_iso(raw: Mapping[str, object], key: str) -> str | None:
    value = _optional_text(raw, key)
    if value is None:
        return None
    if not _ISO_RE.match(value):
        raise ManifestError(f"{key} must be a two-letter ISO code")
    return value.upper()


def _required_coordinate(raw: Mapping[str, object], key: str, *, bound: float, lineno: int) -> float:
    value = raw.get(key)
    if isinstance(value, bool):
        raise ManifestError(f"line {lineno}: {key} must be a number")
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"line {lineno}: {key} must be a number") from exc
    if not math.isfinite(parsed) or parsed < -bound or parsed > bound:
        raise ManifestError(f"line {lineno}: {key} is outside [-{bound}, {bound}]")
    return parsed


def _optional_city_id(raw: Mapping[str, object]) -> str | None:
    value = raw.get("city_id")
    if value is None:
        return None
    # int(1.5) silently truncates; a JSON number must be an actual integer.
    if isinstance(value, (bool, float)):
        raise ManifestError("city_id must be a positive integer when present")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ManifestError("city_id must be a positive integer when present") from exc
    if parsed <= 0:
        raise ManifestError("city_id must be a positive integer when present")
    return str(parsed)


def _optional_tags(raw: Mapping[str, object]) -> tuple[str, ...]:
    value = raw.get("tags")
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ManifestError("tags must be a list of strings when present")
    return tuple(item.strip() for item in value if item.strip())


def _resolve_image(
    image_ref: str,
    *,
    image_root: Path,
    line_label: str,
) -> Path:
    if Path(image_ref).is_absolute():
        raise ManifestError(f"{line_label}: image must be relative to the image root")
    if ".." in Path(image_ref).parts:
        raise ManifestError(f"{line_label}: image must not escape the image root")
    root = image_root.expanduser().resolve()
    target = (root / image_ref).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ManifestError(f"{line_label}: image escapes the image root") from exc
    if not target.is_file():
        raise ManifestError(f"{line_label}: image not found: {image_ref}")
    if target.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        raise ManifestError(f"{line_label}: unsupported image suffix: {target.suffix}")
    return target


def load_manifest(
    path: str | Path,
    *,
    image_root: str | Path | None = None,
    valid_isos: Iterable[str] | None = None,
    allow_unlabeled_country: bool = False,
) -> list[ManifestEntry]:
    """Load and strictly validate a JSONL manifest.

    ``valid_isos`` should be the loaded catalog's source ISO set; passing it
    turns an unknown label into a hard error instead of a silently unmatched
    sample. ``allow_unlabeled_country`` opts in to coordinate-only samples;
    missing/null country labels are then accepted, while non-empty malformed or
    unknown labels are still rejected. Blank lines are allowed; every other
    line must be a JSON object.
    """

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ManifestError(f"manifest not found: {manifest_path}")
    root = Path(image_root) if image_root is not None else manifest_path.parent
    if not root.is_dir():
        raise ManifestError(f"image root not found: {root}")

    allowed = (
        {str(iso).strip().upper() for iso in valid_isos}
        if valid_isos is not None
        else None
    )
    entries: list[ManifestEntry] = []
    seen_ids: set[str] = set()
    seen_images: set[Path] = set()

    with manifest_path.open("r", encoding="utf-8-sig") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            label = f"line {lineno}"
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{label}: invalid JSON: {exc.msg}") from exc
            if not isinstance(raw, dict):
                raise ManifestError(f"{label}: each line must be a JSON object")

            sample_id = _require_text(raw, "id", lineno=lineno)
            if not _ID_RE.match(sample_id):
                raise ManifestError(
                    f"{label}: id must match [A-Za-z0-9._-]+ (got {sample_id!r})"
                )
            if sample_id in seen_ids:
                raise ManifestError(f"{label}: duplicate id: {sample_id}")
            seen_ids.add(sample_id)

            image_ref = _require_text(raw, "image", lineno=lineno)
            image_path = _resolve_image(image_ref, image_root=root, line_label=label)
            if image_path in seen_images:
                raise ManifestError(f"{label}: duplicate image path: {image_ref}")
            seen_images.add(image_path)

            country_iso = (
                _optional_text(raw, "country_iso")
                if allow_unlabeled_country
                else _require_text(raw, "country_iso", lineno=lineno)
            )
            if country_iso is not None:
                country_iso = country_iso.upper()
                if not _ISO_RE.match(country_iso):
                    raise ManifestError(f"{label}: country_iso must be a two-letter code")
                if allowed is not None and country_iso not in allowed:
                    raise ManifestError(
                        f"{label}: country_iso {country_iso} is not in the loaded catalog"
                    )

            city_source_iso = _optional_iso(raw, "city_source_iso")
            if allowed is not None and city_source_iso and city_source_iso not in allowed:
                raise ManifestError(
                    f"{label}: city_source_iso {city_source_iso} is not in the loaded catalog"
                )

            entries.append(
                ManifestEntry(
                    id=sample_id,
                    image_path=image_path,
                    image_ref=image_ref,
                    country_iso=country_iso,
                    lat=_required_coordinate(raw, "lat", bound=90.0, lineno=lineno),
                    lng=_required_coordinate(raw, "lng", bound=180.0, lineno=lineno),
                    city_id=_optional_city_id(raw),
                    city=_optional_text(raw, "city"),
                    admin1_code=_optional_text(raw, "admin1_code"),
                    city_source_iso=city_source_iso,
                    license=_optional_text(raw, "license"),
                    source=_optional_text(raw, "source"),
                    tags=_optional_tags(raw),
                    notes=_optional_text(raw, "notes"),
                )
            )

    if not entries:
        raise ManifestError("manifest contains no samples")
    return entries


__all__ = [
    "ALLOWED_IMAGE_SUFFIXES",
    "ManifestEntry",
    "ManifestError",
    "load_env_file",
    "load_manifest",
]
