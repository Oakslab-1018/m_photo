"""Read-only access to the GeoNames city catalog (``data/cities.json``).

The inference service ranks cities, but the backend owns the authoritative
catalog copy: every city row it returns is checked against this module before
it is shown to a user, so a stale or diverging pair of catalogs surfaces as a
rejected payload instead of fabricated geography on the map.

Pinned in ``data/geo-catalog-sources.json``: GeoNames ``cities15000``
(CC BY 4.0), 34156 rows. ``catalog_digest`` exposes the raw-file SHA-256 that
both services compare, so neither side has to re-serialize JSON to agree.

Parsing is cached per ``(path, mtime, size)``; the catalog is read-only and
shipped inside the backend image, so a given deployment parses it once.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

_ISO_RE = re.compile(r"^[A-Z]{2}$")


class CityCatalogError(ValueError):
    """Raised when the city catalog is missing, malformed or duplicated."""


@dataclass(frozen=True)
class CityRow:
    """One validated GeoNames city row."""

    city_id: int
    iso: str
    name: str
    region: str
    lat: float
    lng: float
    population: int


def _coerce_city_id(value: Any) -> int:
    if isinstance(value, bool):
        raise CityCatalogError("city_id must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CityCatalogError("city_id must be an integer") from exc
    if parsed <= 0:
        raise CityCatalogError("city_id must be positive")
    return parsed


def _coerce_text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if value is None and allow_empty:
        return ""
    if not isinstance(value, str):
        raise CityCatalogError(f"{field} must be a string")
    # Whitespace-normalised so catalog and inference agree byte-for-byte on names.
    text = " ".join(value.split())
    if not text and not allow_empty:
        raise CityCatalogError(f"{field} must not be empty")
    return text


def _coerce_coordinate(value: Any, field: str, *, bound: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CityCatalogError(f"{field} must be a number") from exc
    if not math.isfinite(parsed) or parsed < -bound or parsed > bound:
        raise CityCatalogError(f"{field} is out of range")
    return parsed


def _coerce_population(value: Any) -> int:
    if value is None:
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CityCatalogError("population must be an integer") from exc
    if parsed < 0:
        raise CityCatalogError("population must be non-negative")
    return parsed


def city_from_mapping(item: Any, index: int = 0) -> CityRow:
    """Validate one catalog row into a :class:`CityRow`."""
    if not isinstance(item, dict):
        raise CityCatalogError(f"city catalog item {index} is not an object")
    try:
        city_id = _coerce_city_id(item.get("city_id"))
        iso = _coerce_text(item.get("iso"), "iso").upper()
        name = _coerce_text(item.get("name"), "name")
        region = _coerce_text(item.get("region"), "region", allow_empty=True)
        lat = _coerce_coordinate(item.get("lat"), "lat", bound=90.0)
        lng = _coerce_coordinate(item.get("lng"), "lng", bound=180.0)
        population = _coerce_population(item.get("population"))
    except CityCatalogError as exc:
        raise CityCatalogError(f"city catalog item {index}: {exc}") from exc
    if not _ISO_RE.match(iso):
        raise CityCatalogError(f"city catalog item {index}: iso must be two letters")
    return CityRow(city_id=city_id, iso=iso, name=name, region=region,
                   lat=lat, lng=lng, population=population)


def catalog_digest(path: str | Path) -> str:
    """SHA-256 of the raw catalog bytes (never a re-serialized canonical form)."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CityCatalogError(f"city catalog is unreadable: {Path(path).name}") from exc
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _load_cached(key: tuple[str, int, int]) -> dict[str, dict[int, CityRow]]:
    path = Path(key[0])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CityCatalogError(f"city catalog is unreadable: {path.name}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CityCatalogError(f"city catalog is not valid JSON: {path.name}") from exc
    if isinstance(payload, dict):
        payload = payload.get("cities")
    if not isinstance(payload, list) or not payload:
        raise CityCatalogError("city catalog must be a non-empty JSON array")

    index: dict[str, dict[int, CityRow]] = {}
    seen: set[int] = set()
    for position, item in enumerate(payload):
        row = city_from_mapping(item, position)
        if row.city_id in seen:
            raise CityCatalogError(f"duplicate city_id in city catalog: {row.city_id}")
        seen.add(row.city_id)
        index.setdefault(row.iso, {})[row.city_id] = row
    return index


def load_city_index(path: str | Path) -> dict[str, dict[int, CityRow]]:
    """Return ``{iso: {city_id: CityRow}}`` for the catalog at ``path``.

    Raises :class:`CityCatalogError` for a missing, malformed or duplicated
    catalog; callers treat that exactly like invalid provider output.
    """
    candidate = Path(path)
    try:
        stat = candidate.stat()
    except OSError as exc:
        raise CityCatalogError(f"city catalog not found: {candidate.name}") from exc
    return _load_cached((str(candidate), stat.st_mtime_ns, stat.st_size))


def clear_city_cache() -> None:
    """Drop the parse cache (tests only; the catalog is read-only at runtime)."""
    _load_cached.cache_clear()


__all__ = [
    "CityCatalogError",
    "CityRow",
    "catalog_digest",
    "city_from_mapping",
    "clear_city_cache",
    "load_city_index",
]
