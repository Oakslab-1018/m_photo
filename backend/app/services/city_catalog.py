"""Read-only access to the authoritative GeoNames city catalog (SQLite only).

The inference service ranks cities, but the backend owns the authoritative
catalog copy: every city row it returns is checked against this module before
it is shown to a user, so a stale or diverging catalog surfaces as a rejected
payload instead of fabricated geography on the map.

The catalog now lives in the portable SQLite seed
(``data/geo_catalog.sqlite3``) and is read through the standard-library
:class:`catalog_db.CatalogDatabase`. ``data/cities.json`` is never read.
``catalog_digest`` returns the stored logical fingerprint
(``geo_catalog_source.fingerprint`` for the ``cities`` source), never a hash of
the SQLite file itself, so both services can keep comparing the same revision
string without hashing multi-megabyte files.

Parsing is cached per ``(database path, logical revision)``; the fingerprint
changes when ``--force`` copies an updated seed, which naturally invalidates
the cache.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from catalog_db import CatalogDatabase, CatalogDatabaseError

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


def catalog_digest(path: str | Path | None = None) -> str:
    """Return the stored logical revision of the city catalog.

    ``path`` is a SQLite catalog database path (``None`` selects the immutable
    shipped seed). The value is the fingerprint recorded at seed-build time,
    never a SHA-256 of the database file or a re-serialized JSON document.
    """
    try:
        reader = CatalogDatabase(None if path is None else Path(path))
        return reader.revision("cities")
    except CatalogDatabaseError as exc:
        raise CityCatalogError(f"city catalog is unreadable: {exc}") from exc


def _city_mapping(row: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the original city-object shape from one SQL storage row."""

    city_id = row.get("geoname_id")
    if city_id is None:
        city_id = row.get("place_id")
    return {
        "city_id": city_id,
        "iso": row.get("source_iso"),
        "name": row.get("name"),
        "region": row.get("code") or "",
        "lat": row.get("lat"),
        "lng": row.get("lng"),
        "population": row.get("population") if row.get("population") is not None else 0,
    }


@lru_cache(maxsize=4)
def _load_cached(path_key: str, revision: str) -> dict[str, dict[int, CityRow]]:
    # ``revision`` is part of the cache key; the rows are always re-read from
    # the database on a miss, then validated by the strict converter below.
    try:
        reader = CatalogDatabase(path_key)
        rows = reader.places("city")
    except CatalogDatabaseError as exc:
        raise CityCatalogError(f"city catalog is unreadable: {exc}") from exc

    index: dict[str, dict[int, CityRow]] = {}
    for position, row in enumerate(rows):
        city = city_from_mapping(_city_mapping(row), position)
        bucket = index.setdefault(city.iso, {})
        if city.city_id in bucket:
            raise CityCatalogError(f"duplicate city_id in city catalog: {city.city_id}")
        bucket[city.city_id] = city
    return index


def load_city_index(path: str | Path | None = None) -> dict[str, dict[int, CityRow]]:
    """Return ``{iso: {city_id: CityRow}}`` for the catalog at ``path``.

    ``path`` is a SQLite catalog database path; ``None`` selects the immutable
    shipped seed. Raises :class:`CityCatalogError` for a missing, malformed or
    duplicated catalog; callers treat that exactly like invalid provider
    output. The legacy JSON path form is rejected because the SQLite reader
    cannot open it.
    """
    candidate = None if path is None else Path(path)
    try:
        reader = CatalogDatabase(candidate)
        revision = reader.revision("cities")
    except CatalogDatabaseError as exc:
        raise CityCatalogError(f"city catalog is unreadable: {exc}") from exc
    return _load_cached(str(reader.path), revision)


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
