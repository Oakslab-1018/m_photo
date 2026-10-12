"""Strict helpers for the GeoNames-derived city catalog.

Rows come from the shipped read-only SQLite seed through the shared
``catalog_db`` reader. The helpers keep raw source ISO grouping, preserve the
raw region code and provide deterministic country selection by population
descending and numeric GeoNames identifier ascending.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .catalog import CatalogDatabase, CatalogDatabaseError


class CityCatalogError(ValueError):
    """Raised when the city catalog is missing, malformed or duplicated."""


def _require_label(row: dict[str, Any], field: str, index: int, *, fallback: str | None = None) -> str:
    value = row.get(field)
    if value is None and fallback is not None:
        return fallback
    if not isinstance(value, str):
        raise CityCatalogError(f"city catalog item {index} has a non-string {field}")
    text = value.strip()
    if not text:
        if fallback is not None:
            return fallback
        raise CityCatalogError(f"city catalog item {index} has an empty {field}")
    return text


def _normalize_iso(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CityCatalogError(f"{context} must be a two-letter ISO code")
    iso = value.strip().upper()
    if len(iso) != 2 or not iso.isascii() or not iso.isalpha():
        raise CityCatalogError(f"{context} must be a two-letter ISO code")
    return iso


def _normalize_city_id(value: object, index: int) -> int:
    if isinstance(value, bool):
        raise CityCatalogError(f"city catalog item {index} has an invalid city_id")
    if isinstance(value, int):
        city_id = value
    elif isinstance(value, str) and value.isascii() and value.isdigit():
        city_id = int(value)
    else:
        raise CityCatalogError(f"city catalog item {index} has an invalid city_id")
    if city_id <= 0:
        raise CityCatalogError(f"city catalog item {index} has an invalid city_id")
    return city_id


def _normalize_coordinate(value: object, field: str, index: int, *, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CityCatalogError(f"city catalog item {index} has a non-numeric {field}")
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise CityCatalogError(f"city catalog item {index} has a non-finite or out-of-range {field}")
    return number


def _normalize_population(value: object, index: int) -> int:
    if isinstance(value, bool):
        raise CityCatalogError(f"city catalog item {index} has a non-integer population")
    if isinstance(value, int):
        population = value
    elif isinstance(value, str) and value.isascii() and value.isdigit():
        population = int(value)
    else:
        raise CityCatalogError(f"city catalog item {index} has a non-integer population")
    if population < 0:
        raise CityCatalogError(f"city catalog item {index} has a negative population")
    return population


def validate_cities(payload: object) -> list[dict]:
    """Validate and normalize city rows, preserving database payload extras."""
    if isinstance(payload, dict):
        payload = payload.get("cities")
    if not isinstance(payload, list):
        raise CityCatalogError("city catalog must be an array of city rows")

    cities: list[dict] = []
    seen: set[int] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise CityCatalogError(f"city catalog item {index} is not an object")
        city_id = _normalize_city_id(item.get("city_id"), index)
        if city_id in seen:
            raise CityCatalogError(f"duplicate city_id in city catalog: {city_id}")
        seen.add(city_id)

        iso = _normalize_iso(item.get("iso"), context=f"city catalog item {index} iso")
        name = _require_label(item, "name", index)
        ascii_name = _require_label(item, "ascii_name", index, fallback=name)

        raw_region = item.get("region")
        if raw_region is None:
            region = None
        elif isinstance(raw_region, str):
            region = raw_region
        else:
            raise CityCatalogError(f"city catalog item {index} has a non-string region")

        lat = _normalize_coordinate(item.get("lat"), "lat", index, low=-90.0, high=90.0)
        lng = _normalize_coordinate(item.get("lng"), "lng", index, low=-180.0, high=180.0)
        population = _normalize_population(item.get("population"), index)

        row = dict(item)
        row.update(
            city_id=city_id,
            iso=iso,
            name=name,
            ascii_name=ascii_name,
            region=region,
            lat=lat,
            lng=lng,
            population=population,
        )
        cities.append(row)
    return cities


def _catalog_reader(path: str | Path | CatalogDatabase) -> CatalogDatabase:
    if isinstance(path, CatalogDatabase):
        return path
    try:
        return CatalogDatabase(path)
    except CatalogDatabaseError as exc:
        raise CityCatalogError(f"city catalog database is unreadable: {path}") from exc


def load_cities(path: str | Path | CatalogDatabase) -> list[dict]:
    """Load and validate city rows from the SQLite catalog seed."""
    try:
        payload = _catalog_reader(path).cities()
    except CatalogDatabaseError as exc:
        raise CityCatalogError(f"city catalog database is unreadable: {path}") from exc
    return validate_cities(payload)


def select_country(cities: list[dict] | tuple[dict, ...], iso: str, limit: int = 100) -> list[dict]:
    """Return a country's cities by population desc then numeric id asc."""
    normalized_iso = _normalize_iso(iso, context="country ISO")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise CityCatalogError("city limit must be a positive integer")
    rows = validate_cities(list(cities))
    selected = [row for row in rows if row["iso"] == normalized_iso]
    selected.sort(key=lambda row: (-row["population"], row["city_id"]))
    return selected[:limit]


def file_sha256(path: str | Path | CatalogDatabase) -> str:
    """Return the stored logical city revision, never a SQLite file hash."""
    try:
        return _catalog_reader(path).revision("cities")
    except CatalogDatabaseError as exc:
        raise CityCatalogError(f"city catalog revision is unreadable: {path}") from exc


def catalog_revision(path: str | Path | CatalogDatabase) -> str:
    """Return the stored logical city revision for revision metadata callers."""
    return file_sha256(path)


__all__ = [
    "CityCatalogError",
    "catalog_revision",
    "file_sha256",
    "load_cities",
    "select_country",
    "validate_cities",
]
