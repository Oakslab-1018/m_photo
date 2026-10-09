"""Strict helpers for the flat GeoNames-derived city catalog.

The generated ``data/cities.json`` keeps GeoNames numeric identifiers as
positive decimal strings. That spelling is part of the cross-stack contract:
the frontend and backend receive strings and convert only at the inference
boundary. These helpers validate that contract and provide deterministic
country selection without depending on the inference package.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

_CITY_KEYS = (
    "city_id",
    "iso",
    "name",
    "ascii_name",
    "region",
    "lat",
    "lng",
    "population",
)


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


def _normalize_city_id(value: object, index: int) -> str:
    if not isinstance(value, str):
        raise CityCatalogError(f"city catalog item {index} has a non-string city_id")
    if not value.isascii() or not value.isdigit() or int(value) <= 0:
        raise CityCatalogError(f"city catalog item {index} has an invalid city_id")
    return value


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
    """Validate and normalize the city rows, preserving JSON-compatible extras."""
    if isinstance(payload, dict):
        payload = payload.get("cities")
    if not isinstance(payload, list):
        raise CityCatalogError("city catalog must be a JSON array of city rows")

    cities: list[dict] = []
    seen: set[str] = set()
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
            region = raw_region.strip() or None
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


def load_cities(path: str | Path) -> list[dict]:
    """Load and validate ``data/cities.json``-shaped JSON."""
    path = Path(path)
    if not path.is_file():
        raise CityCatalogError(f"city catalog not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise CityCatalogError(f"city catalog is unreadable: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CityCatalogError(f"city catalog is not valid JSON: {path}") from exc
    return validate_cities(payload)


def select_country(cities: list[dict] | tuple[dict, ...], iso: str, limit: int = 100) -> list[dict]:
    """Return a country's cities by population desc then numeric id asc."""
    normalized_iso = _normalize_iso(iso, context="country ISO")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise CityCatalogError("city limit must be a positive integer")
    rows = validate_cities(list(cities))
    selected = [row for row in rows if row["iso"] == normalized_iso]
    selected.sort(key=lambda row: (-row["population"], int(row["city_id"])))
    return selected[:limit]


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of the raw catalog file bytes."""
    path = Path(path)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CityCatalogError(f"city catalog is unreadable: {path}") from exc
    return digest.hexdigest()


def catalog_revision(path: str | Path) -> str:
    """Alias for :func:`file_sha256` used by revision metadata callers."""
    return file_sha256(path)


__all__ = [
    "CityCatalogError",
    "catalog_revision",
    "file_sha256",
    "load_cities",
    "select_country",
    "validate_cities",
]