"""City-level catalog adapter for the StreetCLIP inference service.

The shipped read-only SQLite seed is the source of truth for the city
directory. This module is the bounded seam between that database and the
inference service: it loads the GeoNames-derived rows, exposes immutable
per-source-ISO tuples, renders the city/region/country text prototypes and
reports the stored logical catalog revision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .catalog import CatalogDatabase, CatalogDatabaseError
from .config import CITY_PROMPT_TEMPLATES, CITY_PROMPT_TEMPLATE_VERSION


class CityCatalogError(ValueError):
    """Raised when the city catalog is missing, malformed or duplicated."""


@dataclass(frozen=True)
class City:
    """One GeoNames city row used for text-prototype scoring."""

    city_id: int
    iso: str
    name: str
    region: str
    lat: float
    lng: float
    population: int


def _catalog_reader(path: Path | str | CatalogDatabase) -> CatalogDatabase:
    if isinstance(path, CatalogDatabase):
        return path
    try:
        return CatalogDatabase(path)
    except CatalogDatabaseError as exc:
        raise CityCatalogError(f"city catalog database is unreadable: {path}") from exc


def catalog_file_hash(path: Path | str | CatalogDatabase) -> str:
    """Return the stored logical city-source revision, never a SQLite file hash."""
    try:
        return _catalog_reader(path).revision("cities")
    except CatalogDatabaseError as exc:
        raise CityCatalogError(f"city catalog revision is unreadable: {path}") from exc


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
    text = " ".join(value.split())
    if not text and not allow_empty:
        raise CityCatalogError(f"{field} must not be empty")
    return text


def _coerce_coordinate(value: Any, field: str, *, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CityCatalogError(f"{field} must be a number") from exc
    if not math.isfinite(parsed) or parsed < low or parsed > high:
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


def city_from_mapping(item: Any, index: int = 0) -> City:
    """Validate and normalize one catalog row (or worker ``City`` object)."""
    if isinstance(item, City):
        return item
    if isinstance(item, dict):
        getter = item.get
    else:
        getter = lambda key: getattr(item, key, None)  # noqa: E731 - tiny adapter
    try:
        city_id = _coerce_city_id(getter("city_id"))
        iso = _coerce_text(getter("iso"), "iso").upper()
        name = _coerce_text(getter("name"), "name")
        region = _coerce_text(getter("region"), "region", allow_empty=True)
        lat = _coerce_coordinate(getter("lat"), "lat", low=-90.0, high=90.0)
        lng = _coerce_coordinate(getter("lng"), "lng", low=-180.0, high=180.0)
        population = _coerce_population(getter("population"))
    except CityCatalogError as exc:
        raise CityCatalogError(f"city catalog item {index}: {exc}") from exc
    if len(iso) != 2:
        raise CityCatalogError(f"city catalog item {index}: iso must be two letters")
    return City(
        city_id=city_id,
        iso=iso,
        name=name,
        region=region,
        lat=lat,
        lng=lng,
        population=population,
    )


def load_cities(path: Path | str | CatalogDatabase) -> dict[str, tuple[City, ...]]:
    """Load the catalog grouped by ISO, sorted by population then city id.

    Every returned group is an immutable tuple already sorted for selection.
    The full mapping is kept so only the selected country's top cities are
    ever turned into text features.
    """
    try:
        payload = _catalog_reader(path).cities()
    except CatalogDatabaseError as exc:
        raise CityCatalogError(f"city catalog database is unreadable: {path}") from exc

    rows: list[City] = []
    seen: set[int] = set()
    for index, item in enumerate(payload):
        city = city_from_mapping(item, index)
        if city.city_id in seen:
            raise CityCatalogError(f"duplicate city_id in city catalog: {city.city_id}")
        seen.add(city.city_id)
        rows.append(city)

    grouped: dict[str, list[City]] = {}
    for city in rows:
        grouped.setdefault(city.iso, []).append(city)
    return {
        iso: tuple(sorted(values, key=lambda city: (-city.population, city.city_id)))
        for iso, values in sorted(grouped.items())
    }


def select_cities(
    grouped: dict[str, tuple[City, ...]],
    iso: str,
    *,
    limit: int,
) -> tuple[City, ...]:
    """Return the top ``limit`` cities for ``iso`` (deterministic)."""
    if limit <= 0:
        raise ValueError("city limit must be positive")
    normalized = iso.strip().upper()
    if len(normalized) != 2:
        raise ValueError("ISO code must be two letters")
    return tuple(grouped.get(normalized, ()))[:limit]


def render_city_prompts(city: City, country_name: str) -> list[str]:
    """Render the deterministic city/region/country text templates."""
    region = city.region or city.name
    return [
        template.format(city=city.name, region=region, country=country_name)
        for template in CITY_PROMPT_TEMPLATES
    ]


def city_prompt_count() -> int:
    """Number of text prototypes rendered per city."""
    return len(CITY_PROMPT_TEMPLATES)


def iter_cities_for_isos(
    grouped: dict[str, tuple[City, ...]],
    isos: Iterable[str],
    *,
    limit: int,
) -> Iterable[tuple[str, City]]:
    """Yield ``(iso, city)`` for selected top cities of each ISO."""
    for iso in sorted({value.strip().upper() for value in isos}):
        for city in select_cities(grouped, iso, limit=limit):
            yield iso, city


__all__ = [
    "City",
    "CityCatalogError",
    "catalog_file_hash",
    "city_from_mapping",
    "city_prompt_count",
    "iter_cities_for_isos",
    "load_cities",
    "render_city_prompts",
    "select_cities",
]
