"""City-level catalog adapter for the StreetCLIP inference service.

The geodata worker owns ``cities_catalog.py`` and ``data/cities.json``. This
module is the bounded seam between that catalog and the inference service: it
loads the flat GeoNames-derived rows once, exposes immutable per-ISO tuples,
renders the city/region/country text prototypes and reports a catalog revision.

``city_catalog_revision`` is the SHA-256 of the raw ``cities.json`` file bytes
(not a re-serialized canonical form). The backend copies that file into its own
image to verify revisions, so both sides must agree byte-for-byte without
importing inference code or re-encoding JSON.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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


def catalog_file_hash(path: Path) -> str:
    """Return the SHA-256 of the raw ``cities.json`` file bytes."""
    path = Path(path)
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CityCatalogError(f"city catalog is unreadable: {path}") from exc
    return digest.hexdigest()


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


def _read_catalog_rows(path: Path) -> list[City]:
    if not path.is_file():
        raise CityCatalogError(f"city catalog not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise CityCatalogError(f"city catalog is unreadable: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CityCatalogError(f"city catalog is not valid JSON: {path}") from exc
    if isinstance(payload, dict):
        payload = payload.get("cities")
    if not isinstance(payload, list):
        raise CityCatalogError("city catalog must be a JSON array of city rows")

    cities: list[City] = []
    seen: set[int] = set()
    for index, item in enumerate(payload):
        city = city_from_mapping(item, index)
        if city.city_id in seen:
            raise CityCatalogError(f"duplicate city_id in city catalog: {city.city_id}")
        seen.add(city.city_id)
        cities.append(city)
    return cities


def _load_via_worker_module(path: Path) -> list[City] | None:
    """Use ``cities_catalog.py`` when present; ``None`` means fall back."""
    try:
        module = importlib.import_module("inference.streetclip.cities_catalog")
    except Exception:
        return None
    loader = getattr(module, "load_cities", None) or getattr(module, "read_cities", None)
    if loader is None:
        return None
    try:
        loaded = loader(path)
    except TypeError:
        loaded = loader(str(path))
    except Exception as exc:
        raise CityCatalogError(f"cities_catalog loader failed: {exc}") from exc

    rows: list[City] = []
    if isinstance(loaded, dict):
        for iso, values in loaded.items():
            if isinstance(values, dict) and "cities" in values:
                values = values["cities"]
            for index, item in enumerate(values or ()):
                row = city_from_mapping(item, index)
                if row.iso != str(iso).strip().upper():
                    row = City(
                        city_id=row.city_id,
                        iso=str(iso).strip().upper(),
                        name=row.name,
                        region=row.region,
                        lat=row.lat,
                        lng=row.lng,
                        population=row.population,
                    )
                rows.append(row)
    elif isinstance(loaded, (list, tuple)):
        rows = [city_from_mapping(item, index) for index, item in enumerate(loaded)]
    else:
        return None
    if not rows:
        return None
    seen: set[int] = set()
    for row in rows:
        if row.city_id in seen:
            raise CityCatalogError(f"duplicate city_id in city catalog: {row.city_id}")
        seen.add(row.city_id)
    return rows


def load_cities(path: Path) -> dict[str, tuple[City, ...]]:
    """Load the catalog grouped by ISO, sorted by population then city id.

    Every returned group is an immutable tuple already sorted for selection.
    The full mapping is kept so only the selected country's top cities are
    ever turned into text features.
    """
    path = Path(path)
    rows = _load_via_worker_module(path)
    if rows is None:
        rows = _read_catalog_rows(path)
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
