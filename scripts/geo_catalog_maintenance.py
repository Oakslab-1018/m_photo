#!/usr/bin/env python3
"""Atomic standard-library maintenance writer for the SQLite geo catalog.

The updater scripts used to rewrite ``data/countries.json``, ``data/cities.json``
and ``data/admin1.json``.  Those generated files are retired: this module is the
single write path that keeps the shipped ``geo_place`` / ``geo_catalog_source``
database coherent while an operator updates one logical dataset at a time.

The writer deliberately uses only :mod:`sqlite3` for writes.  It never hashes the
SQLite file: source revisions are derived from canonical logical rows, and an
untouched dataset keeps its existing fingerprint byte-for-byte.  A dataset whose
updated logical rows are equal to its stored rows also keeps the existing
fingerprint, even when a newly parsed payload has a different JSON member or
record order.

Admin1 representative points are always rebuilt with the frozen
``admin1-largest-city-v1`` policy: the highest-population city matching the raw
source ISO and exact admin1 code wins, with numeric city id ascending on ties.
Changing cities therefore refreshes admin1 coordinates in the same transaction.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATABASE_PATH = REPO_ROOT / "data" / "geo_catalog.sqlite3"

SOURCE_NAMES: tuple[str, ...] = ("countries", "admin1", "cities")
SOURCE_LEVEL: dict[str, str] = {
    "countries": "country",
    "admin1": "admin1",
    "cities": "city",
}
LEVEL_SOURCE: dict[str, str] = {
    "country": "countries",
    "admin1": "admin1",
    "city": "cities",
}

#: Source import/derivation policy version.  This is intentionally the same
#: version already stored in the seed: a maintenance edit must not silently
#: change the meaning of a source version while preserving its rows.
IMPORT_VERSION = "geo-catalog-v1+admin1-largest-city-v1+china-subordinate-regions-v1"

ADMIN1_COORDINATE_SOURCE = "admin1-largest-city-v1"
CATALOG_COORDINATE_SOURCE = "catalog"
UNAVAILABLE_COORDINATE_SOURCE = "unavailable"

_DEFAULT_PROVENANCE = {
    "countries": "legacy-json:countries.json",
    "admin1": "legacy-json:admin1.json",
    "cities": "legacy-json:cities.json",
}

_REGION_PARENT_ISO = {"TW": "CN", "HK": "CN", "MO": "CN"}
_ISO_RE = re.compile(r"^[A-Z]{2}$")
_ISO3_RE = re.compile(r"^[A-Z]{3}$")
_ADMIN1_ID_RE = re.compile(r"^(?P<iso>[A-Z]{2})\.(?P<code>[A-Z0-9]+)$")
_CODE_RE = re.compile(r"^[A-Z0-9]+$")
_FINGERPRINT_RE = re.compile(r"^[0-9a-fA-F]{64}$")

_GEO_PLACE_COLUMNS: tuple[str, ...] = (
    "level",
    "place_id",
    "source_iso",
    "country_iso",
    "parent_id",
    "admin1_id",
    "code",
    "name",
    "ascii_name",
    "name_zh",
    "aliases",
    "payload",
    "population",
    "geoname_id",
    "lat",
    "lng",
    "coordinate_source",
    "coordinate_city_id",
)

_SOURCE_COLUMNS: tuple[str, ...] = (
    "source",
    "path",
    "fingerprint",
    "record_count",
    "import_version",
    "imported_at",
)


class GeoCatalogMaintenanceError(ValueError):
    """Raised when a maintenance input or an existing database is invalid."""


# Compatibility alias: callers can catch either name.
MaintenanceError = GeoCatalogMaintenanceError


@dataclass(frozen=True)
class CatalogMaintenancePlan:
    """A fully derived, read-only preview of one maintenance update."""

    database: Path
    rows: Mapping[str, tuple[dict[str, Any], ...]]
    metadata: Mapping[str, Mapping[str, Any]]
    revisions: Mapping[str, str]
    counts: Mapping[str, int]
    changed_sources: frozenset[str]
    touched_sources: frozenset[str]
    admin1_with_coordinates: int
    admin1_missing_coordinates: int


@dataclass(frozen=True)
class CatalogMaintenanceResult:
    """Outcome of one committed maintenance transaction."""

    database: Path
    counts: Mapping[str, int]
    revisions: Mapping[str, str]
    changed_sources: tuple[str, ...]
    touched_sources: tuple[str, ...]
    admin1_with_coordinates: int
    admin1_missing_coordinates: int


@dataclass(frozen=True)
class CatalogMaintenanceCheck:
    """Result of checking a proposed update without writing anything."""

    ok: bool
    mismatches: tuple[str, ...]
    plan: CatalogMaintenancePlan


def _row_dict(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(row, sqlite3.Row):
        return {str(key): row[key] for key in row.keys()}
    if isinstance(row, Mapping):
        return dict(row)
    raise GeoCatalogMaintenanceError("database row is not a mapping")


def _text(value: Any, field: str, context: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise GeoCatalogMaintenanceError(f"{context}: {field} must be a string")
    text = " ".join(value.split())
    if not text:
        raise GeoCatalogMaintenanceError(f"{context}: {field} must not be blank")
    return text


def _optional_text(value: Any, field: str, context: str) -> str | None:
    return _text(value, field, context, optional=True)


def _region_value(value: Any, field: str, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise GeoCatalogMaintenanceError(f"{context}: {field} must be a string")
    text = " ".join(value.split())
    return text or None


def _positive_int(value: Any, field: str, context: str) -> int:
    if isinstance(value, bool):
        raise GeoCatalogMaintenanceError(f"{context}: {field} must be a positive integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.isascii() and value.isdigit():
        parsed = int(value)
    else:
        raise GeoCatalogMaintenanceError(f"{context}: {field} must be a positive integer")
    if parsed <= 0:
        raise GeoCatalogMaintenanceError(f"{context}: {field} must be a positive integer")
    return parsed


def _nonnegative_int(value: Any, field: str, context: str, *, none_value: int = 0) -> int:
    if value is None:
        return none_value
    if isinstance(value, bool):
        raise GeoCatalogMaintenanceError(f"{context}: {field} must be a non-negative integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.isascii() and value.isdigit():
        parsed = int(value)
    else:
        raise GeoCatalogMaintenanceError(f"{context}: {field} must be a non-negative integer")
    if parsed < 0:
        raise GeoCatalogMaintenanceError(f"{context}: {field} must be a non-negative integer")
    return parsed


def _coordinate(value: Any, field: str, context: str, *, bound: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GeoCatalogMaintenanceError(f"{context}: {field} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < -bound or parsed > bound:
        raise GeoCatalogMaintenanceError(f"{context}: {field} is out of range")
    return parsed


def _reject_zero_zero(context: str, lat: float, lng: float) -> None:
    if lat == 0.0 and lng == 0.0:
        raise GeoCatalogMaintenanceError(
            f"{context}: (0, 0) is not a valid coordinate"
        )


def _payload_json(item: Mapping[str, Any], context: str) -> str:
    try:
        return json.dumps(dict(item), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise GeoCatalogMaintenanceError(f"{context}: payload is not strict JSON: {exc}") from exc


def _aliases_json(value: Any, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise GeoCatalogMaintenanceError(f"{context}: aliases must be an array of strings")
    aliases: list[str] = []
    for position, alias in enumerate(value):
        text = _text(alias, f"aliases[{position}]", context)
        assert text is not None
        aliases.append(text)
    if not aliases:
        return None
    return json.dumps(aliases, ensure_ascii=False)


def _canonical_iso(iso: str) -> str:
    return _REGION_PARENT_ISO.get(iso, iso)


def _parse_json_text(value: Any, field: str, context: str) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        raise GeoCatalogMaintenanceError(f"{context}: {field} must be JSON text")
    try:
        return json.loads(value)
    except ValueError as exc:
        raise GeoCatalogMaintenanceError(f"{context}: {field} is not valid JSON") from exc


def _canonical_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the serialization-order-independent logical form of one row."""

    data = _row_dict(row)
    level = data.get("level")
    if not isinstance(level, str) or level not in LEVEL_SOURCE:
        raise GeoCatalogMaintenanceError(f"invalid logical row level: {level!r}")

    place_id = data.get("place_id")
    if not isinstance(place_id, str) or not place_id:
        raise GeoCatalogMaintenanceError(f"{level}: place_id must be a non-empty string")

    context = f"{level} {place_id}"
    source_iso = data.get("source_iso")
    country_iso = data.get("country_iso")
    if not isinstance(source_iso, str) or _ISO_RE.fullmatch(source_iso) is None:
        raise GeoCatalogMaintenanceError(f"{context}: source_iso must be two uppercase letters")
    if not isinstance(country_iso, str) or _ISO_RE.fullmatch(country_iso) is None:
        raise GeoCatalogMaintenanceError(f"{context}: country_iso must be two uppercase letters")

    aliases = _parse_json_text(data.get("aliases"), "aliases", context)
    if aliases is None:
        canonical_aliases: list[str] | None = None
    elif isinstance(aliases, list) and all(isinstance(item, str) for item in aliases):
        canonical_aliases = list(aliases)
    else:
        raise GeoCatalogMaintenanceError(f"{context}: aliases must be a JSON array of strings")

    payload = _parse_json_text(data.get("payload"), "payload", context)
    if not isinstance(payload, dict):
        raise GeoCatalogMaintenanceError(f"{context}: payload must be a JSON object")

    def optional_string(field: str) -> str | None:
        value = data.get(field)
        if value is None:
            return None
        if not isinstance(value, str):
            raise GeoCatalogMaintenanceError(f"{context}: {field} must be a string or null")
        return value

    population = data.get("population")
    if population is not None:
        population = _nonnegative_int(population, "population", context)

    geoname_id = data.get("geoname_id")
    if geoname_id is not None:
        geoname_id = _positive_int(geoname_id, "geoname_id", context)

    lat = data.get("lat")
    lng = data.get("lng")
    if lat is None or lng is None:
        if not (lat is None and lng is None):
            raise GeoCatalogMaintenanceError(f"{context}: coordinates must be a pair")
        canonical_lat = None
        canonical_lng = None
    else:
        canonical_lat = _coordinate(lat, "lat", context, bound=90.0)
        canonical_lng = _coordinate(lng, "lng", context, bound=180.0)
        _reject_zero_zero(context, canonical_lat, canonical_lng)

    coordinate_source = data.get("coordinate_source")
    if coordinate_source not in {
        ADMIN1_COORDINATE_SOURCE,
        CATALOG_COORDINATE_SOURCE,
        UNAVAILABLE_COORDINATE_SOURCE,
    }:
        raise GeoCatalogMaintenanceError(
            f"{context}: invalid coordinate_source {coordinate_source!r}"
        )

    coordinate_city_id = data.get("coordinate_city_id")
    if coordinate_city_id is not None:
        coordinate_city_id = _positive_int(coordinate_city_id, "coordinate_city_id", context)

    return {
        "level": level,
        "place_id": place_id,
        "source_iso": source_iso,
        "country_iso": country_iso,
        "parent_id": optional_string("parent_id"),
        "admin1_id": optional_string("admin1_id"),
        "code": optional_string("code"),
        "name": optional_string("name"),
        "ascii_name": optional_string("ascii_name"),
        "name_zh": optional_string("name_zh"),
        "aliases": canonical_aliases,
        "payload": payload,
        "population": population,
        "geoname_id": geoname_id,
        "lat": canonical_lat,
        "lng": canonical_lng,
        "coordinate_source": coordinate_source,
        "coordinate_city_id": coordinate_city_id,
    }


def _row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    level = str(row["level"])
    place_id = str(row["place_id"])
    if level == "city" and place_id.isascii() and place_id.isdigit():
        return (0, level, int(place_id), place_id)
    return (1, level, place_id)


def canonical_row_hash(
    rows: Iterable[Mapping[str, Any]], level: str | None = None
) -> str:
    """Return a deterministic SHA-256 of canonical logical rows.

    The row list is sorted by identity and JSON member order is normalized, so
    changing parser record order or payload formatting cannot produce a new
    revision.  The SQLite file itself is never hashed.
    """

    canonical: list[dict[str, Any]] = []
    for row in rows:
        value = _canonical_row(row)
        if level is not None and value["level"] != level:
            raise GeoCatalogMaintenanceError(
                f"expected level {level!r}, got {value['level']!r}"
            )
        canonical.append(value)
    canonical.sort(key=_row_sort_key)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def logical_revision(rows: Iterable[Mapping[str, Any]], level: str | None = None) -> str:
    """Explicit logical-revision terminology for callers."""

    return canonical_row_hash(rows, level)

def _normalise_countries(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not items:
        raise GeoCatalogMaintenanceError("countries update must contain at least one row")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, item in enumerate(items):
        context = f"countries row {position}"
        if not isinstance(item, Mapping):
            raise GeoCatalogMaintenanceError(f"{context}: row must be an object")
        iso = _text(item.get("iso"), "iso", context)
        assert iso is not None
        if _ISO_RE.fullmatch(iso) is None:
            raise GeoCatalogMaintenanceError(f"{context}: iso must be two uppercase letters")
        if iso in seen:
            raise GeoCatalogMaintenanceError(f"duplicate country iso: {iso}")
        seen.add(iso)
        iso3 = item.get("iso3")
        if iso3 is not None:
            iso3 = _text(iso3, "iso3", context)
            assert iso3 is not None
            if _ISO3_RE.fullmatch(iso3) is None:
                raise GeoCatalogMaintenanceError(f"{context}: iso3 must be three uppercase letters")
        name = _text(item.get("name"), "name", context)
        assert name is not None
        name_zh = _optional_text(item.get("name_zh"), "name_zh", context)
        lat = _coordinate(item.get("lat"), "lat", context, bound=90.0)
        lng = _coordinate(item.get("lng"), "lng", context, bound=180.0)
        _reject_zero_zero(context, lat, lng)
        rows.append(
            {
                "level": "country",
                "place_id": iso,
                "source_iso": iso,
                "country_iso": _canonical_iso(iso),
                "parent_id": None,
                "admin1_id": None,
                "code": iso3,
                "name": name,
                "ascii_name": None,
                "name_zh": name_zh,
                "aliases": None,
                "payload": _payload_json(item, context),
                "population": None,
                "geoname_id": None,
                "lat": lat,
                "lng": lng,
                "coordinate_source": CATALOG_COORDINATE_SOURCE,
                "coordinate_city_id": None,
            }
        )
    rows.sort(key=lambda row: str(row["place_id"]))
    return rows


def _normalise_cities(
    items: Sequence[Mapping[str, Any]], admin1_ids: set[str]
) -> list[dict[str, Any]]:
    if not items:
        raise GeoCatalogMaintenanceError("cities update must contain at least one row")
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for position, item in enumerate(items):
        context = f"cities row {position}"
        if not isinstance(item, Mapping):
            raise GeoCatalogMaintenanceError(f"{context}: row must be an object")
        city_id = _positive_int(item.get("city_id"), "city_id", context)
        if city_id in seen:
            raise GeoCatalogMaintenanceError(f"duplicate city_id: {city_id}")
        seen.add(city_id)
        iso = _text(item.get("iso"), "iso", context)
        assert iso is not None
        if _ISO_RE.fullmatch(iso) is None:
            raise GeoCatalogMaintenanceError(f"{context}: iso must be two uppercase letters")
        name = _text(item.get("name"), "name", context)
        assert name is not None
        ascii_name = _optional_text(item.get("ascii_name"), "ascii_name", context)
        region = _region_value(item.get("region"), "region", context)
        lat = _coordinate(item.get("lat"), "lat", context, bound=90.0)
        lng = _coordinate(item.get("lng"), "lng", context, bound=180.0)
        _reject_zero_zero(context, lat, lng)
        population = _nonnegative_int(item.get("population"), "population", context)
        matched = f"{iso}.{region}" if region else None
        if matched is not None and matched not in admin1_ids:
            matched = None
        rows.append(
            {
                "level": "city",
                "place_id": str(city_id),
                "source_iso": iso,
                "country_iso": _canonical_iso(iso),
                "parent_id": matched or iso,
                "admin1_id": matched,
                "code": region,
                "name": name,
                "ascii_name": ascii_name,
                "name_zh": None,
                "aliases": None,
                "payload": _payload_json(item, context),
                "population": population,
                "geoname_id": city_id,
                "lat": lat,
                "lng": lng,
                "coordinate_source": CATALOG_COORDINATE_SOURCE,
                "coordinate_city_id": None,
            }
        )
    rows.sort(key=lambda row: int(row["place_id"]))
    return rows


def _normalise_admin1(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not items:
        raise GeoCatalogMaintenanceError("admin1 update must contain at least one row")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, item in enumerate(items):
        context = f"admin1 row {position}"
        if not isinstance(item, Mapping):
            raise GeoCatalogMaintenanceError(f"{context}: row must be an object")
        admin1_id = _text(item.get("admin1_id"), "admin1_id", context)
        assert admin1_id is not None
        match = _ADMIN1_ID_RE.fullmatch(admin1_id)
        if match is None:
            raise GeoCatalogMaintenanceError(f"{context}: admin1_id must have the form ISO.CODE")
        iso = _text(item.get("iso"), "iso", context)
        code = _text(item.get("code"), "code", context)
        assert iso is not None and code is not None
        if _ISO_RE.fullmatch(iso) is None:
            raise GeoCatalogMaintenanceError(f"{context}: iso must be two uppercase letters")
        if _CODE_RE.fullmatch(code) is None:
            raise GeoCatalogMaintenanceError(f"{context}: code must be uppercase alphanumeric")
        if iso != match.group("iso") or code != match.group("code"):
            raise GeoCatalogMaintenanceError(f"{context}: admin1_id is inconsistent with iso/code")
        if admin1_id in seen:
            raise GeoCatalogMaintenanceError(f"duplicate admin1_id: {admin1_id}")
        seen.add(admin1_id)
        name = _text(item.get("name"), "name", context)
        ascii_name = _text(item.get("ascii_name"), "ascii_name", context)
        assert name is not None and ascii_name is not None
        geoname_id = _positive_int(item.get("geoname_id"), "geoname_id", context)
        name_zh = _optional_text(item.get("name_zh"), "name_zh", context)
        rows.append(
            {
                "level": "admin1",
                "place_id": admin1_id,
                "source_iso": iso,
                "country_iso": _canonical_iso(iso),
                "parent_id": iso,
                "admin1_id": admin1_id,
                "code": code,
                "name": name,
                "ascii_name": ascii_name,
                "name_zh": name_zh,
                "aliases": _aliases_json(item.get("aliases"), context),
                "payload": _payload_json(item, context),
                "population": None,
                "geoname_id": geoname_id,
                "lat": None,
                "lng": None,
                "coordinate_source": UNAVAILABLE_COORDINATE_SOURCE,
                "coordinate_city_id": None,
            }
        )
    rows.sort(key=lambda row: str(row["place_id"]))
    return rows


def _rebind_city_rows(
    rows: Sequence[Mapping[str, Any]], admin1_ids: set[str]
) -> list[dict[str, Any]]:
    rebound: list[dict[str, Any]] = []
    for row in rows:
        item = _row_dict(row)
        iso = item.get("source_iso")
        if not isinstance(iso, str) or _ISO_RE.fullmatch(iso) is None:
            raise GeoCatalogMaintenanceError("city row has an invalid source_iso")
        code = item.get("code")
        if code is not None and not isinstance(code, str):
            raise GeoCatalogMaintenanceError("city row has an invalid code")
        matched = f"{iso}.{code}" if code else None
        if matched is not None and matched not in admin1_ids:
            matched = None
        item["parent_id"] = matched or iso
        item["admin1_id"] = matched
        rebound.append(item)
    return rebound


def _city_winner_map(
    admin1_rows: Sequence[Mapping[str, Any]], city_rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    admin1_ids = {str(row["place_id"]) for row in admin1_rows}
    winners: dict[str, dict[str, Any]] = {}
    for row in city_rows:
        city = _row_dict(row)
        iso = city.get("source_iso")
        code = city.get("code")
        if not isinstance(iso, str) or _ISO_RE.fullmatch(iso) is None:
            raise GeoCatalogMaintenanceError("city row has an invalid source_iso")
        if not code:
            continue
        key = f"{iso}.{code}"
        if key not in admin1_ids:
            continue
        city_id_raw = city.get("geoname_id")
        if city_id_raw is None:
            city_id_raw = city.get("place_id")
        city_id = _positive_int(city_id_raw, "geoname_id", f"city {key}")
        population = _nonnegative_int(
            city.get("population"), "population", f"city {city_id}"
        )
        lat = _coordinate(city.get("lat"), "lat", f"city {city_id}", bound=90.0)
        lng = _coordinate(city.get("lng"), "lng", f"city {city_id}", bound=180.0)
        _reject_zero_zero(f"city {city_id}", lat, lng)
        current = winners.get(key)
        if current is None or (population, -city_id) > (
            int(current["population"]),
            -int(current["geoname_id"]),
        ):
            winners[key] = {
                "city_id": city_id,
                "population": population,
                "lat": lat,
                "lng": lng,
            }
    return winners


def _apply_admin1_winners(
    admin1_rows: Sequence[Mapping[str, Any]], city_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    winners = _city_winner_map(admin1_rows, city_rows)
    result: list[dict[str, Any]] = []
    for row in admin1_rows:
        item = _row_dict(row)
        winner = winners.get(str(item["place_id"]))
        if winner is None:
            item.update(
                {
                    "lat": None,
                    "lng": None,
                    "coordinate_source": UNAVAILABLE_COORDINATE_SOURCE,
                    "coordinate_city_id": None,
                }
            )
        else:
            item.update(
                {
                    "lat": float(winner["lat"]),
                    "lng": float(winner["lng"]),
                    "coordinate_source": ADMIN1_COORDINATE_SOURCE,
                    "coordinate_city_id": int(winner["city_id"]),
                }
            )
        result.append(item)
    return result


def _validate_row_contracts(rows: Sequence[Mapping[str, Any]]) -> None:
    identities: set[tuple[str, str]] = set()
    for row in rows:
        canonical = _canonical_row(row)
        key = (str(canonical["level"]), str(canonical["place_id"]))
        if key in identities:
            raise GeoCatalogMaintenanceError(f"duplicate logical identity: {key}")
        identities.add(key)
        context = f"{canonical['level']} {canonical['place_id']}"
        lat = canonical["lat"]
        lng = canonical["lng"]
        source = canonical["coordinate_source"]
        if canonical["level"] in {"country", "city"}:
            if lat is None or lng is None or source != CATALOG_COORDINATE_SOURCE:
                raise GeoCatalogMaintenanceError(
                    f"{context}: {canonical['level']} rows require catalog coordinates"
                )
        elif source == UNAVAILABLE_COORDINATE_SOURCE:
            if lat is not None or lng is not None or canonical["coordinate_city_id"] is not None:
                raise GeoCatalogMaintenanceError(
                    f"{context}: unavailable coordinates must be NULL"
                )
        elif source == ADMIN1_COORDINATE_SOURCE:
            if lat is None or lng is None or canonical["coordinate_city_id"] is None:
                raise GeoCatalogMaintenanceError(
                    f"{context}: derived coordinates require a provenance city"
                )
        else:
            raise GeoCatalogMaintenanceError(
                f"{context}: admin1 rows use derived or unavailable coordinates"
            )


def _final_rows(
    state_sources: Mapping[str, Mapping[str, Any]],
    current_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    countries: Sequence[Mapping[str, Any]] | None,
    admin1: Sequence[Mapping[str, Any]] | None,
    cities: Sequence[Mapping[str, Any]] | None,
) -> dict[str, list[dict[str, Any]]]:
    del state_sources
    final_countries = (
        _normalise_countries(countries)
        if countries is not None
        else [_row_dict(row) for row in current_rows["countries"]]
    )

    if admin1 is not None:
        admin1_base = _normalise_admin1(admin1)
    else:
        admin1_base = [_row_dict(row) for row in current_rows["admin1"]]
    admin1_ids = {str(row["place_id"]) for row in admin1_base}

    if cities is not None:
        final_cities = _normalise_cities(cities, admin1_ids)
    elif admin1 is not None:
        final_cities = _rebind_city_rows(current_rows["cities"], admin1_ids)
    else:
        final_cities = [_row_dict(row) for row in current_rows["cities"]]

    if admin1 is not None or cities is not None:
        final_admin1 = _apply_admin1_winners(admin1_base, final_cities)
    else:
        final_admin1 = admin1_base
    final_rows = {
        "countries": final_countries,
        "admin1": final_admin1,
        "cities": final_cities,
    }
    for level_rows in final_rows.values():
        _validate_row_contracts(level_rows)
    return final_rows

def _valid_metadata(row: Mapping[str, Any] | None) -> bool:
    if row is None:
        return False
    fingerprint = row.get("fingerprint")
    record_count = row.get("record_count")
    path = row.get("path")
    version = row.get("import_version")
    imported_at = row.get("imported_at")
    return (
        isinstance(path, str)
        and bool(path.strip())
        and isinstance(fingerprint, str)
        and _FINGERPRINT_RE.fullmatch(fingerprint) is not None
        and isinstance(record_count, int)
        and not isinstance(record_count, bool)
        and record_count >= 0
        and isinstance(version, str)
        and bool(version.strip())
        and isinstance(imported_at, str)
        and bool(imported_at.strip())
    )


def _metadata_row(
    source: str,
    *,
    path: str,
    fingerprint: str,
    record_count: int,
    import_version: str,
    imported_at: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "path": path,
        "fingerprint": fingerprint,
        "record_count": record_count,
        "import_version": import_version,
        "imported_at": imported_at,
    }


def _updated_metadata(
    source: str,
    existing: Mapping[str, Any] | None,
    *,
    fingerprint: str,
    record_count: int,
    imported_at: str,
) -> dict[str, Any]:
    current = dict(existing or {})
    path = current.get("path")
    if not isinstance(path, str) or not path.strip():
        path = _DEFAULT_PROVENANCE[source]
    version = current.get("import_version")
    if not isinstance(version, str) or not version.strip():
        version = IMPORT_VERSION
    return _metadata_row(
        source,
        path=path,
        fingerprint=fingerprint,
        record_count=record_count,
        import_version=version,
        imported_at=imported_at,
    )


def _build_plan(
    database: Path,
    sources: Mapping[str, Mapping[str, Any]],
    current_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    countries: Sequence[Mapping[str, Any]] | None,
    admin1: Sequence[Mapping[str, Any]] | None,
    cities: Sequence[Mapping[str, Any]] | None,
    imported_at: str | None,
) -> CatalogMaintenancePlan:
    for name, supplied in (("countries", countries), ("admin1", admin1), ("cities", cities)):
        if supplied is not None and not supplied:
            raise GeoCatalogMaintenanceError(f"{name} update must contain at least one row")

    final_rows = _final_rows(
        sources,
        current_rows,
        countries=countries,
        admin1=admin1,
        cities=cities,
    )

    changed_sources: set[str] = set()
    touched_sources: set[str] = set()
    revisions: dict[str, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    timestamp = imported_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")

    for source in SOURCE_NAMES:
        rows = final_rows[source]
        current = current_rows[source]
        final_hash = canonical_row_hash(rows, SOURCE_LEVEL[source])
        current_hash = canonical_row_hash(current, SOURCE_LEVEL[source])
        row_changed = final_hash != current_hash
        if row_changed:
            changed_sources.add(source)

        existing = sources.get(source)
        if not _valid_metadata(existing):
            touched_sources.add(source)
            revisions[source] = final_hash
            metadata[source] = _updated_metadata(
                source,
                existing,
                fingerprint=final_hash,
                record_count=len(rows),
                imported_at=timestamp,
            )
            continue

        assert existing is not None
        if row_changed:
            touched_sources.add(source)
            revisions[source] = final_hash
            metadata[source] = _updated_metadata(
                source,
                existing,
                fingerprint=final_hash,
                record_count=len(rows),
                imported_at=timestamp,
            )
            continue

        # The logical rows are equal. Preserve the existing fingerprint exactly,
        # even if it came from the retired JSON-byte revision scheme. This keeps
        # inference/backend catalog revisions stable across payload reformatting.
        revisions[source] = str(existing["fingerprint"])
        if int(existing["record_count"]) != len(rows):
            touched_sources.add(source)
            metadata[source] = _updated_metadata(
                source,
                existing,
                fingerprint=str(existing["fingerprint"]),
                record_count=len(rows),
                imported_at=timestamp,
            )
        else:
            metadata[source] = dict(existing)

    with_coordinates = sum(
        1
        for row in final_rows["admin1"]
        if row.get("lat") is not None and row.get("lng") is not None
    )
    missing_coordinates = len(final_rows["admin1"]) - with_coordinates
    counts = {source: len(final_rows[source]) for source in SOURCE_NAMES}
    frozen_rows = {
        source: tuple(dict(row) for row in final_rows[source])
        for source in SOURCE_NAMES
    }
    return CatalogMaintenancePlan(
        database=database,
        rows=frozen_rows,
        metadata={source: dict(metadata[source]) for source in SOURCE_NAMES},
        revisions=dict(revisions),
        counts=counts,
        changed_sources=frozenset(changed_sources),
        touched_sources=frozenset(touched_sources),
        admin1_with_coordinates=with_coordinates,
        admin1_missing_coordinates=missing_coordinates,
    )


def _read_only_uri(path: Path) -> str:
    return "file:" + quote(path.as_posix(), safe="/:") + "?mode=ro"


def _required_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    except sqlite3.Error as exc:
        raise GeoCatalogMaintenanceError(f"cannot inspect database table {table}: {exc}") from exc
    return {str(row["name"]) for row in rows}


def _load_state(
    connection: sqlite3.Connection,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    tables = {
        str(row["name"])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    if "geo_place" not in tables or "geo_catalog_source" not in tables:
        raise GeoCatalogMaintenanceError(
            "database is not a geo catalog: missing geo_place or geo_catalog_source"
        )
    for table, required in (
        ("geo_place", set(_GEO_PLACE_COLUMNS) | {"id"}),
        ("geo_catalog_source", set(_SOURCE_COLUMNS)),
    ):
        missing = sorted(required - _required_columns(connection, table))
        if missing:
            raise GeoCatalogMaintenanceError(
                f"database table {table} is missing columns: {', '.join(missing)}"
            )

    sources: dict[str, dict[str, Any]] = {}
    for row in connection.execute("SELECT * FROM geo_catalog_source"):
        sources[str(row["source"])] = _row_dict(row)

    rows: dict[str, list[dict[str, Any]]] = {source: [] for source in SOURCE_NAMES}
    for row in connection.execute("SELECT * FROM geo_place ORDER BY id"):
        item = _row_dict(row)
        level = item.get("level")
        if not isinstance(level, str) or level not in LEVEL_SOURCE:
            raise GeoCatalogMaintenanceError(f"geo_place contains unsupported level: {level!r}")
        rows[LEVEL_SOURCE[level]].append(item)
    return sources, rows


def _open_read_only(database: Path) -> sqlite3.Connection:
    if not database.is_file():
        raise GeoCatalogMaintenanceError(f"geo catalog database not found: {database}")
    try:
        connection = sqlite3.connect(_read_only_uri(database), uri=True, timeout=5.0)
    except sqlite3.Error as exc:
        raise GeoCatalogMaintenanceError(f"cannot open geo catalog database {database}: {exc}") from exc
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only = ON")
    except sqlite3.Error:
        connection.close()
        raise
    return connection


def plan_catalog_levels(
    database: str | Path = DEFAULT_DATABASE_PATH,
    *,
    countries: Sequence[Mapping[str, Any]] | None = None,
    admin1: Sequence[Mapping[str, Any]] | None = None,
    cities: Sequence[Mapping[str, Any]] | None = None,
) -> CatalogMaintenancePlan:
    """Derive the final rows/metadata without writing the database."""

    path = Path(database).expanduser().resolve()
    connection = _open_read_only(path)
    try:
        sources, current_rows = _load_state(connection)
        return _build_plan(
            path,
            sources,
            current_rows,
            countries=countries,
            admin1=admin1,
            cities=cities,
            imported_at=None,
        )
    finally:
        connection.close()


def check_catalog_levels(
    database: str | Path = DEFAULT_DATABASE_PATH,
    *,
    countries: Sequence[Mapping[str, Any]] | None = None,
    admin1: Sequence[Mapping[str, Any]] | None = None,
    cities: Sequence[Mapping[str, Any]] | None = None,
) -> CatalogMaintenanceCheck:
    """Compare proposed logical rows/revisions with the stored database."""

    path = Path(database).expanduser().resolve()
    plan = plan_catalog_levels(path, countries=countries, admin1=admin1, cities=cities)
    connection = _open_read_only(path)
    try:
        sources, current_rows = _load_state(connection)
    finally:
        connection.close()

    mismatches: list[str] = []
    for source in SOURCE_NAMES:
        current_hash = canonical_row_hash(current_rows[source], SOURCE_LEVEL[source])
        expected_hash = canonical_row_hash(plan.rows[source], SOURCE_LEVEL[source])
        if current_hash != expected_hash:
            mismatches.append(f"{source}:rows")
        current_meta = sources.get(source)
        expected_meta = plan.metadata[source]
        if current_meta is None:
            mismatches.append(f"{source}:metadata")
            continue
        for field in ("fingerprint", "record_count", "path", "import_version"):
            if current_meta.get(field) != expected_meta.get(field):
                mismatches.append(f"{source}:{field}")
    return CatalogMaintenanceCheck(
        ok=not mismatches,
        mismatches=tuple(mismatches),
        plan=plan,
    )

def _insert_geo_place_sql() -> str:
    placeholders = ", ".join("?" for _ in _GEO_PLACE_COLUMNS)
    update_columns = [
        column for column in _GEO_PLACE_COLUMNS if column not in {"level", "place_id"}
    ]
    updates = ", ".join(f"{column} = excluded.{column}" for column in update_columns)
    return (
        f"INSERT INTO geo_place ({', '.join(_GEO_PLACE_COLUMNS)}) "
        f"VALUES ({placeholders}) "
        "ON CONFLICT(level, place_id) DO UPDATE SET "
        + updates
    )


def _replace_level(
    connection: sqlite3.Connection, level: str, rows: Sequence[Mapping[str, Any]]
) -> None:
    existing_ids = [
        str(row["place_id"])
        for row in connection.execute(
            "SELECT place_id FROM geo_place WHERE level = ?", (level,)
        )
    ]
    sql = _insert_geo_place_sql()
    connection.executemany(
        sql,
        [tuple(row.get(column) for column in _GEO_PLACE_COLUMNS) for row in rows],
    )
    incoming = {str(row["place_id"]) for row in rows}
    for place_id in existing_ids:
        if place_id not in incoming:
            connection.execute(
                "DELETE FROM geo_place WHERE level = ? AND place_id = ?",
                (level, place_id),
            )


def _upsert_source(connection: sqlite3.Connection, row: Mapping[str, Any]) -> None:
    connection.execute(
        """
        INSERT INTO geo_catalog_source
            (source, path, fingerprint, record_count, import_version, imported_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(source) DO UPDATE SET
            path = excluded.path,
            fingerprint = excluded.fingerprint,
            record_count = excluded.record_count,
            import_version = excluded.import_version,
            imported_at = excluded.imported_at
        """,
        (
            row["source"],
            row["path"],
            row["fingerprint"],
            row["record_count"],
            row["import_version"],
            row["imported_at"],
        ),
    )


def _validate_committed_state(
    connection: sqlite3.Connection, plan: CatalogMaintenancePlan
) -> None:
    sources, rows = _load_state(connection)
    if set(sources) != set(SOURCE_NAMES):
        raise GeoCatalogMaintenanceError(
            "geo_catalog_source must contain exactly countries, admin1 and cities"
        )
    for source in SOURCE_NAMES:
        meta = sources[source]
        if not _valid_metadata(meta):
            raise GeoCatalogMaintenanceError(f"{source}: invalid source metadata")
        if int(meta["record_count"]) != len(rows[source]):
            raise GeoCatalogMaintenanceError(
                f"{source}: record_count does not match stored rows"
            )
        if str(meta["fingerprint"]) != str(plan.revisions[source]):
            raise GeoCatalogMaintenanceError(f"{source}: fingerprint does not match plan")
        if canonical_row_hash(rows[source], SOURCE_LEVEL[source]) != canonical_row_hash(
            plan.rows[source], SOURCE_LEVEL[source]
        ):
            raise GeoCatalogMaintenanceError(f"{source}: stored rows do not match plan")
        _validate_row_contracts(rows[source])

    admin1_by_id = {str(row["place_id"]): row for row in rows["admin1"]}
    winners = _city_winner_map(rows["admin1"], rows["cities"])
    for admin1_id, admin1_row in admin1_by_id.items():
        winner = winners.get(admin1_id)
        expected_source = (
            ADMIN1_COORDINATE_SOURCE if winner is not None else UNAVAILABLE_COORDINATE_SOURCE
        )
        expected_lat = float(winner["lat"]) if winner is not None else None
        expected_lng = float(winner["lng"]) if winner is not None else None
        expected_city_id = int(winner["city_id"]) if winner is not None else None
        if (
            admin1_row.get("coordinate_source") != expected_source
            or admin1_row.get("lat") != expected_lat
            or admin1_row.get("lng") != expected_lng
            or admin1_row.get("coordinate_city_id") != expected_city_id
        ):
            raise GeoCatalogMaintenanceError(
                f"admin1 {admin1_id}: derived point is not coherent with cities"
            )

    admin1_ids = set(admin1_by_id)
    for city in rows["cities"]:
        admin1_id = city.get("admin1_id")
        if admin1_id is not None and admin1_id not in admin1_ids:
            raise GeoCatalogMaintenanceError(
                f"city {city.get('place_id')}: admin1_{admin1_id} is not present"
            )


def replace_catalog_levels(
    database: str | Path = DEFAULT_DATABASE_PATH,
    *,
    countries: Sequence[Mapping[str, Any]] | None = None,
    admin1: Sequence[Mapping[str, Any]] | None = None,
    cities: Sequence[Mapping[str, Any]] | None = None,
    imported_at: str | None = None,
) -> CatalogMaintenanceResult:
    """Atomically update the supplied logical datasets.

    ``countries``, ``admin1`` and ``cities`` use the parser output shapes.  A
    ``None`` dataset is preserved verbatim.  Cities always cause admin1 points
    to be re-derived; an admin1 edit also rebinds city parent/admin1 keys.
    """

    path = Path(database).expanduser().resolve()
    if not path.is_file():
        raise GeoCatalogMaintenanceError(f"geo catalog database not found: {path}")

    connection: sqlite3.Connection | None = None
    plan: CatalogMaintenancePlan | None = None
    try:
        connection = sqlite3.connect(str(path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.isolation_level = None
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("BEGIN IMMEDIATE")
        sources, current_rows = _load_state(connection)
        plan = _build_plan(
            path,
            sources,
            current_rows,
            countries=countries,
            admin1=admin1,
            cities=cities,
            imported_at=imported_at
            or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f"),
        )

        for source in SOURCE_NAMES:
            if source in plan.changed_sources:
                _replace_level(connection, SOURCE_LEVEL[source], plan.rows[source])

        extra_sources = set(sources) - set(SOURCE_NAMES)
        for source in sorted(extra_sources):
            connection.execute("DELETE FROM geo_catalog_source WHERE source = ?", (source,))
        for source in SOURCE_NAMES:
            if source in plan.touched_sources:
                _upsert_source(connection, plan.metadata[source])

        _validate_committed_state(connection, plan)
        connection.execute("COMMIT")
    except BaseException:
        if connection is not None:
            try:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
        raise
    finally:
        if connection is not None:
            connection.close()

    assert plan is not None
    return CatalogMaintenanceResult(
        database=path,
        counts=dict(plan.counts),
        revisions=dict(plan.revisions),
        changed_sources=tuple(
            source for source in SOURCE_NAMES if source in plan.changed_sources
        ),
        touched_sources=tuple(
            source for source in SOURCE_NAMES if source in plan.touched_sources
        ),
        admin1_with_coordinates=plan.admin1_with_coordinates,
        admin1_missing_coordinates=plan.admin1_missing_coordinates,
    )


def update_countries(
    database: str | Path, countries: Sequence[Mapping[str, Any]]
) -> CatalogMaintenanceResult:
    return replace_catalog_levels(database, countries=countries)


def update_admin1(
    database: str | Path, admin1: Sequence[Mapping[str, Any]]
) -> CatalogMaintenanceResult:
    return replace_catalog_levels(database, admin1=admin1)


def update_cities(
    database: str | Path, cities: Sequence[Mapping[str, Any]]
) -> CatalogMaintenanceResult:
    return replace_catalog_levels(database, cities=cities)


def update_countries_and_cities(
    database: str | Path,
    countries: Sequence[Mapping[str, Any]],
    cities: Sequence[Mapping[str, Any]],
) -> CatalogMaintenanceResult:
    return replace_catalog_levels(database, countries=countries, cities=cities)


write_catalog_levels = replace_catalog_levels
maintain_catalog = replace_catalog_levels


class CatalogMaintenanceWriter:
    """Object wrapper around :func:`replace_catalog_levels`."""

    def __init__(self, database: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.database = Path(database).expanduser().resolve()

    def replace(
        self,
        *,
        countries: Sequence[Mapping[str, Any]] | None = None,
        admin1: Sequence[Mapping[str, Any]] | None = None,
        cities: Sequence[Mapping[str, Any]] | None = None,
    ) -> CatalogMaintenanceResult:
        return replace_catalog_levels(
            self.database,
            countries=countries,
            admin1=admin1,
            cities=cities,
        )

    def plan(
        self,
        *,
        countries: Sequence[Mapping[str, Any]] | None = None,
        admin1: Sequence[Mapping[str, Any]] | None = None,
        cities: Sequence[Mapping[str, Any]] | None = None,
    ) -> CatalogMaintenancePlan:
        return plan_catalog_levels(
            self.database,
            countries=countries,
            admin1=admin1,
            cities=cities,
        )

    def check(
        self,
        *,
        countries: Sequence[Mapping[str, Any]] | None = None,
        admin1: Sequence[Mapping[str, Any]] | None = None,
        cities: Sequence[Mapping[str, Any]] | None = None,
    ) -> CatalogMaintenanceCheck:
        return check_catalog_levels(
            self.database,
            countries=countries,
            admin1=admin1,
            cities=cities,
        )

    def countries(self, rows: Sequence[Mapping[str, Any]]) -> CatalogMaintenanceResult:
        return self.replace(countries=rows)

    def admin1(self, rows: Sequence[Mapping[str, Any]]) -> CatalogMaintenanceResult:
        return self.replace(admin1=rows)

    def cities(self, rows: Sequence[Mapping[str, Any]]) -> CatalogMaintenanceResult:
        return self.replace(cities=rows)


__all__ = [
    "ADMIN1_COORDINATE_SOURCE",
    "CATALOG_COORDINATE_SOURCE",
    "CatalogMaintenanceCheck",
    "CatalogMaintenancePlan",
    "CatalogMaintenanceResult",
    "CatalogMaintenanceWriter",
    "DEFAULT_DATABASE_PATH",
    "GeoCatalogMaintenanceError",
    "IMPORT_VERSION",
    "LEVEL_SOURCE",
    "MaintenanceError",
    "SOURCE_LEVEL",
    "SOURCE_NAMES",
    "UNAVAILABLE_COORDINATE_SOURCE",
    "canonical_row_hash",
    "check_catalog_levels",
    "logical_revision",
    "maintain_catalog",
    "plan_catalog_levels",
    "replace_catalog_levels",
    "update_admin1",
    "update_cities",
    "update_countries",
    "update_countries_and_cities",
    "write_catalog_levels",
]
