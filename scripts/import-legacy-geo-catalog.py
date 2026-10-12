#!/usr/bin/env python3
"""One-time legacy import: legacy JSON catalogs -> portable SQLite seed.

Self-contained and standard-library only. It deliberately does NOT import the
backend bootstrap modules: the legacy JSON validation and row derivation that
used to live in ``backend.app.services.geo_catalog_import`` together with
``admin1_catalog`` / ``city_catalog`` is ported here (bounded to the frozen
seed schema), so this utility keeps working after the backend bootstrap stops
reading the JSON catalogs. No Settings, no JSON path overrides, no SQLAlchemy.

It writes ``data/geo_catalog.sqlite3`` -- exactly the two seed tables
``geo_place`` and ``geo_catalog_source`` plus their indexes -- from explicit
inputs:

* ``--countries-json``, ``--admin1-json``, ``--cities-json`` and ``--output``
  are all required; there are no baked-in JSON defaults and normal startup
  never calls this script;
* ``geo_catalog_source.fingerprint`` records the SHA-256 of the raw input JSON
  bytes (the logical dataset revision consumers compare), never a digest taken
  from publishing metadata and never a hash of the SQLite file;
* ``geo_catalog_source.path`` records portable historical provenance
  (``legacy-json:countries.json`` and friends), never a local absolute path;
* the staged database is read back through the shipped reader
  (:class:`catalog_db.CatalogDatabase`) before an atomic ``os.replace``; on any
  failure the previous asset is left untouched;
* an existing ``--output`` is replaced only when it already looks like a geo
  catalog seed (``geo_place`` + ``geo_catalog_source``); anything else is
  unknown user data and requires an explicit ``--force``.

Admin1 representative coordinates follow the frozen
``admin1-largest-city-v1`` policy: the highest-population city in the same raw
source ISO and exact region code wins (numeric city id ascending on ties);
directory rows without a matching city keep a NULL pair marked ``unavailable``
and are never given an average, a country point or ``(0, 0)``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from catalog_db import CatalogDatabase, CatalogDatabaseError  # noqa: E402

SOURCE_NAMES: tuple[str, ...] = ("countries", "admin1", "cities")

#: Frozen import/derivation policy version stored in geo_catalog_source.
IMPORT_VERSION = "geo-catalog-v1+admin1-largest-city-v1+china-subordinate-regions-v1"

ADMIN1_COORDINATE_SOURCE = "admin1-largest-city-v1"
CATALOG_COORDINATE_SOURCE = "catalog"
UNAVAILABLE_COORDINATE_SOURCE = "unavailable"

#: Portable provenance template for the legacy JSON inputs.
LEGACY_PROVENANCE = "legacy-json:{name}.json"

#: Application-level CN grouping policy (raw source scopes stay distinct).
_REGION_PARENT_ISO = {"TW": "CN", "HK": "CN", "MO": "CN"}

_WRAPPER_KEYS = ("rows", "items", "countries", "admin1", "cities")
_ISO_RE = re.compile(r"^[A-Z]{2}$")
_ISO3_RE = re.compile(r"^[A-Z]{3}$")
_ADMIN1_ID_RE = re.compile(r"^(?P<iso>[A-Z]{2})\.(?P<code>[A-Z0-9]+)$")
_CODE_RE = re.compile(r"^[A-Z0-9]+$")


class LegacyCatalogError(ValueError):
    """Raised when an input catalog is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class Admin1Row:
    admin1_id: str
    iso: str
    code: str
    name: str
    ascii_name: str
    geoname_id: int
    name_zh: str | None
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class CityRow:
    city_id: int
    iso: str
    name: str
    region: str
    lat: float
    lng: float
    population: int


@dataclass(frozen=True)
class SourceRecord:
    name: str
    fingerprint: str
    record_count: int


@dataclass(frozen=True)
class SeedSnapshot:
    sources: tuple[SourceRecord, ...]
    rows: tuple[dict[str, Any], ...]
    counts: dict[str, int]
    admin1_with_coordinates: int
    admin1_missing_coordinates: int


# ---------------------------------------------------------------------------
# raw source reading / shared scalar validation (ported from the old importer)
# ---------------------------------------------------------------------------
def _read_source(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise LegacyCatalogError(f"{label} catalog is unreadable: {path}") from exc


def _decode_source(raw: bytes, path: Path, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LegacyCatalogError(f"{label} catalog is not valid UTF-8: {path}") from exc
    if text.startswith("\ufeff"):
        text = text[1:]
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LegacyCatalogError(f"{label} catalog is not valid JSON: {path}") from exc


def _source_rows(payload: Any, label: str) -> list[Any]:
    rows: Any = None
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        for key in _WRAPPER_KEYS:
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
    if not isinstance(rows, list) or not rows:
        raise LegacyCatalogError(f"{label} catalog must be a non-empty JSON array")
    return rows


def _text(
    value: Any, field: str, label: str, position: int, *, optional: bool = False
) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise LegacyCatalogError(
            f"{label} catalog item {position}: {field} must be a string"
        )
    text = " ".join(value.split())
    if not text:
        raise LegacyCatalogError(
            f"{label} catalog item {position}: {field} must not be blank"
        )
    return text


def _coordinate(
    value: Any, field: str, label: str, position: int, *, bound: float
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LegacyCatalogError(
            f"{label} catalog item {position}: {field} must be a finite number"
        )
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < -bound or parsed > bound:
        raise LegacyCatalogError(
            f"{label} catalog item {position}: {field} is out of range"
        )
    return parsed


def _reject_zero_zero(label: str, position: int, lat: float, lng: float) -> None:
    """``(0, 0)`` is a placeholder, never a catalog coordinate (Null Island)."""

    if lat == 0.0 and lng == 0.0:
        raise LegacyCatalogError(
            f"{label} catalog item {position}: (0, 0) is not a valid coordinate"
        )


def _payload_text(payload: Any, label: str, position: int) -> str:
    try:
        return json.dumps(dict(payload), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LegacyCatalogError(
            f"{label} catalog item {position}: payload is not strict JSON: {exc}"
        ) from exc


def _canonical_iso(iso: str) -> str:
    return _REGION_PARENT_ISO.get(iso, iso)

# ---------------------------------------------------------------------------
# country / admin1 / city validation (ported, bounded to the seed contract)
# ---------------------------------------------------------------------------
def _load_countries(payload: Any) -> list[dict[str, Any]]:
    rows = _source_rows(payload, "countries")
    seen: set[str] = set()
    countries: list[dict[str, Any]] = []
    for position, item in enumerate(rows):
        if not isinstance(item, dict):
            raise LegacyCatalogError(
                f"countries catalog item {position} is not an object"
            )
        iso = item.get("iso")
        if not isinstance(iso, str) or _ISO_RE.fullmatch(iso) is None:
            raise LegacyCatalogError(
                f"countries catalog item {position}: iso must be two uppercase letters"
            )
        if iso in seen:
            raise LegacyCatalogError(f"duplicate iso in countries catalog: {iso}")
        seen.add(iso)
        iso3 = item.get("iso3")
        if iso3 is not None and (
            not isinstance(iso3, str) or _ISO3_RE.fullmatch(iso3) is None
        ):
            raise LegacyCatalogError(
                f"countries catalog item {position}: iso3 must be three uppercase letters"
            )
        name = _text(item.get("name"), "name", "countries", position)
        name_zh = _text(
            item.get("name_zh"), "name_zh", "countries", position, optional=True
        )
        lat = _coordinate(item.get("lat"), "lat", "countries", position, bound=90.0)
        lng = _coordinate(item.get("lng"), "lng", "countries", position, bound=180.0)
        _reject_zero_zero("countries", position, lat, lng)
        countries.append(
            {
                "iso": iso,
                "iso3": iso3,
                "name": name,
                "name_zh": name_zh,
                "lat": lat,
                "lng": lng,
                "payload": item,
            }
        )
    countries.sort(key=lambda country: country["iso"])
    return countries


def _coerce_geoname_id(value: Any, index: int) -> int:
    if isinstance(value, bool):
        raise LegacyCatalogError(
            f"admin1 catalog item {index}: geoname_id must be a positive integer"
        )
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.isascii() and value.isdigit():
        parsed = int(value)
    else:
        raise LegacyCatalogError(
            f"admin1 catalog item {index}: geoname_id must be a positive integer"
        )
    if parsed <= 0:
        raise LegacyCatalogError(
            f"admin1 catalog item {index}: geoname_id must be a positive integer"
        )
    return parsed


def _coerce_aliases(value: Any, index: int) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise LegacyCatalogError(
            f"admin1 catalog item {index}: aliases must be an array of strings"
        )
    aliases: list[str] = []
    for alias_position, alias in enumerate(value):
        text = _text(alias, f"aliases[{alias_position}]", "admin1", index)
        assert text is not None
        aliases.append(text)
    return tuple(aliases)


def _admin1_from_mapping(item: Any, index: int) -> Admin1Row:
    if not isinstance(item, dict):
        raise LegacyCatalogError(f"admin1 catalog item {index}: item is not an object")

    admin1_id = item.get("admin1_id")
    if not isinstance(admin1_id, str):
        raise LegacyCatalogError(
            f"admin1 catalog item {index}: admin1_id must be a string"
        )
    match = _ADMIN1_ID_RE.fullmatch(admin1_id)
    if match is None:
        raise LegacyCatalogError(
            f"admin1 catalog item {index}: admin1_id must have the form ISO.CODE"
        )

    iso = item.get("iso")
    code = item.get("code")
    if not isinstance(iso, str) or _ISO_RE.fullmatch(iso) is None:
        raise LegacyCatalogError(
            f"admin1 catalog item {index}: iso must be two uppercase letters"
        )
    if not isinstance(code, str) or _CODE_RE.fullmatch(code) is None:
        raise LegacyCatalogError(
            f"admin1 catalog item {index}: code must be a non-empty uppercase "
            "alphanumeric string"
        )
    if iso != match.group("iso") or code != match.group("code"):
        raise LegacyCatalogError(
            f"admin1 catalog item {index}: admin1_id is inconsistent with iso/code"
        )

    name = _text(item.get("name"), "name", "admin1", index)
    ascii_name = _text(item.get("ascii_name"), "ascii_name", "admin1", index)
    geoname_id = _coerce_geoname_id(item.get("geoname_id"), index)
    name_zh = _text(item.get("name_zh"), "name_zh", "admin1", index, optional=True)
    aliases = _coerce_aliases(item.get("aliases", ()), index)

    assert name is not None and ascii_name is not None
    return Admin1Row(
        admin1_id=admin1_id,
        iso=iso,
        code=code,
        name=name,
        ascii_name=ascii_name,
        geoname_id=geoname_id,
        name_zh=name_zh,
        aliases=aliases,
    )


def _load_admin1(payload: Any) -> list[tuple[Admin1Row, dict[str, Any]]]:
    rows = _source_rows(payload, "admin1")
    seen: set[str] = set()
    validated: list[tuple[Admin1Row, dict[str, Any]]] = []
    for position, item in enumerate(rows):
        row = _admin1_from_mapping(item, position)
        if row.admin1_id in seen:
            raise LegacyCatalogError(
                f"duplicate admin1_id in admin1 catalog: {row.admin1_id}"
            )
        seen.add(row.admin1_id)
        validated.append((row, item))
    validated.sort(key=lambda entry: entry[0].admin1_id)
    return validated


def _coerce_city_id(value: Any) -> int:
    if isinstance(value, bool):
        raise LegacyCatalogError("city_id must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LegacyCatalogError("city_id must be an integer") from exc
    if parsed <= 0:
        raise LegacyCatalogError("city_id must be positive")
    return parsed


def _city_text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if value is None and allow_empty:
        return ""
    if not isinstance(value, str):
        raise LegacyCatalogError(f"{field} must be a string")
    text = " ".join(value.split())
    if not text and not allow_empty:
        raise LegacyCatalogError(f"{field} must not be empty")
    return text


def _city_coordinate(value: Any, field: str, *, bound: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise LegacyCatalogError(f"{field} must be a number") from exc
    if not math.isfinite(parsed) or parsed < -bound or parsed > bound:
        raise LegacyCatalogError(f"{field} is out of range")
    return parsed


def _city_population(value: Any) -> int:
    if value is None:
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LegacyCatalogError("population must be an integer") from exc
    if parsed < 0:
        raise LegacyCatalogError("population must be non-negative")
    return parsed


def _city_from_mapping(item: Any, index: int) -> CityRow:
    if not isinstance(item, dict):
        raise LegacyCatalogError(f"city catalog item {index} is not an object")
    try:
        city_id = _coerce_city_id(item.get("city_id"))
        iso = _city_text(item.get("iso"), "iso").upper()
        name = _city_text(item.get("name"), "name")
        region = _city_text(item.get("region"), "region", allow_empty=True)
        lat = _city_coordinate(item.get("lat"), "lat", bound=90.0)
        lng = _city_coordinate(item.get("lng"), "lng", bound=180.0)
        population = _city_population(item.get("population"))
    except LegacyCatalogError as exc:
        raise LegacyCatalogError(f"city catalog item {index}: {exc}") from exc
    if not _ISO_RE.match(iso):
        raise LegacyCatalogError(f"city catalog item {index}: iso must be two letters")
    return CityRow(
        city_id=city_id,
        iso=iso,
        name=name,
        region=region,
        lat=lat,
        lng=lng,
        population=population,
    )


def _load_cities(payload: Any) -> list[tuple[CityRow, str | None, dict[str, Any]]]:
    rows = _source_rows(payload, "cities")
    seen: set[int] = set()
    validated: list[tuple[CityRow, str | None, dict[str, Any]]] = []
    for position, item in enumerate(rows):
        row = _city_from_mapping(item, position)
        if row.city_id in seen:
            raise LegacyCatalogError(
                f"duplicate city_id in cities catalog: {row.city_id}"
            )
        seen.add(row.city_id)
        _reject_zero_zero("cities", position, row.lat, row.lng)
        ascii_name = _text(
            item.get("ascii_name"), "ascii_name", "cities", position, optional=True
        )
        validated.append((row, ascii_name, item))
    return validated

# ---------------------------------------------------------------------------
# snapshot building (source rows -> frozen geo_place rows)
# ---------------------------------------------------------------------------
def _build_rows(
    countries: Sequence[dict[str, Any]],
    admin1_rows: Sequence[tuple[Admin1Row, dict[str, Any]]],
    city_rows: Sequence[tuple[CityRow, str | None, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], int, int]:
    """Derive every geo_place row, including admin1 representative points."""

    admin1_ids = {row.admin1_id for row, _ in admin1_rows}
    winners: dict[str, CityRow] = {}
    for city, _, _ in city_rows:
        if not city.region:
            continue
        key = f"{city.iso}.{city.region}"
        if key not in admin1_ids:
            continue
        current = winners.get(key)
        if current is None or (city.population, -city.city_id) > (
            current.population,
            -current.city_id,
        ):
            winners[key] = city

    rows: list[dict[str, Any]] = []
    for position, country in enumerate(countries):
        rows.append(
            {
                "level": "country",
                "place_id": country["iso"],
                "source_iso": country["iso"],
                "country_iso": _canonical_iso(country["iso"]),
                "parent_id": None,
                "admin1_id": None,
                "code": country["iso3"],
                "name": country["name"],
                "ascii_name": None,
                "name_zh": country["name_zh"],
                "aliases": None,
                "payload": _payload_text(country["payload"], "countries", position),
                "population": None,
                "geoname_id": None,
                "lat": country["lat"],
                "lng": country["lng"],
                "coordinate_source": CATALOG_COORDINATE_SOURCE,
                "coordinate_city_id": None,
            }
        )

    for position, (admin1, payload) in enumerate(admin1_rows):
        winner = winners.get(admin1.admin1_id)
        rows.append(
            {
                "level": "admin1",
                "place_id": admin1.admin1_id,
                "source_iso": admin1.iso,
                "country_iso": _canonical_iso(admin1.iso),
                "parent_id": admin1.iso,
                "admin1_id": admin1.admin1_id,
                "code": admin1.code,
                "name": admin1.name,
                "ascii_name": admin1.ascii_name,
                "name_zh": admin1.name_zh,
                "aliases": (
                    json.dumps(list(admin1.aliases), ensure_ascii=False)
                    if admin1.aliases
                    else None
                ),
                "payload": _payload_text(payload, "admin1", position),
                "population": None,
                "geoname_id": admin1.geoname_id,
                "lat": winner.lat if winner is not None else None,
                "lng": winner.lng if winner is not None else None,
                "coordinate_source": (
                    ADMIN1_COORDINATE_SOURCE
                    if winner is not None
                    else UNAVAILABLE_COORDINATE_SOURCE
                ),
                "coordinate_city_id": winner.city_id if winner is not None else None,
            }
        )

    for position, (city, ascii_name, payload) in enumerate(city_rows):
        matched = f"{city.iso}.{city.region}" if city.region else None
        if matched is not None and matched not in admin1_ids:
            matched = None
        rows.append(
            {
                "level": "city",
                "place_id": str(city.city_id),
                "source_iso": city.iso,
                "country_iso": _canonical_iso(city.iso),
                "parent_id": matched or city.iso,
                "admin1_id": matched,
                "code": city.region or None,
                "name": city.name,
                "ascii_name": ascii_name,
                "name_zh": None,
                "aliases": None,
                "payload": _payload_text(payload, "cities", position),
                "population": city.population,
                "geoname_id": city.city_id,
                "lat": city.lat,
                "lng": city.lng,
                "coordinate_source": CATALOG_COORDINATE_SOURCE,
                "coordinate_city_id": None,
            }
        )

    with_coordinates = sum(
        1 for row in rows if row["level"] == "admin1" and row["lat"] is not None
    )
    missing = sum(
        1 for row in rows if row["level"] == "admin1" and row["lat"] is None
    )
    return rows, with_coordinates, missing


def build_snapshot(
    countries_path: Path | str,
    admin1_path: Path | str,
    cities_path: Path | str,
) -> SeedSnapshot:
    """Read, validate and derive the full catalog without touching any database."""

    paths = {
        "countries": Path(countries_path),
        "admin1": Path(admin1_path),
        "cities": Path(cities_path),
    }
    raw = {name: _read_source(path, name) for name, path in paths.items()}
    countries = _load_countries(
        _decode_source(raw["countries"], paths["countries"], "countries")
    )
    admin1_rows = _load_admin1(
        _decode_source(raw["admin1"], paths["admin1"], "admin1")
    )
    city_rows = _load_cities(
        _decode_source(raw["cities"], paths["cities"], "cities")
    )

    rows, with_coordinates, missing = _build_rows(countries, admin1_rows, city_rows)
    counts = {
        "countries": len(countries),
        "admin1": len(admin1_rows),
        "cities": len(city_rows),
    }
    sources = tuple(
        SourceRecord(
            name=name,
            fingerprint=hashlib.sha256(raw[name]).hexdigest(),
            record_count=counts[name],
        )
        for name in SOURCE_NAMES
    )
    return SeedSnapshot(
        sources=sources,
        rows=tuple(rows),
        counts=counts,
        admin1_with_coordinates=with_coordinates,
        admin1_missing_coordinates=missing,
    )


# ---------------------------------------------------------------------------
# frozen seed schema (same DDL as the shipped data/geo_catalog.sqlite3 asset)
# ---------------------------------------------------------------------------
_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE geo_catalog_source (
        source VARCHAR(20) NOT NULL,
        path VARCHAR(500) NOT NULL,
        fingerprint VARCHAR(64) NOT NULL,
        record_count INTEGER NOT NULL,
        import_version VARCHAR(50) NOT NULL,
        imported_at DATETIME NOT NULL,
        PRIMARY KEY (source)
    )
    """,
    """
    CREATE TABLE geo_place (
        id INTEGER NOT NULL,
        level VARCHAR(10) NOT NULL,
        place_id VARCHAR(64) NOT NULL,
        source_iso VARCHAR(2) NOT NULL,
        country_iso VARCHAR(2) NOT NULL,
        parent_id VARCHAR(64),
        admin1_id VARCHAR(64),
        code VARCHAR(16),
        name VARCHAR(255) NOT NULL,
        ascii_name VARCHAR(255),
        name_zh VARCHAR(255),
        aliases TEXT,
        payload TEXT NOT NULL,
        population INTEGER,
        geoname_id INTEGER,
        lat FLOAT,
        lng FLOAT,
        coordinate_source VARCHAR(40) NOT NULL,
        coordinate_city_id INTEGER,
        PRIMARY KEY (id),
        CONSTRAINT uq_geo_place_identity UNIQUE (level, place_id),
        CONSTRAINT ck_geo_place_level CHECK (level IN ('country', 'admin1', 'city')),
        CONSTRAINT ck_geo_place_coordinate_pair CHECK (
            (lat IS NULL AND lng IS NULL) OR (lat IS NOT NULL AND lng IS NOT NULL)
        ),
        CONSTRAINT ck_geo_place_lat_range CHECK (
            lat IS NULL OR (lat >= -90.0 AND lat <= 90.0)
        ),
        CONSTRAINT ck_geo_place_lng_range CHECK (
            lng IS NULL OR (lng >= -180.0 AND lng <= 180.0)
        ),
        CONSTRAINT ck_geo_place_required_coordinates CHECK (
            level = 'admin1' OR (lat IS NOT NULL AND lng IS NOT NULL)
        ),
        CONSTRAINT ck_geo_place_population CHECK (
            population IS NULL OR population >= 0
        ),
        CONSTRAINT ck_geo_place_coordinate_source CHECK (
            coordinate_source IN ('admin1-largest-city-v1', 'catalog', 'unavailable')
        ),
        CONSTRAINT ck_geo_place_coordinate_source_consistency CHECK (
            (coordinate_source = 'unavailable' AND lat IS NULL AND lng IS NULL)
            OR (coordinate_source IN ('admin1-largest-city-v1', 'catalog')
                AND lat IS NOT NULL AND lng IS NOT NULL)
        ),
        CONSTRAINT ck_geo_place_coordinate_provenance CHECK (
            (coordinate_source = 'admin1-largest-city-v1' AND coordinate_city_id IS NOT NULL)
            OR (coordinate_source <> 'admin1-largest-city-v1' AND coordinate_city_id IS NULL)
        ),
        CONSTRAINT ck_geo_place_no_zero_zero CHECK (
            lat IS NULL OR lng IS NULL OR lat <> 0.0 OR lng <> 0.0
        )
    )
    """,
    "CREATE INDEX ix_geo_place_level_country ON geo_place (level, country_iso)",
    "CREATE INDEX ix_geo_place_level_source ON geo_place (level, source_iso)",
    "CREATE INDEX ix_geo_place_level_parent ON geo_place (level, parent_id)",
    "CREATE INDEX ix_geo_place_level_admin1 ON geo_place (level, admin1_id)",
    "CREATE INDEX ix_geo_place_level_population ON geo_place (level, population)",
    "CREATE INDEX ix_geo_place_level_name ON geo_place (level, name)",
)

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


def _insert_sql(table: str, columns: Sequence[str]) -> str:
    return (
        f"INSERT INTO {table} ({', '.join(columns)}) "
        f"VALUES ({', '.join('?' for _ in columns)})"
    )


def _stage_database(snapshot: SeedSnapshot, staged: Path, imported_at: str) -> None:
    """Create the two seed tables in ``staged`` and insert the snapshot rows."""

    connection = sqlite3.connect(staged)
    try:
        connection.isolation_level = None
        connection.execute("BEGIN IMMEDIATE")
        for statement in _SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.executemany(
            _insert_sql("geo_place", _GEO_PLACE_COLUMNS),
            [
                tuple(row[column] for column in _GEO_PLACE_COLUMNS)
                for row in snapshot.rows
            ],
        )
        connection.executemany(
            _insert_sql("geo_catalog_source", _SOURCE_COLUMNS),
            [
                (
                    source.name,
                    LEGACY_PROVENANCE.format(name=source.name),
                    source.fingerprint,
                    source.record_count,
                    IMPORT_VERSION,
                    imported_at,
                )
                for source in snapshot.sources
            ],
        )
        connection.execute("COMMIT")
    except BaseException:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        connection.close()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _read_only_uri(path: Path) -> str:
    return "file:" + quote(path.as_posix(), safe="/:") + "?mode=ro"


def _looks_like_seed(path: Path) -> bool:
    """True when ``path`` is an existing SQLite file with both seed tables."""

    if not path.is_file():
        return False
    try:
        connection = sqlite3.connect(_read_only_uri(path), uri=True, timeout=5.0)
    except sqlite3.Error:
        return False
    try:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        tables = {str(row[0]) for row in rows}
    except sqlite3.Error:
        return False
    finally:
        connection.close()
    return {"geo_place", "geo_catalog_source"} <= tables


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/import-legacy-geo-catalog.py",
        description=(
            "One-time import of the legacy JSON geo catalogs into the portable "
            "SQLite seed read by catalog_db.CatalogDatabase. No downloads."
        ),
    )
    parser.add_argument(
        "--countries-json",
        required=True,
        help="path of the legacy countries JSON catalog (no default)",
    )
    parser.add_argument(
        "--admin1-json",
        required=True,
        help="path of the legacy admin1 JSON catalog (no default)",
    )
    parser.add_argument(
        "--cities-json",
        required=True,
        help="path of the legacy cities JSON catalog (no default)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="target SQLite seed path (no default), e.g. data/geo_catalog.sqlite3",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "replace an existing --output file that is not a recognized geo "
            "catalog seed (refused by default to protect unknown user data)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    countries = Path(args.countries_json).expanduser()
    admin1 = Path(args.admin1_json).expanduser()
    cities = Path(args.cities_json).expanduser()
    output = Path(args.output).expanduser()

    inputs = {
        countries.resolve(): "--countries-json",
        admin1.resolve(): "--admin1-json",
        cities.resolve(): "--cities-json",
    }
    if output.resolve() in inputs:
        print(
            f"refusing to write --output over an input catalog: {output} "
            f"({inputs[output.resolve()]})",
            file=sys.stderr,
        )
        return 2
    if output.exists() and not output.is_file():
        print(f"refusing to replace a non-file --output: {output}", file=sys.stderr)
        return 2
    if output.exists() and not _looks_like_seed(output) and not args.force:
        print(
            f"refusing to replace unknown data at --output: {output}\n"
            "pass --force to replace it with a freshly generated catalog seed",
            file=sys.stderr,
        )
        return 2

    try:
        snapshot = build_snapshot(countries, admin1, cities)
    except LegacyCatalogError as exc:
        print(f"legacy geo catalog import failed: {exc}", file=sys.stderr)
        return 2

    output.parent.mkdir(parents=True, exist_ok=True)
    handle, staged_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
    )
    os.close(handle)
    staged = Path(staged_name)
    imported_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
    try:
        _stage_database(snapshot, staged, imported_at)
        # Read the staged bytes back through the shipped reader before the
        # atomic replacement: a staging bug can never clobber a good asset.
        seed = CatalogDatabase(staged)
        counts = seed.counts()
        sources = seed.sources()
        os.replace(staged, output)
    except (
        LegacyCatalogError,
        CatalogDatabaseError,
        OSError,
        sqlite3.Error,
    ) as exc:
        print(f"legacy geo catalog import failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if staged.exists():
            try:
                staged.unlink()
            except OSError:
                pass

    print(f"geo catalog seed written: {output}")
    print(
        "counts: " + " ".join(f"{name}={counts[name]}" for name in SOURCE_NAMES)
    )
    print(
        "admin1 coordinates: "
        f"with={snapshot.admin1_with_coordinates} "
        f"missing={snapshot.admin1_missing_coordinates}"
    )
    print(
        "import_version: "
        + ", ".join(sorted({str(row["import_version"]) for row in sources}))
    )
    for row in sources:
        print(
            f"source {row['source']}: path={row['path']} "
            f"fingerprint={row['fingerprint']} records={row['record_count']}"
        )
    print(f"sqlite artifact sha256: {_sha256_file(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())