"""Read-only standard-library access to the shipped geographic catalog seed.

``data/geo_catalog.sqlite3`` is generated once from the legacy JSON catalogs
by ``scripts/import-legacy-geo-catalog.py`` and shipped as a portable asset.
Runtime readers -- including the inference service, which has no SQLAlchemy --
import only this module and the standard library:

* every call opens a short-lived SQLite connection with the ``mode=ro`` URI and
  closes it before returning; nothing is cached between calls;
* the required schema, the ``geo_catalog_source`` metadata and the coordinates
  of every row that must carry them are validated before any read;
* logical dataset revisions are the pinned raw-JSON SHA-256 fingerprints stored
  in ``geo_catalog_source.fingerprint``; :meth:`CatalogDatabase.revision`
  returns that stored value and never hashes the database file, never re-reads
  the legacy JSON files and has no JSON fallback;
* failures are raised as :class:`CatalogDatabaseError` (a ``ValueError``).
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parent

#: Shipped portable seed asset (``<repo>/data/geo_catalog.sqlite3``).
DEFAULT_CATALOG_PATH = REPO_ROOT / "data" / "geo_catalog.sqlite3"

#: Environment override for deployments that stage the seed elsewhere.
CATALOG_PATH_ENV = "M_PHOTO_GEO_CATALOG_SEED"

#: Canonical ``geo_place.level`` values.
LEVELS: tuple[str, ...] = ("country", "admin1", "city")

#: Canonical ``geo_catalog_source.source`` names.
SOURCE_NAMES: tuple[str, ...] = ("countries", "admin1", "cities")

#: Source name -> stored place level.
SOURCE_LEVEL: dict[str, str] = {
    "countries": "country",
    "admin1": "admin1",
    "cities": "city",
}

#: Place level -> source name.
LEVEL_SOURCE: dict[str, str] = {
    "country": "countries",
    "admin1": "admin1",
    "city": "cities",
}

#: ``geo_place.coordinate_source`` values produced by the importer.
COORDINATE_SOURCE_DERIVED = "admin1-largest-city-v1"
COORDINATE_SOURCE_CATALOG = "catalog"
COORDINATE_SOURCE_UNAVAILABLE = "unavailable"
_COORDINATE_SOURCES = frozenset(
    {
        COORDINATE_SOURCE_DERIVED,
        COORDINATE_SOURCE_CATALOG,
        COORDINATE_SOURCE_UNAVAILABLE,
    }
)

#: Levels whose rows must always carry finite coordinates.
_LEVELS_REQUIRING_COORDINATES = frozenset({"country", "city"})

_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "geo_place": (
        "id",
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
    ),
    "geo_catalog_source": (
        "source",
        "path",
        "fingerprint",
        "record_count",
        "import_version",
        "imported_at",
    ),
}

_ISO_RE = re.compile(r"^[A-Z]{2}$")
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


class CatalogDatabaseError(ValueError):
    """Raised when the catalog seed is missing, malformed or inconsistent."""


def default_catalog_path() -> Path:
    """Return the seed path: ``M_PHOTO_GEO_CATALOG_SEED`` or the repo default."""

    override = os.environ.get(CATALOG_PATH_ENV, "")
    if override and override.strip():
        return Path(override.strip()).expanduser().resolve()
    return DEFAULT_CATALOG_PATH


def _read_only_uri(path: Path) -> str:
    """SQLite ``file:`` URI that opens ``path`` read-only on POSIX and Windows."""

    return "file:" + quote(path.as_posix(), safe="/:") + "?mode=ro"


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    return {str(row["name"]) for row in rows}


def _payload_object(raw: object, context: str) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw:
        raise CatalogDatabaseError(f"{context}: stored payload is missing")
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise CatalogDatabaseError(f"{context}: stored payload is not valid JSON") from exc
    if not isinstance(value, dict):
        raise CatalogDatabaseError(f"{context}: stored payload is not a JSON object")
    return value


def _decode_aliases(raw: object, context: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, str):
        raise CatalogDatabaseError(f"{context}: aliases column is not text")
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise CatalogDatabaseError(f"{context}: aliases column is not valid JSON") from exc
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise CatalogDatabaseError(
            f"{context}: aliases column must be a JSON array of strings"
        )
    return [item for item in value if item]


def _normalize_iso(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise CatalogDatabaseError(f"{field} must be a two-letter ISO code string")
    code = value.strip().upper()
    if _ISO_RE.fullmatch(code) is None:
        raise CatalogDatabaseError(f"{field} must be a two-letter ISO code, got {value!r}")
    return code


def _normalize_admin1_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogDatabaseError("admin1_id must be a non-empty string")
    return value.strip()


def _require_row_coordinates(level: str, place_id: str, row: sqlite3.Row) -> None:
    """Enforce the stored coordinate contract for one place row (when read)."""

    coordinate_source = row["coordinate_source"]
    if coordinate_source not in _COORDINATE_SOURCES:
        raise CatalogDatabaseError(
            f"{level} {place_id}: unknown coordinate_source {coordinate_source!r}"
        )

    lat = row["lat"]
    lng = row["lng"]
    provenance = row["coordinate_city_id"]

    if coordinate_source == COORDINATE_SOURCE_UNAVAILABLE:
        if level in _LEVELS_REQUIRING_COORDINATES:
            raise CatalogDatabaseError(
                f"{level} {place_id}: {level} rows require coordinates"
            )
        if lat is not None or lng is not None:
            raise CatalogDatabaseError(
                f"{level} {place_id}: unavailable coordinates must be NULL"
            )
        if provenance is not None:
            raise CatalogDatabaseError(
                f"{level} {place_id}: unavailable coordinates must not carry a provenance city"
            )
        return

    if lat is None or lng is None:
        raise CatalogDatabaseError(
            f"{level} {place_id}: coordinates are required for {coordinate_source!r}"
        )
    if (
        isinstance(lat, bool)
        or isinstance(lng, bool)
        or not isinstance(lat, (int, float))
        or not isinstance(lng, (int, float))
    ):
        raise CatalogDatabaseError(f"{level} {place_id}: coordinates must be numeric")
    lat_value = float(lat)
    lng_value = float(lng)
    if not math.isfinite(lat_value) or not math.isfinite(lng_value):
        raise CatalogDatabaseError(f"{level} {place_id}: coordinates must be finite")
    if not (-90.0 <= lat_value <= 90.0) or not (-180.0 <= lng_value <= 180.0):
        raise CatalogDatabaseError(f"{level} {place_id}: coordinates are out of range")

    if coordinate_source == COORDINATE_SOURCE_DERIVED:
        if provenance is None:
            raise CatalogDatabaseError(
                f"{level} {place_id}: derived coordinates are missing coordinate_city_id"
            )
    elif provenance is not None:
        raise CatalogDatabaseError(
            f"{level} {place_id}: {coordinate_source!r} coordinates must not carry "
            "coordinate_city_id"
        )


class CatalogDatabase:
    """Read-only handle over one shipped catalog seed database.

    Construction validates the stored schema and source metadata (a few small
    indexed queries; no payload parsing, no file hashing). Every accessor then
    opens its own ``mode=ro`` connection and closes it before returning.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = (
            default_catalog_path()
            if path is None
            else Path(path).expanduser().resolve()
        )
        self._validate()

    # ----- internal helpers -------------------------------------------------
    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        if not self.path.is_file():
            raise CatalogDatabaseError(f"geo catalog database not found: {self.path}")
        try:
            connection = sqlite3.connect(
                _read_only_uri(self.path), uri=True, timeout=5.0
            )
        except sqlite3.Error as exc:
            raise CatalogDatabaseError(
                f"cannot open geo catalog database {self.path}: {exc}"
            ) from exc
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            yield connection
        except sqlite3.Error as exc:
            raise CatalogDatabaseError(
                f"geo catalog database read failed ({self.path.name}): {exc}"
            ) from exc
        finally:
            connection.close()

    def _fetch(
        self, sql: str, parameters: Sequence[Any] = ()
    ) -> list[sqlite3.Row]:
        with self._connection() as connection:
            return list(connection.execute(sql, tuple(parameters)))

    def _fetch_one(
        self, sql: str, parameters: Sequence[Any] = ()
    ) -> sqlite3.Row | None:
        with self._connection() as connection:
            return connection.execute(sql, tuple(parameters)).fetchone()

    def _validate(self) -> None:
        with self._connection() as connection:
            counts = self._validate_schema(connection)
            self._validate_sources(connection, counts)

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> dict[str, int]:
        for table, required in _REQUIRED_COLUMNS.items():
            columns = _table_columns(connection, table)
            if not columns:
                raise CatalogDatabaseError(
                    f"geo catalog database is missing table: {table}"
                )
            missing = [name for name in required if name not in columns]
            if missing:
                raise CatalogDatabaseError(
                    f"geo catalog table {table} is missing columns: "
                    + ", ".join(missing)
                )

        counts: dict[str, int] = {}
        for row in connection.execute(
            "SELECT level, COUNT(*) AS total FROM geo_place GROUP BY level"
        ):
            level = str(row["level"])
            if level not in LEVELS:
                raise CatalogDatabaseError(
                    f"geo_place contains an unsupported level: {level!r}"
                )
            counts[level] = int(row["total"])
        return counts

    @staticmethod
    def _validate_sources(
        connection: sqlite3.Connection, counts: dict[str, int]
    ) -> None:
        stored = {
            str(row["source"]): row
            for row in connection.execute("SELECT * FROM geo_catalog_source")
        }
        missing = [name for name in SOURCE_NAMES if name not in stored]
        if missing:
            raise CatalogDatabaseError(
                "geo_catalog_source is missing required sources: " + ", ".join(missing)
            )
        for name in SOURCE_NAMES:
            row = stored[name]
            fingerprint = row["fingerprint"]
            if (
                not isinstance(fingerprint, str)
                or _FINGERPRINT_RE.fullmatch(fingerprint) is None
            ):
                raise CatalogDatabaseError(
                    f"geo_catalog_source[{name}].fingerprint must be a SHA-256 hex digest"
                )
            record_count = row["record_count"]
            if (
                isinstance(record_count, bool)
                or not isinstance(record_count, int)
                or record_count < 0
            ):
                raise CatalogDatabaseError(
                    f"geo_catalog_source[{name}].record_count must be a non-negative integer"
                )
            import_version = row["import_version"]
            if not isinstance(import_version, str) or not import_version.strip():
                raise CatalogDatabaseError(
                    f"geo_catalog_source[{name}].import_version must not be blank"
                )
            path = row["path"]
            if not isinstance(path, str) or not path.strip():
                raise CatalogDatabaseError(
                    f"geo_catalog_source[{name}].path must not be blank"
                )
            actual = int(counts.get(SOURCE_LEVEL[name], 0))
            if record_count != actual:
                raise CatalogDatabaseError(
                    f"geo_catalog_source[{name}].record_count={record_count} does not "
                    f"match stored geo_place rows={actual}"
                )

    # ----- public API -------------------------------------------------------
    def countries(self) -> list[dict[str, Any]]:
        """Return the original country payloads sorted by ISO code."""

        rows = self._fetch(
            "SELECT place_id, payload, lat, lng, coordinate_source, "
            "coordinate_city_id FROM geo_place WHERE level = 'country'"
        )
        payloads: list[tuple[str, dict[str, Any]]] = []
        for row in rows:
            place_id = str(row["place_id"])
            _require_row_coordinates("country", place_id, row)
            payloads.append(
                (place_id, _payload_object(row["payload"], f"country {place_id}"))
            )
        payloads.sort(key=lambda item: item[0])
        return [payload for _, payload in payloads]

    def country(self, iso: str) -> dict[str, Any] | None:
        """Return one original country payload, or ``None`` when unknown."""

        code = _normalize_iso(iso, "iso")
        row = self._fetch_one(
            "SELECT place_id, payload, lat, lng, coordinate_source, "
            "coordinate_city_id FROM geo_place "
            "WHERE level = 'country' AND place_id = ?",
            (code,),
        )
        if row is None:
            return None
        _require_row_coordinates("country", code, row)
        return _payload_object(row["payload"], f"country {code}")

    def cities(
        self,
        country_iso: str | None = None,
        admin1_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return original city payloads with corrected typed fields.

        ``country_iso`` is the RAW source scope stored in
        ``geo_place.source_iso``: ``CN`` returns mainland-China rows only and
        ``TW``/``HK``/``MO`` stay distinct. The existing CN grouping policy is
        applied by higher layers, never here. ``admin1_id`` matches the exact
        ``source_iso.code`` directory id stored in ``geo_place.admin1_id``
        (leading zeros significant). Rows keep the stable ``places()`` storage
        order, which is the original catalog order.
        """

        cities: list[dict[str, Any]] = []
        for row in self.places("city", source_iso=country_iso, admin1_id=admin1_id):
            place_id = str(row["place_id"])
            city_id = row["geoname_id"]
            if city_id is None:
                try:
                    city_id = int(place_id)
                except ValueError as exc:
                    raise CatalogDatabaseError(
                        f"city {place_id}: geoname_id is missing"
                    ) from exc
            city = dict(_payload_object(row["payload"], f"city {place_id}"))
            city["city_id"] = int(city_id)
            city["iso"] = str(row["source_iso"])
            city["name"] = str(row["name"])
            city["ascii_name"] = row["ascii_name"]
            city["region"] = row["code"]
            city["lat"] = float(row["lat"])
            city["lng"] = float(row["lng"])
            city["population"] = (
                int(row["population"]) if row["population"] is not None else 0
            )
            cities.append(city)
        return cities

    def admin1(self, source_iso: str | None = None) -> list[dict[str, Any]]:
        """Return admin1 rows with the same shape as the current admin1 JSON."""

        conditions = ["level = 'admin1'"]
        parameters: list[Any] = []
        if source_iso is not None:
            conditions.append("source_iso = ?")
            parameters.append(_normalize_iso(source_iso, "source_iso"))

        rows = self._fetch(
            "SELECT * FROM geo_place WHERE "
            + " AND ".join(conditions)
            + " ORDER BY place_id",
            parameters,
        )
        entries: list[dict[str, Any]] = []
        for row in rows:
            place_id = str(row["place_id"])
            _require_row_coordinates("admin1", place_id, row)
            code = row["code"]
            ascii_name = row["ascii_name"]
            geoname_id = row["geoname_id"]
            if not isinstance(code, str) or not code:
                raise CatalogDatabaseError(f"admin1 {place_id}: code is missing")
            if not isinstance(ascii_name, str) or not ascii_name:
                raise CatalogDatabaseError(f"admin1 {place_id}: ascii_name is missing")
            if isinstance(geoname_id, bool) or not isinstance(geoname_id, int):
                raise CatalogDatabaseError(f"admin1 {place_id}: geoname_id is missing")

            entry = dict(_payload_object(row["payload"], f"admin1 {place_id}"))
            entry["admin1_id"] = place_id
            entry["iso"] = str(row["source_iso"])
            entry["code"] = code
            entry["name"] = str(row["name"])
            entry["ascii_name"] = ascii_name
            entry["geoname_id"] = geoname_id
            if row["name_zh"] is None:
                entry.pop("name_zh", None)
            else:
                entry["name_zh"] = str(row["name_zh"])
            aliases = _decode_aliases(row["aliases"], f"admin1 {place_id}")
            if aliases:
                entry["aliases"] = aliases
            else:
                entry.pop("aliases", None)
            entries.append(entry)
        return entries

    def places(
        self,
        level: str,
        source_iso: str | None = None,
        admin1_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return full ``geo_place`` storage rows (SQL column names).

        ``aliases`` and ``payload`` stay JSON text exactly as stored. Rows come
        back in stable storage order.
        """

        if not isinstance(level, str) or level not in LEVELS:
            raise CatalogDatabaseError(
                f"level must be one of {LEVELS!r}, got {level!r}"
            )
        conditions = ["level = ?"]
        parameters: list[Any] = [level]
        if source_iso is not None:
            conditions.append("source_iso = ?")
            parameters.append(_normalize_iso(source_iso, "source_iso"))
        if admin1_id is not None:
            conditions.append("admin1_id = ?")
            parameters.append(_normalize_admin1_id(admin1_id))

        rows = self._fetch(
            "SELECT * FROM geo_place WHERE "
            + " AND ".join(conditions)
            + " ORDER BY id",
            parameters,
        )
        places: list[dict[str, Any]] = []
        for row in rows:
            place_level = str(row["level"])
            place_id = str(row["place_id"])
            _require_row_coordinates(place_level, place_id, row)
            places.append({key: row[key] for key in row.keys()})
        return places

    def revision(self, source: str) -> str:
        """Return the stored raw-JSON SHA-256 fingerprint for one source."""

        if source not in SOURCE_NAMES:
            raise CatalogDatabaseError(
                f"unknown catalog source {source!r}; expected one of {SOURCE_NAMES!r}"
            )
        row = self._fetch_one(
            "SELECT fingerprint FROM geo_catalog_source WHERE source = ?", (source,)
        )
        if row is None:
            raise CatalogDatabaseError(f"missing source metadata: {source}")
        return str(row["fingerprint"])

    def sources(self) -> list[dict[str, Any]]:
        """Return ``geo_catalog_source`` rows ordered countries/admin1/cities."""

        stored = {
            str(row["source"]): row
            for row in self._fetch("SELECT * FROM geo_catalog_source")
        }
        ordered: list[dict[str, Any]] = []
        for name in SOURCE_NAMES:
            row = stored.pop(name, None)
            if row is not None:
                ordered.append({key: row[key] for key in row.keys()})
        for name in sorted(stored):
            row = stored[name]
            ordered.append({key: row[key] for key in row.keys()})
        return ordered

    def counts(self) -> dict[str, int]:
        """Return ``{'countries': n, 'admin1': n, 'cities': n}`` row counts."""

        counts = {"countries": 0, "admin1": 0, "cities": 0}
        for row in self._fetch(
            "SELECT level, COUNT(*) AS total FROM geo_place GROUP BY level"
        ):
            level = str(row["level"])
            if level not in LEVEL_SOURCE:
                raise CatalogDatabaseError(
                    f"geo_place contains an unsupported level: {level!r}"
                )
            counts[LEVEL_SOURCE[level]] = int(row["total"])
        return counts


__all__ = [
    "CATALOG_PATH_ENV",
    "CatalogDatabase",
    "CatalogDatabaseError",
    "DEFAULT_CATALOG_PATH",
    "LEVELS",
    "SOURCE_NAMES",
    "default_catalog_path",
]