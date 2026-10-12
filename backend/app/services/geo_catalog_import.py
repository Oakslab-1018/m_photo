"""SQLite-only bootstrap and verification of the runtime geo catalog.

The retired JSON catalogs (``data/countries.json``, ``data/admin1.json``,
``data/cities.json``) are never read by this module. The immutable portable
seed ``data/geo_catalog.sqlite3`` is read through the standard-library
:class:`catalog_db.CatalogDatabase`; it carries the complete ``geo_place`` rows
and the logical raw-JSON SHA-256 fingerprints in ``geo_catalog_source``.
Fingerprints are copied verbatim from the seed: this module never derives a
fingerprint from ``data/*sources.json`` metadata and never hashes the SQLite
file itself.

Startup (``ensure_geo_catalog``)
--------------------------------
* A runtime catalog whose tables, source metadata and per-level counts are
  internally consistent is adopted as-is. Only runtime metadata/counts are
  read; the seed file is never opened, so an already-populated 0004 catalog is
  not rebuilt or mutated unnecessarily.
* Otherwise (empty, partial or invalid) both ``geo_place`` and
  ``geo_catalog_source`` are replaced transactionally from the seed. On SQLite
  the write transaction starts with ``BEGIN IMMEDIATE`` so concurrent startups
  cannot interleave two replacements.

CLI
---
::

    python -m backend.app.services.geo_catalog_import           # bootstrap if needed
    python -m backend.app.services.geo_catalog_import --check   # compare DB/seed metadata only
    python -m backend.app.services.geo_catalog_import --force   # copy the seed over the runtime DB

``--check`` compares stored metadata (source, fingerprint, record count,
import version) and per-level counts; it never reads JSON paths.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from sqlalchemy import delete, func, insert, inspect, select
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from catalog_db import (
    SOURCE_LEVEL,
    SOURCE_NAMES,
    CatalogDatabase,
    CatalogDatabaseError,
)

from ..config import Settings, get_settings
from ..database import init_db
from .geo_catalog_models import GeoCatalogSource, GeoPlace

logger = logging.getLogger(__name__)

_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
_PLACE_COLUMNS = frozenset(GeoPlace.__table__.c.keys())
_SOURCE_COLUMNS = frozenset(GeoCatalogSource.__table__.c.keys())


class GeoCatalogImportError(ValueError):
    """Raised when the seed or the runtime catalog cannot be used."""


@dataclass(frozen=True)
class ImportResult:
    """Outcome of one bootstrap/check run."""

    imported: bool
    reasons: tuple[str, ...]
    counts: dict[str, int]
    admin1_with_coordinates: int
    admin1_missing_coordinates: int


@dataclass(frozen=True)
class _RuntimeState:
    initialized: bool
    reasons: tuple[str, ...]
    counts: dict[str, int]
    sources: dict[str, dict[str, Any]]
    admin1_with_coordinates: int = 0
    admin1_missing_coordinates: int = 0


def _empty_counts() -> dict[str, int]:
    return {"countries": 0, "admin1": 0, "cities": 0}


# --------------------------------------------------------------------------
# runtime validation (metadata/counts only; no seed access)
# --------------------------------------------------------------------------


def _read_runtime_state(connection: Connection) -> _RuntimeState:
    """Validate only the runtime DB's schema, source metadata and counts."""

    counts = _empty_counts()
    try:
        inspector = inspect(connection)
        tables = set(inspector.get_table_names())
        if "geo_place" not in tables or "geo_catalog_source" not in tables:
            return _RuntimeState(False, ("runtime:tables-missing",), counts, {})
        place_columns = {column["name"] for column in inspector.get_columns("geo_place")}
        source_columns = {
            column["name"] for column in inspector.get_columns("geo_catalog_source")
        }
    except SQLAlchemyError:
        return _RuntimeState(False, ("runtime:schema-unreadable",), counts, {})

    missing_place = sorted(_PLACE_COLUMNS - place_columns)
    missing_source = sorted(_SOURCE_COLUMNS - source_columns)
    if missing_place or missing_source:
        return _RuntimeState(False, ("runtime:columns-missing",), counts, {})

    place_table = GeoPlace.__table__
    source_table = GeoCatalogSource.__table__
    level_counts: dict[str, int] = {}
    try:
        for level, total in connection.execute(
            select(place_table.c.level, func.count()).group_by(place_table.c.level)
        ):
            level_counts[str(level)] = int(total)
        source_rows = {
            str(row["source"]): dict(row)
            for row in connection.execute(select(source_table)).mappings()
        }
    except SQLAlchemyError:
        return _RuntimeState(False, ("runtime:unreadable",), counts, {})

    reasons: list[str] = []
    unknown_levels = sorted(set(level_counts) - set(SOURCE_LEVEL.values()))
    if unknown_levels:
        reasons.append("runtime:unknown-levels:" + ",".join(unknown_levels))
    for name in SOURCE_NAMES:
        counts[name] = level_counts.get(SOURCE_LEVEL[name], 0)
    if sum(counts.values()) == 0:
        reasons.append("runtime:empty")

    missing_sources = [name for name in SOURCE_NAMES if name not in source_rows]
    if missing_sources:
        reasons.append("runtime:missing-sources:" + ",".join(missing_sources))
    extra_sources = sorted(set(source_rows) - set(SOURCE_NAMES))
    if extra_sources:
        reasons.append("runtime:extra-sources:" + ",".join(extra_sources))

    for name in SOURCE_NAMES:
        row = source_rows.get(name)
        if row is None:
            continue
        fingerprint = row.get("fingerprint")
        if not isinstance(fingerprint, str) or _FINGERPRINT_RE.fullmatch(fingerprint) is None:
            reasons.append(f"{name}:fingerprint-invalid")
        record_count = row.get("record_count")
        if (
            isinstance(record_count, bool)
            or not isinstance(record_count, int)
            or record_count <= 0
        ):
            reasons.append(f"{name}:count-invalid")
        elif record_count != counts[name]:
            reasons.append(f"{name}:count-mismatch")
        import_version = row.get("import_version")
        if not isinstance(import_version, str) or not import_version.strip():
            reasons.append(f"{name}:import-version-invalid")
        stored_path = row.get("path")
        if not isinstance(stored_path, str) or not stored_path.strip():
            reasons.append(f"{name}:path-invalid")

    if reasons:
        return _RuntimeState(False, tuple(reasons), counts, source_rows)

    try:
        missing_coordinates = int(
            connection.execute(
                select(func.count())
                .select_from(place_table)
                .where(place_table.c.level == "admin1", place_table.c.lat.is_(None))
            ).scalar_one()
        )
    except SQLAlchemyError:
        return _RuntimeState(False, ("runtime:unreadable",), counts, source_rows)
    with_coordinates = counts.get("admin1", 0) - missing_coordinates
    return _RuntimeState(
        True,
        (),
        counts,
        source_rows,
        with_coordinates,
        missing_coordinates,
    )


def _state_result(state: _RuntimeState, *, reasons: Sequence[str] | None = None) -> ImportResult:
    return ImportResult(
        imported=False,
        reasons=tuple(reasons) if reasons is not None else (
            ("unchanged",) if state.initialized else state.reasons
        ),
        counts=dict(state.counts),
        admin1_with_coordinates=state.admin1_with_coordinates,
        admin1_missing_coordinates=state.admin1_missing_coordinates,
    )


# --------------------------------------------------------------------------
# seed reading (SQLite only)
# --------------------------------------------------------------------------


def _open_seed(seed_path: str | Path | None) -> CatalogDatabase:
    try:
        return CatalogDatabase(seed_path)
    except CatalogDatabaseError as exc:
        raise GeoCatalogImportError(f"geo catalog seed is unusable: {exc}") from exc


def _source_insert_row(row: Mapping[str, Any]) -> dict[str, Any]:
    imported_at = row.get("imported_at")
    if isinstance(imported_at, datetime):
        parsed = imported_at
    elif isinstance(imported_at, str):
        try:
            parsed = datetime.fromisoformat(imported_at)
        except ValueError as exc:
            raise GeoCatalogImportError(
                "geo catalog seed has an invalid imported_at timestamp"
            ) from exc
    else:
        raise GeoCatalogImportError("geo catalog seed has a missing imported_at timestamp")
    return {**row, "imported_at": parsed}


def _seed_payload(
    seed: CatalogDatabase,
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]], int, int]:
    """Read + validate the seed's metadata and full storage rows (SQLite only)."""

    try:
        stored_sources = {str(row["source"]): dict(row) for row in seed.sources()}
        counts = seed.counts()
        rows = [
            *seed.places("country"),
            *seed.places("admin1"),
            *seed.places("city"),
        ]
    except CatalogDatabaseError as exc:
        raise GeoCatalogImportError(f"geo catalog seed is unusable: {exc}") from exc

    missing_sources = [name for name in SOURCE_NAMES if name not in stored_sources]
    if missing_sources:
        raise GeoCatalogImportError(
            "geo catalog seed is missing sources: " + ", ".join(missing_sources)
        )
    extra_sources = sorted(set(stored_sources) - set(SOURCE_NAMES))
    if extra_sources:
        raise GeoCatalogImportError(
            "geo catalog seed has unexpected sources: " + ", ".join(extra_sources)
        )

    ordered_sources: list[dict[str, Any]] = []
    for name in SOURCE_NAMES:
        row = stored_sources[name]
        if set(row) != _SOURCE_COLUMNS:
            raise GeoCatalogImportError(
                f"geo catalog seed source row {name!r} does not match the runtime schema"
            )
        record_count = row.get("record_count")
        if (
            isinstance(record_count, bool)
            or not isinstance(record_count, int)
            or record_count <= 0
            or record_count != counts.get(name, 0)
        ):
            raise GeoCatalogImportError(
                f"geo catalog seed source row {name!r} has an inconsistent record count"
            )
        ordered_sources.append(_source_insert_row(row))

    if not rows:
        raise GeoCatalogImportError("geo catalog seed contains no place rows")
    for row in rows:
        if set(row) != _PLACE_COLUMNS:
            raise GeoCatalogImportError(
                "geo catalog seed place row does not match the runtime schema"
            )
    rows.sort(key=lambda row: int(row["id"]))

    with_coordinates = sum(
        1 for row in rows if row["level"] == "admin1" and row["lat"] is not None
    )
    missing_coordinates = sum(
        1 for row in rows if row["level"] == "admin1" and row["lat"] is None
    )
    return ordered_sources, dict(counts), rows, with_coordinates, missing_coordinates


# --------------------------------------------------------------------------
# public operations
# --------------------------------------------------------------------------


def ensure_geo_catalog(
    engine: Engine,
    *,
    seed_path: str | Path | None = None,
) -> ImportResult:
    """Adopt a valid runtime catalog or bootstrap it from the SQLite seed.

    The pre-check reads runtime metadata/counts only: when the catalog is
    already fully initialized the seed is never opened.
    """

    with engine.connect() as connection:
        state = _read_runtime_state(connection)
    if state.initialized:
        logger.debug("geo catalog runtime database is already initialized")
        return _state_result(state)
    return bootstrap_geo_catalog(engine, seed_path=seed_path, force=False)


def bootstrap_geo_catalog(
    engine: Engine,
    *,
    seed_path: str | Path | None = None,
    force: bool = False,
) -> ImportResult:
    """Transactionally replace ``geo_place`` + ``geo_catalog_source`` from seed.

    The fingerprint gate is re-checked inside the ``BEGIN IMMEDIATE`` write
    transaction, so a concurrent startup that finished first is adopted rather
    than overwritten. ``force=True`` always copies the seed.
    """

    if not force:
        with engine.connect() as connection:
            state = _read_runtime_state(connection)
        if state.initialized:
            return _state_result(state)

    seed = _open_seed(seed_path)
    source_rows, counts, rows, with_coordinates, missing_coordinates = _seed_payload(seed)

    try:
        with engine.begin() as connection:
            if connection.dialect.name == "sqlite":
                # Python's legacy SQLite mode does not BEGIN for DML-only
                # transactions; take the write lock before re-checking.
                connection.exec_driver_sql("BEGIN IMMEDIATE")
            if not force:
                state = _read_runtime_state(connection)
                if state.initialized:
                    return _state_result(state)
            connection.execute(delete(GeoPlace.__table__))
            connection.execute(delete(GeoCatalogSource.__table__))
            connection.execute(insert(GeoPlace.__table__), rows)
            connection.execute(insert(GeoCatalogSource.__table__), source_rows)
    except SQLAlchemyError as exc:
        raise GeoCatalogImportError(f"geo catalog bootstrap failed: {exc}") from exc

    logger.info(
        "geo catalog copied from SQLite seed (%s; admin1 with coordinates=%d, missing=%d)",
        ", ".join(f"{name}={count}" for name, count in sorted(counts.items())),
        with_coordinates,
        missing_coordinates,
    )
    return ImportResult(
        imported=True,
        reasons=("forced",) if force else ("seed-copied",),
        counts=counts,
        admin1_with_coordinates=with_coordinates,
        admin1_missing_coordinates=missing_coordinates,
    )


def check_geo_catalog(
    engine: Engine,
    *,
    seed_path: str | Path | None = None,
) -> ImportResult:
    """Compare runtime and seed metadata/counts without writing or reading JSON."""

    seed = _open_seed(seed_path)
    try:
        seed_sources = {str(row["source"]): dict(row) for row in seed.sources()}
        seed_counts = seed.counts()
    except CatalogDatabaseError as exc:
        raise GeoCatalogImportError(f"geo catalog seed is unusable: {exc}") from exc

    with engine.connect() as connection:
        state = _read_runtime_state(connection)
    if not state.initialized:
        return ImportResult(
            imported=False,
            reasons=state.reasons or ("runtime:uninitialized",),
            counts=seed_counts,
            admin1_with_coordinates=0,
            admin1_missing_coordinates=0,
        )

    reasons: list[str] = []
    for name in SOURCE_NAMES:
        seed_row = seed_sources.get(name)
        runtime_row = state.sources.get(name)
        if seed_row is None:
            reasons.append(f"seed:{name}:missing")
            continue
        if runtime_row is None:
            reasons.append(f"{name}:missing")
            continue
        if str(runtime_row.get("fingerprint")) != str(seed_row.get("fingerprint")):
            reasons.append(f"{name}:fingerprint")
        if int(runtime_row.get("record_count", -1)) != int(seed_row.get("record_count", -1)):
            reasons.append(f"{name}:count")
        if str(runtime_row.get("import_version")) != str(seed_row.get("import_version")):
            reasons.append(f"{name}:import_version")
    if not reasons and state.counts != seed_counts:
        reasons.append("counts")

    return _state_result(state, reasons=("unchanged",) if not reasons else tuple(reasons))


def import_geo_catalog(
    engine: Engine,
    settings: Settings | None = None,
    *,
    force: bool = False,
) -> ImportResult:
    """Application/CLI entry point: bootstrap if needed, or force copy seed."""

    seed_path = settings.catalog_seed_path if settings is not None else None
    if force:
        return bootstrap_geo_catalog(engine, seed_path=seed_path, force=True)
    return ensure_geo_catalog(engine, seed_path=seed_path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.app.services.geo_catalog_import",
        description=(
            "Bootstrap/verify the runtime geo catalog from the shipped SQLite "
            "seed. No JSON catalogs are read and no downloads are performed."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare runtime and seed metadata/counts (never copies catalog rows)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="copy the seed over the runtime catalog even when it is initialized",
    )
    args = parser.parse_args(argv)
    if args.check and args.force:
        parser.error("--check and --force are mutually exclusive")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()
    engine, _ = init_db(settings)
    try:
        try:
            if args.check:
                result = check_geo_catalog(engine, seed_path=settings.catalog_seed_path)
            else:
                result = import_geo_catalog(engine, settings, force=args.force)
        except GeoCatalogImportError as exc:
            print(f"geo catalog import failed: {exc}", file=sys.stderr)
            return 2
    finally:
        engine.dispose()

    counts = ", ".join(f"{name}={count}" for name, count in sorted(result.counts.items()))
    if args.check:
        if result.reasons == ("unchanged",):
            print(f"geo catalog is up to date ({counts})")
            return 0
        print(f"geo catalog import required: {', '.join(result.reasons)} ({counts})")
        return 1
    state = "imported" if result.imported else "unchanged"
    print(
        f"geo catalog {state} ({counts}; "
        f"admin1 with coordinates={result.admin1_with_coordinates}, "
        f"missing={result.admin1_missing_coordinates})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
