"""Run Alembic and safely adopt the pre-Alembic Geo SQLite schema.

Only an exact match for revision 0001_initial can be adopted. The reference
schema is generated from that immutable migration, never from the current ORM
(which may already describe a future revision). No application data is read.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Connection, Engine

BASELINE_REVISION = "0001_initial"
ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"


class SchemaCompatibilityError(RuntimeError):
    """Refuse to stamp an unknown schema; an operator must inspect it first."""


def _config(connection: Connection) -> Config:
    config = Config(str(ALEMBIC_INI))
    config.attributes["connection"] = connection
    # Application startup must not reset the application's logging handlers.
    config.attributes["configure_logger"] = False
    return config


def _normalise_default(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    while value.startswith("(") and value.endswith(")"):
        value = value[1:-1].strip()
    return value


def _schema(connection: Connection) -> dict[str, Any]:
    inspector = inspect(connection)
    result = {}
    for table in sorted(set(inspector.get_table_names()) - {"alembic_version"}):
        result[table] = {
            "columns": {
                column["name"]: {
                    "type": str(column["type"]).upper(),
                    "nullable": column["nullable"],
                    "default": _normalise_default(column["default"]),
                }
                for column in inspector.get_columns(table)
            },
            "primary_key": tuple(inspector.get_pk_constraint(table)["constrained_columns"]),
            "foreign_keys": sorted(
                (
                    tuple(fk["constrained_columns"]),
                    fk.get("referred_schema"),
                    fk["referred_table"],
                    tuple(fk["referred_columns"]),
                    tuple(sorted(fk.get("options", {}).items())),
                )
                for fk in inspector.get_foreign_keys(table)
            ),
            "indexes": sorted(
                (index["name"], bool(index["unique"]), tuple(index["column_names"]))
                for index in inspector.get_indexes(table)
            ),
            "unique_constraints": sorted(
                tuple(item["column_names"]) for item in inspector.get_unique_constraints(table)
            ),
            "check_constraints": sorted(
                item["sqltext"] for item in inspector.get_check_constraints(table)
            ),
            # Reflection omits expression indexes and some SQLite table options.
            # Retain their metadata so unknown variants cannot be silently stamped.
            "index_definitions": list(connection.execute(
                text("SELECT name, sql FROM sqlite_master WHERE type = 'index' "
                     "AND tbl_name = :table ORDER BY name"), {"table": table}
            ).tuples()),
            "table_options": tuple(connection.execute(
                text("SELECT wr, strict FROM pragma_table_list "
                     "WHERE schema = 'main' AND name = :table"), {"table": table}
            ).one()),
        }
    return result


@lru_cache(maxsize=1)
def _baseline_schema() -> dict[str, Any]:
    # Isolated, disposable reference: never touch the user's database or files.
    reference = create_engine("sqlite://")
    try:
        with reference.begin() as connection:
            command.upgrade(_config(connection), BASELINE_REVISION)
            return _schema(connection)
    finally:
        reference.dispose()


def _validate_unversioned_schema(connection: Connection) -> None:
    if connection.dialect.name != "sqlite":
        raise SchemaCompatibilityError(
            "Automatic adoption of an unversioned database is only supported for SQLite. "
            "Back up and inspect the schema before registering its Alembic revision."
        )
    actual = _schema(connection)
    expected = _baseline_schema()
    problems = []
    views = inspect(connection).get_view_names()
    triggers = connection.exec_driver_sql(
        "SELECT name FROM sqlite_master WHERE type = 'trigger' ORDER BY name"
    ).scalars().all()
    if views:
        problems.append(f"unexpected views: {sorted(views)}")
    if triggers:
        problems.append(f"unexpected triggers: {triggers}")
    if actual.keys() != expected.keys():
        problems.append(f"tables: expected {sorted(expected)}, found {sorted(actual)}")
    for table in sorted(actual.keys() & expected.keys()):
        observed, baseline = actual[table], expected[table]
        if observed["columns"].keys() != baseline["columns"].keys():
            problems.append(f"{table}: column names differ from {BASELINE_REVISION}")
        for name in sorted(observed["columns"].keys() & baseline["columns"].keys()):
            column, wanted = observed["columns"][name], baseline["columns"][name]
            for attribute in ("type", "nullable"):
                if column[attribute] != wanted[attribute]:
                    problems.append(f"{table}.{name}: incompatible {attribute}")
            # create_all() used client-side defaults; the Alembic baseline uses
            # server defaults. Both are known valid forms of this exact schema.
            if column["default"] not in (None, wanted["default"]):
                problems.append(f"{table}.{name}: incompatible server default")
        for attribute in (
            "primary_key", "foreign_keys", "indexes", "unique_constraints",
            "check_constraints", "index_definitions", "table_options",
        ):
            if observed[attribute] != baseline[attribute]:
                problems.append(f"{table}: incompatible {attribute}")
    if problems:
        raise SchemaCompatibilityError(
            "Database schema does not match the Geo Alembic baseline; refusing to stamp or alter it. "
            "Back up the database and resolve the mismatch manually. " + "; ".join(problems)
        )


def migrate_database(engine: Engine) -> None:
    """Upgrade an empty/managed DB, or validate and adopt the old Geo schema.

    A supplied engine keeps migrations on the same database as runtime Settings,
    including test-injected settings, without consulting process environment.
    SQLite acquires an immediate transaction before inspecting the schema, so
    adoption/upgrades are serialized and DDL rolls back on migration failure.
    """
    with engine.begin() as connection:
        if connection.dialect.name == "sqlite":
            # Python's legacy SQLite transaction mode does not BEGIN for DDL.
            # Explicit BEGIN makes migration DDL and revision updates atomic.
            connection.exec_driver_sql("BEGIN IMMEDIATE")
        config = _config(connection)
        heads = MigrationContext.configure(connection).get_current_heads()
        tables = set(inspect(connection).get_table_names()) - {"alembic_version"}
        if not heads and (tables or inspect(connection).get_view_names()):
            _validate_unversioned_schema(connection)
            command.stamp(config, BASELINE_REVISION)
        command.upgrade(config, "head")


if __name__ == "__main__":
    from .config import get_settings
    from .database import init_db

    db_engine, _ = init_db(get_settings())
    db_engine.dispose()
