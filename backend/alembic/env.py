"""Alembic environment.

The database URL comes from :func:`backend.app.config.get_settings`, so
CLI migrations target exactly the same SQLite file as the running application
(``M_PHOTO_DATA_DIR`` / ``M_PHOTO_DB_PATH`` / ``M_PHOTO_DB_URL``).
"""
from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy.engine import Connection

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import backend.app.database  # noqa: E402,F401  (registers the ORM models)
from backend.app.config import get_settings  # noqa: E402
from backend.app.database import Base, create_db_engine  # noqa: E402

config = context.config

if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    return get_settings().database_url


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_on_connection(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # Programmatic startup uses the runtime engine, not environment defaults.
    connection = config.attributes.get("connection")
    if connection is not None:
        _run_on_connection(connection)
        return

    # The CLI uses the same engine settings and SQLite PRAGMAs as the app.
    connectable = create_db_engine(get_settings())
    try:
        with connectable.begin() as connection:
            if connection.dialect.name == "sqlite":
                connection.exec_driver_sql("BEGIN IMMEDIATE")
            _run_on_connection(connection)
    finally:
        connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
