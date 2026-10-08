"""SQLAlchemy 2.x engine / session / Base and the geo data models.

Schema management
-----------------
The authoritative schema is the Alembic baseline migration
(``backend/alembic/versions/0001_initial.py`` — run
``uv run --frozen alembic -c backend/alembic.ini upgrade head``).

``init_db(settings)`` runs Alembic before returning a session factory. Existing
unversioned Geo databases are adopted only after validating their structure
against the initial migration. ``create_schema=False`` skips schema management
for low-level callers; application startup never uses that bypass.

The engine/session factory are created from :class:`Settings`, so importing
this module never creates a database file.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import Settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class GeoSession(Base):
    """One photo-analysis conversation.

    Column names intentionally mirror the legacy Flask ``geo_session`` model
    (``clues_json`` / ``exclusions_json`` / ``confirmed_iso`` / ``status``) so
    the existing JSON payload contract keeps working.
    """

    __tablename__ = "geo_session"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    image_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    clues_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    exclusions_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    model_evidence_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    confirmed_iso: Mapped[str | None] = mapped_column(String(10), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # ---- JSON helpers (same semantics as the legacy model) ----
    @property
    def clues(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.clues_json or "[]")
        except (TypeError, ValueError):
            return []

    @clues.setter
    def clues(self, value: list[dict[str, Any]]) -> None:
        self.clues_json = json.dumps(value, ensure_ascii=False)

    @property
    def exclusions(self) -> list[str]:
        try:
            return json.loads(self.exclusions_json or "[]")
        except (TypeError, ValueError):
            return []

    @exclusions.setter
    def exclusions(self, value: list[str]) -> None:
        self.exclusions_json = json.dumps(value, ensure_ascii=False)

    @property
    def model_evidence(self) -> dict[str, Any] | None:
        if not self.model_evidence_json:
            return None
        try:
            value = json.loads(self.model_evidence_json)
            return value if isinstance(value, dict) else None
        except (TypeError, ValueError):
            return None

    @model_evidence.setter
    def model_evidence(self, value: dict[str, Any] | None) -> None:
        self.model_evidence_json = (
            json.dumps(value, ensure_ascii=False, allow_nan=False) if value is not None else None
        )

    @property
    def media_url(self) -> str:
        return f"/media/{self.image_filename}"


class ChatMessage(Base):
    """A single chat turn (``role`` is ``user`` or ``ai``)."""

    __tablename__ = "chat_message"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("geo_session.id"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    meta_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    @property
    def meta(self) -> dict[str, Any]:
        try:
            return json.loads(self.meta_json or "{}")
        except (TypeError, ValueError):
            return {}

    @meta.setter
    def meta(self, value: dict[str, Any]) -> None:
        self.meta_json = json.dumps(value, ensure_ascii=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "meta": self.meta,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


def create_db_engine(settings: Settings) -> Engine:
    """Create a lazily-connecting SQLite engine for ``settings``."""
    if settings.database_url.startswith("sqlite"):
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)

    connect_args: dict[str, Any] = {}
    if settings.database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False  # sessions stay per-request

    engine = create_engine(
        settings.database_url,
        connect_args=connect_args,
        pool_pre_ping=True,
        future=True,
    )

    if settings.database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, _record):  # pragma: no cover - driver glue
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
            finally:
                cursor.close()

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db(
    settings: Settings, create_schema: bool = True
) -> tuple[Engine, sessionmaker[Session]]:
    """Prepare directories and run migrations before exposing the DB to requests."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    engine = create_db_engine(settings)
    try:
        if create_schema:
            from .migrations import migrate_database

            migrate_database(engine)
        return engine, create_session_factory(engine)
    except BaseException:
        engine.dispose()
        raise


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


__all__ = [
    "Base",
    "ChatMessage",
    "GeoSession",
    "create_db_engine",
    "create_session_factory",
    "ensure_parent_dir",
    "init_db",
]
