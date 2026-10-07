"""SQLAlchemy 2.x engine / session / Base and the geo data models.

Schema management
-----------------
The authoritative schema is the Alembic baseline migration
(``backend/alembic/versions/0001_initial.py`` — run
``uv run --frozen alembic -c backend/alembic.ini upgrade head``).

``init_db(settings)`` still calls ``Base.metadata.create_all`` as a
**temporary convenience** for zero-setup local runs and tests (``create_schema=False``
disables it). It is NOT a migration substitute and does not get applied to an
existing Alembic-managed database with a different revision.

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
    """Create parent directories, engine, session factory and (optionally) schema.

    ``create_schema`` is the temporary ``create_all`` convenience; production
    containers should run the Alembic baseline and may pass ``False``.
    """
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    engine = create_db_engine(settings)
    if create_schema:
        Base.metadata.create_all(engine)
    return engine, create_session_factory(engine)


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
