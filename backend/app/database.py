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

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine, event
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


class GuidedDialogue(Base):
    """Persisted guided-dialogue state for one :class:`GeoSession`.

    A separate table (rather than a ``geo_session`` JSON column) keeps guided
    state additive and leaves the legacy upload/feedback contract untouched.
    The evolving contract state is kept as a JSON document inside this table so
    new review/proposal fields do not require a column per field; event
    idempotency and receipts live in :class:`GuidedEvent` instead. ``revision``
    is the optimistic concurrency token: every effective event performs a
    conditional update on ``(session_id, revision)`` and increments it once.
    """

    __tablename__ = "guided_dialogue"

    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("geo_session.id"), primary_key=True
    )
    revision: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    stage: Mapped[str] = mapped_column(String(20), default="country", nullable=False)
    state_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    @property
    def state(self) -> dict[str, Any]:
        try:
            value = json.loads(self.state_json or "{}")
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self.state_json = json.dumps(value, ensure_ascii=False, allow_nan=False)


class GuidedEvent(Base):
    """Idempotency ledger for guided dialogue events.

    ``(session_id, event_id)`` is unique. The receipt is the original response
    summary for the event and is replayed unchanged when the same event id is
    submitted again, regardless of the caller's later ``expected_revision``.
    """

    __tablename__ = "guided_event"
    __table_args__ = (
        UniqueConstraint("session_id", "event_id", name="uq_guided_event_session_event"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("geo_session.id"), nullable=False, index=True
    )
    event_id: Mapped[str] = mapped_column(String(200), nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    receipt_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    @property
    def receipt(self) -> dict[str, Any]:
        try:
            value = json.loads(self.receipt_json or "{}")
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    @receipt.setter
    def receipt(self, value: dict[str, Any]) -> None:
        self.receipt_json = json.dumps(value, ensure_ascii=False, allow_nan=False)


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


# Imported for its side effect: geo_catalog_models defines the 0004 tables
# (geo_catalog_source / geo_place) on ``Base.metadata`` so Alembic autogenerate
# and the safe baseline-adoption reference see the complete schema. The import
# lives at the bottom because the module maps itself onto ``Base``; the two
# models are re-exported here for callers that import from ``database``.
from .services import geo_catalog_models as _geo_catalog_models  # noqa: E402,F401
from .services.geo_catalog_models import GeoCatalogSource, GeoPlace  # noqa: E402,F401

__all__ = [
    "Base",
    "ChatMessage",
    "GeoCatalogSource",
    "GeoPlace",
    "GeoSession",
    "GuidedDialogue",
    "GuidedEvent",
    "create_db_engine",
    "create_session_factory",
    "ensure_parent_dir",
    "init_db",
]
