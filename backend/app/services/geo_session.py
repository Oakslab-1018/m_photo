"""Session service backed by plain SQLAlchemy 2.x.

The class deliberately implements the same duck-typed interface that the
``services.chat.ChatService`` expects (``get``, ``set_clues``,
``add_clue``, ``add_exclusion``, ``set_confirmed``, ``add_message`` and
``list_messages``) so the existing chat/reasoning orchestration can be reused
as an import dependency instead of being copied.

Every instance is bound to one request-scoped :class:`sqlalchemy.orm.Session`;
the FastAPI dependency layer guarantees a fresh session per request and closes
it afterwards, which keeps SQLite usage thread-safe (routes are sync ``def``
handlers executed in the threadpool).
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import ChatMessage, GeoSession


class GeoSessionService:
    """Request-scoped persistence facade; mutation methods flush, never commit.

    Callers must wrap each logical write operation in :meth:`transaction`.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    # ---- internals ----
    def commit(self) -> None:
        self.session.commit()

    def rollback(self) -> None:
        self.session.rollback()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Commit one complete operation, or roll back all of its mutations.

        Queries may already have started SQLAlchemy's implicit transaction, so
        this deliberately does not call ``Session.begin()``. Routes construct
        their response inside this scope and return only after commit succeeds.
        """
        try:
            yield
            self.commit()
        except BaseException:
            self.rollback()
            raise

    def set_model_evidence(self, session_id: int, evidence: dict[str, Any] | None) -> None:
        row = self.get(session_id)
        if row is None:
            raise ValueError("session not found")
        row.model_evidence = evidence
        self.session.flush()

    def replace_last_ai_message(self, session_id: int, content: str, action: dict[str, Any]) -> None:
        row = self.session.scalars(
            select(ChatMessage).where(ChatMessage.session_id == session_id, ChatMessage.role == "ai")
            .order_by(ChatMessage.id.desc()).limit(1)
        ).first()
        if row is None:
            return
        row.content = content
        row.meta = {**row.meta, "action": action, "conversation_provider": "deepseek"}
        self.session.flush()

    # ---- sessions ----
    def create(
        self,
        image_path: str,
        image_filename: str,
        user_id: int | None = None,
    ) -> GeoSession:
        geo_session = GeoSession(
            user_id=user_id,
            image_path=image_path,
            image_filename=image_filename,
        )
        self.session.add(geo_session)
        self.session.flush()
        self.session.refresh(geo_session)
        return geo_session

    def get(self, session_id: int) -> GeoSession | None:
        return self.session.get(GeoSession, session_id)

    def set_clues(self, session_id: int, clues: list[dict[str, Any]]) -> None:
        geo_session = self.get(session_id)
        if geo_session is not None:
            geo_session.clues = list(clues)
            self.session.flush()

    def add_clue(self, session_id: int, clue: dict[str, Any]) -> None:
        geo_session = self.get(session_id)
        if geo_session is not None:
            clues = geo_session.clues
            clues.append(clue)
            geo_session.clues = clues
            self.session.flush()

    def add_exclusion(self, session_id: int, iso: str) -> None:
        geo_session = self.get(session_id)
        if geo_session is not None and iso not in geo_session.exclusions:
            exclusions = geo_session.exclusions
            exclusions.append(iso)
            geo_session.exclusions = exclusions
            self.session.flush()

    def set_confirmed(self, session_id: int, iso: str) -> None:
        geo_session = self.get(session_id)
        if geo_session is not None:
            geo_session.confirmed_iso = iso
            geo_session.status = "done"
            self.session.flush()

    # ---- messages ----
    def add_message(
        self,
        session_id: int,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
    ) -> ChatMessage:
        message = ChatMessage(session_id=session_id, role=role, content=content)
        if meta:
            message.meta = meta
        self.session.add(message)
        self.session.flush()
        self.session.refresh(message)
        return message

    def list_messages(self, session_id: int) -> list[dict[str, Any]]:
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id.asc())
        )
        return [message.to_dict() for message in self.session.scalars(stmt).all()]


__all__ = ["GeoSessionService"]
