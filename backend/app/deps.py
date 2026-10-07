"""FastAPI dependencies.

Design notes
------------
* Every request gets its **own** SQLAlchemy ``Session`` and it is always
  closed (and rolled back on failure) after the response is produced.
* HTTP handlers are sync ``def`` functions, so FastAPI runs them in its
  threadpool; combined with the per-request session this keeps SQLAlchemy /
  SQLite usage free of cross-thread or shared-session races.
* Heavy legacy services (``ai``, ``services.chat``) are imported lazily or
  constructed per request from objects stored on ``app.state``.
"""
from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from .config import Settings
from .services.chat import ChatService
from .services.geo_session import GeoSessionService
from .services.pipeline import PipelineService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_pipeline_service(request: Request) -> PipelineService:
    return request.app.state.pipeline_service


def get_db_session(request: Request) -> Iterator[Session]:
    """Yield one session per request; always close it."""
    session_factory = request.app.state.session_factory
    session: Session = session_factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_geo_service(session: Session = Depends(get_db_session)) -> GeoSessionService:
    return GeoSessionService(session)


def get_chat_service(geo: GeoSessionService = Depends(get_geo_service)) -> ChatService:
    return ChatService(geo)


__all__ = [
    "get_chat_service",
    "get_db_session",
    "get_geo_service",
    "get_pipeline_service",
    "get_settings",
]
