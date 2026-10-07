"""Adapter around the existing multi-turn chat/reasoning orchestration.

``services.chat.ChatService`` (legacy, repository-level) already implements the
"AI clue -> user feedback -> re-reasoning" loop. We subclass it unchanged and
only note the contract: the injected session service must be bound to a
per-request SQLAlchemy session.
"""
from __future__ import annotations

from services.chat import ChatService as LegacyChatService

__all__ = ["ChatService"]


class ChatService(LegacyChatService):
    """Backend-facing alias of the legacy chat orchestration service."""
