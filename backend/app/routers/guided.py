"""Guided-dialogue HTTP boundary.

Routes stay separate from the legacy ``geo`` router so the existing upload,
feedback, messages and read-only city APIs keep their exact contracts.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ..deps import get_guided_dialogue_service
from ..schemas import GuidedDialogueResponse, GuidedEventRequest
from ..services.guided_dialogue import (
    GuidedConflict,
    GuidedDialogueService,
    GuidedNotFound,
    GuidedUnavailable,
    GuidedValidationError,
)

router = APIRouter(prefix="/geo", tags=["guided-dialogue"])


def _translate(error: Exception) -> HTTPException:
    if isinstance(error, GuidedNotFound):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, GuidedConflict):
        return HTTPException(status_code=409, detail=str(error))
    if isinstance(error, GuidedValidationError):
        return HTTPException(status_code=422, detail=str(error))
    if isinstance(error, GuidedUnavailable):
        return HTTPException(status_code=503, detail=str(error))
    raise error


@router.post("/{session_id}/dialogue/start", response_model=GuidedDialogueResponse)
def start_dialogue(
    session_id: int,
    service: GuidedDialogueService = Depends(get_guided_dialogue_service),
) -> GuidedDialogueResponse:
    try:
        return service.start(session_id)
    except (GuidedNotFound, GuidedConflict, GuidedValidationError, GuidedUnavailable) as exc:
        raise _translate(exc) from exc


@router.get("/{session_id}/dialogue", response_model=GuidedDialogueResponse)
def get_dialogue(
    session_id: int,
    query: str | None = Query(default=None, max_length=200),
    service: GuidedDialogueService = Depends(get_guided_dialogue_service),
) -> GuidedDialogueResponse:
    try:
        return service.get(session_id, query=query)
    except (GuidedNotFound, GuidedConflict, GuidedValidationError, GuidedUnavailable) as exc:
        raise _translate(exc) from exc


@router.post("/{session_id}/dialogue/events", response_model=GuidedDialogueResponse)
def apply_dialogue_event(
    session_id: int,
    payload: GuidedEventRequest,
    service: GuidedDialogueService = Depends(get_guided_dialogue_service),
) -> GuidedDialogueResponse:
    try:
        return service.apply_event(session_id, payload)
    except (GuidedNotFound, GuidedConflict, GuidedValidationError, GuidedUnavailable) as exc:
        raise _translate(exc) from exc


__all__ = ["router"]
