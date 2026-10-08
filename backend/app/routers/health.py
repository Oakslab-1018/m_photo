"""Health endpoint (validates DB availability, not just process liveness)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from .. import __version__
from ..config import Settings
from ..deps import get_db_session, get_settings
from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Round-trip a real query so a locked/missing DB yields 503, not 200."""
    try:
        value = session.execute(text("SELECT 1")).scalar_one()
        if value != 1:
            raise RuntimeError(f"unexpected SELECT 1 result: {value!r}")
    except Exception as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc

    return HealthResponse(
        status="ok",
        database="ok",
        ai_mode=settings.geo_ai_mode,
        data_dir=str(settings.data_dir),
        version=__version__,
        streetclip_enabled=settings.streetclip_enabled,
        deepseek_configured=bool((settings.deepseek_api_key or "").strip()),
        deepseek_model=settings.deepseek_model,
    )


__all__ = ["router"]
