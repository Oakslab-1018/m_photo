"""FastAPI application for the standalone StreetCLIP inference service."""

from __future__ import annotations

import logging
import hmac
from contextlib import asynccontextmanager
from typing import Any

from starlette.concurrency import run_in_threadpool

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from .catalog import CatalogError, Country, compute_catalog_revision
from .config import MODEL_ID, MODEL_REVISION, Settings, load_settings
from .predictor import Predictor, PredictorUnavailableError
from .schema import (
    DecodeError,
    EmptyImageError,
    ExifGpsPresentError,
    ImageTooLargeError,
    ImageValidationError,
)
from .service import StreetClipService

logger = logging.getLogger(__name__)


def _sanitize_error_detail(detail: str) -> str:
    """Keep error text short and free of paths or secrets."""
    text = " ".join(str(detail).split())
    return text[:300]


def _bearer_token(authorization: str | None) -> str | None:
    """Extract a token from ``Bearer <token>`` or a bare ``<token>`` header."""
    if not authorization:
        return None
    value = authorization.strip()
    if not value:
        return None
    parts = value.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        token = parts[1].strip()
        return token or None
    return value


async def require_api_key(
    request: Request,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> None:
    """Reusable bearer-auth dependency bound to ``app.state.settings``.

    The per-app dependency created by :func:`create_app` is preferred; this
    module-level variant exists for callers that compose their own routers
    and is kept behaviourally identical.
    """
    expected = getattr(request.app.state.settings, "api_key", None)
    if not expected:
        return
    supplied = _bearer_token(authorization)
    if supplied is None or not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=401,
            detail="missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def create_app(
    settings: Settings | None = None,
    *,
    predictor: Predictor | None = None,
    service: StreetClipService | None = None,
) -> FastAPI:
    """Create the FastAPI app.

    ``predictor`` and ``service`` are injection points for unit tests. If no
    service is supplied, a lazily populated real StreetCLIP service is created
    during app startup (and failures leave ``/health`` reporting ready=false).
    """
    resolved = settings or load_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """Load catalog and real weights once, before serving requests."""
        active: StreetClipService = application.state.service
        try:
            active.load()
        except Exception:
            logger.exception(
                "StreetCLIP startup load failed; /health reports ready=false"
            )
        yield

    application = FastAPI(
        title="m_photo StreetCLIP inference",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.settings = resolved

    if service is None:
        service = StreetClipService(resolved, predictor=predictor)
    application.state.service = service

    async def api_key_dependency(authorization: str | None = Header(default=None)) -> None:
        expected = resolved.api_key
        if not expected:
            return
        supplied = _bearer_token(authorization)
        if supplied is None or not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
            raise HTTPException(
                status_code=401,
                detail="missing or invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @application.get("/health")
    def health() -> JSONResponse:
        service: StreetClipService = application.state.service
        payload = service.health()
        status_code = 200 if payload.get("ready") else 503
        return JSONResponse(status_code=status_code, content=payload)

    @application.post("/predict")
    async def predict(
        _auth: None = Depends(api_key_dependency),
        file: UploadFile = File(..., description="single street-level image"),
    ) -> JSONResponse:
        service_impl: StreetClipService = application.state.service
        try:
            payload = await file.read(resolved.max_upload_bytes + 1)
        except Exception as exc:  # pragma: no cover - Starlette read errors
            raise HTTPException(status_code=400, detail="unable to read upload") from exc
        finally:
            await file.close()

        try:
            result = await run_in_threadpool(service_impl.predict, payload)
        except ImageTooLargeError as exc:
            raise HTTPException(status_code=413, detail=_sanitize_error_detail(str(exc))) from exc
        except EmptyImageError as exc:
            raise HTTPException(status_code=400, detail=_sanitize_error_detail(str(exc))) from exc
        except ExifGpsPresentError as exc:
            raise HTTPException(status_code=422, detail=_sanitize_error_detail(str(exc))) from exc
        except (DecodeError, ImageValidationError) as exc:
            raise HTTPException(status_code=400, detail=_sanitize_error_detail(str(exc))) from exc
        except PredictorUnavailableError as exc:
            raise HTTPException(status_code=503, detail="model unavailable") from exc
        except Exception:  # pragma: no cover - defensive sanitization
            logger.exception("unhandled prediction failure")
            raise HTTPException(status_code=500, detail="internal inference error") from None
        return JSONResponse(status_code=200, content=result)

    return application


app = create_app()


__all__ = ["app", "create_app", "require_api_key"]
