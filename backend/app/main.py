"""ASGI entry point.

Container contract::

    uv run --frozen uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

The module exposes both a ready-made ``app`` object and a ``create_app``
factory (used by tests, which inject their own :class:`Settings`).

Schema management
-----------------
Application lifespan runs Alembic upgrades before handling requests. A pre-
Alembic Geo database is adopted only after its structure matches the initial
migration; incompatible databases stop startup without being stamped or altered.

"""
from __future__ import annotations

import logging
import math
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import Settings, get_settings
from .database import init_db
from .routers import countries, geo, health
from .services.pipeline import PipelineService

logger = logging.getLogger("backend.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    engine, session_factory = init_db(settings)
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.pipeline_service = PipelineService(settings)
    logger.info(
        "backend ready (ai_mode=%s data_dir=%s upload_dir=%s)",
        settings.geo_ai_mode,
        settings.data_dir,
        settings.upload_dir,
    )
    try:
        yield
    finally:
        engine.dispose()


def _sanitize(value: Any) -> Any:
    """Recursively make a validation-error value strictly JSON safe.

    Two problems are handled:

    * ``float('nan')`` / ``inf`` survive ``jsonable_encoder`` and make
      ``JSONResponse`` (``allow_nan=False``) raise while rendering the 422.
    * Pydantic may embed exception objects (e.g. ``ValueError``) in ``ctx``.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _safe_validation_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make ``RequestValidationError.errors()`` JSON-safe.

    The raw ``input`` is dropped on purpose: it avoids echoing the request
    payload back (privacy) and it is the most common carrier of non-finite
    floats / non-serializable objects.
    """
    safe: list[dict[str, Any]] = []
    for error in errors:
        item = {key: value for key, value in error.items() if key != "input"}
        safe.append(_sanitize(item))
    return safe


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()

    app = FastAPI(
        title="m_photo geo backend",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = resolved

    origins = list(resolved.cors_origins)
    allow_credentials = "*" not in origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Uploaded images are served under /media/<filename>. The directory is
    # created by the lifespan handler (no import-time filesystem side effects).
    app.mount(
        resolved.media_url_prefix,
        StaticFiles(directory=str(resolved.upload_dir), check_dir=False),
        name="media",
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(countries.router, prefix="/api")
    app.include_router(geo.router, prefix="/api")

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        return JSONResponse(
            status_code=exc.status_code,
            content=jsonable_encoder({"error": detail, "detail": detail}),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder(
                {
                    "error": "validation error",
                    "detail": _safe_validation_errors(exc.errors()),
                }
            ),
        )

    return app


app = create_app()


__all__ = ["app", "create_app", "lifespan"]
