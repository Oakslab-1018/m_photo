"""Geo endpoints: upload, feedback, messages and point analysis.

Route handlers are intentionally **sync** (``def``) functions so they run in
FastAPI's threadpool and each one uses a single request-scoped SQLAlchemy
session obtained from :mod:`backend.app.deps`.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from ..config import ALLOWED_CONTENT_TYPES, ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES, Settings
from ..deps import (
    get_chat_service,
    get_geo_service,
    get_pipeline_service,
    get_settings,
)
from ..schemas import (
    AnalyzePointRequest,
    AnalyzePointResponse,
    FeedbackRequest,
    FeedbackResponse,
    MessagesResponse,
    UploadResponse,
)
from ..services.chat import ChatService
from ..services.geo_session import GeoSessionService
from ..services.pipeline import PipelineService
from ..services.model_gateway import StreetClipGateway

router = APIRouter(prefix="/geo", tags=["geo"])

UPLOAD_CHUNK_BYTES = 1024 * 1024


def _check_response_serializable(response: BaseModel) -> None:
    """Fail before committing if the response cannot be rendered as strict JSON."""
    json.dumps(response.model_dump(mode="json"), ensure_ascii=False, allow_nan=False)


def _guess_extension(filename: str | None, content_type: str | None) -> str | None:
    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[1].lower()
        if ext in ALLOWED_EXTENSIONS:
            return ext
    if content_type:
        subtype = content_type.split("/", 1)[-1].lower().split(";")[0].strip()
        if subtype == "jpeg":
            subtype = "jpg"
        if subtype in ALLOWED_EXTENSIONS:
            return subtype
    return None


def _valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def _spool_upload(upload: UploadFile, destination: Path) -> int:
    """Stream the upload to ``destination`` enforcing the size limit."""
    size = 0
    upload.file.seek(0)
    with destination.open("wb") as handle:
        while True:
            chunk = upload.file.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="image exceeds 16MB limit")
            handle.write(chunk)
    if size == 0:
        raise HTTPException(status_code=400, detail="empty upload")
    return size


@router.post("/upload", response_model=UploadResponse, status_code=201)
def upload_image(
    image: UploadFile = File(..., description="Image file (png/jpg/jpeg/gif/webp)"),
    settings: Settings = Depends(get_settings),
    geo: GeoSessionService = Depends(get_geo_service),
    chat: ChatService = Depends(get_chat_service),
    pipeline: PipelineService = Depends(get_pipeline_service),
) -> UploadResponse:
    """Store the image, run the AI pipeline, create a session and start the chat."""
    if image.content_type and image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="unsupported image content type")

    extension = _guess_extension(image.filename, image.content_type)
    if extension is None:
        raise HTTPException(status_code=400, detail="unsupported image extension")

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.{extension}"
    destination = settings.upload_dir / filename

    try:
        _spool_upload(image, destination)
        if not _valid_image(destination):
            raise HTTPException(status_code=400, detail="file is not a valid image")

        try:
            clues = pipeline.analyze(str(destination))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"AI analysis failed: {exc}") from exc

        evidence = StreetClipGateway(settings).rank_image(destination)
        chat.prepare_initial(clues, evidence)
        with geo.transaction():
            geo_session = geo.create(str(destination), filename, user_id=None)
            geo.set_clues(geo_session.id, clues)
            geo.set_model_evidence(geo_session.id, evidence)
            result = chat.initial_analysis(geo_session.id)
            response = UploadResponse(
                session_id=geo_session.id,
                image_url=f"{settings.media_url_prefix}/{filename}",
                image_filename=filename,
                clues=clues,
                scores=result.get("scores") or [],
                action=result.get("action"),
                messages=geo.list_messages(geo_session.id),
            )
            _check_response_serializable(response)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise

    return response


@router.post("/{session_id}/feedback", response_model=FeedbackResponse)
def submit_feedback(
    session_id: int,
    payload: FeedbackRequest,
    geo: GeoSessionService = Depends(get_geo_service),
    chat: ChatService = Depends(get_chat_service),
) -> FeedbackResponse:
    geo_session = geo.get(session_id)
    if geo_session is None:
        raise HTTPException(status_code=404, detail="session not found")

    feedback = payload.model_dump(exclude_none=True)
    chat.prepare_update(session_id, feedback=feedback)
    with geo.transaction():
        result = chat.handle_feedback(session_id, feedback)
        if result.get("error") == "session not found":  # defensive, should not happen
            raise HTTPException(status_code=404, detail="session not found")

        session = geo.get(session_id)
        response = FeedbackResponse(
            scores=result.get("scores") or [],
            action=result.get("action"),
            messages=geo.list_messages(session_id),
            clues=session.clues if session else [],
            exclusions=session.exclusions if session else [],
            done=result.get("done"),
            iso=result.get("iso"),
        )
        _check_response_serializable(response)
    return response


@router.get("/{session_id}/messages", response_model=MessagesResponse)
def list_messages(
    session_id: int,
    geo: GeoSessionService = Depends(get_geo_service),
) -> MessagesResponse:
    if geo.get(session_id) is None:
        raise HTTPException(status_code=404, detail="session not found")
    return MessagesResponse(messages=geo.list_messages(session_id))


@router.post("/{session_id}/analyze-point", response_model=AnalyzePointResponse)
def analyze_point(
    session_id: int,
    payload: AnalyzePointRequest,
    settings: Settings = Depends(get_settings),
    geo: GeoSessionService = Depends(get_geo_service),
    chat: ChatService = Depends(get_chat_service),
    pipeline: PipelineService = Depends(get_pipeline_service),
) -> AnalyzePointResponse:
    """Crop around a normalized click point, analyze it and re-reason."""
    geo_session = geo.get(session_id)
    if geo_session is None:
        raise HTTPException(status_code=404, detail="session not found")

    image_path = Path(geo_session.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="image not found")

    x_ratio, y_ratio = payload.x, payload.y
    crop_path: Path | None = None
    try:
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                crop_size = max(1, int(min(width, height) * 0.35))  # 1x1 images must still crop
                cx, cy = int(x_ratio * width), int(y_ratio * height)
                left = max(0, cx - crop_size // 2)
                top = max(0, cy - crop_size // 2)
                right = min(width, left + crop_size)
                bottom = min(height, top + crop_size)
                if right - left < crop_size:
                    left = max(0, right - crop_size)
                if bottom - top < crop_size:
                    top = max(0, bottom - crop_size)
                crop = image.crop((left, top, right, bottom))
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=500, detail=f"crop failed: {exc}") from exc

        crop_path = settings.upload_dir / f"crop_{uuid.uuid4().hex}.png"
        crop.save(crop_path)

        try:
            new_clues = pipeline.analyze(str(crop_path))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"AI analysis failed: {exc}") from exc
    finally:
        if crop_path is not None:
            crop_path.unlink(missing_ok=True)

    for clue in new_clues:
        clue["source"] = "point_click"
        clue["evidence"] = (clue.get("evidence") or "") + (
            f"（点击区域 {x_ratio * 100:.0f}%,{y_ratio * 100:.0f}%）"
        )
    chat.prepare_update(session_id, new_clues=new_clues, click_pos=(x_ratio, y_ratio))
    with geo.transaction():
        for clue in new_clues:
            geo.add_clue(session_id, clue)

        result = chat.reason_only(session_id, new_clues=new_clues, click_pos=(x_ratio, y_ratio))

        session = geo.get(session_id)
        response = AnalyzePointResponse(
            new_clues=new_clues,
            scores=result.get("scores") or [],
            action=result.get("action"),
            messages=geo.list_messages(session_id),
            clues=session.clues if session else [],
            exclusions=session.exclusions if session else [],
            crop_info={"x": x_ratio, "y": y_ratio, "size": crop_size},
        )
        _check_response_serializable(response)
    return response


__all__ = ["router"]
