"""Geo endpoints: upload, feedback, messages and point analysis.

Route handlers are intentionally **sync** (``def``) functions so they run in
FastAPI's threadpool and each one uses a single request-scoped SQLAlchemy
session obtained from :mod:`backend.app.deps`.
"""
from __future__ import annotations

import hashlib
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
    CitiesRequest,
    CitiesResponse,
    FeedbackRequest,
    FeedbackResponse,
    MessagesResponse,
    UploadResponse,
)
from ..services.chat import ChatService
from ..services.geo_session import GeoSessionService
from ..services.pipeline import PipelineService
from ..services.model_gateway import StreetClipGateway
from ..domain.geolocation.clue import dicts_to_clues
from ..services.city_catalog import CityCatalogError, load_city_index
from ..services.city_reasoning import CityClueReasoner

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


@router.post("/{session_id}/cities", response_model=CitiesResponse)
def list_cities(
    session_id: int,
    payload: CitiesRequest,
    settings: Settings = Depends(get_settings),
    geo: GeoSessionService = Depends(get_geo_service),
) -> CitiesResponse:
    """Rank the requested country's cities for this session's own image.

    Read-only and stateless: it neither writes messages nor changes the country
    ranking, so browsing another country's cities can never alter the analysis
    or the confirmed country. Cities come from the GeoNames catalog the backend
    validates against, and their coordinates are representative points, not the
    photo's location.
    """
    geo_session = geo.get(session_id)
    if geo_session is None:
        raise HTTPException(status_code=404, detail="session not found")

    image_path = Path(geo_session.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="image not found")

    gateway = StreetClipGateway(settings)
    if payload.iso not in gateway.country_isos():
        raise HTTPException(status_code=422, detail="unknown country")

    result = gateway.rank_cities(image_path, payload.iso)
    if result is None:
        return CitiesResponse(city_scores=[], country_iso=payload.iso,
                              scope=None, status="unavailable")

    city_scores = result["city_scores"]
    clue_dimensions = None
    clue_note = None
    if city_scores:
        # Optional clue rerank over already-validated rows: read-only, never
        # adds rows, and degrades to the untouched gateway payload.
        try:
            catalog = load_city_index(settings.cities_path)
        except CityCatalogError:
            catalog = None
        if catalog is not None:
            try:
                session_clues = dicts_to_clues(geo_session.clues)
                reranked = CityClueReasoner(catalog).rerank(
                    city_scores, session_clues, iso=payload.iso
                )
            except (KeyError, TypeError, ValueError):
                reranked = None
            if reranked is not None:
                city_scores = reranked.city_scores
                clue_dimensions = list(reranked.clue_dimensions) or None
                clue_note = reranked.coverage_note

    response = CitiesResponse(
        city_scores=city_scores,
        country_iso=result["country_iso"],
        scope=result["scope"],
        status=result["status"],
        clue_dimensions=clue_dimensions,
        clue_note=clue_note,
    )
    _check_response_serializable(response)
    return response


def _crop_bounds(x_ratio: float, y_ratio: float, width: int, height: int):
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
    return left, top, right, bottom, crop_size


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
    new_clues = []
    crop_evidence = None
    crop_size = 0
    try:
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                left, top, right, bottom, crop_size = _crop_bounds(x_ratio, y_ratio, width, height)
                crop = image.crop((left, top, right, bottom))
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=500, detail=f"crop failed: {exc}") from exc

        crop_path = settings.upload_dir / f"crop_{uuid.uuid4().hex}.png"
        crop.save(crop_path)
        crop_digest = hashlib.sha256(crop_path.read_bytes()).hexdigest()

        try:
            new_clues = pipeline.analyze(str(crop_path))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"AI analysis failed: {exc}") from exc

        gateway = StreetClipGateway(settings)
        if settings.streetclip_enabled:
            # Rank the crop BEFORE the temp file is removed; a model failure
            # keeps the full-image prior while OCR/rules keep working.
            raw_crop = gateway.rank_image(crop_path)
            if raw_crop is not None:
                crop_evidence = gateway.validate_crop(
                    {**raw_crop, "image_sha256": crop_digest},
                    {"x": x_ratio, "y": y_ratio, "size": crop_size},
                )
    finally:
        if crop_path is not None:
            crop_path.unlink(missing_ok=True)

    for clue in new_clues:
        origin = clue.get("source") or "point_click"
        clue["source"] = "point_click"
        clue["evidence"] = (clue.get("evidence") or "") + (
            f"（点击区域 {x_ratio * 100:.0f}%,{y_ratio * 100:.0f}%）"
        )
        extra = dict(clue.get("extra") or {})
        extra["origin_source"] = origin
        clue["extra"] = extra

    gateway = StreetClipGateway(settings)
    existing_evidence = (geo.get(session_id) or geo_session).model_evidence
    updated_evidence = None
    if crop_evidence is not None and existing_evidence is not None:
        # Duplicate regions / repeated imagery are ignored; the immutable
        # full-image prior is never replaced, only extended with crop evidence.
        updated_evidence = gateway.with_crop(existing_evidence, crop_evidence)
    applied_evidence = updated_evidence if updated_evidence is not None else existing_evidence

    # External inference is complete before the write transaction opens; the
    # preview and the transactional reasoning both use the exact same evidence.
    chat.prepare_update(
        session_id, new_clues=new_clues, click_pos=(x_ratio, y_ratio),
        model_evidence=applied_evidence,
    )
    with geo.transaction():
        for clue in new_clues:
            geo.add_clue(session_id, clue)

        if updated_evidence is not None:
            geo.set_model_evidence(session_id, updated_evidence)

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
