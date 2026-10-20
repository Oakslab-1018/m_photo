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
from ..services.model_gateway import StreetClipGateway, build_inference_notice
from ..services.region_analysis import RegionAnalysisService, provider_signature
from ..domain.geolocation.clue import dicts_to_clues
from ..domain.geolocation import knowledge_graph as kg
from ..domain.geolocation.country_policy import (
    canonical_country_iso,
    country_members,
    is_country_member,
    is_excluded,
    normalize_iso,
    region_metadata,
    source_region_iso,
)
from ..services.city_catalog import CityCatalogError, load_city_index
from ..services.city_reasoning import CityClueReasoner

router = APIRouter(prefix="/geo", tags=["geo"])

UPLOAD_CHUNK_BYTES = 1024 * 1024
CITY_PUBLIC_LIMIT = 10
_SOURCE_REGION_LABELS = {
    "CN": "中国大陆",
    "HK": "中国香港",
    "MO": "中国澳门",
    "TW": "中国台湾",
}


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
            scores = result.get("scores") or []
            inference_notice = build_inference_notice(
                settings,
                evidence,
                has_trusted_evidence=bool(scores),
            )
            if inference_notice:
                geo.add_message(
                    geo_session.id,
                    "system",
                    inference_notice,
                    {"inference_notice": True},
                )
            response = UploadResponse(
                session_id=geo_session.id,
                image_url=f"{settings.media_url_prefix}/{filename}",
                image_filename=filename,
                clues=clues,
                scores=scores,
                action=result.get("action"),
                confidence=result.get("confidence"),
                inference_notice=inference_notice,
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
    # Confirmation is a country-level action: raw TW/HK/MO confirmations are
    # recorded as CN. Rejections intentionally keep their raw region scope so
    # "not Taiwan, but mainland China" can be represented.
    if feedback.get("type") == "confirm":
        canonical = canonical_country_iso(feedback.get("iso"))
        if canonical:
            feedback["iso"] = canonical
    chat.prepare_update(session_id, feedback=feedback)
    with geo.transaction():
        result = chat.handle_feedback(session_id, feedback)
        if result.get("error") == "session not found":  # defensive, should not happen
            raise HTTPException(status_code=404, detail="session not found")

        session = geo.get(session_id)
        response = FeedbackResponse(
            scores=result.get("scores") or [],
            action=result.get("action"),
            confidence=result.get("confidence"),
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
    """Rank canonical-country cities, optionally scoped to one raw source.

    Raw city rows are validated by the existing gateway contract before any
    normalization or clue rerank. A canonical query (for example ``CN``)
    merges the member scopes (CN/TW/HK/MO) deterministically; an explicit
    ``source_iso`` narrows the query to one raw scope. A legacy request with
    ``iso=TW`` still returns raw Taiwan cities but presents them under CN.
    """
    geo_session = geo.get(session_id)
    if geo_session is None:
        raise HTTPException(status_code=404, detail="session not found")

    image_path = Path(geo_session.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="image not found")

    gateway = StreetClipGateway(settings)
    available_isos = gateway.country_isos()

    requested_raw = normalize_iso(payload.iso)
    requested_country = canonical_country_iso(requested_raw)
    if not requested_raw or not requested_country or requested_country not in available_isos:
        raise HTTPException(status_code=422, detail="unknown country")

    explicit_source = normalize_iso(payload.source_iso)
    legacy_region = source_region_iso(requested_raw)
    if explicit_source:
        if not is_country_member(requested_country, explicit_source):
            raise HTTPException(status_code=422, detail="source_iso is not a member of the requested country")
        if explicit_source not in available_isos:
            raise HTTPException(status_code=422, detail="unknown source_iso")
        scopes = [explicit_source]
    elif legacy_region:
        # Legacy isoTW remains a useful raw scoped query, but the public
        # country identity below is canonical CN.
        scopes = [legacy_region] if legacy_region in available_isos else []
    else:
        scopes = [
            member for member in country_members(requested_country)
            if member in available_isos
        ]
    persisted_exclusions = list(geo_session.exclusions or [])
    excluded_scopes = [scope for scope in scopes if is_excluded(persisted_exclusions, scope)]
    scopes = [scope for scope in scopes if scope not in excluded_scopes]
    if not scopes:
        # An excluded region/country is an honest empty result, not a provider
        # outage. Keep metadata so the frontend can explain the skipped scope.
        excluded_single = excluded_scopes[0] if len(excluded_scopes) == 1 else None
        excluded_region = (
            region_metadata(excluded_single, kg.get_by_iso(excluded_single))
            if excluded_single else None
        )
        if excluded_single:
            note = f"{_SOURCE_REGION_LABELS.get(excluded_single, excluded_single)}已被排除，未检索城市候选。"
        elif excluded_scopes:
            note = "所选范围已被排除，未检索城市候选。"
        else:
            note = "没有可检索的城市范围。"
        response = CitiesResponse(
            city_scores=[],
            country_iso=requested_country,
            source_iso=excluded_single,
            source_region=excluded_region,
            source_isos=[],
            scope=None,
            status="no_cities",
            note=note,
        )
        _check_response_serializable(response)
        return response

    try:
        catalog = load_city_index(settings.catalog_db_path)
    except CityCatalogError:
        catalog = None
    try:
        session_clues = dicts_to_clues(geo_session.clues)
    except (KeyError, TypeError, ValueError):
        session_clues = []

    rows_by_id: dict[str, dict] = {}
    successful_sources: list[str] = []
    failed_sources: list[str] = []
    empty_sources: list[str] = []
    clue_dimensions: set[str] = set()
    clue_notes: list[str] = []
    limits: list[int] = []
    revisions: list[str] = []
    considered = 0

    for source_iso in scopes:
        result = gateway.rank_cities(image_path, source_iso)
        if result is None:
            # Partial provider failure: keep any other source's rows instead
            # of turning canonical China into a false empty result.
            failed_sources.append(source_iso)
            continue

        successful_sources.append(source_iso)
        scope = result.get("scope") if isinstance(result.get("scope"), dict) else {}
        limit = scope.get("city_limit_per_country")
        if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
            limits.append(limit)
        revision = scope.get("city_catalog_revision")
        if isinstance(revision, str) and revision:
            revisions.append(revision)
        cities_considered = scope.get("cities_considered")
        if isinstance(cities_considered, int) and not isinstance(cities_considered, bool):
            considered += max(0, cities_considered)

        city_scores = list(result.get("city_scores") or [])
        if not city_scores:
            empty_sources.append(source_iso)
            continue

        # Rerank over the validated raw scope first: this retains Taiwan's
        # native city aliases and never operates on normalized CN rows.
        if catalog is not None and session_clues:
            try:
                reranked = CityClueReasoner(catalog).rerank(
                    city_scores, session_clues, iso=source_iso
                )
            except (KeyError, TypeError, ValueError):
                reranked = None
            if reranked is not None:
                city_scores = reranked.city_scores
                clue_dimensions.update(reranked.clue_dimensions)
                if reranked.coverage_note:
                    clue_notes.append(reranked.coverage_note)

        region = region_metadata(source_iso, kg.get_by_iso(source_iso))
        for row in city_scores:
            if not isinstance(row, dict):
                continue
            city_id = row.get("city_id")
            if not isinstance(city_id, str) or not city_id:
                continue
            normalized = {
                **row,
                "iso": requested_country,
                "country_iso": requested_country,
                "source_iso": source_iso,
                "source_region": region,
            }
            current = rows_by_id.get(city_id)
            if current is None or _city_row_order(normalized) < _city_row_order(current):
                rows_by_id[city_id] = normalized

    city_scores = sorted(rows_by_id.values(), key=_city_row_order)[:CITY_PUBLIC_LIMIT]
    if city_scores:
        status = "ok"
    elif successful_sources or empty_sources:
        status = "no_cities"
    else:
        status = "unavailable"

    scope_payload = None
    if successful_sources:
        scope_payload = {
            "city_limit_per_country": max(limits) if limits else None,
            "city_catalog_revision": revisions[0] if revisions else None,
            "cities_considered": considered,
        }

    single_source = scopes[0] if len(scopes) == 1 else None
    source_region = (
        region_metadata(single_source, kg.get_by_iso(single_source))
        if single_source else None
    )

    notes: list[str] = []
    if excluded_scopes:
        notes.append("已跳过排除范围：" + "、".join(
            _SOURCE_REGION_LABELS.get(scope, scope) for scope in excluded_scopes
        ))
    if failed_sources:
        notes.append("以下地区城市模型暂不可用：" + "、".join(
            _SOURCE_REGION_LABELS.get(scope, scope) for scope in failed_sources
        ))
    if empty_sources and len(scopes) > 1:
        notes.append("以下地区没有符合目录的城市候选：" + "、".join(
            _SOURCE_REGION_LABELS.get(scope, scope) for scope in empty_sources
        ))
    if notes and city_scores:
        notes.append("已保留可用地区的结果。")
    if len(scopes) > 1 and scope_payload and scope_payload["city_limit_per_country"]:
        notes.append(f"每个地区最多检索 {scope_payload['city_limit_per_country']} 个城市。")
    if status == "unavailable":
        notes.append("城市模型暂不可用，未返回可用地区结果。")

    response = CitiesResponse(
        city_scores=city_scores,
        country_iso=requested_country,
        source_iso=single_source,
        source_region=source_region,
        source_isos=successful_sources,
        scope=scope_payload,
        status=status,
        note="；".join(dict.fromkeys(notes)) if notes else None,
        clue_dimensions=sorted(clue_dimensions) or None,
        clue_note="；".join(dict.fromkeys(clue_notes)) if clue_notes else None,
    )
    _check_response_serializable(response)
    return response


def _city_row_order(row: dict) -> tuple[float, str]:
    """Deterministic max-by-score order for merged validated city rows."""
    try:
        score = float(row.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return (-score, str(row.get("city_id") or ""))


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


def _point_request_input(payload: AnalyzePointRequest, region_id: str) -> dict:
    image_region = payload.image_region.model_dump() if payload.image_region else None
    return {
        "x": payload.x,
        "y": payload.y,
        "scale": payload.scale,
        "image_region": image_region,
        "purpose": payload.purpose,
        "client_request_id": payload.client_request_id,
        "region_id": region_id,
    }


def _find_persisted_point_report(
    messages: list[dict],
    *,
    region_id: str,
    client_request_id: str | None,
    current_signature: str,
) -> tuple[dict | None, dict | None, str | None]:
    for message in reversed(messages or []):
        if not isinstance(message, dict) or message.get("role") != "ai":
            continue
        meta = message.get("meta") if isinstance(message.get("meta"), dict) else {}
        report = meta.get("point_analysis")
        if not isinstance(report, dict) or report.get("id") != region_id:
            continue
        stored_signature = meta.get("point_analysis_provider_signature")
        if stored_signature != current_signature:
            continue
        request_input = meta.get("point_analysis_input") if isinstance(meta.get("point_analysis_input"), dict) else {}
        if client_request_id and request_input.get("client_request_id") == client_request_id:
            return report, request_input, "duplicate_request"
        if meta.get("point_analysis_cacheable") is True:
            return report, request_input, "duplicate_region"
    return None, None, None


def _latest_ai_state(messages: list[dict]) -> tuple[list, dict | None]:
    scores: list = []
    action: dict | None = None
    for message in reversed(messages or []):
        if not isinstance(message, dict) or message.get("role") != "ai":
            continue
        meta = message.get("meta") if isinstance(message.get("meta"), dict) else {}
        if not scores and isinstance(meta.get("scores"), list):
            scores = list(meta["scores"])
        if action is None and isinstance(meta.get("action"), dict):
            action = dict(meta["action"])
        if scores or action is not None:
            break
    return scores, action


def _point_crop_info(report: dict, request_input: dict | None) -> dict:
    request_input = request_input if isinstance(request_input, dict) else {}
    bbox = report.get("bbox_norm") if isinstance(report.get("bbox_norm"), dict) else {}
    bbox_px = report.get("bbox_px") if isinstance(report.get("bbox_px"), list) and len(report["bbox_px"]) == 4 else [0, 0, 1, 1]
    try:
        x = float(request_input.get("x", float(bbox.get("x", 0.0)) + float(bbox.get("w", 1.0)) / 2.0))
        y = float(request_input.get("y", float(bbox.get("y", 0.0)) + float(bbox.get("h", 1.0)) / 2.0))
    except (TypeError, ValueError):
        x, y = 0.5, 0.5
    try:
        size = max(1, int(max(int(bbox_px[2]) - int(bbox_px[0]), int(bbox_px[3]) - int(bbox_px[1]))))
    except (TypeError, ValueError):
        size = 1
    return {
        "x": x,
        "y": y,
        "size": size,
        "region_id": str(report.get("id") or ""),
        "bbox_px": [int(value) for value in bbox_px],
        "bbox_norm": dict(bbox),
        "scale": float(request_input.get("scale", report.get("scale", 0.25)) or 0.25),
        "purpose": request_input.get("purpose") or report.get("purpose") or "auto",
    }


def _point_response_from_report(
    *,
    geo: GeoSessionService,
    session,
    messages: list[dict],
    report: dict,
    request_input: dict | None,
    dedupe: str,
    applied: bool,
) -> AnalyzePointResponse:
    scores, action = _latest_ai_state(messages)
    confidence = ((action or {}).get("payload") or {}).get("confidence")
    report_payload = {**report, "dedupe": dedupe}
    response = AnalyzePointResponse(
        new_clues=[],
        scores=scores,
        action=action,
        confidence=confidence,
        messages=messages,
        clues=session.clues if session else [],
        exclusions=session.exclusions if session else [],
        crop_info=_point_crop_info(report_payload, request_input),
        focus_region=report_payload,
        model_status=report_payload.get("model"),
        ocr_status=report_payload.get("ocr"),
        applied=applied,
    )
    _check_response_serializable(response)
    return response


def _mark_ranking_not_applied(report: dict, reason: str) -> None:
    model = report.get("model") if isinstance(report.get("model"), dict) else None
    if model is not None:
        model["ranking_applied"] = False
        model["ranking_note"] = reason
    summary = str(report.get("summary") or "")
    summary = summary.replace(
        "已把质量合格的上下文裁剪作为弱国家排序证据，全图原始余弦保持不变；",
        "",
    ).replace("；；", "；").rstrip("。")
    report["summary"] = f"{summary}；{reason}。" if summary else f"{reason}。"


@router.post("/{session_id}/analyze-point", response_model=AnalyzePointResponse)
def analyze_point(
    session_id: int,
    payload: AnalyzePointRequest,
    settings: Settings = Depends(get_settings),
    geo: GeoSessionService = Depends(get_geo_service),
    chat: ChatService = Depends(get_chat_service),
    pipeline: PipelineService = Depends(get_pipeline_service),
) -> AnalyzePointResponse:
    """Analyze a normalized point/ROI, persist a RegionReport and re-reason."""
    geo_session = geo.get(session_id)
    if geo_session is None:
        raise HTTPException(status_code=404, detail="session not found")

    image_path = Path(geo_session.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="image not found")

    x_ratio, y_ratio = payload.x, payload.y
    gateway = StreetClipGateway(settings)
    region_service = RegionAnalysisService(settings, pipeline, gateway)
    image_region = payload.image_region.model_dump() if payload.image_region else None
    try:
        plan = region_service.plan(
            image_path,
            x=x_ratio,
            y=y_ratio,
            scale=payload.scale,
            image_region=image_region,
            purpose=payload.purpose,
        )
    except (UnidentifiedImageError, OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=500, detail=f"crop planning failed: {type(exc).__name__}") from exc

    request_input = _point_request_input(payload, plan.region_id)
    current_signature = provider_signature(settings, pipeline)
    messages = geo.list_messages(session_id)
    existing_report, existing_input, dedupe = _find_persisted_point_report(
        messages,
        region_id=plan.region_id,
        client_request_id=payload.client_request_id,
        current_signature=current_signature,
    )
    if existing_report is not None:
        return _point_response_from_report(
            geo=geo,
            session=geo.get(session_id),
            messages=messages,
            report=existing_report,
            request_input=existing_input,
            dedupe=dedupe or "duplicate_region",
            applied=False,
        )

    existing_evidence = (geo.get(session_id) or geo_session).model_evidence
    full_margin = None
    full_uncertain = None
    if isinstance(existing_evidence, dict):
        try:
            full_margin = float(existing_evidence.get("margin"))
        except (TypeError, ValueError):
            full_margin = None
        if isinstance(existing_evidence.get("uncertain"), bool):
            full_uncertain = existing_evidence["uncertain"]

    try:
        analysis = region_service.analyze(
            image_path,
            plan,
            allow_crop_ranking=existing_evidence is not None,
            full_margin=full_margin,
            full_uncertain=full_uncertain,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"local analysis failed: {type(exc).__name__}") from exc

    report = dict(analysis["report"])
    new_clues = list(analysis.get("clues") or [])
    crop_evidence = analysis.get("crop_evidence")
    for clue in new_clues:
        origin = clue.get("source") or "point_click"
        clue["source"] = "point_click"
        bbox_text = (
            f"{plan.bbox_norm['x']:.3f},{plan.bbox_norm['y']:.3f},"
            f"{plan.bbox_norm['w']:.3f},{plan.bbox_norm['h']:.3f}"
        )
        clue["evidence"] = (clue.get("evidence") or "") + (
            f"（局部区域 {x_ratio * 100:.0f}%,{y_ratio * 100:.0f}%；归一化框 {bbox_text}）"
        )
        extra = dict(clue.get("extra") or {})
        extra["origin_source"] = extra.get("origin_source") or origin
        extra["region_id"] = plan.region_id
        extra["region_bbox_norm"] = dict(plan.bbox_norm)
        clue["extra"] = extra

    updated_evidence = None
    if crop_evidence is not None and existing_evidence is not None:
        updated_evidence = gateway.with_crop(existing_evidence, crop_evidence)
    if crop_evidence is not None and updated_evidence is None:
        _mark_ranking_not_applied(report, "上下文裁剪未写入，因为区域重复或裁剪证据已达到上限")
    applied_evidence = updated_evidence if updated_evidence is not None else existing_evidence
    applied = bool(new_clues or updated_evidence is not None)
    report["dedupe"] = "new" if applied else "no_new_evidence"

    # External inference is complete before the write transaction opens; the
    # preview and the transactional reasoning both use the exact same evidence.
    chat.prepare_update(
        session_id, new_clues=new_clues, click_pos=(x_ratio, y_ratio),
        model_evidence=applied_evidence,
    )
    with geo.transaction():
        current_messages = geo.list_messages(session_id)
        duplicate_report, duplicate_input, duplicate_kind = _find_persisted_point_report(
            current_messages,
            region_id=plan.region_id,
            client_request_id=payload.client_request_id,
            current_signature=current_signature,
        )
        if duplicate_report is not None:
            return _point_response_from_report(
                geo=geo,
                session=geo.get(session_id),
                messages=current_messages,
                report=duplicate_report,
                request_input=duplicate_input,
                dedupe=duplicate_kind or "duplicate_region",
                applied=False,
            )
        for clue in new_clues:
            geo.add_clue(session_id, clue)

        if updated_evidence is not None:
            geo.set_model_evidence(session_id, updated_evidence)

        result = chat.reason_only(
            session_id,
            new_clues=new_clues,
            click_pos=(x_ratio, y_ratio),
            point_analysis=report,
            point_analysis_input=request_input,
            point_analysis_cacheable=bool(analysis.get("cacheable")),
            point_analysis_provider_signature=current_signature,
        )

        session = geo.get(session_id)
        response = AnalyzePointResponse(
            new_clues=new_clues,
            scores=result.get("scores") or [],
            action=result.get("action"),
            confidence=result.get("confidence"),
            messages=geo.list_messages(session_id),
            clues=session.clues if session else [],
            exclusions=session.exclusions if session else [],
            crop_info={
                "x": x_ratio,
                "y": y_ratio,
                "size": max(1, plan.bbox_px[2] - plan.bbox_px[0], plan.bbox_px[3] - plan.bbox_px[1]),
                "region_id": plan.region_id,
                "bbox_px": list(plan.bbox_px),
                "bbox_norm": plan.bbox_norm,
                "scale": payload.scale,
                "purpose": payload.purpose,
            },
            focus_region=report,
            model_status=report.get("model"),
            ocr_status=report.get("ocr"),
            applied=applied,
        )
        _check_response_serializable(response)
    return response


__all__ = ["router"]
