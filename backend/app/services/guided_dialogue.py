"""Persisted country -> city guided dialogue.

The service is deliberately separate from ``services.chat``. Guided free text is
parsed into proposals/reviewable clues only; it never calls the legacy feedback
path and therefore cannot trigger that path's explicit country confirmation.

All location identities are resolved against local catalogs. City retrieval is
country-wide: model rankings are merged across the raw catalog scopes of the
confirmed country, while catalog search can surface exact matches outside the
model's bounded top rows. Persisted admin1 values remain readable as deprecated
provenance, but they never gate or validate the country -> city path.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from services.chat import ChineseFreeTextParser, PendingQuestion, PlaceMention

from ..config import Settings
from ..database import GeoSession, GuidedDialogue, GuidedEvent
from ..domain.geolocation.clue import Clue, DIM_LABELS, VALUE_LABELS, dicts_to_clues
from ..domain.geolocation.confidence import route_confidence
from ..domain.geolocation.country_policy import (
    canonical_country_iso,
    country_members,
    is_excluded,
    normalize_iso,
    source_region_iso,
)
from ..domain.geolocation.engine import normalize_place_text
from ..domain.geolocation.model_ranking import ModelAwareEngine, ModelAwareHypotheses
from ..schemas import (
    ActionModel,
    ChatMessageModel,
    ClueModel,
    ConfirmedPathModel,
    ConfidenceRouteModel,
    CountryScoreModel,
    GuidedCandidateModel,
    GuidedDialogueModel,
    GuidedDialogueResponse,
    GuidedEventRequest,
    GuidedEventTarget,
    GuidedLocationTarget,
    GuidedQuestionModel,
    GuidedQuestionOptionModel,
    GuidedReceiptModel,
    ReviewedClueModel,
)
from .city_catalog import CityRow
from .city_labels import city_name_zh
from .city_reasoning import CityClueReasoner
from .deepseek import DeepSeekClient, narration_matches_action
from .geo_session import GeoSessionService
from .geocoding import GeocodingConfig, get_shared_geocoder
from .guided_city_cache import CITY_RANKING_CACHE, CityRankingKey
from .place_resolution import CityEvidence, PlaceResolutionBatch, PlaceResolutionService
from .region_observation import (
    AdoptedOcrObservation,
    ObservationAdoptionError,
    find_persisted_ocr_observation,
    is_user_confirmed_observation,
)
from .model_gateway import (
    MAX_TARGET_CITY_IDS,
    MODEL_REVISION,
    PROMPT_VERSION,
    StreetClipGateway,
    build_inference_notice,
)

logger = logging.getLogger(__name__)

DIALOGUE_VERSION = 1
COUNTRY_CANDIDATE_LIMIT = 20
COUNTRY_CATALOG_FALLBACK_LIMIT = 50
CITY_CANDIDATE_LIMIT = 20
CITY_SEARCH_LIMIT = 50
PROPOSAL_LIMIT = 20
MAX_QUESTION_OPTIONS = 5
MAX_COUNTRY_CLUE_QUESTIONS = 3
_STREET_SUFFIXES = ("大道", "路", "街", "巷", "弄")

_CITY_ID_RE = re.compile(r"^[1-9][0-9]*$")
_ADMIN1_ID_RE = re.compile(r"^([A-Z]{2})\.(.+)$")
_UPPER_ISO_RE = re.compile(r"(?<![A-Za-z])([A-Z]{2})(?![A-Za-z])")
_POI_SUFFIXES = (
    "分店", "店", "馆", "大厦", "商厦", "商城", "市场", "商场",
    "酒店", "宾馆", "餐厅", "医院", "学校", "大学", "中心",
)
_CLUE_PRIORITY = {
    "place_name": 0,
    "language": 1,
    "driving_side": 2,
    "culture": 3,
    "building": 4,
    "vegetation": 5,
    "climate": 6,
    "vehicle": 7,
}
_LEVEL_LABELS = {"country": "国家", "admin1": "一级行政区", "city": "城市"}

# 面向用户的复核文案复用共享中文取值标签；个别展示口径在此覆盖。
_VALUE_LABEL_OVERRIDES = {"generic": "普通 / 不明显"}


def _value_label(value: object) -> str:
    raw = str(value or "").strip()
    return _VALUE_LABEL_OVERRIDES.get(raw) or VALUE_LABELS.get(raw) or raw


def _looks_like_street_text(value: object) -> bool:
    text = str(value or "").strip()
    return bool(text) and any(suffix in text for suffix in _STREET_SUFFIXES)


class GuidedNotFound(RuntimeError):
    """会话/引导状态不存在（HTTP 404）。"""


class GuidedConflict(RuntimeError):
    """Stale revision/question conflict (HTTP 409)."""


class GuidedValidationError(RuntimeError):
    """Unknown catalog identity, invalid ancestry/event (HTTP 422)."""


class GuidedUnavailable(RuntimeError):
    """Required local catalog is unavailable (HTTP 503)."""


class _ReconcileRetry(RuntimeError):
    """Internal CAS retry marker for evidence reconciliation."""


def _empty_state() -> dict[str, Any]:
    return {
        "confirmed": {"country": None, "admin1": None, "city": None},
        "exclusions": [],
        "clues": [],
        "pending_question": None,
        "asked_question_ids": [],
        "asked_clue_questions": [],
        "confidence_confirm_isos": [],
        "proposals": [],
        "place_resolutions": [],
        "proposal_meta": {},
        "conflict": None,
        "pending_correction": None,
        "session_evidence_fingerprint": "",
        "last_text": "",
    }


def _normalise_state(value: object) -> dict[str, Any]:
    result = _empty_state()
    if not isinstance(value, Mapping):
        return result
    confirmed = value.get("confirmed")
    if isinstance(confirmed, Mapping):
        for level in ("country", "admin1", "city"):
            target = confirmed.get(level)
            result["confirmed"][level] = dict(target) if isinstance(target, Mapping) else None
    for key in (
        "exclusions", "clues", "asked_question_ids", "asked_clue_questions",
        "confidence_confirm_isos", "proposals", "place_resolutions",
    ):
        raw = value.get(key)
        if isinstance(raw, list):
            result[key] = [dict(item) if isinstance(item, Mapping) else item for item in raw]
    # Legacy alias-derived metadata must never recreate a mapping.  New
    # resolution metadata always carries resolution_ids; stale curated entries
    # are dropped rather than converted into hidden static knowledge.
    for record in result.get("clues") or []:
        if not isinstance(record, dict):
            continue
        extra = record.get("_extra")
        if isinstance(extra, dict):
            extra.pop("place_alias", None)
            record["_extra"] = extra
    proposal_meta = value.get("proposal_meta")
    if isinstance(proposal_meta, Mapping):
        result["proposal_meta"] = {
            str(key): dict(item)
            for key, item in proposal_meta.items()
            if isinstance(item, Mapping)
            and (
                item.get("resolution_ids")
                or not str(key).startswith("city:")
            )
        }
    valid_meta_keys = set(result["proposal_meta"])
    result["proposals"] = [
        item for item in result.get("proposals") or []
        if not isinstance(item, Mapping)
        or item.get("level") != "city"
        or _target_state_key(item) in valid_meta_keys
    ]
    question = value.get("pending_question")
    result["pending_question"] = dict(question) if isinstance(question, Mapping) else None
    conflict = value.get("conflict")
    result["conflict"] = dict(conflict) if isinstance(conflict, Mapping) else None
    correction = value.get("pending_correction")
    result["pending_correction"] = dict(correction) if isinstance(correction, Mapping) else None
    if isinstance(result.get("place_resolutions"), list):
        result["place_resolutions"] = [
            item for item in result["place_resolutions"] if isinstance(item, Mapping)
        ][-20:]
    result["session_evidence_fingerprint"] = str(value.get("session_evidence_fingerprint") or "")
    result["last_text"] = str(value.get("last_text") or "")
    return result


def _json_hash(prefix: str, value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:24]}"


def _strict_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _semantic_event_payload(event: GuidedEventRequest | Mapping[str, Any]) -> dict[str, Any]:
    payload = (
        event.model_dump(mode="json", exclude_none=True)
        if isinstance(event, GuidedEventRequest)
        else dict(event)
    )
    payload.pop("expected_revision", None)
    return payload


def _event_fingerprint(event: GuidedEventRequest | Mapping[str, Any]) -> str:
    return _json_hash("event", _semantic_event_payload(event))


def _clamp_confidence(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not (number == number) or number in (float("inf"), float("-inf")):
        return None
    return min(1.0, max(0.0, number))


def _raw_clue_dict(value: object) -> dict[str, Any] | None:
    if isinstance(value, Clue):
        return value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _clue_id(clue: Mapping[str, Any], occurrence: int = 0) -> str:
    identity = {
        "dimension": clue.get("dimension"),
        "value": clue.get("value"),
        "source": clue.get("source"),
        "evidence": clue.get("evidence"),
        "occurrence": occurrence,
    }
    return _json_hash("clue", identity)


def _reviewed_record(raw: Mapping[str, Any], occurrence: int = 0) -> dict[str, Any]:
    dimension = str(raw.get("dimension") or "").strip()
    value = str(raw.get("value") or "").strip()
    source = str(raw.get("source") or "").strip() or "machine"
    evidence = str(raw.get("evidence") or raw.get("description") or "").strip()
    confidence = _clamp_confidence(raw.get("confidence"))
    extra = raw.get("extra") if isinstance(raw.get("extra"), Mapping) else {}
    extra = dict(extra)
    extra.pop("place_alias", None)
    return {
        "id": _clue_id(raw, occurrence),
        "dimension": dimension,
        "value": value,
        "description": evidence,
        "source": source,
        "confidence": confidence,
        "status": "pending",
        "_extra": dict(extra),
    }


def _reviewed_to_clue(record: Mapping[str, Any]) -> Clue | None:
    dimension = str(record.get("dimension") or "").strip()
    value = str(record.get("value") or "").strip()
    if not dimension or not value:
        return None
    extra = record.get("_extra")
    extra = dict(extra) if isinstance(extra, Mapping) else {}
    confidence = _clamp_confidence(record.get("confidence"))
    return Clue(
        dimension=dimension,
        value=value,
        confidence=confidence if confidence is not None else 0.0,
        evidence=str(record.get("description") or ""),
        source=str(record.get("source") or "machine"),
        extra=extra,
    )


def _target_key(target: Mapping[str, Any]) -> tuple[str, str]:
    return (str(target.get("level") or ""), str(target.get("id") or ""))


def _target_state_key(target: Mapping[str, Any]) -> str:
    level, identifier = _target_key(target)
    return f"{level}:{identifier}"


def _question_model(question: Mapping[str, Any] | None) -> GuidedQuestionModel | None:
    if not isinstance(question, Mapping):
        return None
    options = None
    raw_options = question.get("options")
    if isinstance(raw_options, list):
        options = [
            GuidedQuestionOptionModel(value=str(item.get("value")), label=str(item.get("label")))
            for item in raw_options
            if isinstance(item, Mapping) and item.get("value") and item.get("label")
        ] or None
    raw_target = question.get("target")
    target = GuidedLocationTarget(**dict(raw_target)) if isinstance(raw_target, Mapping) else None
    return GuidedQuestionModel(
        id=str(question.get("id") or ""),
        kind=str(question.get("kind") or "request_clue"),
        message=str(question.get("message") or "请进一步说明地点。"),
        dimension=str(question["dimension"]) if question.get("dimension") else None,
        clue_id=str(question["clue_id"]) if question.get("clue_id") else None,
        target=target,
        options=options,
    )


def _review_question(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build the authoritative review question for one clue record."""
    clue_id = str(record.get("id") or "")
    dimension = str(record.get("dimension") or "")
    value = str(record.get("value") or "")
    return {
        "id": f"review:{clue_id}",
        "kind": "review_clue",
        "message": (
            f"我们初步判断这是「{_value_label(value)}」的"
            f"{DIM_LABELS.get(dimension, dimension)}，您觉得准确吗？"
        ),
        "dimension": dimension,
        "clue_id": clue_id,
        "options": [
            {"value": "confirmed", "label": "确认"},
            {"value": "rejected", "label": "否定"},
        ],
    }


@dataclass
class _FreeTextPreflight:
    parsed: Any
    resolution: PlaceResolutionBatch | None = None
    notice: str | None = None
    mentions: tuple[PlaceMention, ...] = ()


@dataclass
class _ObservationAdoptionPreflight:
    observation: AdoptedOcrObservation
    place_term: str
    resolution: PlaceResolutionBatch | None = None
    notice: str | None = None


@dataclass
class _Prepared:
    state: dict[str, Any]
    stage: str
    candidates: list[dict[str, Any]]
    scores: list[Any]
    action: ActionModel
    confidence: dict[str, Any] | None
    legacy_confirmed_iso: str | None
    legacy_status: str
    inference_notice: str | None = None


class GuidedDialogueService:
    """Request-scoped transactional guided-dialogue service."""

    def __init__(self, geo: GeoSessionService, settings: Settings) -> None:
        self.geo = geo
        self.settings = settings
        self.parser = ChineseFreeTextParser()
        self.gateway = StreetClipGateway(settings)
        self.provider = DeepSeekClient(settings)
        self.geocoding_config = GeocodingConfig.from_env()
        self.place_resolver = PlaceResolutionService(
            geocoder=get_shared_geocoder(self.geocoding_config),
            cache_ttl_seconds=self.geocoding_config.cache_ttl_seconds,
            negative_cache_ttl_seconds=self.geocoding_config.negative_cache_ttl_seconds,
        )
        self._store_instance: Any | None = None
        self._country_records_cache: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------- lifecycle
    def start(self, session_id: int) -> GuidedDialogueResponse:
        session = self.geo.get(session_id)
        if session is None:
            raise GuidedNotFound("会话不存在")
        row = self.geo.session.get(GuidedDialogue, session_id)
        if row is not None:
            return self.get(session_id)

        state, _ = self._reconcile_session_evidence(_empty_state(), session)
        prepared = self._prepare(state, session, query=None, ensure_question=True)
        prepared = self._maybe_narrate(prepared, session_id)
        now = datetime.utcnow()
        row = GuidedDialogue(
            session_id=session_id,
            revision=0,
            stage=prepared.stage,
            state_json="{}",
            created_at=now,
            updated_at=now,
        )
        row.state = prepared.state
        try:
            with self.geo.transaction():
                self.geo.session.add(row)
                self.geo.session.flush()
                self._add_guided_message(
                    session_id,
                    "ai",
                    prepared.action.message,
                    {
                        "guided": True,
                        "guided_start": True,
                        "action": prepared.action.model_dump(mode="json"),
                        "dialogue_revision": 0,
                    },
                )
                response = self._response(session_id, prepared, revision=0, receipt=None)
                self._strict_response(response)
            return response
        except IntegrityError:
            self.geo.rollback()
            existing = self.geo.session.get(GuidedDialogue, session_id)
            if existing is None:
                raise
            state, _ = self._reconcile_session_evidence(_normalise_state(existing.state), session)
            prepared = self._prepare(state, session, query=None, ensure_question=True)
            return self._response(session_id, prepared, revision=existing.revision, receipt=None)

    def get(self, session_id: int, *, query: str | None = None) -> GuidedDialogueResponse:
        for _attempt in range(3):
            session = self.geo.get(session_id)
            if session is None:
                raise GuidedNotFound("会话不存在")
            row = self.geo.session.get(GuidedDialogue, session_id)
            if row is None:
                raise GuidedNotFound("引导对话尚未初始化")
            state, changed = self._reconcile_session_evidence(_normalise_state(row.state), session)
            prepared = self._prepare(state, session, query=query, ensure_question=changed)
            if not changed:
                response = self._response(session_id, prepared, revision=row.revision, receipt=None)
                self._strict_response(response)
                return response
            base_revision = int(row.revision)
            new_revision = base_revision + 1
            try:
                with self.geo.transaction():
                    result = self.geo.session.execute(
                        update(GuidedDialogue)
                        .where(
                            GuidedDialogue.session_id == session_id,
                            GuidedDialogue.revision == base_revision,
                        )
                        .values(
                            revision=new_revision,
                            stage=prepared.stage,
                            state_json=json.dumps(prepared.state, ensure_ascii=False, allow_nan=False),
                            updated_at=datetime.utcnow(),
                        )
                    )
                    if result.rowcount != 1:
                        raise _ReconcileRetry
                    response = self._response(session_id, prepared, revision=new_revision, receipt=None)
                    self._strict_response(response)
                return response
            except _ReconcileRetry:
                self.geo.rollback()
                continue
        raise GuidedConflict("对话版本已过期，请重新加载后再试")

    def apply_event(self, session_id: int, event: GuidedEventRequest) -> GuidedDialogueResponse:
        session = self.geo.get(session_id)
        if session is None:
            raise GuidedNotFound("会话不存在")
        row = self.geo.session.get(GuidedDialogue, session_id)
        if row is None:
            raise GuidedNotFound("引导对话尚未初始化")

        existing = self._event_by_id(session_id, event.event_id)
        if existing is not None:
            return self._replay_existing(session_id, event, existing)

        base_revision = int(row.revision)
        if int(event.expected_revision) != base_revision:
            raise GuidedConflict("对话版本已过期，请重新加载后再试")

        state = _normalise_state(row.state)
        state, evidence_changed = self._reconcile_session_evidence(state, session)
        if evidence_changed:
            raise GuidedConflict("底层证据已更新，请重新加载对话后再试")
        preflight: _FreeTextPreflight | _ObservationAdoptionPreflight | None = None
        if event.type == "free_text":
            preflight = self._preflight_free_text(state, session, event)
        elif event.type == "adopt_observation":
            try:
                observation = find_persisted_ocr_observation(
                    self.geo.list_messages(session_id),
                    report_id=event.report_id or "",
                    observation_id=event.observation_id or "",
                )
            except ObservationAdoptionError as exc:
                raise GuidedValidationError(str(exc)) from exc
            preflight = self._preflight_observation_adoption(
                state,
                session,
                observation,
            )
        outcome = self._apply_event(state, session, event, preflight=preflight)
        query = None
        prepared = self._prepare(
            outcome["state"],
            session,
            query=query,
            ensure_question=bool(outcome["effective"]),
        )
        if event.type == "free_text" and outcome["effective"]:
            prepared = self._maybe_narrate(
                prepared,
                session_id,
                feedback=(event.text or event.value or outcome["summary"]),
            )
        receipt = {
            "event_id": event.event_id,
            "status": outcome["status"],
            "summary": outcome["summary"],
        }
        audit_target = None
        if event.target is not None:
            try:
                audit_target = self._resolve_target(event.target)
            except GuidedValidationError:
                audit_target = None
        event_payload = event.model_dump(mode="json", exclude_none=True)
        adoption_audit = (
            preflight.observation.audit_payload()
            if isinstance(preflight, _ObservationAdoptionPreflight)
            else None
        )
        event_clue_id = event.clue_id or outcome.get("adopted_clue_id")
        audit = {
            "payload": event_payload,
            "fingerprint": _event_fingerprint(event),
            "validated_target": audit_target,
            "clue_id": event_clue_id,
            "level": event.level,
            "proposals": prepared.state.get("proposals") or [],
            "place_resolutions": prepared.state.get("place_resolutions") or [],
        }
        if adoption_audit is not None:
            adoption_audit = {**adoption_audit, "clue_id": event_clue_id}
            audit["adopted_observation"] = adoption_audit
        if event.type == "free_text" and event.text:
            user_content = event.text.strip()
        elif event.type == "adopt_observation" and isinstance(
            preflight, _ObservationAdoptionPreflight
        ):
            # The user's turn is the authoritative stored OCR text, never the
            # client-provided placeholder/summary.
            user_content = preflight.observation.text
        elif event.type == "answer" and event.value:
            user_content = f"回答：{event.value.strip()}"
        else:
            user_content = outcome["summary"]
        new_revision = base_revision + (1 if outcome["effective"] else 0)
        try:
            with self.geo.transaction():
                update_values: dict[str, Any] = {"updated_at": datetime.utcnow()}
                if outcome["effective"]:
                    update_values.update(
                        revision=new_revision,
                        stage=prepared.stage,
                        state_json=json.dumps(
                            prepared.state,
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                    )
                result = self.geo.session.execute(
                    update(GuidedDialogue)
                    .where(
                        GuidedDialogue.session_id == session_id,
                        GuidedDialogue.revision == base_revision,
                    )
                    .values(**update_values)
                )
                if result.rowcount != 1:
                    raise GuidedConflict("对话版本已过期，请重新加载后再试")
                ledger = GuidedEvent(
                    session_id=session_id,
                    event_id=event.event_id,
                    revision=new_revision,
                    status=outcome["status"],
                    summary=outcome["summary"],
                    receipt_json="{}",
                )
                ledger.receipt = {**receipt, "audit": audit}
                self.geo.session.add(ledger)
                self.geo.session.flush()
                self._sync_legacy_session(session, prepared.legacy_confirmed_iso, prepared.legacy_status)
                self._add_guided_message(
                    session_id,
                    "user",
                    user_content,
                    {
                        "guided": True,
                        "guided_event_id": event.event_id,
                        "event_type": event.type,
                        "event_payload": event_payload,
                        "validated_target": audit_target,
                        "clue_id": event_clue_id,
                        "adopted_observation": adoption_audit,
                        "proposals": prepared.state.get("proposals") or [],
                        "receipt": receipt,
                    },
                )
                self._add_guided_message(
                    session_id,
                    "ai",
                    prepared.action.message,
                    {
                        "guided": True,
                        "guided_event_id": event.event_id,
                        "event_type": event.type,
                        "action": prepared.action.model_dump(mode="json"),
                        "dialogue_revision": new_revision,
                        "event_payload": event_payload,
                        "validated_target": audit_target,
                        "clue_id": event_clue_id,
                        "adopted_observation": adoption_audit,
                    },
                )
                response = self._response(
                    session_id,
                    prepared,
                    revision=new_revision,
                    receipt=GuidedReceiptModel(**receipt),
                )
                self._strict_response(response)
            return response
        except GuidedConflict:
            existing = self._event_by_id(session_id, event.event_id)
            if existing is not None:
                return self._replay_existing(session_id, event, existing)
            raise
        except IntegrityError:
            self.geo.rollback()
            existing = self._event_by_id(session_id, event.event_id)
            if existing is None:
                raise
            return self._replay_existing(session_id, event, existing)

    # ------------------------------------------------------ external preflight
    def _preflight_free_text(
        self,
        state: dict[str, Any],
        session: GeoSession,
        event: GuidedEventRequest,
    ) -> _FreeTextPreflight:
        """Parse and resolve user text before the write transaction.

        Provider I/O happens here, never in GET and never while a database
        transaction is held open.  The resulting state is still only a
        proposal; a later explicit confirm_location is required.
        """

        text = (event.text or "").strip()
        if not text:
            raise GuidedValidationError("缺少自由文本内容")
        pending = self._parser_pending(state.get("pending_question"))
        parsed = self.parser.parse(text, pending=pending)
        session_id = int(session.id)
        history = self.geo.list_messages(session_id)
        catalog = self.parser.remote_catalog()
        if (self.settings.deepseek_api_key or "").strip():
            # Snapshot bounded context first, then end the read transaction
            # before any external extraction request.
            self.geo.rollback()
            try:
                raw_plan = self.provider.extract({
                    "text": text,
                    "history": history,
                    "pending": pending.to_dict() if pending is not None else None,
                    "catalog": catalog,
                })
                remote_plan = self.parser.validate_remote(raw_plan, text, pending)
                if remote_plan.clues or remote_plan.reject_isos or remote_plan.place_mentions:
                    parsed = parsed.merged(remote_plan)
            except Exception as exc:
                logger.warning(
                    "Guided DeepSeek extraction unavailable; using rules (%s)",
                    type(exc).__name__,
                )

        mentions = tuple(parsed.place_mentions)
        if not mentions:
            return _FreeTextPreflight(parsed=parsed)
        resolution, notice = self._resolve_place_mentions(
            state,
            session,
            mentions,
            include_prior_context=True,
        )
        return _FreeTextPreflight(
            parsed=parsed,
            resolution=resolution,
            notice=notice,
            mentions=mentions,
        )

    def _preflight_observation_adoption(
        self,
        state: dict[str, Any],
        session: GeoSession,
        observation: AdoptedOcrObservation,
    ) -> _ObservationAdoptionPreflight:
        """Build literal place evidence from stored OCR text, before writes.

        This path never invokes the free-text LLM extractor. It passes only the
        adopted OCR string as a legal ``PlaceMention`` and reuses the same local
        catalog/network resolver as guided free text.
        """

        compact = " ".join(observation.text.split())
        place_term = compact[:60].strip()
        if not place_term:
            raise GuidedValidationError("OCR 观察没有可检索的文字")
        preflight = _ObservationAdoptionPreflight(
            observation=observation,
            place_term=place_term,
        )
        # A different event id for the exact same confirmed observation is an
        # acknowledged no-op and must not trigger another network lookup.
        if self._confirmed_adoption_record(state, observation) is not None:
            return preflight
        mention = PlaceMention(
            quote=observation.text[:200],
            search_terms=(place_term,),
            parser="ocr_adopted",
        )
        resolution, notice = self._resolve_place_mentions(
            state,
            session,
            (mention,),
            include_prior_context=True,
        )
        return _ObservationAdoptionPreflight(
            observation=observation,
            place_term=place_term,
            resolution=resolution,
            notice=notice,
        )

    def _resolve_place_mentions(
        self,
        state: dict[str, Any],
        session: GeoSession,
        mentions: Sequence[PlaceMention],
        *,
        include_prior_context: bool,
    ) -> tuple[PlaceResolutionBatch | None, str | None]:
        """Resolve literal mentions through the existing bounded resolver.

        All provider I/O remains outside the CAS write transaction. Both
        free-text parsing and observation adoption may combine a bounded set of
        still-unresolved prior literal terms when the current term looks like a
        partial POI or street reference.
        """

        resolved_mentions = list(mentions)
        if include_prior_context and self._mentions_need_context(resolved_mentions):
            resolved_mentions.extend(self._prior_unresolved_mentions(state))
        confirmed = state.get("confirmed") or {}
        country = confirmed.get("country") if isinstance(confirmed, Mapping) else None
        country_iso = normalize_iso(country.get("id")) if isinstance(country, Mapping) else ""
        scopes = self._country_source_scopes(state)
        try:
            resolution = self.place_resolver.resolve(
                self._store(),
                resolved_mentions,
                country_iso=country_iso or None,
                source_isos=scopes,
                between_local_and_network=self.geo.rollback,
            )
        except Exception as exc:
            logger.warning("Place resolution unavailable (%s)", type(exc).__name__)
            return (
                None,
                "地名检索暂时不可用；文字线索已保留，请补充城市、区名或附近地标。",
            )
        return resolution, resolution.notice

    @staticmethod
    def _mentions_need_context(mentions: Sequence[PlaceMention]) -> bool:
        markers = ("路牌", "门牌", "招牌", "标牌", "指示牌", "旁边", "附近", "对面", "隔壁", "周围")
        suffixes = (
            "大道", "大街", "路", "街", "巷", "弄", "胡同", "广场", "公园", "站", "桥",
            *_POI_SUFFIXES,
        )
        for mention in mentions:
            quote = str(getattr(mention, "quote", "") or "")
            if any(marker in quote for marker in markers):
                return True
            for term in getattr(mention, "search_terms", ()) or ():
                if any(str(term).endswith(suffix) for suffix in suffixes):
                    return True
        return False

    def _prior_unresolved_mentions(self, state: Mapping[str, Any]) -> list[PlaceMention]:
        """Combine only recent unresolved user terms, never all catalog text."""

        active_ids = self._active_guided_clue_ids(state)
        result: list[PlaceMention] = []
        seen_terms: set[str] = set()
        for entry in reversed(list(state.get("place_resolutions") or [])[-4:]):
            if not isinstance(entry, Mapping):
                continue
            if self._resolution_expired(entry):
                continue
            if str(entry.get("status") or "") not in {"unresolved", "ambiguous", "unavailable"}:
                continue
            clue_ids = {str(item) for item in entry.get("clue_ids") or [] if str(item)}
            if clue_ids and not clue_ids.issubset(active_ids):
                continue
            terms = tuple(str(item) for item in entry.get("query_terms") or [] if str(item))
            if not terms:
                continue
            fresh = tuple(
                term for term in terms
                if normalize_place_text(term) not in seen_terms
            )
            if not fresh:
                continue
            seen_terms.update(normalize_place_text(term) for term in fresh)
            result.append(PlaceMention(
                quote=str(entry.get("quote") or " ".join(fresh))[:200],
                search_terms=fresh[:2],
                parser="state",
            ))
            if len(result) >= 1:
                break
        return result

    @staticmethod
    def _active_guided_clue_ids(state: Mapping[str, Any]) -> set[str]:
        active: set[str] = set()
        for record in state.get("clues") or []:
            if not isinstance(record, Mapping) or not record.get("id"):
                continue
            origin_user = str(record.get("_origin") or "") == "guided_user"
            adopted_observation = is_user_confirmed_observation(record.get("_extra"))
            if not origin_user and not adopted_observation:
                continue
            status = str(record.get("status") or "pending")
            if adopted_observation:
                # A retracted/uncertain adopted observation must stop keeping
                # its catalog resolution active; confirming can re-enable it
                # within the existing TTL.
                if status == "confirmed":
                    active.add(str(record.get("id")))
            elif status in {"pending", "confirmed"}:
                active.add(str(record.get("id")))
        return active

    # ----------------------------------------------------------- persistence
    def _event_by_id(self, session_id: int, event_id: str) -> GuidedEvent | None:
        return self.geo.session.scalars(
            select(GuidedEvent).where(
                GuidedEvent.session_id == session_id,
                GuidedEvent.event_id == event_id,
            )
        ).first()

    def _replay_existing(
        self,
        session_id: int,
        event: GuidedEventRequest,
        existing: GuidedEvent,
    ) -> GuidedDialogueResponse:
        receipt = existing.receipt or {}
        audit = receipt.get("audit") if isinstance(receipt.get("audit"), Mapping) else {}
        stored_fingerprint = str(audit.get("fingerprint") or "")
        if not stored_fingerprint and isinstance(audit.get("payload"), Mapping):
            stored_fingerprint = _event_fingerprint(audit["payload"])
        if stored_fingerprint != _event_fingerprint(event):
            raise GuidedConflict("该 event_id 已绑定其他事件内容")
        response = self.get(session_id, query=None)
        return response.model_copy(update={
            "receipt": GuidedReceiptModel(
                event_id=existing.event_id,
                status=existing.status,
                summary=existing.summary,
            )
        })

    def _add_guided_message(
        self,
        session_id: int,
        role: str,
        content: str,
        meta: dict[str, Any],
    ) -> None:
        self.geo.add_message(session_id, role, content, meta=meta)

    def _sync_legacy_session(self, session: GeoSession, confirmed_iso: str | None, status: str) -> None:
        session.confirmed_iso = confirmed_iso
        session.status = status
        self.geo.session.flush()

    @staticmethod
    def _strict_response(response: GuidedDialogueResponse) -> None:
        _strict_json(response.model_dump(mode="json"))

    # ------------------------------------------------------------- catalogs
    def _store(self) -> Any:
        if self._store_instance is None:
            try:
                from .geo_catalog_store import GeoCatalogStore

                store = GeoCatalogStore(self.geo.session)
                revisions = store.revisions()
                required = (
                    revisions.get("countries"),
                    revisions.get("cities"),
                    revisions.get("import_version"),
                )
                if not all(required):
                    raise GuidedUnavailable("地理目录数据库尚未完成导入")
                self._store_instance = store
            except GuidedUnavailable:
                raise
            except Exception as exc:
                raise GuidedUnavailable("地理目录数据库不可用") from exc
        return self._store_instance

    @staticmethod
    def _unavailable(exc: Exception) -> GuidedUnavailable:
        return GuidedUnavailable("地理目录数据库不可用")

    def _admin_rows(self, iso: str) -> list[Any]:
        source = normalize_iso(iso)
        if not source:
            return []
        try:
            return list(self._store().admin1_rows(source))
        except Exception as exc:
            raise self._unavailable(exc) from exc

    def _admin_by_id(self, admin1_id: str) -> Any | None:
        try:
            return self._store().get_place("admin1", admin1_id)
        except Exception as exc:
            raise self._unavailable(exc) from exc

    @staticmethod
    def _admin_iso(row: Any) -> str:
        iso = normalize_iso(getattr(row, "iso", None) or getattr(row, "source_iso", None))
        if iso:
            return iso
        admin1_id = str(getattr(row, "admin1_id", "") or getattr(row, "place_id", "") or "")
        return normalize_iso(admin1_id.split(".", 1)[0])

    @staticmethod
    def _admin_code(row: Any) -> str:
        code = getattr(row, "code", None)
        if code is not None:
            return str(code)
        admin1_id = str(getattr(row, "admin1_id", "") or getattr(row, "place_id", "") or "")
        return admin1_id.split(".", 1)[1] if "." in admin1_id else ""

    @staticmethod
    def _admin_label(row: Any) -> str:
        label = getattr(row, "label", None)
        if callable(label):
            label = label()
        if label:
            return str(label)
        for key in ("name_zh", "name", "ascii_name"):
            value = getattr(row, key, None)
            if value:
                return str(value)
        return GuidedDialogueService._admin_code(row)

    @staticmethod
    def _admin_aliases(row: Any) -> list[str]:
        values: list[str] = []
        aliases = getattr(row, "aliases", None)
        if isinstance(aliases, str):
            values.append(aliases)
        elif isinstance(aliases, Iterable):
            values.extend(str(item) for item in aliases if item)
        for key in ("name", "ascii_name", "name_zh"):
            value = getattr(row, key, None)
            if value:
                values.append(str(value))
        return list(dict.fromkeys(values))

    def _admin1_target(self, row: Any, country_iso: str | None = None) -> dict[str, Any]:
        iso = self._admin_iso(row)
        code = self._admin_code(row)
        admin1_id = str(
            getattr(row, "admin1_id", "")
            or getattr(row, "place_id", "")
            or f"{iso}.{code}"
        )
        canonical = canonical_country_iso(country_iso or iso)
        try:
            place = self._store().get_place("admin1", admin1_id)
        except Exception as exc:
            raise self._unavailable(exc) from exc
        if place is None:
            raise GuidedUnavailable("一级行政区目录条目缺失")
        coordinate = self._coordinate_fields(place)
        return {
            "level": "admin1",
            "id": admin1_id,
            "label": self._admin_label(row),
            "country_iso": canonical,
            "source_iso": iso,
            "admin1_id": admin1_id,
            "lat": coordinate["lat"],
            "lng": coordinate["lng"],
            **coordinate["provenance"],
        }

    @staticmethod
    def _coordinate_fields(place: Any | None) -> dict[str, Any]:
        if place is None:
            return {
                "lat": None,
                "lng": None,
                "provenance": {
                    "coordinate_source": None,
                    "coordinate_quality": None,
                    "coordinate_city_id": None,
                },
            }
        quality = getattr(place, "coordinate_quality", None)
        if callable(quality):
            quality = quality()
        city_id = getattr(place, "coordinate_city_id", None)
        source = getattr(place, "coordinate_source", None)
        return {
            "lat": getattr(place, "lat", None),
            "lng": getattr(place, "lng", None),
            "provenance": {
                "coordinate_source": str(source) if source is not None else None,
                "coordinate_quality": str(quality) if quality is not None else None,
                "coordinate_city_id": int(city_id) if city_id is not None else None,
            },
        }

    def _upgrade_target_provenance(self, target: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(target)
        for key in ("coordinate_source", "coordinate_quality", "coordinate_city_id"):
            result.setdefault(key, None)
        level = str(result.get("level") or "")
        try:
            if level == "country":
                place = self._store().get_place("country", str(result.get("id") or ""))
            elif level == "admin1":
                place = self._store().get_place("admin1", str(result.get("id") or ""))
            elif level == "city":
                place = self._store().get_place(
                    "city",
                    str(result.get("city_id") or result.get("id") or ""),
                )
            else:
                place = None
        except Exception as exc:
            if level == "admin1":
                return result
            raise self._unavailable(exc) from exc
        if place is None:
            if level == "admin1":
                return result
            raise GuidedUnavailable("地理目录条目缺失")
        fields = self._coordinate_fields(place)
        if result.get("lat") is None:
            result["lat"] = fields["lat"]
        if result.get("lng") is None:
            result["lng"] = fields["lng"]
        result.update(fields["provenance"])
        if level == "city":
            result["label"] = city_name_zh(
                str(result.get("city_id") or result.get("id") or ""),
                str(getattr(place, "name", "") or ""),
            )
        return result

    def _country_records(self) -> list[dict[str, Any]]:
        if self._country_records_cache is not None:
            return self._country_records_cache
        try:
            records = self._store().countries()
        except Exception as exc:
            raise self._unavailable(exc) from exc
        self._country_records_cache = [
            dict(item) for item in records if isinstance(item, Mapping)
        ]
        return self._country_records_cache

    def _country_record(self, iso: str) -> dict[str, Any] | None:
        canonical = canonical_country_iso(iso)
        if not canonical:
            return None
        for record in self._country_records():
            if canonical_country_iso(record.get("iso")) == canonical:
                return record
        return None

    def _country_exists(self, iso: str) -> bool:
        return self._country_record(iso) is not None

    def _country_label(self, iso: str) -> str:
        country = self._country_record(iso) or {}
        return str(country.get("name_zh") or country.get("name") or iso)

    def _country_target(self, iso: str) -> dict[str, Any]:
        canonical = canonical_country_iso(iso)
        country = self._country_record(canonical)
        if country is None:
            raise GuidedUnavailable("国家目录条目缺失")
        try:
            place = self._store().get_place("country", canonical)
        except Exception as exc:
            raise self._unavailable(exc) from exc
        if place is None:
            raise GuidedUnavailable("国家目录条目缺失")
        coordinate = self._coordinate_fields(place)
        return {
            "level": "country",
            "id": canonical,
            "label": str(country.get("name_zh") or country.get("name") or canonical),
            "country_iso": canonical,
            "source_iso": None,
            "admin1_id": None,
            "city_id": None,
            "lat": coordinate["lat"] if coordinate["lat"] is not None else (
                country.get("lat") if isinstance(country.get("lat"), (int, float)) else None
            ),
            "lng": coordinate["lng"] if coordinate["lng"] is not None else (
                country.get("lng") if isinstance(country.get("lng"), (int, float)) else None
            ),
            **coordinate["provenance"],
        }

    @staticmethod
    def _city_row_from_place(place: Any) -> CityRow:
        if place is None or place.lat is None or place.lng is None or place.population is None:
            raise GuidedUnavailable("城市目录条目不可用")
        return CityRow(
            city_id=int(place.place_id),
            iso=str(place.source_iso),
            name=str(place.name),
            region=str(place.code or ""),
            lat=float(place.lat),
            lng=float(place.lng),
            population=int(place.population),
        )

    def _city_rows_for_source(self, source_iso: str) -> dict[int, CityRow]:
        source = normalize_iso(source_iso)
        if not source:
            return {}
        try:
            return dict(self._store().city_rows(source))
        except Exception as exc:
            raise self._unavailable(exc) from exc

    def _city_rows_for_scope(self, source_iso: str, admin1_id: str | None) -> dict[int, CityRow]:
        source = normalize_iso(source_iso)
        if not source:
            return {}
        try:
            return dict(self._store().city_rows(source, admin1_id))
        except Exception as exc:
            raise self._unavailable(exc) from exc

    def _city_row_by_id(self, city_id: str) -> CityRow | None:
        try:
            place = self._store().get_place("city", city_id)
        except Exception as exc:
            raise self._unavailable(exc) from exc
        return self._city_row_from_place(place) if place is not None else None

    def _city_target(self, row: CityRow) -> dict[str, Any]:
        canonical = canonical_country_iso(row.iso)
        admin1_id = f"{row.iso}.{row.region}" if row.region else None
        try:
            place = self._store().get_place("city", str(row.city_id))
        except Exception as exc:
            raise self._unavailable(exc) from exc
        if place is None:
            raise GuidedUnavailable("城市目录条目缺失")
        coordinate = self._coordinate_fields(place)
        return {
            "level": "city",
            "id": str(row.city_id),
            "label": city_name_zh(
                row.city_id,
                str(getattr(place, "name", "") or row.name),
            ),
            "country_iso": canonical,
            "source_iso": row.iso,
            "admin1_id": admin1_id,
            "city_id": str(row.city_id),
            "lat": coordinate["lat"] if coordinate["lat"] is not None else row.lat,
            "lng": coordinate["lng"] if coordinate["lng"] is not None else row.lng,
            **coordinate["provenance"],
        }

    # --------------------------------------------------------------- evidence
    @staticmethod
    def _session_evidence_fingerprint(session: GeoSession) -> str:
        payload = {
            "clues": session.clues or [],
            "model_evidence": session.model_evidence,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _session_review_records(session: GeoSession) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        occurrences: dict[str, int] = {}
        for item in session.clues or []:
            raw = _raw_clue_dict(item)
            if raw is None:
                continue
            identity = json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str)
            occurrence = occurrences.get(identity, 0)
            occurrences[identity] = occurrence + 1
            record = _reviewed_record(raw, occurrence)
            record["_origin"] = "session"
            records.append(record)
        return records

    def _reconcile_session_evidence(
        self,
        state: dict[str, Any],
        session: GeoSession,
    ) -> tuple[dict[str, Any], bool]:
        """Merge authoritative session evidence without losing review status."""

        fingerprint = self._session_evidence_fingerprint(session)
        existing = {
            str(record.get("id")): dict(record)
            for record in state.get("clues") or []
            if isinstance(record, Mapping) and record.get("id")
        }
        reconciled_session: list[dict[str, Any]] = []
        session_ids: set[str] = set()
        for record in self._session_review_records(session):
            clue_id = str(record["id"])
            previous = existing.get(clue_id)
            if previous is not None:
                preserved = dict(record)
                preserved["status"] = str(previous.get("status") or "pending")
                preserved["_origin"] = "session"
                previous_extra = previous.get("_extra")
                if is_user_confirmed_observation(previous_extra):
                    # Immutable machine evidence is rebuilt from the session,
                    # but user-confirmed observation provenance belongs to the
                    # review layer and must survive later reconciliation.
                    preserved["_extra"] = dict(previous_extra)
                reconciled_session.append(preserved)
            else:
                reconciled_session.append(record)
            session_ids.add(clue_id)
        guided_only = [
            dict(record)
            for record in state.get("clues") or []
            if isinstance(record, Mapping)
            and record.get("id")
            and str(record.get("_origin") or "") != "session"
        ]
        clues = [*reconciled_session, *guided_only]
        pending_clue_ids = {
            str(record.get("id"))
            for record in clues
            if isinstance(record, Mapping)
            and record.get("id")
            and str(record.get("status") or "pending") == "pending"
        }
        stale_review_ids = {f"review:{clue_id}" for clue_id in pending_clue_ids}
        asked_question_ids = list(state.get("asked_question_ids") or [])
        pruned_asked_question_ids = [
            item for item in asked_question_ids if str(item) not in stale_review_ids
        ]
        asked_reviews_changed = pruned_asked_question_ids != asked_question_ids
        changed = (
            str(state.get("session_evidence_fingerprint") or "") != fingerprint
            or list(state.get("clues") or []) != clues
            or asked_reviews_changed
        )
        state["clues"] = clues
        state["session_evidence_fingerprint"] = fingerprint
        if asked_reviews_changed:
            state["asked_question_ids"] = pruned_asked_question_ids
        if changed:
            state["pending_question"] = None
        return state, changed

    def _ranking_clues(self, state: Mapping[str, Any]) -> list[Clue]:
        result: list[Clue] = []
        for record in state.get("clues") or []:
            if not isinstance(record, Mapping):
                continue
            if str(record.get("status") or "") == "rejected":
                continue
            clue = _reviewed_to_clue(record)
            if clue is not None:
                result.append(clue)
        return result

    def _country_scores(
        self,
        state: Mapping[str, Any],
        session: GeoSession,
        clues: Sequence[Clue],
    ) -> tuple[list[Any], str | None, dict[str, Any] | None]:
        engine = ModelAwareEngine()
        evidence = session.model_evidence
        validated_evidence = None
        validation_failed = False
        if evidence is not None:
            try:
                validated_evidence = self.gateway.validate(evidence)
                engine.evidence = validated_evidence
            except Exception:
                engine.evidence = None
                validation_failed = True
        guided_country_exclusions = [
            str(target.get("id"))
            for target in state.get("exclusions") or []
            if isinstance(target, Mapping) and target.get("level") == "country"
        ]
        exclusions = list(session.exclusions or []) + guided_country_exclusions
        scores = engine.infer(clues, exclusions=exclusions, top_n=COUNTRY_CANDIDATE_LIMIT)
        notice = build_inference_notice(
            self.settings,
            evidence,
            validation_failed=validation_failed,
            has_trusted_evidence=bool(scores),
        )
        return scores, notice, validated_evidence

    # ------------------------------------------------------------- candidates
    @staticmethod
    def _candidate(target: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        result = {"target": dict(target)}
        for key in ("score", "score_type", "reason", "selection_basis", "place_match"):
            value = kwargs.get(key)
            if value is not None:
                result[key] = value
        return result

    @staticmethod
    def _merge_candidates(
        primary: Sequence[Mapping[str, Any]],
        secondary: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in [*primary, *secondary]:
            target = item.get("target") if isinstance(item, Mapping) else None
            if not isinstance(target, Mapping):
                continue
            key = _target_key(target)
            if key in seen:
                continue
            seen.add(key)
            result.append(dict(item))
        return result

    def _country_candidates(
        self,
        state: Mapping[str, Any],
        scores: Sequence[Any],
        *,
        query: str | None,
        allow_catalog_fallback: bool = True,
    ) -> list[dict[str, Any]]:
        query_key = normalize_place_text(query or "")
        result: list[dict[str, Any]] = []
        seen: set[str] = set()

        def excluded(iso: str) -> bool:
            if is_excluded(state.get("exclusions") or [], iso):
                return True
            for target in state.get("exclusions") or []:
                if isinstance(target, Mapping) and target.get("level") == "country" and target.get("id") == iso:
                    return True
            return False

        for score in scores:
            iso = canonical_country_iso(getattr(score, "iso", ""))
            if not iso or iso in seen or excluded(iso):
                continue
            target = self._country_target(iso)
            country = self._country_record(iso) or {}
            haystack = " ".join(
                str(part) for part in (country.get("name"), country.get("name_zh"), iso) if part
            )
            if query_key and query_key not in normalize_place_text(haystack):
                continue
            payload = score.to_dict()
            result.append(self._candidate(
                target,
                score=payload.get("score"),
                score_type=payload.get("score_type"),
                reason="国家模型或规则证据",
            ))
            seen.add(iso)

        # Catalog-only fallback keeps no-provider operation usable and performs
        # query filtering over the complete country directory before any limit.
        # A degraded model session with no trusted text evidence must instead
        # produce an empty candidate list and request a usable clue.
        catalog: list[dict[str, Any]] = []
        if not allow_catalog_fallback:
            return self._merge_candidates(result, [])
        for country in self._country_records():
            iso = canonical_country_iso(country.get("iso"))
            if not iso or iso in seen or excluded(iso):
                continue
            target = self._country_target(iso)
            haystack = " ".join(
                part for part in (
                    str(country.get("name") or ""),
                    str(country.get("name_zh") or ""),
                    iso,
                ) if part
            )
            if query_key and query_key not in normalize_place_text(haystack):
                continue
            catalog.append(self._candidate(target, reason="国家目录"))
            seen.add(iso)
        catalog.sort(key=lambda item: (str(item["target"].get("label") or ""), str(item["target"].get("id") or "")))
        result.extend(catalog[:COUNTRY_CATALOG_FALLBACK_LIMIT])
        if query_key:
            result.sort(key=lambda item: (
                normalize_place_text(item["target"].get("label")) != query_key,
                str(item["target"].get("id") or ""),
            ))
        return self._merge_candidates(result, [])

    def _country_source_scopes(self, state: Mapping[str, Any]) -> tuple[str, ...]:
        confirmed = state.get("confirmed") or {}
        country = confirmed.get("country") if isinstance(confirmed, Mapping) else None
        if not isinstance(country, Mapping):
            return ()
        raw = normalize_iso(country.get("id"))
        canonical = canonical_country_iso(raw)
        if not canonical:
            return ()
        # Persisted raw TW/HK/MO keeps its narrow source identity. A newly
        # confirmed country is canonical CN, which intentionally fans out to all
        # available member scopes (CN/HK/MO/TW).
        if raw and source_region_iso(raw):
            return (raw,)
        return country_members(canonical)

    def _source_excluded(
        self,
        state: Mapping[str, Any],
        session: GeoSession | None,
        source_iso: str,
    ) -> bool:
        guided = [
            str(target.get("id"))
            for target in state.get("exclusions") or []
            if isinstance(target, Mapping) and target.get("level") == "country"
        ]
        persisted = list(session.exclusions or []) if session is not None else []
        return is_excluded([*persisted, *guided], source_iso)

    def _city_rows_for_country(
        self,
        state: Mapping[str, Any],
        session: GeoSession | None = None,
    ) -> dict[int, CityRow]:
        result: dict[int, CityRow] = {}
        for source_iso in self._country_source_scopes(state):
            if self._source_excluded(state, session, source_iso):
                continue
            for city_id, row in self._city_rows_for_source(source_iso).items():
                result[city_id] = row
        return result

    def _catalog_city_search_targets(
        self,
        state: Mapping[str, Any],
        session: GeoSession | None,
        text: str,
        *,
        limit: int = CITY_SEARCH_LIMIT,
    ) -> list[dict[str, Any]]:
        term = str(text or "").strip()
        scopes = self._country_source_scopes(state)
        if not term or not scopes:
            return []
        canonical = canonical_country_iso(scopes[0])
        found: list[tuple[CityRow, dict[str, Any]]] = []
        seen: set[tuple[str, str]] = set()
        try:
            store = self._store()
            for source_iso in scopes:
                if self._source_excluded(state, session, source_iso):
                    continue
                page = store.query_places(
                    "city",
                    country=canonical,
                    source_iso=source_iso,
                    q=term,
                    limit=limit,
                    coordinates_only=True,
                )
                for place in page.get("rows") or []:
                    try:
                        row = self._city_row_from_place(place)
                    except GuidedUnavailable:
                        continue
                    target = self._city_target(row)
                    key = _target_key(target)
                    if key in seen or self._excluded_by_state(state, target) is not None:
                        continue
                    seen.add(key)
                    found.append((row, target))
        except GuidedUnavailable:
            raise
        except Exception as exc:
            raise self._unavailable(exc) from exc

        normalized = normalize_place_text(term)

        def order(item: tuple[CityRow, dict[str, Any]]) -> tuple[Any, ...]:
            row, target = item
            match_labels = {
                normalize_place_text(row.name),
                normalize_place_text(str(target.get("label") or "")),
            }
            return (
                normalized not in match_labels,
                -row.population,
                row.city_id,
            )

        found.sort(key=order)
        return [target for _row, target in found]

    def _cached_city_ranking(
        self,
        image_path: Path,
        source_iso: str,
        cache_key: CityRankingKey | None,
        *,
        city_ids: tuple[int, ...] | None = None,
    ) -> Mapping[str, Any] | None:
        """Load one validated raw city payload without holding locks over I/O."""

        requested_city_ids = tuple(city_ids or ())
        if cache_key is not None and cache_key.city_ids != requested_city_ids:
            # Ordinary and targeted requests are distinct provider operations.
            cache_key = replace(cache_key, city_ids=requested_city_ids)
        if cache_key is not None:
            hit, cached = CITY_RANKING_CACHE.get(cache_key)
            if hit:
                return cached
        try:
            if requested_city_ids:
                result = self.gateway.rank_cities(
                    image_path,
                    source_iso,
                    city_ids=requested_city_ids,
                )
            else:
                # Keep old fake gateways compatible on the default path.
                result = self.gateway.rank_cities(image_path, source_iso)
        except Exception:
            result = None
        if cache_key is not None:
            if isinstance(result, Mapping) and result:
                CITY_RANKING_CACHE.put_success(cache_key, result)
            else:
                # A short-lived miss prevents repeated timeout waits, without
                # making a provider outage permanent.
                CITY_RANKING_CACHE.put_failure(cache_key)
        return result if isinstance(result, Mapping) else None

    def _ranked_city_rows(
        self,
        state: Mapping[str, Any],
        session: GeoSession,
        clues: Sequence[Clue],
    ) -> dict[str, Mapping[str, Any]]:
        image_path = Path(session.image_path)
        if not image_path.exists():
            return {}
        fingerprint = CITY_RANKING_CACHE.image_fingerprint(image_path)
        catalog_revision = ""
        endpoint = ""
        if fingerprint is not None:
            try:
                catalog_revision = str(self._store().revisions().get("cities") or "")
            except Exception:
                catalog_revision = ""
            endpoint = (
                str(self.settings.streetclip_url or "").rstrip("/") + "/predict-cities"
            )
        merged: dict[str, Mapping[str, Any]] = {}
        for source_iso in self._country_source_scopes(state):
            if self._source_excluded(state, session, source_iso):
                continue
            cache_key = (
                CityRankingKey(
                    image_sha256=fingerprint.sha256,
                    image_size=fingerprint.size,
                    image_mtime_ns=fingerprint.mtime_ns,
                    model_revision=MODEL_REVISION,
                    prompt_revision=PROMPT_VERSION,
                    catalog_revision=catalog_revision,
                    source_iso=normalize_iso(source_iso) or str(source_iso),
                    endpoint=endpoint,
                )
                if fingerprint is not None and catalog_revision and endpoint
                else None
            )
            result = self._cached_city_ranking(image_path, source_iso, cache_key)
            rows = [
                dict(row)
                for row in (result.get("city_scores") if result else []) or []
                if isinstance(row, Mapping)
            ]
            scoped_rows = self._city_rows_for_source(source_iso)
            scoped_evidence = [
                item for item in self._place_city_evidence(state)
                if normalize_iso(item.source_iso) == source_iso
                and item.city_id in scoped_rows
            ]

            raw_scored_ids: set[int] = set()
            for row in rows:
                try:
                    city_id = int(str(row.get("city_id") or ""))
                    score = float(row.get("score"))
                except (TypeError, ValueError):
                    continue
                if (
                    city_id in scoped_rows
                    and normalize_iso(row.get("iso")) == source_iso
                    and math.isfinite(score)
                ):
                    raw_scored_ids.add(city_id)

            best_missing_evidence: dict[int, tuple[float, CityEvidence]] = {}
            for item in scoped_evidence:
                if item.city_id in raw_scored_ids:
                    continue
                try:
                    confidence = float(item.confidence)
                except (TypeError, ValueError):
                    confidence = 0.0
                if not math.isfinite(confidence):
                    confidence = 0.0
                current = best_missing_evidence.get(item.city_id)
                if current is None or confidence > current[0]:
                    best_missing_evidence[item.city_id] = (confidence, item)

            target_ids = tuple(
                city_id
                for city_id, _ in sorted(
                    best_missing_evidence.items(),
                    key=lambda item: (-item[1][0], item[0]),
                )[:MAX_TARGET_CITY_IDS]
            )
            if target_ids:
                targeted_key = (
                    replace(cache_key, city_ids=target_ids)
                    if cache_key is not None
                    else None
                )
                targeted = self._cached_city_ranking(
                    image_path,
                    source_iso,
                    targeted_key,
                    city_ids=target_ids,
                )
                seen_ids = {str(row.get("city_id") or "") for row in rows}
                for row in (targeted.get("city_scores") if targeted else []) or []:
                    if not isinstance(row, Mapping):
                        continue
                    city_id = str(row.get("city_id") or "")
                    try:
                        city_key = int(city_id)
                    except (TypeError, ValueError):
                        continue
                    if city_id in seen_ids or city_key not in scoped_rows:
                        continue
                    if normalize_iso(row.get("iso")) != source_iso:
                        continue
                    rows.append(dict(row))
                    seen_ids.add(city_id)

            if not rows:
                continue
            if scoped_rows and (clues or scoped_evidence) and rows:
                try:
                    reranked = CityClueReasoner({source_iso: scoped_rows}).rerank(
                        rows,
                        clues,
                        iso=source_iso,
                        city_evidence=scoped_evidence,
                    )
                    if reranked is not None:
                        rows = list(reranked.city_scores)
                except (KeyError, TypeError, ValueError):
                    pass
            for row in rows:
                city_id = str(row.get("city_id") or "")
                try:
                    city_key = int(city_id)
                except (TypeError, ValueError):
                    continue
                if city_key not in scoped_rows:
                    continue
                if normalize_iso(row.get("iso")) != source_iso:
                    continue
                current = merged.get(city_id)
                if current is None:
                    merged[city_id] = row
                    continue
                try:
                    current_score = float(current.get("score"))
                except (TypeError, ValueError):
                    current_score = -1.0
                try:
                    row_score = float(row.get("score"))
                except (TypeError, ValueError):
                    row_score = -1.0
                if row_score > current_score:
                    merged[city_id] = row
        return merged

    def _city_candidates(
        self,
        state: Mapping[str, Any],
        session: GeoSession,
        clues: Sequence[Clue],
        *,
        query: str | None,
        proposals: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        rows = self._city_rows_for_country(state, session)
        if not rows:
            return []
        ranks = self._ranked_city_rows(state, session, clues)
        query_key = normalize_place_text(query or "")
        country_scopes = set(self._country_source_scopes(state))
        canonical = canonical_country_iso(next(iter(country_scopes), ""))
        candidates: dict[tuple[str, str], dict[str, Any]] = {}

        def add(target: Mapping[str, Any], **kwargs: Any) -> None:
            if normalize_iso(target.get("source_iso")) not in country_scopes:
                return
            if canonical_country_iso(target.get("country_iso")) != canonical:
                return
            if self._excluded_by_state(state, target) is not None:
                return
            key = _target_key(target)
            existing = candidates.get(key)
            if existing is None:
                candidates[key] = self._candidate(target, **kwargs)
                return
            for field in ("score", "score_type", "reason", "selection_basis", "place_match"):
                value = kwargs.get(field)
                if value is not None and (
                    existing.get(field) is None
                    or (field == "reason" and kwargs.get("score") is not None)
                ):
                    existing[field] = value

        # Explicit proposal provenance is retained, but real scores determine
        # city-stage ordering. An unscored proposal is displayed after scored
        # rows rather than being allowed to masquerade above the model.
        for raw in proposals:
            if not isinstance(raw, Mapping) or raw.get("level") != "city":
                continue
            row = self._city_row_by_id(str(raw.get("id") or ""))
            if row is None or row.iso not in country_scopes:
                continue
            add(
                self._city_target(row),
                reason=self._proposal_reason(state, row),
                selection_basis="user_text",
                place_match=self._place_match_from_meta(state, row),
            )

        if query_key:
            for target in self._catalog_city_search_targets(state, session, query or ""):
                add(
                    target,
                    reason="全国城市目录检索",
                    selection_basis="catalog_search",
                )

        evidence_city_ids = {
            str(item.city_id)
            for item in self._place_city_evidence(state)
            if normalize_iso(item.source_iso) in country_scopes
            and item.city_id in rows
        }
        for city_id, ranked in ranks.items():
            try:
                row = rows[int(city_id)]
            except (KeyError, TypeError, ValueError):
                continue
            add(
                self._city_target(row),
                score=ranked.get("score"),
                score_type=ranked.get("score_type"),
                reason=(
                    "图片+文字地点证据"
                    if (
                        str(city_id) in evidence_city_ids
                        or ranked.get("matched_clues")
                        or ranked.get("fusion") is not None
                    )
                    else "图片城市排序"
                ),
            )

        fallback = sorted(
            rows.values(),
            key=lambda row: (-row.population, row.city_id),
        )[: CITY_CANDIDATE_LIMIT * 2]
        for row in fallback:
            add(self._city_target(row), reason="城市目录")

        def sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
            target = item.get("target") or {}
            label_key = normalize_place_text(str(target.get("label") or ""))
            try:
                row = rows[int(str(target.get("id") or ""))]
            except (KeyError, TypeError, ValueError):
                row = None
            original_name_key = normalize_place_text(row.name) if row is not None else ""
            basis = str(item.get("selection_basis") or "")
            basis_rank = 0 if basis == "user_text" else 1 if basis == "catalog_search" else 2
            # Display labels are localized, while exact matching must continue
            # to recognize the original catalog/model name.
            exact = (
                0 if query_key and query_key in {label_key, original_name_key}
                else 1
            )
            try:
                raw_score = float(item.get("score"))
            except (TypeError, ValueError):
                raw_score = None
            if raw_score is not None and not math.isfinite(raw_score):
                raw_score = None
            meta = (state.get("proposal_meta") or {}).get(
                f"city:{target.get('id')}"
            )
            ambiguous = bool(
                isinstance(meta, Mapping) and meta.get("ambiguous")
            )
            population = 0 if ambiguous or row is None else -row.population
            label_order = label_key if ambiguous else ""
            return (
                0 if raw_score is not None else 1,
                -raw_score if raw_score is not None else 0.0,
                basis_rank,
                exact,
                0 if ambiguous else 1,
                label_order, population, str(target.get("id") or ""),
            )

        ordered = sorted(candidates.values(), key=sort_key)
        return ordered[:CITY_CANDIDATE_LIMIT]

    # --------------------------------------------------------------- target
    def _resolve_target(self, raw: GuidedEventTarget | Mapping[str, Any]) -> dict[str, Any]:
        level = str(getattr(raw, "level", None) or (raw.get("level") if isinstance(raw, Mapping) else "") or "")
        identifier = str(getattr(raw, "id", None) or (raw.get("id") if isinstance(raw, Mapping) else "") or "").strip()
        if level == "country":
            iso = normalize_iso(identifier)
            canonical = canonical_country_iso(iso)
            if not canonical or not self._country_exists(canonical):
                raise GuidedValidationError("未知国家 ID")
            return self._country_target(canonical)
        if level == "admin1":
            admin1_id = identifier.upper()
            match = _ADMIN1_ID_RE.fullmatch(admin1_id)
            if not match:
                raise GuidedValidationError("一级行政区 ID 格式无效")
            row = self._admin_by_id(admin1_id)
            if row is None:
                raise GuidedValidationError("未知一级行政区 ID")
            return self._admin1_target(row)
        if level == "city":
            if not _CITY_ID_RE.fullmatch(identifier):
                raise GuidedValidationError("城市 ID 格式无效")
            row = self._city_row_by_id(identifier)
            if row is None:
                raise GuidedValidationError("未知城市 ID")
            return self._city_target(row)
        raise GuidedValidationError("未知地点层级")

    def _validate_confirmation(self, state: Mapping[str, Any], target: Mapping[str, Any]) -> None:
        level = str(target.get("level") or "")
        confirmed = state.get("confirmed") or {}
        country = confirmed.get("country") if isinstance(confirmed, Mapping) else None
        if level == "country":
            current = canonical_country_iso(country.get("id")) if isinstance(country, Mapping) else ""
            if current and current != canonical_country_iso(target.get("id")):
                raise GuidedValidationError("确认其他国家前请先回退")
            return
        if level == "admin1":
            raise GuidedValidationError("已取消一级行政区确认，请直接确认城市")
        if not isinstance(country, Mapping):
            raise GuidedValidationError("请先确认国家，再确认城市")
        country_iso = canonical_country_iso(country.get("id"))
        target_country = canonical_country_iso(target.get("country_iso"))
        if not country_iso or target_country != country_iso:
            raise GuidedValidationError("目标国家与已确认国家不一致")
        if level != "city":
            raise GuidedValidationError("未知地点层级")

    def _validate_active_scope(self, state: Mapping[str, Any], target: Mapping[str, Any]) -> None:
        """Reject exclusions from outside the confirmed canonical country."""

        level = str(target.get("level") or "")
        confirmed = state.get("confirmed") or {}
        country = confirmed.get("country") if isinstance(confirmed, Mapping) else None
        if not isinstance(country, Mapping):
            if level != "country":
                raise GuidedValidationError("尚未确认国家，不能排除下级地点")
            return
        country_iso = canonical_country_iso(country.get("id"))
        target_country = canonical_country_iso(target.get("country_iso"))
        if not country_iso or target_country != country_iso:
            raise GuidedValidationError("排除地点不属于当前已确认国家")

    @staticmethod
    def _excluded_by_state(
        state: Mapping[str, Any],
        target: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        for exclusion in state.get("exclusions") or []:
            if not isinstance(exclusion, Mapping):
                continue
            level = str(exclusion.get("level") or "")
            if level == "country":
                excluded_iso = normalize_iso(exclusion.get("id"))
                if excluded_iso == canonical_country_iso(target.get("country_iso")):
                    return dict(exclusion)
                if source_region_iso(excluded_iso) and normalize_iso(target.get("source_iso")) == excluded_iso:
                    return dict(exclusion)
            if level == "admin1":
                if target.get("level") == "admin1" and target.get("id") == exclusion.get("id"):
                    return dict(exclusion)
                if target.get("level") == "city" and target.get("admin1_id") == exclusion.get("admin1_id"):
                    return dict(exclusion)
            if level == "city" and target.get("level") == "city" and target.get("id") == exclusion.get("id"):
                return dict(exclusion)
        return None

    @staticmethod
    def _stage_for_state(state: Mapping[str, Any]) -> str:
        confirmed = state.get("confirmed") or {}
        if isinstance(confirmed, Mapping) and isinstance(confirmed.get("city"), Mapping):
            return "resolved"
        if isinstance(confirmed, Mapping) and isinstance(confirmed.get("country"), Mapping):
            return "city"
        return "country"

    @staticmethod
    def _legacy_fields(state: Mapping[str, Any]) -> tuple[str | None, str]:
        confirmed = state.get("confirmed") or {}
        country = confirmed.get("country") if isinstance(confirmed, Mapping) else None
        city = confirmed.get("city") if isinstance(confirmed, Mapping) else None
        if not isinstance(country, Mapping):
            return None, "active"
        return str(country.get("id")), ("done" if isinstance(city, Mapping) else "active")

    # ---------------------------------------------------------------- prepare
    @staticmethod
    def _route_blocked(state: Mapping[str, Any]) -> bool:
        """Keep correction/conflict/pinned re-review ahead of confidence routing."""

        if isinstance(state.get("pending_correction"), Mapping):
            return True
        if isinstance(state.get("conflict"), Mapping):
            return True
        pending = state.get("pending_question")
        if not isinstance(pending, Mapping):
            return False
        if pending.get("_pinned") is True:
            return True
        return bool(pending.get("correction_id") or pending.get("conflict_id"))

    @staticmethod
    def _asked_clue_history(state: Mapping[str, Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for item in state.get("asked_clue_questions") or []:
            if not isinstance(item, Mapping):
                continue
            question_id = str(item.get("id") or "")
            if not question_id:
                continue
            result.append({
                "id": question_id,
                "dimension": str(item.get("dimension") or ""),
            })
        return result

    @staticmethod
    def _pending_low_clue_question(state: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if isinstance(state.get("pending_correction"), Mapping) or isinstance(state.get("conflict"), Mapping):
            return None
        pending = state.get("pending_question")
        if not isinstance(pending, Mapping):
            return None
        if not str(pending.get("id") or "").startswith("clue:country:"):
            return None
        if pending.get("correction_id") or pending.get("conflict_id"):
            return None
        return pending

    @staticmethod
    def _uncertain_free_text(text: str) -> bool:
        lowered = text.strip().lower()
        return any(token in lowered for token in (
            "不知道", "不确定", "看不清", "看不见", "无法判断", "无法确定",
            "说不清", "不清楚", "拿不准", "没法判断", "猜不出", "辨不清",
        ))

    @classmethod
    def _record_clue_question(
        cls,
        state: dict[str, Any],
        question: Mapping[str, Any],
    ) -> None:
        question_id = str(question.get("id") or "")
        if not question_id.startswith("clue:country:"):
            return
        history = cls._asked_clue_history(state)
        if not any(item.get("id") == question_id for item in history):
            history.append({
                "id": question_id,
                "dimension": str(question.get("dimension") or ""),
            })
            state["asked_clue_questions"] = history
        asked = [str(item) for item in state.get("asked_question_ids") or []]
        if question_id not in asked:
            asked.append(question_id)
            state["asked_question_ids"] = asked

    @staticmethod
    def _evidence_rationale(clues: Sequence[Clue], top_iso: str | None) -> str:
        """Label only trusted clues that actually matched the predicted country."""

        engine = ModelAwareEngine()
        trusted = engine._trusted_clues(list(clues))
        if not trusted or not top_iso:
            return "全图模型与线索融合首选一致"
        country = next(
            (item for item in engine.countries if item.get("iso") == top_iso),
            None,
        )
        if country is None:
            return "全图模型与线索融合首选一致"
        scored = engine._score_country(country, trusted, [])
        labels: list[str] = []
        for item in (scored.matched_clues if scored is not None else []):
            if not isinstance(item, Mapping):
                continue
            raw_value = str(item.get("value") or "")
            label = _value_label(raw_value)
            if str(item.get("dimension") or "") == "place_name" and raw_value:
                resolved_iso = engine.resolve_country_key(raw_value)
                resolved = engine._country_by_iso.get(resolved_iso or "")
                if resolved:
                    label = str(resolved.get("name_zh") or resolved.get("name") or label)
            if label and label not in labels:
                labels.append(label)
            if len(labels) >= 3:
                break
        return "可信线索：" + "、".join(labels) if labels else "全图模型与线索融合首选一致"

    @staticmethod
    def _guided_options(options: Any) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in options or []:
            if not isinstance(item, Mapping):
                continue
            value = str(item.get("value") or "").strip()[:200]
            label = str(item.get("label") or value).strip()[:200]
            if not value or not label:
                continue
            key = (value, label)
            if key in seen:
                continue
            seen.add(key)
            result.append({"value": value, "label": label})
            if len(result) >= MAX_QUESTION_OPTIONS:
                break
        return result

    @staticmethod
    def _route_candidates(
        candidates: Sequence[Mapping[str, Any]],
        route: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        by_iso: dict[str, dict[str, Any]] = {}
        for item in candidates:
            target = item.get("target") if isinstance(item, Mapping) else None
            if not isinstance(target, Mapping):
                continue
            iso = canonical_country_iso(target.get("id") or target.get("country_iso"))
            if iso and iso not in by_iso:
                by_iso[iso] = dict(item)
        result: list[dict[str, Any]] = []
        for row in route.get("top3") or []:
            if not isinstance(row, Mapping):
                continue
            iso = canonical_country_iso(row.get("iso"))
            item = by_iso.get(iso)
            if item is not None and item not in result:
                result.append(item)
        return result

    @staticmethod
    def _confirm_question(
        candidates: Sequence[Mapping[str, Any]],
        *,
        low_uncertainty: bool = False,
        rationale: str = "",
    ) -> dict[str, Any] | None:
        ordered = list(candidates)[:3]
        if not ordered:
            return None
        target = ordered[0].get("target") if isinstance(ordered[0], Mapping) else None
        if not isinstance(target, Mapping):
            return None
        options: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in ordered:
            item_target = item.get("target") if isinstance(item, Mapping) else None
            if not isinstance(item_target, Mapping):
                continue
            value = str(item_target.get("id") or "").strip()
            label = str(item_target.get("label") or value).strip()
            if not value or value in seen:
                continue
            seen.add(value)
            options.append({"value": value, "label": label})
        reason = f"（{rationale}）" if rationale else ""
        if low_uncertainty:
            message = (
                "已询问 3 个线索问题，当前证据仍不足以自动判断国家，也不会自动猜测。"
                f"{reason}你可以从下方三个候选中手动选择一个并明确确认，或继续补充文字/重新上传。"
            )
        else:
            message = (
                f"候选最可能是「{target.get('label') or target.get('id')}」{reason}。"
                "你可以直接确认，也可从下方三个候选中选择；只有确认地点后才会进入下一步。"
            )
        question_id = _json_hash("confidence-confirm", {
            "targets": [option["value"] for option in options],
            "low": low_uncertainty,
        })
        return {
            "id": question_id,
            "kind": "confirm_location",
            "message": message,
            "target": dict(target),
            "options": options or None,
        }

    def _low_clue_question(
        self,
        state: Mapping[str, Any],
        scores: Sequence[Any],
        clues: Sequence[Clue],
        candidates: Sequence[Mapping[str, Any]],
        validated_evidence: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        history = self._asked_clue_history(state)
        if len(history) >= MAX_COUNTRY_CLUE_QUESTIONS:
            return None
        asked_dimensions = [item["dimension"] for item in history if item.get("dimension")]
        engine = ModelAwareEngine()
        engine.evidence = dict(validated_evidence) if isinstance(validated_evidence, Mapping) else None
        hypothesis = ModelAwareHypotheses(engine).next_action(
            list(scores),
            list(clues),
            asked=asked_dimensions,
        )
        payload = dict(hypothesis.payload or {})
        dimensions = [
            str(item) for item in payload.get("dimensions") or []
            if isinstance(item, str) and item
        ]
        dimension = dimensions[0] if dimensions else "place_name"
        options = self._guided_options(payload.get("options"))
        candidate_isos = [str(item) for item in payload.get("candidates") or [] if item]
        if not candidate_isos:
            for item in candidates:
                target = item.get("target") if isinstance(item, Mapping) else None
                if isinstance(target, Mapping) and target.get("id"):
                    candidate_isos.append(str(target.get("id")))
        if dimension == "place_name" and not options:
            place_options = []
            for item in candidates:
                target = item.get("target") if isinstance(item, Mapping) else None
                if isinstance(target, Mapping) and target.get("id"):
                    place_options.append({
                        "value": str(target.get("id")),
                        "label": str(target.get("label") or target.get("id")),
                    })
            options = self._guided_options(place_options)
        question_id = _json_hash("clue:country", {
            "index": len(history),
            "dimension": dimension,
            "candidates": candidate_isos[:3],
        })
        return {
            "id": question_id,
            "kind": "ask" if len(candidate_isos) > 1 else "request_clue",
            "message": str(hypothesis.message or f"请补充{dimension}线索。"),
            "dimension": dimension,
            "options": options or None,
        }

    def _maybe_narrate(
        self,
        prepared: _Prepared,
        session_id: int,
        *,
        feedback: str = "",
    ) -> _Prepared:
        """Narrate on POST only; never let narration alter action or state."""

        if self.provider is None:
            return prepared
        if str((prepared.confidence or {}).get("tier") or "") == "high":
            return prepared
        action = prepared.action.model_dump(mode="json")
        message = self.provider.narrate({
            "action": action,
            "scores": [score.to_dict() for score in prepared.scores],
            "clues": prepared.state.get("clues") or [],
            "history": self.geo.list_messages(session_id),
            "feedback": feedback,
        })
        if message and narration_matches_action(message, action):
            prepared.action = prepared.action.model_copy(update={"message": message})
        return prepared

    def _city_catalog_name(self, city_id: object, fallback: object = "") -> str:
        """Recover the authoritative original name for an old persisted label."""

        identifier = str(city_id or "").strip()
        fallback_name = str(fallback or "").strip()
        if not _CITY_ID_RE.fullmatch(identifier):
            return fallback_name
        try:
            place = self._store().get_place("city", identifier)
        except Exception as exc:
            raise self._unavailable(exc) from exc
        if place is None:
            return fallback_name
        return str(getattr(place, "name", "") or "").strip() or fallback_name

    def _localize_city_target(self, value: object) -> tuple[dict[str, Any] | None, str]:
        if not isinstance(value, Mapping) or value.get("level") != "city":
            return None, ""
        target = dict(value)
        old_label = str(target.get("label") or "")
        city_id = str(target.get("id") or target.get("city_id") or "")
        target["label"] = city_name_zh(
            city_id,
            self._city_catalog_name(city_id, old_label),
        )
        return target, old_label

    def _localize_city_labels(self, state: dict[str, Any]) -> None:
        """Localize legacy city display labels without changing identities."""

        confirmed = state.get("confirmed")
        if isinstance(confirmed, Mapping):
            city, _ = self._localize_city_target(confirmed.get("city"))
            if city is not None:
                updated = dict(confirmed)
                updated["city"] = city
                state["confirmed"] = updated

        pending = state.get("pending_question")
        if isinstance(pending, Mapping):
            question = dict(pending)
            target, old_label = self._localize_city_target(question.get("target"))
            if target is not None:
                question["target"] = target
                new_label = str(target.get("label") or "")
                message = str(question.get("message") or "")
                if old_label and old_label != new_label and old_label in message:
                    question["message"] = message.replace(old_label, new_label)
            options = []
            for option in question.get("options") or []:
                if not isinstance(option, Mapping):
                    continue
                localized = dict(option)
                city_id = str(localized.get("value") or "")
                if _CITY_ID_RE.fullmatch(city_id):
                    localized["label"] = city_name_zh(
                        city_id,
                        self._city_catalog_name(city_id, localized.get("label")),
                    )
                options.append(localized)
            if options:
                question["options"] = options
            state["pending_question"] = question

        proposals = []
        for item in state.get("proposals") or []:
            target, _ = self._localize_city_target(item)
            proposals.append(target if target is not None else item)
        state["proposals"] = proposals

    def _prepare(
        self,
        state: dict[str, Any],
        session: GeoSession,
        *,
        query: str | None,
        ensure_question: bool = True,
    ) -> _Prepared:
        state = _normalise_state(state)
        self._localize_city_labels(state)
        clues = self._ranking_clues(state)
        scores, inference_notice, validated_evidence = self._country_scores(state, session, clues)
        stage = self._stage_for_state(state)
        route_blocked = stage == "country" and self._route_blocked(state)
        confidence = None
        if stage == "country" and validated_evidence is not None and not route_blocked:
            confidence = route_confidence(scores, validated_evidence).to_dict()
        proposals = [dict(item) for item in state.get("proposals") or [] if isinstance(item, Mapping)]
        country_candidates = self._country_candidates(
            state,
            scores,
            query=query,
            allow_catalog_fallback=bool(scores) or inference_notice is None,
        )
        if stage == "country":
            country_targets = [
                target for target in proposals
                if target.get("level") == "country"
                and self._excluded_by_state(state, target) is None
            ]
            country_targets.extend(self._resolved_country_targets(state))
            proposed_countries = [
                self._candidate(target, reason="自由文本或地名检索提议", selection_basis="user_text")
                for target in country_targets
            ]
            country_candidates = self._merge_candidates(proposed_countries, country_candidates)
        city_candidates = (
            self._city_candidates(state, session, clues, query=query, proposals=proposals)
            if stage == "city"
            else []
        )
        if inference_notice and not scores:
            pending = state.get("pending_question")
            if isinstance(pending, Mapping) and str(pending.get("kind") or "") == "confirm_location":
                state["pending_question"] = None
                state["asked_question_ids"] = [
                    item
                    for item in state.get("asked_question_ids") or []
                    if not str(item).startswith(f"request:{stage}")
                ]
                ensure_question = True

        question = None
        action = None
        state["confidence_confirm_isos"] = []
        rationale = (
            self._evidence_rationale(clues, str(confidence.get("top1_iso") or "") or None)
            if confidence is not None else ""
        )
        if confidence is not None and not route_blocked:
            routed_candidates = self._route_candidates(country_candidates, confidence)
            if not routed_candidates:
                routed_candidates = list(country_candidates[:3])
            confirm_isos = [
                str(item.get("iso")) for item in confidence.get("top3") or []
                if isinstance(item, Mapping) and item.get("iso")
            ]
            tier = str(confidence.get("tier") or "low")
            if tier == "high":
                state["pending_question"] = None
                state["confidence_confirm_isos"] = confirm_isos
                top = (confidence.get("top3") or [{}])[0]
                action = ActionModel(
                    kind="result",
                    message=(
                        f"高置信预测：{top.get('name_zh') or top.get('iso') or '候选国家'}。"
                        f"{rationale}。这是模型候选排序，不是已确认地点；"
                        "你可以直接确认，或选择其他候选继续定位。"
                    ),
                    payload={
                        "stage": stage,
                        "iso": top.get("iso"),
                        "name_zh": top.get("name_zh"),
                        "score": top.get("score"),
                        "evidence_rationale": rationale,
                        "requires_explicit_confirmation": True,
                    },
                )
                country_candidates = routed_candidates
            elif tier == "medium":
                question = self._confirm_question(routed_candidates, rationale=rationale)
                if question is not None:
                    state["pending_question"] = question
                    state["confidence_confirm_isos"] = confirm_isos
                    action = self._action_for_state(state, question)
                    country_candidates = routed_candidates
            else:
                history = self._asked_clue_history(state)
                if len(history) >= MAX_COUNTRY_CLUE_QUESTIONS:
                    confidence["action"] = "confirm_candidates"
                    confidence["reason"] = "问题预算已用完；当前仍为低置信，只能由用户显式选择候选。"
                    question = self._confirm_question(
                        routed_candidates,
                        low_uncertainty=True,
                        rationale=rationale,
                    )
                    if question is not None:
                        state["pending_question"] = question
                        state["confidence_confirm_isos"] = confirm_isos
                        action = self._action_for_state(state, question)
                        country_candidates = routed_candidates
                    else:
                        state["pending_question"] = None
                        action = ActionModel(
                            kind="uncertain",
                            message=(
                                "当前证据不足，无法给出候选国家，也不会自动猜测。"
                                "请继续补充可辨认文字/地名，或重新上传图片。"
                            ),
                            payload={"stage": stage},
                        )
                else:
                    existing = state.get("pending_question")
                    if (
                        isinstance(existing, Mapping)
                        and str(existing.get("id") or "").startswith("clue:country:")
                        and self._question_is_valid(state, existing)
                    ):
                        question = dict(existing)
                    else:
                        question = self._low_clue_question(
                            state,
                            scores,
                            clues,
                            country_candidates,
                            validated_evidence,
                        )
                    if question is not None:
                        state["pending_question"] = question
                        action = self._action_for_state(state, question)

        if action is None:
            pending = state.get("pending_question")
            legacy_pending = (
                isinstance(pending, Mapping)
                and isinstance(pending.get("target"), Mapping)
                and pending["target"].get("level") == "admin1"
            )
            question = (
                self._ensure_question(state, country_candidates, city_candidates)
                if ensure_question or legacy_pending
                else pending
            )
            action = self._action_for_state(state, question)

        if confidence is not None:
            top3 = list(confidence.get("top3") or [])
            action = action.model_copy(
                update={
                    "payload": {
                        **dict(action.payload or {}),
                        "confidence": confidence,
                        "candidate_details": top3,
                        "candidates": [item.get("iso") for item in top3 if item.get("iso")],
                        "reason": confidence.get("reason"),
                        "evidence_rationale": rationale,
                        "requires_explicit_confirmation": True,
                    }
                }
            )
        if inference_notice:
            action = action.model_copy(
                update={"message": f"{inference_notice}\n{action.message}"}
            )
        legacy_iso, legacy_status = self._legacy_fields(state)
        return _Prepared(
            state=state,
            stage=stage,
            candidates=country_candidates if stage == "country" else city_candidates,
            scores=scores,
            action=action,
            confidence=confidence,
            legacy_confirmed_iso=legacy_iso,
            legacy_status=legacy_status,
            inference_notice=inference_notice,
        )

    # -------------------------------------------------------------- questions
    def _question_is_valid(self, state: Mapping[str, Any], question: Mapping[str, Any]) -> bool:
        kind = str(question.get("kind") or "")
        question_id = str(question.get("id") or "")
        if not question_id:
            return False
        if kind == "review_clue":
            clue_id = str(question.get("clue_id") or "")
            for record in state.get("clues") or []:
                if isinstance(record, Mapping) and str(record.get("id")) == clue_id:
                    return str(record.get("status") or "") == "pending"
            return False
        asked = {str(item) for item in state.get("asked_question_ids") or []}
        if question_id in asked:
            return False
        if kind == "confirm_location":
            target = question.get("target")
            if not isinstance(target, Mapping):
                return False
            stage = self._stage_for_state(state)
            if stage == "resolved":
                return False
            return str(target.get("level") or "") == stage and self._excluded_by_state(state, target) is None
        if kind in {"ask", "request_clue"}:
            correction = state.get("pending_correction")
            conflict = state.get("conflict")
            if kind == "ask" and question.get("correction_id"):
                return (
                    isinstance(correction, Mapping)
                    and str(correction.get("id") or "") == str(question.get("correction_id") or "")
                )
            if kind == "ask" and question.get("conflict_id"):
                return (
                    isinstance(conflict, Mapping)
                    and str(conflict.get("id") or "") == str(question.get("conflict_id") or "")
                )
            return True
        return False

    @staticmethod
    def _correction_question(plan: Mapping[str, Any]) -> dict[str, Any]:
        parts: list[str] = []
        reject_isos = plan.get("reject_isos") or []
        clear_exclusions = plan.get("clear_exclusions") or []
        replace_dimensions = plan.get("replace_dimensions") or []
        remove_clue_values = plan.get("remove_clue_values") or []
        if reject_isos:
            parts.append("排除：" + "、".join(str(item) for item in reject_isos))
        if clear_exclusions:
            parts.append("撤回排除：" + "、".join(str(item) for item in clear_exclusions))
        if replace_dimensions:
            parts.append(
                "替换线索维度："
                + "、".join(DIM_LABELS.get(str(item), str(item)) for item in replace_dimensions)
            )
        if remove_clue_values:
            parts.append("移除线索值：" + "、".join(
                f"{DIM_LABELS.get(str(item[0]), str(item[0]))}「{_value_label(item[1])}」"
                for item in remove_clue_values
                if isinstance(item, (list, tuple)) and len(item) == 2
            ))
        detail = "；".join(parts) or "文本中的修正"
        return {
            "id": f"correction:{plan.get('id')}",
            "kind": "ask",
            "message": f"检测到可能的修正或排除（{detail}）。是否应用？",
            "dimension": "correction",
            "correction_id": str(plan.get("id") or ""),
            "options": [
                {"value": "apply", "label": "应用修正"},
                {"value": "keep", "label": "保持原样"},
            ],
        }

    def _place_clarification_question(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        entries = self._active_resolution_entries(state)
        if not entries:
            return None
        latest = entries[-1]
        status = str(latest.get("status") or "")
        if status not in {"ambiguous", "unresolved", "unavailable", "foreign_conflict"}:
            return None
        question_id = f"place-clarify:{latest.get('id')}"
        asked = {str(item) for item in state.get("asked_question_ids") or []}
        if question_id in asked:
            return None
        message = str(latest.get("notice") or "").strip()
        if not message:
            if status == "foreign_conflict":
                message = "检索结果与当前国家不一致；请补充正确国家或回退后重新选择。"
            elif status == "unavailable":
                message = "地名检索暂不可用；请补充城市、区名或附近地标。"
            elif status == "ambiguous":
                message = "这个地点可能对应多个城市；请补充城市、区名或附近地标。"
            else:
                message = "暂未找到可核验城市；请补充城市、区名或附近地标。"
        return {
            "id": question_id,
            "kind": "request_clue",
            "message": message,
            "dimension": "place_name",
            "options": None,
        }

    def _ensure_question(
        self,
        state: dict[str, Any],
        country_candidates: Sequence[Mapping[str, Any]],
        city_candidates: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any] | None:
        stage = self._stage_for_state(state)
        if stage == "resolved":
            state["pending_question"] = None
            return None
        existing = state.get("pending_question")
        if isinstance(existing, Mapping) and self._question_is_valid(state, existing):
            return dict(existing)

        correction = state.get("pending_correction")
        if isinstance(correction, Mapping):
            question = self._correction_question(correction)
            if self._question_is_valid(state, question):
                state["pending_question"] = question
                return question

        conflict = state.get("conflict")
        if isinstance(conflict, Mapping):
            question = {
                "id": f"clarify:{conflict.get('id')}",
                "kind": "ask",
                "message": str(conflict.get("message") or "现有线索与已确认路径冲突，请先澄清；已确认路径不会自动改变。"),
                "dimension": "place_name",
                "conflict_id": str(conflict.get("id") or ""),
                "proposed": conflict.get("proposed"),
            }
            if self._question_is_valid(state, question):
                state["pending_question"] = question
                return question

        asked = {str(item) for item in state.get("asked_question_ids") or []}
        pending_clues = [
            dict(record)
            for record in state.get("clues") or []
            if isinstance(record, Mapping) and str(record.get("status") or "") == "pending"
        ]
        pending_clues.sort(key=lambda item: (
            _CLUE_PRIORITY.get(str(item.get("dimension") or ""), 99),
            -float(item.get("confidence") or 0.0),
            str(item.get("id") or ""),
        ))
        for record in pending_clues:
            clue_id = str(record.get("id") or "")
            if not clue_id:
                continue
            question = _review_question(record)
            state["pending_question"] = question
            return question

        place_question = self._place_clarification_question(state)
        if place_question is not None:
            state["pending_question"] = place_question
            return place_question

        candidate_sets = {
            "country": list(country_candidates),
            "city": list(city_candidates),
        }
        candidates = candidate_sets.get(stage, [])
        proposals = [
            dict(item) for item in state.get("proposals") or []
            if isinstance(item, Mapping) and item.get("level") == stage
        ]
        ordered_targets: list[dict[str, Any]] = list(proposals)
        for candidate in candidates:
            target = candidate.get("target") if isinstance(candidate, Mapping) else None
            if isinstance(target, Mapping):
                ordered_targets.append(dict(target))
        seen: set[tuple[str, str]] = set()
        for target in ordered_targets:
            key = _target_key(target)
            if key in seen:
                continue
            seen.add(key)
            if self._excluded_by_state(state, target) is not None:
                continue
            question_id = f"confirm:{stage}:{target.get('id')}"
            if question_id in asked:
                continue
            options = []
            option_keys: set[tuple[str, str]] = set()
            for item in candidates:
                item_target = item.get("target") if isinstance(item, Mapping) else None
                if isinstance(item_target, Mapping):
                    option_key = _target_key(item_target)
                    if option_key in option_keys:
                        continue
                    option_keys.add(option_key)
                    options.append({
                        "value": str(item_target.get("id")),
                        "label": str(item_target.get("label") or item_target.get("id")),
                    })
                if len(options) >= MAX_QUESTION_OPTIONS:
                    break
            question = {
                "id": question_id,
                "kind": "confirm_location",
                "message": f"是否确认「{target.get('label') or target.get('id')}」？确认后才会进入下一步，也可以改选其他候选。",
                "target": dict(target),
                "options": options or None,
            }
            state["pending_question"] = question
            return question

        request_ids = [f"request:{stage}", f"request:{stage}:2", f"request:{stage}:3"]
        for request_id in request_ids:
            if request_id in asked:
                continue
            question = {
                "id": request_id,
                "kind": "request_clue",
                "message": "请描述画面中可见的招牌、地名或文字线索；排序结果不是地点确认。",
                "dimension": "place_name",
                "options": None,
            }
            state["pending_question"] = question
            return question
        state["pending_question"] = None
        return None

    def _action_for_state(
        self,
        state: Mapping[str, Any],
        question: Mapping[str, Any] | None,
    ) -> ActionModel:
        stage = self._stage_for_state(state)
        if stage == "resolved":
            city = (state.get("confirmed") or {}).get("city")
            target = dict(city) if isinstance(city, Mapping) else {}
            return ActionModel(
                kind="done",
                message=f"已确认城市：{target.get('label') or target.get('id') or '未知'}。",
                payload={"stage": stage, "target": target},
            )
        if not isinstance(question, Mapping):
            return ActionModel(
                kind="request_clue",
                message="请描述画面中可见的地名或文字线索。",
                payload={"stage": stage},
            )
        kind_map = {
            "review_clue": "ask",
            "confirm_location": "confirm",
            "ask": "ask",
            "request_clue": "request_clue",
        }
        payload = {
            "stage": stage,
            "question_id": question.get("id"),
            "question_kind": question.get("kind"),
        }
        for key in ("dimension", "clue_id", "options", "target", "proposed", "correction_id"):
            value = question.get(key)
            if value is not None:
                payload[key] = value
        return ActionModel(
            kind=kind_map.get(str(question.get("kind") or ""), "request_clue"),
            message=str(question.get("message") or "请进一步说明。"),
            payload=payload,
        )

    # -------------------------------------------------------------- event apply
    def _apply_event(
        self,
        state: dict[str, Any],
        session: GeoSession,
        event: GuidedEventRequest,
        *,
        preflight: _FreeTextPreflight | None = None,
    ) -> dict[str, Any]:
        event_type = str(event.type)
        if event_type == "confirm_clue":
            if event.question_id is not None and event.clue_id is None:
                raise GuidedConflict("question_id 与线索不匹配")
            self._check_question_reference(
                state,
                event,
                allowed_kinds={"review_clue"},
                clue_id=event.clue_id,
            )
            return self._apply_clue_status(state, event.clue_id, "confirmed")
        if event_type == "reject_clue":
            if event.question_id is not None and event.clue_id is None:
                raise GuidedConflict("question_id 与线索不匹配")
            self._check_question_reference(
                state,
                event,
                allowed_kinds={"review_clue"},
                clue_id=event.clue_id,
            )
            return self._apply_clue_status(state, event.clue_id, "rejected")
        if event_type == "confirm_location":
            return self._apply_confirm_location(state, event)
        if event_type == "reject_location":
            return self._apply_reject_location(state, event)
        if event_type == "retract":
            return self._apply_retract(state, event)
        if event_type == "backtrack":
            return self._apply_backtrack(state, event)
        if event_type == "free_text":
            return self._apply_free_text(state, session, event, preflight=preflight)
        if event_type == "adopt_observation":
            return self._apply_adopted_observation(state, event, preflight=preflight)
        if event_type == "answer":
            return self._apply_answer(state, event)
        if event_type == "unknown":
            return self._apply_unknown(state, event)
        raise GuidedValidationError("未知事件类型")

    def _apply_clue_status(
        self,
        state: dict[str, Any],
        clue_id: str | None,
        status: str,
    ) -> dict[str, Any]:
        if not clue_id:
            raise GuidedValidationError("缺少线索 ID（clue_id）")
        for record in state.get("clues") or []:
            if not isinstance(record, Mapping) or str(record.get("id")) != str(clue_id):
                continue
            if str(record.get("status") or "pending") == status:
                return self._outcome(state, False, "unchanged", "该线索状态没有变化。")
            updated = dict(record)
            updated["status"] = status
            state["clues"] = [
                updated if isinstance(item, Mapping) and str(item.get("id")) == str(clue_id) else item
                for item in state["clues"]
            ]
            self._refresh_derived_proposals(state)
            state["pending_question"] = None
            self._refresh_conflict(state)
            label = "已确认" if status == "confirmed" else "已否定"
            return self._outcome(state, True, "applied", f"线索{label}。")
        raise GuidedValidationError("未知线索 ID")

    def _apply_confirm_location(
        self,
        state: dict[str, Any],
        event: GuidedEventRequest,
    ) -> dict[str, Any]:
        if event.target is None:
            raise GuidedValidationError("缺少目标地点")
        pending = state.get("pending_question")
        if isinstance(pending, Mapping) and str(pending.get("kind") or "") == "ask":
            raise GuidedConflict("请先处理当前澄清问题，再确认地点。")
        if isinstance(state.get("pending_correction"), Mapping) or isinstance(state.get("conflict"), Mapping):
            raise GuidedConflict("请先处理当前修正或冲突问题，再确认地点。")
        self._check_question_reference(
            state,
            event,
            allowed_kinds={"confirm_location"},
            target={"level": event.target.level, "id": event.target.id},
        )
        if self._has_pending_clue_review(state):
            confirm_iso = canonical_country_iso(event.target.id)
            routed_isos = {
                canonical_country_iso(item) for item in state.get("confidence_confirm_isos") or []
            }
            pinned_review = (
                isinstance(pending, Mapping)
                and pending.get("kind") == "review_clue"
                and pending.get("_pinned") is True
            )
            if event.target.level != "country" or not confirm_iso or confirm_iso not in routed_isos or pinned_review:
                raise GuidedConflict("请先完成当前线索核对，再确认地点。")
        if event.target.level == "admin1":
            raise GuidedValidationError("已取消一级行政区确认，请直接确认城市")
        target = self._resolve_target(event.target)
        self._validate_confirmation(state, target)
        excluded = self._excluded_by_state(state, target)
        if excluded is not None:
            raise GuidedValidationError("该地点已被排除，请先撤回排除再确认")
        level = str(target["level"])
        confirmed = state.get("confirmed") or {}
        current = confirmed.get(level) if isinstance(confirmed, Mapping) else None
        if isinstance(current, Mapping) and _target_key(current) == _target_key(target):
            return self._outcome(state, False, "unchanged", f"「{target.get('label')}」已确认。")
        new_confirmed = {
            "country": confirmed.get("country") if isinstance(confirmed, Mapping) else None,
            "admin1": confirmed.get("admin1") if isinstance(confirmed, Mapping) else None,
            "city": confirmed.get("city") if isinstance(confirmed, Mapping) else None,
        }
        if level == "country":
            new_confirmed["country"] = dict(target)
            new_confirmed["admin1"] = None
            new_confirmed["city"] = None
        elif level == "city":
            # Deprecated admin1 provenance is retained, never used as a parent
            # gate. A city outside that old metadata remains confirmable.
            new_confirmed["city"] = dict(target)
        else:
            raise GuidedValidationError("已取消一级行政区确认，请直接确认城市")
        state["confirmed"] = new_confirmed
        state["pending_question"] = None
        state["confidence_confirm_isos"] = []
        if level == "country":
            retained = []
            for item in state.get("proposals") or []:
                if not isinstance(item, Mapping) or item.get("level") != "city":
                    continue
                if canonical_country_iso(item.get("country_iso")) == canonical_country_iso(target.get("id")):
                    retained.append(dict(item))
            retained_keys = {_target_state_key(item) for item in retained}
            state["proposals"] = retained
            state["proposal_meta"] = {
                str(key): dict(value)
                for key, value in (state.get("proposal_meta") or {}).items()
                if str(key) in retained_keys and isinstance(value, Mapping)
            }
        else:
            state["proposals"] = []
            state["proposal_meta"] = {}
        state["pending_correction"] = None
        self._refresh_conflict(state)
        return self._outcome(state, True, "applied", f"已确认「{target.get('label')}」。")

    def _apply_reject_location(
        self,
        state: dict[str, Any],
        event: GuidedEventRequest,
    ) -> dict[str, Any]:
        if event.target is None:
            raise GuidedValidationError("缺少目标地点")
        self._check_question_reference(
            state,
            event,
            allowed_kinds={"confirm_location"},
            target={"level": event.target.level, "id": event.target.id},
        )
        target = self._resolve_target(event.target)
        self._validate_active_scope(state, target)
        if any(
            _target_key(item) == _target_key(target)
            for item in state.get("exclusions") or []
            if isinstance(item, Mapping)
        ):
            return self._outcome(state, False, "unchanged", f"「{target.get('label')}」此前已排除。")
        exclusions = [dict(item) for item in state.get("exclusions") or [] if isinstance(item, Mapping)]
        exclusions.append(dict(target))
        state["exclusions"] = exclusions
        state["pending_question"] = None
        state["pending_correction"] = None
        self._clear_path_invalidated_by_exclusion(state, target)
        self._refresh_derived_proposals(state)
        self._refresh_conflict(state)
        return self._outcome(state, True, "applied", f"已排除「{target.get('label')}」。")

    def _apply_retract(
        self,
        state: dict[str, Any],
        event: GuidedEventRequest,
    ) -> dict[str, Any]:
        if event.target is not None and event.clue_id is not None:
            raise GuidedValidationError("撤回操作只能提供 target 或 clue_id 其中一个")
        if event.clue_id is not None:
            for record in state.get("clues") or []:
                if not isinstance(record, Mapping) or str(record.get("id")) != str(event.clue_id):
                    continue
                if str(record.get("status") or "pending") == "pending":
                    return self._outcome(state, False, "unchanged", "线索已经处于待复核状态。")
                updated = dict(record)
                updated["status"] = "pending"
                state["clues"] = [
                    updated if isinstance(item, Mapping) and str(item.get("id")) == str(event.clue_id) else item
                    for item in state["clues"]
                ]
                self._refresh_derived_proposals(state)
                review_question_id = f"review:{event.clue_id}"
                state["asked_question_ids"] = [
                    item
                    for item in state.get("asked_question_ids") or []
                    if str(item) != review_question_id
                ]
                # Pin the restored clue as the current review question. The
                # generic priority scan could otherwise return a different
                # pending clue while the correction/conflict remains queued.
                pinned = _review_question(updated)
                pinned["_pinned"] = True
                state["pending_question"] = pinned
                self._refresh_conflict(state)
                return self._outcome(state, True, "applied", "线索已恢复为待复核。")
            raise GuidedValidationError("未知线索 ID")
        if event.target is None:
            raise GuidedValidationError("需要提供 target 或 clue_id")
        target = self._resolve_target(event.target)
        before = len(state.get("exclusions") or [])
        state["exclusions"] = [
            item for item in state.get("exclusions") or []
            if not (isinstance(item, Mapping) and _target_key(item) == _target_key(target))
        ]
        if len(state["exclusions"]) == before:
            return self._outcome(state, False, "unchanged", "该地点此前未被排除。")
        state["pending_question"] = None
        self._refresh_derived_proposals(state)
        self._refresh_conflict(state)
        return self._outcome(state, True, "applied", f"已恢复「{target.get('label')}」的候选资格。")

    def _apply_backtrack(
        self,
        state: dict[str, Any],
        event: GuidedEventRequest,
    ) -> dict[str, Any]:
        requested = event.level
        if requested not in {"country", "admin1", "city"}:
            raise GuidedValidationError("缺少回退层级")
        # admin1 was the old city-stage parent. It now aliases the two-step
        # country stage instead of leaving an invalid country+admin1-only path.
        level = "country" if requested == "admin1" else requested
        confirmed = state.get("confirmed") or {}
        if level == "country":
            changed = any(isinstance(confirmed.get(key), Mapping) for key in ("country", "admin1", "city"))
            new_confirmed = {"country": None, "admin1": None, "city": None}
        else:
            changed = isinstance(confirmed.get("city"), Mapping)
            new_confirmed = {
                "country": confirmed.get("country") if isinstance(confirmed, Mapping) else None,
                "admin1": confirmed.get("admin1") if isinstance(confirmed, Mapping) else None,
                "city": None,
            }
        if not changed:
            return self._outcome(state, False, "unchanged", f"当前{_LEVEL_LABELS[level]}阶段没有可回退的确认。")
        state["confirmed"] = new_confirmed
        state["pending_question"] = None
        state["asked_question_ids"] = []
        state["pending_correction"] = None
        state["proposals"] = []
        state["proposal_meta"] = {}
        self._refresh_derived_proposals(state)
        self._refresh_conflict(state)
        return self._outcome(state, True, "applied", f"已回退到{_LEVEL_LABELS[level]}阶段。")

    def _apply_free_text(
        self,
        state: dict[str, Any],
        session: GeoSession,
        event: GuidedEventRequest,
        *,
        preflight: _FreeTextPreflight | None = None,
    ) -> dict[str, Any]:
        text = (event.text or "").strip()
        if not text:
            raise GuidedValidationError("缺少自由文本内容")
        pending_state = self._pending_low_clue_question(state)
        if pending_state is not None and self._uncertain_free_text(text):
            self._record_clue_question(state, pending_state)
            state["pending_question"] = None
            state["conflict"] = None
            self._refresh_derived_proposals(state)
            return self._outcome(
                state,
                True,
                "applied",
                "已记录为无法判断并计入本轮问题；不会自动推进地点。",
            )
        pending = self._parser_pending(state.get("pending_question"))
        parsed = preflight.parsed if preflight is not None else self.parser.parse(
            text, pending=pending
        )
        added: list[str] = []
        parsed_answer = bool(
            pending_state is not None
            and not parsed.question
            and any(
                str(raw.get("dimension") or "") == str(pending_state.get("dimension") or "")
                for raw in parsed.clues
                if isinstance(raw, Mapping)
            )
        )
        existing_ids = {
            str(record.get("id")) for record in state.get("clues") or []
            if isinstance(record, Mapping) and record.get("id")
        }
        existing_evidence = {
            (
                str(record.get("dimension") or ""),
                str(record.get("value") or ""),
                str(record.get("source") or ""),
            )
            for record in state.get("clues") or []
            if isinstance(record, Mapping)
            and str(record.get("_origin") or "") == "guided_user"
        }
        duplicate_evidence = False
        previous_text = str(state.get("last_text") or "")

        # Persist every literal search term as a display-only clue even when the
        # LLM supplied the mention without a separate signage clue.
        for mention in parsed.place_mentions:
            for term in mention.search_terms:
                normalized = normalize_place_text(term)
                if not normalized:
                    continue
                covered = any(
                    isinstance(record, Mapping)
                    and (
                        normalize_place_text(record.get("value")) == normalized
                        or normalized in {
                            normalize_place_text(item)
                            for item in (record.get("_extra") or {}).get("place_terms", [])
                            if isinstance(item, str)
                        }
                    )
                    for record in state.get("clues") or []
                ) or any(
                    isinstance(raw, Mapping)
                    and (
                        normalize_place_text(raw.get("value")) == normalized
                        or normalized in {
                            normalize_place_text(item)
                            for item in (raw.get("extra") or {}).get("place_terms", [])
                            if isinstance(item, str)
                        }
                    )
                    for raw in parsed.clues
                )
                if covered:
                    continue
                extra = {
                    "literal_kind": "place_search_term",
                    "origin_source": "user_text",
                    "display_only": True,
                    "affects_ranking": False,
                    "reliable": False,
                    "validated": False,
                    "parser": mention.parser,
                    "place_terms": [term],
                    "place_quote": mention.quote,
                }
                parsed.clues.append({
                    "dimension": "signage",
                    "value": term[:64],
                    "confidence": 0.75,
                    "evidence": mention.quote[:200],
                    "source": "user",
                    "extra": extra,
                })

        for raw in parsed.clues:
            if not isinstance(raw, Mapping):
                continue
            record = _reviewed_record(raw)
            if str(record["id"]) in existing_ids:
                continue
            record["source"] = "user"
            evidence_key = (
                str(record.get("dimension") or ""),
                str(record.get("value") or ""),
                str(record.get("source") or ""),
            )
            if evidence_key in existing_evidence:
                duplicate_evidence = True
                continue
            extra = dict(record.get("_extra") or {})
            extra["parser"] = str(extra.get("parser") or "guided_rules")
            record["_extra"] = extra
            record["_origin"] = "guided_user"
            state.setdefault("clues", []).append(record)
            existing_ids.add(str(record["id"]))
            existing_evidence.add(evidence_key)
            added.append(str(record.get("id")))

        old_proposals = [
            dict(item) for item in state.get("proposals") or [] if isinstance(item, Mapping)
        ]
        proposed = self._propose_targets(state, text, parsed.clues)
        country_keys = {
            _target_state_key(item) for item in old_proposals
            if isinstance(item, Mapping) and item.get("level") != "city"
        }
        retained_country = [
            dict(item) for item in old_proposals
            if isinstance(item, Mapping) and item.get("level") != "city"
        ]
        for target in proposed:
            key = _target_state_key(target)
            if key not in country_keys:
                retained_country.append(target)
                country_keys.add(key)
        state["proposals"] = retained_country

        resolution_added = False
        if preflight is not None and preflight.resolution is not None:
            resolution_added = self._record_place_resolution(state, preflight.resolution)

        self._refresh_derived_proposals(state)
        proposals = [
            dict(item) for item in state.get("proposals") or [] if isinstance(item, Mapping)
        ]
        conflict = self._detect_conflict(state, proposals)
        correction = self._correction_plan(text, parsed, added)
        old_conflict = (
            dict(state.get("conflict")) if isinstance(state.get("conflict"), Mapping) else None
        )
        old_correction = (
            dict(state.get("pending_correction"))
            if isinstance(state.get("pending_correction"), Mapping) else None
        )
        effective = bool(
            added
            or resolution_added
            or proposals != old_proposals
            or conflict != old_conflict
            or correction is not None
            or old_correction is not None
        )
        if (
            pending_state is not None
            and parsed_answer
            and correction is None
            and conflict is None
        ):
            self._record_clue_question(state, pending_state)
            state["pending_question"] = None
            effective = True
        if not effective:
            if duplicate_evidence or (previous_text and text == previous_text):
                return self._outcome(
                    state,
                    False,
                    "unchanged",
                    "该文字线索已存在，无需重复记录。",
                )
            if bool(getattr(parsed, "question", False)):
                return self._outcome(
                    state,
                    False,
                    "unchanged",
                    "这是询问，未据此新增地点线索。",
                )
            return self._outcome(
                state,
                False,
                "unrecognized",
                "没有识别到新的目录证据、地点检索结果或修正计划。",
            )
        state["conflict"] = conflict
        state["pending_correction"] = correction
        state["last_text"] = text
        state["pending_question"] = None
        resolution = preflight.resolution if preflight is not None else None
        if correction is not None:
            summary = "检测到可能的排除或修正，需要你确认是否应用。"
        elif conflict is not None:
            summary = str(conflict.get("message") or "线索与已确认路径冲突，未自动改变路径。")
        elif resolution is not None and resolution.status == "resolved":
            labels = [
                str(item.get("label") or item.get("id") or "")
                for item in state.get("proposals") or []
                if isinstance(item, Mapping) and item.get("level") == "city"
            ]
            summary = (
                f"已记录文字线索，检索到目录城市候选{'、'.join(labels)}，需你确认。"
                if labels else "已记录文字线索，但候选当前不可见；请补充城市或区名。"
            )
        elif resolution is not None and resolution.status == "ambiguous":
            summary = resolution.notice or "地点不唯一；请补充城市、区名或附近地标。"
        elif resolution is not None and resolution.status == "foreign_conflict":
            summary = resolution.notice or "检索结果与当前国家不一致，未自动改变路径。"
        elif resolution is not None and resolution.status in {"unresolved", "unavailable"}:
            summary = resolution.notice or "暂未找到可核验城市；请补充城市、区名或附近地标。"
        elif proposals:
            summary = "已记录自由文本提议；必须明确确认地点后才会推进。"
        elif added:
            summary = "已记录文字线索；暂未匹配城市，可补充城市、区名或附近地标。"
        else:
            summary = "没有识别到新的目录证据或地点提议。"
        if preflight is not None and preflight.notice and resolution is None:
            summary = f"{preflight.notice} {summary}".strip()
        return self._outcome(state, True, "applied", summary)

    @staticmethod
    def _literal_ocr_text(record: Mapping[str, Any]) -> str:
        extra = record.get("_extra") if isinstance(record.get("_extra"), Mapping) else {}
        for value in (extra.get("adopted_text"), extra.get("literal"), record.get("value")):
            text = str(value or "").strip()
            if text:
                return text
        return ""

    @staticmethod
    def _is_literal_ocr_record(record: Mapping[str, Any]) -> bool:
        if str(record.get("dimension") or "") == "signage":
            return True
        extra = record.get("_extra") if isinstance(record.get("_extra"), Mapping) else {}
        return bool(
            extra.get("observation_kind") == "ocr_text"
            or extra.get("literal_kind") in {
                "adopted_ocr_observation",
                "reported_signage",
                "standalone_street_label",
            }
        )

    @classmethod
    def _confirmed_adoption_record(
        cls,
        state: Mapping[str, Any],
        observation: AdoptedOcrObservation,
    ) -> Mapping[str, Any] | None:
        for record in state.get("clues") or []:
            if not isinstance(record, Mapping):
                continue
            if str(record.get("status") or "") != "confirmed":
                continue
            extra = record.get("_extra")
            if is_user_confirmed_observation(extra) and observation.matches_extra(extra):
                return record
        return None

    @classmethod
    def _find_literal_ocr_record(
        cls,
        state: Mapping[str, Any],
        observation: AdoptedOcrObservation,
    ) -> Mapping[str, Any] | None:
        assert_key = normalize_place_text(observation.text) or observation.text.casefold()
        matches: list[tuple[tuple[Any, ...], Mapping[str, Any]]] = []
        for index, record in enumerate(state.get("clues") or []):
            if not isinstance(record, Mapping) or not cls._is_literal_ocr_record(record):
                continue
            candidates = {
                normalize_place_text(cls._literal_ocr_text(record)),
                normalize_place_text(record.get("value")),
            }
            candidates.discard("")
            if assert_key not in candidates:
                continue
            extra = record.get("_extra") if isinstance(record.get("_extra"), Mapping) else {}
            status = str(record.get("status") or "pending")
            matches.append((
                (
                    0 if observation.matches_extra(extra) else 1,
                    0 if status == "pending" else 1 if status == "confirmed" else 2,
                    0 if str(record.get("_origin") or "") == "guided_user" else 1,
                    index,
                ),
                record,
            ))
        return min(matches, key=lambda item: item[0])[1] if matches else None

    def _apply_adopted_observation(
        self,
        state: dict[str, Any],
        event: GuidedEventRequest,
        *,
        preflight: _FreeTextPreflight | _ObservationAdoptionPreflight | None = None,
    ) -> dict[str, Any]:
        if not isinstance(preflight, _ObservationAdoptionPreflight):
            raise GuidedValidationError("缺少已校验的 OCR 观察")
        observation = preflight.observation
        existing = self._confirmed_adoption_record(state, observation)
        if existing is not None:
            outcome = self._outcome(
                state,
                False,
                "unchanged",
                "该 OCR 文字及来源此前已经采用，无需重复操作。",
            )
            outcome["adopted_clue_id"] = str(existing.get("id") or "")
            return outcome

        record = self._find_literal_ocr_record(state, observation)
        clue_id = str(record.get("id") or "") if isinstance(record, Mapping) else ""
        status_changed = False
        provenance_changed = False
        if record is not None:
            updated = dict(record)
            extra, provenance_changed = observation.merge_into_extra(
                updated.get("_extra"),
                event_id=event.event_id,
                place_term=preflight.place_term,
            )
            status_changed = str(updated.get("status") or "pending") != "confirmed"
            updated["status"] = "confirmed"
            updated["_extra"] = extra
            state["clues"] = [
                updated
                if isinstance(item, Mapping) and str(item.get("id") or "") == clue_id
                else item
                for item in state.get("clues") or []
            ]
        else:
            extra, provenance_changed = observation.merge_into_extra(
                {},
                event_id=event.event_id,
                place_term=preflight.place_term,
            )
            evidence = observation.evidence or (
                f"用户确认局部 OCR 文字「{observation.text[:160]}」。"
            )
            raw = {
                "dimension": "signage",
                "value": observation.text[:256],
                "confidence": observation.confidence if observation.confidence is not None else 0.8,
                "evidence": evidence,
                "source": "user",
                "extra": extra,
            }
            updated = _reviewed_record(raw)
            updated["status"] = "confirmed"
            updated["_origin"] = "guided_user"
            clue_id = str(updated.get("id") or "")
            state.setdefault("clues", []).append(updated)
            status_changed = True
        final_extra = dict(updated.get("_extra") or {})
        final_extra.update({
            "literal_kind": str(final_extra.get("literal_kind") or "adopted_ocr_observation"),
            "display_only": True,
            "affects_ranking": False,
            "reliable": True,
            "validated": True,
        })
        updated["_extra"] = final_extra
        if record is not None:
            state["clues"] = [
                updated
                if isinstance(item, Mapping) and str(item.get("id") or "") == clue_id
                else item
                for item in state.get("clues") or []
            ]

        resolution_added = False
        if preflight.resolution is not None:
            resolution_added = self._record_place_resolution(state, preflight.resolution)
        self._refresh_derived_proposals(state)
        state["pending_question"] = None
        self._refresh_conflict(state)
        effective = bool(status_changed or provenance_changed or resolution_added)
        if not effective:
            outcome = self._outcome(
                state,
                False,
                "unchanged",
                "该 OCR 文字及来源此前已经采用，无需重复操作。",
            )
        else:
            outcome = self._outcome(
                state,
                True,
                "applied",
                self._adoption_summary(
                    state,
                    observation,
                    preflight,
                    preflight.resolution,
                ),
            )
        outcome["adopted_clue_id"] = clue_id
        return outcome

    def _adoption_summary(
        self,
        state: Mapping[str, Any],
        observation: AdoptedOcrObservation,
        preflight: _ObservationAdoptionPreflight,
        resolution: PlaceResolutionBatch | None,
    ) -> str:
        text = observation.text if len(observation.text) <= 80 else observation.text[:77] + "…"
        prefix = f"文字已采用（OCR 原文「{text}」，来源已记录）"
        if resolution is None:
            notice = str(preflight.notice or "地点检索暂时不可用。").strip()
            return (
                f"{prefix}；地点检索失败：{notice} "
                "文字已保存，但尚无地点检索结果，不声称已影响排序。"
            )
        status = str(resolution.status or "")
        if status == "resolved":
            labels = [
                str(item.get("label") or item.get("id") or "")
                for item in state.get("proposals") or []
                if isinstance(item, Mapping) and item.get("level") == "city"
            ]
            suffix = f"命中目录城市候选{'、'.join(labels)}" if labels else "已产生可核验目录候选"
            return (
                f"{prefix}；地点检索成功，{suffix}。"
                "已进入既有地点候选/统一评分链，仍需你明确确认，不会自动确认城市或坐标。"
            )
        if status == "ambiguous":
            notice = str(resolution.notice or "命中的地点不唯一。").strip()
            return (
                f"{prefix}；地点检索结果有歧义：{notice} "
                "文字已保存，需补充城市、区名或附近地标，不会自动选择。"
            )
        if status == "unavailable":
            notice = str(resolution.notice or "地名检索暂时不可用。").strip()
            return (
                f"{prefix}；地点检索失败：{notice} "
                "文字已保存，但尚无地点检索结果，不声称已影响排序。"
            )
        if status == "foreign_conflict":
            notice = str(resolution.notice or "检索结果与当前国家不一致。").strip()
            return (
                f"{prefix}；地点检索结果与当前国家冲突：{notice} "
                "文字已保存，未自动改变已确认路径。"
            )
        notice = str(resolution.notice or "未找到可核验的城市结果。").strip()
        return (
            f"{prefix}；地点检索无结果：{notice} "
            "文字已保存，尚未产生地点候选，不声称已影响排序。"
        )

    def _apply_answer(
        self,
        state: dict[str, Any],
        event: GuidedEventRequest,
    ) -> dict[str, Any]:
        question = self._require_question(state, event)
        value = (event.value or "").strip()
        if not value:
            raise GuidedValidationError("回答事件缺少 value")
        if question.get("correction_id") or question.get("dimension") == "correction":
            return self._apply_correction_decision(state, question, value)
        self._record_clue_question(state, question)
        if question.get("kind") == "ask" and isinstance(state.get("conflict"), Mapping):
            state["conflict"] = None
            state["proposals"] = []
            state["pending_question"] = None
            record = _reviewed_record({
                "dimension": str(question.get("dimension") or "place_name"),
                "value": value,
                "confidence": 0.9,
                "evidence": "conflict clarification",
                "source": "user",
                "extra": {"parser": "guided_conflict_answer"},
            })
            record["_origin"] = "guided_user"
            state.setdefault("clues", []).append(record)
            self._refresh_derived_proposals(state)
            return self._outcome(state, True, "applied", "已记录澄清；已确认路径保持不变，冲突问题结束。")
        if question.get("kind") == "review_clue":
            token = value.lower()
            if token in {"confirm", "confirmed", "yes", "true", "1"}:
                return self._apply_clue_status(state, str(question.get("clue_id") or ""), "confirmed")
            if token in {"reject", "rejected", "no", "false", "0"}:
                return self._apply_clue_status(state, str(question.get("clue_id") or ""), "rejected")
            if token in {"unknown", "unsure", "maybe"}:
                return self._apply_clue_status(state, str(question.get("clue_id") or ""), "uncertain")
        dimension = str(question.get("dimension") or "place_name")
        record = _reviewed_record({
            "dimension": dimension,
            "value": value,
            "confidence": 0.9,
            "evidence": "user answer",
            "source": "user",
            "extra": {"parser": "guided_answer"},
        })
        record["_origin"] = "guided_user"
        for item in state.get("clues") or []:
            if isinstance(item, Mapping) and item.get("dimension") == dimension and str(item.get("value")) == value:
                return self._outcome(state, False, "unchanged", "该回答已记录。")
        state.setdefault("clues", []).append(record)
        state["pending_question"] = None
        return self._outcome(state, True, "applied", "已记录为待复核线索；没有确认任何地点。")

    def _source_region_target(self, raw_iso: str) -> dict[str, Any]:
        raw = normalize_iso(raw_iso)
        canonical = canonical_country_iso(raw)
        if not source_region_iso(raw) or not canonical:
            raise GuidedValidationError("未知地区范围")
        record = next(
            (
                item for item in self._country_records()
                if normalize_iso(item.get("iso")) == raw
            ),
            {},
        )
        try:
            place = self._store().get_place("country", raw)
        except Exception as exc:
            raise self._unavailable(exc) from exc
        coordinate = self._coordinate_fields(place)
        return {
            "level": "country",
            "id": raw,
            "label": str(record.get("name_zh") or record.get("name") or raw),
            "country_iso": canonical,
            "source_iso": raw,
            "admin1_id": None,
            "city_id": None,
            "lat": coordinate["lat"],
            "lng": coordinate["lng"],
            **coordinate["provenance"],
        }

    def _correction_targets_for_iso(self, value: object) -> list[dict[str, Any]]:
        raw = normalize_iso(value)
        canonical = canonical_country_iso(raw)
        if not raw or not canonical or not self._country_exists(canonical):
            raise GuidedValidationError("修正计划包含未知国家或地区")
        if raw != canonical:
            return [self._source_region_target(raw)]
        return [self._country_target(canonical)]

    def _apply_correction_decision(
        self,
        state: dict[str, Any],
        question: Mapping[str, Any],
        value: str,
    ) -> dict[str, Any]:
        plan = state.get("pending_correction")
        if not isinstance(plan, Mapping):
            raise GuidedConflict("修正计划已失效，请重新输入")
        token = value.strip().lower()
        apply_tokens = {"apply", "yes", "true", "1", "应用", "确认", "是", "确定"}
        keep_tokens = {"keep", "no", "false", "0", "保持", "取消", "不应用", "否"}
        if token not in apply_tokens and token not in keep_tokens:
            raise GuidedValidationError("请选择应用修正或保持原样")
        if token in keep_tokens:
            state["pending_correction"] = None
            state["pending_question"] = None
            self._refresh_conflict(state)
            return self._outcome(state, True, "applied", "已保留原状态，未应用修正计划。")

        reject_targets: list[dict[str, Any]] = []
        clear_targets: list[dict[str, Any]] = []
        for iso in plan.get("reject_isos") or []:
            reject_targets.extend(self._correction_targets_for_iso(iso))
        for iso in plan.get("clear_exclusions") or []:
            clear_targets.extend(self._correction_targets_for_iso(iso))
        for target in reject_targets:
            self._validate_active_scope(state, target)

        clear_keys = {_target_key(target) for target in clear_targets}
        if clear_keys:
            state["exclusions"] = [
                item for item in state.get("exclusions") or []
                if not (isinstance(item, Mapping) and _target_key(item) in clear_keys)
            ]

        existing_keys = {
            _target_key(item) for item in state.get("exclusions") or []
            if isinstance(item, Mapping)
        }
        for target in reject_targets:
            key = _target_key(target)
            if key not in existing_keys:
                state.setdefault("exclusions", []).append(dict(target))
                existing_keys.add(key)
            self._clear_path_invalidated_by_exclusion(state, target)

        replace_dimensions = {
            str(item) for item in plan.get("replace_dimensions") or [] if str(item)
        }
        keep_new_ids = {str(item) for item in plan.get("add_clue_ids") or []}
        remove_values = {
            (str(item[0]), str(item[1]))
            for item in plan.get("remove_clue_values") or []
            if isinstance(item, (list, tuple)) and len(item) == 2
        }
        if replace_dimensions or remove_values:
            remaining: list[Any] = []
            for record in state.get("clues") or []:
                if (
                    isinstance(record, Mapping)
                    and str(record.get("_origin") or "") == "guided_user"
                    and str(record.get("id") or "") not in keep_new_ids
                    and (
                        str(record.get("dimension") or "") in replace_dimensions
                        or (str(record.get("dimension") or ""), str(record.get("value") or "")) in remove_values
                    )
                ):
                    continue
                remaining.append(record)
            state["clues"] = remaining

        self._refresh_derived_proposals(state)
        state["pending_correction"] = None
        state["pending_question"] = None
        self._refresh_conflict(state)
        return self._outcome(state, True, "applied", "已应用修正计划；当前阶段不会自动推进。")

    def _apply_unknown(
        self,
        state: dict[str, Any],
        event: GuidedEventRequest,
    ) -> dict[str, Any]:
        question = self._require_question(state, event)
        question_id = str(question.get("id") or "")
        effective = False
        if question.get("correction_id") and isinstance(state.get("pending_correction"), Mapping):
            state["pending_correction"] = None
            effective = True
        if question.get("kind") == "review_clue":
            clue_id = str(question.get("clue_id") or "")
            for record in state.get("clues") or []:
                if isinstance(record, Mapping) and str(record.get("id")) == clue_id:
                    if str(record.get("status") or "pending") != "uncertain":
                        updated = dict(record)
                        updated["status"] = "uncertain"
                        state["clues"] = [
                            updated if isinstance(item, Mapping) and str(item.get("id")) == clue_id else item
                            for item in state["clues"]
                        ]
                        self._refresh_derived_proposals(state)
                        effective = True
                    break
        history_before = len(self._asked_clue_history(state))
        asked_before = {str(item) for item in state.get("asked_question_ids") or []}
        self._record_clue_question(state, question)
        if len(self._asked_clue_history(state)) > history_before:
            effective = True
        asked = list(state.get("asked_question_ids") or [])
        if question_id and question_id not in asked_before:
            if question_id not in asked:
                asked.append(question_id)
            effective = True
        state["asked_question_ids"] = asked
        state["pending_question"] = None
        if question.get("kind") == "ask":
            state["conflict"] = None
            state["proposals"] = []
            state["proposal_meta"] = {}
        return self._outcome(
            state,
            effective,
            "applied" if effective else "unchanged",
            "已标记为无法判断并切换到其他问题；当前阶段保持不变。",
        )

    @staticmethod
    def _outcome(
        state: dict[str, Any],
        effective: bool,
        status: str,
        summary: str,
    ) -> dict[str, Any]:
        return {"state": state, "effective": effective, "status": status, "summary": summary}

    def _check_question_reference(
        self,
        state: Mapping[str, Any],
        event: GuidedEventRequest,
        *,
        allowed_kinds: set[str] | None = None,
        target: Mapping[str, Any] | None = None,
        clue_id: str | None = None,
    ) -> None:
        if event.question_id is None:
            return
        pending = state.get("pending_question")
        if not isinstance(pending, Mapping) or str(pending.get("id")) != str(event.question_id):
            raise GuidedConflict("问题已失效，请重新加载对话")
        if allowed_kinds is not None and str(pending.get("kind") or "") not in allowed_kinds:
            raise GuidedConflict("question_id 与当前问题类型不匹配")
        if clue_id is not None and str(pending.get("clue_id") or "") != str(clue_id):
            raise GuidedConflict("question_id 与线索不匹配")
        if target is not None:
            pending_target = pending.get("target")
            if not isinstance(pending_target, Mapping) or _target_key(pending_target) != _target_key(target):
                raise GuidedConflict("question_id 与目标地点不匹配")

    def _has_pending_clue_review(self, state: Mapping[str, Any]) -> bool:
        """True while any observation clue still awaits review.

        Covers both the current ``review_clue`` question and correction/conflict
        asks that take precedence while unreviewed clues are queued.
        """
        pending = state.get("pending_question")
        if isinstance(pending, Mapping) and str(pending.get("kind") or "") == "review_clue":
            return True
        for record in state.get("clues") or []:
            if isinstance(record, Mapping) and str(record.get("status") or "pending") == "pending":
                return True
        return False

    def _require_question(self, state: Mapping[str, Any], event: GuidedEventRequest) -> dict[str, Any]:
        pending = state.get("pending_question")
        if not isinstance(pending, Mapping) or not event.question_id or str(pending.get("id")) != str(event.question_id):
            raise GuidedConflict("回答或无法判断事件必须引用当前待回答问题")
        return dict(pending)

    @staticmethod
    def _parser_pending(question: Mapping[str, Any] | None) -> PendingQuestion | None:
        if not isinstance(question, Mapping):
            return None
        dimension = question.get("dimension")
        target = question.get("target")
        iso = target.get("country_iso") if isinstance(target, Mapping) else None
        return PendingQuestion(
            kind=str(question.get("kind") or ""),
            dimensions=[str(dimension)] if dimension else [],
            options=[],
            iso=str(iso) if iso else None,
            name_zh=str(target.get("label") or "") if isinstance(target, Mapping) else "",
        )

    def _clear_path_invalidated_by_exclusion(self, state: dict[str, Any], exclusion: Mapping[str, Any]) -> None:
        confirmed = state.get("confirmed") or {}
        level = str(exclusion.get("level") or "")
        country = confirmed.get("country") if isinstance(confirmed, Mapping) else None
        city = confirmed.get("city") if isinstance(confirmed, Mapping) else None
        if level == "country":
            excluded_iso = normalize_iso(exclusion.get("id"))
            country_iso = canonical_country_iso(country.get("id")) if isinstance(country, Mapping) else ""
            if excluded_iso == country_iso:
                state["confirmed"] = {"country": None, "admin1": None, "city": None}
            elif (
                isinstance(city, Mapping)
                and source_region_iso(excluded_iso)
                and normalize_iso(city.get("source_iso")) == excluded_iso
            ):
                state["confirmed"] = {
                    "country": country,
                    "admin1": confirmed.get("admin1"),
                    "city": None,
                }
        elif level == "admin1":
            if isinstance(city, Mapping) and city.get("admin1_id") == exclusion.get("admin1_id"):
                state["confirmed"] = {
                    "country": country,
                    "admin1": confirmed.get("admin1"),
                    "city": None,
                }
            if isinstance(confirmed.get("admin1"), Mapping) and confirmed["admin1"].get("id") == exclusion.get("id"):
                state["confirmed"] = {"country": country, "admin1": None, "city": None}
        elif level == "city" and isinstance(city, Mapping) and city.get("id") == exclusion.get("id"):
            state["confirmed"] = {
                "country": country,
                "admin1": confirmed.get("admin1"),
                "city": None,
            }
        state["pending_question"] = None

    @staticmethod
    def _correction_plan(text: str, parsed: Any, added_clue_ids: Sequence[str]) -> dict[str, Any] | None:
        reject_isos = list(dict.fromkeys(str(item).strip().upper() for item in (parsed.reject_isos or []) if item))
        clear_exclusions = list(dict.fromkeys(str(item).strip().upper() for item in (parsed.clear_exclusions or []) if item))
        replace_dimensions = list(dict.fromkeys(str(item).strip() for item in (parsed.replace_dimensions or []) if item))
        remove_values = [
            [str(dimension).strip(), str(value).strip()]
            for dimension, value in (parsed.remove_clue_values or [])
            if dimension and value is not None
        ]
        if not (reject_isos or clear_exclusions or replace_dimensions or remove_values):
            return None
        payload = {
            "source_text": text[:500],
            "reject_isos": reject_isos,
            "clear_exclusions": clear_exclusions,
            "replace_dimensions": replace_dimensions,
            "remove_clue_values": remove_values,
            "add_clue_ids": [str(item) for item in added_clue_ids],
        }
        return {**payload, "id": _json_hash("correction", payload)}

    # -------------------------------------------------------------- proposals
    def _propose_targets(
        self,
        state: Mapping[str, Any],
        text: str,
        parsed_clues: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return only explicit country proposals from accepted parsed clues."""

        proposals: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for raw in parsed_clues:
            if not isinstance(raw, Mapping):
                continue
            if str(raw.get("dimension") or "") != "place_name":
                continue
            value = str(raw.get("value") or "").strip()
            extra = raw.get("extra") if isinstance(raw.get("extra"), Mapping) else {}
            if extra.get("place_kind") != "country":
                continue
            iso = normalize_iso(value)
            if not iso or not self._country_exists(iso):
                continue
            target = self._country_target(iso)
            key = _target_key(target)
            if key in seen:
                continue
            seen.add(key)
            proposals.append(target)
        return proposals[:PROPOSAL_LIMIT]

    def _record_place_resolution(
        self,
        state: dict[str, Any],
        resolution: PlaceResolutionBatch,
    ) -> bool:
        """Persist bounded provenance and link it to active literal clues."""

        terms = {
            normalize_place_text(term)
            for term in resolution.query_terms
            if normalize_place_text(term)
        }
        clue_ids: list[str] = []
        quote = ""
        for record in state.get("clues") or []:
            if not isinstance(record, Mapping):
                continue
            extra = record.get("_extra") if isinstance(record.get("_extra"), Mapping) else {}
            if (
                str(record.get("_origin") or "") != "guided_user"
                and not is_user_confirmed_observation(extra)
            ):
                continue
            if str(record.get("status") or "pending") not in {"pending", "confirmed"}:
                continue
            record_terms = {
                normalize_place_text(item)
                for item in extra.get("place_terms", [])
                if isinstance(item, str)
            }
            if not record_terms:
                record_terms.add(normalize_place_text(record.get("value")))
            if not terms or not (record_terms & terms):
                continue
            clue_id = str(record.get("id") or "")
            if clue_id and clue_id not in clue_ids:
                clue_ids.append(clue_id)
            if not quote:
                quote = str(extra.get("place_quote") or record.get("description") or "")[:200]
        if not clue_ids:
            logger.warning("place resolution had no active literal clue linkage")
            return False
        cities = [
            {
                "city_id": str(evidence.city_id),
                "source_iso": normalize_iso(evidence.source_iso),
                "confidence": float(evidence.confidence),
                "match_kind": str(evidence.match_kind),
            }
            for evidence in resolution.city_evidence
        ]
        sources = [dict(item) for item in resolution.sources if isinstance(item, Mapping)]
        identity = {
            "status": resolution.status,
            "query_terms": list(resolution.query_terms),
            "city_ids": list(resolution.city_id_strings),
            "foreign_country_isos": list(resolution.foreign_country_isos),
        }
        resolution_id = _json_hash("place-resolution", identity)
        entries = [
            dict(item) for item in state.get("place_resolutions") or []
            if isinstance(item, Mapping)
        ]
        if resolution.status == "resolved":
            for item in entries:
                if str(item.get("status") or "") in {"unresolved", "ambiguous", "unavailable"}:
                    item["superseded_by"] = resolution_id
        else:
            current_terms = {normalize_place_text(item) for item in resolution.query_terms}
            for item in entries:
                previous_terms = {
                    normalize_place_text(value) for value in item.get("query_terms") or []
                }
                if (
                    str(item.get("status") or "") in {"unresolved", "ambiguous", "unavailable"}
                    and previous_terms == current_terms
                    and str(item.get("id") or "") != resolution_id
                ):
                    item["superseded_by"] = resolution_id
        existing_index = next(
            (
                index for index, item in enumerate(entries)
                if str(item.get("id") or "") == resolution_id
            ),
            None,
        )
        if existing_index is not None:
            existing = entries[existing_index]
            merged_ids = sorted(set(existing.get("clue_ids") or []) | set(clue_ids))
            changed = merged_ids != list(existing.get("clue_ids") or [])
            if changed:
                existing["clue_ids"] = merged_ids
            known_sources = {
                (str(item.get("provider")), str(item.get("provider_place_id")))
                for item in existing.get("sources") or [] if isinstance(item, Mapping)
            }
            for source in sources:
                key = (
                    str(source.get("provider")),
                    str(source.get("provider_place_id")),
                )
                if key not in known_sources:
                    existing.setdefault("sources", []).append(source)
                    known_sources.add(key)
                    changed = True
            if cities:
                existing["cities"] = cities
                changed = True
            if changed:
                state["place_resolutions"] = entries
            return changed
        entries.append({
            "id": resolution_id,
            "version": int(resolution.version),
            "status": resolution.status,
            "query_terms": list(resolution.query_terms),
            "quote": quote or " ".join(resolution.query_terms),
            "clue_ids": sorted(clue_ids),
            "city_ids": list(resolution.city_id_strings),
            "cities": cities,
            "sources": sources,
            "foreign_country_isos": list(resolution.foreign_country_isos),
            "notice": resolution.notice,
            "fetched_at": resolution.fetched_at,
            "expires_at": resolution.expires_at,
        })
        state["place_resolutions"] = entries[-20:]
        return True

    @staticmethod
    def _resolution_expired(entry: Mapping[str, Any]) -> bool:
        raw = str(entry.get("expires_at") or "").strip()
        if not raw:
            return False
        try:
            expires = datetime.fromisoformat(raw)
        except ValueError:
            return True
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return expires <= datetime.now(timezone.utc)

    def _active_resolution_entries(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        active_ids = self._active_guided_clue_ids(state)
        result: list[dict[str, Any]] = []
        for entry in state.get("place_resolutions") or []:
            if not isinstance(entry, Mapping):
                continue
            if entry.get("superseded_by") or self._resolution_expired(entry):
                continue
            clue_ids = {str(item) for item in entry.get("clue_ids") or [] if str(item)}
            if clue_ids and clue_ids.issubset(active_ids):
                result.append(dict(entry))
        return result

    def _place_city_evidence(self, state: Mapping[str, Any]) -> list[CityEvidence]:
        evidence: list[CityEvidence] = []
        for entry in self._active_resolution_entries(state):
            if str(entry.get("status") or "") != "resolved":
                continue
            clue_ids = tuple(str(item) for item in entry.get("clue_ids") or [] if str(item))
            sources = tuple(
                item for item in entry.get("sources") or [] if isinstance(item, Mapping)
            )
            for city in entry.get("cities") or []:
                if not isinstance(city, Mapping):
                    continue
                try:
                    city_id = int(str(city.get("city_id")))
                except (TypeError, ValueError):
                    continue
                source_iso = normalize_iso(city.get("source_iso"))
                if not source_iso:
                    continue
                evidence.append(CityEvidence(
                    city_id=city_id,
                    source_iso=source_iso,
                    text=" ".join(str(item) for item in entry.get("query_terms") or []),
                    confidence=float(city.get("confidence") or 0.0),
                    match_kind=str(city.get("match_kind") or "catalog_exact"),
                    clue_ids=clue_ids,
                    sources=sources,
                ))
        return evidence

    def _resolved_country_targets(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Project a unique resolved catalog city back to its country proposal."""

        countries: set[str] = set()
        for entry in self._active_resolution_entries(state):
            if str(entry.get("status") or "") != "resolved":
                continue
            for city_id in entry.get("city_ids") or []:
                row = self._city_row_by_id(str(city_id))
                if row is None:
                    continue
                canonical = canonical_country_iso(row.iso)
                if canonical:
                    countries.add(canonical)
        if len(countries) != 1:
            return []
        country = next(iter(countries))
        if not self._country_exists(country):
            return []
        return [self._country_target(country)]

    def _refresh_derived_proposals(self, state: dict[str, Any]) -> None:
        """Rebuild city proposals and metadata from active literal evidence."""

        retained = [
            dict(item) for item in state.get("proposals") or []
            if isinstance(item, Mapping) and item.get("level") != "city"
        ]
        proposals: dict[str, dict[str, Any]] = {}
        meta: dict[str, dict[str, Any]] = {}
        scopes = set(self._country_source_scopes(state))
        for entry in self._active_resolution_entries(state):
            status = str(entry.get("status") or "")
            if status not in {"resolved", "ambiguous"}:
                continue
            sources = [
                dict(item) for item in entry.get("sources") or [] if isinstance(item, Mapping)
            ]
            city_details = {
                str(item.get("city_id")): dict(item)
                for item in entry.get("cities") or [] if isinstance(item, Mapping)
            }
            for city_id in entry.get("city_ids") or []:
                row = self._city_row_by_id(str(city_id))
                if row is None:
                    continue
                if scopes and normalize_iso(row.iso) not in scopes:
                    continue
                target = self._city_target(row)
                if self._excluded_by_state(state, target) is not None:
                    continue
                key = _target_state_key(target)
                proposals[key] = target
                row_label = str(target.get("label") or row.name)
                detail = city_details.get(str(city_id), {})
                provider = next(
                    (str(item.get("provider") or "") for item in sources if item.get("provider")),
                    "local_catalog",
                )
                if status == "ambiguous":
                    reason = (
                        "文字可能指向多个目录城市；需补充城市、区名或附近地标，"
                        "不会按人口或检索排名自动选择。"
                    )
                elif provider == "local_catalog":
                    reason = f"目录名称精确匹配「{row_label}」，待确认。"
                else:
                    reason = (
                        f"用户文字经 {provider} 检索并映射到目录城市「{row_label}」，待确认。"
                    )
                current = meta.get(key)
                if current is None:
                    meta[key] = {
                        "reason": reason,
                        "clue_ids": list(entry.get("clue_ids") or []),
                        "resolution_ids": [str(entry.get("id") or "")],
                        "selection_basis": "user_text",
                        "ambiguous": status == "ambiguous",
                        "query_terms": list(entry.get("query_terms") or []),
                        "match_kind": str(detail.get("match_kind") or "geocoder_locality"),
                        "sources": sources,
                    }
                else:
                    current["clue_ids"] = sorted(
                        set(current.get("clue_ids") or []) | set(entry.get("clue_ids") or [])
                    )
                    current["resolution_ids"] = sorted(
                        set(current.get("resolution_ids") or []) | {str(entry.get("id") or "")}
                    )
                    current["ambiguous"] = bool(current.get("ambiguous") or status == "ambiguous")
                    known = {
                        (str(item.get("provider")), str(item.get("provider_place_id")))
                        for item in current.get("sources") or []
                        if isinstance(item, Mapping)
                    }
                    for source in sources:
                        source_key = (
                            str(source.get("provider")),
                            str(source.get("provider_place_id")),
                        )
                        if source_key not in known:
                            current.setdefault("sources", []).append(source)
                            known.add(source_key)
        state["proposals"] = [*retained, *proposals.values()][:PROPOSAL_LIMIT]
        state["proposal_meta"] = meta

    def _proposal_reason(self, state: Mapping[str, Any], row: CityRow) -> str:
        key = f"city:{row.city_id}"
        meta = (state.get("proposal_meta") or {}).get(key)
        if isinstance(meta, Mapping) and meta.get("reason"):
            reason = str(meta["reason"])
            localized = city_name_zh(row.city_id, row.name)
            if row.name and row.name != localized:
                reason = reason.replace(row.name, localized)
            return reason
        return "用户文字候选"

    def _place_match_from_meta(
        self,
        state: Mapping[str, Any],
        row: CityRow,
    ) -> dict[str, Any] | None:
        key = f"city:{row.city_id}"
        meta = (state.get("proposal_meta") or {}).get(key)
        if not isinstance(meta, Mapping):
            return None
        return {
            "query_terms": [str(item)[:60] for item in meta.get("query_terms") or []][:4],
            "clue_ids": [str(item)[:120] for item in meta.get("clue_ids") or []][:20],
            "match_kind": str(meta.get("match_kind") or "catalog_exact"),
            "ambiguous": bool(meta.get("ambiguous")),
            "sources": [
                {
                    "provider": str(item.get("provider") or "local_catalog")[:40],
                    "provider_place_id": (
                        str(item.get("provider_place_id"))[:200]
                        if item.get("provider_place_id") is not None else None
                    ),
                    "label": str(item.get("label") or row.name)[:200],
                    "query": str(item.get("query") or " ".join(meta.get("query_terms") or []))[:200],
                    "attribution": (
                        str(item.get("attribution"))[:200]
                        if item.get("attribution") is not None else None
                    ),
                    "license": (
                        str(item.get("license"))[:80]
                        if item.get("license") is not None else None
                    ),
                    "fetched_at": str(item.get("fetched_at") or "")[:80],
                }
                for item in meta.get("sources") or []
                if isinstance(item, Mapping)
            ][:4],
        }

    def _refresh_conflict(self, state: dict[str, Any]) -> None:
        proposals = [
            dict(item) for item in state.get("proposals") or []
            if isinstance(item, Mapping)
        ]
        conflict = self._detect_conflict(state, proposals)
        state["conflict"] = conflict
        pending = state.get("pending_question")
        if isinstance(pending, Mapping) and pending.get("conflict_id"):
            if not isinstance(conflict, Mapping) or str(conflict.get("id")) != str(pending.get("conflict_id")):
                state["pending_question"] = None

    def _detect_conflict(
        self,
        state: Mapping[str, Any],
        proposals: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any] | None:
        confirmed = state.get("confirmed") or {}
        country = confirmed.get("country") if isinstance(confirmed, Mapping) else None
        city = confirmed.get("city") if isinstance(confirmed, Mapping) else None
        conflicts: list[dict[str, Any]] = []
        proposal_countries = {
            canonical_country_iso(item.get("country_iso"))
            for item in proposals if isinstance(item, Mapping)
        }
        proposal_countries.discard("")
        if len(proposal_countries) > 1 and isinstance(country, Mapping):
            conflicts.append({"reason": "multiple countries proposed", "proposed": list(proposals)})
        for item in proposals:
            if not isinstance(item, Mapping):
                continue
            if isinstance(country, Mapping) and canonical_country_iso(item.get("country_iso")) != canonical_country_iso(country.get("id")):
                conflicts.append({"reason": "different country proposed", "proposed": dict(item)})
                continue
            if isinstance(city, Mapping) and item.get("level") == "city" and item.get("id") != city.get("id"):
                conflicts.append({"reason": "different city proposed", "proposed": dict(item)})
        if isinstance(country, Mapping):
            confirmed_country = canonical_country_iso(country.get("id"))
            for entry in self._active_resolution_entries(state):
                foreign = {
                    canonical_country_iso(item)
                    for item in entry.get("foreign_country_isos") or []
                }
                foreign.discard("")
                if foreign and confirmed_country not in foreign:
                    conflicts.append({
                        "reason": "different country proposed",
                        "proposed": {
                            "source": "place_resolution",
                            "resolution_id": entry.get("id"),
                            "country_iso": sorted(foreign),
                            "query_terms": entry.get("query_terms") or [],
                        },
                    })
        if not conflicts:
            return None
        reasons = [str(item["reason"]) for item in conflicts]
        conflict_id = _json_hash("conflict", conflicts)
        if any(reason in {"multiple countries proposed", "different country proposed"} for reason in reasons):
            message = (
                "文本指向与当前国家不一致的地点，已保持当前路径不变；"
                "如需选择该国家，请先回退到国家阶段。"
            )
        else:
            message = "文本与已确认路径冲突，请先澄清；已确认路径不会自动改变。"
        return {
            "id": conflict_id,
            "message": message,
            "proposed": [dict(item) for item in proposals],
            "confirmed": dict(confirmed),
            "reasons": reasons,
        }

    # -------------------------------------------------------------- response
    @staticmethod
    def _public_clue_rows(state: Mapping[str, Any], session: GeoSession) -> list[dict[str, Any]]:
        """Merge immutable machine clues with reviewable guided-user clues."""

        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()

        def add(row: Mapping[str, Any]) -> None:
            dimension = str(row.get("dimension") or "")
            value = str(row.get("value") or "")
            if not dimension or not value:
                return
            key = (
                dimension,
                value,
                str(row.get("source") or ""),
                str(row.get("evidence") or ""),
            )
            if key in seen:
                return
            seen.add(key)
            rows.append(dict(row))

        # User-confirmed observation provenance may belong to an otherwise
        # immutable session clue. Surface the review-layer copy first so the
        # public clue carries the adopted source instead of losing it in the
        # duplicate-key check against the raw machine row.
        for record in state.get("clues") or []:
            if not isinstance(record, Mapping):
                continue
            if not is_user_confirmed_observation(record.get("_extra")):
                continue
            clue = _reviewed_to_clue(record)
            if clue is None:
                continue
            row = clue.to_dict()
            extra = dict(row.get("extra") or {})
            extra.update({
                "guided": True,
                "guided_clue_id": str(record.get("id") or ""),
                "guided_status": str(record.get("status") or "pending"),
            })
            row["extra"] = extra
            add(row)
        for item in session.clues or []:
            if isinstance(item, Mapping):
                add(item)
        for record in state.get("clues") or []:
            if not isinstance(record, Mapping) or str(record.get("_origin") or "") != "guided_user":
                continue
            clue = _reviewed_to_clue(record)
            if clue is None:
                continue
            row = clue.to_dict()
            extra = dict(row.get("extra") or {})
            extra.update({
                "guided": True,
                "guided_clue_id": str(record.get("id") or ""),
                "guided_status": str(record.get("status") or "pending"),
            })
            row["extra"] = extra
            add(row)
        return rows

    def _response(
        self,
        session_id: int,
        prepared: _Prepared,
        *,
        revision: int,
        receipt: GuidedReceiptModel | None,
    ) -> GuidedDialogueResponse:
        session = self.geo.get(session_id)
        if session is None:
            raise GuidedNotFound("会话不存在")
        state = prepared.state
        confirmed = state.get("confirmed") or {}
        dialogue = GuidedDialogueModel(
            version=DIALOGUE_VERSION,
            revision=revision,
            stage=prepared.stage,
            confirmed=ConfirmedPathModel(
                country=(
                    self._upgrade_target_provenance(confirmed["country"])
                    if isinstance(confirmed, Mapping) and isinstance(confirmed.get("country"), Mapping)
                    else None
                ),
                admin1=(
                    self._upgrade_target_provenance(confirmed["admin1"])
                    if isinstance(confirmed, Mapping) and isinstance(confirmed.get("admin1"), Mapping)
                    else None
                ),
                city=(
                    self._upgrade_target_provenance(confirmed["city"])
                    if isinstance(confirmed, Mapping) and isinstance(confirmed.get("city"), Mapping)
                    else None
                ),
            ),
            exclusions=[
                GuidedLocationTarget(**self._upgrade_target_provenance(item))
                for item in state.get("exclusions") or []
                if isinstance(item, Mapping)
            ],
            clues=[
                ReviewedClueModel(
                    id=str(item.get("id")),
                    dimension=str(item.get("dimension")),
                    value=str(item.get("value") or ""),
                    description=str(item.get("description") or ""),
                    source=str(item.get("source") or ""),
                    confidence=_clamp_confidence(item.get("confidence")),
                    status=str(item.get("status") or "pending"),
                )
                for item in state.get("clues") or []
                if isinstance(item, Mapping)
                and item.get("id")
                and item.get("dimension")
            ],
            pending_question=_question_model(state.get("pending_question")),
        )
        score_models = [
            CountryScoreModel(**score.to_dict())
            for score in prepared.scores
        ]
        clue_models = [
            ClueModel(**dict(item))
            for item in self._public_clue_rows(state, session)
        ]
        candidate_models = []
        for item in prepared.candidates:
            candidate = dict(item)
            if isinstance(candidate.get("target"), Mapping):
                candidate["target"] = self._upgrade_target_provenance(candidate["target"])
            candidate_models.append(GuidedCandidateModel(**candidate))
        return GuidedDialogueResponse(
            dialogue=dialogue,
            candidates=candidate_models,
            action=prepared.action,
            confidence=ConfidenceRouteModel(**prepared.confidence) if prepared.confidence else None,
            messages=[ChatMessageModel(**item) for item in self.geo.list_messages(session_id)],
            scores=score_models,
            clues=clue_models,
            exclusions=list(session.exclusions or []),
            receipt=receipt,
            inference_notice=prepared.inference_notice,
        )


__all__ = [
    "GuidedConflict",
    "GuidedDialogueService",
    "GuidedNotFound",
    "GuidedUnavailable",
    "GuidedValidationError",
]
