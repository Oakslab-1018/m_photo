"""Persisted country -> admin1 -> city guided dialogue.

The service is deliberately separate from ``services.chat``. Guided free text is
parsed into proposals/reviewable clues only; it never calls the legacy feedback
path and therefore cannot trigger that path's explicit country confirmation.

All location identities are resolved against local catalogs. The optional city
model ranking is an enrichment layered on top of the complete admin1-scoped
catalog; it is never used before that scope filter and catalog-only rows never
receive invented scores.
"""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from services.chat import ChineseFreeTextParser, PendingQuestion

from ..config import Settings
from ..database import GeoSession, GuidedDialogue, GuidedEvent
from ..domain.geolocation.clue import Clue, DIM_LABELS, dicts_to_clues
from ..domain.geolocation.country_policy import (
    canonical_country_iso,
    country_members,
    is_excluded,
    normalize_iso,
    source_region_iso,
)
from ..domain.geolocation.engine import find_exact_phrase, normalize_place_text
from ..domain.geolocation.model_ranking import ModelAwareEngine
from ..schemas import (
    ActionModel,
    ChatMessageModel,
    ClueModel,
    ConfirmedPathModel,
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
from .city_reasoning import CityClueReasoner
from .geo_session import GeoSessionService
from .model_gateway import StreetClipGateway

DIALOGUE_VERSION = 1
COUNTRY_CANDIDATE_LIMIT = 20
COUNTRY_CATALOG_FALLBACK_LIMIT = 50
CITY_CANDIDATE_LIMIT = 20
PROPOSAL_LIMIT = 20
MAX_QUESTION_OPTIONS = 5

_CITY_ID_RE = re.compile(r"^[1-9][0-9]*$")
_ADMIN1_ID_RE = re.compile(r"^([A-Z]{2})\.(.+)$")
_UPPER_ISO_RE = re.compile(r"(?<![A-Za-z])([A-Z]{2})(?![A-Za-z])")
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

# Explicit, catalog-validated Chinese aliases for guided city proposals. The
# legacy parser's alias map is country-level and intentionally not reused as a
# location confirmation source. These IDs are checked against the read-only
# SQLite city catalog and still require an explicit confirm_location event.
_GUIDED_CITY_ALIAS_IDS = {
    "杭州": "1808926",
}


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
        "proposals": [],
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
    for key in ("exclusions", "clues", "asked_question_ids", "proposals"):
        raw = value.get(key)
        if isinstance(raw, list):
            result[key] = [dict(item) if isinstance(item, Mapping) else item for item in raw]
    question = value.get("pending_question")
    result["pending_question"] = dict(question) if isinstance(question, Mapping) else None
    conflict = value.get("conflict")
    result["conflict"] = dict(conflict) if isinstance(conflict, Mapping) else None
    correction = value.get("pending_correction")
    result["pending_correction"] = dict(correction) if isinstance(correction, Mapping) else None
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
    return GuidedQuestionModel(
        id=str(question.get("id") or ""),
        kind=str(question.get("kind") or "request_clue"),
        message=str(question.get("message") or "请进一步说明地点。"),
        dimension=str(question["dimension"]) if question.get("dimension") else None,
        clue_id=str(question["clue_id"]) if question.get("clue_id") else None,
        options=options,
    )


@dataclass
class _Prepared:
    state: dict[str, Any]
    stage: str
    candidates: list[dict[str, Any]]
    scores: list[Any]
    action: ActionModel
    legacy_confirmed_iso: str | None
    legacy_status: str


class GuidedDialogueService:
    """Request-scoped transactional guided-dialogue service."""

    def __init__(self, geo: GeoSessionService, settings: Settings) -> None:
        self.geo = geo
        self.settings = settings
        self.parser = ChineseFreeTextParser()
        self.gateway = StreetClipGateway(settings)
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
            try:
                with self.geo.transaction():
                    result = self.geo.session.execute(
                        update(GuidedDialogue)
                        .where(
                            GuidedDialogue.session_id == session_id,
                            GuidedDialogue.revision == row.revision,
                        )
                        .values(
                            revision=row.revision + 1,
                            stage=prepared.stage,
                            state_json=json.dumps(prepared.state, ensure_ascii=False, allow_nan=False),
                            updated_at=datetime.utcnow(),
                        )
                    )
                    if result.rowcount != 1:
                        raise _ReconcileRetry
                    response = self._response(session_id, prepared, revision=row.revision + 1, receipt=None)
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

        if int(event.expected_revision) != int(row.revision):
            raise GuidedConflict("对话版本已过期，请重新加载后再试")

        state = _normalise_state(row.state)
        state, evidence_changed = self._reconcile_session_evidence(state, session)
        if evidence_changed:
            raise GuidedConflict("底层证据已更新，请重新加载对话后再试")
        outcome = self._apply_event(state, session, event)
        query = None
        prepared = self._prepare(
            outcome["state"],
            session,
            query=query,
            ensure_question=bool(outcome["effective"]),
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
        audit = {
            "payload": event_payload,
            "fingerprint": _event_fingerprint(event),
            "validated_target": audit_target,
            "clue_id": event.clue_id,
            "level": event.level,
            "proposals": prepared.state.get("proposals") or [],
        }
        if event.type == "free_text" and event.text:
            user_content = event.text.strip()
        elif event.type == "answer" and event.value:
            user_content = f"回答：{event.value.strip()}"
        else:
            user_content = outcome["summary"]
        new_revision = row.revision + (1 if outcome["effective"] else 0)
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
                        GuidedDialogue.revision == row.revision,
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
                        "clue_id": event.clue_id,
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
                        "clue_id": event.clue_id,
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
                    revisions.get("admin1"),
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
                place = self._store().get_place("city", str(result.get("id") or ""))
            else:
                place = None
        except Exception as exc:
            raise self._unavailable(exc) from exc
        if place is None:
            raise GuidedUnavailable("地理目录条目缺失")
        fields = self._coordinate_fields(place)
        if result.get("lat") is None:
            result["lat"] = fields["lat"]
        if result.get("lng") is None:
            result["lng"] = fields["lng"]
        result.update(fields["provenance"])
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

    def _city_target(self, row: CityRow, *, label: str | None = None) -> dict[str, Any]:
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
            "label": label or row.name,
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
        changed = (
            str(state.get("session_evidence_fingerprint") or "") != fingerprint
            or list(state.get("clues") or []) != clues
        )
        state["clues"] = clues
        state["session_evidence_fingerprint"] = fingerprint
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
    ) -> list[Any]:
        engine = ModelAwareEngine()
        if session.model_evidence is not None:
            try:
                engine.evidence = self.gateway.validate(session.model_evidence)
            except Exception:
                engine.evidence = None
        guided_country_exclusions = [
            str(target.get("id"))
            for target in state.get("exclusions") or []
            if isinstance(target, Mapping) and target.get("level") == "country"
        ]
        exclusions = list(session.exclusions or []) + guided_country_exclusions
        return engine.infer(clues, exclusions=exclusions, top_n=COUNTRY_CANDIDATE_LIMIT)

    # ------------------------------------------------------------- candidates
    @staticmethod
    def _candidate(target: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        result = {"target": dict(target)}
        for key in ("score", "score_type", "reason"):
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
        catalog: list[dict[str, Any]] = []
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

    def _admin1_candidates(
        self,
        state: Mapping[str, Any],
        proposals: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        confirmed = state.get("confirmed") or {}
        country = confirmed.get("country") if isinstance(confirmed, Mapping) else None
        if not isinstance(country, Mapping):
            return []
        canonical = canonical_country_iso(country.get("id"))
        candidates: list[dict[str, Any]] = []
        for source_iso in country_members(canonical):
            for row in self._admin_rows(source_iso):
                target = self._admin1_target(row, canonical)
                if self._excluded_by_state(state, target) is not None:
                    continue
                candidates.append(self._candidate(target, reason="官方一级行政区目录"))
        candidates.sort(key=lambda item: (
            str(item["target"].get("source_iso") or ""),
            str(item["target"].get("admin1_id") or ""),
        ))
        proposed = [self._candidate(target, reason="自由文本提议") for target in proposals
                    if isinstance(target, Mapping) and target.get("level") == "admin1"]
        return self._merge_candidates(proposed, candidates)

    def _selected_admin1(self, state: Mapping[str, Any]) -> dict[str, Any] | None:
        confirmed = state.get("confirmed") or {}
        target = confirmed.get("admin1") if isinstance(confirmed, Mapping) else None
        return dict(target) if isinstance(target, Mapping) else None

    def _city_rows_scoped(self, state: Mapping[str, Any]) -> list[CityRow]:
        admin1 = self._selected_admin1(state)
        if admin1 is None:
            return []
        source_iso = normalize_iso(admin1.get("source_iso")) or normalize_iso(str(admin1.get("admin1_id") or "").split(".", 1)[0])
        admin1_id = str(admin1.get("admin1_id") or admin1.get("id") or "")
        if not source_iso or not admin1_id:
            return []
        return list(self._city_rows_for_scope(source_iso, admin1_id).values())

    def _ranked_city_rows(
        self,
        state: Mapping[str, Any],
        session: GeoSession,
        clues: Sequence[Clue],
    ) -> dict[str, Mapping[str, Any]]:
        admin1 = self._selected_admin1(state)
        if admin1 is None:
            return {}
        source_iso = normalize_iso(admin1.get("source_iso"))
        image_path = Path(session.image_path)
        if not source_iso or not image_path.exists():
            return {}
        try:
            result = self.gateway.rank_cities(image_path, source_iso)
        except Exception:
            result = None
        if not result:
            return {}
        rows = [dict(row) for row in result.get("city_scores") or [] if isinstance(row, Mapping)]
        scoped_rows = self._city_rows_for_scope(source_iso, str(admin1.get("id") or admin1.get("admin1_id") or ""))
        catalog = {source_iso: scoped_rows}
        if scoped_rows and clues:
            try:
                reranked = CityClueReasoner(catalog).rerank(rows, clues, iso=source_iso)
                if reranked is not None:
                    rows = list(reranked.city_scores)
            except (KeyError, TypeError, ValueError):
                pass
        return {str(row.get("city_id")): row for row in rows if row.get("city_id") is not None}

    def _city_candidates(
        self,
        state: Mapping[str, Any],
        session: GeoSession,
        clues: Sequence[Clue],
        *,
        query: str | None,
        proposals: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        rows = self._city_rows_scoped(state)
        if not rows:
            return []
        ranks = self._ranked_city_rows(state, session, clues)
        query_key = normalize_place_text(query or "")
        alias_row_ids: set[int] = set()
        if query_key:
            for alias, city_id in _GUIDED_CITY_ALIAS_IDS.items():
                alias_key = normalize_place_text(alias)
                if alias_key and (alias_key == query_key or alias_key in query_key or query_key in alias_key):
                    alias_row_ids.add(int(city_id))
        scoped_rows: list[tuple[CityRow, Mapping[str, Any] | None]] = []
        for row in rows:
            if self._excluded_by_state(state, self._city_target(row)) is not None:
                continue
            if query_key and row.city_id not in alias_row_ids and query_key not in normalize_place_text(row.name):
                continue
            scoped_rows.append((row, ranks.get(str(row.city_id))))

        proposed = []
        selected_admin1_id = str((self._selected_admin1(state) or {}).get("id") or "")
        for target in proposals:
            if not isinstance(target, Mapping) or target.get("level") != "city":
                continue
            if selected_admin1_id and str(target.get("admin1_id") or "") != selected_admin1_id:
                continue
            if self._excluded_by_state(state, target) is not None:
                continue
            proposed.append(self._candidate(target, reason="自由文本提议"))

        candidates: list[dict[str, Any]] = []
        for row, ranked in scoped_rows:
            payload = self._candidate(
                self._city_target(row),
                score=ranked.get("score") if isinstance(ranked, Mapping) else None,
                score_type=ranked.get("score_type") if isinstance(ranked, Mapping) else None,
                reason=(
                    "图片城市排序"
                    if isinstance(ranked, Mapping)
                    else ("目录检索匹配" if query_key else "城市目录")
                ),
            )
            candidates.append(payload)

        def sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
            target = item.get("target") or {}
            row = next((r for r in rows if str(r.city_id) == str(target.get("id"))), None)
            label_key = normalize_place_text(str(target.get("label") or ""))
            exact = 0 if query_key and (label_key == query_key or (row is not None and row.city_id in alias_row_ids)) else 1
            try:
                score = -float(item.get("score"))
            except (TypeError, ValueError):
                score = 1.0
            population = -(row.population if row is not None else 0)
            return (exact, score, population, str(target.get("id") or ""))

        candidates.sort(key=sort_key)
        return self._merge_candidates(proposed, candidates)[:CITY_CANDIDATE_LIMIT]

    # --------------------------------------------------------------- target
    def _resolve_target(self, raw: GuidedEventTarget | Mapping[str, Any]) -> dict[str, Any]:
        level = str(getattr(raw, "level", None) or (raw.get("level") if isinstance(raw, Mapping) else "") or "")
        identifier = str(getattr(raw, "id", None) or (raw.get("id") if isinstance(raw, Mapping) else "") or "").strip()
        if level == "country":
            iso = normalize_iso(identifier)
            if not iso or canonical_country_iso(iso) != iso or not self._country_exists(iso):
                raise GuidedValidationError("未知国家 ID")
            return self._country_target(iso)
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
        admin1 = confirmed.get("admin1") if isinstance(confirmed, Mapping) else None
        if level == "country":
            if isinstance(country, Mapping) and str(country.get("id")) != str(target.get("id")):
                raise GuidedValidationError("确认其他国家前请先回退")
            return
        if not isinstance(country, Mapping):
            raise GuidedValidationError("请先确认国家，再确认一级行政区或城市")
        if str(target.get("country_iso")) != str(country.get("id")):
            raise GuidedValidationError("目标国家与已确认国家不一致")
        if level == "admin1":
            if isinstance(admin1, Mapping) and str(admin1.get("id")) != str(target.get("id")):
                raise GuidedValidationError("确认其他一级行政区前请先回退")
            return
        if level == "city":
            if not isinstance(admin1, Mapping):
                raise GuidedValidationError("请先确认一级行政区，再确认城市")
            if str(target.get("admin1_id")) != str(admin1.get("id")):
                raise GuidedValidationError("城市不属于已确认的一级行政区")
            if str(target.get("country_iso")) != str(admin1.get("country_iso")):
                raise GuidedValidationError("城市的国家归属不一致")
            return
        raise GuidedValidationError("未知地点层级")

    def _validate_active_scope(self, state: Mapping[str, Any], target: Mapping[str, Any]) -> None:
        """Reject exclusions whose ancestry is outside the active guided path."""

        level = str(target.get("level") or "")
        confirmed = state.get("confirmed") or {}
        country = confirmed.get("country") if isinstance(confirmed, Mapping) else None
        admin1 = confirmed.get("admin1") if isinstance(confirmed, Mapping) else None
        if not isinstance(country, Mapping):
            if level != "country":
                raise GuidedValidationError("尚未确认国家，不能排除下级地点")
            return
        if canonical_country_iso(target.get("country_iso")) != str(country.get("id")):
            raise GuidedValidationError("排除地点不属于当前已确认国家")
        if isinstance(admin1, Mapping):
            if level == "admin1" and str(target.get("id")) != str(admin1.get("id")):
                raise GuidedValidationError("排除的行政区不属于当前已确认行政区")
            if level == "city" and str(target.get("admin1_id")) != str(admin1.get("id")):
                raise GuidedValidationError("排除的城市不属于当前已确认行政区")
        if level == "city":
            admin1_id = target.get("admin1_id")
            if not admin1_id or self._admin_by_id(str(admin1_id)) is None:
                raise GuidedValidationError("城市缺少官方一级行政区归属，不能执行排除")

    @staticmethod
    def _excluded_by_state(
        state: Mapping[str, Any],
        target: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        for exclusion in state.get("exclusions") or []:
            if not isinstance(exclusion, Mapping):
                continue
            level = str(exclusion.get("level") or "")
            if level == "country" and target.get("country_iso") == exclusion.get("id"):
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
        if isinstance(confirmed, Mapping) and isinstance(confirmed.get("admin1"), Mapping):
            return "city"
        if isinstance(confirmed, Mapping) and isinstance(confirmed.get("country"), Mapping):
            return "admin1"
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
    def _prepare(
        self,
        state: dict[str, Any],
        session: GeoSession,
        *,
        query: str | None,
        ensure_question: bool = True,
    ) -> _Prepared:
        state = _normalise_state(state)
        clues = self._ranking_clues(state)
        scores = self._country_scores(state, session, clues)
        stage = self._stage_for_state(state)
        proposals = [dict(item) for item in state.get("proposals") or [] if isinstance(item, Mapping)]
        country_candidates = self._country_candidates(state, scores, query=query)
        if stage == "country":
            proposed_countries = [
                self._candidate(target, reason="自由文本提议")
                for target in proposals
                if target.get("level") == "country"
                and self._excluded_by_state(state, target) is None
            ]
            country_candidates = self._merge_candidates(proposed_countries, country_candidates)
        admin1_candidates = self._admin1_candidates(state, proposals) if stage in {"admin1", "city", "resolved"} else []
        city_candidates = (
            self._city_candidates(state, session, clues, query=query, proposals=proposals)
            if stage in {"city", "resolved"}
            else []
        )
        question = (
            self._ensure_question(state, country_candidates, admin1_candidates, city_candidates)
            if ensure_question
            else state.get("pending_question")
        )
        action = self._action_for_state(state, question)
        legacy_iso, legacy_status = self._legacy_fields(state)
        return _Prepared(
            state=state,
            stage=stage,
            candidates=(
                country_candidates if stage == "country"
                else admin1_candidates if stage == "admin1"
                else city_candidates
            ),
            scores=scores,
            action=action,
            legacy_confirmed_iso=legacy_iso,
            legacy_status=legacy_status,
        )

    # -------------------------------------------------------------- questions
    def _question_is_valid(self, state: Mapping[str, Any], question: Mapping[str, Any]) -> bool:
        kind = str(question.get("kind") or "")
        question_id = str(question.get("id") or "")
        asked = {str(item) for item in state.get("asked_question_ids") or []}
        if not question_id or question_id in asked:
            return False
        if kind == "review_clue":
            clue_id = str(question.get("clue_id") or "")
            for record in state.get("clues") or []:
                if isinstance(record, Mapping) and str(record.get("id")) == clue_id:
                    return str(record.get("status") or "") == "pending"
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
            parts.append("替换线索维度：" + "、".join(str(item) for item in replace_dimensions))
        if remove_clue_values:
            parts.append("移除线索值：" + "、".join(
                f"{item[0]}={item[1]}" for item in remove_clue_values
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

    def _ensure_question(
        self,
        state: dict[str, Any],
        country_candidates: Sequence[Mapping[str, Any]],
        admin1_candidates: Sequence[Mapping[str, Any]],
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
            if not clue_id or f"review:{clue_id}" in asked:
                continue
            dimension = str(record.get("dimension") or "")
            value = str(record.get("value") or "")
            question = {
                "id": f"review:{clue_id}",
                "kind": "review_clue",
                "message": f"请复核线索：{DIM_LABELS.get(dimension, dimension)}={value}。这条线索是否正确？",
                "dimension": dimension,
                "clue_id": clue_id,
                "options": [
                    {"value": "confirmed", "label": "确认"},
                    {"value": "rejected", "label": "否定"},
                ],
            }
            state["pending_question"] = question
            return question

        candidate_sets = {
            "country": list(country_candidates),
            "admin1": list(admin1_candidates),
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
            return self._apply_free_text(state, session, event)
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
        self._check_question_reference(
            state,
            event,
            allowed_kinds={"confirm_location"},
            target={"level": event.target.level, "id": event.target.id},
        )
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
        elif level == "admin1":
            new_confirmed["admin1"] = dict(target)
            new_confirmed["city"] = None
        else:
            new_confirmed["city"] = dict(target)
        state["confirmed"] = new_confirmed
        state["pending_question"] = None
        state["proposals"] = []
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
                state["pending_question"] = None
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
        self._refresh_conflict(state)
        return self._outcome(state, True, "applied", f"已恢复「{target.get('label')}」的候选资格。")

    def _apply_backtrack(
        self,
        state: dict[str, Any],
        event: GuidedEventRequest,
    ) -> dict[str, Any]:
        level = event.level
        if level not in {"country", "admin1", "city"}:
            raise GuidedValidationError("缺少回退层级")
        confirmed = state.get("confirmed") or {}
        if level == "country":
            changed = any(isinstance(confirmed.get(key), Mapping) for key in ("country", "admin1", "city"))
            new_confirmed = {"country": None, "admin1": None, "city": None}
        elif level == "admin1":
            changed = any(isinstance(confirmed.get(key), Mapping) for key in ("admin1", "city"))
            new_confirmed = {
                "country": confirmed.get("country") if isinstance(confirmed, Mapping) else None,
                "admin1": None,
                "city": None,
            }
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
        self._refresh_conflict(state)
        return self._outcome(state, True, "applied", f"已回退到{_LEVEL_LABELS[level]}阶段。")

    def _apply_free_text(
        self,
        state: dict[str, Any],
        session: GeoSession,
        event: GuidedEventRequest,
    ) -> dict[str, Any]:
        text = (event.text or "").strip()
        if not text:
            raise GuidedValidationError("缺少自由文本内容")
        parsed = self.parser.parse(text, pending=self._parser_pending(state.get("pending_question")))
        added: list[str] = []
        existing_ids = {
            str(record.get("id")) for record in state.get("clues") or []
            if isinstance(record, Mapping) and record.get("id")
        }
        for raw in parsed.clues:
            if not isinstance(raw, Mapping):
                continue
            record = _reviewed_record(raw)
            if str(record["id"]) in existing_ids:
                continue
            record["source"] = "user"
            extra = dict(record.get("_extra") or {})
            extra["parser"] = "guided_rules"
            record["_extra"] = extra
            record["_origin"] = "guided_user"
            state.setdefault("clues", []).append(record)
            existing_ids.add(str(record["id"]))
            added.append(str(record.get("id")))

        proposals = self._propose_targets(state, text, parsed.clues)
        conflict = self._detect_conflict(state, proposals)
        correction = self._correction_plan(text, parsed, added)
        old_proposals = [dict(item) for item in state.get("proposals") or [] if isinstance(item, Mapping)]
        old_conflict = dict(state.get("conflict")) if isinstance(state.get("conflict"), Mapping) else None
        old_correction = dict(state.get("pending_correction")) if isinstance(state.get("pending_correction"), Mapping) else None
        effective = bool(
            added
            or proposals != old_proposals
            or conflict != old_conflict
            or correction is not None
            or old_correction is not None
        )
        if not effective:
            return self._outcome(state, False, "unrecognized", "没有识别到新的目录证据、地点提议或修正计划。")
        state["proposals"] = proposals
        state["conflict"] = conflict
        state["pending_correction"] = correction
        state["last_text"] = text
        state["pending_question"] = None
        if correction is not None:
            summary = "检测到可能的排除或修正，需要你确认是否应用。"
        elif conflict is not None:
            summary = str(conflict.get("message") or "线索与已确认路径冲突，未自动改变路径。")
        elif proposals:
            summary = "已记录自由文本提议；必须明确确认地点后才会推进。"
        elif added:
            summary = "已记录待复核的自由文本线索；没有确认任何地点。"
        else:
            summary = "没有识别到新的目录证据或地点提议。"
        return self._outcome(state, True, "applied", summary)

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

    def _correction_targets_for_iso(self, value: object) -> list[dict[str, Any]]:
        raw = normalize_iso(value)
        canonical = canonical_country_iso(raw)
        if not raw or not canonical or not self._country_exists(canonical):
            raise GuidedValidationError("修正计划包含未知国家或地区")
        if raw != canonical:
            rows = self._admin_rows(raw)
            if not rows:
                raise GuidedValidationError("该地区没有可用的官方一级行政区，不能应用排除修正")
            return [self._admin1_target(row, canonical) for row in rows]
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
                        effective = True
                    break
        asked = list(state.get("asked_question_ids") or [])
        if question_id and question_id not in asked:
            asked.append(question_id)
            effective = True
        state["asked_question_ids"] = asked
        state["pending_question"] = None
        if question.get("kind") == "ask":
            state["conflict"] = None
            state["proposals"] = []
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
        if level == "country" and isinstance(confirmed.get("country"), Mapping) and confirmed["country"].get("id") == exclusion.get("id"):
            state["confirmed"] = {"country": None, "admin1": None, "city": None}
        elif level == "admin1" and isinstance(confirmed.get("admin1"), Mapping) and confirmed["admin1"].get("id") == exclusion.get("id"):
            state["confirmed"] = {"country": confirmed.get("country"), "admin1": None, "city": None}
        elif level == "city" and isinstance(confirmed.get("city"), Mapping) and confirmed["city"].get("id") == exclusion.get("id"):
            state["confirmed"] = {"country": confirmed.get("country"), "admin1": confirmed.get("admin1"), "city": None}
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
        proposals: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        def add(target: Mapping[str, Any] | None) -> None:
            if not isinstance(target, Mapping):
                return
            key = _target_key(target)
            if key in seen:
                return
            seen.add(key)
            proposals.append(dict(target))

        for raw in parsed_clues:
            if not isinstance(raw, Mapping) or raw.get("dimension") != "place_name":
                continue
            value = str(raw.get("value") or "").strip()
            if not value:
                continue
            iso = normalize_iso(value) if value.isascii() and value.isalpha() and value == value.upper() else ""
            if iso and iso == canonical_country_iso(iso) and self._country_exists(iso):
                add(self._country_target(iso))
            for target in self._match_admin1_text(state, value):
                add(target)
            for target in self._match_city_text(state, value):
                add(target)
                parent = target.get("admin1_id")
                if parent:
                    row = self._admin_by_id(str(parent))
                    if row is not None:
                        add(self._admin1_target(row, target.get("country_iso")))

        normalized = normalize_place_text(text)
        if normalized:
            for token in _UPPER_ISO_RE.findall(text):
                iso = normalize_iso(token)
                if iso and iso == canonical_country_iso(iso) and self._country_exists(iso):
                    add(self._country_target(iso))
            for country in self._country_records():
                iso = canonical_country_iso(country.get("iso"))
                if not iso or not self._country_exists(iso):
                    continue
                for label in (country.get("name"), country.get("name_zh")):
                    if label and find_exact_phrase(normalized, normalize_place_text(str(label))):
                        add(self._country_target(iso))
                        break
            for target in self._match_admin1_text(state, text):
                add(target)
            for target in self._match_city_text(state, text):
                add(target)
                parent = target.get("admin1_id")
                if parent:
                    row = self._admin_by_id(str(parent))
                    if row is not None:
                        add(self._admin1_target(row, target.get("country_iso")))
        return proposals[:PROPOSAL_LIMIT]

    def _all_admin1_rows(self) -> list[Any]:
        try:
            store = self._store()
            rows: list[Any] = []
            offset = 0
            page_size = 2000
            while True:
                page = store.query_places(
                    "admin1",
                    limit=page_size,
                    offset=offset,
                    coordinates_only=False,
                )
                batch = list(page.get("rows") or [])
                rows.extend(batch)
                total = int(page.get("total") or 0)
                offset += len(batch)
                if not batch or offset >= total:
                    return rows
        except Exception as exc:
            raise self._unavailable(exc) from exc

    def _match_admin1_text(self, state: Mapping[str, Any], text: str) -> list[dict[str, Any]]:
        normalized = normalize_place_text(text)
        if not normalized:
            return []
        confirmed = state.get("confirmed") or {}
        country = confirmed.get("country") if isinstance(confirmed, Mapping) else None
        country_iso = canonical_country_iso(country.get("id")) if isinstance(country, Mapping) else ""
        source_isos = country_members(country_iso) if country_iso else sorted({
            self._admin_iso(row) for row in self._all_admin1_rows() if self._admin_iso(row)
        })
        result: list[dict[str, Any]] = []
        for source_iso in source_isos:
            for row in self._admin_rows(source_iso):
                for label in self._admin_aliases(row):
                    if label and find_exact_phrase(normalized, normalize_place_text(label)):
                        result.append(self._admin1_target(row, country_iso or source_iso))
                        break
        return result

    def _match_city_text(self, state: Mapping[str, Any], text: str) -> list[dict[str, Any]]:
        normalized = normalize_place_text(text)
        if not normalized:
            return []
        confirmed = state.get("confirmed") or {}
        country = confirmed.get("country") if isinstance(confirmed, Mapping) else None
        if not isinstance(country, Mapping):
            return []
        selected = self._selected_admin1(state)
        selected_source = normalize_iso(selected.get("source_iso")) if isinstance(selected, Mapping) else ""
        selected_id = str(selected.get("id") or selected.get("admin1_id") or "") if isinstance(selected, Mapping) else ""
        result: list[dict[str, Any]] = []
        for source_iso in country_members(canonical_country_iso(country.get("id"))):
            if not source_iso:
                continue
            if selected_source == source_iso and selected_id:
                scope = self._city_rows_for_scope(source_iso, selected_id)
            else:
                scope = self._city_rows_for_source(source_iso)
            for alias, city_id in _GUIDED_CITY_ALIAS_IDS.items():
                if find_exact_phrase(normalized, normalize_place_text(alias)):
                    row = scope.get(int(city_id))
                    if row is not None:
                        result.append(self._city_target(row))
            for row in scope.values():
                if find_exact_phrase(normalized, normalize_place_text(row.name)):
                    result.append(self._city_target(row))
        return result

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
        admin1 = confirmed.get("admin1") if isinstance(confirmed, Mapping) else None
        city = confirmed.get("city") if isinstance(confirmed, Mapping) else None
        conflicts: list[dict[str, Any]] = []
        proposal_countries = {
            canonical_country_iso(item.get("country_iso"))
            for item in proposals if isinstance(item, Mapping)
        }
        proposal_countries.discard("")
        if len(proposal_countries) > 1:
            conflicts.append({"reason": "multiple countries proposed", "proposed": list(proposals)})
        for item in proposals:
            if not isinstance(item, Mapping):
                continue
            if isinstance(country, Mapping) and canonical_country_iso(item.get("country_iso")) != str(country.get("id")):
                conflicts.append({"reason": "different country proposed", "proposed": dict(item)})
                continue
            if isinstance(admin1, Mapping):
                if item.get("level") == "admin1" and item.get("id") != admin1.get("id"):
                    conflicts.append({"reason": "different admin1 proposed", "proposed": dict(item)})
                if item.get("level") == "city" and item.get("admin1_id") != admin1.get("id"):
                    conflicts.append({"reason": "city outside confirmed admin1 proposed", "proposed": dict(item)})
            if isinstance(city, Mapping) and item.get("level") == "city" and item.get("id") != city.get("id"):
                conflicts.append({"reason": "different city proposed", "proposed": dict(item)})
        if not conflicts:
            return None
        conflict_id = _json_hash("conflict", conflicts)
        return {
            "id": conflict_id,
            "message": "文本与已确认路径冲突，请先澄清；已确认路径不会自动改变。",
            "proposed": [dict(item) for item in proposals],
            "confirmed": dict(confirmed),
            "reasons": [item["reason"] for item in conflicts],
        }

    # -------------------------------------------------------------- response
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
            for item in session.clues or []
            if isinstance(item, Mapping) and item.get("dimension") and item.get("value") is not None
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
            messages=[ChatMessageModel(**item) for item in self.geo.list_messages(session_id)],
            scores=score_models,
            clues=clue_models,
            exclusions=list(session.exclusions or []),
            receipt=receipt,
        )


__all__ = [
    "GuidedConflict",
    "GuidedDialogueService",
    "GuidedNotFound",
    "GuidedUnavailable",
    "GuidedValidationError",
]
