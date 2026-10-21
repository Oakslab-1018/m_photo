"""Validation and provenance helpers for adopting persisted region OCR text.

The frontend submits only ``report_id`` and ``observation_id``.  This module
looks those identifiers up in chat messages belonging to the current session,
validates the stored point-analysis OCR report, and returns a bounded plain
data object that can be used to build a :class:`PlaceMention` without trusting
client-supplied text or location fields.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

MAX_ADOPTED_OBSERVATION_TEXT = 2000
_BLOCKED_OCR_STATES = frozenset({"mock", "disabled", "unavailable"})


class ObservationAdoptionError(ValueError):
    """A stored observation cannot be adopted by the current session."""


def _clean_identifier(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 120:
        raise ObservationAdoptionError(f"{label}无效")
    return text


def _bounded_bbox(value: object) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        x = float(value.get("x"))
        y = float(value.get("y"))
        w = float(value.get("w"))
        h = float(value.get("h"))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x, y, w, h)):
        return None
    if x < 0.0 or y < 0.0 or w <= 0.0 or h <= 0.0:
        return None
    if x + w > 1.0 + 1e-9 or y + h > 1.0 + 1e-9:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def _bounded_bbox_px(value: object) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return []
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return []


def _bounded_confidence(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        return None
    return number

@dataclass(frozen=True)
class AdoptedOcrObservation:
    """A validated OCR observation copied from one persisted AI report."""

    report_id: str
    observation_id: str
    text: str
    provider: str
    origin_source: str
    evidence: str
    bbox_norm: dict[str, float] | None
    quad_norm: dict[str, float] | None
    report_bbox_norm: dict[str, float] | None
    bbox_px: list[int]
    ocr_provider: str
    ocr_state: str
    confidence: float | None
    message_id: int | None

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def audit_payload(self) -> dict[str, Any]:
        """Return bounded source evidence for the event ledger."""

        return {
            "report_id": self.report_id,
            "observation_id": self.observation_id,
            "kind": "ocr_text",
            "text": self.text,
            "text_sha256": self.text_sha256,
            "provider": self.provider,
            "ocr_provider": self.ocr_provider,
            "ocr_state": self.ocr_state,
            "origin_source": self.origin_source,
            "bbox_norm": dict(self.bbox_norm) if self.bbox_norm else None,
            "quad_norm": dict(self.quad_norm) if self.quad_norm else None,
            "message_id": self.message_id,
        }

    def merge_into_extra(
        self,
        extra: object,
        *,
        event_id: str,
        place_term: str,
    ) -> tuple[dict[str, Any], bool]:
        """Merge adoption provenance into one reviewed clue's ``_extra``.

        The direct source fields describe the observation adopted by this
        event. ``adopted_observations`` retains a small deduplicated history so
        the same stored observation can be recognized after unrelated clues or
        provider metadata change.
        """

        current = dict(extra) if isinstance(extra, Mapping) else {}
        before = dict(current)
        history = [
            dict(item)
            for item in current.get("adopted_observations") or []
            if isinstance(item, Mapping)
        ]
        source_entry = {
            "event_id": str(event_id),
            "report_id": self.report_id,
            "observation_id": self.observation_id,
            "kind": "ocr_text",
            "origin_source": self.origin_source,
            "provider": self.provider,
            "bbox_norm": dict(self.bbox_norm) if self.bbox_norm else None,
            "quad_norm": dict(self.quad_norm) if self.quad_norm else None,
            "text_sha256": self.text_sha256,
        }
        source_key = (self.report_id, self.observation_id, self.text_sha256)
        if not any(
            (
                str(item.get("report_id") or ""),
                str(item.get("observation_id") or ""),
                str(item.get("text_sha256") or ""),
            ) == source_key
            for item in history
        ):
            history.append(source_entry)
        current["adopted_observations"] = history[-8:]
        current.update({
            "adoption": "user_confirmed_region_observation",
            "adoption_origin_source": "region_observation",
            "adopted_text": self.text,
            "observation_kind": "ocr_text",
            "region_id": self.report_id,
            "report_id": self.report_id,
            "observation_id": self.observation_id,
            "origin_source": self.origin_source,
            "provider": self.provider,
            "ocr_provider": self.ocr_provider,
            "ocr_state": self.ocr_state,
            "bbox": (
                dict(self.bbox_norm or self.quad_norm or self.report_bbox_norm)
                if (self.bbox_norm or self.quad_norm or self.report_bbox_norm)
                else None
            ),
            "bbox_norm": dict(self.bbox_norm) if self.bbox_norm else None,
            "quad_norm": dict(self.quad_norm) if self.quad_norm else None,
            "region_bbox_norm": dict(self.report_bbox_norm) if self.report_bbox_norm else None,
            "bbox_px": list(self.bbox_px),
            "user_confirmed": True,
            "user_confirmed_provenance": source_entry,
            "place_quote": self.text[:200],
        })
        existing_terms = [
            str(item).strip()
            for item in current.get("place_terms") or []
            if str(item).strip()
        ]
        normalized_terms = {item.casefold() for item in existing_terms}
        term = str(place_term or "").strip()
        if term and term.casefold() not in normalized_terms:
            existing_terms.append(term)
        if existing_terms:
            current["place_terms"] = existing_terms[:4]
        return current, current != before

    def matches_extra(self, extra: object) -> bool:
        """True when ``extra`` already records this exact adopted text source."""

        current = extra if isinstance(extra, Mapping) else {}
        entries = current.get("adopted_observations")
        if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
            for item in entries:
                if not isinstance(item, Mapping):
                    continue
                if (
                    str(item.get("report_id") or "") == self.report_id
                    and str(item.get("observation_id") or "") == self.observation_id
                    and str(item.get("text_sha256") or "") == self.text_sha256
                ):
                    return True
        return bool(
            str(current.get("report_id") or "") == self.report_id
            and str(current.get("observation_id") or "") == self.observation_id
            and str(current.get("adopted_text") or "") == self.text
        )


def find_persisted_ocr_observation(
    messages: Sequence[Mapping[str, Any]],
    *,
    report_id: str,
    observation_id: str,
) -> AdoptedOcrObservation:
    """Find one OCR observation in the latest AI report with ``report_id``.

    ``messages`` must already be scoped to the current session. A matching
    report with an unusable OCR state, duplicate observation ID, non-OCR kind,
    or empty/oversized text is rejected instead of falling back to older data.
    """

    wanted_report = _clean_identifier(report_id, "report_id")
    wanted_observation = _clean_identifier(observation_id, "observation_id")
    for message in reversed(list(messages or ())):
        if not isinstance(message, Mapping) or message.get("role") != "ai":
            continue
        meta = message.get("meta") if isinstance(message.get("meta"), Mapping) else {}
        report = meta.get("point_analysis") if isinstance(meta.get("point_analysis"), Mapping) else None
        if report is None or str(report.get("id") or "").strip() != wanted_report:
            continue

        ocr = report.get("ocr") if isinstance(report.get("ocr"), Mapping) else None
        state = str((ocr or {}).get("state") or "").strip().lower()
        if ocr is None or ocr.get("available") is not True or not state or state in _BLOCKED_OCR_STATES:
            raise ObservationAdoptionError("该报告没有可采用的真实 OCR 结果")

        observations = report.get("observations") if isinstance(report.get("observations"), list) else []
        matches = [
            item for item in observations
            if isinstance(item, Mapping)
            and str(item.get("id") or "").strip() == wanted_observation
        ]
        if not matches:
            raise ObservationAdoptionError("当前报告的观察 ID 不存在")
        if len(matches) != 1:
            raise ObservationAdoptionError("当前报告的观察 ID 不唯一")
        observation = matches[0]
        if str(observation.get("kind") or "") != "ocr_text":
            raise ObservationAdoptionError("只能采用 OCR 文字观察")

        raw_text = observation.get("text")
        if not isinstance(raw_text, str):
            raise ObservationAdoptionError("OCR 观察缺少文字")
        text = raw_text.strip()
        if not text:
            raise ObservationAdoptionError("OCR 观察文字为空")
        if len(text) > MAX_ADOPTED_OBSERVATION_TEXT:
            raise ObservationAdoptionError("OCR 观察文字过长")

        trust = observation.get("trust") if isinstance(observation.get("trust"), Mapping) else {}
        provider = str(observation.get("provider") or (ocr or {}).get("provider") or "unknown").strip()
        origin_source = str(observation.get("source") or "ocr").strip() or "ocr"
        ocr_provider = str((ocr or {}).get("provider") or "").strip()
        message_id = message.get("id") if isinstance(message.get("id"), int) else None
        return AdoptedOcrObservation(
            report_id=wanted_report,
            observation_id=wanted_observation,
            text=text,
            provider=provider[:120],
            origin_source=origin_source[:80],
            evidence=str(observation.get("evidence") or "")[:1000],
            bbox_norm=_bounded_bbox(observation.get("bbox_norm")),
            quad_norm=_bounded_bbox(observation.get("quad_norm")),
            report_bbox_norm=_bounded_bbox(report.get("bbox_norm")),
            bbox_px=_bounded_bbox_px(report.get("bbox_px")),
            ocr_provider=ocr_provider[:80],
            ocr_state=state[:80],
            confidence=_bounded_confidence(trust.get("confidence")),
            message_id=message_id,
        )

    raise ObservationAdoptionError("当前会话中没有对应的局部分析报告")


def is_user_confirmed_observation(extra: object) -> bool:
    """True when a clue carries provenance from an adopted region observation."""

    current = extra if isinstance(extra, Mapping) else {}
    return bool(
        current.get("user_confirmed") is True
        and (
            current.get("adoption_origin_source") == "region_observation"
            or current.get("observation_id")
            or current.get("adopted_observations")
        )
    )


__all__ = [
    "AdoptedOcrObservation",
    "MAX_ADOPTED_OBSERVATION_TEXT",
    "ObservationAdoptionError",
    "find_persisted_ocr_observation",
    "is_user_confirmed_observation",
]
