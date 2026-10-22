"""Bounded model-similarity reranking for city scene observations.

This module deliberately does not turn user feedback or StreetCLIP scene
labels into claimed city attributes. A scene clue is only a query used by the
gateway to compare the photo against a city-conditioned visual neutral. The
resulting centered residual is a bounded model-similarity signal.

The shared ``services.city_scene_contract`` module owns the vocabulary and
normalization. This module owns session-evidence selection, residual math and
candidate provenance. It is deterministic, read-only and network-free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from services.city_scene_contract import (
    CITY_SCENE_VERSION,
    MAX_CITY_SCENE_CLUES,
    normalize_city_scene_clue,
    city_scene_value,
)

from ..domain.geolocation.clue import DIM_LABELS
from .region_observation import is_user_confirmed_observation


SCENE_SIGNAL_KIND = "model_similarity"
SCENE_MATCH_KIND = "model_match"
SCENE_SCORE_TYPE = "city_scene_clue_fusion"
SCENE_DELTA_SCALE = 0.6
SCENE_DIMENSION_CAP = 0.04
SCENE_TOTAL_CAP = 0.08
SCENE_MIN_SPAN = 0.005
SCENE_MAX_NOTES = 8

_PLACE_DIMENSIONS = frozenset({"place_name", "signage"})
_NON_DISCRIMINATING_DIMENSIONS = frozenset({"language", "driving_side"})
_EXCLUSIVE_VALUE_GROUPS = {
    "building": (frozenset({"high_rise", "mid_rise", "low_rise"}),),
    "terrain": (frozenset({"flat_ground", "mountainous"}),),
}
_DIMENSION_ORDER = (
    "building",
    "terrain",
    "waterbody",
    "vegetation",
    "road",
    "vehicle",
    "climate",
    "infrastructure",
)


@dataclass(frozen=True)
class SceneObservation:
    """One user-reported or user-confirmed scene observation."""

    clue_id: str
    dimension: str
    value: str
    confidence: float
    evidence: str


@dataclass(frozen=True)
class SceneSelection:
    observations: tuple[SceneObservation, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class SceneRerankResult:
    city_scores: list[dict[str, Any]]
    clue_notes: tuple[str, ...]
    dimensions: tuple[str, ...]
    attempted: bool = False


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _record_extra(record: Mapping[str, Any]) -> dict[str, Any]:
    extra = record.get("_extra")
    if not isinstance(extra, Mapping):
        extra = record.get("extra")
    return dict(extra) if isinstance(extra, Mapping) else {}


def _dedupe_notes(notes: Sequence[str]) -> tuple[str, ...]:
    unique: list[str] = []
    for raw in notes:
        note = str(raw or "").strip()
        if note and note not in unique:
            unique.append(note)
        if len(unique) >= SCENE_MAX_NOTES:
            break
    return tuple(unique)


def _dimension_label(dimension: str) -> str:
    return DIM_LABELS.get(dimension, dimension or "场景")


def _normalize(dimension: str, value: str) -> tuple[str, str] | None:
    try:
        normalized = normalize_city_scene_clue(dimension, value)
    except Exception:
        return None
    if not isinstance(normalized, tuple) or len(normalized) != 2:
        return None
    dim, canonical = normalized
    if not isinstance(dim, str) or not isinstance(canonical, str):
        return None
    dim = dim.strip()
    canonical = canonical.strip()
    return (dim, canonical) if dim and canonical else None

def select_scene_observations(
    records: Sequence[Mapping[str, Any]],
) -> SceneSelection:
    """Select bounded, non-conflicting scene observations from review state.

    A pending ``guided_user`` observation is an explicitly reported observation
    and may seed a model comparison. A machine observation is eligible only
    after the user confirmed its review state or adopted it explicitly. Rejected, uncertain
    and ranking-suspended records never enter the model comparison.
    """

    by_dimension: dict[str, list[SceneObservation]] = {}
    notes: list[str] = []

    for record in records or ():
        if not isinstance(record, Mapping):
            continue
        extra = _record_extra(record)
        status = str(record.get("status") or "pending").strip().lower()
        source = str(record.get("source") or "").strip().lower()
        origin = str(record.get("_origin") or "").strip().lower()
        adopted = is_user_confirmed_observation(extra)
        user_reported = source == "user" and origin == "guided_user"
        reviewed_machine = origin == "session" and status == "confirmed"
        if not user_reported and not adopted and not reviewed_machine:
            # Raw OCR, LLM guesses and unreviewed analyzer output are not scene
            # observations even when their dimension happens to match.
            continue
        if extra.get("ranking_suspended") is True:
            notes.append("已撤回的结构化观察暂不参与城市画面匹配")
            continue
        if status in {"rejected", "uncertain"}:
            notes.append("已否定或无法判断的结构化观察不参与城市画面匹配")
            continue
        if status not in {"pending", "confirmed"}:
            continue
        if adopted and status != "confirmed":
            continue

        dimension = str(record.get("dimension") or "").strip()
        if not dimension or dimension in _PLACE_DIMENSIONS:
            continue
        if dimension in _NON_DISCRIMINATING_DIMENSIONS:
            notes.append(f"{_dimension_label(dimension)}暂未建立独立的城市对应，未计入城市重排")
            continue

        value = str(record.get("value") or "").strip()
        normalized = _normalize(dimension, value)
        if normalized is None:
            notes.append(f"{_dimension_label(dimension)}观察当前没有受控画面词条，未计分")
            continue
        normalized_dimension, canonical_value = normalized
        confidence = _finite_float(record.get("confidence"))
        if confidence is None:
            continue
        confidence = _clamp(confidence, 0.0, 1.0)
        if confidence <= 0.0:
            continue
        by_dimension.setdefault(normalized_dimension, []).append(
            SceneObservation(
                clue_id=str(record.get("id") or ""),
                dimension=normalized_dimension,
                value=canonical_value,
                confidence=confidence,
                evidence=str(
                    record.get("description")
                    or record.get("evidence")
                    or ""
                )[:500],
            )
        )

    selected: list[SceneObservation] = []
    for dimension in _DIMENSION_ORDER:
        observations = by_dimension.get(dimension) or []
        if not observations:
            continue
        values = {item.value for item in observations}
        if len(values) > 1:
            exclusive = any(
                len(values & group) > 1
                for group in _EXCLUSIVE_VALUE_GROUPS.get(dimension, ())
            )
            if exclusive:
                notes.append(
                    f"{_dimension_label(dimension)}观察包含互斥取值，整维暂不计分"
                )
                continue
            notes.append("同维多项观察，仅取一项参与，避免重复计分")
        selected.append(
            sorted(
                observations,
                key=lambda item: (-item.confidence, item.value, item.clue_id),
            )[0]
        )

    if len(selected) > MAX_CITY_SCENE_CLUES:
        notes.append(
            f"城市画面匹配最多使用{MAX_CITY_SCENE_CLUES}个维度，其余维度未计分"
        )
        selected = selected[:MAX_CITY_SCENE_CLUES]
    return SceneSelection(tuple(selected), _dedupe_notes(notes))


def scene_clue_pairs(
    observations: Sequence[SceneObservation],
) -> tuple[tuple[str, str], ...]:
    return tuple((item.dimension, item.value) for item in observations[:MAX_CITY_SCENE_CLUES])


def _payload_scores(payload: object) -> dict[tuple[str, str], dict[str, tuple[float, float]]]:
    if not isinstance(payload, Mapping):
        return {}
    result: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    for item in payload.get("clues") or ():
        if not isinstance(item, Mapping):
            continue
        dimension = str(item.get("dimension") or "").strip()
        value = str(item.get("value") or "").strip()
        if not dimension or not value:
            continue
        scores: dict[str, tuple[float, float]] = {}
        for score in item.get("scores") or ():
            if not isinstance(score, Mapping):
                continue
            city_id = str(score.get("city_id") or "").strip()
            raw = _finite_float(score.get("raw_cosine"))
            neutral = _finite_float(score.get("neutral_cosine"))
            if city_id and raw is not None and neutral is not None:
                scores[city_id] = (raw, neutral)
        if scores:
            result[(dimension, value)] = scores
    return result

def _city_id(row: Mapping[str, Any]) -> str:
    return str(row.get("city_id") or "").strip()


def _score(row: Mapping[str, Any]) -> float:
    value = _finite_float(row.get("score"))
    return value if value is not None else 0.0


def _numeric_city_id(row: Mapping[str, Any]) -> int:
    try:
        return int(_city_id(row))
    except (TypeError, ValueError):
        return 1 << 62


def _base_score(row: Mapping[str, Any]) -> float:
    fusion = row.get("fusion")
    if isinstance(fusion, Mapping):
        value = _finite_float(fusion.get("model_base_score"))
        if value is None:
            value = _finite_float(fusion.get("base_score"))
        if value is not None:
            return value
    return _score(row)


def _final_score(row: Mapping[str, Any]) -> float:
    fusion = row.get("fusion")
    if isinstance(fusion, Mapping):
        value = _finite_float(fusion.get("final_score"))
        if value is not None:
            return value
    return _score(row)


def _order(rows: Sequence[Mapping[str, Any]], *, use_scene: bool) -> list[Mapping[str, Any]]:
    def key(row: Mapping[str, Any]) -> tuple[float, int]:
        score = _final_score(row) if use_scene else _base_score(row)
        return (-score, _numeric_city_id(row))

    return sorted(rows, key=key)


def apply_scene_residual(
    city_scores: Sequence[Mapping[str, Any]],
    payload: object,
    observations: Sequence[SceneObservation],
    *,
    notes: Sequence[str] = (),
    version: str = CITY_SCENE_VERSION,
) -> SceneRerankResult:
    """Apply a bounded, centered scene-conditioned model similarity residual."""

    rows = [dict(row) for row in city_scores]
    all_notes = list(notes)
    if not rows or not observations:
        return SceneRerankResult(rows, _dedupe_notes(all_notes), (), False)

    payload_scores = _payload_scores(payload)
    if not payload_scores:
        all_notes.append("城市画面模型结果暂不可用，未改变城市分数")
        return SceneRerankResult(rows, _dedupe_notes(all_notes), (), False)

    row_ids = {_city_id(row) for row in rows if _city_id(row)}
    dimension_gains: list[tuple[SceneObservation, dict[str, float]]] = []
    dimensions: list[str] = []
    for observation in observations[:MAX_CITY_SCENE_CLUES]:
        scores = payload_scores.get((observation.dimension, observation.value)) or {}
        if not scores:
            all_notes.append(
                f"{_dimension_label(observation.dimension)}画面匹配结果缺失，未计分"
            )
            continue
        deltas: dict[str, float] = {}
        for city_id in row_ids:
            pair = scores.get(city_id)
            if pair is not None:
                deltas[city_id] = pair[0] - pair[1]
        dimensions.append(observation.dimension)
        gains = {city_id: 0.0 for city_id in row_ids}
        if len(deltas) < 2 or not deltas:
            all_notes.append(
                f"{_dimension_label(observation.dimension)}画面匹配候选不足，整维不计分"
            )
            dimension_gains.append((observation, gains))
            continue
        span = max(deltas.values()) - min(deltas.values())
        if span < SCENE_MIN_SPAN:
            all_notes.append(
                f"{_dimension_label(observation.dimension)}画面匹配无区分度，整维不计分"
            )
            dimension_gains.append((observation, gains))
            continue
        mean_delta = sum(deltas.values()) / len(deltas)
        for city_id, delta in deltas.items():
            centered = delta - mean_delta
            gain = (
                _clamp(SCENE_DELTA_SCALE * centered, -SCENE_DIMENSION_CAP, SCENE_DIMENSION_CAP)
                * observation.confidence
            )
            gains[city_id] = round(gain, 6)
        dimension_gains.append((observation, gains))

    if not dimension_gains:
        return SceneRerankResult(rows, _dedupe_notes(all_notes), (), False)

    before_order = _order(rows, use_scene=False)
    before_rank = {_city_id(row): index for index, row in enumerate(before_order, 1)}
    final_by_id: dict[str, float] = {}
    applied_delta_by_id: dict[str, float] = {}
    requested_delta_by_id: dict[str, float] = {}
    any_delta = False
    dimension_count = max(1, len(dimension_gains))
    for row in rows:
        city_id = _city_id(row)
        requested = sum(gains.get(city_id, 0.0) for _, gains in dimension_gains)
        requested = _clamp(requested / dimension_count, -SCENE_TOTAL_CAP, SCENE_TOTAL_CAP)
        current = _score(row)
        final = _clamp(current + requested, 0.0, 1.0)
        applied = final - current
        final_by_id[city_id] = round(final, 6)
        requested_delta_by_id[city_id] = round(requested, 6)
        applied_delta_by_id[city_id] = round(applied, 6)
        any_delta = any_delta or abs(applied) > 1e-12

    for row in rows:
        city_id = _city_id(row)
        delta = applied_delta_by_id.get(city_id, 0.0)
        final = final_by_id.get(city_id, _score(row))
        fusion = dict(row.get("fusion")) if isinstance(row.get("fusion"), Mapping) else {}
        base_score = _finite_float(fusion.get("base_score"))
        if base_score is None:
            base_score = _score(row)
        place_boost = _finite_float(fusion.get("clue_boost"))
        if place_boost is None:
            place_boost = 0.0
        fusion.update({
            "base_score": round(base_score, 6),
            "model_base_score": round(base_score, 6),
            "place_boost": round(place_boost, 6),
            "clue_boost": round(place_boost, 6),
            "scene_delta": delta,
            "final_score": final,
            "scene_version": str(version),
            "signal_kind": SCENE_SIGNAL_KIND,
            "match_kind": SCENE_MATCH_KIND,
            "scene_dimensions": list(dict.fromkeys(dimensions)),
            "scene_delta_cap": SCENE_TOTAL_CAP,
            "rank_before": before_rank.get(city_id),
        })
        terms = [
            dict(item)
            for item in (row.get("matched_clues") or ())
            if isinstance(item, Mapping)
        ]
        requested = requested_delta_by_id.get(city_id, 0.0)
        applied = applied_delta_by_id.get(city_id, 0.0)
        apply_fraction = (applied / requested) if abs(requested) > 1e-12 else 0.0
        for observation, gains in dimension_gains:
            raw_gain = gains.get(city_id, 0.0)
            effective_gain = raw_gain * apply_fraction / dimension_count
            if abs(effective_gain) <= 1e-12:
                continue
            terms.append({
                "dimension": observation.dimension,
                "value": observation.value,
                "value_label": (
                    city_scene_value(observation.dimension, observation.value).label
                    if city_scene_value(observation.dimension, observation.value) is not None
                    else observation.value
                ),
                "clue_id": observation.clue_id,
                "user_evidence": observation.evidence,
                "gain": round(effective_gain, 6),
                "raw_gain": round(raw_gain, 6),
                "weight": round(apply_fraction / dimension_count, 6),
                "evidence": (
                    "StreetCLIP 画面条件模型相似度；"
                    "这是模型相似度残差，不是已核验的城市属性"
                ),
                "signal_kind": SCENE_SIGNAL_KIND,
                "match_kind": SCENE_MATCH_KIND,
            })
        row["score"] = final
        row["fusion"] = fusion
        row["rank_before"] = before_rank.get(city_id)
        if terms:
            row["matched_clues"] = terms
        if any_delta:
            row["score_type"] = SCENE_SCORE_TYPE

    after_order = _order(rows, use_scene=True)
    after_rank = {_city_id(row): index for index, row in enumerate(after_order, 1)}
    for row in rows:
        city_id = _city_id(row)
        row["rank_after"] = after_rank.get(city_id)
        if isinstance(row.get("fusion"), dict):
            row["fusion"]["rank_after"] = after_rank.get(city_id)
    return SceneRerankResult(
        [dict(row) for row in after_order],
        _dedupe_notes(all_notes),
        tuple(dict.fromkeys(dimensions)),
        True,
    )


__all__ = [
    "SCENE_DIMENSION_CAP",
    "SCENE_MATCH_KIND",
    "SCENE_SCORE_TYPE",
    "SCENE_SIGNAL_KIND",
    "SCENE_TOTAL_CAP",
    "SceneObservation",
    "SceneRerankResult",
    "SceneSelection",
    "apply_scene_residual",
    "scene_clue_pairs",
    "select_scene_observations",
]
