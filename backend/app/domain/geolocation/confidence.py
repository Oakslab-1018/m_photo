"""Deterministic confidence routing for country-level results.

The route is a product policy, not a probability. It uses the minimum of the
canonical StreetCLIP model-only margin and the fused top1-top2 margin so OCR
cannot create a high-confidence answer from an uncertain prior. Thresholds are
provisional until the full paired dev/holdout analysis is complete.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from .country_policy import canonical_country_iso

POLICY_VERSION = "confidence-tier-v1-provisional"
HIGH_CONFIDENCE_MARGIN = 0.02
MEDIUM_CONFIDENCE_MARGIN = 0.005
_MARGIN_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ConfidenceRoute:
    tier: str
    action: str
    model_only_margin: float | None
    fused_margin: float | None
    effective_margin: float | None
    prior_top_iso: str | None
    top1_iso: str | None
    top3: tuple[dict[str, Any], ...]
    leader_changed: bool
    reason: str
    policy_version: str = POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "action": self.action,
            "model_only_margin": self.model_only_margin,
            "fused_margin": self.fused_margin,
            "effective_margin": self.effective_margin,
            "prior_top_iso": self.prior_top_iso,
            "top1_iso": self.top1_iso,
            "top3": [dict(item) for item in self.top3],
            "leader_changed": self.leader_changed,
            "reason": self.reason,
            "policy_version": self.policy_version,
        }


def _margin(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _score_row(score: Any) -> dict[str, Any] | None:
    if isinstance(score, dict):
        row = dict(score)
    else:
        row = score.to_dict() if callable(getattr(score, "to_dict", None)) else {}
    iso = canonical_country_iso(row.get("iso"))
    if not iso:
        return None
    payload = {
        key: row.get(key)
        for key in (
            "name",
            "name_zh",
            "score",
            "raw_cosine",
            "score_type",
            "matched_clues",
            "fusion",
            "parent_iso",
            "source_regions",
            "is_grouped",
        )
        if row.get(key) is not None
    }
    payload["iso"] = iso
    return payload


def _canonical_fused_rows(scores: Sequence[Any]) -> tuple[list[dict[str, Any]], float | None]:
    """Return canonical fused rows and their top1-top2 margin."""

    rows: list[dict[str, Any]] = []
    values: list[float] = []
    for score in list(scores):
        row = _score_row(score)
        if row is None:
            return [], None
        value = _margin(row.get("score"))
        if value is None:
            return [], None
        values.append(value)
        rows.append({**row, "score": value})
    if len(values) < 2 or any(values[index] < values[index + 1] for index in range(len(values) - 1)):
        return [], None

    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        iso = str(row["iso"])
        current = best.get(iso)
        if current is None or float(row["score"]) > float(current["score"]):
            best[iso] = row
    canonical = sorted(best.values(), key=lambda item: (-float(item["score"]), str(item["iso"])))
    if len(canonical) < 2:
        return canonical, None
    return canonical, float(canonical[0]["score"]) - float(canonical[1]["score"])


def _canonical_model_margin(model_evidence: dict[str, Any] | None) -> tuple[str, float] | None:
    """Validate evidence and return canonical ``(leader, top1-top2 margin)``."""

    if not isinstance(model_evidence, dict):
        return None
    rankings = model_evidence.get("rankings")
    if not isinstance(rankings, list) or len(rankings) < 2:
        return None

    raw_values: list[float] = []
    seen_raw: set[str] = set()
    canonical_best: dict[str, float] = {}
    for row in rankings:
        if not isinstance(row, dict):
            return None
        raw_iso = str(row.get("iso") or "").strip().upper()
        iso = canonical_country_iso(raw_iso)
        if not iso or raw_iso in seen_raw:
            return None
        value = _margin(row.get("raw_cosine"))
        if value is None or not -1.0 <= value <= 1.0:
            return None
        seen_raw.add(raw_iso)
        raw_values.append(value)
        canonical_best[iso] = max(canonical_best.get(iso, value), value)

    if any(raw_values[index] < raw_values[index + 1] for index in range(len(raw_values) - 1)):
        return None
    raw_margin = raw_values[0] - raw_values[1]
    reported = _margin(model_evidence.get("margin"))
    if reported is None or reported < 0 or abs(reported - raw_margin) > _MARGIN_TOLERANCE:
        return None

    canonical = sorted(canonical_best.items(), key=lambda item: (-item[1], item[0]))
    if len(canonical) < 2:
        return None
    return canonical[0][0], canonical[0][1] - canonical[1][1]


def route_confidence(
    scores: Sequence[Any],
    model_evidence: dict[str, Any] | None,
) -> ConfidenceRoute:
    """Return direct/confirm/ask routing without consulting ground truth."""

    fused_rows, fused_margin = _canonical_fused_rows(scores)
    top3 = tuple(fused_rows[:3])
    top1_iso = (str(top3[0].get("iso") or "") or None) if top3 else None
    model_route = _canonical_model_margin(model_evidence)
    prior_top_iso, model_margin = model_route if model_route is not None else (None, None)
    effective_margin = (
        min(model_margin, fused_margin)
        if model_margin is not None and fused_margin is not None
        else None
    )
    leader_changed = bool(prior_top_iso and top1_iso and prior_top_iso != top1_iso)

    if effective_margin is None:
        tier, action = "low", "ask_clue"
        reason = "模型证据或融合排序缺少一致且有效的 margin/leader，不能自动给出国家结论。"
    elif effective_margin >= HIGH_CONFIDENCE_MARGIN and not leader_changed:
        tier, action = "high", "direct_top1"
        reason = "model-only 与 fused 的保守 margin 均达到高置信区间。"
    elif effective_margin >= MEDIUM_CONFIDENCE_MARGIN or leader_changed:
        tier, action = "medium", "confirm_candidates"
        reason = (
            "OCR 改变了候选首位，需用户确认。"
            if leader_changed
            else "候选接近，应展示 top3 并让用户确认。"
        )
    else:
        tier, action = "low", "ask_clue"
        reason = "model-only 与 fused 的保守 margin 低于中置信区间。"

    return ConfidenceRoute(
        tier=tier,
        action=action,
        model_only_margin=round(model_margin, 6) if model_margin is not None else None,
        fused_margin=round(fused_margin, 6) if fused_margin is not None else None,
        effective_margin=round(effective_margin, 6) if effective_margin is not None else None,
        prior_top_iso=prior_top_iso,
        top1_iso=top1_iso,
        top3=top3,
        leader_changed=leader_changed,
        reason=reason,
    )


__all__ = [
    "ConfidenceRoute",
    "HIGH_CONFIDENCE_MARGIN",
    "MEDIUM_CONFIDENCE_MARGIN",
    "POLICY_VERSION",
    "route_confidence",
]
