"""Full-image StreetCLIP prior fused with unique crop cosines and trusted clues.

Trusted clues are user answers and explicitly reliable/catalog-validated
evidence (OCR, text_language, SemanticClueAnalyzer). Heuristic analyzers
(vegetation color, building color, ...) never count as model evidence, and
observation-only dimensions (signage/road/terrain/waterbody/infrastructure)
never score because they have no country-catalog metadata.

Boost scales are deterministic and bounded, never probabilistic:
- ordinary catalog clues keep the historical 0.02 scale, total cap 0.15;
- validated semantic evidence is weighted per dimension
  (place_name 0.12, driving_side 0.08, culture 0.06) with a 0.35 total cap.
"""
from __future__ import annotations

from .engine import ReasoningEngine, CountryScore, group_country_scores
from .country_policy import canonical_country_iso, is_excluded, normalize_iso
from .hypotheses import HypothesisManager, HypothesisAction
from .clue import DIM_LABELS, is_automated_heuristic_clue
from config import MIN_CLUE_CONFIDENCE, TOP_N_HYPOTHESES

# Modest, deterministic fusion weights: the immutable full-image cosine stays
# the dominant term and every secondary contribution is explicitly capped.
CROP_WEIGHT = 0.25
CLUE_BOOST_SCALE = 0.02
CLUE_BOOST_CAP = 0.15
# Validated semantic evidence gets a larger, still bounded, deterministic term.
SEMANTIC_BOOST_SCALES = {"place_name": 0.12, "driving_side": 0.08, "culture": 0.06}
SEMANTIC_BOOST_CAP = 0.35
_RELIABLE_SOURCES = {"ocr", "text_language"}


def _crop_region_identity(region: dict):
    """Identity used to keep one crop vote per ROI, not per repeated call."""
    region_id = region.get("region_id")
    if isinstance(region_id, str) and region_id:
        return ("region_id", region_id)
    bbox = region.get("bbox_norm")
    if isinstance(bbox, dict):
        try:
            return (
                "bbox_norm",
                round(float(bbox["x"]), 6),
                round(float(bbox["y"]), 6),
                round(float(bbox["w"]), 6),
                round(float(bbox["h"]), 6),
                str(region.get("purpose") or ""),
            )
        except (KeyError, TypeError, ValueError):
            return None
    try:
        return (
            "legacy_click",
            round(float(region["x"]), 6),
            round(float(region["y"]), 6),
            int(region["size"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _reference_by_iso(reference: dict) -> tuple[dict[str, list], dict[str, list]]:
    """Group a validated reference sidecar into ``(matches, regions)`` per ISO.

    Diagnostics are attached to the candidate row of their own country, so a
    country that is not among the returned candidates contributes nothing.
    """
    matches: dict[str, list] = {}
    regions: dict[str, list] = {}
    for key, bucket in (("matches", matches), ("regions", regions)):
        for row in reference.get(key) or []:
            if isinstance(row, dict) and isinstance(row.get("iso"), str):
                bucket.setdefault(row["iso"], []).append(row)
    return matches, regions


class ModelAwareEngine(ReasoningEngine):
    """Rank countries from the full prior, unique crop means and trusted clues."""

    evidence: dict | None = None

    @staticmethod
    def _is_trusted(clue) -> bool:
        """User clues plus explicitly reliable/catalog-validated evidence only.

        Color/vegetation/building heuristics are never trusted model evidence;
        observation-only semantic clues (``display_only``) stay display only.
        """
        extra = clue.extra if isinstance(clue.extra, dict) else {}
        if (
            is_automated_heuristic_clue(clue)
            or extra.get("display_only") is True
            or extra.get("affects_ranking") is False
        ):
            # 颜色/亮度启发式以及观察型证据只展示，绝不进入融合排序。
            return False
        if clue.source == "user":
            return True
        origin = extra.get("origin_source", clue.source)
        if extra.get("validated") is True:
            # Catalog-checked semantic evidence (place_name / culture / driving_side).
            return (origin in _RELIABLE_SOURCES or clue.source in _RELIABLE_SOURCES
                    or clue.source == "semantic")
        if extra.get("reliable") is not True:
            return False
        return origin in _RELIABLE_SOURCES or clue.source in _RELIABLE_SOURCES

    def _dedupe_key(self, clue) -> tuple[str, str]:
        """Canonical key while preserving distinct regional source scopes."""
        if clue.dimension == "place_name":
            extra = clue.extra if isinstance(clue.extra, dict) else {}
            claimed_source = normalize_iso(extra.get("source_iso"))
            if claimed_source:
                return (clue.dimension, claimed_source)
            resolved = self.resolve_country_key(clue.value)
            if resolved:
                return (clue.dimension, resolved)
        if clue.dimension == "driving_side" and isinstance(clue.value, str):
            return (clue.dimension, clue.value.strip().lower())
        return (clue.dimension, clue.value)

    def _trusted_clues(self, clues) -> list:
        """Deduplicate trusted clues per canonical (dimension, value).

        The strongest confidence wins; a user clue wins ties, then validated
        evidence wins over plain reliable evidence. Output order is
        deterministic regardless of clue insertion order.
        """
        best: dict[tuple[str, str], object] = {}
        for clue in clues:
            if clue.confidence < MIN_CLUE_CONFIDENCE or not self._is_trusted(clue):
                continue
            key = self._dedupe_key(clue)
            current = best.get(key)
            if current is None:
                best[key] = clue
                continue
            if clue.confidence > current.confidence:
                best[key] = clue
            elif clue.confidence == current.confidence:
                clue_extra = clue.extra if isinstance(clue.extra, dict) else {}
                current_extra = current.extra if isinstance(current.extra, dict) else {}
                if clue.source == "user" and current.source != "user":
                    best[key] = clue
                elif (clue_extra.get("validated") is True
                      and current_extra.get("validated") is not True):
                    best[key] = clue
        return [best[key] for key in sorted(best)]

    def _crop_means(self, prior: dict) -> dict[str, float]:
        """Per-region crop means, then one vote per region for each country."""
        region_values: dict[tuple, dict[str, list[float]]] = {}
        for crop in self.evidence.get("crops") or []:
            if not isinstance(crop, dict):
                continue
            region = crop.get("region") if isinstance(crop.get("region"), dict) else {}
            if region.get("ranking_applied") is False:
                continue
            purpose = region.get("purpose")
            if purpose is not None and purpose not in {"auto", "scene"}:
                continue
            key = _crop_region_identity(region)
            if key is None:
                continue
            for row in crop.get("rankings") or []:
                if not isinstance(row, dict):
                    continue
                iso = row.get("iso")
                value = row.get("raw_cosine")
                if iso not in prior or isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                region_values.setdefault(key, {}).setdefault(iso, []).append(float(value))

        country_region_means: dict[str, list[float]] = {}
        for values_by_iso in region_values.values():
            for iso, items in values_by_iso.items():
                if items:
                    country_region_means.setdefault(iso, []).append(sum(items) / len(items))
        return {
            iso: sum(items) / len(items)
            for iso, items in country_region_means.items()
            if items
        }

    def _crop_region_count(self) -> int:
        regions = set()
        for crop in self.evidence.get("crops") or []:
            if not isinstance(crop, dict):
                continue
            region = crop.get("region") if isinstance(crop.get("region"), dict) else {}
            if region.get("ranking_applied") is False:
                continue
            purpose = region.get("purpose")
            if purpose is not None and purpose not in {"auto", "scene"}:
                continue
            key = _crop_region_identity(region)
            if key is not None:
                regions.add(key)
        return len(regions)

    def infer(self, clues, exclusions=None, top_n=TOP_N_HYPOTHESES):
        if not self.evidence:
            return super().infer(clues, exclusions, top_n)
        prior = {row["iso"]: row["raw_cosine"] for row in self.evidence["rankings"]}
        crop_means = self._crop_means(prior)
        crop_region_count = self._crop_region_count()
        trusted = self._trusted_clues(clues)
        reference = self.evidence.get("reference")
        reference_matches, reference_regions = (
            _reference_by_iso(reference) if isinstance(reference, dict) else ({}, {})
        )
        scores = []
        for country in self.countries:
            iso = country["iso"]
            if is_excluded(exclusions, iso) or iso not in prior:
                continue
            # Retrieval-only diagnostics: never a term in the ranking below.
            diagnostics: dict = {}
            if isinstance(reference, dict):
                iso_regions = reference_regions.get(iso, [])
                iso_matches = reference_matches.get(iso, [])
                diagnostics = {
                    "reference_matches": iso_matches,
                    "reference_regions": iso_regions,
                    # Region prototypes are the aggregate view; fall back to the
                    # individual matches only when no region row covers this ISO.
                    "reference_cosine": max(
                        (row["raw_cosine"] for row in (iso_regions or iso_matches)), default=None
                    ),
                }
            full_cosine = float(prior[iso])
            crop_mean = crop_means.get(iso)
            if crop_mean is None:
                base_score = full_cosine
                crop_contribution = 0.0
            else:
                # Weighted mean keeps the composite inside the cosine range;
                # crops can only nudge the immutable full-image prior.
                crop_contribution = CROP_WEIGHT * crop_mean
                base_score = (full_cosine + crop_contribution) / (1.0 + CROP_WEIGHT)
            clue_score = 0.0
            matched_clues = []
            boost_terms = []
            if trusted:
                clue_result = self._score_country(country, trusted, [])
                if clue_result is not None:
                    clue_score = clue_result.score
                    matched_clues = clue_result.matched_clues
                    for item in matched_clues:
                        if not item.get("counted", True):
                            # 同一 OCR 行派生出的相关证据只计最强的一条。
                            continue
                        scale = SEMANTIC_BOOST_SCALES.get(item["dimension"], CLUE_BOOST_SCALE)
                        boost_terms.append({
                            "dimension": item["dimension"],
                            "gain": item["gain"],
                            "scale": scale,
                            "boost": round(float(item["gain"]) * scale, 6),
                        })
            boost_cap = (
                SEMANTIC_BOOST_CAP
                if any(term["dimension"] in SEMANTIC_BOOST_SCALES for term in boost_terms)
                else CLUE_BOOST_CAP
            )
            clue_boost = min(boost_cap, sum(term["boost"] for term in boost_terms))
            # Honest semantics: only countries with actual secondary evidence
            # are labelled hybrid; a pure full-image prior stays "cosine".
            hybrid = crop_mean is not None or clue_boost > 0
            fusion = {
                "full_weight": 1.0,
                "full_cosine": round(full_cosine, 6),
                "crop_weight": CROP_WEIGHT,
                "crop_count": crop_region_count,
                "crop_mean": round(crop_mean, 6) if crop_mean is not None else None,
                "crop_contribution": round(crop_contribution, 6),
                "clue_score": round(clue_score, 6),
                "clue_boost": round(clue_boost, 6),
            } if hybrid else {}
            if boost_terms:
                fusion["clue_boost_terms"] = boost_terms
                fusion["clue_boost_cap"] = boost_cap
            scores.append(CountryScore(
                iso=iso, name=country["name"], name_zh=country["name_zh"],
                score=base_score + clue_boost,
                raw_cosine=full_cosine,
                score_type="hybrid_rank" if hybrid else "cosine",
                matched_clues=matched_clues,
                lat=country.get("lat", 0), lng=country.get("lng", 0),
                fusion=fusion,
                **diagnostics,
            ))
        # Group raw source scopes before the top-N slice. A persisted CN
        # exclusion removes all children; a raw TW/HK/MO exclusion removes
        # only that source region before the max aggregation.
        grouped = group_country_scores(scores, self._country_by_iso)
        return sorted(grouped, key=lambda item: (-item.score, item.iso))[:top_n]


class ModelAwareHypotheses(HypothesisManager):
    """Fused-score question policy: distinguishing, non-repeating, option-backed.

    The StreetCLIP prior is ranking evidence, never confirmation evidence, so
    this manager keeps asking for user-verifiable clues. Heuristic image clues
    never count as user knowledge, so a color/climate guess cannot suppress a
    better question to the user.
    """

    def next_action(self, scores, clues, asked=None):
        asked_dims = {d for d in (asked or ()) if isinstance(d, str)}
        if not scores or scores[0].raw_cosine is None:
            return super().next_action(scores, clues, asked=asked_dims)
        top = scores[0]
        candidates = [item.iso for item in scores[:2]]
        blocked = self._user_known_dimensions(clues) | asked_dims
        dimension = None
        if len(scores) > 1:
            dimension = self._discriminating_dimension(scores[0], scores[1], clues, blocked)
        if dimension:
            second = scores[1]
            options = self._options_for(dimension, scores[:2], clues)
            reason = (f"「{top.name_zh}」与「{second.name_zh}」在这个维度上的目录特征不同，"
                      f"可以用它区分候选。")
            return HypothesisAction(
                "ask",
                f"StreetCLIP 把「{top.name_zh}」排在候选首位（全图余弦 {top.raw_cosine:.3f}，"
                f"融合排序不是概率）。还不能确定拍摄地，请补充{self.dimension_label(dimension)}线索。",
                {"dimensions": [dimension], "candidates": candidates, "options": options,
                 "reason": reason, "score_type": top.score_type},
            )
        # No distinguishing dimension left: ask the highest-priority dimension
        # that still lacks trustworthy user evidence, without repeating what
        # the dialogue already asked.
        remaining = [d for d in self.QUESTION_ORDER if d not in blocked]
        lead = remaining[0] if remaining else None
        options = self._options_for(lead, scores[:2], clues) if lead else []
        if lead:
            reason = f"当前缺少{self.dimension_label(lead)}维度的用户证据。"
            tail = f"请补充{self.dimension_label(lead)}等线索，或手动确认。"
        else:
            reason = "所有优先维度都已询问过；请直接描述画面或点击可疑区域。"
            tail = "请直接描述画面中的可辨识线索，或手动确认。"
        return HypothesisAction(
            "request_clue",
            f"当前候选首位为「{top.name_zh}」，这仅是当前候选国家目录"
            f"（{len({canonical_country_iso(c['iso']) for c in self.engine.countries})} 国）内的排序，不是已确认的拍摄地。{tail}",
            {"dimensions": [lead] if lead else [], "candidates": candidates,
             "options": options, "reason": reason, "score_type": top.score_type},
        )

__all__ = [
    "CLUE_BOOST_CAP",
    "CLUE_BOOST_SCALE",
    "CROP_WEIGHT",
    "SEMANTIC_BOOST_CAP",
    "SEMANTIC_BOOST_SCALES",
    "ModelAwareEngine",
    "ModelAwareHypotheses",
]
