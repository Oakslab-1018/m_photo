"""Full-image StreetCLIP prior fused with unique crop cosines and trusted clues."""
from __future__ import annotations

from .engine import ReasoningEngine, CountryScore
from .hypotheses import HypothesisManager, HypothesisAction
from .clue import DIM_LABELS
from config import MIN_CLUE_CONFIDENCE, TOP_N_HYPOTHESES

# Modest, deterministic fusion weights: the immutable full-image cosine stays
# the dominant term and every secondary contribution is explicitly capped.
CROP_WEIGHT = 0.25
CLUE_BOOST_SCALE = 0.02
CLUE_BOOST_CAP = 0.15
_RELIABLE_SOURCES = {"ocr", "text_language"}


class ModelAwareEngine(ReasoningEngine):
    """Rank countries from the full prior, unique crop means and trusted clues."""

    evidence: dict | None = None

    @staticmethod
    def _is_trusted(clue) -> bool:
        """User clues and explicitly reliable OCR/text clues only.

        Color/vegetation/building heuristics are never trusted model evidence.
        """
        extra = clue.extra if isinstance(clue.extra, dict) else {}
        if clue.source == "user":
            return True
        if extra.get("reliable") is not True:
            return False
        origin = extra.get("origin_source", clue.source)
        return origin in _RELIABLE_SOURCES or clue.source in _RELIABLE_SOURCES

    def _trusted_clues(self, clues) -> list:
        """Deduplicate trusted clues per (dimension, value).

        The strongest confidence wins; a user clue wins ties so a weak OCR
        observation can never mask a stronger manual one. Output order is
        deterministic regardless of clue insertion order.
        """
        best: dict[tuple[str, str], object] = {}
        for clue in clues:
            if clue.confidence < MIN_CLUE_CONFIDENCE or not self._is_trusted(clue):
                continue
            key = (clue.dimension, clue.value)
            current = best.get(key)
            if current is None:
                best[key] = clue
                continue
            if clue.confidence > current.confidence:
                best[key] = clue
            elif clue.confidence == current.confidence and clue.source == "user" and current.source != "user":
                best[key] = clue
        return [best[key] for key in sorted(best)]

    def _crop_means(self, prior: dict) -> dict[str, float]:
        """Mean unique crop cosine per country from validated crop evidence."""
        values: dict[str, list[float]] = {}
        for crop in self.evidence.get("crops") or []:
            if not isinstance(crop, dict):
                continue
            for row in crop.get("rankings") or []:
                if not isinstance(row, dict):
                    continue
                iso = row.get("iso")
                value = row.get("raw_cosine")
                if iso not in prior or isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                values.setdefault(iso, []).append(float(value))
        return {iso: sum(items) / len(items) for iso, items in values.items() if items}

    def infer(self, clues, exclusions=None, top_n=TOP_N_HYPOTHESES):
        if not self.evidence:
            return super().infer(clues, exclusions, top_n)
        prior = {row["iso"]: row["raw_cosine"] for row in self.evidence["rankings"]}
        crop_means = self._crop_means(prior)
        trusted = self._trusted_clues(clues)
        crops = self.evidence.get("crops") or []
        scores = []
        for country in self.countries:
            iso = country["iso"]
            if iso in (exclusions or []) or iso not in prior:
                continue
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
            if trusted:
                clue_result = self._score_country(country, trusted, [])
                if clue_result is not None:
                    clue_score = clue_result.score
                    matched_clues = clue_result.matched_clues
            clue_boost = min(CLUE_BOOST_CAP, clue_score * CLUE_BOOST_SCALE)
            # Honest semantics: only countries with actual secondary evidence
            # are labelled hybrid; a pure full-image prior stays "cosine".
            hybrid = crop_mean is not None or clue_boost > 0
            fusion = {
                "full_weight": 1.0,
                "full_cosine": round(full_cosine, 6),
                "crop_weight": CROP_WEIGHT,
                "crop_count": len(crops),
                "crop_mean": round(crop_mean, 6) if crop_mean is not None else None,
                "crop_contribution": round(crop_contribution, 6),
                "clue_score": round(clue_score, 6),
                "clue_boost": round(clue_boost, 6),
            } if hybrid else {}
            scores.append(CountryScore(
                iso=iso, name=country["name"], name_zh=country["name_zh"],
                score=base_score + clue_boost,
                raw_cosine=full_cosine,
                score_type="hybrid_rank" if hybrid else "cosine",
                matched_clues=matched_clues,
                lat=country.get("lat", 0), lng=country.get("lng", 0),
                fusion=fusion,
            ))
        return sorted(scores, key=lambda item: (-item.score, item.iso))[:top_n]


class ModelAwareHypotheses(HypothesisManager):
    def next_action(self, scores, clues):
        if not scores or scores[0].raw_cosine is None:
            return super().next_action(scores, clues)
        top = scores[0]
        candidates = [item.iso for item in scores[:2]]
        if len(scores) > 1:
            dimension = self._discriminating_dimension(scores[0], scores[1], clues)
            if dimension:
                return HypothesisAction("ask",
                    f"StreetCLIP 将「{top.name_zh}」排在候选首位（全图余弦 {top.raw_cosine:.3f}，融合排序不是概率）。"
                    f"还不能确定拍摄地，请补充{DIM_LABELS.get(dimension, dimension)}线索。",
                    {"dimensions": [dimension], "candidates": candidates, "score_type": top.score_type})
        missing = self._missing_dimensions(clues)
        return HypothesisAction("request_clue",
            f"当前候选首位为「{top.name_zh}」，这仅是当前候选国家目录（{len(self.engine.countries)} 国）内的排序，不是已确认的拍摄地。请补充可辨识线索或手动确认。",
            {"dimensions": missing or ["language", "culture"], "candidates": candidates,
             "score_type": top.score_type})
