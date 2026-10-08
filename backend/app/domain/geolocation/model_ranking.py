"""Full-image StreetCLIP prior with explicit, non-probabilistic ranking semantics."""
from __future__ import annotations

from .engine import ReasoningEngine, CountryScore
from .hypotheses import HypothesisManager, HypothesisAction
from .clue import DIM_LABELS
from config import TOP_N_HYPOTHESES


class ModelAwareEngine(ReasoningEngine):
    evidence: dict | None = None

    def infer(self, clues, exclusions=None, top_n=TOP_N_HYPOTHESES):
        if not self.evidence:
            return super().infer(clues, exclusions, top_n)
        prior = {r["iso"]: r["raw_cosine"] for r in self.evidence["rankings"]}
        # Color/brightness heuristics are not learned evidence: do not swamp the image prior.
        user_clues = [c for c in clues if c.source == "user"]
        scores = []
        for country in self.countries:
            iso = country["iso"]
            if iso in (exclusions or []) or iso not in prior:
                continue
            user_score = self._score_country(country, user_clues, [])
            boost = min(0.1, (user_score.score if user_score else 0) * 0.025)
            scores.append(CountryScore(
                iso=iso, name=country["name"], name_zh=country["name_zh"],
                score=prior[iso] + boost, raw_cosine=prior[iso],
                score_type="hybrid_rank" if user_clues else "cosine",
                matched_clues=user_score.matched_clues if user_score else [],
                lat=country.get("lat", 0), lng=country.get("lng", 0),
            ))
        return sorted(scores, key=lambda s: (-s.score, s.iso))[:top_n]


class ModelAwareHypotheses(HypothesisManager):
    def next_action(self, scores, clues):
        if not scores or scores[0].raw_cosine is None:
            return super().next_action(scores, clues)
        top = scores[0]
        candidates = [s.iso for s in scores[:2]]
        if len(scores) > 1:
            dimension = self._discriminating_dimension(scores[0], scores[1], clues)
            if dimension:
                return HypothesisAction("ask",
                    f"StreetCLIP 将「{top.name_zh}」排在候选首位（余弦相似度 {top.raw_cosine:.3f}，不是概率）。"
                    f"还不能确定拍摄地，请补充{DIM_LABELS.get(dimension, dimension)}线索。",
                    {"dimensions": [dimension], "candidates": candidates, "score_type": top.score_type})
        missing = self._missing_dimensions(clues)
        return HypothesisAction("request_clue",
            f"当前候选首位为「{top.name_zh}」，这仅是固定 20 国目录内的排序，不是已确认的拍摄地。请补充可辨识线索或手动确认。",
            {"dimensions": missing or ["language", "culture"], "candidates": candidates})
