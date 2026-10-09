"""推理引擎：将线索映射到国家并评分排序。

核心逻辑（单一真源）：
- 每个国家的每个维度有一组可能的特征值；
- 每条线索 = (dimension, value, confidence)；
- 国家得分 = Σ 线索置信度 × 匹配权重，匹配权重：
    * 完全匹配该维度某特征值 -> 1.0
    * 该维度无该特征但 value 属于该维度通用值 -> 0.1（弱信号）
    * 不匹配 -> 0
- 语言维度额外加权（语言是强地理信号）；
- 排除型线索（用户反馈"不是 X"）会扣减对应国家分数。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import MIN_CLUE_CONFIDENCE, TOP_N_HYPOTHESES
from .clue import Clue
from . import knowledge_graph as kg


LANGUAGE_WEIGHT = 2.0  # 语言线索权重更高
CLIMATE_WEIGHT = 1.2
VEGETATION_WEIGHT = 1.0
BUILDING_WEIGHT = 1.0
VEHICLE_WEIGHT = 0.8

DIMENSION_WEIGHTS = {
    "language": LANGUAGE_WEIGHT,
    "climate": CLIMATE_WEIGHT,
    "vegetation": VEGETATION_WEIGHT,
    "building": BUILDING_WEIGHT,
    "vehicle": VEHICLE_WEIGHT,
    "culture": 1.5,
    "signage": 1.2,
}


@dataclass
class CountryScore:
    iso: str
    name: str
    name_zh: str
    score: float
    matched_clues: List[Dict[str, Any]] = field(default_factory=list)
    lat: float = 0.0
    lng: float = 0.0

    raw_cosine: float | None = None
    score_type: str = "rule_weight"
    # Optional, non-probabilistic component breakdown for hybrid rankings.
    fusion: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "iso": self.iso,
            "name": self.name,
            "name_zh": self.name_zh,
            "score": round(self.score, 4),
            "matched_clues": self.matched_clues,
            "lat": self.lat,
            "lng": self.lng,
            "raw_cosine": self.raw_cosine,
            "score_type": self.score_type,
        }
        if self.fusion:
            result["fusion"] = self.fusion
        return result


class ReasoningEngine:
    """线索 -> 候选国家评分。"""

    def __init__(self):
        self.countries = kg.load_countries()

    def _score_country(self, country: Dict[str, Any], clues: List[Clue],
                       exclusions: List[str]) -> Optional[CountryScore]:
        iso = country["iso"]
        if iso in exclusions:
            return None

        features = country.get("features", {})
        score = 0.0
        matched: List[Dict[str, Any]] = []

        for clue in clues:
            if clue.confidence < MIN_CLUE_CONFIDENCE:
                continue
            dim = clue.dimension
            value = clue.value
            dim_weight = DIMENSION_WEIGHTS.get(dim, 1.0)

            possible_values: List[str] = []
            # 语言维度用 languages_spoken
            if dim == "language":
                possible_values = country.get("languages_spoken", [])
            else:
                possible_values = features.get(dim, [])

            if value in possible_values:
                gain = clue.confidence * dim_weight
                score += gain
                matched.append({
                    "dimension": dim,
                    "value": value,
                    "gain": round(gain, 4),
                    "evidence": clue.evidence,
                })
            # 不匹配则不加分（保持 0），不惩罚以避免弱线索误杀

        if score <= 0:
            return None

        return CountryScore(
            iso=iso,
            name=country["name"],
            name_zh=country.get("name_zh", country["name"]),
            score=score,
            matched_clues=matched,
            lat=country.get("lat", 0.0),
            lng=country.get("lng", 0.0),
        )

    def infer(self, clues: List[Clue],
              exclusions: Optional[List[str]] = None,
              top_n: int = TOP_N_HYPOTHESES) -> List[CountryScore]:
        exclusions = exclusions or []
        scores: List[CountryScore] = []
        for country in self.countries:
            cs = self._score_country(country, clues, exclusions)
            if cs is not None:
                scores.append(cs)

        scores.sort(key=lambda x: x.score, reverse=True)
        # 归一化到 [0,1] 便于展示
        if scores:
            max_score = scores[0].score
            if max_score > 0:
                for s in scores:
                    s.score = s.score / max_score

        return scores[:top_n]

    def infer_as_dict(self, clues: List[Clue],
                      exclusions: Optional[List[str]] = None,
                      top_n: int = TOP_N_HYPOTHESES) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.infer(clues, exclusions, top_n)]
