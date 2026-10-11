"""假设生成与验证：驱动人机协同的多轮交互。

每次推理后，引擎根据候选国家分布决定下一步动作：
- 若 Top1 置信度 >> Top2，且超过阈值 -> 提议确认（confirm）
- 若多个国家候选接近 -> 提出区分性问题（ask）
- 若线索不足 -> 主动询问关键维度（request_clue）

本轮增强（保持既有 confirm/ask 语义不变）：

- 只有 *用户证据*（source=user 或显式标记 reliable 的 OCR/文字体系证据）才算
  "已知"，启发式图像线索不会阻止 AI 向用户追问更可靠的线索；
- 已经问过的维度（由会话历史 meta.action.payload.dimensions 汇总）不会重复问；
- 提问附带结构化选项 payload.options = [{dimension, value, label}]，
  由前端本地化维度并渲染为答案 chip；每条问题最多 MAX_OPTIONS 个选项；
- 每个 ask/request_clue 附 payload.reason 说明为什么问这个维度。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

from config import CONFIRMATION_THRESHOLD
from .clue import Clue, DIM_LABELS, VALUE_LABELS
from .engine import CountryScore, ReasoningEngine


# 新增共享维度在 clue.py 落地前的本地回退标签（落地后自动优先使用 clue.py）。
LOCAL_DIM_LABELS = {
    "place_name": "地名文字",
    "road": "道路特征",
    "driving_side": "行驶方向",
    "terrain": "地形地貌",
    "waterbody": "水体特征",
    "infrastructure": "基础设施",
}

LOCAL_VALUE_LABELS = {
    "left": "靠左行驶",
    "right": "靠右行驶",
    "devanagari": "天城文",
    "mountain": "山地",
    "mountains": "山地",
    "hills": "丘陵",
    "plateau": "高原",
    "plain": "平原",
    "plains": "平原",
    "valley": "山谷",
    "coast": "海岸",
    "coastal": "海岸",
    "sea": "海洋",
    "ocean": "海洋",
    "lake": "湖泊",
    "river": "河流",
    "canal": "运河",
    "harbor": "港口",
    "waterfall": "瀑布",
    "asphalt": "沥青路面",
    "concrete": "水泥路面",
    "gravel": "砾石路面",
    "dirt": "土路",
    "cobblestone": "石板路",
    "paved": "铺装道路",
    "unpaved": "未铺装道路",
    "railway": "铁路",
    "subway": "地铁",
    "tram": "有轨电车",
    "power_lines": "输电线路",
    "wind_turbine": "风力发电机",
    "bridge": "桥梁",
    "tunnel": "隧道",
    "airport": "机场",
}

# 提问维度的优先顺序：先问能直接区分候选的强线索维度。
QUESTION_DIMENSION_ORDER = (
    "language", "place_name", "driving_side", "signage", "road", "terrain",
    "waterbody", "vegetation", "building", "vehicle", "climate", "culture",
    "infrastructure",
)

# 无候选可区分时的默认选项（值域与共享线索目录保持一致）。
DEFAULT_OPTION_VALUES = {
    "language": ("ja", "ko", "zh", "latin", "cyrillic", "arabic", "thai", "devanagari"),
    "driving_side": ("left", "right"),
    "vegetation": ("temperate_forest", "grassland", "desert", "tropical_rainforest", "savanna"),
    "building": ("gothic", "modern_glass", "mediterranean", "colonial", "generic"),
    "climate": ("temperate", "mediterranean_sunny", "temperate_maritime", "cold_polar", "arid"),
    "terrain": ("mountain", "hills", "plain", "coast", "valley"),
    "waterbody": ("sea", "lake", "river", "canal", "harbor"),
    "road": ("asphalt", "concrete", "gravel", "dirt", "cobblestone"),
    "infrastructure": ("railway", "subway", "bridge", "power_lines", "tram"),
    "vehicle": ("dark_vehicle", "white_silver_vehicle"),
}

MAX_OPTIONS = 4
_RELIABLE_ORIGINS = {"ocr", "text_language"}


def dimension_label(dimension: str) -> str:
    """维度中文名：优先共享目录，其次本地回退，最后原样返回。"""
    return DIM_LABELS.get(dimension) or LOCAL_DIM_LABELS.get(dimension) or dimension


def value_label(value: str) -> str:
    """取值中文名：优先共享目录，其次本地回退，最后原样返回。"""
    return VALUE_LABELS.get(value) or LOCAL_VALUE_LABELS.get(value) or value


def is_user_evidence(clue: Clue) -> bool:
    """True only for clues that count as trustworthy user/OCR evidence.

    Heuristic image analyzers (color/climate guesses) deliberately return False
    so that dialogue keeps asking the user for a better observation.
    """
    source = str(getattr(clue, "source", "") or "").strip().lower()
    if source == "user":
        return True
    extra = clue.extra if isinstance(getattr(clue, "extra", None), dict) else {}
    if extra.get("reliable") is not True:
        return False
    origin = str(extra.get("origin_source") or source or "").strip().lower()
    return origin in _RELIABLE_ORIGINS


def user_known_dimensions(clues: Sequence[Clue]) -> Set[str]:
    """Dimensions with trustworthy user/OCR evidence (heuristics excluded)."""
    return {
        clue.dimension
        for clue in clues
        if getattr(clue, "confidence", 0.0) >= 0.3 and is_user_evidence(clue)
    }


@dataclass
class HypothesisAction:
    kind: str  # "confirm" | "ask" | "request_clue" | "done"
    message: str
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "message": self.message, "payload": self.payload}


class HypothesisManager:
    # Shared helpers exposed on the class so subclasses in other modules
    # (model_ranking.ModelAwareHypotheses) need no extra imports.
    dimension_label = staticmethod(dimension_label)
    value_label = staticmethod(value_label)
    _user_known_dimensions = staticmethod(user_known_dimensions)
    _is_user_evidence = staticmethod(is_user_evidence)
    QUESTION_ORDER = QUESTION_DIMENSION_ORDER
    DEFAULT_VALUES = DEFAULT_OPTION_VALUES
    MAX_OPTIONS = MAX_OPTIONS

    def __init__(self, engine: ReasoningEngine):
        self.engine = engine

    # ---- public entry point ----
    def next_action(self, scores: List[CountryScore],
                    clues: List[Clue],
                    asked: Optional[Sequence[str]] = None) -> HypothesisAction:
        asked_dims = {d for d in (asked or ()) if isinstance(d, str)}
        if not scores:
            missing = self._missing_dimensions(clues, asked_dims)
            lead = missing[0] if missing else None
            dims = missing[:3] or ["language", "culture"]
            detail = "、".join(dimension_label(d) for d in dims)
            return HypothesisAction(
                kind="request_clue",
                message=f"目前线索不足以推断任何国家，请补充{detail}等线索。",
                payload={
                    "dimensions": dims,
                    "options": self._options_for(lead, [], clues),
                    "reason": "当前没有任何匹配的候选国家",
                },
            )

        top1 = scores[0]
        top2 = scores[1] if len(scores) > 1 else None

        # 确认型：Top1 显著领先且置信度够高
        if top1.score >= CONFIRMATION_THRESHOLD and (top2 is None or top1.score - top2.score >= 0.2):
            return HypothesisAction(
                kind="confirm",
                message=f"我推断这张照片拍摄于「{top1.name_zh}」，是否正确？",
                payload={"iso": top1.iso, "name_zh": top1.name_zh, "score": top1.score},
            )

        # 区分型：候选接近，找一个能区分它们且尚未问过/已知的维度
        if top2 is not None and top1.score - top2.score < 0.3:
            dim = self._discriminating_dimension(top1, top2, clues, asked_dims)
            if dim:
                return HypothesisAction(
                    kind="ask",
                    message=f"候选国家「{top1.name_zh}」与「{top2.name_zh}」分数接近，"
                            f"请问该地点的{dimension_label(dim)}更符合哪一种特征？",
                    payload={
                        "dimensions": [dim],
                        "candidates": [top1.iso, top2.iso],
                        "options": self._options_for(dim, [top1, top2], clues),
                        "reason": f"该维度在「{top1.name_zh}」与「{top2.name_zh}」之间存在差异",
                    },
                )

        # 请求更多线索（跳过用户已知/已问过的维度）
        missing = self._missing_dimensions(clues, asked_dims)
        if missing:
            lead = missing[0]
            dims = missing[:3]
            return HypothesisAction(
                kind="request_clue",
                message=f"为了进一步缩小范围，请确认以下线索：{'、'.join(dimension_label(d) for d in dims)}。",
                payload={
                    "dimensions": dims,
                    "options": self._options_for(lead, [], clues),
                    "reason": "这些维度还没有可靠的用户证据",
                },
            )

        return HypothesisAction(
            kind="confirm",
            message=f"综合线索，最可能的拍摄地是「{top1.name_zh}」。",
            payload={"iso": top1.iso, "name_zh": top1.name_zh, "score": top1.score},
        )

    # ---- dimension helpers ----
    def _feature_values(self, iso: str, dimension: str) -> Set[str]:
        """Catalog values for one country/dimension (never invents values)."""
        from . import knowledge_graph as kg
        country = kg.get_by_iso(iso)
        if not country:
            return set()
        if dimension == "language":
            values = set(country.get("languages_spoken") or [])
            values |= set(country.get("features", {}).get("language") or [])
            return {v for v in values if isinstance(v, str) and v}
        if dimension == "driving_side":
            side = country.get("driving_side")
            return {side} if isinstance(side, str) and side else set()
        values = country.get("features", {}).get(dimension) or []
        return {v for v in values if isinstance(v, str) and v}

    def _discriminating_dimension(self, a: CountryScore, b: CountryScore,
                                  clues: List[Clue],
                                  asked: Optional[Set[str]] = None) -> Optional[str]:
        """找一个 a、b 特征不同且用户尚未给出可靠证据、也没问过的维度。"""
        blocked = user_known_dimensions(clues) | set(asked or ())
        for dim in self.QUESTION_ORDER:
            if dim in blocked or dim in {"place_name", "signage"}:
                continue
            va = self._feature_values(a.iso, dim)
            vb = self._feature_values(b.iso, dim)
            if va and vb and va != vb:
                return dim
        return None

    def _missing_dimensions(self, clues: List[Clue],
                            asked: Optional[Set[str]] = None) -> List[str]:
        blocked = user_known_dimensions(clues) | set(asked or ())
        return [d for d in self.QUESTION_ORDER if d not in blocked]

    # ---- structured options ----
    def _options_for(self, dimension: Optional[str], candidates: Sequence[CountryScore],
                     clues: List[Clue]) -> List[Dict[str, str]]:
        if not dimension:
            return []
        options: List[Dict[str, str]] = []
        if dimension == "place_name" and candidates:
            for country in candidates[:2]:
                options.append({"dimension": dimension, "value": country.iso,
                                "label": country.name_zh})
            return options[:self.MAX_OPTIONS]
        if candidates:
            sets = [self._feature_values(country.iso, dimension) for country in candidates]
            if len(sets) >= 2 and sets[0] and sets[1]:
                unique_a = sorted(sets[0] - sets[1])
                unique_b = sorted(sets[1] - sets[0])
                shared = sorted(sets[0] & sets[1])
                values = unique_a + unique_b + shared
            else:
                values = sorted(set().union(*sets)) if sets else []
        else:
            values = list(self.DEFAULT_VALUES.get(dimension, ()))
        for value in values:
            options.append({"dimension": dimension, "value": value, "label": value_label(value)})
            if len(options) >= self.MAX_OPTIONS:
                break
        return options
