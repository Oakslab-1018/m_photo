"""假设生成与验证：驱动人机协同的多轮交互。

每次推理后，引擎根据候选国家分布决定下一步动作：
- 若 Top1 置信度 >> Top2，且超过阈值 -> 提议确认（confirm）
- 若多个国家候选接近 -> 提出区分性问题（ask）
- 若线索不足 -> 主动询问关键维度（request_clue）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config import CONFIRMATION_THRESHOLD
from .clue import Clue, DIM_LABELS
from .engine import CountryScore, ReasoningEngine


@dataclass
class HypothesisAction:
    kind: str  # "confirm" | "ask" | "request_clue" | "done"
    message: str
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "message": self.message, "payload": self.payload}


class HypothesisManager:
    def __init__(self, engine: ReasoningEngine):
        self.engine = engine

    def next_action(self, scores: List[CountryScore],
                    clues: List[Clue]) -> HypothesisAction:
        if not scores:
            return HypothesisAction(
                kind="request_clue",
                message="目前线索不足以推断任何国家，请补充更多线索（如文字语言、植被类型等）。",
                payload={"dimensions": ["language", "vegetation", "climate", "building"]},
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

        # 区分型：候选接近，找一个能区分它们的维度提问
        if top2 is not None and top1.score - top2.score < 0.3:
            dim = self._discriminating_dimension(top1, top2, clues)
            if dim:
                return HypothesisAction(
                    kind="ask",
                    message=f"候选国家「{top1.name_zh}」与「{top2.name_zh}」分数接近，"
                            f"请问该地点的{DIM_LABELS.get(dim, dim)}更符合哪一种特征？",
                    payload={"dimensions": [dim], "candidates": [top1.iso, top2.iso]},
                )

        # 请求更多线索
        missing = self._missing_dimensions(clues)
        if missing:
            return HypothesisAction(
                kind="request_clue",
                message=f"为了进一步缩小范围，请确认以下线索：{', '.join(DIM_LABELS.get(d, d) for d in missing)}。",
                payload={"dimensions": missing},
            )

        return HypothesisAction(
            kind="confirm",
            message=f"综合线索，最可能的拍摄地是「{top1.name_zh}」。",
            payload={"iso": top1.iso, "name_zh": top1.name_zh, "score": top1.score},
        )

    def _discriminating_dimension(self, a: CountryScore, b: CountryScore,
                                  clues: List[Clue]) -> Optional[str]:
        """找一个 a、b 特征不同且用户尚未确认的维度。"""
        from . import knowledge_graph as kg
        ca = kg.get_by_iso(a.iso)
        cb = kg.get_by_iso(b.iso)
        if not ca or not cb:
            return None
        have_dims = {c.dimension for c in clues}
        for dim in ("language", "climate", "vegetation", "building", "vehicle"):
            if dim in have_dims:
                continue
            va = set(ca.get("features", {}).get(dim, []))
            vb = set(cb.get("features", {}).get(dim, []))
            if va != vb and (va - vb or vb - va):
                return dim
        return None

    def _missing_dimensions(self, clues: List[Clue]) -> List[str]:
        have = {c.dimension for c in clues if c.confidence >= 0.3}
        all_dims = ["language", "climate", "vegetation", "building", "vehicle", "culture"]
        return [d for d in all_dims if d not in have]
