"""线索卡（Clue Card）数据结构。

AI 感知结果被统一结构化为 Clue，每个 Clue 包含：
- dimension: 维度（如 vegetation / building / vehicle / language / climate）
- value: 线索取值（如 "temperate_forest" / "gothic" / "sedan"）
- confidence: 置信度 [0, 1]
- evidence: 支撑证据（OCR 文本、检测框描述等）
- source: 来源分析器
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


# 线索维度枚举（单一真源，所有分析器与推理引擎共享）
DIMENSIONS = (
    "language",       # 文字语言
    "vegetation",     # 植被类型
    "building",       # 建筑风格
    "vehicle",        # 车辆类型/特征
    "climate",        # 气候特征
    "signage",        # 标志/招牌文字
    "culture",        # 文化符号
)

# 维度中文名映射（与 DIMENSIONS 保持一致）
DIM_LABELS = {
    "language": "文字语言",
    "vegetation": "植被类型",
    "building": "建筑风格",
    "vehicle": "车辆特征",
    "climate": "气候特征",
    "signage": "招牌文字",
    "culture": "文化符号",
}

# 线索取值中文名映射（前端展示与后端消息共用，保持语言统一）
VALUE_LABELS = {
    # vegetation
    "temperate_forest": "温带森林",
    "grassland": "草原",
    "savanna": "稀树草原",
    "desert": "荒漠",
    "tropical_rainforest": "热带雨林",
    "urban_sparse": "城市稀疏植被",
    # vehicle
    "dark_vehicle": "深色车辆",
    "white_silver_vehicle": "白/银色车辆",
    "unknown_vehicle": "未识别车辆",
    # building
    "mediterranean": "地中海风格",
    "colonial": "殖民风格",
    "gothic": "哥特式",
    "modern_glass": "现代玻璃幕墙",
    "generic": "通用/不明显",
    # climate
    "mediterranean_sunny": "地中海晴朗气候",
    "temperate_maritime": "温带海洋性气候",
    "cold_polar": "寒冷气候",
    "arid": "干旱气候",
    "temperate": "温带气候",
    # language
    "zh": "中文",
    "ja": "日语",
    "ko": "韩语",
    "latin": "拉丁语系",
    "cyrillic": "西里尔语系",
    "arabic": "阿拉伯语",
    "thai": "泰语",
    "unknown": "未知",
}


@dataclass
class Clue:
    dimension: str
    value: str
    confidence: float
    evidence: str = ""
    source: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Clue":
        return cls(
            dimension=d["dimension"],
            value=d["value"],
            confidence=float(d.get("confidence", 0.0)),
            evidence=d.get("evidence", ""),
            source=d.get("source", ""),
            extra=d.get("extra", {}),
        )


def clues_to_dicts(clues: List[Clue]) -> List[Dict[str, Any]]:
    return [c.to_dict() for c in clues]


def dicts_to_clues(items: List[Dict[str, Any]]) -> List[Clue]:
    return [Clue.from_dict(i) for i in items]
