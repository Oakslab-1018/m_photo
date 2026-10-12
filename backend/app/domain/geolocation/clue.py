"""线索卡（Clue Card）数据结构。

AI 感知结果被统一结构化为 Clue，每个 Clue 包含：
- dimension: 维度（如 place_name / signage / vegetation / driving_side）
- value: 线索取值（目录特征值、规范化词条或 OCR 原文）
- confidence: 置信度 [0, 1]
- evidence: 支撑证据（OCR 文本、检测框描述等）
- source: 来源分析器 / 语义分析器
- extra: 结构化补充（provenance、validated、display_only 等）

维度分成两类（单一真源，所有分析器与推理引擎共享）：

- ``CATALOG_DIMENSIONS``：有国家目录元数据支撑，可参与国家/城市排序；
- ``OBSERVATION_DIMENSIONS``：只有 OCR 文本观察、没有目录元数据，
  只能展示为观察，绝不允许暗示改变了排序。

``extra`` 约定（SemanticClueAnalyzer 输出）：
- 目录校验证据：``validated=True``、``reliable=True``、``affects_ranking=True``；
- 纯观察证据：``validated=False``、``reliable=False``、``display_only=True``、
  ``affects_ranking=False``；
- 任何来源都保留 ``origin_source`` / ``origin_confidence`` / ``ocr_occurrences``。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


# 线索维度枚举（单一真源，所有分析器与推理引擎共享）
DIMENSIONS = (
    "language",       # 文字语言
    "signage",        # 标志/招牌原文（观察）
    "place_name",     # 目录校验过的国家/地名（国家名、name_zh、ISO3）
    "road",           # 道路词汇（观察）
    "driving_side",   # 通行方向（目录 driving_side 字段）
    "terrain",        # 地形词汇（观察）
    "waterbody",      # 水体词汇（观察）
    "vegetation",     # 植被类型
    "building",       # 建筑风格
    "vehicle",        # 车辆类型/特征
    "climate",        # 气候特征
    "culture",        # 文化符号/地标（目录 culture 特征值）
    "infrastructure", # 基础设施词汇（观察）
)

# 有目录元数据、允许参与排序的维度（engine / city_reasoning 的评分白名单）
CATALOG_DIMENSIONS = (
    "language",
    "place_name",
    "driving_side",
    "vegetation",
    "building",
    "vehicle",
    "climate",
    "culture",
)

# 没有目录元数据的观察维度：只展示，不参与任何排序
OBSERVATION_DIMENSIONS = (
    "signage",
    "road",
    "terrain",
    "waterbody",
    "infrastructure",
)

# Color/brightness analyzers are display observations only. Their categorical
# guesses must never become country-ranking evidence, including for older
# persisted clues created before the metadata flags existed.
_AUTOMATED_HEURISTIC_SOURCES = frozenset({"climate", "vegetation", "building", "vehicle"})


def is_automated_heuristic_clue(clue: object) -> bool:
    """True for analyzer-produced color/brightness guesses.

    Checks source and origin_source for current analyzer output, then falls
    back to the heuristic dimensions so legacy persisted clues with a generic
    ``pipeline`` source stay display-only too. User/OCR/semantic evidence is
    exempt unless it originated from one of the automated analyzers.
    """
    source = str(getattr(clue, "source", "") or "").strip().lower()
    extra = getattr(clue, "extra", None)
    extra = extra if isinstance(extra, dict) else {}
    origin = str(extra.get("origin_source") or source or "").strip().lower()
    if source in _AUTOMATED_HEURISTIC_SOURCES or origin in _AUTOMATED_HEURISTIC_SOURCES:
        return True
    dimension = str(getattr(clue, "dimension", "") or "").strip().lower()
    trusted_source = source in {"user", "ocr", "text_language", "semantic"}
    explicit_trust = extra.get("validated") is True or extra.get("reliable") is True
    return dimension in _AUTOMATED_HEURISTIC_SOURCES and not (trusted_source or explicit_trust)

# 维度中文名映射（与 DIMENSIONS 保持一致）
DIM_LABELS = {
    "language": "文字语言",
    "signage": "招牌文字",
    "place_name": "地名证据",
    "road": "道路词汇",
    "driving_side": "通行方向",
    "terrain": "地形词汇",
    "waterbody": "水体词汇",
    "vegetation": "植被类型",
    "building": "建筑风格",
    "vehicle": "车辆特征",
    "climate": "气候特征",
    "culture": "文化符号",
    "infrastructure": "基础设施",
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
    # driving_side（目录字段）
    "left": "左行交通",
    "right": "右行交通",
    # road（观察词条）
    "street": "街道",
    "road": "道路",
    "avenue": "大道",
    "boulevard": "林荫大道",
    "highway": "高速公路",
    "freeway": "高速公路",
    "motorway": "高速公路",
    "route": "路线",
    "alley": "小巷",
    "lane": "车道",
    "square": "广场",
    "plaza": "广场",
    "大道": "大道",
    "公路": "公路",
    "街道": "街道",
    "高速公路": "高速公路",
    "快速路": "快速路",
    "环路": "环路",
    # terrain（观察词条）
    "mountain": "山地",
    "hill": "丘陵",
    "valley": "山谷",
    "coast": "海岸",
    "island": "岛屿",
    "glacier": "冰川",
    "volcano": "火山",
    "canyon": "峡谷",
    "plateau": "高原",
    "cliff": "悬崖",
    "山脉": "山脉",
    "山谷": "山谷",
    "火山": "火山",
    "冰川": "冰川",
    "峡谷": "峡谷",
    "高原": "高原",
    "岛屿": "岛屿",
    "海岸": "海岸",
    # waterbody（观察词条）
    "lake": "湖泊",
    "river": "河流",
    "sea": "海洋",
    "ocean": "海洋",
    "bay": "海湾",
    "harbor": "港口",
    "harbour": "港口",
    "beach": "海滩",
    "waterfall": "瀑布",
    "reservoir": "水库",
    "strait": "海峡",
    "fjord": "峡湾",
    "湖泊": "湖泊",
    "河流": "河流",
    "海湾": "海湾",
    "海滩": "海滩",
    "瀑布": "瀑布",
    "水库": "水库",
    "海峡": "海峡",
    # infrastructure（观察词条）
    "station": "车站",
    "airport": "机场",
    "port": "港口",
    "bridge": "桥梁",
    "tunnel": "隧道",
    "dam": "水坝",
    "metro": "地铁",
    "subway": "地铁",
    "stadium": "体育场",
    "terminal": "航站楼",
    "lighthouse": "灯塔",
    "observatory": "天文台",
    "university": "大学",
    "hospital": "医院",
    "museum": "博物馆",
    "车站": "车站",
    "机场": "机场",
    "港口": "港口",
    "大桥": "大桥",
    "隧道": "隧道",
    "水坝": "水坝",
    "地铁": "地铁",
    "体育场": "体育场",
    "博物馆": "博物馆",
    "大学": "大学",
    "医院": "医院",
    "灯塔": "灯塔",
    # culture（国家目录 culture 特征值）
    "acropolis": "卫城",
    "aztec": "阿兹特克",
    "big_ben": "大本钟",
    "cafe": "咖啡馆",
    "castle": "城堡",
    "colosseum": "罗马斗兽场",
    "eiffel": "埃菲尔铁塔",
    "elephant": "大象",
    "flamenco": "弗拉门戈",
    "hanbok": "韩服",
    "kremlin": "克里姆林宫",
    "lantern": "灯笼",
    "maple": "枫叶",
    "opera_house": "歌剧院",
    "pagoda": "佛塔",
    "pizza": "披萨",
    "pyramid": "金字塔",
    "red_bus": "红色巴士",
    "sakura": "樱花",
    "samba": "桑巴",
    "sari": "纱丽",
    "shrine": "神社",
    "statue_of_liberty": "自由女神像",
    "temple": "寺庙",
    "viking": "维京",
    "windmill": "风车",
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


__all__ = [
    "CATALOG_DIMENSIONS",
    "DIMENSIONS",
    "DIM_LABELS",
    "OBSERVATION_DIMENSIONS",
    "VALUE_LABELS",
    "is_automated_heuristic_clue",
    "Clue",
    "clues_to_dicts",
    "dicts_to_clues",
]