"""人机协同多轮对话服务。

实现 "AI 线索输出 -> 用户反馈 -> AI 迭代分析" 的假设-验证循环。

支持的用户反馈类型：

- confirm  : 确认某国家（设置 confirmed_iso，结束会话）
- reject   : 排除某国家（加入 exclusions，重新推理）
- answer   : 回答某维度特征（追加线索，重新推理）
- free_text: 中文自由文本，做保守的自然语言解析

自由文本解析是不可绕过的确定性底线（可选 LLM 抽取只作为补充，并且要
重新通过同一套校验）：

1. 疑问句（？/吗/是不是/…）不产生任何正向线索，也不产生排除；
2. 否定与不确定（不是/可能/好像/应该/…）不产生正向线索；
3. 排除只在显式否定（"不是日本"/"排除日本"）且确有地点语境时成立，
   顺带提及（"日本品牌"/"去过日本"）不算；
4. 修正语句（"说错了，其实是日本"）可以收回此前的排除；
5. 地点线索只在拍摄/画面语境、"是/在/位于"直陈或回答上一轮地名问题时
   成立，并且必须校验到国家目录内的国家/ISO；
6. 上一轮 AI 问题的维度/选项会成为语境：短回答（"左边"、点击选项 chip）
   也能转化为对应维度的线索。
"""
from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from backend.app.domain.geolocation.clue import Clue, DIM_LABELS, VALUE_LABELS, dicts_to_clues
from backend.app.domain.geolocation.engine import ReasoningEngine
from backend.app.domain.geolocation.hypotheses import HypothesisManager, dimension_label, value_label
from backend.app.domain.geolocation import knowledge_graph as kg


# ---------------------------------------------------------------------------
# 共享维度与词汇（只覆盖请求的维度；值域与共享线索目录一致，不发明新取值）
# ---------------------------------------------------------------------------
REMOTE_DIMENSIONS = (
    "language", "signage", "place_name", "road", "driving_side", "terrain",
    "waterbody", "vegetation", "building", "vehicle", "climate", "culture",
    "infrastructure",
)

DIMENSION_CONFIDENCE = {
    "language": 0.8,
    "place_name": 0.85,
    "driving_side": 0.8,
    "vegetation": 0.7,
    "climate": 0.7,
    "building": 0.7,
    "culture": 0.7,
    "road": 0.7,
    "terrain": 0.7,
    "waterbody": 0.7,
    "infrastructure": 0.7,
}

_QUESTION_TOKENS = (
    "？", "?", "吗", "呢", "是不是", "是否", "什么", "哪", "怎么", "为什么",
    "请问", "如何", "多少",
)
_UNCERTAIN_TOKENS = (
    "不确定", "不清楚", "不知道", "记不清", "说不好", "不觉得", "不认为",
    "可能", "也许", "或许", "大概", "好像", "似乎", "估计", "推测", "猜测",
    "说不定", "应该", "怀疑", "有点像",
)
_REJECT_TOKENS = (
    "绝对不是", "肯定不是", "不可能是", "绝不可能是", "绝非", "并非", "排除",
    "不在", "不是",
)
_NEGATION_TOKENS = (
    "不是", "并非", "没有", "没在", "不在", "排除", "不可能是", "绝非", "不像是",
)
_CORRECTION_TOKENS = ("说错", "搞错", "记错", "说反", "更正", "修正", "收回", "改成", "重新说", "其实是")
# 强自我修正标记：声明"之前说错了"，同批新线索覆盖对应维度的旧用户线索。
_SELF_CORRECTION_TOKENS = ("说错", "搞错", "记错", "说反", "更正", "修正", "收回", "改成", "重新说")
_SEGMENT_BREAKS = ("但是", "但", "然而", "而是", "只有")
_NON_EVIDENCE_TOKENS = (
    "想", "打算", "准备", "计划", "希望", "要去", "会去", "去过", "听说", "据说",
    "留学", "旅游", "度假",
)
_PLACE_CONTEXT_TOKENS = (
    "照片", "图片", "图里", "画面", "拍摄", "拍的", "拍照", "地点", "定位", "取景",
    "背景里", "写着", "招牌", "路牌", "标牌", "路标", "地标", "街景", "门牌",
)
_PLACE_SUFFIX_GUARDS = (
    "品牌", "产品", "人", "菜", "料理", "车", "队", "公司", "企业", "货", "电影",
    "动漫", "漫画", "游戏", "文化", "历史", "游客", "旅客", "使馆", "签证", "航空",
    "语言", "语", "文", "国籍", "护照", "国旗", "军队", "政府", "电视剧", "音乐",
    "球队", "明星", "口音", "风味", "进口", "片",
)
_AFFIRMATIVE_REPLIES = {
    "是", "是的", "对", "对的", "没错", "正确", "确认", "确定", "就是",
    "是这里", "是这个", "嗯", "嗯嗯",
}
_NEGATIVE_REPLIES = {
    "不是", "不对", "不是的", "不对的", "错了", "不正确", "不准确", "否",
    "不是这里", "不是这个", "不是它",
}

_ASCII_RE = re.compile(r"[A-Za-z]")
_CLAUSE_BOUNDARY = re.compile(r"[。！!？?；;，,\n\r、]+")

# 中文国家别名（共享目录之外的常见称谓；ISO 始终来自目录校验）
_EXTRA_PLACE_ALIASES = {
    "澳洲": "AU", "南韩": "KR", "北韩": "KP", "大韩民国": "KR", "日本国": "JP",
    "美利坚": "US", "英格兰": "GB", "苏格兰": "GB", "威尔士": "GB", "大不列颠": "GB",
}

# 常见城市/地区 -> 国家 ISO（仅用于把用户写出的地名落到目录内国家）
_CITY_ALIASES = {
    "东京": "JP", "大阪": "JP", "京都": "JP", "北海道": "JP", "冲绳": "JP",
    "首尔": "KR", "釜山": "KR", "济州": "KR",
    "北京": "CN", "上海": "CN", "广州": "CN", "深圳": "CN", "杭州": "CN",
    "西安": "CN", "成都": "CN", "香港": "HK", "澳门": "MO", "台北": "TW",
    "曼谷": "TH", "清迈": "TH", "普吉": "TH",
    "巴黎": "FR", "马赛": "FR", "尼斯": "FR",
    "伦敦": "GB", "曼彻斯特": "GB", "爱丁堡": "GB",
    "柏林": "DE", "慕尼黑": "DE", "法兰克福": "DE",
    "罗马": "IT", "米兰": "IT", "威尼斯": "IT", "佛罗伦萨": "IT",
    "马德里": "ES", "巴塞罗那": "ES", "雅典": "GR",
    "莫斯科": "RU", "圣彼得堡": "RU",
    "纽约": "US", "洛杉矶": "US", "旧金山": "US", "华盛顿": "US", "芝加哥": "US",
    "波士顿": "US", "西雅图": "US",
    "多伦多": "CA", "温哥华": "CA", "蒙特利尔": "CA",
    "悉尼": "AU", "墨尔本": "AU", "布里斯班": "AU", "珀斯": "AU",
    "圣保罗": "BR", "里约热内卢": "BR", "开罗": "EG", "孟买": "IN", "新德里": "IN",
    "墨西哥城": "MX", "阿姆斯特丹": "NL", "鹿特丹": "NL", "斯德哥尔摩": "SE",
}

# 常见城市/地区英文写法（仍限定在目录内国家，仅作为字面地名线索）
_CITY_ALIASES_EN = {
    "tokyo": "JP", "osaka": "JP", "kyoto": "JP", "sapporo": "JP",
    "seoul": "KR", "busan": "KR", "bangkok": "TH", "chiang mai": "TH",
    "beijing": "CN", "shanghai": "CN", "guangzhou": "CN", "shenzhen": "CN",
    "hong kong": "HK", "macau": "MO", "taipei": "TW",
    "paris": "FR", "london": "GB", "edinburgh": "GB",
    "berlin": "DE", "munich": "DE", "frankfurt": "DE",
    "rome": "IT", "milan": "IT", "venice": "IT",
    "madrid": "ES", "barcelona": "ES", "athens": "GR",
    "moscow": "RU", "st petersburg": "RU",
    "new york": "US", "los angeles": "US", "san francisco": "US", "chicago": "US",
    "toronto": "CA", "vancouver": "CA", "sydney": "AU", "melbourne": "AU",
    "sao paulo": "BR", "rio de janeiro": "BR", "cairo": "EG", "mumbai": "IN",
    "new delhi": "IN", "mexico city": "MX", "amsterdam": "NL", "stockholm": "SE",
}

# 城市名后紧跟这些字时更像街名，而不是城市证据（"北京路"）。
_CITY_SUFFIX_GUARDS = ("路", "道", "巷")

# 用户明确描述招牌/路牌文字时才抽取字面 signage 线索。
_SIGNAGE_MARKERS = ("写着", "写的是", "招牌", "路牌", "标牌", "门牌", "告示", "指示牌")
_QUOTE_PATTERN = re.compile(r"[“\"'「『](?P<text>[^”\"'」』]{2,40})[”\"'」』]")
_INLINE_SIGNAGE_RE = re.compile(
    r"(?:上?写着|写的是)[:：]?(?P<text>[A-Za-z0-9\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff .,·'\-]{1,30})"
)


def _has_any(text: str, tokens: Sequence[str]) -> bool:
    return any(token in text for token in tokens)


def _segment_before(text: str, index: int) -> str:
    """Clause-local prefix before ``index`` that resets at coordination words.

    "没有日语只有韩文" must suppress 日语 but keep 韩文, so the negation check
    only looks at the text since the last "但/而是/只有" style break.
    """
    prefix = text[:max(0, index)]
    start = 0
    for token in _SEGMENT_BREAKS:
        pos = prefix.rfind(token)
        if pos >= 0:
            start = max(start, pos + len(token))
    return prefix[start:]


def _is_negated(segment: str) -> bool:
    return _has_any(segment, _NEGATION_TOKENS)


def _split_clauses(text: str) -> List[str]:
    """Split on Chinese/ASCII punctuation, keeping the punctuation with the
    preceding clause so "？" and "吗" still mark questions."""
    clauses: List[str] = []
    start = 0
    for match in _CLAUSE_BOUNDARY.finditer(text):
        segment = text[start:match.end()]
        if segment.strip():
            clauses.append(segment)
        start = match.end()
    tail = text[start:]
    if tail.strip():
        clauses.append(tail)
    return clauses


def _clause_containing(text: str, index: int) -> str:
    start = 0
    for match in _CLAUSE_BOUNDARY.finditer(text):
        if match.end() > index:
            return text[start:match.end()]
        start = match.end()
    return text[start:]


def _normalize(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower(), flags=re.UNICODE)


def _truncate(text: Any, limit: int) -> str:
    value = str(text or "")
    return value if len(value) <= limit else value[:limit]


def _find_anchor(text: str, candidates: Sequence[Any]) -> Optional[Tuple[int, str]]:
    lowered = text.lower()
    best: Optional[Tuple[int, str]] = None
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        pos = lowered.find(candidate.lower())
        if pos >= 0 and (best is None or pos < best[0]):
            best = (pos, candidate)
    return best


@lru_cache(maxsize=1)
def _place_alias_matcher() -> Tuple[re.Pattern, Dict[str, Tuple[str, str]]]:
    """Bounded alias table from the country catalog plus curated city names.

    Each alias maps to ``(kind, hint)``: country names/aliases carry a
    catalog-validated ISO code; city names stay literal text (a later,
    catalog-backed city resolver owns turning a literal into a city/country).
    Bare ISO codes are deliberately excluded from free-text matching.
    """
    table: Dict[str, Tuple[str, str]] = {}
    for country in kg.load_countries():
        iso = country.get("iso")
        if not isinstance(iso, str):
            continue
        for key in (country.get("name_zh"), country.get("name")):
            if isinstance(key, str) and key.strip():
                table.setdefault(key.strip(), ("country", iso))
    valid = set(kg.all_isos())
    for alias, iso in _EXTRA_PLACE_ALIASES.items():
        if iso in valid:
            table.setdefault(alias, ("country", iso))
    for alias, iso in _CITY_ALIASES.items():
        if iso in valid:
            table.setdefault(alias, ("city", iso))
    for alias, iso in _CITY_ALIASES_EN.items():
        if iso in valid:
            table.setdefault(alias, ("city", iso))
    entries: List[str] = []
    for alias in sorted(table, key=len, reverse=True):
        escaped = re.escape(alias)
        if _ASCII_RE.search(alias):
            entries.append(r"(?<![A-Za-z0-9])" + escaped + r"(?![A-Za-z0-9])")
        else:
            entries.append(escaped)
    if not entries:
        return re.compile(r"(?!)"), {}
    pattern = re.compile("|".join(entries), re.IGNORECASE)
    lookup = {alias.lower(): table[alias] for alias in table}
    return pattern, lookup


@dataclass
class PendingQuestion:
    """The previous AI question, used as dialogue context for short answers."""

    kind: str = ""
    dimensions: List[str] = field(default_factory=list)
    options: List[Dict[str, str]] = field(default_factory=list)
    iso: Optional[str] = None
    name_zh: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "dimensions": list(self.dimensions),
                "options": list(self.options), "iso": self.iso, "name_zh": self.name_zh}

    @classmethod
    def from_action(cls, action: Any) -> Optional["PendingQuestion"]:
        if not isinstance(action, dict):
            return None
        payload = action.get("payload") if isinstance(action.get("payload"), dict) else {}
        kind = action.get("kind") if isinstance(action.get("kind"), str) else ""
        dimensions = [d for d in payload.get("dimensions") or []
                      if isinstance(d, str) and d]
        options: List[Dict[str, str]] = []
        for option in payload.get("options") or []:
            if not isinstance(option, dict):
                continue
            dimension = option.get("dimension")
            value = option.get("value")
            if not (isinstance(dimension, str) and dimension and isinstance(value, str) and value):
                continue
            entry = {"dimension": dimension, "value": value}
            label = option.get("label")
            if isinstance(label, str) and label:
                entry["label"] = label
            options.append(entry)
        iso = payload.get("iso") if isinstance(payload.get("iso"), str) else None
        name = payload.get("name_zh") if isinstance(payload.get("name_zh"), str) else ""
        return cls(kind=kind, dimensions=dimensions, options=options, iso=iso, name_zh=name)


@dataclass
class ParsedFeedback:
    """Deterministic parse result for one free-text message."""

    clues: List[Dict[str, Any]] = field(default_factory=list)
    reject_isos: List[str] = field(default_factory=list)
    clear_exclusions: List[str] = field(default_factory=list)
    # Bounded correction plan: drop only these *user* clues before appending
    # the new evidence (OCR/heuristic evidence is never touched).
    replace_dimensions: List[str] = field(default_factory=list)
    remove_clue_values: List[Tuple[str, str]] = field(default_factory=list)
    confirm_iso: Optional[str] = None
    question: bool = False

    def add_clue(self, dimension: str, value: str, confidence: float, evidence: str = "",
                 literal: str = "", parser: str = "rules",
                 extra: Optional[Dict[str, Any]] = None) -> None:
        dimension = str(dimension or "").strip()
        value = str(value or "").strip()
        if not dimension or not value:
            return
        for existing in self.clues:
            if existing["dimension"] == dimension and existing["value"] == value:
                if confidence > existing["confidence"]:
                    existing["confidence"] = round(float(confidence), 3)
                    if evidence:
                        existing["evidence"] = _truncate(evidence, 200)
                return
        entry_extra: Dict[str, Any] = {"literal": _truncate(literal or value, 60), "parser": parser}
        if isinstance(extra, dict):
            entry_extra.update(extra)
        self.clues.append({
            "dimension": dimension,
            "value": value[:64],
            "confidence": round(float(confidence), 3),
            "evidence": _truncate(evidence or literal, 200),
            "source": "user",
            "extra": entry_extra,
        })

    def add_reject(self, iso: str) -> None:
        iso = str(iso or "").strip().upper()
        if len(iso) == 2 and iso not in self.reject_isos:
            self.reject_isos.append(iso)

    def add_clear_exclusion(self, iso: str) -> None:
        iso = str(iso or "").strip().upper()
        if len(iso) == 2 and iso not in self.clear_exclusions and iso not in self.reject_isos:
            self.clear_exclusions.append(iso)

    def add_replace_dimension(self, dimension: str) -> None:
        dimension = str(dimension or "").strip()
        if dimension and dimension not in self.replace_dimensions:
            self.replace_dimensions.append(dimension)

    def add_removal(self, dimension: str, value: str) -> None:
        dimension = str(dimension or "").strip()
        value = str(value or "").strip()
        if dimension and value and (dimension, value) not in self.remove_clue_values:
            self.remove_clue_values.append((dimension, value))

    def merged(self, other: "ParsedFeedback") -> "ParsedFeedback":
        """Deterministic result plus validated remote additions.

        Only clues/rejections are merged: a remote payload can never confirm a
        country or remove an exclusion.
        """
        result = ParsedFeedback(
            clues=[dict(clue) for clue in self.clues],
            reject_isos=list(self.reject_isos),
            clear_exclusions=list(self.clear_exclusions),
            replace_dimensions=list(self.replace_dimensions),
            remove_clue_values=list(self.remove_clue_values),
            confirm_iso=self.confirm_iso,
            question=self.question,
        )
        if not isinstance(other, ParsedFeedback):
            return result
        for clue in other.clues:
            extra = clue.get("extra") if isinstance(clue.get("extra"), dict) else {}
            result.add_clue(clue.get("dimension"), clue.get("value"),
                            clue.get("confidence", 0.0), clue.get("evidence", ""),
                            extra.get("literal", ""), extra.get("parser", "rules"))
        for iso in other.reject_isos:
            result.add_reject(iso)
        for iso in other.clear_exclusions:
            result.add_clear_exclusion(iso)
        for dimension in other.replace_dimensions:
            result.add_replace_dimension(dimension)
        for dimension, value in other.remove_clue_values:
            result.add_removal(dimension, value)
        return result


class ChineseFreeTextParser:
    """Deterministic, bounded Chinese free-text parser for dialogue clues."""

    PHRASES: Dict[str, Tuple[Tuple[str, str], ...]] = {
        "language": (
            ("日本语", "ja"), ("日语", "ja"), ("日文", "ja"), ("假名", "ja"),
            ("片假名", "ja"), ("平假名", "ja"),
            ("韩语", "ko"), ("韩文", "ko"), ("朝鲜语", "ko"), ("谚文", "ko"),
            ("中文", "zh"), ("汉语", "zh"), ("汉字", "zh"), ("简体字", "zh"),
            ("繁体字", "zh"),
            ("俄语", "cyrillic"), ("俄文", "cyrillic"), ("西里尔", "cyrillic"),
            ("阿拉伯语", "arabic"), ("阿拉伯文", "arabic"),
            ("泰语", "thai"), ("泰文", "thai"),
            ("英语", "latin"), ("英文", "latin"), ("法语", "latin"), ("法文", "latin"),
            ("德语", "latin"), ("德文", "latin"), ("西班牙语", "latin"),
            ("西班牙文", "latin"), ("葡萄牙语", "latin"), ("意大利语", "latin"),
            ("荷兰语", "latin"), ("越南语", "latin"), ("拉丁字母", "latin"),
            ("拉丁语", "latin"), ("印地语", "devanagari"), ("天城文", "devanagari"),
        ),
        "vegetation": (
            ("热带雨林", "tropical_rainforest"), ("雨林", "tropical_rainforest"),
            ("稀树草原", "savanna"),
            ("温带森林", "temperate_forest"), ("阔叶林", "temperate_forest"),
            ("针叶林", "temperate_forest"), ("森林", "temperate_forest"),
            ("草原", "grassland"), ("草地", "grassland"), ("牧场", "grassland"),
            ("荒漠", "desert"), ("沙漠", "desert"), ("戈壁", "desert"),
            ("植被稀少", "urban_sparse"), ("城市稀疏植被", "urban_sparse"),
        ),
        "climate": (
            ("地中海气候", "mediterranean_sunny"), ("地中海", "mediterranean_sunny"),
            ("温带海洋", "temperate_maritime"), ("海洋性气候", "temperate_maritime"),
            ("寒冷", "cold_polar"), ("极地", "cold_polar"), ("寒带", "cold_polar"),
            ("干旱", "arid"), ("干燥", "arid"),
            ("温带", "temperate"),
        ),
        "building": (
            ("哥特式", "gothic"), ("哥特", "gothic"),
            ("地中海风格", "mediterranean"), ("地中海建筑", "mediterranean"),
            ("殖民风格", "colonial"), ("殖民地风格", "colonial"), ("红砖", "colonial"),
            ("玻璃幕墙", "modern_glass"), ("现代建筑", "modern_glass"), ("高楼", "modern_glass"),
        ),
        "culture": (
            ("神社", "shrine"), ("鸟居", "shrine"), ("樱花", "sakura"),
            ("佛塔", "pagoda"), ("宝塔", "pagoda"), ("灯笼", "lantern"),
            ("韩服", "hanbok"), ("埃菲尔铁塔", "eiffel"), ("咖啡馆", "cafe"),
            ("城堡", "castle"), ("斗兽场", "colosseum"), ("披萨", "pizza"),
            ("大本钟", "big_ben"), ("双层巴士", "red_bus"), ("弗拉门戈", "flamenco"),
            ("卫城", "acropolis"), ("克里姆林宫", "kremlin"), ("自由女神", "statue_of_liberty"),
            ("枫叶", "maple"), ("桑巴", "samba"), ("歌剧院", "opera_house"),
            ("寺庙", "temple"), ("神庙", "temple"), ("纱丽", "sari"),
            ("大象", "elephant"), ("金字塔", "pyramid"), ("阿兹特克", "aztec"),
            ("风车", "windmill"), ("维京", "viking"),
        ),
        "driving_side": (
            ("靠左行驶", "left"), ("靠左", "left"), ("左侧通行", "left"), ("左行", "left"),
            ("右舵", "left"), ("方向盘在右", "left"),
            ("靠右行驶", "right"), ("靠右", "right"), ("右侧通行", "right"), ("右行", "right"),
            ("左舵", "right"), ("方向盘在左", "right"),
        ),
        "road": (
            ("未铺装", "unpaved"),
            ("沥青", "asphalt"), ("柏油", "asphalt"),
            ("水泥路", "concrete"), ("水泥路面", "concrete"),
            ("砂石路", "gravel"), ("碎石路", "gravel"),
            ("土路", "dirt"), ("泥路", "dirt"),
            ("石板路", "cobblestone"), ("石板", "cobblestone"),
            ("铺装", "paved"),
        ),
        "terrain": (
            ("山地", "mountain"), ("山脉", "mountain"), ("群山", "mountain"),
            ("山区", "mountain"), ("雪山", "mountain"),
            ("丘陵", "hills"), ("高原", "plateau"), ("平原", "plain"),
            ("盆地", "valley"), ("山谷", "valley"),
            ("海岸", "coast"), ("海滨", "coast"), ("沿海", "coast"),
        ),
        "waterbody": (
            ("运河", "canal"), ("水库", "lake"), ("湖泊", "lake"), ("湖边", "lake"),
            ("河流", "river"), ("河边", "river"),
            ("海洋", "sea"), ("大海", "sea"), ("海边", "sea"), ("海滩", "sea"),
            ("港口", "harbor"), ("码头", "harbor"), ("瀑布", "waterfall"),
        ),
        "infrastructure": (
            ("铁路", "railway"), ("火车", "railway"), ("铁道", "railway"),
            ("地铁", "subway"), ("有轨电车", "tram"), ("电车", "tram"),
            ("输电线路", "power_lines"), ("高压线", "power_lines"), ("电线杆", "power_lines"),
            ("风力发电机", "wind_turbine"), ("风力发电", "wind_turbine"),
            ("桥梁", "bridge"), ("大桥", "bridge"), ("隧道", "tunnel"), ("机场", "airport"),
        ),
    }

    CONTEXT_PHRASES: Dict[str, Tuple[Tuple[str, str], ...]] = {
        "driving_side": (
            ("左", "left"), ("左边", "left"), ("靠左", "left"),
            ("右", "right"), ("右边", "right"), ("靠右", "right"),
        ),
        "waterbody": (
            ("湖", "lake"), ("海", "sea"), ("河", "river"), ("江", "river"),
        ),
        "terrain": (
            ("山", "mountain"), ("丘陵", "hills"), ("平原", "plain"), ("高原", "plateau"),
        ),
    }

    LOCAL_VOCAB: Dict[str, Tuple[str, ...]] = {
        "language": ("zh", "ja", "ko", "latin", "cyrillic", "arabic", "thai", "devanagari"),
        "driving_side": ("left", "right"),
        "terrain": ("mountain", "hills", "plateau", "plain", "valley", "coast"),
        "waterbody": ("sea", "ocean", "lake", "river", "canal", "harbor", "waterfall"),
        "road": ("asphalt", "concrete", "gravel", "dirt", "cobblestone", "paved", "unpaved"),
        "infrastructure": ("railway", "subway", "tram", "power_lines", "wind_turbine",
                           "bridge", "tunnel", "airport"),
        "vegetation": ("temperate_forest", "grassland", "savanna", "desert",
                       "tropical_rainforest", "urban_sparse"),
        "building": ("gothic", "modern_glass", "mediterranean", "colonial", "generic"),
        "vehicle": ("dark_vehicle", "white_silver_vehicle", "unknown_vehicle"),
        "climate": ("temperate", "mediterranean_sunny", "temperate_maritime",
                    "cold_polar", "arid"),
        "culture": (),
    }

    EXTRA_VALUE_ANCHORS: Dict[Tuple[str, str], Tuple[str, ...]] = {
        ("language", "ja"): ("假名", "片假名", "平假名", "日文", "和文"),
        ("language", "ko"): ("谚文", "朝鲜文", "朝鲜语"),
        ("language", "zh"): ("汉字", "简体", "繁体", "汉语"),
        ("language", "latin"): ("拉丁", "字母"),
        ("driving_side", "left"): ("靠左", "左侧", "左行", "右舵"),
        ("driving_side", "right"): ("靠右", "右侧", "右行", "左舵"),
        ("terrain", "mountain"): ("山", "雪山", "群山"),
        ("terrain", "hills"): ("丘陵", "起伏"),
        ("terrain", "coast"): ("海", "沿海"),
        ("waterbody", "sea"): ("海", "海边", "海滩"),
        ("waterbody", "lake"): ("湖", "水库"),
        ("waterbody", "river"): ("河", "江"),
        ("road", "dirt"): ("土路", "泥路"),
        ("road", "gravel"): ("砂石", "碎石"),
        ("infrastructure", "railway"): ("铁路", "火车", "铁道"),
    }

    REMOTE_CONFIDENCE = 0.85
    _REMOTE_INTENTS = {"assert", "answer", "reject", "uncertain", "question", "other"}

    def __init__(self) -> None:
        entries: List[Tuple[str, str, str]] = []
        for dimension, phrases in self.PHRASES.items():
            for phrase, value in phrases:
                entries.append((phrase, dimension, value))
        # 长短语优先，保证 "稀树草原" 不会被 "草原" 抢先匹配。
        entries.sort(key=lambda item: (-len(item[0]), item[0]))
        self._phrase_index = tuple(entries)

    # ---- history / context ----
    @classmethod
    def pending_from_history(cls, messages: Sequence[Dict[str, Any]]) -> Optional[PendingQuestion]:
        for message in reversed(list(messages or ())):
            if not isinstance(message, dict) or message.get("role") != "ai":
                continue
            meta = message.get("meta") if isinstance(message.get("meta"), dict) else {}
            return PendingQuestion.from_action(meta.get("action"))
        return None

    # ---- deterministic parse ----
    def parse(self, text: str, pending: Optional[PendingQuestion] = None) -> ParsedFeedback:
        parsed = ParsedFeedback()
        raw = (text or "").strip()
        if not raw:
            return parsed
        evidence = _truncate(raw, 200)
        clauses = _split_clauses(raw)
        saw_question = False
        processed = False
        for clause in clauses:
            if _has_any(clause, _QUESTION_TOKENS):
                # 疑问句既不产生证据，也不产生排除（引号里的词同样不算）。
                saw_question = True
                continue
            processed = True
            self._parse_place_mentions(clause, evidence, pending, parsed)
            self._parse_phrase_mentions(clause, evidence, parsed)
            self._parse_signage_mentions(clause, evidence, parsed)
        if _has_any(raw, _SELF_CORRECTION_TOKENS):
            # "刚才说错了，其实是X"：同批新线索覆盖对应维度的旧用户线索。
            for dimension in {clue["dimension"] for clue in parsed.clues}:
                parsed.add_replace_dimension(dimension)
        parsed.question = saw_question and not processed
        self._apply_pending_context(raw, parsed, pending)
        return parsed

    def _parse_place_mentions(self, clause: str, evidence: str,
                              pending: Optional[PendingQuestion],
                              parsed: ParsedFeedback) -> None:
        for kind, hint, pos, literal in self._place_matches(clause):
            guards = _PLACE_SUFFIX_GUARDS if kind == "country" else _CITY_SUFFIX_GUARDS
            if self._is_location_reject(clause, pos, literal, guards):
                if kind == "country":
                    # 只有显式否定国家才是硬排除；否定城市不等于否定国家。
                    parsed.add_reject(hint)
                    # 同时清掉此前同国家的用户地名线索（城市字面量不动）。
                    parsed.add_removal("place_name", hint)
                continue
            segment = _segment_before(clause, pos)
            if _has_any(segment, _UNCERTAIN_TOKENS) or _is_negated(segment):
                continue
            if not self._is_place_evidence(clause, pos, literal, pending, guards):
                continue
            if kind == "country":
                value, extra = hint, {"place_kind": "country"}
            else:
                # 城市/地名保留用户字面写法，交给带目录校验的 city resolver。
                value, extra = literal, {"place_kind": "city", "country_hint": hint}
            if kind == "country" and _has_any(clause, _CORRECTION_TOKENS):
                parsed.add_clear_exclusion(hint)
            parsed.add_clue("place_name", value, DIMENSION_CONFIDENCE["place_name"],
                            evidence, literal, parser="rules", extra=extra)

    def _parse_phrase_mentions(self, clause: str, evidence: str,
                               parsed: ParsedFeedback) -> None:
        for pos, dimension, value, phrase in self._phrase_matches(clause):
            segment = _segment_before(clause, pos)
            if _has_any(segment, _UNCERTAIN_TOKENS):
                # 不确定：既不新增证据，也不删除旧证据。
                continue
            if _is_negated(segment):
                # 明确否定该取值：修正时可移除此前同值的用户线索，但不新增。
                parsed.add_removal(dimension, value)
                continue
            if _has_any(segment, _NON_EVIDENCE_TOKENS):
                continue
            parsed.add_clue(dimension, value, DIMENSION_CONFIDENCE.get(dimension, 0.7),
                            evidence, phrase, parser="rules")

    def _place_matches(self, clause: str) -> List[Tuple[str, str, int, str]]:
        """Return ``(kind, hint, position, literal)`` matches.

        Country matches carry a validated ISO hint; city matches stay literal
        (keeping a trailing 市/区/县/都) with only a validated country hint in
        metadata.
        """
        pattern, lookup = _place_alias_matcher()
        matches: List[Tuple[str, str, int, str]] = []
        for match in pattern.finditer(clause):
            entry = lookup.get(match.group(0).lower())
            if not entry:
                continue
            kind, hint = entry
            end = match.end()
            if kind == "city":
                while end < len(clause) and end - match.end() < 2 and clause[end] in "市区县镇都":
                    end += 1
            matches.append((kind, hint, match.start(), clause[match.start():end]))
        return matches

    def _parse_signage_mentions(self, clause: str, evidence: str,
                                parsed: ParsedFeedback) -> None:
        """Literal signage text the user explicitly quotes/reports.

        Only clauses that mention signage (写着/招牌/路牌/...) are considered
        and the value is the user's literal text; nothing is inferred.
        """
        if not _has_any(clause, _SIGNAGE_MARKERS):
            return
        existing = {_normalize(clue.get("value")) for clue in parsed.clues}
        existing |= {_normalize((clue.get("extra") or {}).get("literal")) for clue in parsed.clues}
        candidates: List[Tuple[int, str]] = []
        for match in _QUOTE_PATTERN.finditer(clause):
            candidates.append((match.start("text"), match.group("text").strip()))
        if not candidates:
            for match in _INLINE_SIGNAGE_RE.finditer(clause):
                candidates.append((match.start("text"), match.group("text").strip()))
        for pos, literal in candidates:
            if not (2 <= len(literal) <= 40):
                continue
            normalized = _normalize(literal)
            if not normalized or normalized in existing:
                continue
            segment = _segment_before(clause, pos)
            if _has_any(segment, _UNCERTAIN_TOKENS) or _is_negated(segment):
                continue
            if any(phrase in literal for phrase, _dim, _value in self._phrase_index):
                # 该文字本身已经是明确的维度线索（如"靠左行驶"），避免重复。
                continue
            parsed.add_clue("signage", literal, 0.8, evidence, literal, parser="rules")
            existing.add(normalized)

    def _phrase_matches(self, clause: str) -> List[Tuple[int, str, str, str]]:
        raw_matches: List[Tuple[int, str, str, str]] = []
        for phrase, dimension, value in self._phrase_index:
            start = 0
            while True:
                pos = clause.find(phrase, start)
                if pos < 0:
                    break
                raw_matches.append((pos, dimension, value, phrase))
                start = pos + len(phrase)
        raw_matches.sort(key=lambda item: (item[0], -len(item[3])))
        accepted: List[Tuple[int, str, str, str]] = []
        occupied: List[Tuple[int, int]] = []
        for pos, dimension, value, phrase in raw_matches:
            end = pos + len(phrase)
            if any(pos < taken_end and end > taken_start for taken_start, taken_end in occupied):
                continue
            occupied.append((pos, end))
            accepted.append((pos, dimension, value, phrase))
        accepted.sort(key=lambda item: item[0])
        return accepted

    def _is_location_reject(self, clause: str, pos: int, literal: str,
                            guards: Sequence[str] = _PLACE_SUFFIX_GUARDS) -> bool:
        """Explicit rejection only ("不是日本" / "排除日本").

        Incidental mentions ("日本品牌", "去过日本") and hedged rejections
        ("可能不是日本") never become hard exclusions.
        """
        end = pos + len(literal)
        if any(clause.startswith(guard, end) for guard in guards):
            return False
        segment = _segment_before(clause, pos)
        if _has_any(segment, _UNCERTAIN_TOKENS):
            return False
        if len(segment) > 12:
            return False
        return _has_any(segment, _REJECT_TOKENS)

    def _is_place_evidence(self, clause: str, pos: int, literal: str,
                           pending: Optional[PendingQuestion],
                           guards: Sequence[str] = _PLACE_SUFFIX_GUARDS) -> bool:
        """A place is evidence only with shooting/scene context, a direct
        "是/在/位于" statement, or as a short answer to the place question."""
        end = pos + len(literal)
        if any(clause.startswith(guard, end) for guard in guards):
            return False
        if _has_any(clause, _PLACE_CONTEXT_TOKENS):
            return True
        window = clause[max(0, pos - 3):pos]
        if _has_any(window, ("是", "在", "就在", "就是", "来自", "位于", "属于")):
            return True
        if pending is not None and "place_name" in pending.dimensions:
            stripped = clause.strip()
            if len(stripped) <= 12:
                return True
        return False

    def _rejects_iso(self, text: str, iso: str) -> bool:
        for kind, hint, pos, literal in self._place_matches(text):
            if kind != "country" or hint != iso:
                continue
            if self._is_location_reject(text, pos, literal, _PLACE_SUFFIX_GUARDS):
                return True
        return False

    # ---- dialogue context (answers to the previous question) ----
    def _apply_pending_context(self, text: str, parsed: ParsedFeedback,
                               pending: Optional[PendingQuestion]) -> None:
        if pending is None or not text:
            return
        if _has_any(text, _QUESTION_TOKENS):
            # A question is never an answer.
            return
        if pending.kind == "confirm" and pending.iso:
            normalized = _normalize(text)
            if normalized in _AFFIRMATIVE_REPLIES and not _has_any(text, _NEGATION_TOKENS):
                parsed.confirm_iso = pending.iso
                return
            if normalized in _NEGATIVE_REPLIES or self._rejects_iso(text, pending.iso):
                parsed.add_reject(pending.iso)
                return
        if not pending.dimensions:
            return
        dimensions = {d for d in pending.dimensions if isinstance(d, str)}
        for option in pending.options:
            dimension = option.get("dimension")
            value = option.get("value")
            if dimension not in dimensions or not isinstance(value, str) or not value:
                continue
            found = _find_anchor(text, (option.get("label"), value_label(value), value))
            if found is None:
                continue
            segment = _segment_before(text, found[0])
            if _has_any(segment, _UNCERTAIN_TOKENS) or _is_negated(segment):
                continue
            parsed.add_clue(dimension, value, 0.85, _truncate(text, 200), found[1], parser="rules")
        for dimension in sorted(dimensions):
            for phrase, value in self.CONTEXT_PHRASES.get(dimension, ()):
                pos = text.find(phrase)
                if pos < 0:
                    continue
                segment = _segment_before(text, pos)
                if _has_any(segment, _UNCERTAIN_TOKENS) or _is_negated(segment):
                    continue
                parsed.add_clue(dimension, value, 0.8, _truncate(text, 200), phrase, parser="rules")

    # ---- optional remote (DeepSeek) extraction validation ----
    def remote_catalog(self) -> Dict[str, Any]:
        values: Dict[str, List[str]] = {}
        for dimension in REMOTE_DIMENSIONS:
            if dimension in {"place_name", "signage"}:
                continue
            values[dimension] = sorted(self.supported_values(dimension))[:40]
        countries = {
            country["iso"]: country.get("name_zh") or country.get("name") or country["iso"]
            for country in kg.load_countries()
            if country.get("features")
        }
        return {"dimensions": list(REMOTE_DIMENSIONS), "values": values, "countries": countries}

    def validate_remote(self, payload: Any, text: str,
                        pending: Optional[PendingQuestion] = None) -> ParsedFeedback:
        """Strictly validate a remote extraction payload.

        Only entries whose quote is a literal substring of the user text that
        passes the same question/negation/uncertainty checks survive, only
        catalog values (validated country ISO for place_name) are accepted, and
        a remote payload can never confirm a country.
        """
        result = ParsedFeedback()
        if not isinstance(payload, dict) or not isinstance(text, str) or not text.strip():
            return result
        intent = payload.get("intent")
        if intent not in self._REMOTE_INTENTS or intent in {"question", "uncertain"}:
            return result
        entries = payload.get("clues")
        if isinstance(entries, list):
            for entry in entries[:6]:
                clue = self._validate_remote_clue(entry, text)
                if clue:
                    result.add_clue(clue["dimension"], clue["value"], self.REMOTE_CONFIDENCE,
                                    _truncate(text, 200), clue["literal"], parser="deepseek")
        rejects = payload.get("reject")
        if isinstance(rejects, list):
            for entry in rejects[:4]:
                iso = self._validate_remote_reject(entry, text)
                if iso:
                    result.add_reject(iso)
        return result

    def _validate_remote_clue(self, entry: Any, text: str) -> Optional[Dict[str, str]]:
        if not isinstance(entry, dict):
            return None
        dimension = entry.get("dimension")
        raw_value = entry.get("value")
        quote = entry.get("quote")
        if not all(isinstance(item, str) and item.strip()
                   for item in (dimension, raw_value, quote)):
            return None
        dimension = dimension.strip()
        raw_value = raw_value.strip()
        quote = quote.strip()
        if dimension not in REMOTE_DIMENSIONS or len(raw_value) > 40 or len(quote) > 60:
            return None
        if _has_any(quote, _QUESTION_TOKENS):
            return None
        pos = text.find(quote)
        if pos < 0:
            return None
        if _has_any(_clause_containing(text, pos), _QUESTION_TOKENS):
            return None
        segment = _segment_before(text, pos)
        if _has_any(segment, _UNCERTAIN_TOKENS) or _is_negated(segment):
            return None
        if dimension == "signage":
            if _normalize(raw_value) and _normalize(raw_value) in _normalize(text):
                return {"dimension": dimension, "value": raw_value, "literal": quote}
            return None
        if dimension == "place_name":
            iso = self._resolve_iso(raw_value)
            if iso and self._anchor_supported(dimension, iso, quote):
                return {"dimension": dimension, "value": iso, "literal": quote}
            literal = raw_value.strip()
            if self._literal_place_supported(literal, quote, text):
                return {"dimension": dimension, "value": literal, "literal": quote}
            return None
        value = self._resolve_value(dimension, raw_value)
        if value and self._anchor_supported(dimension, value, quote):
            return {"dimension": dimension, "value": value, "literal": quote}
        return None

    def _validate_remote_reject(self, entry: Any, text: str) -> Optional[str]:
        if not isinstance(entry, dict):
            return None
        iso = self._resolve_iso(entry.get("iso"))
        quote = entry.get("quote")
        if not iso or not isinstance(quote, str) or not quote.strip():
            return None
        quote = quote.strip()
        if len(quote) > 60 or text.find(quote) < 0:
            return None
        if _has_any(_clause_containing(text, text.find(quote)), _QUESTION_TOKENS):
            return None
        for kind, hint, pos, literal in self._place_matches(quote):
            if kind == "country" and hint == iso and self._is_location_reject(quote, pos, literal):
                return iso
        return None

    # ---- catalog helpers ----
    def supported_values(self, dimension: str) -> Set[str]:
        values = set(self.LOCAL_VOCAB.get(dimension, ()))
        if dimension == "language":
            for country in kg.load_countries():
                values.update(token for token in country.get("languages_spoken") or []
                              if isinstance(token, str) and token)
        elif dimension == "driving_side":
            for country in kg.load_countries():
                side = country.get("driving_side")
                if isinstance(side, str) and side:
                    values.add(side)
        else:
            values.update(kg.feature_values(dimension))
        return values

    def _resolve_value(self, dimension: str, raw_value: str) -> Optional[str]:
        supported = self.supported_values(dimension)
        if not supported:
            return None
        token = raw_value.strip()
        normalized = re.sub(r"[\s\-]+", "_", token).lower()
        if normalized in supported:
            return normalized
        lowered = token.lower()
        for candidate in sorted(supported):
            if value_label(candidate).lower() == lowered:
                return candidate
        for phrase, value in self.PHRASES.get(dimension, ()):
            if phrase and phrase in token and value in supported:
                return value
        return None

    def _resolve_iso(self, raw: Any) -> Optional[str]:
        if not isinstance(raw, str) or not raw.strip():
            return None
        token = raw.strip()
        upper = token.upper()
        if kg.get_by_iso(upper):
            return upper
        lowered = token.lower()
        for country in kg.load_countries():
            iso3 = country.get("iso3")
            if isinstance(iso3, str) and iso3.lower() == lowered:
                return country["iso"]
            for key in (country.get("name_zh"), country.get("name")):
                if isinstance(key, str) and key.lower() == lowered:
                    return country["iso"]
        alias = _EXTRA_PLACE_ALIASES.get(token) or _CITY_ALIASES.get(token)
        if alias and kg.get_by_iso(alias):
            return alias
        return None

    def _literal_place_supported(self, literal: str, quote: str, text: str) -> bool:
        """Literal (city/signage) place text must itself be user-grounded."""
        if not (2 <= len(literal) <= 40):
            return False
        normalized = _normalize(literal)
        if not normalized or normalized not in _normalize(quote):
            return False
        aliases = set(_CITY_ALIASES) | set(_CITY_ALIASES_EN)
        if literal in aliases or literal.lower() in aliases:
            return True
        if any(alias in literal for alias in aliases):
            return True
        clause = _clause_containing(text, max(0, text.find(quote)))
        return _has_any(clause, _SIGNAGE_MARKERS)

    def _anchors(self, dimension: str, value: str) -> Set[str]:
        anchors = {str(value).lower(), str(value).replace("_", " ").lower()}
        label = value_label(value)
        if label:
            anchors.add(label.lower())
        for phrase, phrase_value in self.PHRASES.get(dimension, ()):
            if phrase_value == value:
                anchors.add(phrase.lower())
        for extra in self.EXTRA_VALUE_ANCHORS.get((dimension, value), ()):
            anchors.add(str(extra).lower())
        if dimension == "place_name":
            country = kg.get_by_iso(value.upper())
            if country:
                for key in (country.get("name_zh"), country.get("name")):
                    if isinstance(key, str) and key:
                        anchors.add(key.lower())
            for alias, alias_iso in _CITY_ALIASES.items():
                if alias_iso == value.upper():
                    anchors.add(alias.lower())
        return {anchor for anchor in anchors if anchor}

    def _anchor_supported(self, dimension: str, value: str, quote: str) -> bool:
        lowered = quote.lower()
        return any(anchor in lowered for anchor in self._anchors(dimension, value))

class ChatService:
    """对话编排：确定性 free_text 解析 + 假设-验证循环。"""

    def __init__(self, session_service):
        self.session_service = session_service
        self.engine = ReasoningEngine()
        self.hypothesis = HypothesisManager(self.engine)
        self.parser = ChineseFreeTextParser()
        self._free_text_plan: Optional[Tuple[str, ParsedFeedback]] = None

    def _clues_of(self, session) -> List[Clue]:
        return dicts_to_clues(session.clues)

    @staticmethod
    def _country_name(iso: str) -> str:
        country = kg.get_by_iso(iso)
        return country["name_zh"] if country else iso

    def _safe_messages(self, session_id: int) -> List[Dict[str, Any]]:
        try:
            messages = self.session_service.list_messages(session_id)
        except Exception:
            return []
        return list(messages or [])

    def _pending_question(self, session_id: int) -> Optional[PendingQuestion]:
        return self.parser.pending_from_history(self._safe_messages(session_id))

    @staticmethod
    def _asked_dimensions(messages: Sequence[Dict[str, Any]]) -> List[str]:
        asked: List[str] = []
        for message in messages or []:
            if not isinstance(message, dict) or message.get("role") != "ai":
                continue
            meta = message.get("meta") if isinstance(message.get("meta"), dict) else {}
            action = meta.get("action") if isinstance(meta.get("action"), dict) else {}
            payload = action.get("payload") if isinstance(action.get("payload"), dict) else {}
            for dimension in payload.get("dimensions") or []:
                if isinstance(dimension, str) and dimension not in asked:
                    asked.append(dimension)
        return asked

    def _take_free_text_plan(self, text: str) -> Optional[ParsedFeedback]:
        plan = self._free_text_plan
        self._free_text_plan = None
        if not plan:
            return None
        stored_text, parsed = plan
        return parsed if stored_text == text else None

    def _remove_exclusion(self, session_id: int, iso: str) -> None:
        remover = getattr(self.session_service, "remove_exclusion", None)
        if callable(remover):
            remover(session_id, iso)

    @staticmethod
    def _call_next_action(hypothesis, scores, clues, asked):
        """Call ``next_action`` with asked-dims when the signature supports it.

        Hypothesis managers (including lightweight test doubles) may implement
        only ``next_action(scores, clues)``; passing ``asked`` unconditionally
        would raise TypeError, so it is supplied only when accepted.
        """
        try:
            parameters = inspect.signature(hypothesis.next_action).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_asked = "asked" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        if accepts_asked:
            return hypothesis.next_action(scores, clues, asked=asked)
        return hypothesis.next_action(scores, clues)

    def _remove_corrected_clues(self, session_id: int, plan: ParsedFeedback) -> None:
        """Apply the bounded correction plan before appending new evidence.

        Only *user-sourced* clues are dropped: OCR/heuristic evidence and
        unrelated dimensions are always preserved.
        """
        remover = getattr(self.session_service, "remove_user_clues", None)
        if not callable(remover):
            return
        for dimension, value in plan.remove_clue_values:
            remover(session_id, dimension=dimension, value=value)
        for dimension in plan.replace_dimensions:
            remover(session_id, dimension=dimension)

    def initial_analysis(self, session_id: int) -> Dict[str, Any]:
        """AI 首次分析后生成初始假设动作。"""
        session = self.session_service.get(session_id)
        if not session:
            return {"error": "session not found"}
        clues = self._clues_of(session)
        scores = self.engine.infer(clues, exclusions=session.exclusions)
        action = self.hypothesis.next_action(scores, clues)
        self.session_service.add_message(
            session_id, "ai", action.message,
            meta={"action": action.to_dict(), "scores": [s.to_dict() for s in scores]},
        )
        return {
            "scores": [s.to_dict() for s in scores],
            "action": action.to_dict(),
        }

    def handle_feedback(self, session_id: int, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """处理用户反馈并迭代推理。

        feedback 结构：
        {"type": "confirm"|"reject"|"answer"|"free_text", "iso"?, "dimension"?, "value"?, "text"?}
        """
        session = self.session_service.get(session_id)
        if not session:
            return {"error": "session not found"}

        ftype = feedback.get("type")
        user_msg = feedback.get("text") or ""

        if ftype == "confirm":
            iso = feedback.get("iso")
            if iso:
                self.session_service.set_confirmed(session_id, iso)
                cname = self._country_name(iso)
                self.session_service.add_message(session_id, "user", f"确认：{cname}")
                self.session_service.add_message(
                    session_id, "ai",
                    f"已记录你确认的国家为「{cname}」。"
                    "这是国家候选目录内的国家确认，不是照片 GPS 坐标。",
                    meta={"action": {"kind": "done", "iso": iso}},
                )
                return {"done": True, "iso": iso,
                        "messages": self.session_service.list_messages(session_id)}

        elif ftype == "reject":
            iso = feedback.get("iso")
            if iso:
                self.session_service.add_exclusion(session_id, iso)
                cname = self._country_name(iso)
                self.session_service.add_message(session_id, "user", f"排除：{cname}")

        elif ftype == "answer":
            dim = feedback.get("dimension")
            value = feedback.get("value")
            if isinstance(dim, str) and isinstance(value, str) and dim.strip() and value.strip():
                dim = dim.strip()
                value = value.strip()[:64]
                self.session_service.add_clue(session_id, {
                    "dimension": dim,
                    "value": value,
                    "confidence": 0.9,
                    "evidence": "用户确认",
                    "source": "user",
                    "extra": {"parser": "answer"},
                })
                self.session_service.add_message(
                    session_id, "user",
                    f"补充线索：{dimension_label(dim)} = {value_label(value)}")

        elif ftype == "free_text":
            plan = self._take_free_text_plan(user_msg)
            if plan is None:
                plan = self.parser.parse(user_msg, pending=self._pending_question(session_id))
            self.session_service.add_message(session_id, "user", user_msg)
            self._remove_corrected_clues(session_id, plan)
            for iso in plan.clear_exclusions:
                self._remove_exclusion(session_id, iso)
            for iso in plan.reject_isos:
                self.session_service.add_exclusion(session_id, iso)
            for clue in plan.clues:
                self.session_service.add_clue(session_id, clue)
            if plan.confirm_iso:
                # 用户对确认问题的明确肯定：这是用户本人的确认，不是 LLM 自动确认。
                iso = plan.confirm_iso
                cname = self._country_name(iso)
                self.session_service.set_confirmed(session_id, iso)
                self.session_service.add_message(
                    session_id, "ai",
                    f"已记录你确认的国家为「{cname}」。"
                    "这是国家候选目录内的国家确认，不是照片 GPS 坐标。",
                    meta={"action": {"kind": "done", "iso": iso}},
                )
                return {"done": True, "iso": iso,
                        "messages": self.session_service.list_messages(session_id)}

        # 重新推理
        session = self.session_service.get(session_id)  # 刷新
        clues = self._clues_of(session)
        scores = self.engine.infer(clues, exclusions=session.exclusions)
        asked = self._asked_dimensions(self._safe_messages(session_id))
        action = self._call_next_action(self.hypothesis, scores, clues, asked)

        self.session_service.add_message(
            session_id, "ai", action.message,
            meta={"action": action.to_dict(), "scores": [s.to_dict() for s in scores]},
        )
        return {
            "scores": [s.to_dict() for s in scores],
            "action": action.to_dict(),
        }

    def reason_only(self, session_id: int,
                    new_clues: Optional[List[Dict[str, Any]]] = None,
                    click_pos: Optional[tuple] = None) -> Dict[str, Any]:
        """仅根据当前线索重新推理（用于局部点击分析后更新评分）。

        new_clues: 本次局部分析新发现的线索，用于在对话中给出分析依据。
        click_pos: (x_ratio, y_ratio) 点击位置，用于消息中说明。
        """
        session = self.session_service.get(session_id)
        if not session:
            return {"error": "session not found"}
        clues = self._clues_of(session)
        scores = self.engine.infer(clues, exclusions=session.exclusions)
        asked = self._asked_dimensions(self._safe_messages(session_id))
        action = self._call_next_action(self.hypothesis, scores, clues, asked)

        # 构造分析依据消息
        pos_text = ""
        if click_pos:
            pos_text = f"你点击了图片 {click_pos[0]*100:.0f}%, {click_pos[1]*100:.0f}% 的位置，"

        basis_lines = []
        if new_clues:
            basis_lines.append("该区域分析出以下线索：")
            for c in new_clues:
                dim_zh = dimension_label(c.get("dimension"))
                val_zh = value_label(c.get("value"))
                conf = c.get("confidence", 0)
                basis_lines.append(f"• {dim_zh}：{val_zh}（置信度 {conf*100:.0f}%）")
        if scores:
            top = scores[0]
            basis_lines.append(f"\n基于以上线索，当前最可能的国家是「{top.name_zh}」。")

        msg = f"{pos_text}已完成局部区域分析。" + "\n".join(basis_lines) + f"\n{action.message}"

        self.session_service.add_message(
            session_id, "ai", msg,
            meta={"action": action.to_dict(), "scores": [s.to_dict() for s in scores]},
        )
        return {
            "scores": [s.to_dict() for s in scores],
            "action": action.to_dict(),
        }

    # ---- 向后兼容：旧的简单关键词接口 ----
    def _parse_free_text(self, text: str) -> List[Dict[str, Any]]:
        return self.parser.parse(text).clues
