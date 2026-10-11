"""从可靠 OCR 线索中抽取语义线索（目录校验证据 / 纯观察两级）。

调用契约（OCR / pipeline worker 集成用，保持稳定、纯函数式）::

    from .semantic import SemanticClueAnalyzer

    analyzer = SemanticClueAnalyzer()      # 只读国家目录，无网络、无图片访问
    derived = analyzer.analyze(all_clues)  # 不修改入参，可重复调用
    all_clues.extend(derived)

- 构造：``SemanticClueAnalyzer(countries=None, min_confidence=0.3)``。
  ``countries`` 只用于离线测试注入；默认读取缓存的 ``data/countries.json``。
- 输入：任意 ``Clue`` 列表。只消费 ``dimension == "signage"`` 且
  ``extra.reliable is True``（或 source 为 ``ocr``/``point_click``）
  且置信度不低于 ``min_confidence`` 的 OCR 文本行；其余线索一律忽略。
- 输出：仅派生线索（不复制入参）。幂等：对输出再次调用不会产生新线索。
- 确定性：输出按 (dimension, value) 去重后按 DIMENSIONS 顺序排序；同一
  (dimension, value) 的多条 OCR 行只保留最高置信度并记录 ``ocr_occurrences``，
  绝不叠加置信度。

证据分级：
- ``place_name``：目录 ``name``/``name_zh`` 规范化完整边界匹配，或已校验的
  城市名（含有限中文别名；北京市/東京駅允许一个行政后缀）。跨国家同名
  城市 fail closed，只有唯一 ISO 或显式目录别名才提供国家；裸 3 字母大写
  词元（OPEN/AND/CLOSED 等）不再推断国家。
  ``validated=True``、``affects_ranking=True``，并携带 ``origin_group``
  以便引擎对同一 OCR 行的相关证据只计最强的一条。
- ``culture``：精确命中目录 ``features.culture`` 且在目录中只属于 1 个国家。
  ``validated=True``、``affects_ranking=True``。
- ``driving_side``：仅显式短语（drive on the left / left-hand traffic 等），
  取值校验为 left/right。``validated=True``、``affects_ranking=True``。
- ``road`` / ``terrain`` / ``waterbody`` / ``infrastructure``：只有 OCR 词汇
  证据、没有目录元数据。``display_only=True``、``affects_ranking=False``、
  ``reliable=False``，只用于线索卡展示，绝不参与排序。

保守性：单字 CJK 词条不收；否定语境（not/no/不是/并非…）跳过；英文词做
alnum 完整边界 + 否定窗口 + 专名延续防护（New Mexico / Northern Ireland /
Little Italy 不再命中 Mexico/Ireland/Italy）；常见同形英文地名
（china/turkey/chad/jordan/georgia）只在中文名命中时才输出，避免把普通词
当成国家。road/terrain/waterbody/infrastructure 一律 `display_only`，
`affects_ranking=False`，只展示不排序。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from config import BASE_DIR
from ..clue import Clue, DIMENSIONS
from .. import knowledge_graph as kg
from ..engine import (
    CITY_NAME_ALIASES,
    build_city_name_index,
    find_exact_phrase,
    find_place_phrase,
    load_city_name_index,
    normalize_place_text,
    resolve_city_name,
)


MIN_SEMANTIC_CONFIDENCE = 0.3
_OCR_SOURCES = frozenset({"ocr", "point_click"})

# 置信度上限：派生线索永远不高于原始 OCR 行
_CONFIDENCE_CAPS = {
    "place_name": 0.95,
    "culture": 0.85,
    "driving_side": 0.8,
    "road": 0.6,
    "terrain": 0.55,
    "waterbody": 0.55,
    "infrastructure": 0.6,
}

# 常见英文同形词：只有 ISO3（大写词元）或中文名命中才允许输出国家
_AMBIGUOUS_ENGLISH_PLACE_NAMES = frozenset({
    "china", "turkey", "chad", "jordan", "georgia",
})

# 观察词表：canonical token -> 需要精确匹配的规范化词条（含 CJK 多字词）
OBSERVATION_TOKENS: Dict[str, Tuple[str, ...]] = {
    "road": (
        "street", "road", "avenue", "boulevard", "highway", "freeway",
        "motorway", "route", "alley", "lane", "square", "plaza",
        "大道", "公路", "街道", "高速公路", "快速路", "环路",
    ),
    "terrain": (
        "mountain", "hill", "valley", "coast", "island", "glacier",
        "volcano", "canyon", "plateau", "cliff",
        "山脉", "山谷", "火山", "冰川", "峡谷", "高原", "岛屿", "海岸",
    ),
    "waterbody": (
        "lake", "river", "sea", "ocean", "bay", "harbor", "harbour",
        "beach", "waterfall", "reservoir", "strait", "fjord",
        "湖泊", "河流", "海湾", "海滩", "瀑布", "水库", "海峡",
    ),
    "infrastructure": (
        "station", "airport", "port", "bridge", "tunnel", "dam", "metro",
        "subway", "stadium", "terminal", "lighthouse", "observatory",
        "university", "hospital", "museum",
        "车站", "机场", "港口", "大桥", "隧道", "水坝", "地铁",
        "体育场", "博物馆", "大学", "医院", "灯塔",
    ),
}

_DRIVING_SIDE_PHRASES: Tuple[Tuple[str, str], ...] = (
    ("drive on the left", "left"),
    ("driving on the left", "left"),
    ("left hand traffic", "left"),
    ("left-hand traffic", "left"),
    ("keep left", "left"),
    ("drive on the right", "right"),
    ("driving on the right", "right"),
    ("right hand traffic", "right"),
    ("right-hand traffic", "right"),
    ("keep right", "right"),
)


def _is_reliable_signage(clue: Clue, min_confidence: float) -> bool:
    if clue.dimension != "signage" or not isinstance(clue.value, str):
        return False
    if not clue.value.strip() or clue.confidence < min_confidence:
        return False
    extra = clue.extra if isinstance(clue.extra, dict) else {}
    if extra.get("reliable") is True:
        return True
    return clue.source in _OCR_SOURCES and extra.get("reliable") is not False


def _matched_texts(*values: str) -> List[str]:
    seen: List[str] = []
    for value in values:
        text = " ".join(str(value).split())
        if text and text not in seen:
            seen.append(text[:60])
    return seen


class SemanticClueAnalyzer:
    """Deterministic OCR-text -> semantic clue derivation (no image access)."""

    def __init__(self, countries: Optional[List[Dict[str, Any]]] = None, *,
                 cities: Optional[List[Dict[str, Any]]] = None,
                 min_confidence: float = MIN_SEMANTIC_CONFIDENCE) -> None:
        self.countries = list(countries) if countries is not None else kg.load_countries()
        self.min_confidence = float(min_confidence)
        # 城市目录懒加载：只有出现可靠 OCR 文本时才读取 data/cities.json。
        self._cities_arg = list(cities) if cities is not None else None
        self._city_index_cache = None
        self._place_phrases: List[Tuple[str, str, str]] = self._build_place_phrases()
        self._culture_phrases: List[Tuple[str, str, str]] = self._build_culture_phrases()
        self._driving_phrases: List[Tuple[str, str]] = [
            (normalize_place_text(phrase), side) for phrase, side in _DRIVING_SIDE_PHRASES
            if normalize_place_text(phrase)
        ]
        self._observation_phrases: Dict[str, List[Tuple[str, str]]] = {
            dimension: sorted(
                ((normalize_place_text(token), token) for token in tokens),
                key=lambda item: (-len(item[0]), item[0]),
            )
            for dimension, tokens in OBSERVATION_TOKENS.items()
        }

    # ---- index construction -------------------------------------------------
    def _build_place_phrases(self) -> List[Tuple[str, str, str]]:
        buckets: Dict[str, set] = {}
        for country in self.countries:
            iso = country.get("iso")
            if not isinstance(iso, str) or not iso:
                continue
            candidates = ((country.get("name"), "name"), (country.get("name_zh"), "name_zh"))
            for candidate, kind in candidates:
                key = normalize_place_text(candidate)
                if key:
                    buckets.setdefault(key, set()).add((iso, kind))
        phrases: List[Tuple[str, str, str]] = []
        for phrase, entries in buckets.items():
            isos = {iso for iso, _kind in entries}
            if len(isos) != 1:
                continue  # ambiguous catalog name -> never derive
            if phrase in _AMBIGUOUS_ENGLISH_PLACE_NAMES:
                continue  # 同形英文词只允许 ISO3/中文名命中
            iso = next(iter(isos))
            kinds = sorted({kind for entry_iso, kind in entries if entry_iso == iso})
            phrases.append((phrase, iso, kinds[0] if len(kinds) == 1 else "name"))
        phrases.sort(key=lambda item: (-len(item[0]), item[0]))
        return phrases


    def _build_culture_phrases(self) -> List[Tuple[str, str, str]]:
        buckets: Dict[str, set] = {}
        for country in self.countries:
            iso = country.get("iso")
            if not isinstance(iso, str) or not iso:
                continue
            for value in (country.get("features", {}) or {}).get("culture", []) or []:
                if isinstance(value, str) and value:
                    key = normalize_place_text(value.replace("_", " "))
                    if key:
                        buckets.setdefault(key, set()).add(iso)
        # 只有唯一指向 1 个国家的文化词条才算“可验证的地标证据”
        phrases = [
            (key, _canonical_value(key, self.countries, "culture"), next(iter(isos)))
            for key, isos in buckets.items() if len(isos) == 1
        ]
        phrases.sort(key=lambda item: (-len(item[0]), item[0]))
        return phrases

    # ---- analysis -----------------------------------------------------------
    def analyze(self=None, clues: Optional[Sequence[Clue]] = None) -> List[Clue]:
        """Derive semantic clues; usable as instance method *and* class call.

        ``analyzer.analyze(all_clues)`` is the preferred form. The OCR worker's
        current ``SemanticClueAnalyzer.analyze(list(clues))`` class-call form is
        also supported (it builds a default analyzer instead of failing
        silently), and ``SemanticClueAnalyzer.analyze(clues=...)`` works too.
        """
        if not isinstance(self, SemanticClueAnalyzer):
            payload = clues if clues is not None else self
            if payload is None:
                return []
            return SemanticClueAnalyzer().analyze(payload)
        collected: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for clue in clues or ():
            if not isinstance(clue, Clue) or not _is_reliable_signage(clue, self.min_confidence):
                continue
            text = str(clue.value)
            norm = normalize_place_text(text)
            if not norm:
                continue
            self._collect_places(text, norm, clue, collected)
            self._collect_culture(norm, clue, collected)
            self._collect_driving_side(norm, clue, collected)
            self._collect_observations(norm, clue, collected)
        order = {dimension: index for index, dimension in enumerate(DIMENSIONS)}
        return [
            self._to_clue(dimension, value, entry)
            for (dimension, value), entry in sorted(
                collected.items(), key=lambda item: (order.get(item[0][0], 99), item[0][1])
            )
        ]

    def _collect_places(self, raw_text: str, norm: str, clue: Clue,
                        collected: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
        for phrase, iso, kind in self._place_phrases:
            if find_place_phrase(norm, phrase):
                self._add(collected, "place_name", iso, clue,
                          matched_text=self._matched_for(norm, phrase, raw_text),
                          validated=True, affects_ranking=True,
                          catalog_iso=iso, match_kind=kind)
                break
        # 不再从裸 3 字母大写词元推断国家（OPEN/AND/CLOSED、CAN/EXIT 等
        # 同形词误报率过高）；ISO3 只在用户显式回答等输入路径中使用。
        self._collect_city_places(norm, clue, collected)

    def _city_index(self):
        if self._city_index_cache is None:
            try:
                if self._cities_arg is not None:
                    self._city_index_cache = build_city_name_index(self._cities_arg)
                else:
                    self._city_index_cache = load_city_name_index(BASE_DIR / "data" / "cities.json")
            except (OSError, ValueError, TypeError):
                self._city_index_cache = {}
        return self._city_index_cache

    def _collect_city_places(self, norm: str, clue: Clue,
                             collected: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
        """Bounded city evidence: curated CJK aliases + exact English n-grams.

        Every hit is re-resolved against data/cities.json and fails closed on
        multi-ISO same-name cities; only unique-ISO names or explicit aliases
        produce a clue, and no coordinate or row is ever invented here.
        """
        index = self._city_index()
        if not index:
            return
        seen_iso: set = set()
        for alias in sorted(CITY_NAME_ALIASES):
            alias_norm = normalize_place_text(alias)
            if not alias_norm or alias in seen_iso:
                continue
            if not find_place_phrase(norm, alias_norm):
                continue
            resolution = resolve_city_name(index, alias)
            if resolution is None:
                continue
            seen_iso.add(alias)
            self._add(collected, "place_name", resolution.iso, clue,
                      matched_text=alias, validated=True, affects_ranking=True,
                      catalog_iso=resolution.iso, match_kind="city_alias",
                      city_name=resolution.city_name, city_id=resolution.city_id,
                      confidence_factor=resolution.confidence_factor)
        tokens = norm.split()
        for size in (1, 2, 3):
            for start in range(len(tokens) - size + 1):
                phrase = " ".join(tokens[start:start + size])
                if phrase in seen_iso:
                    continue
                resolution = resolve_city_name(index, phrase)
                if resolution is None or resolution.resolution == "city_unique_country":
                    continue
                if not find_place_phrase(norm, phrase):
                    continue
                seen_iso.add(phrase)
                self._add(collected, "place_name", resolution.iso, clue,
                          matched_text=phrase, validated=True, affects_ranking=True,
                          catalog_iso=resolution.iso, match_kind="city_name",
                          city_name=resolution.city_name, city_id=resolution.city_id,
                          confidence_factor=resolution.confidence_factor)

    def _collect_culture(self, norm: str, clue: Clue,
                         collected: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
        for phrase, value, iso in self._culture_phrases:
            if find_place_phrase(norm, phrase):
                self._add(collected, "culture", value, clue,
                          matched_text=self._matched_for(norm, phrase, str(clue.value)),
                          validated=True, affects_ranking=True,
                          catalog_iso=iso, match_kind="culture")
                break

    def _collect_driving_side(self, norm: str, clue: Clue,
                              collected: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
        for phrase, side in self._driving_phrases:
            if find_exact_phrase(norm, phrase):
                self._add(collected, "driving_side", side, clue,
                          matched_text=self._matched_for(norm, phrase, str(clue.value)),
                          validated=True, affects_ranking=True,
                          match_kind="driving_side")
                break

    def _collect_observations(self, norm: str, clue: Clue,
                              collected: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
        for dimension, phrases in self._observation_phrases.items():
            for phrase, token in phrases:
                if find_exact_phrase(norm, phrase):
                    self._add(collected, dimension, token, clue,
                              matched_text=self._matched_for(norm, phrase, str(clue.value)),
                              validated=False, affects_ranking=False)
                    break

    # ---- helpers ------------------------------------------------------------
    @staticmethod
    def _matched_for(norm: str, phrase: str, raw: str) -> str:
        index = norm.find(phrase)
        if index < 0:
            return phrase
        # 用规范化位置在原文里做近似定位：仅用于展示，失败时退回词条本身。
        words = norm[:index].split()
        return " ".join(words[-2:] + [phrase]) if words else phrase

    @staticmethod
    def _add(collected: Dict[Tuple[str, str], Dict[str, Any]], dimension: str,
             value: str, clue: Clue, *, matched_text: str,
             validated: bool, affects_ranking: bool,
             catalog_iso: Optional[str] = None,
             match_kind: Optional[str] = None,
             city_name: Optional[str] = None,
             city_id: Optional[str] = None,
             confidence_factor: Optional[float] = None) -> None:
        key = (dimension, value)
        confidence = min(float(clue.confidence), _CONFIDENCE_CAPS.get(dimension, 0.6))
        entry = collected.get(key)
        if entry is None:
            entry = {
                "confidence": confidence,
                "best_origin": float(clue.confidence),
                "evidence": clue.evidence or f"OCR 文本证据：{matched_text}",
                "matched_texts": _matched_texts(matched_text),
                "occurrences": 0,
                "validated": validated,
                "affects_ranking": affects_ranking,
                "catalog_iso": catalog_iso,
                "match": match_kind,
                "city_name": city_name,
                "city_id": city_id,
                "confidence_factor": confidence_factor,
                "origin_group": origin_group_for_clue(clue),
            }
            collected[key] = entry
        else:
            if entry.get("catalog_iso") is None and catalog_iso is not None:
                entry["catalog_iso"] = catalog_iso
            if entry.get("match") is None and match_kind is not None:
                entry["match"] = match_kind
            if entry.get("origin_group") is None:
                entry["origin_group"] = origin_group_for_clue(clue)
            if entry.get("city_name") is None and city_name is not None:
                entry["city_name"] = city_name
            if entry.get("city_id") is None and city_id is not None:
                entry["city_id"] = city_id
            if confidence_factor is not None:
                current_factor = entry.get("confidence_factor")
                entry["confidence_factor"] = min(
                    float(current_factor) if current_factor is not None else 1.0,
                    float(confidence_factor),
                )
        entry["occurrences"] += 1
        if confidence > entry["confidence"]:
            entry["confidence"] = confidence
            entry["evidence"] = clue.evidence or f"OCR 文本证据：{matched_text}"
        entry["best_origin"] = max(entry["best_origin"], float(clue.confidence))
        for text in _matched_texts(matched_text):
            if text not in entry["matched_texts"] and len(entry["matched_texts"]) < 3:
                entry["matched_texts"].append(text)

    @staticmethod
    def _to_clue(dimension: str, value: str, entry: Dict[str, Any]) -> Clue:
        validated = bool(entry["validated"])
        extra: Dict[str, Any] = {
            "origin_source": "ocr",
            "origin_confidence": round(float(entry["best_origin"]), 3),
            "matched_text": entry["matched_texts"][0],
            "matched_texts": list(entry["matched_texts"]),
            "ocr_occurrences": int(entry["occurrences"]),
            "validated": validated,
            "reliable": validated,
            "affects_ranking": bool(entry["affects_ranking"]),
        }
        if entry.get("catalog_iso"):
            # Validated catalog provenance: engine/city layers can scope the
            # evidence to the exact ISO it was checked against.
            extra["catalog_iso"] = str(entry["catalog_iso"])
        if entry.get("match"):
            extra["match"] = str(entry["match"])
        if entry.get("city_name"):
            extra["city_name"] = str(entry["city_name"])
        if entry.get("city_id"):
            extra["city_id"] = str(entry["city_id"])
        if entry.get("confidence_factor") is not None:
            extra["confidence_factor"] = round(float(entry["confidence_factor"]), 3)
        if entry.get("origin_group"):
            # Stable identity of the OCR source line: used by the engine to
            # avoid counting correlated evidence from one line twice.
            extra["origin_group"] = str(entry["origin_group"])
        if not validated:
            extra["display_only"] = True
        return Clue(
            dimension=dimension,
            value=value,
            confidence=round(float(entry["confidence"]), 3),
            evidence=str(entry["evidence"]),
            source="semantic",
            extra=extra,
        )


def origin_group_for_clue(clue: Clue) -> str:
    """Stable OCR-line identity: normalized text plus line start when known.

    Public helper: pipeline tags per-line derived language clues with the same
    identity so the engine counts all descendants of one OCR line once.
    """
    extra = clue.extra if isinstance(clue.extra, dict) else {}
    key = normalize_place_text(clue.value)[:80]
    quad = extra.get("quad") or extra.get("bbox")
    if (isinstance(quad, (list, tuple)) and quad
            and isinstance(quad[0], (list, tuple)) and len(quad[0]) >= 2):
        try:
            key += f"|{round(float(quad[0][0]), 1)},{round(float(quad[0][1]), 1)}"
        except (TypeError, ValueError):
            pass
    return "ocr:" + key


def _canonical_value(phrase: str, countries: Iterable[Dict[str, Any]], dimension: str) -> str:
    """Return the exact catalog token whose normalized form equals ``phrase``."""
    for country in countries:
        for value in (country.get("features", {}) or {}).get(dimension, []) or []:
            if isinstance(value, str) and normalize_place_text(value.replace("_", " ")) == phrase:
                return value
    return phrase.replace(" ", "_")


def derive_semantic_clues(clues: Sequence[Clue], *,
                          countries: Optional[List[Dict[str, Any]]] = None,
                          min_confidence: float = MIN_SEMANTIC_CONFIDENCE) -> List[Clue]:
    """One-shot convenience wrapper for pipeline integration."""
    return SemanticClueAnalyzer(countries, min_confidence=min_confidence).analyze(clues)


__all__ = [
    "MIN_SEMANTIC_CONFIDENCE",
    "OBSERVATION_TOKENS",
    "SemanticClueAnalyzer",
    "derive_semantic_clues",
    "origin_group_for_clue",
]