"""推理引擎：将线索映射到国家并评分排序。

核心逻辑（单一真源）：
- 每个国家的每个维度有一组可能的特征值；
- 每条线索 = (dimension, value, confidence)；
- 国家得分 = Σ 线索置信度 × 匹配权重 × 地名置信因子；
- 只有 :data:`CATALOG_DIMENSIONS` 中的维度参与评分：
    * language      -> 目录 ``languages_spoken``
    * place_name    -> 目录 ``iso``/``name``/``name_zh``/ISO3，或
                       ``data/cities.json`` 中校验过的城市名（含中文别名）
    * driving_side  -> 目录 ``driving_side`` 字段（仅 left/right）
    * culture 等    -> 目录 ``features[dimension]`` 特征值
- 观察维度（signage/road/terrain/waterbody/infrastructure）没有目录元数据，
  永不参与匹配，绝不改变排序。

地名歧义防护：
- 2 字母 ISO 只接受大小写完全一致的 ``"JP"`` 形式（避免英文常用词 in/it/us/me）；
- 国家名/中文名经规范化（大小写、变音符、标点）后只做完整匹配；
- 城市名来自 ``data/cities.json``：全局唯一 ISO → 全额置信；跨国家同名
  （Paris/York 等）一律视为歧义、保持未解析，人口不是照片国家的证据；
  只有 :data:`CITY_NAME_ALIASES` 中显式且经目录 ID 核对的别名
  （巴黎→FR、东京→JP 等）才提供国家范围；
- 当线索携带 ``extra.city_name`` 时，引擎必须用目录重新解析核实其 ISO，
  查不到或不一致就整条作废，绝不信任外部声明的城市。
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from config import BASE_DIR, MIN_CLUE_CONFIDENCE, TOP_N_HYPOTHESES
from .clue import CATALOG_DIMENSIONS, Clue
from . import knowledge_graph as kg
from .country_policy import (
    canonical_country_iso,
    country_members,
    is_excluded,
    normalize_iso,
    source_region_iso,
)


LANGUAGE_WEIGHT = 2.0  # 语言线索权重更高
PLACE_NAME_WEIGHT = 3.0  # 目录校验过的地名是最强的规则信号
DRIVING_SIDE_WEIGHT = 1.5  # 通行方向直接来自目录字段
CLIMATE_WEIGHT = 1.2
VEGETATION_WEIGHT = 1.0
BUILDING_WEIGHT = 1.0
VEHICLE_WEIGHT = 0.8

DIMENSION_WEIGHTS = {
    "language": LANGUAGE_WEIGHT,
    "place_name": PLACE_NAME_WEIGHT,
    "driving_side": DRIVING_SIDE_WEIGHT,
    "climate": CLIMATE_WEIGHT,
    "vegetation": VEGETATION_WEIGHT,
    "building": BUILDING_WEIGHT,
    "vehicle": VEHICLE_WEIGHT,
    "culture": 1.5,
    # 观察维度（signage/road/terrain/waterbody/infrastructure）刻意不在此表中：
    # 它们没有目录元数据，只能展示，不能暗示改变了国家排序。
}

# 城市名解析：跨国家同名（Paris/York 等）不靠人口猜测国家，一律 fail
# closed；只有下面显式且经目录核对的别名才提供国家范围。
CITY_NAME_ALIASES: Dict[str, Tuple[str, str]] = {
    "北京": ("Beijing", "CN"),
    "上海": ("Shanghai", "CN"),
    "广州": ("Guangzhou", "CN"),
    "深圳": ("Shenzhen", "CN"),
    "东京": ("Tokyo", "JP"),
    "東京": ("Tokyo", "JP"),
    "大阪": ("Osaka", "JP"),
    "巴黎": ("Paris", "FR"),
    "伦敦": ("London", "GB"),
}

# 拉丁专名延续词防护：New Mexico -> mexico、Northern Ireland -> ireland、
# Little Italy -> italy 均不再当作孤立国家名；连接词/介词除外。
LATIN_JOINER_STOPWORDS = frozenset({
    "in", "at", "of", "the", "to", "from", "near", "by", "on", "and", "or",
    "a", "an", "is", "this", "welcome", "visit", "city", "downtown", "made",
    "de", "la", "el",
})
# CJK 行政区/车站后缀：北京市、東京都、東京駅
ADMIN_SUFFIX_CHARS = frozenset("市駅站城区都港府道县縣国")

_ASCII_NEGATIONS = frozenset({"not", "no", "never", "without", "isnt", "non"})
_CJK_NEGATIONS = ("不是", "并非", "不在", "没有", "非", "无")
_NEGATION_WINDOW = 12


def normalize_place_text(value: object) -> str:
    """Casefold/diacritic/punctuation-insensitive canonical form for matching."""
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKD", value)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text)
    return " ".join(cleaned.split())


def _is_negated(prefix: str) -> bool:
    words = prefix.split()
    if any(word in _ASCII_NEGATIONS for word in words[-2:]):
        return True
    return any(token in prefix[-4:] for token in _CJK_NEGATIONS)


def is_negated_prefix(prefix: str) -> bool:
    """Public wrapper: True when a normalized prefix ends in a negation cue."""
    return _is_negated(prefix)


def find_exact_phrase(normalized_text: str, phrase: str) -> bool:
    """True only for an alnum-boundary phrase match outside a negation window.

    ``normalized_text`` and ``phrase`` must both come from
    :func:`normalize_place_text`; no substring, prefix or fuzzy matching.
    """
    if not normalized_text or not phrase:
        return False
    start = 0
    while True:
        index = normalized_text.find(phrase, start)
        if index < 0:
            return False
        end = index + len(phrase)
        left_ok = index == 0 or not normalized_text[index - 1].isalnum()
        right_ok = end == len(normalized_text) or not normalized_text[end].isalnum()
        if left_ok and right_ok and not _is_negated(
            normalized_text[max(0, index - _NEGATION_WINDOW):index]
        ):
            return True
        start = index + 1


def find_place_phrase(normalized_text: str, phrase: str) -> bool:
    """Boundary/negation match with place-specific ambiguity guards.

    Extends :func:`find_exact_phrase` with:
    - Latin single-word phrases must not continue a longer proper noun
      (``New Mexico``, ``Northern Ireland``, ``Little Italy`` are rejected
      unless the preceding word is a known joiner/stopword);
    - CJK phrases may be followed by one administrative/station suffix
      (``北京市``, ``東京都``, ``東京駅``) while the left boundary stays strict
      (``東京都`` never matches ``京都``).
    """
    if not normalized_text or not phrase:
        return False
    single_latin = phrase.isascii() and " " not in phrase
    start = 0
    while True:
        index = normalized_text.find(phrase, start)
        if index < 0:
            return False
        start = index + 1
        if index > 0 and normalized_text[index - 1].isalnum():
            continue  # strict left boundary always applies
        end = index + len(phrase)
        if end < len(normalized_text) and normalized_text[end].isalnum():
            if phrase.isascii() or normalized_text[end] not in ADMIN_SUFFIX_CHARS:
                continue
        if _is_negated(normalized_text[max(0, index - _NEGATION_WINDOW):index]):
            continue
        if single_latin:
            previous = normalized_text[:index].split()
            if previous and previous[-1] not in LATIN_JOINER_STOPWORDS:
                continue
        return True


def build_country_name_index(countries: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map normalized country name/中文名 -> unique ISO; ambiguous keys dropped."""
    buckets: Dict[str, set] = {}
    for country in countries:
        iso = country.get("iso")
        for candidate in (country.get("name"), country.get("name_zh")):
            key = normalize_place_text(candidate)
            if key and isinstance(iso, str) and iso:
                buckets.setdefault(key, set()).add(iso)
    return {key: next(iter(values)) for key, values in buckets.items() if len(values) == 1}


@dataclass(frozen=True)
class PlaceResolution:
    """Catalog-validated country resolution for one place value."""

    iso: str
    confidence_factor: float = 1.0
    resolution: str = "country_name"
    city_id: Optional[str] = None
    city_name: Optional[str] = None


def build_city_name_index(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Tuple[Tuple[str, int, str, str], ...]]:
    """``{normalized name: ((iso, population, city_id, name), ...)}`` by population."""
    buckets: Dict[str, List[Tuple[str, int, str, str]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name, iso, city_id = row.get("name"), row.get("iso"), row.get("city_id")
        if not isinstance(name, str) or not isinstance(iso, str) or city_id is None:
            continue
        try:
            population = int(row.get("population") or 0)
        except (TypeError, ValueError):
            population = 0
        key = normalize_place_text(name)
        if key:
            buckets.setdefault(key, []).append((iso.strip().upper(), population, str(city_id), name))
    return {
        key: tuple(sorted(values, key=lambda item: (-item[1], item[2])))
        for key, values in buckets.items()
    }


@lru_cache(maxsize=4)
def _load_city_name_index_cached(path: str, mtime_ns: int, size: int):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("cities") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return {}
    return build_city_name_index(rows)


def load_city_name_index(path: str | Path) -> Dict[str, Tuple[Tuple[str, int, str, str], ...]]:
    """Load ``data/cities.json`` name index once per (path, mtime, size)."""
    candidate = Path(path)
    stat = candidate.stat()
    return _load_city_name_index_cached(str(candidate), stat.st_mtime_ns, stat.st_size)


def resolve_city_name(index: Mapping[str, Tuple[Tuple[str, int, str, str], ...]],
                      value: object) -> Optional[PlaceResolution]:
    """Resolve a city name/alias to a catalog country, failing closed.

    Multi-ISO same-name cities (Paris, York, ...) return ``None``: population
    is never used as evidence of the photographed country. Only the bounded
    :data:`CITY_NAME_ALIASES` table provides an explicit country scope, and
    even aliases must match a real ``data/cities.json`` row for that ISO.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    rows = index.get(normalize_place_text(raw))
    if not rows:
        alias = CITY_NAME_ALIASES.get(raw)
        if not alias:
            return None
        catalog_name, alias_iso = alias
        rows = tuple(
            row for row in index.get(normalize_place_text(catalog_name), ())
            if row[0] == alias_iso
        )
        if not rows:
            return None
        if len(rows) == 1:
            row = rows[0]
            return PlaceResolution(row[0], 1.0, "city_alias", row[2], row[3])
        # Explicit alias scope, but the catalog holds several same-named rows
        # in that ISO: country evidence only, never a single city id.
        return PlaceResolution(rows[0][0], 1.0, "city_alias_country")
    isos = {row[0] for row in rows}
    if len(isos) != 1:
        # 同名城市跨越多个国家：人口/相似度都不足以指认国家，保持歧义。
        return None
    if len(rows) == 1:
        row = rows[0]
        return PlaceResolution(row[0], 1.0, "city_unique", row[2], row[3])
    # Same country, several same-named cities: country evidence only.
    return PlaceResolution(rows[0][0], 1.0, "city_unique_country")


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
    # Optional labeled-reference diagnostics for this country. ``None`` means no
    # reference index is configured and the keys are omitted entirely, so
    # consumers can tell that apart from "configured, nothing matched here".
    reference_matches: Optional[List[Dict[str, Any]]] = None
    reference_regions: Optional[List[Dict[str, Any]]] = None
    reference_cosine: Optional[float] = None
    # Present only when a policy group was formed from subordinate source
    # scopes.  The public ``iso`` remains the canonical product country.
    parent_iso: Optional[str] = None
    source_regions: Optional[List[Dict[str, Any]]] = None
    is_grouped: bool = False

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
        if self.parent_iso:
            result["parent_iso"] = self.parent_iso
        if self.source_regions is not None:
            result["source_regions"] = self.source_regions
        if self.is_grouped:
            result["is_grouped"] = True
        if self.reference_matches is not None or self.reference_regions is not None:
            result["reference_matches"] = self.reference_matches or []
            result["reference_regions"] = self.reference_regions or []
            result["reference_cosine"] = self.reference_cosine
        return result


def _source_region_record(score: CountryScore, parent_iso: str) -> Dict[str, Any]:
    """Return one source-scope record for grouped candidate metadata."""
    record: Dict[str, Any] = {
        "source_iso": score.iso,
        "parent_iso": parent_iso,
        "name": score.name,
        "name_zh": score.name_zh,
        "score": round(score.score, 4),
        "raw_cosine": score.raw_cosine,
        "score_type": score.score_type,
    }
    if score.matched_clues:
        record["matched_clues"] = score.matched_clues
    if score.fusion:
        record["fusion"] = score.fusion
    return record


def _merge_group_clues(members: List[CountryScore], parent_iso: str) -> List[Dict[str, Any]]:
    """Merge member clue diagnostics without changing their scoring semantics."""
    merged: List[Dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for member in members:
        for item in member.matched_clues:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            row.setdefault("source_iso", member.iso)
            row.setdefault("parent_iso", parent_iso)
            key = (
                row.get("source_iso"),
                row.get("dimension"),
                row.get("value"),
                row.get("evidence"),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    return merged


def _merge_group_reference(members: List[CountryScore]) -> Dict[str, Any]:
    """Merge retrieval-only reference diagnostics for a grouped candidate."""
    has_matches = any(member.reference_matches is not None for member in members)
    has_regions = any(member.reference_regions is not None for member in members)
    if not has_matches and not has_regions:
        return {}

    matches: List[Dict[str, Any]] = []
    regions: List[Dict[str, Any]] = []
    seen_matches: set[tuple[Any, ...]] = set()
    seen_regions: set[tuple[Any, ...]] = set()
    for member in members:
        for row in member.reference_matches or []:
            if not isinstance(row, dict):
                continue
            key = (row.get("reference_id"), row.get("iso"), row.get("region"), row.get("place"))
            if key in seen_matches:
                continue
            seen_matches.add(key)
            matches.append(row)
        for row in member.reference_regions or []:
            if not isinstance(row, dict):
                continue
            key = (row.get("iso"), row.get("region"), row.get("city"))
            if key in seen_regions:
                continue
            seen_regions.add(key)
            regions.append(row)

    matches.sort(
        key=lambda row: (-float(row.get("raw_cosine") or 0.0), str(row.get("reference_id") or ""))
    )
    regions.sort(
        key=lambda row: (
            -float(row.get("raw_cosine") or 0.0),
            str(row.get("iso") or ""),
            str(row.get("region") or ""),
            str(row.get("city") or ""),
        )
    )
    cosine = max(
        (float(row.get("raw_cosine")) for row in (regions or matches)
         if isinstance(row.get("raw_cosine"), (int, float))),
        default=None,
    )
    return {
        "reference_matches": matches if has_matches else None,
        "reference_regions": regions if has_regions else None,
        "reference_cosine": cosine,
    }


def group_country_scores(
    scores: Iterable[CountryScore],
    countries_by_iso: Mapping[str, Mapping[str, Any]] | None = None,
) -> List[CountryScore]:
    """Group raw source scores into canonical product countries.

    Grouping happens before top-N slicing and normalization.  The group score is
    the maximum member score: summing members would double-count related
    Taiwan/mainland prompts, while max preserves every source scope's actual
    evidence in ``source_regions`` and ``fusion.members``.
    """
    grouped: Dict[str, List[CountryScore]] = {}
    for score in scores:
        canonical = canonical_country_iso(score.iso)
        if not canonical:
            continue
        grouped.setdefault(canonical, []).append(score)

    result: List[CountryScore] = []
    lookup = countries_by_iso or {}
    for canonical, members in grouped.items():
        members = sorted(members, key=lambda item: (-item.score, item.iso))
        best = members[0]
        policy_members = country_members(canonical)
        needs_group = len(members) > 1 or best.iso != canonical or policy_members != (canonical,)
        if not needs_group:
            result.append(best)
            continue

        root = lookup.get(canonical) or {}
        name = str(root.get("name") or best.name)
        name_zh = str(root.get("name_zh") or best.name_zh)
        lat = root.get("lat", best.lat)
        lng = root.get("lng", best.lng)
        source_regions = [
            _source_region_record(member, canonical)
            for member in members
        ]
        fusion = dict(best.fusion or {})
        fusion.update({
            "aggregation": "max",
            "group_iso": canonical,
            "member_count": len(members),
            "members": [
                {
                    "source_iso": member.iso,
                    "score": round(member.score, 6),
                    "raw_cosine": member.raw_cosine,
                    "score_type": member.score_type,
                }
                for member in members
            ],
        })
        reference = _merge_group_reference(members)
        result.append(replace(
            best,
            iso=canonical,
            name=name,
            name_zh=name_zh,
            lat=float(lat) if isinstance(lat, (int, float)) else 0.0,
            lng=float(lng) if isinstance(lng, (int, float)) else 0.0,
            matched_clues=_merge_group_clues(members, canonical),
            fusion=fusion,
            parent_iso=None,
            source_regions=source_regions,
            is_grouped=True,
            reference_matches=reference.get("reference_matches"),
            reference_regions=reference.get("reference_regions"),
            reference_cosine=reference.get("reference_cosine"),
        ))

    return sorted(result, key=lambda item: (-item.score, item.iso))


class ReasoningEngine:
    """线索 -> 候选国家评分。"""

    def __init__(self):
        self.countries = kg.load_countries()
        self._country_by_iso = {
            c["iso"]: c for c in self.countries
            if isinstance(c.get("iso"), str)
        }
        self._iso_set = {c["iso"] for c in self.countries if isinstance(c.get("iso"), str)}
        self._iso3_index = {
            str(c["iso3"]).upper(): c["iso"]
            for c in self.countries
            if isinstance(c.get("iso3"), str) and isinstance(c.get("iso"), str)
        }
        self._name_index = build_country_name_index(self.countries)
        # 城市目录懒加载：只有出现非国家名的 place_name 线索时才读取。
        self._city_index_cache: Optional[Dict[str, Tuple[Tuple[str, int, str, str], ...]]] = None

    def _city_name_index(self) -> Dict[str, Tuple[Tuple[str, int, str, str], ...]]:
        if self._city_index_cache is None:
            try:
                self._city_index_cache = load_city_name_index(BASE_DIR / "data" / "cities.json")
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                self._city_index_cache = {}
        return self._city_index_cache

    def resolve_place(self, value: object) -> Optional[PlaceResolution]:
        """Resolve country names/ISO codes first, then catalog city names."""
        if not isinstance(value, str):
            return None
        raw = value.strip()
        if not raw:
            return None
        if len(raw) == 2 and raw.isascii() and raw.isalpha():
            if raw == raw.upper() and raw in self._iso_set:
                return PlaceResolution(raw, 1.0, "iso")
            return None
        if len(raw) == 3 and raw.isascii() and raw.isalpha() and raw == raw.upper():
            iso = self._iso3_index.get(raw)
            return PlaceResolution(iso, 1.0, "iso3") if iso else None
        candidates = [raw]
        # 允许一个 CJK 行政/车站后缀：北京市 -> 北京，東京都/東京駅 -> 東京。
        if (not raw.isascii()) and raw[-1] in ADMIN_SUFFIX_CHARS:
            candidates.append(raw[:-1])
        city_index = None
        for candidate in candidates:
            iso = self._name_index.get(normalize_place_text(candidate))
            if iso:
                return PlaceResolution(iso, 1.0, "country_name")
            if city_index is None:
                city_index = self._city_name_index()
            city = resolve_city_name(city_index, candidate)
            if city is not None:
                return city
        return None

    def resolve_country_key(self, value: object) -> Optional[str]:
        """Canonical ISO for a place value, or None when not an exact match."""
        resolved = self.resolve_place(value)
        return resolved.iso if resolved is not None else None

    def _score_country(self, country: Dict[str, Any], clues: List[Clue],
                       exclusions: List[str]) -> Optional[CountryScore]:
        iso = country["iso"]
        if is_excluded(exclusions, iso):
            return None

        features = country.get("features", {})
        score = 0.0
        matched: List[Dict[str, Any]] = []

        for clue in clues:
            if clue.confidence < MIN_CLUE_CONFIDENCE:
                continue
            dim = clue.dimension
            value = clue.value
            if dim not in CATALOG_DIMENSIONS:
                # 观察维度没有目录元数据：明确不参与任何评分。
                continue
            dim_weight = DIMENSION_WEIGHTS.get(dim, 1.0)
            extra = clue.extra if isinstance(clue.extra, dict) else {}
            if extra.get("display_only") is True or extra.get("affects_ranking") is False:
                # 观察型证据（road/terrain/waterbody/infrastructure/signage）
                # 只展示，绝不参与任何评分。
                continue

            matched_value: Optional[str] = None
            gain_factor = 1.0
            provenance: Dict[str, Any] = {}
            if dim == "language":
                if isinstance(value, str) and value in country.get("languages_spoken", []):
                    matched_value = value
            elif dim == "driving_side":
                side = country.get("driving_side")
                if (isinstance(value, str) and isinstance(side, str)
                        and value.strip().lower() == side.strip().lower()):
                    matched_value = value
            elif dim == "place_name":
                # Validated regional provenance may arrive with a canonical
                # value (for example value=CN, source_iso=TW) while the raw
                # source row is TW. Scope the evidence to that source row
                # before grouping so it cannot boost mainland CN and so an
                # exclusion of TW neutralizes the evidence.
                claimed_source = normalize_iso(extra.get("source_iso"))
                if claimed_source and claimed_source not in self._iso_set:
                    continue
                if claimed_source and claimed_source != iso:
                    continue

                resolved = self.resolve_place(value)
                if resolved is not None and claimed_source:
                    # A canonical country value is allowed for a regional
                    # source only when it resolves to the same parent group.
                    # A different region source is never cross-credited.
                    if resolved.iso != claimed_source:
                        if source_region_iso(resolved.iso):
                            resolved = None
                        elif canonical_country_iso(resolved.iso) != canonical_country_iso(claimed_source):
                            resolved = None
                elif resolved is not None and resolved.iso != iso:
                    resolved = None

                if resolved is not None:
                    gain_factor = resolved.confidence_factor
                    provenance = {"resolution": resolved.resolution}
                    if claimed_source:
                        provenance["source_iso"] = claimed_source
                        parent = canonical_country_iso(claimed_source)
                        if parent:
                            provenance["parent_iso"] = parent
                    if resolved.city_name:
                        provenance["city_name"] = resolved.city_name
                    if resolved.city_id:
                        provenance["city_id"] = resolved.city_id
                    # 外部声明的城市必须由目录重新核实，不信任 extra 本身。
                    claimed_iso = normalize_iso(extra.get("catalog_iso"))
                    if claimed_iso:
                        if (claimed_iso != resolved.iso
                                and canonical_country_iso(claimed_iso) != canonical_country_iso(resolved.iso)):
                            resolved = None
                    claimed_name = extra.get("city_name")
                    if isinstance(claimed_name, str) and claimed_name.strip():
                        recheck = self.resolve_place(claimed_name)
                        if recheck is None or recheck.iso != iso:
                            resolved = None
                        else:
                            gain_factor = min(gain_factor, recheck.confidence_factor)
                            provenance = {"resolution": recheck.resolution}
                            if claimed_source:
                                provenance["source_iso"] = claimed_source
                                parent = canonical_country_iso(claimed_source)
                                if parent:
                                    provenance["parent_iso"] = parent
                            if recheck.city_name:
                                provenance["city_name"] = recheck.city_name
                            if recheck.city_id:
                                provenance["city_id"] = recheck.city_id
                    claimed_id = extra.get("city_id")
                    if resolved is not None and claimed_id is not None:
                        if str(claimed_id) != str(provenance.get("city_id")):
                            resolved = None
                    if resolved is not None:
                        matched_value = value
            else:
                if isinstance(value, str) and value in features.get(dim, []):
                    matched_value = value

            if matched_value is None:
                continue
            gain = clue.confidence * dim_weight * gain_factor
            score += gain
            entry = {
                "dimension": dim,
                "value": matched_value,
                "gain": round(gain, 4),
                "evidence": clue.evidence,
            }
            entry.update(provenance)
            origin_group = extra.get("origin_group")
            if isinstance(origin_group, str) and origin_group:
                entry["origin_group"] = origin_group
            matched.append(entry)

        # 同一 OCR 行派生的证据（如 "Eiffel Tower France" -> place_name FR +
        # culture eiffel）彼此相关：卡片全部保留，但每个 origin_group 只计
        # 最强的一条贡献，避免同一行证据被重复计分。
        score = 0.0
        best_by_group: Dict[str, int] = {}
        for index, entry in enumerate(matched):
            group = entry.get("origin_group")
            if not group:
                entry["counted"] = True
                score += entry["gain"]
                continue
            best = best_by_group.get(group)
            if best is None:
                best_by_group[group] = index
                entry["counted"] = True
                score += entry["gain"]
            elif entry["gain"] > matched[best]["gain"]:
                score += entry["gain"] - matched[best]["gain"]
                matched[best]["counted"] = False
                best_by_group[group] = index
                entry["counted"] = True
            else:
                entry["counted"] = False

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
        raw_scores: List[CountryScore] = []
        for country in self.countries:
            cs = self._score_country(country, clues, exclusions)
            if cs is not None:
                raw_scores.append(cs)

        # Group source scopes before normalization and top-N slicing. A raw
        # TW/HK/MO exclusion removes only that member; canonical CN removes
        # the complete product country.
        scores = group_country_scores(raw_scores, self._country_by_iso)
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


__all__ = [
    "CITY_NAME_ALIASES",
    "DIMENSION_WEIGHTS",
    "DRIVING_SIDE_WEIGHT",
    "PLACE_NAME_WEIGHT",
    "CountryScore",
    "PlaceResolution",
    "ReasoningEngine",
    "ADMIN_SUFFIX_CHARS",
    "LATIN_JOINER_STOPWORDS",
    "build_city_name_index",
    "build_country_name_index",
    "canonical_country_iso",
    "country_members",
    "find_exact_phrase",
    "group_country_scores",
    "find_place_phrase",
    "is_negated_prefix",
    "load_city_name_index",
    "normalize_place_text",
    "resolve_city_name",
]