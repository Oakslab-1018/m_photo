"""会话线索 -> 已校验城市行的重排（read-only，绝不新增/伪造城市行）。

设计约束（城市身份由 ``data/cities.json`` 和 gateway 校验负责）：

- 只在 gateway 已通过 :meth:`StreetClipGateway.validate_city_ranking` 的行上
  调整 ``score`` / ``score_type`` / ``matched_clues`` / ``fusion``；
  ``raw_cosine``、``city_id``、``name``、``region``、坐标、参考诊断保持原样。
- 匹配证据只来自：用户回答的 ``place_name``、带可靠标记的 OCR ``signage``
  文本；先按 ``catalog_iso``（若线索携带）限定国家，再在该 ISO 的完整目录里
  要求城市名唯一。
- 精确匹配规则：
    * 拉丁单字城市名必须 >= 4 字符，且不得紧跟在另一个普通词之后
      （防止 ``York`` 命中 ``New York``）；常见连接词（in/at/of/...）除外。
    * CJK 城市名（>= 2 字）允许紧跟行政区/车站后缀：北京市、東京駅、大阪駅。
    * 否定语境（not/no/不是/并非…）跳过。
- 同一城市名只计一次（重复 OCR 行不叠加）；单个城市总加成封顶
  :data:`CITY_CLUE_BOOST_CAP`，得分封顶 1.0；score_type 改写为
  :data:`CITY_CLUE_SCORE_TYPE` 且 ``fusion.base_score`` 保留原始分数。
- 命中目录但不在模型返回行内（例如模型只返回前 10 行）时：只写
  ``coverage_note`` 说明“未参与重排”，绝不凭空补一行没有模型分数的候选。
- 无匹配证据时原样返回（顺序、字段形状完全不变）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..domain.geolocation.clue import Clue
from ..domain.geolocation.engine import (
    CITY_NAME_ALIASES,
    find_place_phrase,
    normalize_place_text,
)


CITY_CLUE_BOOST_SCALE = 0.12
CITY_CLUE_BOOST_CAP = 0.25
CITY_CLUE_SCORE_TYPE = "city_clue_fusion"
MIN_ASCII_CITY_NAME = 4
MIN_NON_ASCII_CITY_NAME = 2
MIN_EVIDENCE_CONFIDENCE = 0.3
_OCR_SOURCES = frozenset({"ocr", "point_click"})


@dataclass(frozen=True)
class CityRerankResult:
    """Reranked rows plus honest coverage metadata (never fabricated rows)."""

    city_scores: List[Dict[str, Any]]
    clue_dimensions: Tuple[str, ...] = ()
    coverage_note: Optional[str] = None


@dataclass(frozen=True)
class _Evidence:
    dimension: str
    text: str
    confidence: float
    evidence: str
    catalog_iso: Optional[str] = None


class CityClueReasoner:
    """Deterministic, bounded clue rerank over gateway-validated city rows."""

    def __init__(self, catalog: Optional[Mapping[str, Mapping[Any, Any]]] = None) -> None:
        # ``catalog`` is ``{iso: {city_id: CityRow}}`` (see services.city_catalog).
        # ``None`` means "cannot scope/validate names" -> fail closed, no rerank.
        self._catalog = catalog

    def rerank(self, city_scores: Sequence[Dict[str, Any]],
               clues: Sequence[Clue], *, iso: str) -> CityRerankResult:
        rows = list(city_scores or [])
        iso_key = str(iso or "").strip().upper()
        if not rows or self._catalog is None:
            return CityRerankResult(rows)
        scope = self._catalog.get(iso_key) or {}
        if not scope:
            return CityRerankResult(rows)
        evidence = self._extract_evidence(clues, iso_key)
        if not evidence:
            return CityRerankResult(rows)

        buckets = self._name_buckets(scope, iso_key)
        rows_by_id = {str(row.get("city_id")): row for row in rows if row.get("city_id") is not None}
        matched: Dict[str, Dict[str, Dict[str, Any]]] = {}
        notes: List[str] = []

        for item in evidence:
            norm = normalize_place_text(item.text)
            if not norm:
                continue
            hit = self._match_name(norm, buckets)
            if hit is None:
                continue
            ids = buckets[hit]
            if len(ids) > 1:
                notes.append(f"线索中的「{hit}」在该国目录中不唯一，未参与重排")
                continue
            display = self._row_name(scope.get(ids[0])) or hit
            city_key = str(ids[0])
            if city_key not in rows_by_id:
                notes.append(
                    f"线索命中的「{display}」不在本次模型返回的前 {len(rows)} 个城市候选内，未参与重排"
                )
                continue
            entries = matched.setdefault(city_key, {})
            entry = entries.get(hit)
            if entry is None:
                entries[hit] = {
                    "dimension": item.dimension,
                    "confidence": min(1.0, item.confidence),
                    "evidence": item.evidence,
                    "value": display,
                }
            elif item.confidence > entry["confidence"]:
                entry.update(dimension=item.dimension,
                             confidence=min(1.0, item.confidence),
                             evidence=item.evidence,
                             value=display)

        if not matched:
            return CityRerankResult(rows, (), self._join_notes(notes))

        boosted: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
        dimensions: set = set()
        for city_key, entries in matched.items():
            terms: List[Dict[str, Any]] = []
            for name, entry in sorted(entries.items()):
                term = round(min(1.0, float(entry["confidence"])) * CITY_CLUE_BOOST_SCALE, 6)
                terms.append({
                    "dimension": entry["dimension"],
                    "value": entry.get("value", name),
                    "gain": term,
                    "evidence": entry["evidence"] or f"城市名精确匹配：{name}",
                })
                dimensions.add(entry["dimension"])
            total = min(CITY_CLUE_BOOST_CAP, sum(term["gain"] for term in terms))
            boosted[city_key] = (round(total, 6), terms)

        result: List[Dict[str, Any]] = []
        for row in rows:
            city_key = str(row.get("city_id"))
            if city_key not in boosted:
                result.append(row)
                continue
            boost, terms = boosted[city_key]
            base_score = float(row.get("score", 0.0))
            result.append({
                **row,
                "score": min(1.0, base_score + boost),
                "score_type": CITY_CLUE_SCORE_TYPE,
                "matched_clues": terms,
                "fusion": {
                    "base_score": base_score,
                    "clue_boost": boost,
                    "clue_boost_cap": CITY_CLUE_BOOST_CAP,
                    "clue_boost_scale": CITY_CLUE_BOOST_SCALE,
                },
            })
        result.sort(key=_city_order_key)
        return CityRerankResult(result, tuple(sorted(dimensions)), self._join_notes(notes))

    # ---- evidence -----------------------------------------------------------
    def _extract_evidence(self, clues: Sequence[Clue], iso: str) -> List[_Evidence]:
        evidence: List[_Evidence] = []
        for clue in clues or ():
            if not isinstance(clue, Clue):
                continue
            extra = clue.extra if isinstance(clue.extra, dict) else {}
            dimension = clue.dimension
            if dimension == "place_name":
                allowed = clue.source == "user" or extra.get("validated") is True
            elif dimension == "signage":
                allowed = extra.get("reliable") is True or clue.source in _OCR_SOURCES
            else:
                continue
            if not allowed or not isinstance(clue.value, str) or not clue.value.strip():
                continue
            confidence = float(clue.confidence) if isinstance(clue.confidence, (int, float)) else 0.0
            if not (confidence >= MIN_EVIDENCE_CONFIDENCE):
                continue
            catalog_iso = extra.get("catalog_iso")
            catalog_iso = str(catalog_iso).strip().upper() if catalog_iso else None
            if catalog_iso and catalog_iso != iso:
                continue  # 线索自己声明了别的国家，不在本次城市范围内
            texts = [clue.value]
            city_name = extra.get("city_name")
            if isinstance(city_name, str) and city_name.strip():
                # 语义线索（value=ISO）把已校验城市名放在 extra 里。
                texts.append(city_name)
            for text in texts:
                evidence.append(_Evidence(
                    dimension=dimension,
                    text=text,
                    confidence=min(1.0, confidence),
                    evidence=str(clue.evidence or ""),
                    catalog_iso=catalog_iso,
                ))
        return evidence

    # ---- name matching ------------------------------------------------------
    @staticmethod
    def _row_name(row: Any) -> str:
        name = getattr(row, "name", None)
        if isinstance(name, str):
            return name
        if isinstance(row, Mapping) and isinstance(row.get("name"), str):
            return row["name"]
        return ""

    def _name_buckets(self, scope: Mapping[Any, Any], iso: str) -> Dict[str, List[Any]]:
        buckets: Dict[str, List[Any]] = {}
        for city_id, row in scope.items():
            name = normalize_place_text(self._row_name(row))
            if not _usable_name(name):
                continue
            buckets.setdefault(name, []).append(city_id)
        # 有限的中文/日文别名仍必须解析到本 ISO 目录里已存在的城市行。
        for alias, (catalog_name, alias_iso) in CITY_NAME_ALIASES.items():
            if alias_iso != iso:
                continue
            target = buckets.get(normalize_place_text(catalog_name))
            alias_key = normalize_place_text(alias)
            if target and alias_key and alias_key not in buckets:
                buckets[alias_key] = list(target)
        return buckets

    @staticmethod
    def _match_name(norm: str, buckets: Mapping[str, List[Any]]) -> Optional[str]:
        # 最长优先：New York City > York；命中后不再考虑更短名字
        for name in sorted(buckets, key=lambda item: (-len(item), item)):
            if _city_phrase_found(norm, name):
                return name
        return None

    @staticmethod
    def _join_notes(notes: List[str]) -> Optional[str]:
        unique: List[str] = []
        for note in notes:
            if note not in unique:
                unique.append(note)
        if not unique:
            return None
        if len(unique) == 1:
            return unique[0]
        suffix = " 等" if len(unique) > 2 else ""
        return "；".join(unique[:2]) + suffix


def _usable_name(normalized_name: str) -> bool:
    if not normalized_name:
        return False
    compact = normalized_name.replace(" ", "")
    if normalized_name.isascii():
        return len(compact) >= MIN_ASCII_CITY_NAME
    return len(compact) >= MIN_NON_ASCII_CITY_NAME


def _city_phrase_found(norm_text: str, name: str) -> bool:
    """Delegate to the shared place matcher (boundaries, suffix, Latin guard)."""
    return find_place_phrase(norm_text, name)


def _city_order_key(row: Mapping[str, Any]) -> Tuple[float, int]:
    try:
        city_id = int(str(row.get("city_id")))
    except (TypeError, ValueError):
        city_id = 1 << 62
    try:
        score = float(row.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return (-score, city_id)


__all__ = [
    "CITY_CLUE_BOOST_CAP",
    "CITY_CLUE_BOOST_SCALE",
    "CITY_CLUE_SCORE_TYPE",
    "CityClueReasoner",
    "CityRerankResult",
]