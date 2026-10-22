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

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from ..domain.geolocation.clue import Clue
from ..domain.geolocation.engine import (
    CITY_NAME_ALIASES,
    find_place_phrase,
    normalize_place_text,
)
from ..domain.geolocation.country_policy import canonical_country_iso, normalize_iso
from .place_resolution import CityEvidence
from .city_scene_reasoning import SceneObservation, apply_scene_residual


CITY_CLUE_BOOST_SCALE = 0.12
CITY_CLUE_BOOST_CAP = 0.25
CITY_CLUE_SCORE_TYPE = "city_clue_fusion"
MIN_ASCII_CITY_NAME = 4
MIN_NON_ASCII_CITY_NAME = 2
MIN_EVIDENCE_CONFIDENCE = 0.3
_OCR_SOURCES = frozenset({"ocr", "point_click"})
_NETWORK_SOURCE_TYPES = frozenset({"web_search", "geocoder"})
_NETWORK_MATCH_KINDS = frozenset({
    "geocoder_locality", "geocoder_candidate", "web_candidate",
})


@dataclass(frozen=True)
class CityRerankResult:
    """Reranked rows plus honest coverage metadata (never fabricated rows)."""

    city_scores: List[Dict[str, Any]]
    clue_dimensions: Tuple[str, ...] = ()
    coverage_note: Optional[str] = None
    clue_notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class _Evidence:
    dimension: str
    text: str
    confidence: float
    evidence: str
    catalog_iso: Optional[str] = None
    city_id: Optional[int] = None
    source_iso: Optional[str] = None
    source: str = ""
    clue_id: str = ""
    source_refs: Tuple[Mapping[str, Any], ...] = ()
    match_kind: str = ""


_SOURCE_REF_FIELDS = (
    "provider", "provider_place_id", "label", "query", "attribution", "license",
    "fetched_at", "source_type", "url", "title", "excerpt", "observation_source",
)


def _source_refs_are_network(items: Sequence[Mapping[str, Any]]) -> bool:
    for item in items:
        source_type = str(item.get("source_type") or "").strip().lower()
        if source_type in _NETWORK_SOURCE_TYPES:
            return True
    return False


def _source_refs_from_clue(
    clue: Clue, extra: Mapping[str, Any]
) -> Tuple[Mapping[str, Any], ...]:
    raw_refs = extra.get("source_refs")
    if isinstance(raw_refs, Mapping):
        raw_refs = (raw_refs,)
    refs = _merge_source_refs(tuple(raw_refs or ()))
    if refs:
        return refs
    origin = str(extra.get("origin_source") or clue.source or "").strip().lower()
    observation_source = "ocr" if origin == "ocr" or clue.source == "ocr" else (
        "user" if origin in {"user_text", "user"} or clue.source == "user" else None
    )
    provider = str(extra.get("provider") or clue.source or "unknown").strip()[:40]
    if not provider:
        provider = "unknown"
    ref = {
        "provider": provider,
        "label": str(extra.get("matched_text") or clue.evidence or provider)[:200],
        "fetched_at": str(extra.get("fetched_at") or "")[:80],
    }
    if observation_source:
        ref["observation_source"] = observation_source
    return (ref,)


def _merge_source_refs(
    *groups: object,
) -> Tuple[Mapping[str, Any], ...]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for group in groups:
        if isinstance(group, Mapping):
            group = (group,)
        for item in group or ():
            if not isinstance(item, Mapping):
                continue
            ref = {
                key: item.get(key)
                for key in _SOURCE_REF_FIELDS
                if item.get(key) is not None
            }
            raw_url = str(ref.get("url") or "").strip()
            if raw_url:
                try:
                    parts = urlsplit(raw_url)
                    hostname = parts.hostname
                except ValueError:
                    parts = None
                    hostname = None
                if (
                    parts is None
                    or parts.scheme.lower() not in {"http", "https"}
                    or not hostname
                    or parts.username is not None
                    or parts.password is not None
                    or len(raw_url) > 1000
                ):
                    ref.pop("url", None)
            else:
                ref.pop("url", None)
            if not ref.get("provider"):
                ref["provider"] = str(item.get("origin_source") or "unknown")[:40]
            if not ref.get("label"):
                ref["label"] = str(ref.get("title") or ref["provider"])[:200]
            key = tuple(str(ref.get(field) or "") for field in (
                "source_type", "provider", "url", "title", "observation_source",
            ))
            if key in seen:
                continue
            seen.add(key)
            merged.append(ref)
    return tuple(merged[:8])


class CityClueReasoner:
    """Deterministic, bounded clue rerank over gateway-validated city rows."""

    def __init__(self, catalog: Optional[Mapping[str, Mapping[Any, Any]]] = None) -> None:
        # ``catalog`` is ``{iso: {city_id: CityRow}}`` (see services.city_catalog).
        # ``None`` means "cannot scope/validate names" -> fail closed, no rerank.
        self._catalog = catalog

    def rerank(self, city_scores: Sequence[Dict[str, Any]],
               clues: Sequence[Clue], *, iso: str,
               city_evidence: Sequence[CityEvidence] = (),
               scene_evidence: Mapping[str, Any] | None = None,
               scene_observations: Sequence[SceneObservation] = (),
               scene_notes: Sequence[str] = ()) -> CityRerankResult:
        rows = [dict(row) for row in (city_scores or ())]
        iso_key = str(iso or "").strip().upper()
        dimensions: set[str] = set()
        all_notes: List[str] = list(scene_notes)
        place_notes: List[str] = []
        changed = False

        scope: Mapping[Any, Any] = {}
        if self._catalog is not None and iso_key:
            scope = self._catalog.get(iso_key) or {}
        if scope:
            evidence = self._extract_evidence(clues, iso_key)
            evidence.extend(self._extract_city_evidence(city_evidence, iso_key, scope))
            if evidence:
                buckets = self._name_buckets(scope, iso_key)
                rows_by_id = {
                    str(row.get("city_id")): row
                    for row in rows
                    if row.get("city_id") is not None
                }
                matched: Dict[str, Dict[str, Dict[str, Any]]] = {}
                for item in evidence:
                    if item.city_id is not None:
                        ids = [item.city_id]
                        hit = normalize_place_text(self._row_name(scope.get(item.city_id)))
                        if not hit:
                            continue
                    else:
                        norm = normalize_place_text(item.text)
                        if not norm:
                            continue
                        hit = self._match_name(norm, buckets)
                        if hit is None:
                            continue
                        ids = buckets[hit]
                    if len(ids) > 1:
                        place_notes.append(
                            f"线索中的「{hit}」在该国目录中不唯一，未参与重排"
                        )
                        continue
                    display = self._row_name(scope.get(ids[0])) or hit
                    city_key = str(ids[0])
                    if city_key not in rows_by_id:
                        place_notes.append(
                            f"线索命中的「{display}」不在本次模型返回的前 {len(rows)} 个城市候选内，未参与重排"
                        )
                        continue
                    entries = matched.setdefault(city_key, {})
                    entry = entries.get("__city__")
                    refs = _merge_source_refs(item.source_refs)
                    if entry is None:
                        entries["__city__"] = {
                            "dimension": item.dimension,
                            "confidence": min(1.0, item.confidence),
                            "evidence": item.evidence,
                            "value": display,
                            "source": item.source,
                            "clue_id": item.clue_id,
                            "source_refs": refs,
                            "match_kind": item.match_kind,
                        }
                    else:
                        # Multiple OCR/network rows for one city support one
                        # bounded contribution; only provenance is merged.
                        entry["source_refs"] = _merge_source_refs(
                            tuple(entry.get("source_refs") or ()), refs
                        )
                        if item.confidence > entry["confidence"]:
                            entry.update(
                                dimension=item.dimension,
                                confidence=min(1.0, item.confidence),
                                evidence=item.evidence,
                                value=display,
                                source=item.source,
                                clue_id=item.clue_id,
                                match_kind=item.match_kind,
                            )

                if matched:
                    boosted: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
                    for city_key, entries in matched.items():
                        terms: List[Dict[str, Any]] = []
                        for name, entry in sorted(entries.items()):
                            term = round(
                                min(1.0, float(entry["confidence"])) * CITY_CLUE_BOOST_SCALE,
                                6,
                            )
                            terms.append({
                                "dimension": entry["dimension"],
                                "value": entry.get("value", name),
                                "gain": term,
                                "evidence": entry["evidence"]
                                or f"城市名精确匹配：{name}",
                                "source": entry.get("source") or "",
                                "clue_id": entry.get("clue_id") or "",
                                "source_refs": list(entry.get("source_refs") or ()),
                                "match_kind": entry.get("match_kind") or "city_name",
                            })
                            dimensions.add(entry["dimension"])
                        total = min(CITY_CLUE_BOOST_CAP, sum(term["gain"] for term in terms))
                        boosted[city_key] = (round(total, 6), terms)

                    before_order = sorted(
                        rows,
                        key=lambda row: (
                            -_model_score(row),
                            _numeric_city_id(row),
                        ),
                    )
                    before_rank = {
                        str(row.get("city_id")): index
                        for index, row in enumerate(before_order, 1)
                    }
                    result: List[Dict[str, Any]] = []
                    for row in rows:
                        city_key = str(row.get("city_id"))
                        boost, terms = boosted.get(city_key, (0.0, []))
                        base_score = _model_score(row)
                        final_score = min(1.0, base_score + boost)
                        fusion = dict(row.get("fusion")) if isinstance(row.get("fusion"), Mapping) else {}
                        fusion.update({
                            "base_score": round(base_score, 6),
                            "model_base_score": round(base_score, 6),
                            "place_boost": round(boost, 6),
                            "place_match_boost": round(boost, 6),
                            "clue_boost": round(boost, 6),
                            "clue_boost_cap": CITY_CLUE_BOOST_CAP,
                            "clue_boost_scale": CITY_CLUE_BOOST_SCALE,
                            "scene_delta": 0.0,
                            "final_score": round(final_score, 6),
                            "rank_before": before_rank.get(city_key),
                        })
                        updated = {
                            **row,
                            "score": final_score,
                            "fusion": fusion,
                        }
                        if boost > 0.0:
                            updated["score_type"] = CITY_CLUE_SCORE_TYPE
                            updated["matched_clues"] = terms
                        result.append(updated)
                    after_order = sorted(
                        result,
                        key=lambda row: (
                            -_candidate_score(row),
                            _numeric_city_id(row),
                        ),
                    )
                    after_rank = {
                        str(row.get("city_id")): index
                        for index, row in enumerate(after_order, 1)
                    }
                    for row in result:
                        row["fusion"]["rank_after"] = after_rank.get(
                            str(row.get("city_id"))
                        )
                        row["rank_before"] = before_rank.get(str(row.get("city_id")))
                        row["rank_after"] = after_rank.get(str(row.get("city_id")))
                    rows = result
                    changed = True

        if scene_observations or scene_notes:
            scene_result = apply_scene_residual(
                rows,
                scene_evidence,
                scene_observations,
                notes=[*place_notes, *scene_notes],
            )
            rows = scene_result.city_scores
            for row in rows:
                fusion = row.get("fusion")
                if isinstance(fusion, dict):
                    place_match_boost = fusion.get("place_match_boost")
                    if place_match_boost is None and fusion.get("place_boost") is not None:
                        fusion["place_match_boost"] = fusion.get("place_boost")
            dimensions.update(scene_result.dimensions)
            all_notes = list(scene_result.clue_notes)
            changed = changed or scene_result.attempted
        else:
            all_notes = _unique_notes([*place_notes, *all_notes])

        if changed:
            rows.sort(key=_city_order_key)
        if all_notes:
            rows = [{**row, "clue_notes": list(all_notes)} for row in rows]
        return CityRerankResult(
            rows,
            tuple(sorted(dimensions)),
            self._join_notes(all_notes),
            tuple(all_notes),
        )

    # ---- evidence -----------------------------------------------------------
    def _extract_evidence(self, clues: Sequence[Clue], iso: str) -> List[_Evidence]:
        evidence: List[_Evidence] = []
        for clue in clues or ():
            if not isinstance(clue, Clue):
                continue
            extra = clue.extra if isinstance(clue.extra, dict) else {}
            # Display-only observations must not be promoted from literal text
            # into a city match. Resolver-validated CityEvidence below remains
            # an independent, catalog-scoped path.
            if extra.get("display_only") is True or extra.get("affects_ranking") is False:
                continue
            if (
                extra.get("candidate_only") is True
                or str(extra.get("match_kind") or "").strip().lower() in _NETWORK_MATCH_KINDS
                or _source_refs_are_network(
                    tuple(extra.get("source_refs") or ())
                    if not isinstance(extra.get("source_refs"), Mapping)
                    else (extra.get("source_refs"),)
                )
            ):
                continue
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
            catalog_iso = normalize_iso(extra.get("catalog_iso"))
            source_iso = normalize_iso(extra.get("source_iso"))
            if source_iso:
                # A validated regional clue is scoped to its raw source only.
                # A canonical country value/parent may still match the same
                # source scope, but another child region never cross-matches.
                if source_iso != iso:
                    continue
                if catalog_iso and (
                    catalog_iso != iso
                    and canonical_country_iso(catalog_iso) != canonical_country_iso(iso)
                ):
                    continue
            elif catalog_iso and catalog_iso != iso:
                continue  # 线索自己声明了别的国家，不在本次城市范围内
            texts = [clue.value]
            city_name = extra.get("city_name")
            if isinstance(city_name, str) and city_name.strip():
                # 语义线索（value=ISO）把已校验城市名放在 extra 里。
                texts.append(city_name)
            for text in texts:
                source_refs = _source_refs_from_clue(clue, extra)
                evidence.append(_Evidence(
                    dimension=dimension,
                    text=text,
                    confidence=min(1.0, confidence),
                    evidence=str(clue.evidence or ""),
                    catalog_iso=catalog_iso,
                    source=str(clue.source or ""),
                    clue_id=str(extra.get("guided_clue_id") or extra.get("clue_id") or ""),
                    source_refs=source_refs,
                    match_kind=str(extra.get("match_kind") or extra.get("match") or ""),
                ))
        return evidence

    def _extract_city_evidence(
        self,
        city_evidence: Sequence[CityEvidence],
        iso: str,
        scope: Mapping[Any, Any],
    ) -> List[_Evidence]:
        """Accept only resolver-validated, unique city evidence in this scope."""

        best: Dict[int, _Evidence] = {}
        for item in city_evidence or ():
            if not isinstance(item, CityEvidence):
                continue
            if str(item.source_iso or "").strip().upper() != iso:
                continue
            item_source_type = str(getattr(item, "source_type", "") or "").strip().lower()
            if (
                item_source_type in _NETWORK_SOURCE_TYPES
                or str(item.match_kind or "").strip().lower() in _NETWORK_MATCH_KINDS
                or _source_refs_are_network(tuple(item.sources or ()))
            ):
                continue
            try:
                city_id = int(item.city_id)
            except (TypeError, ValueError):
                continue
            if city_id not in scope:
                continue
            confidence = min(1.0, max(0.0, float(item.confidence)))
            if confidence < MIN_EVIDENCE_CONFIDENCE:
                continue
            refs = _merge_source_refs(tuple(item.sources or ()))
            if not refs and item_source_type:
                refs = _merge_source_refs(({
                    "provider": item_source_type,
                    "label": item.text,
                    "source_type": item_source_type,
                },))
            current = best.get(city_id)
            if current is None:
                best[city_id] = _Evidence(
                    dimension="place_name",
                    text=item.text,
                    confidence=confidence,
                    evidence=item.text,
                    catalog_iso=iso,
                    city_id=city_id,
                    source_iso=iso,
                    source="place_resolution",
                    clue_id=(item.clue_ids[0] if item.clue_ids else ""),
                    source_refs=refs,
                    match_kind=str(item.match_kind or ""),
                )
            else:
                best[city_id] = replace(
                    current,
                    confidence=max(current.confidence, confidence),
                    evidence=current.evidence or item.text,
                    source_refs=_merge_source_refs(current.source_refs, refs),
                )
        return list(best.values())
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


def _numeric_city_id(row: Mapping[str, Any]) -> int:
    try:
        return int(str(row.get("city_id")))
    except (TypeError, ValueError):
        return 1 << 62


def _model_score(row: Mapping[str, Any]) -> float:
    fusion = row.get("fusion")
    if isinstance(fusion, Mapping):
        value = fusion.get("model_base_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if number == number and number not in (float("inf"), float("-inf")):
                return number
    value = row.get("score", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _candidate_score(row: Mapping[str, Any]) -> float:
    fusion = row.get("fusion")
    if isinstance(fusion, Mapping):
        value = fusion.get("final_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if number == number and number not in (float("inf"), float("-inf")):
                return number
    try:
        return float(row.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _unique_notes(notes: Sequence[str]) -> List[str]:
    unique: List[str] = []
    for raw in notes:
        note = str(raw or "").strip()
        if note and note not in unique:
            unique.append(note)
    return unique


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
    return (-_candidate_score(row), _numeric_city_id(row))


__all__ = [
    "CITY_CLUE_BOOST_CAP",
    "CITY_CLUE_BOOST_SCALE",
    "CITY_CLUE_SCORE_TYPE",
    "CityClueReasoner",
    "CityRerankResult",
]
