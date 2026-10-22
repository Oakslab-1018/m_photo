"""Turn literal user place text into catalog city evidence or candidates.

Local source-backed matches are the only path allowed to produce confirmed
``CityEvidence`` and score boosts.  Geocoder and web results are recall-only:
they may produce catalog city candidates with confidence zero, but never a
resolved location or direct place-match boost.  External provider coordinates
and identities are always ignored.
"""
from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, Sequence

from ..domain.geolocation.country_policy import canonical_country_iso, normalize_iso
from .city_catalog import CityRow
from .city_labels import city_ids_for_name_zh
from .geocoding import (
    GeocoderHit,
    GeocoderRequest,
    GeocodingUnavailable,
    PlaceGeocoder,
)
from .local_place_index import LocalPlaceIndex, get_local_place_index
from .web_place_candidates import candidates_from_web_result

logger = logging.getLogger(__name__)

_RESOLUTION_VERSION = 3
_MAX_TERMS = 4
_MAX_EXACT_QUERY_LIMIT = 2000
_SOURCE_TYPES = frozenset({
    "local_catalog", "local_poi", "local_road", "web_search", "geocoder",
})
_LOCATIVE_MARKERS = (
    "路牌", "门牌", "招牌", "标牌", "指示牌", "旁边", "附近", "对面", "隔壁", "周围",
)
_LOCATION_SUFFIXES = (
    "大道", "大街", "路", "街", "巷子", "巷", "弄", "胡同", "广场", "公园",
    "景区", "车站", "站", "机场", "码头", "寺", "祠", "桥", "里", "片区",
    "社区", "村", "镇", "乡", "road", "street", "avenue", "lane", "square",
    "park", "station", "bridge", "temple", "市", "区", "县", "馆", "店", "塔", "大厦", "楼",
)


def _external_search_allowed(mentions: Sequence[MentionLike]) -> bool:
    for mention in mentions:
        if getattr(mention, "parser", "") in {"deepseek", "ocr_adopted"}:
            return True
        quote = str(getattr(mention, "quote", "") or "").casefold()
        if any(marker in quote for marker in _LOCATIVE_MARKERS):
            return True
        for term in getattr(mention, "search_terms", ()) or ():
            lowered = str(term or "").strip().casefold()
            if any(lowered.endswith(suffix) for suffix in _LOCATION_SUFFIXES):
                return True
    return False


if TYPE_CHECKING:
    from .geo_catalog_store import GeoCatalogStore


class MentionLike(Protocol):
    quote: str
    search_terms: tuple[str, ...]


class ResolutionError(ValueError):
    """A resolution input was malformed."""


@dataclass(frozen=True)
class CityEvidence:
    """One catalog-resolved city and the bounded sources supporting it."""

    city_id: int
    source_iso: str
    text: str
    confidence: float
    match_kind: str
    clue_ids: tuple[str, ...] = ()
    sources: tuple[Mapping[str, Any], ...] = ()
    source_type: str = ""


@dataclass(frozen=True)
class PlaceResolutionBatch:
    """Bounded result of resolving one message's literal place terms."""

    version: int
    status: str
    query_terms: tuple[str, ...]
    city_evidence: tuple[CityEvidence, ...] = ()
    city_ids: tuple[int, ...] = ()
    sources: tuple[Mapping[str, Any], ...] = ()
    foreign_country_isos: tuple[str, ...] = ()
    notice: str | None = None
    fetched_at: str = ""
    expires_at: str = ""
    candidate_cities: tuple[CityEvidence, ...] = ()

    @property
    def city_id_strings(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.city_ids)

    @property
    def is_unique(self) -> bool:
        return self.status == "resolved" and len(self.city_ids) == 1


@dataclass(frozen=True)
class _ExactCity:
    row: CityRow
    canonical_country: str
    names: tuple[str, ...]
    match_kind: str = "catalog_exact"
    confidence: float = 1.0
    source_type: str = "local_catalog"
    sources: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class _LocalState:
    nonempty: tuple[tuple[_ExactCity, ...], ...]
    union: tuple[_ExactCity, ...]
    intersection: tuple[_ExactCity, ...]
    all_terms_matched: bool
    ambiguous: bool
    conflict: bool

    @property
    def has_matches(self) -> bool:
        return bool(self.union)


def _normalize_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in text if character.isalnum())


def _city_key(city: _ExactCity) -> tuple[str, int]:
    return (city.row.iso, city.row.city_id)


def _source_key(source: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(source.get("source_type") or ""),
        str(source.get("provider") or ""),
        str(source.get("provider_place_id") or ""),
        str(source.get("url") or source.get("label") or ""),
    )


def _unique_sources(
    sources: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    result: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        key = _source_key(source)
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(source))
    return tuple(result)


def _merge_exact_city(existing: _ExactCity, incoming: _ExactCity) -> _ExactCity:
    winner = incoming if incoming.confidence > existing.confidence else existing
    return _ExactCity(
        row=winner.row,
        canonical_country=winner.canonical_country,
        names=tuple(dict.fromkeys((*existing.names, *incoming.names))),
        match_kind=winner.match_kind,
        confidence=winner.confidence,
        source_type=winner.source_type,
        sources=_unique_sources((*existing.sources, *incoming.sources)),
    )


def _unique_cities(cities: Sequence[_ExactCity]) -> tuple[_ExactCity, ...]:
    result: dict[tuple[str, int], _ExactCity] = {}
    for city in cities:
        key = _city_key(city)
        existing = result.get(key)
        result[key] = city if existing is None else _merge_exact_city(existing, city)
    return tuple(result.values())


def _unavailable_notice(error: GeocodingUnavailable) -> str:
    reason = str(getattr(error, "reason", "unavailable") or "unavailable")
    status = getattr(error, "status_code", None)
    cooldown = bool(getattr(error, "cooldown", False))
    if reason == "connect_timeout":
        summary = "地名检索连接超时（ConnectTimeout）"
    elif reason == "read_timeout":
        summary = "地名检索响应超时（ReadTimeout）"
    elif reason == "timeout":
        summary = "地名检索超时"
    elif reason == "connect_error":
        summary = "地名检索网络连接失败（ConnectError）"
    elif reason == "rate_limited" or (reason == "http_status" and status == 429):
        summary = "地名检索服务限流（HTTP 429），请稍后重试"
    elif reason == "http_status" and status == 400:
        summary = "地名检索请求未被服务接受（HTTP 400），这不是“没有结果”"
    elif reason == "http_status" and status in {401, 403}:
        summary = f"地名检索服务拒绝访问（HTTP {status}）"
    elif reason == "http_status" and status == 404:
        summary = "地名检索接口不存在（HTTP 404）"
    elif reason == "http_status" and status == 408:
        summary = "地名检索请求超时（HTTP 408）"
    elif reason == "http_status" and isinstance(status, int) and status >= 500:
        summary = f"地名检索服务暂时异常（HTTP {status}）"
    elif reason == "http_status" and status is not None:
        summary = f"地名检索服务返回 HTTP {status}"
    elif reason == "unexpected_redirect":
        summary = f"地名检索接口返回意外跳转（HTTP {status}）"
    elif reason == "invalid_json":
        summary = "地名检索返回格式异常（InvalidJSON）"
    elif reason == "malformed_response":
        summary = "地名检索返回内容异常（MalformedResponse）"
    elif reason == "response_too_large":
        summary = "地名检索响应过大，已停止处理"
    elif reason == "invalid_config":
        summary = "地名检索配置无效"
    elif reason == "transport_error":
        summary = "地名检索网络传输失败"
    else:
        summary = "地名检索暂时不可用"
    if cooldown:
        summary = f"{summary}；服务正在短暂冷却"
    return f"{summary}。文字线索已保留，未自动选择城市。"


def _bounded_terms(mentions: Sequence[MentionLike]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for mention in mentions:
        terms = getattr(mention, "search_terms", ()) or ()
        for item in terms:
            text = " ".join(str(item or "").split())
            key = _normalize_text(text)
            if not text or not key or len(text) > 60 or key in seen:
                continue
            seen.add(key)
            result.append(text)
            if len(result) >= _MAX_TERMS:
                return tuple(result)
    return tuple(result)


def _source_ref(
    *,
    provider: str,
    label: str,
    query: str,
    source_type: str,
    provider_place_id: str | None = None,
    attribution: str | None = None,
    license_name: str | None = None,
    title: str | None = None,
    url: str | None = None,
    excerpt: str | None = None,
    fetched_at: str,
) -> dict[str, Any]:
    normalized_type = source_type if source_type in _SOURCE_TYPES else ""
    return {
        "provider": str(provider or "")[:40],
        "provider_place_id": (
            str(provider_place_id)[:200] if provider_place_id is not None else None
        ),
        "label": str(label or "")[:200],
        "query": str(query or "")[:200],
        "attribution": str(attribution)[:200] if attribution is not None else None,
        "license": str(license_name)[:80] if license_name is not None else None,
        "fetched_at": str(fetched_at or "")[:80],
        "source_type": normalized_type,
        "title": str(title or label or "")[:200],
        "url": str(url)[:1000] if url else None,
        "excerpt": str(excerpt or "")[:300],
    }


def _city_row(place: Any) -> CityRow | None:
    city_id = getattr(place, "place_id", None)
    lat = getattr(place, "lat", None)
    lng = getattr(place, "lng", None)
    population = getattr(place, "population", None)
    if city_id is None or lat is None or lng is None or population is None:
        return None
    try:
        return CityRow(
            city_id=int(city_id),
            iso=str(getattr(place, "source_iso", "") or "").upper(),
            name=str(getattr(place, "name", "") or ""),
            region=str(getattr(place, "code", "") or ""),
            lat=float(lat),
            lng=float(lng),
            population=int(population),
        )
    except (TypeError, ValueError):
        return None


def _place_names(place: Any, extra_names: Sequence[str] = ()) -> tuple[str, ...]:
    aliases = getattr(place, "aliases", ()) or ()
    if isinstance(aliases, str):
        aliases = (aliases,)
    elif not isinstance(aliases, (list, tuple, set)):
        aliases = ()
    values = (
        *extra_names,
        str(getattr(place, "name", "") or ""),
        str(getattr(place, "ascii_name", "") or ""),
        str(getattr(place, "name_zh", "") or ""),
        *aliases,
    )
    result: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


class PlaceResolutionService:
    """Local source-backed resolution first; external providers are recall-only."""

    def __init__(
        self,
        *,
        geocoder: PlaceGeocoder,
        cache_ttl_seconds: float = 24 * 60 * 60,
        negative_cache_ttl_seconds: float = 10 * 60,
    ) -> None:
        self.geocoder = geocoder
        self.cache_ttl_seconds = max(60.0, float(cache_ttl_seconds))
        self.negative_cache_ttl_seconds = max(30.0, float(negative_cache_ttl_seconds))

    def resolve(
        self,
        store: "GeoCatalogStore",
        mentions: Sequence[MentionLike],
        *,
        country_iso: str | None = None,
        source_isos: Sequence[str] = (),
        between_local_and_network: Callable[[], None] | None = None,
    ) -> PlaceResolutionBatch:
        terms = _bounded_terms(mentions)
        if not terms:
            return self._batch("unresolved", (), notice="没有可检索的地点文字。")

        canonical = canonical_country_iso(country_iso or "")
        scopes = tuple(
            item for item in (normalize_iso(value) for value in source_isos) if item
        )
        local_index = get_local_place_index()
        local_by_term = {
            term: self._local_matches(store, term, canonical, scopes, local_index)
            for term in terms
        }
        local_state = self._local_state(terms, local_by_term)
        local_decision = self._local_resolution(terms, local_state)
        if local_decision is not None:
            return local_decision

        network_allowed = _external_search_allowed(mentions) or local_state.has_matches
        if not network_allowed:
            return self._batch(
                "unresolved",
                terms,
                notice="目录中没有精确匹配；请补充城市、区名或附近地标。",
            )

        geocoder_cities: tuple[_ExactCity, ...] = ()
        geocoder_sources: tuple[Mapping[str, Any], ...] = ()
        foreign: tuple[str, ...] = ()
        geocoder_failure: str | None = None
        if between_local_and_network is not None:
            between_local_and_network()
        try:
            hits = tuple(self.geocoder.search(GeocoderRequest(
                terms=terms,
                quote=" ".join(
                    str(getattr(mention, "quote", "") or "")[:200]
                    for mention in mentions[:2]
                    if str(getattr(mention, "quote", "") or "").strip()
                ),
                country_iso=canonical or None,
                source_isos=scopes,
                limit=5,
            )))
        except GeocodingUnavailable as exc:
            hits = ()
            geocoder_failure = _unavailable_notice(exc)
        except Exception as exc:
            hits = ()
            logger.warning("Place geocoder unavailable (%s)", type(exc).__name__)
            geocoder_failure = "地名检索暂时不可用。文字线索已保留，未自动选择城市。"
        if hits:
            geocoder_cities, geocoder_sources, foreign = self._map_geocoder_hits(
                store,
                hits,
                canonical=canonical,
                scopes=scopes,
                local_index=local_index,
            )

        local_needs_clarification = local_state.has_matches and (
            local_state.ambiguous or local_state.conflict
        )
        geocoder_unique_clean = (
            len(geocoder_cities) == 1
            and not local_needs_clarification
            and not foreign
        )
        if geocoder_unique_clean:
            candidate_cities = self._geocoder_candidate_evidence(geocoder_cities)
            local_sources = tuple(
                source for city in local_state.union for source in city.sources
            )
            return self._candidate_batch(
                terms,
                candidate_cities,
                extra_city_ids=tuple(city.row.city_id for city in local_state.union),
                sources=_unique_sources((
                    *local_sources,
                    *geocoder_sources,
                    *(source for item in candidate_cities for source in item.sources),
                )),
                notice=self._candidate_notice(
                    local_state,
                    geocoder_failure=geocoder_failure,
                    web_status="skipped",
                    web_result=None,
                    web_failure=None,
                    foreign_country_isos=foreign,
                ),
            )

        web_result: object | None = None
        web_failure: str | None = None
        if between_local_and_network is not None:
            between_local_and_network()
        try:
            from .web_place_search import get_web_place_search

            web_result = get_web_place_search().search(
                terms,
                country_iso=canonical or None,
            )
        except Exception as exc:
            web_failure = "网页检索暂时不可用。"
            logger.warning("Place web search unavailable (%s)", type(exc).__name__)
        web_status = str(getattr(web_result, "status", "") or "").strip().lower()
        if web_status not in {"disabled", "unavailable", "ok", "empty"}:
            web_status = "unavailable" if web_failure else "disabled"

        candidate_cities = self._merge_candidate_evidence((
            *self._geocoder_candidate_evidence(geocoder_cities),
            *self._web_candidate_evidence(
                store,
                web_result,
                canonical=canonical,
                scopes=scopes,
                local_index=local_index,
            ),
        ))
        if candidate_cities:
            extra_ids = tuple(city.row.city_id for city in local_state.union)
            notice = self._candidate_notice(
                local_state,
                geocoder_failure=geocoder_failure,
                web_status=web_status,
                web_result=web_result,
                web_failure=web_failure,
                foreign_country_isos=foreign,
            )
            local_sources = tuple(
                source for city in local_state.union for source in city.sources
            )
            return self._candidate_batch(
                terms,
                candidate_cities,
                extra_city_ids=extra_ids,
                sources=_unique_sources((
                    *local_sources,
                    *geocoder_sources,
                    *(source for item in candidate_cities for source in item.sources),
                )),
                foreign_country_isos=foreign,
                notice=notice,
            )

        if local_state.has_matches:
            cities = _unique_cities((
                *local_state.union,
                *geocoder_cities,
            ))
            notice = self._local_notice(local_state)
            if foreign:
                notice = (
                    f"{notice} 地名检索同时命中当前国家范围外的结果，"
                    "已保留冲突诊断但未采用。"
                )
            failures = [value for value in (
                geocoder_failure,
                self._web_failure_notice(web_status, web_result, web_failure),
            ) if value]
            if failures:
                notice = f"{notice} {' '.join(failures)}"
            local_sources = tuple(
                source for city in local_state.union for source in city.sources
            )
            return self._ambiguous_batch(
                terms,
                cities,
                provider="local_catalog",
                sources=_unique_sources((*local_sources, *geocoder_sources)),
                foreign_country_isos=foreign,
                notice=notice,
            )

        if foreign:
            return self._batch(
                "foreign_conflict",
                terms,
                sources=geocoder_sources,
                foreign_country_isos=foreign,
                notice="地名检索结果与当前国家不一致；外部结果只作候选，未自动改变路径。",
            )
        if geocoder_failure or web_status == "unavailable":
            notice = geocoder_failure or self._web_failure_notice(
                web_status, web_result, web_failure
            ) or "地点检索暂时不可用。"
            return self._batch("unavailable", terms, notice=notice)
        return self._batch(
            "unresolved",
            terms,
            notice="未找到可核验的城市结果；外部网页结果也不会自动选择城市。",
        )

    # ---------------------------------------------------------------- local
    def _exact_city(
        self,
        place: Any,
        term: str,
        canonical_country: str,
        source_isos: Sequence[str],
        *,
        extra_names: Sequence[str] = (),
    ) -> _ExactCity | None:
        row = _city_row(place)
        if row is None:
            return None
        if source_isos and row.iso not in source_isos:
            return None
        names = _place_names(place, extra_names)
        normalized = _normalize_text(term)
        matched_name = next(
            (name for name in names if _normalize_text(name) == normalized),
            "",
        )
        if not normalized or not matched_name:
            return None
        country = canonical_country_iso(row.iso)
        if canonical_country and country != canonical_country:
            return None
        source = _source_ref(
            provider="local_catalog",
            provider_place_id=str(row.city_id),
            label=matched_name,
            query=term,
            source_type="local_catalog",
            license_name="catalog",
            title=f"本地城市目录：{matched_name}",
            fetched_at=self._now().isoformat(),
        )
        return _ExactCity(
            row=row,
            canonical_country=country,
            names=names,
            match_kind="catalog_exact",
            confidence=1.0,
            source_type="local_catalog",
            sources=(source,),
        )

    def _local_matches(
        self,
        store: "GeoCatalogStore",
        term: str,
        canonical_country: str,
        source_isos: Sequence[str],
        local_index: LocalPlaceIndex,
    ) -> tuple[_ExactCity, ...]:
        found: dict[tuple[str, int], _ExactCity] = {}

        def add(city: _ExactCity | None) -> None:
            if city is None:
                return
            key = _city_key(city)
            existing = found.get(key)
            found[key] = city if existing is None else _merge_exact_city(existing, city)

        label_ids = city_ids_for_name_zh(term)
        label_places: Sequence[Any] = ()
        get_places = getattr(store, "get_places", None)
        if label_ids and callable(get_places):
            label_places = get_places("city", label_ids)
        elif label_ids:
            get_place = getattr(store, "get_place", None)
            if callable(get_place):
                label_places = tuple(
                    place for place in (get_place("city", city_id) for city_id in label_ids)
                    if place is not None
                )
        for place in label_places:
            add(self._exact_city(
                place,
                term,
                canonical_country,
                source_isos,
                extra_names=(term,),
            ))

        for match in local_index.lookup(term, source_isos=source_isos):
            place = store.get_place("city", str(match.city_id))
            if place is None:
                continue
            row = _city_row(place)
            if row is None or row.iso != match.source_iso:
                continue
            if source_isos and row.iso not in source_isos:
                continue
            country = canonical_country_iso(row.iso)
            if canonical_country and country != canonical_country:
                continue
            add(_ExactCity(
                row=row,
                canonical_country=country,
                names=_place_names(place, (match.name,)),
                match_kind=match.match_kind,
                confidence=match.confidence,
                source_type=match.source_type,
                sources=match.sources,
            ))

        queries = list(source_isos) if source_isos else [None]
        for source_iso in queries:
            page = store.query_places(
                "city",
                country=canonical_country or None,
                source_iso=source_iso,
                q=term,
                limit=_MAX_EXACT_QUERY_LIMIT,
                coordinates_only=True,
            )
            for place in page.get("rows") or []:
                add(self._exact_city(place, term, canonical_country, source_isos))
        return _unique_cities(tuple(found.values()))

    def _local_state(
        self,
        terms: tuple[str, ...],
        local_by_term: Mapping[str, tuple[_ExactCity, ...]],
    ) -> _LocalState:
        nonempty = tuple(
            tuple(local_by_term.get(term, ()))
            for term in terms
            if local_by_term.get(term)
        )
        if not nonempty:
            return _LocalState((), (), (), False, False, False)
        all_terms_matched = all(bool(local_by_term.get(term)) for term in terms)
        keysets = [set(_city_key(city) for city in cities) for cities in nonempty]
        intersection_keys = set.intersection(*keysets) if keysets else set()
        union_cities = _unique_cities(tuple(city for cities in nonempty for city in cities))
        intersection = tuple(
            city for city in union_cities if _city_key(city) in intersection_keys
        )
        conflict = all_terms_matched and len(nonempty) > 1 and not intersection
        ambiguous = not all_terms_matched or len(intersection) > 1
        return _LocalState(
            nonempty=nonempty,
            union=union_cities,
            intersection=intersection,
            all_terms_matched=all_terms_matched,
            ambiguous=ambiguous,
            conflict=conflict,
        )

    def _local_resolution(
        self,
        terms: tuple[str, ...],
        state: _LocalState,
    ) -> PlaceResolutionBatch | None:
        if state.all_terms_matched and len(state.intersection) == 1:
            return self._resolved_batch(terms, state.intersection)
        if len(terms) == 1 and len(state.nonempty) == 1 and len(state.nonempty[0]) == 1:
            return self._resolved_batch(terms, state.nonempty[0])
        return None

    @staticmethod
    def _local_notice(state: _LocalState) -> str:
        if state.conflict:
            return (
                "多个本地文字线索指向不同目录城市；全部原线索与冲突均保留，"
                "联网结果只生成候选，不会自动选择。"
            )
        if state.ambiguous:
            return (
                "本地目录命中不唯一或仍有未匹配项；全部本地候选保留，"
                "联网结果只生成候选，不会自动选择。"
            )
        return "本地目录未形成唯一且无冲突的城市结果。"


    # ------------------------------------------------------------- external
    def _map_geocoder_hits(
        self,
        store: "GeoCatalogStore",
        hits: Sequence[GeocoderHit],
        *,
        canonical: str,
        scopes: Sequence[str],
        local_index: LocalPlaceIndex,
    ) -> tuple[tuple[_ExactCity, ...], tuple[Mapping[str, Any], ...], tuple[str, ...]]:
        by_city: dict[tuple[str, int], _ExactCity] = {}
        sources: list[Mapping[str, Any]] = []
        foreign: set[str] = set()
        known_sources: set[tuple[str, str]] = set()
        for hit in hits[:5]:
            source = _source_ref(
                provider=hit.provider,
                provider_place_id=hit.provider_place_id,
                label=hit.display_name or hit.name,
                query=hit.query,
                source_type="geocoder",
                attribution=hit.attribution,
                license_name=hit.license,
                title=hit.display_name or hit.name,
                excerpt=hit.display_name or hit.name,
                fetched_at=self._now().isoformat(),
            )
            source_key = (hit.provider, str(hit.provider_place_id or ""))
            if source_key not in known_sources:
                sources.append(source)
                known_sources.add(source_key)
            hit_country = canonical_country_iso(hit.country_code or "")
            if canonical and hit_country and hit_country != canonical:
                foreign.add(hit_country)
                continue
            country = canonical or hit_country
            if not country:
                continue
            for locality in hit.locality_names:
                for city in self._local_matches(
                    store, locality, country, scopes, local_index
                ):
                    candidate = _ExactCity(
                        row=city.row,
                        canonical_country=city.canonical_country,
                        names=city.names,
                        match_kind="geocoder_candidate",
                        confidence=0.0,
                        source_type="geocoder",
                        sources=_unique_sources((*city.sources, source)),
                    )
                    key = _city_key(candidate)
                    existing = by_city.get(key)
                    by_city[key] = (
                        candidate if existing is None
                        else _merge_exact_city(existing, candidate)
                    )
        return (
            tuple(by_city.values()),
            _unique_sources(sources),
            tuple(sorted(foreign)),
        )

    def _geocoder_candidate_evidence(
        self,
        cities: Sequence[_ExactCity],
    ) -> tuple[CityEvidence, ...]:
        return tuple(
            CityEvidence(
                city_id=city.row.city_id,
                source_iso=city.row.iso,
                text=next(iter(city.names), city.row.name),
                confidence=0.0,
                match_kind="geocoder_candidate",
                sources=_unique_sources(city.sources),
                source_type="geocoder",
            )
            for city in cities
        )

    def _web_candidate_evidence(
        self,
        store: "GeoCatalogStore",
        web_result: object | None,
        *,
        canonical: str,
        scopes: Sequence[str],
        local_index: LocalPlaceIndex,
    ) -> tuple[CityEvidence, ...]:
        if web_result is None:
            return ()
        candidates = candidates_from_web_result(
            store,
            web_result,
            country_iso=canonical,
            source_isos=scopes,
            local_index=local_index,
        )
        return tuple(
            CityEvidence(
                city_id=candidate.city_id,
                source_iso=candidate.source_iso,
                text=candidate.name,
                confidence=0.0,
                match_kind="web_candidate",
                sources=_unique_sources(candidate.sources),
                source_type="web_search",
            )
            for candidate in candidates
        )

    @staticmethod
    def _merge_candidate_evidence(
        candidates: Sequence[CityEvidence],
    ) -> tuple[CityEvidence, ...]:
        merged: dict[str, CityEvidence] = {}
        for item in candidates:
            if item.confidence != 0.0 or item.match_kind not in {
                "geocoder_candidate", "web_candidate"
            }:
                continue
            key = str(item.city_id)
            existing = merged.get(key)
            if existing is None:
                merged[key] = item
                continue
            match_kind = (
                "geocoder_candidate"
                if "geocoder_candidate" in {existing.match_kind, item.match_kind}
                else "web_candidate"
            )
            source_type = (
                "geocoder" if match_kind == "geocoder_candidate" else "web_search"
            )
            merged[key] = CityEvidence(
                city_id=item.city_id,
                source_iso=item.source_iso,
                text=existing.text or item.text,
                confidence=0.0,
                match_kind=match_kind,
                sources=_unique_sources((*existing.sources, *item.sources)),
                source_type=source_type,
            )
        return tuple(merged.values())

    @staticmethod
    def _web_failure_notice(
        web_status: str,
        web_result: object | None,
        web_failure: str | None,
    ) -> str | None:
        if web_status != "unavailable":
            return None
        notice = str(getattr(web_result, "notice", "") or "").strip()
        return notice or web_failure or "网页检索暂时不可用。"

    def _candidate_notice(
        self,
        local_state: _LocalState,
        *,
        geocoder_failure: str | None,
        web_status: str,
        web_result: object | None,
        web_failure: str | None,
        foreign_country_isos: Sequence[str],
    ) -> str:
        parts: list[str] = []
        if local_state.has_matches:
            parts.append(self._local_notice(local_state))
        if foreign_country_isos:
            parts.append("检索结果与当前国家范围存在冲突；冲突国家仅保留在诊断中，未采用。")
        parts.append("联网结果仅用于召回，不直接增加地点匹配分，也不会确认城市。")
        failure = self._web_failure_notice(web_status, web_result, web_failure)
        if geocoder_failure:
            parts.append(geocoder_failure)
        if failure:
            parts.append(failure)
        return " ".join(parts)

    # -------------------------------------------------------------- batches
    def _resolved_batch(
        self,
        terms: tuple[str, ...],
        cities: Sequence[_ExactCity],
    ) -> PlaceResolutionBatch:
        if len(cities) != 1:
            return self._ambiguous_batch(terms, cities, provider="catalog")
        now = self._now()
        city = cities[0]
        source_list = _unique_sources(city.sources) or (
            _source_ref(
                provider="local_catalog",
                provider_place_id=str(city.row.city_id),
                label=city.names[0] if city.names else city.row.name,
                query=" ".join(terms),
                source_type="local_catalog",
                license_name="catalog",
                title=f"本地城市目录：{city.names[0] if city.names else city.row.name}",
                fetched_at=now.isoformat(),
            ),
        )
        evidence = CityEvidence(
            city_id=city.row.city_id,
            source_iso=city.row.iso,
            text=" ".join(terms),
            confidence=city.confidence,
            match_kind=city.match_kind,
            sources=source_list,
            source_type=city.source_type,
        )
        return PlaceResolutionBatch(
            version=_RESOLUTION_VERSION,
            status="resolved",
            query_terms=terms,
            city_evidence=(evidence,),
            city_ids=(city.row.city_id,),
            sources=source_list,
            fetched_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=self.cache_ttl_seconds)).isoformat(),
            notice=None,
        )

    def _ambiguous_batch(
        self,
        terms: tuple[str, ...],
        cities: Sequence[_ExactCity],
        *,
        provider: str,
        sources: Sequence[Mapping[str, Any]] = (),
        foreign_country_isos: Sequence[str] = (),
        notice: str | None = None,
    ) -> PlaceResolutionBatch:
        now = self._now()
        unique = {_city_key(city): city for city in cities}
        source_list = _unique_sources(sources)
        if not source_list and unique:
            source_list = (
                _source_ref(
                    provider=provider,
                    provider_place_id=None,
                    label=" / ".join(
                        next(iter(item.names), item.row.name) for item in unique.values()
                    )[:200],
                    query=" ".join(terms),
                    source_type="local_catalog" if provider == "local_catalog" else "geocoder",
                    license_name="catalog" if provider == "local_catalog" else None,
                    title="本地目录歧义" if provider == "local_catalog" else "外部候选",
                    fetched_at=now.isoformat(),
                ),
            )
        return PlaceResolutionBatch(
            version=_RESOLUTION_VERSION,
            status="ambiguous",
            query_terms=terms,
            city_ids=tuple(city.row.city_id for city in unique.values()),
            sources=source_list,
            foreign_country_isos=tuple(foreign_country_isos),
            notice=notice or "命中的地点不唯一；请补充城市、区名或附近地标，不会自动选择。",
            fetched_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=self.cache_ttl_seconds)).isoformat(),
        )

    def _candidate_batch(
        self,
        terms: tuple[str, ...],
        candidates: Sequence[CityEvidence],
        *,
        extra_city_ids: Sequence[int] = (),
        sources: Sequence[Mapping[str, Any]] = (),
        foreign_country_isos: Sequence[str] = (),
        notice: str | None = None,
    ) -> PlaceResolutionBatch:
        now = self._now()
        candidate_list = tuple(candidates)
        city_ids = tuple(dict.fromkeys((
            *(item.city_id for item in candidate_list),
            *extra_city_ids,
        )))
        return PlaceResolutionBatch(
            version=_RESOLUTION_VERSION,
            status="candidates",
            query_terms=terms,
            city_evidence=(),
            city_ids=city_ids,
            sources=_unique_sources(sources),
            foreign_country_isos=tuple(foreign_country_isos),
            notice=notice or "联网结果仅用于召回，不直接增加地点匹配分，也不会确认城市。",
            fetched_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=self.cache_ttl_seconds)).isoformat(),
            candidate_cities=candidate_list,
        )

    def _batch(
        self,
        status: str,
        terms: tuple[str, ...],
        *,
        sources: Sequence[Mapping[str, Any]] = (),
        foreign_country_isos: Sequence[str] = (),
        notice: str | None = None,
        detail: str | None = None,
    ) -> PlaceResolutionBatch:
        now = self._now()
        suffix = f"（{detail}）" if detail else ""
        ttl = (
            self.negative_cache_ttl_seconds
            if status in {"unresolved", "unavailable", "foreign_conflict"}
            else self.cache_ttl_seconds
        )
        return PlaceResolutionBatch(
            version=_RESOLUTION_VERSION,
            status=status,
            query_terms=terms,
            sources=_unique_sources(sources),
            foreign_country_isos=tuple(foreign_country_isos),
            notice=f"{notice}{suffix}" if notice else notice,
            fetched_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=ttl)).isoformat(),
        )

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)


__all__ = [
    "CityEvidence",
    "MentionLike",
    "PlaceResolutionBatch",
    "PlaceResolutionService",
    "ResolutionError",
]
