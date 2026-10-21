"""Turn literal user place text into catalog city evidence.

The resolver never accepts an external provider's identity or coordinates as a
map target. It first performs exact matching against the authoritative scoped
city catalog and the local Chinese-name stable-ID index, then (only when
necessary) asks the configured bounded provider for locality names and maps
those names back to catalog rows. Ambiguity is a first-class result and is never
resolved by population, provider rank or nearest-city guesses.
"""
from __future__ import annotations

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

_RESOLUTION_VERSION = 2
_MAX_TERMS = 4
_MAX_EXACT_QUERY_LIMIT = 2000
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
        # LLM terms and explicitly adopted OCR text have already passed literal
        # validation. They may be POIs with no generic road/store suffix.
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
    """One catalog-resolved city and the user-text sources supporting it."""

    city_id: int
    source_iso: str
    text: str
    confidence: float
    match_kind: str
    clue_ids: tuple[str, ...] = ()
    sources: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class PlaceResolutionBatch:
    """Bounded result of resolving one user message's literal place terms."""

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


def _normalize_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in text if character.isalnum())


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
    return f"{summary}。文字线索已保留，请补充城市、区名或附近地标。"


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
    provider_place_id: str | None = None,
    attribution: str | None = None,
    license_name: str | None = None,
    fetched_at: str,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "provider_place_id": provider_place_id,
        "label": label[:200],
        "query": query[:200],
        "attribution": attribution,
        "license": license_name,
        "fetched_at": fetched_at,
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
    """Local catalog exact matching followed by one optional external search."""

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
        local_by_term: dict[str, tuple[_ExactCity, ...]] = {}
        for term in terms:
            local_by_term[term] = self._local_matches(store, term, canonical, scopes)

        local_decision = self._local_decision(terms, local_by_term)
        if local_decision is not None:
            return local_decision

        if not _external_search_allowed(mentions):
            return self._batch(
                "unresolved",
                terms,
                notice="目录中没有精确匹配；请补充城市、区名或附近地标。",
            )

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
            return self._batch(
                "unavailable",
                terms,
                notice=_unavailable_notice(exc),
            )
        except Exception:
            return self._batch(
                "unavailable",
                terms,
                notice="地名检索暂时不可用。文字线索已保留，请补充城市、区名或附近地标。",
            )
        if not hits:
            return self._batch(
                "unresolved",
                terms,
                notice="未找到可核验的城市结果；请补充城市、区名或附近地标。",
            )
        return self._map_external_hits(
            store,
            terms,
            hits,
            canonical=canonical,
            scopes=scopes,
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
        if not normalized or not any(
            _normalize_text(name) == normalized for name in names
        ):
            return None
        country = canonical_country_iso(row.iso)
        if canonical_country and country != canonical_country:
            return None
        return _ExactCity(
            row=row,
            canonical_country=country,
            names=names,
        )

    def _local_matches(
        self,
        store: "GeoCatalogStore",
        term: str,
        canonical_country: str,
        source_isos: Sequence[str],
    ) -> tuple[_ExactCity, ...]:
        found: dict[tuple[str, int], _ExactCity] = {}

        # Local Chinese display names are stable-ID aliases, but they are not
        # duplicated into GeoPlace. Resolve the IDs through the same scoped
        # catalog and keep every duplicate so country filtering and ambiguity
        # remain explicit.
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
            city = self._exact_city(
                place,
                term,
                canonical_country,
                source_isos,
                extra_names=(term,),
            )
            if city is not None:
                found.setdefault((city.row.iso, city.row.city_id), city)

        # Keep the existing catalog-name/alias path and use it as the fallback
        # when a label ID is absent from the current catalog snapshot.
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
                city = self._exact_city(
                    place, term, canonical_country, source_isos
                )
                if city is not None:
                    found.setdefault((city.row.iso, city.row.city_id), city)
        return tuple(found.values())

    def _local_decision(
        self,
        terms: tuple[str, ...],
        local_by_term: Mapping[str, tuple[_ExactCity, ...]],
    ) -> PlaceResolutionBatch | None:
        nonempty = [set(item) for item in local_by_term.values() if item]
        if not nonempty:
            return None
        all_terms_matched = all(bool(local_by_term.get(term)) for term in terms)
        if not all_terms_matched and len(terms) > 1:
            return None
        intersection = set.intersection(*nonempty) if nonempty else set()
        if all_terms_matched and len(intersection) == 1:
            return self._resolved_batch(terms, tuple(intersection), match_kind="catalog_exact")
        if all_terms_matched and len(intersection) > 1:
            return self._ambiguous_batch(terms, tuple(intersection), provider="local_catalog")
        if len(terms) == 1 and len(nonempty[0]) == 1:
            return self._resolved_batch(terms, tuple(nonempty[0]), match_kind="catalog_exact")
        if len(terms) == 1 and len(nonempty[0]) > 1:
            return self._ambiguous_batch(terms, tuple(nonempty[0]), provider="local_catalog")
        if all_terms_matched and not intersection:
            return self._ambiguous_batch(
                terms,
                tuple(item for values in nonempty for item in values),
                provider="local_catalog",
                notice="多个文字线索指向不同城市；请补充城市或区名后再确认。",
            )
        return None

    # ------------------------------------------------------------- external
    def _map_external_hits(
        self,
        store: "GeoCatalogStore",
        terms: tuple[str, ...],
        hits: Sequence[GeocoderHit],
        *,
        canonical: str,
        scopes: Sequence[str],
    ) -> PlaceResolutionBatch:
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
                attribution=hit.attribution,
                license_name=hit.license,
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
                for city in self._local_matches(store, locality, country, scopes):
                    by_city[(city.row.iso, city.row.city_id)] = city

        if foreign and not by_city:
            return self._batch(
                "foreign_conflict",
                terms,
                sources=tuple(sources),
                foreign_country_isos=tuple(sorted(foreign)),
                notice="地名检索结果与当前国家不一致；请先回退国家阶段再选择。",
            )
        if not by_city:
            return self._batch(
                "unresolved",
                terms,
                sources=tuple(sources),
                foreign_country_isos=tuple(sorted(foreign)),
                notice="地名结果没有可核验的城市；请补充城市、区名或附近地标。",
            )
        if foreign:
            return self._ambiguous_batch(
                terms,
                tuple(by_city.values()),
                provider=hits[0].provider if hits else "external",
                sources=tuple(sources),
                foreign_country_isos=tuple(sorted(foreign)),
                notice="同名地点跨越多个国家；请先明确国家，不会自动选择。",
            )
        if len(by_city) == 1:
            return self._resolved_batch(
                terms,
                tuple(by_city.values()),
                match_kind="geocoder_locality",
                sources=tuple(sources),
            )
        return self._ambiguous_batch(
            terms,
            tuple(by_city.values()),
            provider=hits[0].provider if hits else "external",
            sources=tuple(sources),
            notice="地名检索命中多个城市；请补充城市、区名或附近地标。",
        )

    # -------------------------------------------------------------- batches
    def _resolved_batch(
        self,
        terms: tuple[str, ...],
        cities: Sequence[_ExactCity],
        *,
        match_kind: str,
        sources: Sequence[Mapping[str, Any]] = (),
    ) -> PlaceResolutionBatch:
        if len(cities) != 1:
            return self._ambiguous_batch(terms, cities, provider="catalog")
        now = self._now()
        city = cities[0]
        source_list = tuple(sources) or (
            _source_ref(
                provider="local_catalog",
                provider_place_id=str(city.row.city_id),
                label=city.names[0] if city.names else city.row.name,
                query=" ".join(terms),
                license_name="catalog",
                fetched_at=now.isoformat(),
            ),
        )
        evidence = CityEvidence(
            city_id=city.row.city_id,
            source_iso=city.row.iso,
            text=" ".join(terms),
            confidence=1.0 if match_kind == "catalog_exact" else 0.8,
            match_kind=match_kind,
            sources=source_list,
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
        unique: dict[tuple[str, int], _ExactCity] = {
            (city.row.iso, city.row.city_id): city for city in cities
        }
        source_list = tuple(sources) or (
            _source_ref(
                provider=provider,
                provider_place_id=None,
                label=" / ".join(
                    next(iter(item.names), item.row.name) for item in unique.values()
                )[:200],
                query=" ".join(terms),
                license_name="catalog" if provider == "local_catalog" else None,
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
            notice=notice or "命中的地点不唯一；请补充城市、区名或附近地标，不会按人口自动选择。",
            fetched_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=self.cache_ttl_seconds)).isoformat(),
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
            sources=tuple(sources),
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
