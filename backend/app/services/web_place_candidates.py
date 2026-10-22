"""Literal web-snippet to authoritative catalog-city candidates.

Web results are recall only: this module never trusts provider coordinates,
place IDs or inferred names.  It matches only catalog names, local Chinese
labels, catalog aliases and imported city aliases that appear literally in
bounded titles/excerpts.  Candidate confidence is always zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from ..domain.geolocation.country_policy import normalize_iso
from ..domain.geolocation.engine import find_place_phrase, normalize_place_text
from .city_labels import city_name_zh
from .local_place_index import LocalPlaceIndex

MAX_WEB_HITS = 5
MAX_WEB_CANDIDATES = 8
CATALOG_PAGE_SIZE = 2000
MAX_CATALOG_CITY_SCAN = 50000
_GENERIC_CJK_NAMES = frozenset({
    "中国", "日本", "城市", "中心", "广场", "新区", "地区",
})
_CJK_CONTEXT_MARKERS = (
    "坐落于", "位于", "地处", "地址", "城市", "来自", "属于", "在", "于", "到",
)
_CJK_ADMIN_FOLLOWERS = (
    "市", "区", "县", "省", "州", "镇", "乡", "都", "的", "有",
)
_CJK_NON_CITY_FOLLOWERS = (
    "大道", "大街", "胡同", "广场", "公园", "车站", "机场", "码头",
    "分店", "酒店", "宾馆", "餐厅", "医院", "学校", "大学", "市场", "商场",
    "中心", "大厦", "路", "街", "巷", "弄", "站", "桥", "寺", "祠", "店", "馆", "楼",
)


@dataclass(frozen=True)
class WebCityCandidate:
    city_id: int
    source_iso: str
    name: str
    sources: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class _CityName:
    key: str
    city_id: int
    source_iso: str
    display: str
    cjk: bool


def _bounded_text(value: object, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _valid_url(value: object) -> str:
    text = _bounded_text(value, 1000)
    if (
        not text
        or any(character.isspace() for character in text)
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
    ):
        return ""
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError:
        return ""
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port is not None and not (1 <= port <= 65535)
    ):
        return ""
    return text


def _usable_name(value: object) -> tuple[str, str, bool]:
    text = _bounded_text(value, 120)
    key = normalize_place_text(text)
    if not key:
        return "", "", False
    compact = key.replace(" ", "")
    cjk = any("\u3400" <= character <= "\u9fff" for character in text)
    if key.isascii():
        if len(compact) < 4:
            return "", "", False
    elif cjk:
        if len(compact) < 2 or text in _GENERIC_CJK_NAMES:
            return "", "", False
    elif len(compact) < 3:
        return "", "", False
    return key, text, cjk


def _iter_catalog_cities(
    store: Any,
    *,
    country_iso: str,
    source_isos: Sequence[str],
):
    scopes = tuple(dict.fromkeys(
        normalized for normalized in (normalize_iso(item) for item in source_isos)
        if normalized
    ))
    queries: tuple[str | None, ...] = scopes or (None,)
    seen = 0
    for source_iso in queries:
        offset = 0
        while seen < MAX_CATALOG_CITY_SCAN:
            page = store.query_places(
                "city",
                country=country_iso or None,
                source_iso=source_iso,
                limit=min(CATALOG_PAGE_SIZE, MAX_CATALOG_CITY_SCAN - seen),
                offset=offset,
                coordinates_only=True,
            )
            rows = list(page.get("rows") or [])
            if not rows:
                break
            for place in rows:
                seen += 1
                yield place
                if seen >= MAX_CATALOG_CITY_SCAN:
                    break
            total = int(page.get("total") or 0)
            offset += len(rows)
            if offset >= total:
                break


def _city_names(
    store: Any,
    *,
    country_iso: str,
    source_isos: Sequence[str],
    local_index: LocalPlaceIndex,
) -> tuple[_CityName, ...]:
    result: list[_CityName] = []
    seen: set[tuple[str, str, int]] = set()
    for place in _iter_catalog_cities(
        store, country_iso=country_iso, source_isos=source_isos
    ):
        try:
            city_id = int(str(getattr(place, "place_id", "")))
        except (TypeError, ValueError):
            continue
        iso = normalize_iso(getattr(place, "source_iso", ""))
        if not iso:
            continue
        display = city_name_zh(city_id, str(getattr(place, "name", "") or ""))
        aliases = getattr(place, "aliases", ()) or ()
        if isinstance(aliases, str):
            aliases = (aliases,)
        elif not isinstance(aliases, (list, tuple, set)):
            aliases = ()
        raw_names = (
            display,
            str(getattr(place, "name", "") or ""),
            str(getattr(place, "ascii_name", "") or ""),
            str(getattr(place, "name_zh", "") or ""),
            *(str(item) for item in aliases),
            *local_index.city_aliases_for_city(city_id, iso),
        )
        for raw_name in raw_names:
            key, shown, cjk = _usable_name(raw_name)
            if not key:
                continue
            dedupe_key = (key, iso, city_id)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            result.append(_CityName(
                key=key,
                city_id=city_id,
                source_iso=iso,
                display=shown,
                cjk=cjk,
            ))
    return tuple(result)


def _cjk_left_context(text: str, index: int) -> tuple[bool, bool]:
    if index == 0 or not text[index - 1].isalnum():
        return True, False
    prefix = text[max(0, index - 12):index]
    if any(prefix.endswith(marker) for marker in _CJK_CONTEXT_MARKERS):
        return True, True
    if text[index - 1] in "省市县区州国":
        return True, True
    return False, False


def _cjk_match_allowed(text: str, key: str, index: int) -> bool:
    end = index + len(key)
    left_allowed, explicit_context = _cjk_left_context(text, index)
    if not left_allowed:
        return False
    remainder = text[end:]
    if any(remainder.startswith(suffix) for suffix in _CJK_NON_CITY_FOLLOWERS):
        return False
    if explicit_context:
        return True
    if not remainder or not remainder[0].isalnum():
        return True
    return any(remainder.startswith(suffix) for suffix in _CJK_ADMIN_FOLLOWERS)


def _literal_city_matches(text: str, names: Sequence[_CityName]) -> tuple[_CityName, ...]:
    occurrences: list[tuple[_CityName, int]] = []
    for name in names:
        if name.cjk:
            start = 0
            while True:
                index = text.find(name.key, start)
                if index < 0:
                    break
                if _cjk_match_allowed(text, name.key, index):
                    occurrences.append((name, index))
                start = index + 1
        elif find_place_phrase(text, name.key):
            index = text.find(name.key)
            if index >= 0:
                occurrences.append((name, index))

    occurrences.sort(key=lambda item: (
        -len(item[0].key), item[1], item[0].city_id, item[0].source_iso,
    ))
    occupied: list[tuple[int, int, str]] = []
    accepted: list[_CityName] = []
    for name, index in occurrences:
        end = index + len(name.key)
        blocked = False
        for start, stop, key in occupied:
            if index < stop and start < end:
                if not (start == index and stop == end and key == name.key):
                    blocked = True
                    break
        if blocked:
            continue
        occupied.append((index, end, name.key))
        accepted.append(name)
    return tuple(accepted)


def _hit_source(hit: Any) -> Mapping[str, Any] | None:
    title = _bounded_text(getattr(hit, "title", ""), 200)
    url = _valid_url(getattr(hit, "url", ""))
    if not title or not url:
        return None
    excerpt = _bounded_text(getattr(hit, "excerpt", ""), 300)
    provider = _bounded_text(getattr(hit, "provider", ""), 40) or "web_search"
    query = _bounded_text(getattr(hit, "query", ""), 200)
    fetched_at = _bounded_text(getattr(hit, "fetched_at", ""), 80)
    return {
        "provider": provider,
        "provider_place_id": None,
        "label": title,
        "query": query,
        "attribution": f"Web search: {provider}"[:200],
        "license": None,
        "fetched_at": fetched_at,
        "source_type": "web_search",
        "title": title,
        "url": url,
        "excerpt": excerpt,
    }


def candidates_from_web_result(
    store: Any,
    result: object,
    *,
    country_iso: str = "",
    source_isos: Sequence[str] = (),
    local_index: LocalPlaceIndex,
) -> tuple[WebCityCandidate, ...]:
    """Map literal web mentions to at most eight catalog cities."""

    if str(getattr(result, "status", "") or "").strip().lower() != "ok":
        return ()
    raw_hits = getattr(result, "hits", ()) or ()
    if not isinstance(raw_hits, (tuple, list)) or not raw_hits:
        return ()

    names = _city_names(
        store,
        country_iso=normalize_iso(country_iso),
        source_isos=source_isos,
        local_index=local_index,
    )
    if not names:
        return ()

    matched: dict[tuple[str, int], dict[str, Any]] = {}
    for hit_index, hit in enumerate(raw_hits[:MAX_WEB_HITS]):
        source = _hit_source(hit)
        if source is None:
            continue
        title = _bounded_text(getattr(hit, "title", ""), 400)
        excerpt = _bounded_text(getattr(hit, "excerpt", ""), 1200)
        normalized_text = normalize_place_text("\n".join(filter(None, (title, excerpt))))
        if not normalized_text:
            continue
        for name in _literal_city_matches(normalized_text, names):
            key = (name.source_iso, name.city_id)
            entry = matched.get(key)
            if entry is None:
                entry = {
                    "first_hit": hit_index,
                    "name": name.display,
                    "sources": [],
                }
                matched[key] = entry
            sources = entry["sources"]
            source_key = (
                str(source.get("source_type")),
                str(source.get("url") or ""),
                str(source.get("title") or ""),
            )
            if all(
                (
                    str(item.get("source_type")),
                    str(item.get("url") or ""),
                    str(item.get("title") or ""),
                ) != source_key
                for item in sources
            ):
                sources.append(dict(source))

    ordered = sorted(
        matched.items(),
        key=lambda item: (int(item[1]["first_hit"]), item[0][1]),
    )[:MAX_WEB_CANDIDATES]
    return tuple(
        WebCityCandidate(
            city_id=city_id,
            source_iso=iso,
            name=str(entry["name"]),
            sources=tuple(entry["sources"]),
        )
        for (iso, city_id), entry in ordered
    )


__all__ = [
    "MAX_WEB_CANDIDATES",
    "MAX_WEB_HITS",
    "WebCityCandidate",
    "candidates_from_web_result",
]
