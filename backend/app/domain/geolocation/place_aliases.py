"""Curated, catalog-validated place-name aliases.

This module is intentionally data-only: it performs no network, database, model,
or coordinate lookup.  A match is a proposal hint, never proof that a label is
globally unique and never an automatic location confirmation.

Flow callers must pass the matched entry through
:func:`validate_curated_place_alias` with their authoritative city index before
using it.  Missing rows, country/ISO mismatches, or name mismatches fail closed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple
import unicodedata


PLACE_ALIAS_VERSION = "2026-10-21.1"

_QUOTE_WRAPPERS = (
    ("\u201c", "\u201d"),
    ('"', '"'),
    ("'", "'"),
    ("\u300c", "\u300d"),
    ("\u300e", "\u300f"),
    ("\u2018", "\u2019"),
)


@dataclass(frozen=True)
class CuratedPlaceAliasSource:
    """One human-reviewed source supporting a curated association."""

    publisher: str
    title: str
    url: str
    verified_on: str


@dataclass(frozen=True)
class CuratedPlaceAlias:
    """One exact literal -> catalog city proposal.

    ``proposal_only`` and ``globally_unique=False`` are deliberate: the source
    material supports a Chengdu association for this use case, not a claim that
    every same-named road anywhere belongs to Chengdu.
    """

    kind: str
    aliases: Tuple[str, ...]
    city_id: str
    country_iso: str
    city_name: str
    display_label: str
    version: str
    sources: Tuple[CuratedPlaceAliasSource, ...] = ()
    proposal_only: bool = True
    globally_unique: bool = False


_YULIN_SOURCES = (
    CuratedPlaceAliasSource(
        publisher="四川省商务厅（商务部网站发布）",
        title="玉林街道：助力成都打造国际消费中心城市",
        url=(
            "https://cif.mofcom.gov.cn/newsite/html/sichuan/html/282519/"
            "2023/11/10/1699605482453.html"
        ),
        verified_on="2026-10-21",
    ),
    CuratedPlaceAliasSource(
        publisher="国家体育总局体育彩票管理中心",
        title="如果你要写成都",
        url="https://www.sport.gov.cn/cpzx/n5651/c959725/content.html",
        verified_on="2026-10-21",
    ),
)


CURATED_PLACE_ALIASES: Tuple[CuratedPlaceAlias, ...] = (
    CuratedPlaceAlias(
        kind="street_label",
        aliases=("玉林路", "玉林西路"),
        city_id="1815286",
        country_iso="CN",
        city_name="Chengdu",
        display_label="成都",
        version=PLACE_ALIAS_VERSION,
        sources=_YULIN_SOURCES,
    ),
    CuratedPlaceAlias(
        kind="city_name",
        aliases=("成都", "成都市"),
        city_id="1815286",
        country_iso="CN",
        city_name="Chengdu",
        display_label="成都",
        version=PLACE_ALIAS_VERSION,
    ),
)


def _strip_wrappers(value: object) -> str:
    text = str(value or "").strip()
    for opening, closing in _QUOTE_WRAPPERS:
        if len(text) >= 2 and text.startswith(opening) and text.endswith(closing):
            return text[len(opening):-len(closing)].strip()
    return text


def _normalize(value: object) -> str:
    """Exact-match normalization; punctuation/spacing do not create substrings."""

    text = unicodedata.normalize("NFKC", _strip_wrappers(value)).casefold()
    return "".join(ch for ch in text if ch.isalnum())


def match_curated_place_alias(literal: object) -> Optional[CuratedPlaceAlias]:
    """Return one exact curated alias match, or ``None``.

    Matching is whole-value equality after quote/whitespace normalization.  In
    particular, ``玉林`` and ``玉林路附近`` never match ``玉林路``.
    """

    normalized = _normalize(literal)
    if not normalized:
        return None
    for entry in CURATED_PLACE_ALIASES:
        if any(normalized == _normalize(alias) for alias in entry.aliases):
            return entry
    return None


def _row_value(row: Any, key: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(key)
    return getattr(row, key, None)


def validate_curated_place_alias(
    entry: CuratedPlaceAlias,
    city_index: Mapping[str, Mapping[Any, Any]],
) -> bool:
    """Return true only when the authoritative city row matches this entry.

    ``city_index`` is the existing ``{iso: {city_id: CityRow}}`` shape used by
    the city catalog.  No lookup fallback or population guess is performed.
    """

    if not isinstance(entry, CuratedPlaceAlias):
        return False
    country_iso = str(entry.country_iso or "").strip().upper()
    scope = city_index.get(country_iso) if isinstance(city_index, Mapping) else None
    if not isinstance(scope, Mapping):
        return False
    row = scope.get(int(entry.city_id)) if entry.city_id.isdigit() else None
    if row is None and entry.city_id in scope:
        row = scope.get(entry.city_id)
    if row is None:
        return False
    row_iso = _row_value(row, "iso") or _row_value(row, "country_iso")
    row_name = _row_value(row, "name")
    row_id = _row_value(row, "city_id") or _row_value(row, "place_id")
    return (
        str(row_iso or "").strip().upper() == country_iso
        and str(row_id or "") == str(entry.city_id)
        and _normalize(row_name) == _normalize(entry.city_name)
    )


def match_and_validate_curated_place_alias(
    literal: object,
    city_index: Mapping[str, Mapping[Any, Any]],
) -> Optional[CuratedPlaceAlias]:
    """Convenience helper for flow callers that must fail closed."""

    entry = match_curated_place_alias(literal)
    if entry is None:
        return None
    return entry if validate_curated_place_alias(entry, city_index) else None


__all__ = [
    "CURATED_PLACE_ALIASES",
    "PLACE_ALIAS_VERSION",
    "CuratedPlaceAlias",
    "CuratedPlaceAliasSource",
    "match_and_validate_curated_place_alias",
    "match_curated_place_alias",
    "validate_curated_place_alias",
]
