"""Application-level country grouping policy.

Raw source/catalog identities from the read-only SQLite catalog stay immutable:
``TW``, ``HK`` and ``MO`` remain valid two-letter source/catalog codes so
model revisions, reference sidecars and city-catalog validation continue to
work.  This module only maps those source scopes into the product country
hierarchy used by rankings, confirmations and city queries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

# Only these explicit source scopes are grouped.  The table is intentionally
# small: it is an app policy, not a general political-status adjudication.
REGION_PARENT_ISO: dict[str, str] = {
    "TW": "CN",
    "HK": "CN",
    "MO": "CN",
}

POLICY_VERSION = "china-subordinate-regions-v1"


def normalize_iso(value: object) -> str:
    """Return an uppercase two-letter ASCII code, or ``""`` when invalid."""
    if not isinstance(value, str):
        return ""
    candidate = value.strip().upper()
    if len(candidate) != 2 or not candidate.isascii() or not candidate.isalpha():
        return ""
    return candidate


def canonical_country_iso(value: object) -> str:
    """Map a raw source/catalog code to its product country code.

    Unknown and non-region codes are returned unchanged (normalized).  This is
    the helper dialogue/parsers should call before persisting a country-level
    confirmation.
    """
    code = normalize_iso(value)
    if not code:
        return ""
    return REGION_PARENT_ISO.get(code, code)


def source_region_iso(value: object) -> str:
    """Return the raw source code when ``value`` is a subordinate region."""
    code = normalize_iso(value)
    return code if code in REGION_PARENT_ISO else ""


def is_region_iso(value: object) -> bool:
    """True when ``value`` identifies a grouped raw region scope."""
    return bool(source_region_iso(value))


def country_members(value: object) -> tuple[str, ...]:
    """Return the raw source scopes for a product country, root first.

    ``country_members("CN") == ("CN", "HK", "MO", "TW")``.  A grouped source
    code resolves to the same tuple; unknown codes resolve to themselves.
    """
    canonical = canonical_country_iso(value)
    if not canonical:
        return ()
    children = tuple(
        sorted(code for code, parent in REGION_PARENT_ISO.items() if parent == canonical)
    )
    if canonical in children:
        return children
    return (canonical, *children)


def region_members(value: object) -> tuple[str, ...]:
    """Return only the subordinate raw scopes of a product country."""
    return tuple(member for member in country_members(value) if member != canonical_country_iso(value))


def is_country_member(country: object, source: object) -> bool:
    """True when ``source`` is an allowed raw scope for ``country``."""
    canonical = canonical_country_iso(country)
    source_code = normalize_iso(source)
    return bool(canonical and source_code and source_code in country_members(canonical))


@dataclass(frozen=True)
class ExclusionScope:
    """One persisted exclusion interpreted without changing its raw identity."""

    value: str
    kind: str  # "country" or "region"
    country_iso: str
    source_iso: str | None = None


def exclusion_scope(value: object) -> ExclusionScope | None:
    """Classify one persisted exclusion value.

    A product country code (for example ``CN``) excludes the whole grouped
    country.  A raw subordinate code (``TW``/``HK``/``MO``) excludes only that
    source region before grouping.  Other codes behave as country exclusions,
    preserving all existing non-policy behavior.
    """
    code = normalize_iso(value)
    if not code:
        return None
    parent = REGION_PARENT_ISO.get(code)
    if parent:
        return ExclusionScope(value=code, kind="region", country_iso=parent, source_iso=code)
    return ExclusionScope(value=code, kind="country", country_iso=code, source_iso=None)


def is_excluded(exclusions: Iterable[object] | None, source_iso: object) -> bool:
    """Return whether a raw source scope is removed by persisted exclusions.

    Country-level exclusions remove every member.  Region exclusions remove
    only their exact raw source scope, which keeps e.g. "not Taiwan, but
    mainland China" representable.
    """
    source = normalize_iso(source_iso)
    if not source:
        return False
    country = canonical_country_iso(source)
    for raw in exclusions or ():
        scope = exclusion_scope(raw)
        if scope is None:
            continue
        if scope.kind == "country" and scope.country_iso == country:
            return True
        if scope.kind == "region" and scope.source_iso == source:
            return True
    return False


def region_metadata(
    source_iso: object,
    country: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return JSON-safe source-scope metadata for a raw catalog code.

    Child scopes (TW/HK/MO) carry their canonical parent; a canonical root
    source (CN) carries itself as ``parent_iso`` so the same contract can be
    used for merged city rows.
    """
    source = normalize_iso(source_iso)
    if not source:
        return {}
    parent = REGION_PARENT_ISO.get(source, source)
    payload: dict[str, object] = {
        "source_iso": source,
        "parent_iso": parent,
    }
    if isinstance(country, Mapping):
        name = country.get("name")
        name_zh = country.get("name_zh")
        if isinstance(name, str) and name:
            payload["name"] = name
        if isinstance(name_zh, str) and name_zh:
            payload["name_zh"] = name_zh
        for key in ("lat", "lng", "continent", "subregion"):
            value = country.get(key)
            if value is not None:
                payload[key] = value
    return payload


__all__ = [
    "ExclusionScope",
    "POLICY_VERSION",
    "REGION_PARENT_ISO",
    "canonical_country_iso",
    "country_members",
    "exclusion_scope",
    "is_country_member",
    "is_excluded",
    "is_region_iso",
    "normalize_iso",
    "region_members",
    "region_metadata",
    "source_region_iso",
]
