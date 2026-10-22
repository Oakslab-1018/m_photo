"""Optional, source-backed local place names used before network lookup.

The index is intentionally exact-match only and defaults to an absent
``data/place-evidence.json`` file.  A configured file may add high-confidence
city aliases, POIs and roads, but every row still resolves through the
authoritative city catalog at use time.  Missing or malformed data degrades to
an empty index; this module never downloads or invents place names.
"""
from __future__ import annotations

import json
import logging
import os
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

MIN_LOCAL_CONFIDENCE = 0.9
MAX_LOCAL_NAME_LENGTH = 120
_KIND_TO_SOURCE_TYPE = {
    "city_alias": "local_catalog",
    "poi": "local_poi",
    "road": "local_road",
}
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_INDEX_PATH = _REPO_ROOT / "data" / "place-evidence.json"


def _normalize_name(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in text if character.isalnum())


def _source_iso(value: object) -> str:
    text = str(value or "").strip().upper()
    return text if len(text) == 2 and text.isalpha() else ""


def _positive_city_id(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        city_id = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return city_id if city_id > 0 else None


def _confidence(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not (MIN_LOCAL_CONFIDENCE <= number <= 1.0):
        return None
    return number


def _trusted_url(value: object) -> str:
    text = str(value or "").strip()
    if (
        not text
        or len(text) > 1000
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


def _source_records(value: object, source_type: str) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Mapping):
        raw_sources: Sequence[object] = (value,)
    elif isinstance(value, list):
        raw_sources = value
    else:
        return ()
    result: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in raw_sources[:8]:
        if not isinstance(raw, Mapping):
            continue
        label = str(raw.get("label") or "").strip()[:200]
        url = _trusted_url(raw.get("url"))
        if not label or (source_type in {"local_poi", "local_road"} and not url):
            continue
        source = {
            "provider": str(raw.get("provider") or "local_evidence").strip()[:40] or "local_evidence",
            "provider_place_id": (
                str(raw.get("provider_place_id"))[:200]
                if raw.get("provider_place_id") is not None else None
            ),
            "label": label,
            "query": str(raw.get("query") or "")[:200],
            "attribution": (
                str(raw.get("attribution"))[:200]
                if raw.get("attribution") is not None else None
            ),
            "license": (
                str(raw.get("license"))[:80]
                if raw.get("license") is not None else None
            ),
            "fetched_at": str(raw.get("fetched_at") or "")[:80],
            "source_type": source_type,
            "title": str(raw.get("title") or label)[:200],
            "url": url or None,
            "excerpt": str(raw.get("excerpt") or "")[:300],
        }
        key = (
            str(source.get("source_type")),
            str(source.get("url") or ""),
            str(source.get("label") or ""),
        )
        if key not in seen:
            seen.add(key)
            result.append(source)
    return tuple(result)


@dataclass(frozen=True)
class LocalPlaceMatch:
    """One exact local name bound to a catalog city and source evidence."""

    name: str
    city_id: int
    source_iso: str
    kind: str
    confidence: float
    sources: tuple[Mapping[str, Any], ...]

    @property
    def source_type(self) -> str:
        return _KIND_TO_SOURCE_TYPE[self.kind]

    @property
    def match_kind(self) -> str:
        return {
            "city_alias": "city_alias",
            "poi": "poi",
            "road": "road",
        }[self.kind]


class LocalPlaceIndex:
    """Immutable exact-name index; an empty instance is a valid fail-closed state."""

    def __init__(
        self,
        matches: Sequence[LocalPlaceMatch] = (),
        *,
        path: Path | None = None,
    ) -> None:
        by_name: dict[str, dict[tuple[str, int, str], LocalPlaceMatch]] = {}
        aliases: dict[tuple[str, int], list[str]] = {}
        for match in matches:
            key = _normalize_name(match.name)
            if not key:
                continue
            by_name.setdefault(key, {})[(match.source_iso, match.city_id, match.kind)] = match
            if match.kind == "city_alias":
                values = aliases.setdefault((match.source_iso, match.city_id), [])
                if match.name not in values:
                    values.append(match.name)
        self._by_name = {
            key: tuple(values.values()) for key, values in by_name.items()
        }
        self._aliases = {
            key: tuple(sorted(values)) for key, values in aliases.items()
        }
        self.path = path

    def lookup(
        self,
        value: object,
        *,
        source_isos: Sequence[str] = (),
    ) -> tuple[LocalPlaceMatch, ...]:
        key = _normalize_name(value)
        if not key:
            return ()
        allowed = {
            normalized for normalized in (_source_iso(item) for item in source_isos)
            if normalized
        }
        return tuple(
            match for match in self._by_name.get(key, ())
            if not allowed or match.source_iso in allowed
        )

    def city_aliases_for_city(
        self,
        city_id: str | int,
        source_iso: str,
    ) -> tuple[str, ...]:
        canonical_id = _positive_city_id(city_id)
        iso = _source_iso(source_iso)
        if canonical_id is None or not iso:
            return ()
        return self._aliases.get((iso, canonical_id), ())

    def __bool__(self) -> bool:
        return bool(self._by_name)


def _empty_index(path: Path | None = None) -> LocalPlaceIndex:
    return LocalPlaceIndex(path=path)


def _parse_index(payload: object, path: Path) -> LocalPlaceIndex:
    if isinstance(payload, Mapping):
        raw_entries = payload.get("entries")
    else:
        raw_entries = payload
    if not isinstance(raw_entries, list):
        return _empty_index(path)

    matches: list[LocalPlaceMatch] = []
    seen: set[tuple[str, str, int, str]] = set()
    for raw in raw_entries:
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("name") or "").strip()
        normalized = _normalize_name(name)
        kind = str(raw.get("kind") or "").strip().lower()
        city_id = _positive_city_id(raw.get("city_id"))
        iso = _source_iso(raw.get("source_iso"))
        confidence = _confidence(raw.get("confidence"))
        if (
            not normalized
            or len(name) > MAX_LOCAL_NAME_LENGTH
            or kind not in _KIND_TO_SOURCE_TYPE
            or city_id is None
            or not iso
            or confidence is None
        ):
            continue
        source_type = _KIND_TO_SOURCE_TYPE[kind]
        sources = _source_records(raw.get("sources", raw.get("source")), source_type)
        if not sources:
            continue
        key = (normalized, iso, city_id, kind)
        if key in seen:
            continue
        seen.add(key)
        matches.append(LocalPlaceMatch(
            name=name,
            city_id=city_id,
            source_iso=iso,
            kind=kind,
            confidence=confidence,
            sources=sources,
        ))
    return LocalPlaceIndex(matches, path=path)


@lru_cache(maxsize=8)
def _load_cached(path_text: str, mtime_ns: int, size: int) -> LocalPlaceIndex:
    del mtime_ns, size  # cache key only; avoids stale reads after atomic replacement
    path = Path(path_text)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Local place index ignored (%s)", type(exc).__name__)
        return _empty_index(path)
    return _parse_index(payload, path)


def configured_index_path() -> Path:
    override = str(os.environ.get("M_PHOTO_LOCAL_PLACE_INDEX") or "").strip()
    return Path(override).expanduser() if override else _DEFAULT_INDEX_PATH


def get_local_place_index(path: str | Path | None = None) -> LocalPlaceIndex:
    candidate = Path(path).expanduser() if path is not None else configured_index_path()
    try:
        resolved = candidate.resolve()
        stat = resolved.stat()
    except OSError:
        return _empty_index(candidate)
    if not resolved.is_file():
        return _empty_index(candidate)
    return _load_cached(str(resolved), stat.st_mtime_ns, stat.st_size)


def clear_local_place_index_cache() -> None:
    _load_cached.cache_clear()


__all__ = [
    "LocalPlaceIndex",
    "LocalPlaceMatch",
    "MIN_LOCAL_CONFIDENCE",
    "clear_local_place_index_cache",
    "configured_index_path",
    "get_local_place_index",
]
