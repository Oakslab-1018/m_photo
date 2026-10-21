"""Offline Chinese city labels and exact search aliases keyed by GeoNames IDs.

Presentation prefers curated Chinese names over the local Wikidata snapshot,
then falls back to the authoritative source name. The same immutable local
index also powers exact Chinese-name search without treating display text as an
identity: every lookup resolves back to a stable decimal GeoNames city ID. No
translation or network access happens at runtime.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_NAME_FILES = (
    _DATA_DIR / "city-names.zh-CN.json",
    _DATA_DIR / "city-names.wikidata.zh.json",
)
_HAN = re.compile(r"[\u3400-\u9fff]")
_POSITIVE_ID = re.compile(r"^[1-9][0-9]*$")
_LEGACY_PLACEHOLDER = "中文名待补充"
_UNNAMED_CITY = "未命名城市"


def _usable_name(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text or _LEGACY_PLACEHOLDER in text:
        return ""
    return text


def _city_id(value: object) -> str | None:
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not _POSITIVE_ID.fullmatch(text):
        return None
    return str(int(text))


def _normalize_label(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in text if character.isalnum())


def _read_names(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(key).strip(): value
        for key, value in data.items()
        if isinstance(key, str) and key.strip() and isinstance(value, str)
    }


@lru_cache(maxsize=1)
def _names() -> dict[str, str]:
    names: dict[str, str] = {}
    for path in _NAME_FILES:
        for raw_city_id, raw_label in _read_names(path).items():
            city_id = _city_id(raw_city_id)
            label = _usable_name(raw_label)
            if city_id is None or city_id in names or not label or not _HAN.search(label):
                continue
            names[city_id] = label
    return names


@lru_cache(maxsize=1)
def _name_index() -> dict[str, tuple[str, ...]]:
    index: dict[str, list[str]] = {}
    for city_id, label in _names().items():
        key = _normalize_label(label)
        if key:
            index.setdefault(key, []).append(city_id)
    return {
        key: tuple(sorted(city_ids, key=lambda item: int(item)))
        for key, city_ids in index.items()
    }


def city_name_zh(city_id: str | int, original: str = "") -> str:
    """Return a local Chinese label, then the authoritative original name."""

    canonical_id = _city_id(city_id)
    label = _names().get(canonical_id) if canonical_id is not None else None
    if label:
        return label
    return _usable_name(original) or _UNNAMED_CITY


def city_ids_for_name_zh(value: object) -> tuple[str, ...]:
    """Return every stable city ID whose local Chinese label exactly matches.

    Duplicate labels are deliberately preserved so callers can scope them by
    country and report ambiguity instead of silently choosing a population
    winner or the first file entry.
    """

    key = _normalize_label(value)
    if not key:
        return ()
    return _name_index().get(key, ())


__all__ = [
    "city_ids_for_name_zh",
    "city_name_zh",
]
