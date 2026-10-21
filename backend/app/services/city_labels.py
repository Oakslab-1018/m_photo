"""Offline Chinese city display names keyed by authoritative GeoNames IDs.

Presentation only: the source catalog, raw model names and coordinates stay
intact. Curated Chinese names take precedence over the local Wikidata label
snapshot, which takes precedence over the authoritative source name. No
translation or network access happens at runtime.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_NAME_FILES = (
    _DATA_DIR / "city-names.zh-CN.json",
    _DATA_DIR / "city-names.wikidata.zh.json",
)
_HAN = re.compile(r"[\u3400-\u9fff]")
_LEGACY_PLACEHOLDER = "中文名待补充"
_UNNAMED_CITY = "未命名城市"


def _usable_name(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text or _LEGACY_PLACEHOLDER in text:
        return ""
    return text


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
        for city_id, raw_label in _read_names(path).items():
            label = _usable_name(raw_label)
            if city_id in names or not label or not _HAN.search(label):
                continue
            names[city_id] = label
    return names


def city_name_zh(city_id: str | int, original: str = "") -> str:
    """Return a local Chinese label, then the authoritative original name."""
    label = _names().get(str(city_id).strip())
    if label:
        return label
    return _usable_name(original) or _UNNAMED_CITY
