"""Offline Chinese city display names keyed by authoritative GeoNames IDs.

Presentation only: the source catalog, raw model names and coordinates stay intact.
Attribution and reproducible source fingerprint live alongside the name data.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_DATA_FILE = Path(__file__).resolve().parents[3] / "data" / "city-names.zh-CN.json"
_HAN = re.compile(r"[\u3400-\u9fff]")


@lru_cache(maxsize=1)
def _names() -> dict[str, str]:
    data = json.loads(_DATA_FILE.read_text(encoding="utf-8"))
    return {key: value for key, value in data.items()
            if isinstance(key, str) and isinstance(value, str) and _HAN.search(value)}


def city_name_zh(city_id: str | int, original: str = "") -> str:
    """Use a verified language-tagged Chinese name; never invent a translation."""
    label = _names().get(str(city_id))
    if label:
        return label
    # A source may already provide a Chinese label absent from the translation snapshot.
    if original and _HAN.search(original) and not re.search(r"[A-Za-z]", original):
        return original
    return "中文名待补充"
