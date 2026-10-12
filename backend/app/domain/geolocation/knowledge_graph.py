"""知识图谱：从只读 SQLite 目录加载国家知识库并提供线索匹配查询。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from catalog_db import CatalogDatabase, default_catalog_path

_CONFIGURED_CATALOG_PATH: Path | None = None


def catalog_path() -> Path:
    """Return the process catalog path: configured runtime DB or seed fallback."""

    return _CONFIGURED_CATALOG_PATH or default_catalog_path()


def configure_catalog(path: str | Path) -> None:
    """Select the process catalog DB and invalidate the country payload cache.

    Backend startup calls this once after the runtime SQLite catalog has been
    imported. Offline callers that never configure the process keep reading the
    immutable shipped seed.
    """

    global _CONFIGURED_CATALOG_PATH
    candidate = Path(path).expanduser().resolve()
    _CONFIGURED_CATALOG_PATH = candidate
    load_countries.cache_clear()


@lru_cache(maxsize=1)
def load_countries() -> List[Dict[str, Any]]:
    """加载只读国家目录（带缓存），数据直接来自 SQLite seed/runtime 目录。"""

    return CatalogDatabase(catalog_path()).countries()


def get_by_iso(iso: str) -> Dict[str, Any] | None:
    for c in load_countries():
        if c["iso"] == iso:
            return c
    return None


def all_isos() -> List[str]:
    return [c["iso"] for c in load_countries()]


def countries_by_feature(dimension: str, value: str) -> List[Dict[str, Any]]:
    """返回在指定维度上包含该特征值的所有国家。"""
    result = []
    for c in load_countries():
        features = c.get("features", {})
        if value in features.get(dimension, []):
            result.append(c)
    return result


def countries_by_language(lang: str) -> List[Dict[str, Any]]:
    return [c for c in load_countries() if lang in c.get("languages_spoken", [])]


def countries_by_driving_side(side: str) -> List[Dict[str, Any]]:
    return [c for c in load_countries() if c.get("driving_side") == side]


def feature_values(dimension: str) -> List[str]:
    """返回某维度下所有出现过的特征值（去重）。"""
    values = set()
    for c in load_countries():
        for v in c.get("features", {}).get(dimension, []):
            values.add(v)
    return sorted(values)
