"""知识图谱：加载国家知识库并提供线索匹配查询。"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, List

from config import BASE_DIR

_DATA_PATH = os.path.join(BASE_DIR, "data", "countries.json")


@lru_cache(maxsize=1)
def load_countries() -> List[Dict[str, Any]]:
    """加载国家知识库（带缓存）。"""
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


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
