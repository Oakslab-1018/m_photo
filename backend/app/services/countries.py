"""Country knowledge-base access.

Delegates to the existing ``reasoning.knowledge_graph`` module (single source
of truth, cached) and falls back to reading ``data/countries.json`` directly
if the legacy module cannot be imported.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from ..config import Settings


class CountriesService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def list_countries(self) -> list[dict[str, Any]]:
        try:
            from reasoning import knowledge_graph as kg

            return list(kg.load_countries())
        except Exception:  # pragma: no cover - fallback path
            return list(_load_countries_file(str(self._settings.countries_path)))


@lru_cache(maxsize=4)
def _load_countries_file(path: str) -> tuple[dict[str, Any], ...]:
    text = open(path, "r", encoding="utf-8").read()
    return tuple(json.loads(text))


__all__ = ["CountriesService"]
