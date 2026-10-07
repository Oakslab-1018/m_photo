"""Country knowledge-base access.

Delegates to the geolocation domain knowledge graph (single source of truth,
cached, and rooted at the repository-level ``data/countries.json``).
"""
from __future__ import annotations

from typing import Any

from ..config import Settings
from ..domain.geolocation import knowledge_graph as kg


class CountriesService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def list_countries(self) -> list[dict[str, Any]]:
        return list(kg.load_countries())


__all__ = ["CountriesService"]
