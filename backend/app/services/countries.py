"""Country knowledge-base access with canonical product grouping.

The read-only SQLite catalog remains the source of catalog identities and
revision material. This service only presents the product-country directory:
subordinate China source scopes are listed once under ``CN`` in
``source_regions`` instead of appearing as independent top-level countries.
"""
from __future__ import annotations

from typing import Any

from ..domain.geolocation.country_policy import (
    canonical_country_iso,
    region_members,
)
from .catalog_access import catalog_for_settings


def _region_entry(source_iso: str, parent_iso: str, row: dict[str, Any]) -> dict[str, Any]:
    """Return the bounded region metadata consumed by the frontend directory."""
    return {
        "source_iso": source_iso,
        "parent_iso": parent_iso,
        "name": row.get("name", source_iso),
        "name_zh": row.get("name_zh", row.get("name", source_iso)),
        "lat": row.get("lat"),
        "lng": row.get("lng"),
    }


class CountriesService:
    def __init__(self, settings) -> None:
        self._settings = settings

    def list_countries(self) -> list[dict[str, Any]]:
        raw_countries = [
            country for country in catalog_for_settings(self._settings).countries()
            if isinstance(country, dict) and isinstance(country.get("iso"), str)
        ]
        by_iso = {country["iso"].strip().upper(): country for country in raw_countries}
        canonical_isos = sorted({
            canonical_country_iso(iso)
            for iso in by_iso
            if canonical_country_iso(iso)
        })

        countries: list[dict[str, Any]] = []
        for iso in canonical_isos:
            root = by_iso.get(iso)
            if root is None:
                continue
            entry = dict(root)
            regions = [
                _region_entry(child, iso, by_iso[child])
                for child in region_members(iso)
                if child in by_iso
            ]
            if regions:
                entry["source_regions"] = regions
            countries.append(entry)
        return countries


__all__ = ["CountriesService"]
