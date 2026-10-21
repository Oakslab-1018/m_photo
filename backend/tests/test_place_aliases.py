"""Focused tests for the pure curated place-alias module."""
from __future__ import annotations

from pathlib import Path

from backend.app.domain.geolocation.place_aliases import (
    CURATED_PLACE_ALIASES,
    PLACE_ALIAS_VERSION,
    match_and_validate_curated_place_alias,
    match_curated_place_alias,
    validate_curated_place_alias,
)
from backend.app.services.city_catalog import load_city_index

ROOT = Path(__file__).resolve().parents[2]


def test_exact_street_aliases_carry_catalog_and_provenance():
    for literal in ("玉林路", "玉林西路", "“玉林路”", "‘玉林西路’"):
        entry = match_curated_place_alias(literal)
        assert entry is not None
        assert entry.kind == "street_label"
        assert entry.city_id == "1815286"
        assert entry.country_iso == "CN"
        assert entry.city_name == "Chengdu"
        assert entry.display_label == "成都"
        assert entry.version == PLACE_ALIAS_VERSION
        assert entry.proposal_only is True
        assert entry.globally_unique is False
        assert [source.title for source in entry.sources] == [
            "玉林街道：助力成都打造国际消费中心城市",
            "如果你要写成都",
        ]
        assert [source.publisher for source in entry.sources] == [
            "四川省商务厅（商务部网站发布）",
            "国家体育总局体育彩票管理中心",
        ]
        assert all(source.verified_on == "2026-10-21" for source in entry.sources)
        assert not hasattr(entry, "lat")
        assert not hasattr(entry, "lng")


def test_direct_chengdu_names_match_the_city_entry():
    for literal in ("成都", "成都市"):
        entry = match_curated_place_alias(literal)
        assert entry is not None
        assert entry.kind == "city_name"
        assert entry.city_id == "1815286"
        assert entry.display_label == "成都"


def test_aliases_are_exact_not_substrings():
    for literal in ("玉林", "玉林路附近", "成都玉林路", "梧桐巷", "", None):
        assert match_curated_place_alias(literal) is None


def test_validation_accepts_the_committed_catalog_row():
    index = load_city_index(ROOT / "data" / "geo_catalog.sqlite3")
    for literal in ("玉林路", "玉林西路", "成都", "成都市"):
        entry = match_curated_place_alias(literal)
        assert entry is not None
        assert validate_curated_place_alias(entry, index) is True


def test_validation_fails_closed_for_missing_or_mismatched_rows():
    entry = next(item for item in CURATED_PLACE_ALIASES if item.kind == "street_label")
    assert validate_curated_place_alias(entry, {}) is False
    assert validate_curated_place_alias(entry, {"CN": {}}) is False
    assert validate_curated_place_alias(
        entry,
        {"CN": {1815286: {"city_id": 1815286, "iso": "JP", "name": "Chengdu"}}},
    ) is False
    assert validate_curated_place_alias(
        entry,
        {"CN": {1815286: {"city_id": 1815286, "iso": "CN", "name": "Chengdu"}}},
    ) is True
    assert match_and_validate_curated_place_alias("玉林路", {"CN": {}}) is None
