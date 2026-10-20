"""Focused tests for OCR-derived place evidence and its ranking boundary."""
from __future__ import annotations

import pytest

from backend.app.domain.geolocation.analysis import semantic as semantic_module
from backend.app.domain.geolocation.analysis.semantic import SemanticClueAnalyzer
from backend.app.domain.geolocation.clue import Clue
from backend.app.domain.geolocation.engine import ReasoningEngine


def _country(iso: str, name: str, name_zh: str = "") -> dict:
    return {"iso": iso, "name": name, "name_zh": name_zh, "features": {}}


def _city(name: str, iso: str, city_id: int, population: int = 1000) -> dict:
    return {
        "name": name,
        "iso": iso,
        "city_id": city_id,
        "population": population,
    }


def _signage(value: str, confidence: float = 0.95) -> Clue:
    return Clue(
        dimension="signage",
        value=value,
        confidence=confidence,
        evidence=f"OCR 文本证据：{value}",
        source="ocr",
        extra={
            "origin_source": "ocr",
            "provider": "rapidocr",
            "reliable": True,
            "line_confidence": confidence,
        },
    )


def _analyze(clues: list[Clue], countries: list[dict], cities: list[dict]) -> list[Clue]:
    analyzer = SemanticClueAnalyzer(countries=countries, cities=cities)
    return analyzer.analyze(clues)


def _place_clues(clues: list[Clue]) -> list[Clue]:
    return [clue for clue in clues if clue.dimension == "place_name"]


def _pandan_catalog() -> tuple[list[dict], list[dict]]:
    return (
        [_country("PH", "Philippines", "菲律宾")],
        [_city("Pandan", "PH", 8299649, 37325)],
    )


def test_pandan_mewah_regression_keeps_signage_without_place_evidence():
    countries, cities = _pandan_catalog()
    signage = _signage("PANDAN MEWAH")
    clues = [signage]

    derived = _analyze(clues, countries, cities)

    assert clues == [signage]
    assert _place_clues(derived) == []


def test_city_name_whole_line_without_country_is_not_evidence():
    countries, cities = _pandan_catalog()

    derived = _analyze([_signage("  Pandan!  ")], countries, cities)

    assert _place_clues(derived) == []


@pytest.mark.parametrize(
    ("text", "iso", "name", "city_name"),
    [
        ("GEORGE", "ZA", "South Africa", "George"),
        ("FOREST", "BE", "Belgium", "Forest"),
    ],
)
def test_whole_line_catalog_city_without_country_is_not_generic_evidence(
    text: str, iso: str, name: str, city_name: str
):
    countries = [_country(iso, name)]
    cities = [_city(city_name, iso, 9000)]

    derived = _analyze([_signage(text)], countries, cities)

    assert _place_clues(derived) == []


def test_direct_country_name_marks_explicit_country_context():
    countries, cities = _pandan_catalog()

    derived = _analyze([_signage("PHILIPPINES")], countries, cities)

    places = _place_clues(derived)
    assert [clue.value for clue in places] == ["PH"]
    assert places[0].extra["match"] == "name"
    assert places[0].extra["match_context"] == "explicit_country"


def test_multiword_city_requires_explicit_country_but_keeps_it_when_present():
    countries = [
        _country("PH", "Philippines", "菲律宾"),
        _country("ID", "Indonesia", "印度尼西亚"),
    ]
    cities = [
        _city("Pandan", "PH", 8299649, 37325),
        _city("Tanjung Pandan", "ID", 1624877, 103062),
    ]

    without_country = _analyze([_signage("TANJUNG PANDAN")], countries, cities)
    with_country = _analyze(
        [_signage("TANJUNG PANDAN IN INDONESIA")], countries, cities
    )

    assert _place_clues(without_country) == []
    places = _place_clues(with_country)
    assert [clue.value for clue in places] == ["ID"]
    assert places[0].extra["match_context"] == "explicit_country"
    assert places[0].extra["city_name"] == "Tanjung Pandan"


def test_city_unique_country_keeps_existing_fail_closed_behavior():
    countries = [_country("US", "United States")]
    cities = [
        _city("Springfield", "US", 1, 1000),
        _city("Springfield", "US", 2, 2000),
    ]

    derived = _analyze([_signage("SPRINGFIELD")], countries, cities)

    assert _place_clues(derived) == []


def test_city_name_inside_longer_sign_requires_explicit_country():
    countries = [
        _country("PH", "Philippines", "菲律宾"),
        _country("SG", "Singapore", "新加坡"),
    ]
    cities = [_city("Pandan", "PH", 8299649, 37325)]

    consistent = _analyze([_signage("PANDAN IN PHILIPPINES")], countries, cities)
    conflicting = _analyze([_signage("PANDAN IN SINGAPORE")], countries, cities)

    consistent_places = _place_clues(consistent)
    assert [clue.value for clue in consistent_places] == ["PH"]
    assert consistent_places[0].extra["match_context"] == "explicit_country"

    conflicting_places = _place_clues(conflicting)
    assert [clue.value for clue in conflicting_places] == ["SG"]
    assert all(clue.extra.get("catalog_iso") != "PH" for clue in conflicting_places)


def test_city_name_requires_one_explicit_country_on_the_line():
    countries = [
        _country("PH", "Philippines", "菲律宾"),
        _country("SG", "Singapore", "新加坡"),
    ]
    cities = [_city("Pandan", "PH", 8299649, 37325)]

    derived = _analyze(
        [_signage("PANDAN IN PHILIPPINES AND SINGAPORE")], countries, cities
    )

    places = _place_clues(derived)
    assert [clue.value for clue in places] == ["PH"]  # country-name path only
    assert all("city_name" not in clue.extra for clue in places)


def test_conflicting_explicit_country_suppresses_same_named_city():
    countries = [
        _country("SG", "Singapore", "新加坡"),
        _country("MY", "Malaysia", "马来西亚"),
    ]
    cities = [_city("Singapore", "MY", 9002)]

    derived = _analyze([_signage("SINGAPORE")], countries, cities)

    places = _place_clues(derived)
    assert [clue.value for clue in places] == ["SG"]
    assert all(clue.extra.get("catalog_iso") != "MY" for clue in places)


def test_cn_context_merges_tw_and_accepts_taipei_alias(monkeypatch):
    monkeypatch.setitem(semantic_module.CITY_NAME_ALIASES, "台北", ("Taipei", "TW"))
    countries = [
        _country("CN", "China", "中国"),
        _country("TW", "Taiwan", "台湾"),
        _country("JP", "Japan", "日本"),
    ]
    cities = [_city("Taipei", "TW", 1668341, 7871900)]
    analyzer = SemanticClueAnalyzer(countries=countries, cities=cities)

    assert analyzer._explicit_country_isos("taiwan 中国") == {"CN"}

    compatible = analyzer.analyze([_signage("台北 中国")])
    tw_place = next(clue for clue in _place_clues(compatible) if clue.value == "TW")
    assert tw_place.extra["catalog_iso"] == "TW"
    assert tw_place.extra["match_context"] == "city_alias"
    assert tw_place.extra["city_name"] == "Taipei"

    conflicting = analyzer.analyze([_signage("台北 日本")])
    assert all(
        clue.extra.get("catalog_iso") != "TW"
        for clue in _place_clues(conflicting)
    )


@pytest.mark.parametrize(
    ("text", "iso", "expected_alias"),
    [
        ("北京市", "CN", "北京"),
        ("東京駅", "JP", "東京"),
    ],
)
def test_cjk_aliases_and_one_admin_suffix_remain_strong(
    text: str, iso: str, expected_alias: str
):
    countries = [
        _country("CN", "China", "中国"),
        _country("JP", "Japan", "日本"),
    ]
    cities = [
        _city("Beijing", "CN", 1816670, 11716620),
        _city("Tokyo", "JP", 1850147, 8336599),
    ]

    derived = _analyze([_signage(text)], countries, cities)

    places = _place_clues(derived)
    assert [clue.value for clue in places] == [iso]
    assert places[0].extra["match"] == "city_alias"
    assert places[0].extra["match_context"] == "city_alias"
    assert places[0].extra["matched_text"] == expected_alias


def test_cjk_alias_is_suppressed_by_conflicting_explicit_country():
    countries = [
        _country("CN", "China", "中国"),
        _country("SG", "Singapore", "新加坡"),
    ]
    cities = [_city("Beijing", "CN", 1816670, 11716620)]

    derived = _analyze([_signage("北京市 IN SINGAPORE")], countries, cities)

    places = _place_clues(derived)
    assert [clue.value for clue in places] == ["SG"]
    assert all(clue.extra.get("catalog_iso") != "CN" for clue in places)


@pytest.mark.parametrize("text", ["NOT PANDAN", "NOT PANDAN PHILIPPINES"])
def test_negated_city_and_country_do_not_create_place_evidence(text: str):
    countries, cities = _pandan_catalog()

    derived = _analyze([_signage(text)], countries, cities)

    assert _place_clues(derived) == []


def test_synthetic_city_inside_sign_is_not_a_blacklist_special_case():
    countries = [_country("XX", "Exampleland")]
    cities = [_city("Harbor", "XX", 9001)]

    derived = _analyze([_signage("HARBOR VIEW")], countries, cities)

    assert _place_clues(derived) == []
    assert any(
        clue.dimension == "waterbody" and clue.value == "harbor"
        for clue in derived
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_same_iso_aggregation_keeps_explicit_country_over_alias(reverse: bool):
    countries = [_country("JP", "Japan", "日本")]
    cities = [_city("Tokyo", "JP", 1850147, 8336599)]
    alias = _signage("東京駅")
    explicit = _signage("JAPAN")
    clues = [explicit, alias] if reverse else [alias, explicit]

    derived = _analyze(clues, countries, cities)

    places = _place_clues(derived)
    assert [clue.value for clue in places] == ["JP"]
    assert places[0].extra["match_context"] == "explicit_country"


def test_rejected_city_token_does_not_enter_rule_scores():
    countries, cities = _pandan_catalog()
    signage = _signage("PANDAN MEWAH")
    derived = _analyze([signage], countries, cities)

    scores = ReasoningEngine().infer([signage, *derived], top_n=5)

    assert scores == []
