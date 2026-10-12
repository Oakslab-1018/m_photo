"""Focused ranking policy checks against the current SQLite catalog.

These tests deliberately avoid the retired JSON-era catalog fixtures: they use
the real runtime catalog bootstrapped by the application lifespan, then verify
that color/brightness heuristics are display-only while trusted text evidence
and the full-image model prior keep their intended ranking behavior.
"""
from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from backend.app.domain.geolocation.clue import Clue
from backend.app.domain.geolocation.engine import ReasoningEngine
from backend.app.domain.geolocation.model_ranking import ModelAwareEngine
from backend.app.main import create_app
from backend.app.services.catalog_access import catalog_for_settings
from backend.app.services.model_gateway import (
    MODEL_ID,
    MODEL_REVISION,
    StreetClipGateway,
    catalog_revision,
)
from backend.tests.support import make_settings


HEURISTICS = [
    Clue("climate", "cold_polar", 0.45, source="climate"),
    Clue("vegetation", "urban_sparse", 0.3, source="vegetation"),
    Clue("vehicle", "dark_vehicle", 0.4, source="vehicle"),
    Clue("building", "generic", 0.25, source="building"),
]


@pytest.fixture
def runtime_settings(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings)
    with TestClient(app):
        yield settings


def full_evidence(settings, leader: str = "FR") -> dict:
    countries = catalog_for_settings(settings).countries()
    rows = [
        {"iso": country["iso"], "raw_cosine": 0.40 if country["iso"] == leader else 0.10}
        for country in countries
    ]
    rows.sort(key=lambda row: (-row["raw_cosine"], row["iso"]))
    return {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "catalog_revision": catalog_revision(countries),
        "rankings": rows,
        "margin": round(rows[0]["raw_cosine"] - rows[1]["raw_cosine"], 6),
        "uncertain": False,
    }


def test_reasoning_engine_ignores_automated_color_heuristics(runtime_settings):
    engine = ReasoningEngine()
    assert engine.infer(HEURISTICS, top_n=5) == []
    legacy_pipeline = Clue(
        "vegetation",
        "urban_sparse",
        0.3,
        source="pipeline",
    )
    assert engine.infer([legacy_pipeline], top_n=5) == []

    trusted = Clue(
        "language",
        "ja",
        0.9,
        source="ocr",
        extra={"origin_source": "ocr", "reliable": True},
    )
    scores = engine.infer([*HEURISTICS, trusted], top_n=5)
    assert scores
    assert scores[0].iso == "JP"
    assert scores[0].score_type == "rule_weight"


def test_model_fusion_ignores_automated_color_heuristics(runtime_settings):
    gateway = StreetClipGateway(runtime_settings)
    evidence = gateway.validate(full_evidence(runtime_settings, leader="FR"))
    engine = ModelAwareEngine()
    engine.evidence = evidence

    scores = engine.infer(HEURISTICS, top_n=5)
    assert scores[0].iso == "FR"
    assert scores[0].score_type == "cosine"
    assert scores[0].raw_cosine == pytest.approx(0.40)
    assert all(not score.fusion for score in scores)


def test_analyzer_metadata_marks_color_clues_display_only(runtime_settings, tmp_path):
    from pathlib import Path
    from PIL import Image

    from backend.app.domain.geolocation.analysis.building import BuildingAnalyzer
    from backend.app.domain.geolocation.analysis.climate import ClimateAnalyzer
    from backend.app.domain.geolocation.analysis.vegetation import VegetationAnalyzer
    from backend.app.domain.geolocation.analysis.vehicle import VehicleAnalyzer

    image = Path(tmp_path) / "color.png"
    Image.new("RGB", (64, 64), (20, 20, 20)).save(image)
    for analyzer in (ClimateAnalyzer(), VegetationAnalyzer(), BuildingAnalyzer(), VehicleAnalyzer()):
        clues = analyzer.analyze(str(image))
        assert clues
        for clue in clues:
            assert clue.extra["display_only"] is True
            assert clue.extra["affects_ranking"] is False
            assert clue.extra["reliable"] is False
