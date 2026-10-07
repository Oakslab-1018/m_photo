"""Domain layout and migration-contract checks."""
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.domain.geolocation.analysis import AnalysisPipeline  # noqa: E402
from backend.app.domain.geolocation.engine import ReasoningEngine  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


class DomainLayoutTest(unittest.TestCase):
    def test_new_domain_modules_are_importable(self):
        for name in (
            "backend.app.domain.geolocation",
            "backend.app.domain.geolocation.clue",
            "backend.app.domain.geolocation.engine",
            "backend.app.domain.geolocation.hypotheses",
            "backend.app.domain.geolocation.knowledge_graph",
            "backend.app.domain.geolocation.analysis",
            "backend.app.domain.geolocation.analysis.base",
            "backend.app.domain.geolocation.analysis.ocr",
            "backend.app.domain.geolocation.analysis.building",
            "backend.app.domain.geolocation.analysis.climate",
            "backend.app.domain.geolocation.analysis.vegetation",
            "backend.app.domain.geolocation.analysis.vehicle",
            "backend.app.domain.geolocation.analysis.text_language",
            "backend.app.domain.geolocation.analysis.pipeline",
        ):
            with self.subTest(module=name):
                self.assertIsNotNone(importlib.import_module(name))

    def test_reasoning_public_api_is_reexported(self):
        domain = importlib.import_module("backend.app.domain.geolocation")
        expected = {
            "Clue", "DIMENSIONS", "clues_to_dicts", "dicts_to_clues",
            "ReasoningEngine", "CountryScore",
            "HypothesisManager", "HypothesisAction", "knowledge_graph",
        }
        for name in expected:
            with self.subTest(name=name):
                self.assertTrue(hasattr(domain, name), f"missing re-export: {name}")
        self.assertEqual(set(domain.__all__), expected)
        self.assertFalse(hasattr(domain, "AnalysisPipeline"))

    def test_root_source_packages_were_removed(self):
        self.assertFalse((REPO_ROOT / "ai").exists(), "root ai/ must be removed")
        self.assertFalse((REPO_ROOT / "reasoning").exists(), "root reasoning/ must be removed")

    def test_pipeline_service_uses_new_analysis_pipeline(self):
        settings = mock.Mock(geo_ai_mode="mock")
        service = importlib.import_module("backend.app.services.pipeline").PipelineService(settings)
        pipeline = service._get_pipeline()
        self.assertIsInstance(pipeline, AnalysisPipeline)

    def test_countries_service_uses_domain_knowledge_graph(self):
        settings = mock.Mock()
        service = importlib.import_module("backend.app.services.countries").CountriesService(settings)
        fake_countries = [{"iso": "ZZ", "name": "Testland"}]
        with mock.patch(
            "backend.app.domain.geolocation.knowledge_graph.load_countries",
            return_value=fake_countries,
        ) as load_countries:
            self.assertEqual(service.list_countries(), fake_countries)
            load_countries.assert_called_once_with()

    def test_mock_analysis_does_not_import_easyocr(self):
        sys.modules.pop("easyocr", None)
        pipeline = AnalysisPipeline(mode="mock")
        self.assertEqual(pipeline.mode, "mock")
        self.assertNotIn("easyocr", sys.modules)

    def test_new_engine_loads_root_country_data(self):
        engine = ReasoningEngine()
        self.assertTrue(engine.countries)
        self.assertIn("JP", {country["iso"] for country in engine.countries})


if __name__ == "__main__":
    unittest.main()
