"""Static import checks and the "no easyocr required" guarantee."""
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class StaticImportTest(unittest.TestCase):
    def test_backend_app_imports_without_side_effects(self):
        app_module = importlib.import_module("backend.app.main")
        self.assertTrue(hasattr(app_module, "app"))
        self.assertTrue(hasattr(app_module, "create_app"))

    def test_area_and_module_layout_importable(self):
        for name in (
            "backend.app.config",
            "backend.app.database",
            "backend.app.deps",
            "backend.app.schemas",
            "backend.app.routers.health",
            "backend.app.routers.countries",
            "backend.app.routers.geo",
            "backend.app.services.chat",
            "backend.app.services.countries",
            "backend.app.services.geo_session",
            "backend.app.services.pipeline",
        ):
            with self.subTest(module=name):
                self.assertIsNotNone(importlib.import_module(name))

    def test_domain_dependencies_are_reused_not_copied(self):
        domain = importlib.import_module("backend.app.domain.geolocation")
        analysis = importlib.import_module("backend.app.domain.geolocation.analysis")
        chat = importlib.import_module("services.chat")
        self.assertFalse(hasattr(domain, "AnalysisPipeline"))
        self.assertTrue(hasattr(analysis, "AnalysisPipeline"))
        self.assertTrue(hasattr(chat, "ChatService"))
        self.assertTrue(issubclass(
            importlib.import_module("backend.app.services.chat").ChatService,
            chat.ChatService,
        ))

    def test_mock_mode_does_not_import_easyocr(self):
        sys.modules.pop("easyocr", None)
        backend = importlib.reload(importlib.import_module("backend.app.services.pipeline"))
        settings = mock.Mock(geo_ai_mode="mock")
        service = backend.PipelineService(settings)
        self.assertEqual(service.mode, "mock")
        # Building the pipeline in mock mode must never touch easyocr.
        service._get_pipeline()
        self.assertNotIn("easyocr", sys.modules)
        self.assertNotIn("easyocr", sys.modules)


if __name__ == "__main__":
    unittest.main()
