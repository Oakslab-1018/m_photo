"""Health + countries endpoint tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.tests.support import ApiTestCase  # noqa: E402


class HealthApiTest(unittest.TestCase, ApiTestCase):
    def setUp(self):
        self._setup_client()

    def tearDown(self):
        self._teardown_client()

    def test_health_reports_db_ok(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["database"], "ok")
        self.assertEqual(payload["ai_mode"], "mock")
        self.assertTrue(Path(payload["data_dir"]).exists())

    def test_health_creates_sqlite_file(self):
        self.client.get("/api/health")
        self.assertTrue(self.settings.db_path.exists())


class CountriesApiTest(unittest.TestCase, ApiTestCase):
    def setUp(self):
        self._setup_client()

    def tearDown(self):
        self._teardown_client()

    def test_countries_listed(self):
        response = self.client.get("/api/geo/countries")
        self.assertEqual(response.status_code, 200, response.text)
        countries = response.json()["countries"]
        self.assertGreaterEqual(len(countries), 10)
        isos = {country["iso"] for country in countries}
        self.assertIn("JP", isos)
        japan = next(country for country in countries if country["iso"] == "JP")
        self.assertEqual(japan["name_zh"], "日本")
        self.assertIn("features", japan)


if __name__ == "__main__":
    unittest.main()
