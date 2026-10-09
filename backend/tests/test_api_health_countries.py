"""Health + countries endpoint tests."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.services.model_gateway import catalog_revision  # noqa: E402
from backend.tests.support import ApiTestCase  # noqa: E402

# Offline snapshot of the canonical Google countries_csv representative points
# (https://developers.google.com/public-data/docs/canonical/countries_csv),
# retrieved 2026-10-09.
# Hardcoded on purpose: the regression test must not require network access.
CANONICAL_COUNTRY_POINTS = {
    "JP": (36.204824, 138.252924),
    "CN": (35.86166, 104.195397),
    "KR": (35.907757, 127.766922),
    "FR": (46.227638, 2.213749),
    "DE": (51.165691, 10.451526),
    "IT": (41.87194, 12.56738),
    "GB": (55.378051, -3.435973),
    "ES": (40.463667, -3.74922),
    "GR": (39.074208, 21.824312),
    "RU": (61.52401, 105.318756),
    "US": (37.09024, -95.712891),
    "CA": (56.130366, -106.346771),
    "BR": (-14.235004, -51.92528),
    "AU": (-25.274398, 133.775136),
    "IN": (20.593684, 78.96288),
    "TH": (15.870032, 100.992541),
    "EG": (26.820553, 30.802498),
    "MX": (23.634501, -102.552784),
    "NL": (52.132633, 5.291266),
    "SE": (60.128161, 18.643501),
}

# ISO + English-name catalog identity, unaffected by coordinate corrections.
EXPECTED_CATALOG_REVISION = "d0e88b0d4ba7fe5261278aa79a2546511fdb4de3a9032758f6dc987551a8d43d"


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

    def test_all_20_countries_match_offline_canonical_coordinates(self):
        countries = self.client.get("/api/geo/countries").json()["countries"]
        by_iso = {country["iso"]: country for country in countries}
        self.assertEqual(set(by_iso), set(CANONICAL_COUNTRY_POINTS))
        for iso, (lat, lng) in CANONICAL_COUNTRY_POINTS.items():
            with self.subTest(iso=iso):
                self.assertAlmostEqual(by_iso[iso]["lat"], lat, delta=1e-4)
                self.assertAlmostEqual(by_iso[iso]["lng"], lng, delta=1e-4)

    def test_coordinate_edits_keep_catalog_revision_unchanged(self):
        path = Path(__file__).resolve().parents[2] / "data" / "countries.json"
        countries = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(catalog_revision(countries), EXPECTED_CATALOG_REVISION)


class ConfirmationSemanticsTest(unittest.TestCase, ApiTestCase):
    """User confirmation is a COUNTRY pick, never a GPS/photo-location claim."""

    def setUp(self):
        self._setup_client()
        self.session_id = self.upload_image().json()["session_id"]

    def tearDown(self):
        self._teardown_client()

    def test_confirm_records_country_without_claiming_gps_or_locked_location(self):
        response = self.client.post(
            f"/api/geo/{self.session_id}/feedback",
            json={"type": "confirm", "iso": "JP"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["done"])
        self.assertEqual(payload["iso"], "JP")
        message = payload["messages"][-1]["content"]
        self.assertIn("日本", message)
        self.assertIn("确认", message)
        self.assertIn("国家", message)
        self.assertIn("不是照片 GPS", message)
        self.assertNotIn("已锁定拍摄地", message)
        self.assertNotIn("锁定", message)


if __name__ == "__main__":
    unittest.main()
