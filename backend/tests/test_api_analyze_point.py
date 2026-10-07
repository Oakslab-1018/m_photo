"""analyze-point endpoint tests (crop analysis, 404 and 422 handling)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.tests.support import ApiTestCase  # noqa: E402


class AnalyzePointApiTest(unittest.TestCase, ApiTestCase):
    def setUp(self):
        self._setup_client()
        self.session_id = self.upload_image().json()["session_id"]

    def tearDown(self):
        self._teardown_client()

    def test_analyze_point_returns_crop_info_and_clues(self):
        response = self.client.post(
            f"/api/geo/{self.session_id}/analyze-point",
            json={"x": 0.5, "y": 0.5},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()

        self.assertEqual(payload["crop_info"]["x"], 0.5)
        self.assertEqual(payload["crop_info"]["y"], 0.5)
        self.assertGreater(payload["crop_info"]["size"], 0)

        self.assertGreaterEqual(len(payload["new_clues"]), 4)
        self.assertTrue(all(clue["source"] == "point_click" for clue in payload["new_clues"]))
        self.assertTrue(all("点击区域" in clue["evidence"] for clue in payload["new_clues"]))

        dims = {clue["dimension"] for clue in payload["clues"]}
        self.assertIn("vegetation", dims)
        self.assertIsNotNone(payload["action"])
        self.assertGreaterEqual(len(payload["messages"]), 2)

    def test_analyze_point_rejects_out_of_range_xy(self):
        for payload in ({"x": -0.1, "y": 0.5}, {"x": 0.5, "y": 1.5}):
            response = self.client.post(
                f"/api/geo/{self.session_id}/analyze-point", json=payload
            )
            self.assertEqual(response.status_code, 422, response.text)
            self.assertIn("error", response.json())

    def test_analyze_point_rejects_non_numeric_xy(self):
        response = self.client.post(
            f"/api/geo/{self.session_id}/analyze-point",
            json={"x": "left", "y": 0.5},
        )
        self.assertEqual(response.status_code, 422)

    def test_analyze_point_unknown_session_404(self):
        response = self.client.post(
            "/api/geo/999999/analyze-point",
            json={"x": 0.5, "y": 0.5},
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn("error", response.json())

    def test_analyze_point_missing_image_404(self):
        session_id = self.upload_image().json()["session_id"]
        for image in self.settings.upload_dir.glob("*.png"):
            image.unlink()
        response = self.client.post(
            f"/api/geo/{session_id}/analyze-point",
            json={"x": 0.5, "y": 0.5},
        )
        self.assertEqual(response.status_code, 404)

    def test_crop_files_are_cleaned_up(self):
        before = set(self.settings.upload_dir.glob("crop_*.png"))
        self.client.post(
            f"/api/geo/{self.session_id}/analyze-point",
            json={"x": 0.3, "y": 0.7},
        )
        after = set(self.settings.upload_dir.glob("crop_*.png"))
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
