"""Edge cases: JSON-safe 422 payloads, NaN rejection, tiny-image crops."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.tests.support import ApiTestCase, make_test_image_bytes  # noqa: E402


class ValidationPayloadTest(unittest.TestCase, ApiTestCase):
    def setUp(self):
        self._setup_client()
        self.session_id = self.upload_image().json()["session_id"]

    def tearDown(self):
        self._teardown_client()

    def test_missing_feedback_field_returns_json_safe_422(self):
        """A model-level ValueError lands in ``ctx`` and must be serializable."""
        response = self.client.post(
            f"/api/geo/{self.session_id}/feedback",
            json={"type": "confirm"},  # missing "iso"
        )
        self.assertEqual(response.status_code, 422, response.text)
        payload = json.loads(response.text)  # would raise if ctx held a ValueError
        self.assertEqual(payload["error"], "validation error")
        self.assertTrue(payload["detail"])
        for item in payload["detail"]:
            for value in (item.get("ctx") or {}).values():
                self.assertIsInstance(value, str)

    def test_answer_feedback_missing_dimension_or_value_422(self):
        for body in (
            {"type": "answer", "value": "ja"},
            {"type": "answer", "dimension": "language"},
            {"type": "answer"},
        ):
            with self.subTest(body=body):
                response = self.client.post(
                    f"/api/geo/{self.session_id}/feedback", json=body
                )
                self.assertEqual(response.status_code, 422, response.text)
                self.assertEqual(response.json()["error"], "validation error")

    def test_free_text_feedback_missing_text_422(self):
        for body in ({"type": "free_text"}, {"type": "free_text", "text": "   "}):
            with self.subTest(body=body):
                response = self.client.post(
                    f"/api/geo/{self.session_id}/feedback", json=body
                )
                self.assertEqual(response.status_code, 422)

    def test_analyze_point_rejects_nan(self):
        response = self.client.post(
            f"/api/geo/{self.session_id}/analyze-point",
            content=b'{"x": NaN, "y": 0.5}',
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 422, response.text)
        payload = json.loads(response.text)
        self.assertEqual(payload["error"], "validation error")

    def test_analyze_point_rejects_infinity(self):
        response = self.client.post(
            f"/api/geo/{self.session_id}/analyze-point",
            content=b'{"x": Infinity, "y": 0.5}',
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 422, response.text)


class TinyImageTest(unittest.TestCase, ApiTestCase):
    def setUp(self):
        self._setup_client()

    def tearDown(self):
        self._teardown_client()

    def test_one_pixel_image_crop_does_not_collapse(self):
        upload = self.client.post(
            "/api/geo/upload",
            files={
                "image": ("dot.png", make_test_image_bytes(size=(1, 1)), "image/png")
            },
        )
        self.assertEqual(upload.status_code, 201, upload.text)
        session_id = upload.json()["session_id"]

        response = self.client.post(
            f"/api/geo/{session_id}/analyze-point", json={"x": 0.5, "y": 0.5}
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertGreaterEqual(payload["crop_info"]["size"], 1)
        self.assertTrue(payload["new_clues"])


if __name__ == "__main__":
    unittest.main()
