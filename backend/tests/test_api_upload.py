"""Upload endpoint tests (valid image, validation errors, media path)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.tests.support import ApiTestCase, make_test_image_bytes  # noqa: E402


class UploadApiTest(unittest.TestCase, ApiTestCase):
    def setUp(self):
        self._setup_client()

    def tearDown(self):
        self._teardown_client()

    def test_upload_creates_session_and_clues(self):
        response = self.upload_image()
        self.assertEqual(response.status_code, 201, response.text)
        payload = response.json()

        self.assertIsInstance(payload["session_id"], int)
        self.assertTrue(payload["image_filename"].endswith(".png"))
        self.assertEqual(payload["image_url"], f"/media/{payload['image_filename']}")

        dimensions = {clue["dimension"] for clue in payload["clues"]}
        self.assertGreaterEqual(len(payload["clues"]), 4)
        self.assertIn("vegetation", dimensions)
        self.assertIn("climate", dimensions)

        self.assertIsInstance(payload["scores"], list)
        self.assertIsNotNone(payload["action"])
        self.assertGreaterEqual(len(payload["messages"]), 1)
        self.assertEqual(payload["messages"][0]["role"], "ai")

        stored = self.settings.upload_dir / payload["image_filename"]
        self.assertTrue(stored.exists())
        self.assertTrue(self.settings.db_path.exists())

    def test_uploaded_image_served_under_media(self):
        payload = self.upload_image().json()
        response = self.client.get(payload["image_url"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, make_test_image_bytes())

    def test_upload_rejects_unsupported_extension(self):
        response = self.client.post(
            "/api/geo/upload",
            files={"image": ("notes.txt", b"hello", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_upload_rejects_invalid_image_payload(self):
        response = self.client.post(
            "/api/geo/upload",
            files={"image": ("fake.png", b"not-an-image", "image/png")},
        )
        self.assertEqual(response.status_code, 400)

    def test_upload_requires_file(self):
        response = self.client.post("/api/geo/upload")
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
