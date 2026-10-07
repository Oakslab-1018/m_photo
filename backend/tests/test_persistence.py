"""Storage persistence: sessions/messages survive an application restart."""
from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.main import create_app  # noqa: E402
from backend.tests.support import make_settings, make_test_image_bytes  # noqa: E402


class StoragePersistenceTest(unittest.TestCase):
    def test_session_messages_and_image_survive_restart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = make_settings(tmpdir)

            # ---- first "process" ----
            with ExitStack() as stack:
                client = stack.enter_context(TestClient(create_app(settings)))
                upload = client.post(
                    "/api/geo/upload",
                    files={
                        "image": (
                            "persist.png",
                            make_test_image_bytes(color=(200, 170, 90)),
                            "image/png",
                        )
                    },
                )
                self.assertEqual(upload.status_code, 201, upload.text)
                payload = upload.json()
                session_id = payload["session_id"]
                image_url = payload["image_url"]
                messages_before = len(payload["messages"])

            self.assertTrue(settings.db_path.exists())
            self.assertTrue((settings.upload_dir / payload["image_filename"]).exists())

            # ---- second "process" over the same storage root ----
            with ExitStack() as stack:
                client = stack.enter_context(TestClient(create_app(make_settings(tmpdir))))

                health = client.get("/api/health")
                self.assertEqual(health.status_code, 200)

                messages = client.get(f"/api/geo/{session_id}/messages")
                self.assertEqual(messages.status_code, 200, messages.text)
                self.assertEqual(len(messages.json()["messages"]), messages_before)

                feedback = client.post(
                    f"/api/geo/{session_id}/feedback",
                    json={"type": "reject", "iso": "JP"},
                )
                self.assertEqual(feedback.status_code, 200, feedback.text)
                self.assertIn("JP", feedback.json()["exclusions"])

                image = client.get(image_url)
                self.assertEqual(image.status_code, 200)

            # exclusions written by the second process are still on disk
            with ExitStack() as stack:
                client = stack.enter_context(TestClient(create_app(make_settings(tmpdir))))
                response = client.get(f"/api/geo/{session_id}/messages")
                self.assertEqual(response.status_code, 200)
                self.assertGreaterEqual(len(response.json()["messages"]), messages_before + 2)


if __name__ == "__main__":
    unittest.main()
