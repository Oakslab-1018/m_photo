"""Feedback + messages endpoint tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.tests.support import ApiTestCase  # noqa: E402


class FeedbackApiTest(unittest.TestCase, ApiTestCase):
    def setUp(self):
        self._setup_client()
        self.session_id = self.upload_image().json()["session_id"]

    def tearDown(self):
        self._teardown_client()

    def test_confirm_feedback_closes_session(self):
        response = self.client.post(
            f"/api/geo/{self.session_id}/feedback",
            json={"type": "confirm", "iso": "JP"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["done"])
        self.assertEqual(payload["iso"], "JP")
        self.assertGreaterEqual(len(payload["messages"]), 2)
        self.assertIn("日本", payload["messages"][-1]["content"])

    def test_reject_feedback_adds_exclusion_and_reasons(self):
        response = self.client.post(
            f"/api/geo/{self.session_id}/feedback",
            json={"type": "reject", "iso": "JP"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertIn("JP", payload["exclusions"])
        self.assertIsInstance(payload["scores"], list)
        self.assertIsNotNone(payload["action"])

    def test_answer_feedback_appends_user_clue(self):
        response = self.client.post(
            f"/api/geo/{self.session_id}/feedback",
            json={"type": "answer", "dimension": "language", "value": "ja"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        user_sources = [clue for clue in payload["clues"] if clue.get("source") == "user"]
        self.assertTrue(user_sources)
        self.assertTrue(any(msg["role"] == "user" for msg in payload["messages"]))

    def test_free_text_feedback_parses_keywords(self):
        response = self.client.post(
            f"/api/geo/{self.session_id}/feedback",
            json={"type": "free_text", "text": "这里有日语招牌和温带森林"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        dimensions = {clue["dimension"] for clue in response.json()["clues"]}
        self.assertIn("language", dimensions)

    def test_feedback_unknown_session_404(self):
        response = self.client.post(
            "/api/geo/999999/feedback",
            json={"type": "confirm", "iso": "JP"},
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn("error", response.json())

    def test_feedback_invalid_type_422(self):
        response = self.client.post(
            f"/api/geo/{self.session_id}/feedback",
            json={"type": "shrug"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("error", response.json())

    def test_messages_returns_history(self):
        self.client.post(
            f"/api/geo/{self.session_id}/feedback",
            json={"type": "reject", "iso": "JP"},
        )
        response = self.client.get(f"/api/geo/{self.session_id}/messages")
        self.assertEqual(response.status_code, 200, response.text)
        messages = response.json()["messages"]
        self.assertGreaterEqual(len(messages), 3)
        roles = [msg["role"] for msg in messages]
        self.assertIn("ai", roles)
        self.assertIn("user", roles)
        message_ids = [msg["id"] for msg in messages]
        self.assertEqual(message_ids, sorted(message_ids))

    def test_messages_unknown_session_404(self):
        response = self.client.get("/api/geo/999999/messages")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
