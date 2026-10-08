"""Every logical write is committed once; failures leave no partial state."""
from __future__ import annotations

import unittest
from unittest import mock

from pydantic import ValidationError
from sqlalchemy import select

from backend.app.database import ChatMessage, GeoSession
from backend.app.services.chat import ChatService
from backend.app.services.geo_session import GeoSessionService
from backend.tests.support import ApiTestCase


class TransactionTest(unittest.TestCase, ApiTestCase):
    def setUp(self):
        self._setup_client()

    def tearDown(self):
        self._teardown_client()

    def snapshot(self):
        # A fresh session observes persisted data, not the failed request's cache.
        with self.app.state.session_factory() as session:
            return (
                [tuple(row) for row in session.execute(
                    select(GeoSession.__table__).order_by(GeoSession.id)
                )],
                [tuple(row) for row in session.execute(
                    select(ChatMessage.__table__).order_by(ChatMessage.id)
                )],
            )

    def assert_empty(self):
        self.assertEqual(self.snapshot(), ([], []))
        self.assertEqual(list(self.settings.upload_dir.iterdir()), [])

    def test_upload_initial_analysis_failure_rolls_back_rows_and_image(self):
        def fail(chat, session_id):
            chat.session_service.add_message(session_id, "ai", "partial analysis")
            raise RuntimeError("injected analysis failure")

        with mock.patch.object(ChatService, "initial_analysis", autospec=True, side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "injected analysis failure"):
                self.upload_image()
        self.assert_empty()

    def test_upload_commit_failure_rolls_back_rows_and_image(self):
        with mock.patch.object(GeoSessionService, "commit", side_effect=RuntimeError("commit failed")):
            with self.assertRaisesRegex(RuntimeError, "commit failed"):
                self.upload_image()
        self.assert_empty()

    def test_upload_response_validation_failure_rolls_back_rows_and_image(self):
        original = ChatService.initial_analysis

        def invalid_result(chat, session_id):
            result = original(chat, session_id)
            result["scores"] = [{"iso": "JP"}]  # Missing required response fields.
            return result

        with mock.patch.object(ChatService, "initial_analysis", autospec=True, side_effect=invalid_result):
            with self.assertRaises(ValidationError):
                self.upload_image()
        self.assert_empty()

    def test_upload_json_render_failure_rolls_back_rows_and_image(self):
        original = ChatService.initial_analysis

        def invalid_result(chat, session_id):
            result = original(chat, session_id)
            result["scores"] = [{
                "iso": "JP", "name": "Japan", "name_zh": "日本", "score": float("nan"),
            }]
            return result

        with mock.patch.object(ChatService, "initial_analysis", autospec=True, side_effect=invalid_result):
            with self.assertRaises(ValueError):
                self.upload_image()
        self.assert_empty()

    def test_feedback_message_failure_rolls_back_confirmation_and_messages(self):
        session_id = self.upload_image().json()["session_id"]
        before = self.snapshot()
        original = GeoSessionService.add_message

        def fail_second_message(geo, *args, **kwargs):
            message = original(geo, *args, **kwargs)
            if args[1] == "ai":
                raise RuntimeError("injected message failure")
            return message

        with mock.patch.object(GeoSessionService, "add_message", autospec=True, side_effect=fail_second_message):
            with self.assertRaisesRegex(RuntimeError, "injected message failure"):
                self.client.post(f"/api/geo/{session_id}/feedback", json={"type": "confirm", "iso": "JP"})
        self.assertEqual(self.snapshot(), before)

    def test_feedback_commit_failure_leaves_previous_state(self):
        session_id = self.upload_image().json()["session_id"]
        before = self.snapshot()
        with mock.patch.object(GeoSessionService, "commit", side_effect=RuntimeError("commit failed")):
            with self.assertRaisesRegex(RuntimeError, "commit failed"):
                self.client.post(f"/api/geo/{session_id}/feedback", json={"type": "reject", "iso": "JP"})
        self.assertEqual(self.snapshot(), before)

    def test_point_reasoning_failure_rolls_back_clues_and_messages_and_cleans_crop(self):
        payload = self.upload_image().json()
        session_id = payload["session_id"]
        before = self.snapshot()

        def fail(chat, session_id, **kwargs):
            chat.session_service.add_message(session_id, "ai", "partial point analysis")
            raise RuntimeError("injected reasoning failure")

        with mock.patch.object(ChatService, "reason_only", autospec=True, side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "injected reasoning failure"):
                self.client.post(f"/api/geo/{session_id}/analyze-point", json={"x": 0.5, "y": 0.5})
        self.assertEqual(self.snapshot(), before)
        self.assertEqual([p.name for p in self.settings.upload_dir.iterdir()], [payload["image_filename"]])

    def test_point_commit_failure_leaves_previous_state_and_cleans_crop(self):
        payload = self.upload_image().json()
        session_id = payload["session_id"]
        before = self.snapshot()
        with mock.patch.object(GeoSessionService, "commit", side_effect=RuntimeError("commit failed")):
            with self.assertRaisesRegex(RuntimeError, "commit failed"):
                self.client.post(f"/api/geo/{session_id}/analyze-point", json={"x": 0.5, "y": 0.5})
        self.assertEqual(self.snapshot(), before)
        self.assertEqual([p.name for p in self.settings.upload_dir.iterdir()], [payload["image_filename"]])

    def test_successful_operations_commit_exactly_once(self):
        original = GeoSessionService.commit
        with mock.patch.object(GeoSessionService, "commit", autospec=True, side_effect=original) as commit:
            response = self.upload_image()
            self.assertEqual(response.status_code, 201, response.text)
            self.assertEqual(commit.call_count, 1)
            session_id = response.json()["session_id"]
            # Flushed messages are already present in the response before commit.
            self.assertTrue(response.json()["messages"])

            commit.reset_mock()
            response = self.client.post(f"/api/geo/{session_id}/feedback", json={"type": "reject", "iso": "JP"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(commit.call_count, 1)

            commit.reset_mock()
            response = self.client.post(f"/api/geo/{session_id}/analyze-point", json={"x": 0.5, "y": 0.5})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(commit.call_count, 1)


if __name__ == "__main__":
    unittest.main()
