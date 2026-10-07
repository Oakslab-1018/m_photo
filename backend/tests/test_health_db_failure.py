"""Health endpoint must fail (503) when the database is not usable."""
from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from backend.app.main import create_app  # noqa: E402
from backend.tests.support import make_settings  # noqa: E402


class HealthDatabaseCheckTest(unittest.TestCase):
    def test_health_reports_503_when_select_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = make_settings(tmpdir)
            with ExitStack() as stack:
                client = stack.enter_context(TestClient(create_app(settings)))
                ok = client.get("/api/health")
                self.assertEqual(ok.status_code, 200, ok.text)

                app = client.app
                real_factory = app.state.session_factory
                app.state.session_factory = _factory_with_broken_db(real_factory)

                broken = client.get("/api/health")
                self.assertEqual(broken.status_code, 503, broken.text)
                self.assertEqual(broken.json()["error"], "database unavailable")


def _factory_with_broken_db(session_factory):
    def factory():
        session = session_factory()
        session.execute = lambda *args, **kwargs: _raise()  # type: ignore[assignment]
        return session

    return factory


def _raise():
    raise RuntimeError("database is gone")


if __name__ == "__main__":
    unittest.main()
