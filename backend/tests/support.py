"""Shared helpers for the backend test-suite.

Written with :mod:`unittest` primitives so the suite runs both with ``pytest``
and with ``python -m unittest``.
"""
from __future__ import annotations

import io
import tempfile
from contextlib import ExitStack
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.config import Settings
from backend.app.main import create_app


def make_settings(tmpdir: str | Path) -> Settings:
    """Build isolated test settings rooted at ``tmpdir``.

    The catalog is database-only now; dedicated catalog tests inject a
    temporary SQLite seed (``catalog_seed_path_override``) instead of the
    retired JSON path overrides.
    """
    root = Path(tmpdir).resolve()
    data_dir = root / "data"
    upload_dir = data_dir / "uploads"
    db_path = data_dir / "test.db"
    return Settings(
        data_dir=data_dir,
        upload_dir=upload_dir,
        db_path=db_path,
        database_url=f"sqlite:///{db_path.as_posix()}",
        media_url_prefix="/media",
        cors_origins=("http://localhost:5173",),
        geo_ai_mode="mock",
    )


def make_test_image_bytes(color=(60, 160, 60), size=(64, 64)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class ApiTestCase:
    """Mixin providing a lifespan-managed ``TestClient`` per test."""

    def _setup_client(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.settings = make_settings(self._tmpdir.name)
        self.app = create_app(self.settings)
        self._stack = ExitStack()
        self.client = self._stack.enter_context(TestClient(self.app))

    def _teardown_client(self) -> None:
        self._stack.close()
        self._tmpdir.cleanup()

    # ---- helpers ----
    def upload_image(self, name: str = "street.png", color=(60, 160, 60)):
        return self.client.post(
            "/api/geo/upload",
            files={"image": (name, make_test_image_bytes(color), "image/png")},
        )


__all__ = ["ApiTestCase", "make_settings", "make_test_image_bytes"]
