"""Configuration / env-alias tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.config import DEFAULT_DATA_DIRNAME, REPO_ROOT, Settings  # noqa: E402


class SettingsTest(unittest.TestCase):
    def test_defaults_use_mock_mode_and_storage_root(self):
        settings = Settings.from_env({})
        expected_root = (REPO_ROOT / "backend" / DEFAULT_DATA_DIRNAME).resolve()
        self.assertEqual(settings.geo_ai_mode, "mock")
        self.assertEqual(expected_root.name, "storage")
        self.assertEqual(settings.data_dir, expected_root)
        self.assertEqual(settings.upload_dir, (expected_root / "uploads").resolve())
        self.assertEqual(settings.db_path, (expected_root / "m_photo.db").resolve())
        self.assertEqual(settings.media_url_prefix, "/media")
        # the read-only country knowledge base must NOT live under the data root
        self.assertEqual(settings.countries_path, (REPO_ROOT / "data" / "countries.json").resolve())
        self.assertNotEqual(settings.data_dir, settings.countries_path.parent)

    def test_m_photo_data_dir_controls_uploads_and_db(self):
        settings = Settings.from_env({"M_PHOTO_DATA_DIR": "/data"})
        self.assertEqual(settings.data_dir, Path("/data").resolve())
        self.assertEqual(settings.upload_dir, (Path("/data") / "uploads").resolve())
        self.assertEqual(settings.db_path, (Path("/data") / "m_photo.db").resolve())
        self.assertTrue(settings.database_url.endswith("/data/m_photo.db"))

    def test_relative_data_dir_is_resolved_against_repo(self):
        settings = Settings.from_env({"M_PHOTO_DATA_DIR": "var/data"})
        self.assertEqual(settings.data_dir, (REPO_ROOT / "var/data").resolve())

    def test_env_aliases(self):
        settings = Settings.from_env(
            {
                "DATA_DIR": "/tmp/legacy-data",
                "UPLOAD_FOLDER": "pics",
                "DATABASE_PATH": "legacy.db",
                "CORS_ORIGINS": "http://a.test, http://b.test",
                "AI_MODE": "real",
            }
        )
        self.assertEqual(settings.data_dir, Path("/tmp/legacy-data").resolve())
        self.assertEqual(settings.upload_dir, (Path("/tmp/legacy-data") / "pics").resolve())
        self.assertEqual(settings.db_path, (Path("/tmp/legacy-data") / "legacy.db").resolve())
        self.assertEqual(settings.cors_origins, ("http://a.test", "http://b.test"))
        self.assertEqual(settings.geo_ai_mode, "real")

    def test_geo_ai_mode_defaults_to_mock_and_can_be_overridden(self):
        self.assertEqual(Settings.from_env({"GEO_AI_MODE": "MOCK"}).geo_ai_mode, "mock")
        self.assertEqual(Settings.from_env({"GEO_AI_MODE": "real"}).geo_ai_mode, "real")

    def test_unknown_ai_mode_is_a_passthrough_without_ocr_guarantee(self):
        """Only "mock" is exercised/supported in this stage."""
        settings = Settings.from_env({"GEO_AI_MODE": "mystery"})
        self.assertEqual(settings.geo_ai_mode, "mystery")  # resolved, not crash
        self.assertNotEqual(settings.geo_ai_mode, "mock")

    def test_database_url_override_wins(self):
        settings = Settings.from_env(
            {"M_PHOTO_DATA_DIR": "/data", "M_PHOTO_DB_URL": "sqlite:////tmp/custom.db"}
        )
        self.assertEqual(settings.database_url, "sqlite:////tmp/custom.db")
        self.assertEqual(settings.db_path, Path("/tmp/custom.db").resolve())


if __name__ == "__main__":
    unittest.main()
