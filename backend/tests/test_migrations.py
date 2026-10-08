"""Alembic startup, safe adoption, fail-closed checks and CLI integration."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from alembic import command
from alembic.runtime.migration import MigrationContext
from alembic.util.exc import CommandError
from fastapi.testclient import TestClient
from sqlalchemy import CheckConstraint, DefaultClause, MetaData, String, inspect, text

from backend.app import migrations
from backend.app.database import Base, ChatMessage, GeoSession, create_db_engine, create_session_factory, init_db
from backend.app.main import create_app
from backend.app.migrations import BASELINE_REVISION, SchemaCompatibilityError
from backend.tests.support import make_settings, make_test_image_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAD_REVISION = "0002_model_evidence"


class MigrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = make_settings(self.tmp.name)

    def make_legacy(self, metadata=None):
        engine = create_db_engine(self.settings)
        self.addCleanup(engine.dispose)
        (metadata if metadata is not None else self.copied_metadata()).create_all(engine)
        return engine

    def revision(self, engine):
        with engine.connect() as connection:
            return MigrationContext.configure(connection).get_current_heads()

    def definition(self, engine):
        with engine.connect() as connection:
            return list(connection.execute(text(
                "SELECT type, name, tbl_name, sql FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            )).tuples())

    def copied_metadata(self):
        # Freeze legacy fixtures at immutable 0001, not the evolving ORM.
        from sqlalchemy import create_engine
        reference = create_engine("sqlite://")
        try:
            with reference.begin() as connection:
                command.upgrade(migrations._config(connection), BASELINE_REVISION)
                metadata = MetaData()
                metadata.reflect(connection, only=["geo_session", "chat_message"])
            return metadata
        finally:
            reference.dispose()

    def test_empty_database_uses_alembic_not_create_all(self):
        with mock.patch.object(Base.metadata, "create_all", side_effect=AssertionError("must use migrations")):
            engine, _ = init_db(self.settings)
        self.addCleanup(engine.dispose)
        self.assertEqual(self.revision(engine), (HEAD_REVISION,))
        self.assertEqual(set(inspect(engine).get_table_names()), {"geo_session", "chat_message", "alembic_version"})

    def test_repeated_initialization_is_idempotent(self):
        first, _ = init_db(self.settings)
        before = self.definition(first)
        first.dispose()
        second, _ = init_db(self.settings)
        self.addCleanup(second.dispose)
        self.assertEqual(self.definition(second), before)
        self.assertEqual(self.revision(second), (HEAD_REVISION,))

    def test_legacy_rows_messages_and_image_survive_adoption_and_restart(self):
        engine = self.make_legacy()
        self.settings.upload_dir.mkdir(parents=True)
        image = self.settings.upload_dir / "old.png"
        image.write_bytes(make_test_image_bytes())
        legacy = self.copied_metadata()
        with engine.begin() as connection:
            inserted = connection.execute(legacy.tables["geo_session"].insert().values(
                image_path=str(image), image_filename=image.name, created_at=datetime(2026, 1, 1)))
            session_id = inserted.inserted_primary_key[0]
            connection.execute(legacy.tables["chat_message"].insert().values(
                session_id=session_id, role="ai", content="existing message", created_at=datetime(2026, 1, 1)))
        self.assertNotIn("alembic_version", inspect(engine).get_table_names())
        before = image.read_bytes()
        for _ in range(2):
            with TestClient(create_app(self.settings)) as client:
                self.assertEqual(self.revision(client.app.state.engine), (HEAD_REVISION,))
                response = client.get(f"/api/geo/{session_id}/messages")
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["messages"][0]["content"], "existing message")
                self.assertEqual(client.get("/media/old.png").content, before)
                response = client.post(f"/api/geo/{session_id}/feedback", json={"type": "confirm", "iso": "JP"})
                self.assertEqual(response.status_code, 200, response.text)
        with create_session_factory(engine)() as session:
            self.assertEqual(session.get(GeoSession, session_id).confirmed_iso, "JP")
        self.assertEqual(image.read_bytes(), before)

    def test_unversioned_alembic_baseline_defaults_are_accepted(self):
        engine, _ = init_db(self.settings)
        self.addCleanup(engine.dispose)
        with engine.begin() as connection:
            command.downgrade(migrations._config(connection), BASELINE_REVISION)
            connection.execute(text("DROP TABLE alembic_version"))
        migrated, _ = init_db(self.settings)
        self.addCleanup(migrated.dispose)
        self.assertEqual(self.revision(migrated), (HEAD_REVISION,))

    def test_empty_version_table_with_legacy_schema_is_adopted(self):
        engine = self.make_legacy()
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)"))
        migrated, _ = init_db(self.settings)
        self.addCleanup(migrated.dispose)
        self.assertEqual(self.revision(migrated), (HEAD_REVISION,))

    def test_incompatible_unversioned_schema_is_not_stamped_or_changed(self):
        statements = {
            "missing table": "DROP TABLE chat_message",
            "extra column": "ALTER TABLE geo_session ADD COLUMN unexpected TEXT",
            "missing index": "DROP INDEX ix_chat_message_session_id",
            "wrong index": "CREATE UNIQUE INDEX extra_unique ON geo_session (image_filename)",
            "partial index": (
                "DROP INDEX ix_chat_message_session_id",
                "CREATE INDEX ix_chat_message_session_id ON chat_message (session_id) WHERE role = 'ai'",
            ),
            "foreign table": "CREATE TABLE unrelated (id INTEGER PRIMARY KEY)",
            "foreign view": "CREATE VIEW unrelated_view AS SELECT id FROM geo_session",
            "trigger": "CREATE TRIGGER unrelated_trigger AFTER INSERT ON geo_session BEGIN SELECT 1; END",
        }
        for label, statement in statements.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as root:
                settings = make_settings(root)
                engine = create_db_engine(settings)
                try:
                    self.copied_metadata().create_all(engine)
                    with engine.begin() as connection:
                        for sql in statement if isinstance(statement, tuple) else (statement,):
                            connection.execute(text(sql))
                    before = self.definition(engine)
                    with self.assertRaises(SchemaCompatibilityError):
                        unexpected_engine, _ = init_db(settings)
                        unexpected_engine.dispose()
                    self.assertEqual(self.definition(engine), before)
                    self.assertNotIn("alembic_version", inspect(engine).get_table_names())
                finally:
                    engine.dispose()

    def test_incompatible_column_type_nullability_default_and_constraints_are_rejected(self):
        for kind in ("type", "nullable", "default", "foreign_key", "check", "without_rowid"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                metadata = self.copied_metadata()
                if kind == "type":
                    metadata.tables["chat_message"].c.role.type = String(80)
                elif kind == "nullable":
                    metadata.tables["geo_session"].c.user_id.nullable = False
                elif kind == "default":
                    metadata.tables["geo_session"].c.status.server_default = DefaultClause(text("'unknown'"))
                elif kind == "foreign_key":
                    table = metadata.tables["chat_message"]
                    for constraint in list(table.foreign_key_constraints):
                        table.constraints.remove(constraint)
                    table.foreign_keys.clear()
                    table.c.session_id.foreign_keys.clear()
                elif kind == "check":
                    metadata.tables["geo_session"].append_constraint(CheckConstraint("user_id > 0"))
                else:
                    metadata.tables["geo_session"].dialect_options["sqlite"]["with_rowid"] = False
                settings = make_settings(root)
                engine = create_db_engine(settings)
                try:
                    metadata.create_all(engine)
                    before = self.definition(engine)
                    with self.assertRaises(SchemaCompatibilityError):
                        unexpected_engine, _ = init_db(settings)
                        unexpected_engine.dispose()
                    self.assertEqual(self.definition(engine), before)
                finally:
                    engine.dispose()

    def test_incompatible_schema_stops_application_startup(self):
        engine = self.make_legacy()
        with engine.begin() as connection:
            connection.execute(text("DROP INDEX ix_chat_message_session_id"))
        with self.assertRaises(SchemaCompatibilityError):
            with TestClient(create_app(self.settings)):
                self.fail("the application must not become ready")
        self.assertNotIn("alembic_version", inspect(engine).get_table_names())

    def test_unknown_revision_is_not_rewritten(self):
        engine, _ = init_db(self.settings)
        self.addCleanup(engine.dispose)
        with engine.begin() as connection:
            connection.execute(text("UPDATE alembic_version SET version_num = 'unknown_revision'"))
        with self.assertRaises(CommandError):
            init_db(self.settings)
        self.assertEqual(self.revision(engine), ("unknown_revision",))

    def test_programmatic_migrations_use_injected_settings_not_environment(self):
        unrelated = Path(self.tmp.name) / "unrelated"
        unrelated_url = f"sqlite:///{(unrelated / 'wrong.db').as_posix()}"
        with mock.patch.dict(os.environ, {"M_PHOTO_DATA_DIR": str(unrelated), "M_PHOTO_DB_URL": unrelated_url}):
            engine, _ = init_db(self.settings)
        self.addCleanup(engine.dispose)
        self.assertEqual(self.revision(engine), (HEAD_REVISION,))
        self.assertFalse(unrelated.exists())

    def test_startup_does_not_reconfigure_logging(self):
        with mock.patch("logging.config.fileConfig", side_effect=AssertionError("must not reset logging")):
            engine, _ = init_db(self.settings)
        self.addCleanup(engine.dispose)
        self.assertEqual(self.revision(engine), (HEAD_REVISION,))

    def test_legacy_is_stamped_at_baseline_then_upgraded_to_future_revision(self):
        engine = self.make_legacy()
        fixture = Path(self.tmp.name) / "migration-fixture"
        fixture.mkdir()
        shutil.copy2(migrations.ALEMBIC_INI, fixture / "alembic.ini")
        shutil.copytree(migrations.ALEMBIC_INI.parent / "alembic", fixture / "alembic")
        (fixture / "alembic" / "versions" / "0003_test.py").write_text(
            "from alembic import op\nimport sqlalchemy as sa\n"
            "revision = '0003_test'\ndown_revision = '0002_model_evidence'\n"
            "def upgrade():\n    op.add_column('geo_session', sa.Column('note', sa.Text(), nullable=True))\n"
            "def downgrade():\n    op.drop_column('geo_session', 'note')\n",
            encoding="utf-8",
        )
        with mock.patch.object(migrations, "ALEMBIC_INI", fixture / "alembic.ini"):
            migrations.migrate_database(engine)
            migrations.migrate_database(engine)
        self.assertEqual(self.revision(engine), ("0003_test",))
        self.assertIn("note", {c["name"] for c in inspect(engine).get_columns("geo_session")})

    def test_failed_migration_rolls_back_schema_and_revision_changes(self):
        fixture = Path(self.tmp.name) / "failed-migration"
        fixture.mkdir()
        shutil.copy2(migrations.ALEMBIC_INI, fixture / "alembic.ini")
        shutil.copytree(migrations.ALEMBIC_INI.parent / "alembic", fixture / "alembic")
        (fixture / "alembic" / "versions" / "0003_failure.py").write_text(
            "from alembic import op\nimport sqlalchemy as sa\n"
            "revision = '0003_failure'\ndown_revision = '0002_model_evidence'\n"
            "def upgrade():\n    op.add_column('geo_session', sa.Column('note', sa.Text()))\n"
            "    raise RuntimeError('injected migration failure')\n"
            "def downgrade():\n    pass\n",
            encoding="utf-8",
        )
        # Both an empty DB and an adopted DB must remain at their pre-run schema.
        for legacy in (False, True):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as root:
                engine = create_db_engine(make_settings(root))
                try:
                    if legacy:
                        self.copied_metadata().create_all(engine)
                    before = self.definition(engine)
                    with mock.patch.object(migrations, "ALEMBIC_INI", fixture / "alembic.ini"):
                        with self.assertRaisesRegex(RuntimeError, "injected migration failure"):
                            migrations.migrate_database(engine)
                    self.assertEqual(self.definition(engine), before)
                    self.assertNotIn("alembic_version", inspect(engine).get_table_names())
                finally:
                    engine.dispose()

    def test_cli_works_outside_repo_with_percent_in_database_path(self):
        root = Path(self.tmp.name) / "percent%data"
        root.mkdir()
        settings = make_settings(root)
        env = dict(os.environ, M_PHOTO_DATA_DIR=str(settings.data_dir), M_PHOTO_DB_URL=settings.database_url, PYTHONPATH=str(REPO_ROOT))
        command_line = [sys.executable, "-m", "alembic", "-c", str(migrations.ALEMBIC_INI), "upgrade", "head"]
        for _ in range(2):
            result = subprocess.run(command_line, cwd=root, env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        engine = create_db_engine(settings)
        self.addCleanup(engine.dispose)
        self.assertEqual(self.revision(engine), (HEAD_REVISION,))

    def test_cli_failed_upgrade_rolls_back_ddl(self):
        fixture = Path(self.tmp.name) / "cli-failure"
        fixture.mkdir()
        shutil.copy2(migrations.ALEMBIC_INI, fixture / "alembic.ini")
        shutil.copytree(migrations.ALEMBIC_INI.parent / "alembic", fixture / "alembic")
        (fixture / "alembic" / "versions" / "0003_failure.py").write_text(
            "from alembic import op\nimport sqlalchemy as sa\n"
            "revision = '0003_failure'\ndown_revision = '0002_model_evidence'\n"
            "def upgrade():\n    op.add_column('geo_session', sa.Column('note', sa.Text()))\n"
            "    raise RuntimeError('injected CLI migration failure')\n"
            "def downgrade():\n    pass\n",
            encoding="utf-8",
        )
        env = dict(os.environ, M_PHOTO_DATA_DIR=str(self.settings.data_dir), M_PHOTO_DB_URL=self.settings.database_url, PYTHONPATH=str(REPO_ROOT))
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "-c", str(fixture / "alembic.ini"), "upgrade", "head"],
            cwd=self.tmp.name, env=env, text=True, capture_output=True, timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("injected CLI migration failure", result.stderr)
        engine = create_db_engine(self.settings)
        self.addCleanup(engine.dispose)
        self.assertEqual(inspect(engine).get_table_names(), [])

    def test_safe_migration_module_cli_adopts_legacy(self):
        engine = self.make_legacy()
        env = dict(os.environ, M_PHOTO_DATA_DIR=str(self.settings.data_dir), M_PHOTO_DB_URL=self.settings.database_url, PYTHONPATH=str(REPO_ROOT))
        result = subprocess.run([sys.executable, "-m", "backend.app.migrations"], cwd=self.tmp.name, env=env, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.revision(engine), (HEAD_REVISION,))


if __name__ == "__main__":
    unittest.main()
