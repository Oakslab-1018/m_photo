"""Runtime configuration for the FastAPI backend.

All writable state (uploaded images + SQLite database) lives under a single
configurable root directory. The default is ``<repo>/backend/storage`` — the
repo-level ``data/`` directory is deliberately avoided because it holds the
read-only country knowledge base (``data/countries.json``).

======================  ==============================  ======================
Primary env var         Accepted aliases                Purpose
======================  ==============================  ======================
M_PHOTO_DATA_DIR        DATA_DIR, APP_DATA_DIR,         root for uploads + db
                        UPLOAD_ROOT, M_PHOTO_DATA
M_PHOTO_UPLOAD_DIR      UPLOAD_FOLDER                   uploaded image directory
M_PHOTO_DB_PATH         DATABASE_PATH                   SQLite file path
M_PHOTO_DB_URL          DATABASE_URL                    full SQLAlchemy URL
                                                        (wins over db path)
M_PHOTO_CORS_ORIGINS    CORS_ORIGINS                    comma separated origins
GEO_AI_MODE             AI_MODE                         "mock" (default) / "real"
======================  ==============================  ======================

``GEO_AI_MODE`` deliberately defaults to ``mock`` so the service starts (and
stays importable) without ``easyocr`` or any downloaded model weights. Other
values pass through unchanged; only ``mock`` is exercised in this stage.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_AI_MODE = "mock"
DEFAULT_DATA_DIRNAME = "storage"  # resolved as <repo>/backend/storage
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}
ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/gif",
    "image/webp",
    "application/octet-stream",
}
MAX_UPLOAD_BYTES = 16 * 1024 * 1024  # 16 MB, mirrors the legacy MAX_CONTENT_LENGTH


def _resolve(path: str | os.PathLike[str], base: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


@dataclass(frozen=True)
class Settings:
    """Immutable, per-process settings snapshot."""

    data_dir: Path
    upload_dir: Path
    db_path: Path
    database_url: str
    media_url_prefix: str
    cors_origins: tuple[str, ...]
    geo_ai_mode: str

    @property
    def countries_path(self) -> Path:
        """Location of the read-only country knowledge base."""
        return REPO_ROOT / "data" / "countries.json"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        source: dict[str, str] = dict(os.environ) if env is None else dict(env)

        def get(*names: str) -> str | None:
            for name in names:
                value = source.get(name)
                if value:
                    return value
            return None

        data_raw = get(
            "M_PHOTO_DATA_DIR", "DATA_DIR", "APP_DATA_DIR", "UPLOAD_ROOT", "M_PHOTO_DATA"
        )
        data_dir = (
            _resolve(data_raw, REPO_ROOT)
            if data_raw
            else (REPO_ROOT / "backend" / DEFAULT_DATA_DIRNAME)
        )

        upload_raw = get("M_PHOTO_UPLOAD_DIR", "UPLOAD_FOLDER")
        upload_dir = _resolve(upload_raw, data_dir) if upload_raw else (data_dir / "uploads")

        db_url_raw = get("M_PHOTO_DB_URL", "DATABASE_URL", "M_PHOTO_DATABASE_URL")
        db_path_raw = get("M_PHOTO_DB_PATH", "DATABASE_PATH", "M_PHOTO_DATABASE_PATH")
        if db_url_raw:
            database_url = db_url_raw
            if db_path_raw:
                db_path = _resolve(db_path_raw, data_dir)
            elif database_url.startswith("sqlite:///"):
                db_path = _resolve(database_url[len("sqlite:///"):], data_dir)
            else:
                db_path = data_dir / "m_photo.db"
        else:
            db_path = _resolve(db_path_raw, data_dir) if db_path_raw else (data_dir / "m_photo.db")
            database_url = f"sqlite:///{db_path.as_posix()}"

        origins = get("M_PHOTO_CORS_ORIGINS", "CORS_ORIGINS")
        if origins:
            cors_origins = tuple(item.strip() for item in origins.split(",") if item.strip())
        else:
            cors_origins = (
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:8000",
                "http://127.0.0.1:8000",
            )

        ai_mode = (get("GEO_AI_MODE", "AI_MODE") or DEFAULT_AI_MODE).strip().lower()
        media_prefix = get("M_PHOTO_MEDIA_URL") or "/media"

        return cls(
            data_dir=data_dir,
            upload_dir=upload_dir,
            db_path=db_path,
            database_url=database_url,
            media_url_prefix=media_prefix.rstrip("/") or "/media",
            cors_origins=cors_origins,
            geo_ai_mode=ai_mode,
        )


def get_settings(env: dict[str, str] | None = None) -> Settings:
    """Build :class:`Settings`; ``env`` is only used by tests."""
    return Settings.from_env(env)


__all__ = [
    "ALLOWED_CONTENT_TYPES",
    "ALLOWED_EXTENSIONS",
    "DEFAULT_AI_MODE",
    "DEFAULT_DATA_DIRNAME",
    "MAX_UPLOAD_BYTES",
    "REPO_ROOT",
    "Settings",
    "get_settings",
]
