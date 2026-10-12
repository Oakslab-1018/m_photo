"""Runtime configuration for the FastAPI backend.

All writable state (uploaded images + the runtime SQLite database) lives under
a single configurable root directory. The default is ``<repo>/backend/storage``.
The geographic catalog is shipped as the immutable SQLite seed
``<repo>/data/geo_catalog.sqlite3`` (read through the standard-library
:mod:`catalog_db` reader); the legacy JSON catalogs are no longer read at
runtime and are no longer referenced by this module.

======================  ==============================  ======================
Primary env var         Accepted aliases                Purpose
======================  ==============================  ======================
M_PHOTO_DATA_DIR        DATA_DIR, APP_DATA_DIR,         root for uploads + db
                        UPLOAD_ROOT, M_PHOTO_DATA
M_PHOTO_UPLOAD_DIR      UPLOAD_FOLDER                   uploaded image directory
M_PHOTO_DB_PATH         DATABASE_PATH                   SQLite file path
M_PHOTO_DB_URL          DATABASE_URL                    full SQLAlchemy URL
                                                        (wins over db path)
M_PHOTO_GEO_CATALOG_SEED -                              shipped geo catalog seed
                                                        SQLite path override
M_PHOTO_CORS_ORIGINS    CORS_ORIGINS                    comma separated origins
GEO_AI_MODE             AI_MODE                         "mock" (default) / "real"
OCR_PROVIDER            GEO_OCR_PROVIDER                "mock" (default) / "rapidocr"
OCR_MODEL_TYPE          GEO_OCR_MODEL_TYPE              "server" (default) / "mobile"
M_PHOTO_OCR_MODEL_DIR   OCR_MODEL_DIR                   local ONNX cache dir
OCR_DET_MODEL_PATH      -                               explicit det .onnx path
OCR_REC_MODEL_PATH      -                               explicit rec .onnx path
OCR_CLS_MODEL_PATH      -                               explicit cls .onnx path
OCR_MIN_TEXT_SCORE      -                               line score floor (0.5)
OCR_LANGUAGE_MIN_CONFIDENCE -                           language clue floor (0.7)
M_PHOTO_GEO_CATALOG_MODE -                              only "db" is supported;
                                                        "json" is rejected
======================  ==============================  ======================

``GEO_AI_MODE`` picks the general analyzer mode and ``OCR_PROVIDER`` the OCR
provider independently. Both default to ``mock`` so the service starts (and is
fully covered by tests) without RapidOCR or downloaded model weights.
``rapidocr`` runs the configurable PP-OCRv5 det/cls/rec ONNX Runtime (CPU)
profile; the default profile is ``server`` for accuracy. ``mobile`` remains
available for lower-latency deployments. Models are cached under
``data_dir/ocr-models`` unless explicit ``.onnx`` paths are given.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from catalog_db import default_catalog_path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_AI_MODE = "mock"
DEFAULT_OCR_PROVIDER = "mock"
DEFAULT_OCR_MODEL_TYPE = "server"
VALID_OCR_MODEL_TYPES = ("server", "mobile")
DEFAULT_OCR_MIN_TEXT_SCORE = 0.5
DEFAULT_OCR_LANGUAGE_MIN_CONFIDENCE = 0.7
DEFAULT_GEO_CATALOG_MODE = "db"
#: Deprecated compatibility constant. Only the database-backed catalog exists;
#: ``json`` is rejected explicitly instead of falling back to file reads.
VALID_GEO_CATALOG_MODES = (DEFAULT_GEO_CATALOG_MODE,)
DEFAULT_DATA_DIRNAME = "storage"  # resolved as <repo>/backend/storage
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}


def normalize_ocr_model_type(value: str | None) -> str:
    """Return a validated PP-OCRv5 model profile name.

    ``server`` is the accuracy-oriented default; ``mobile`` is retained as a
    supported lower-latency profile. Unknown values fail fast so a deployment
    never silently runs a profile it did not request.
    """
    candidate = str(value or DEFAULT_OCR_MODEL_TYPE).strip().lower()
    if candidate not in VALID_OCR_MODEL_TYPES:
        raise ValueError(
            f"OCR_MODEL_TYPE must be one of {VALID_OCR_MODEL_TYPES!r}; got {candidate!r}"
        )
    return candidate


def normalize_geo_catalog_mode(value: str | None) -> str:
    """Return the only supported geo catalog mode (``db``).

    The JSON catalog fallback has been removed: runtime lookups and startup
    bootstrap both read SQLite exclusively, so an explicit ``json``
    configuration is rejected instead of being silently ignored.
    """
    candidate = str(value or DEFAULT_GEO_CATALOG_MODE).strip().lower()
    if candidate not in VALID_GEO_CATALOG_MODES:
        raise ValueError(
            "GEO_CATALOG_MODE only supports 'db'; the JSON catalog mode was removed"
        )
    return candidate


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


def _sqlite_database_name(database_url: str) -> str | None:
    """Return the file name from a file-backed SQLite SQLAlchemy URL.

    ``None`` means the URL does not address a file-backed SQLite database
    (non-SQLite URL, in-memory database or SQLite URI with ``file:`` scheme),
    in which case :attr:`Settings.catalog_db_path` fails explicitly instead of
    falling back to an unrelated path.
    """
    try:
        from sqlalchemy.engine import make_url
        from sqlalchemy.exc import ArgumentError
    except ImportError:  # pragma: no cover - SQLAlchemy is a runtime dependency
        return None
    try:
        url = make_url(database_url)
    except (ArgumentError, TypeError, ValueError):
        return None
    if url.get_backend_name() != "sqlite":
        return None
    database = url.database
    if not database or database == ":memory:" or database.startswith("file:"):
        return None
    return database


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
    geo_catalog_mode: str = DEFAULT_GEO_CATALOG_MODE
    # Deployment override for the shipped read-only seed asset
    # (M_PHOTO_GEO_CATALOG_SEED). It must point at a SQLite catalog seed, never
    # at a JSON catalog file.
    catalog_seed_path_override: Path | None = None
    ocr_provider: str = DEFAULT_OCR_PROVIDER
    ocr_model_type: str = DEFAULT_OCR_MODEL_TYPE
    ocr_model_dir: Path | None = None
    ocr_det_model_path: Path | None = None
    ocr_rec_model_path: Path | None = None
    ocr_cls_model_path: Path | None = None
    ocr_min_text_score: float = DEFAULT_OCR_MIN_TEXT_SCORE
    ocr_language_min_confidence: float = DEFAULT_OCR_LANGUAGE_MIN_CONFIDENCE
    streetclip_enabled: bool = False
    streetclip_url: str = "http://127.0.0.1:8001"
    streetclip_api_key: str = field(default="", repr=False)
    streetclip_timeout_seconds: float = 45.0
    deepseek_api_key: str = field(default="", repr=False)
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"
    deepseek_timeout_seconds: float = 15.0

    def __post_init__(self) -> None:
        if self.geo_catalog_mode != DEFAULT_GEO_CATALOG_MODE:
            raise ValueError(
                "GEO_CATALOG_MODE only supports 'db'; the JSON catalog mode was removed"
            )

    @property
    def catalog_seed_path(self) -> Path:
        """Immutable shipped geo catalog seed (SQLite).

        The deployment override (``M_PHOTO_GEO_CATALOG_SEED``) is captured by
        :meth:`from_env`; direct construction may set
        ``catalog_seed_path_override``. When neither is present the root
        :func:`catalog_db.default_catalog_path` default is used.
        """
        return self.catalog_seed_path_override or default_catalog_path()

    @property
    def catalog_db_path(self) -> Path:
        """File path of the runtime SQLite catalog database.

        The path always comes from ``database_url`` so a URL override can
        never be shadowed by an unrelated ``db_path`` value. Non-SQLite or
        in-memory URLs raise: there is deliberately no fallback to the seed or
        to an uninitialized runtime database.
        """
        database = _sqlite_database_name(self.database_url)
        if database is None:
            raise ValueError(
                "catalog_db_path requires a file-backed SQLite database_url; "
                f"got {self.database_url!r}"
            )
        return _resolve(database, self.data_dir)

    # ------------------------------------------------------------------
    # Deprecated JSON-era path aliases. They intentionally resolve to the
    # runtime SQLite catalog so old call sites can never read the retired JSON
    # files; new call sites should use ``catalog_for_settings(settings)``.
    # ------------------------------------------------------------------
    @property
    def countries_path(self) -> Path:
        """Deprecated alias for the runtime catalog SQLite path."""
        return self.catalog_db_path

    @property
    def admin1_path(self) -> Path:
        """Deprecated alias for the runtime catalog SQLite path."""
        return self.catalog_db_path

    @property
    def cities_path(self) -> Path:
        """Deprecated alias for the runtime catalog SQLite path."""
        return self.catalog_db_path

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
        geo_catalog_mode = normalize_geo_catalog_mode(get("M_PHOTO_GEO_CATALOG_MODE"))
        catalog_seed_raw = get("M_PHOTO_GEO_CATALOG_SEED")
        catalog_seed_path_override = (
            _resolve(catalog_seed_raw, REPO_ROOT) if catalog_seed_raw else None
        )
        media_prefix = get("M_PHOTO_MEDIA_URL") or "/media"

        def normalize_provider(value: str) -> str:
            candidate = value.strip().lower()
            if candidate in {"real", "true", "1"}:
                return "rapidocr"
            return candidate

        ocr_provider_raw = get("OCR_PROVIDER", "GEO_OCR_PROVIDER")
        ocr_provider = normalize_provider(
            ocr_provider_raw if ocr_provider_raw else ai_mode
        ) or DEFAULT_OCR_PROVIDER
        ocr_model_type = normalize_ocr_model_type(
            get("OCR_MODEL_TYPE", "GEO_OCR_MODEL_TYPE")
        )

        ocr_model_raw = get("M_PHOTO_OCR_MODEL_DIR", "OCR_MODEL_DIR")
        ocr_model_dir = (
            _resolve(ocr_model_raw, data_dir) if ocr_model_raw else (data_dir / "ocr-models")
        )

        def optional_model_path(name: str) -> Path | None:
            raw = get(name)
            return _resolve(raw, ocr_model_dir) if raw else None

        def bounded_float(name: str, default: float) -> float:
            raw = get(name)
            if not raw:
                return default
            try:
                value = float(raw)
            except ValueError:
                return default
            return min(1.0, max(0.0, value))

        ocr_min_text_score = bounded_float("OCR_MIN_TEXT_SCORE", DEFAULT_OCR_MIN_TEXT_SCORE)
        ocr_language_min_confidence = bounded_float(
            "OCR_LANGUAGE_MIN_CONFIDENCE", DEFAULT_OCR_LANGUAGE_MIN_CONFIDENCE
        )

        return cls(
            data_dir=data_dir,
            upload_dir=upload_dir,
            db_path=db_path,
            database_url=database_url,
            media_url_prefix=media_prefix.rstrip("/") or "/media",
            cors_origins=cors_origins,
            geo_ai_mode=ai_mode,
            geo_catalog_mode=geo_catalog_mode,
            catalog_seed_path_override=catalog_seed_path_override,
            ocr_provider=ocr_provider,
            ocr_model_type=ocr_model_type,
            ocr_model_dir=ocr_model_dir,
            ocr_det_model_path=optional_model_path("OCR_DET_MODEL_PATH"),
            ocr_rec_model_path=optional_model_path("OCR_REC_MODEL_PATH"),
            ocr_cls_model_path=optional_model_path("OCR_CLS_MODEL_PATH"),
            ocr_min_text_score=ocr_min_text_score,
            ocr_language_min_confidence=ocr_language_min_confidence,
            streetclip_enabled=(get("STREETCLIP_ENABLED") or "false").lower() in {"1", "true", "yes", "on"},
            streetclip_url=(get("STREETCLIP_URL") or "http://127.0.0.1:8001").rstrip("/"),
            streetclip_api_key=(get("STREETCLIP_API_KEY") or "").strip(),
            streetclip_timeout_seconds=float(get("STREETCLIP_TIMEOUT_SECONDS") or "45"),
            deepseek_api_key=(get("DEEPSEEK_API_KEY") or "").strip(),
            deepseek_base_url=(get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/"),
            deepseek_model=get("DEEPSEEK_MODEL") or "deepseek-flash",
            deepseek_timeout_seconds=float(get("DEEPSEEK_TIMEOUT_SECONDS") or "15"),
        )


def get_settings(env: dict[str, str] | None = None) -> Settings:
    """Build :class:`Settings`; ``env`` is only used by tests."""
    return Settings.from_env(env)


__all__ = [
    "ALLOWED_CONTENT_TYPES",
    "ALLOWED_EXTENSIONS",
    "DEFAULT_AI_MODE",
    "DEFAULT_GEO_CATALOG_MODE",
    "DEFAULT_OCR_LANGUAGE_MIN_CONFIDENCE",
    "DEFAULT_OCR_MIN_TEXT_SCORE",
    "DEFAULT_OCR_MODEL_TYPE",
    "DEFAULT_OCR_PROVIDER",
    "VALID_GEO_CATALOG_MODES",
    "VALID_OCR_MODEL_TYPES",
    "normalize_geo_catalog_mode",
    "normalize_ocr_model_type",
    "DEFAULT_DATA_DIRNAME",
    "MAX_UPLOAD_BYTES",
    "REPO_ROOT",
    "Settings",
    "get_settings",
]
