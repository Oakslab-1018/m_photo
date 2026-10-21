"""Runtime settings bridge for the standalone StreetCLIP service.

This module is deliberately lightweight: it imports only the standard library
so ``app.py`` and its tests can import without torch, transformers, CUDA or
model weights. The settings object is shared by the FastAPI application and the
real predictor only after the model is loaded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

MODEL_ID = "geolocal/StreetCLIP"
MODEL_REVISION = "e3561ba2ad9bf14c9efd6b0092607b8497efbfea"
PROMPT_TEMPLATE_VERSION = "streetclip-country-prompts-v1"
PROMPT_TEMPLATES = (
    "a photo from {country}",
    "a street-level photo taken in {country}",
    "a photograph of a street scene in {country}",
)
UNCERTAIN_MARGIN_THRESHOLD = 0.015
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
DEFAULT_BATCH_MAX_SIZE = 8
MAX_BATCH_MAX_SIZE = 32
DEFAULT_BATCH_MAX_AGGREGATE_BYTES = 64 * 1024 * 1024
DEFAULT_BATCH_PROCESSOR_CHUNK = 8
MAX_BATCH_PROCESSOR_CHUNK = 32
DEFAULT_CITY_LIMIT_PER_COUNTRY = 100
DEFAULT_CITY_PROMPT_CHUNK = 128
CITY_REFERENCE_FUSION_WEIGHT = 0.6
CITY_TEXT_FUSION_WEIGHT = 0.4
CITY_PROMPT_TEMPLATE_VERSION = "streetclip-city-prompts-v1"
CITY_TEXT_FEATURE_CACHE_COUNTRIES = 8

CITY_PROMPT_TEMPLATES = (
    "a photo taken in {city}, {region}, {country}",
    "a street-level photograph of {city}, {region}, {country}",
    "a photograph of a street scene in {city}, in the {region} region of {country}",
    "a cityscape photo of {city}, {country}",
)

_SERVICE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = _SERVICE_ROOT.parents[1]
DEFAULT_CATALOG_DB_PATH = REPO_ROOT / "data" / "geo_catalog.sqlite3"
DEFAULT_STORAGE_DIR = _SERVICE_ROOT / "storage"

# Deprecated compatibility names. They point at the SQLite seed, never at the
# retired JSON catalogs, so older imports cannot reintroduce a file dependency.
DEFAULT_COUNTRIES_PATH = DEFAULT_CATALOG_DB_PATH
DEFAULT_CITIES_PATH = DEFAULT_CATALOG_DB_PATH


def _env_value(env: dict[str, str], *names: str) -> str | None:
    """Return the first non-blank env value among ``names``."""
    for name in names:
        value = env.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _positive_int(raw: str | None, name: str, default: int) -> int:
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def load_root_dotenv(repo_root: Path | None = None) -> Path | None:
    """Load the repository root ``.env`` when present.

    Explicit ``os.environ`` values always win; ``python-dotenv`` never
    overrides existing variables. Missing or unreadable files are ignored so
    tests remain importable on machines without local secrets.
    """
    root = repo_root or REPO_ROOT
    env_path = root / ".env"
    if not env_path.is_file():
        return None
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dependency is pinned for real runs
        return None
    load_dotenv(dotenv_path=env_path, override=False, encoding="utf-8-sig")
    return env_path


@dataclass(frozen=True)
class Settings:
    """Immutable process settings resolved from environment variables."""

    catalog_db_path: Path = DEFAULT_CATALOG_DB_PATH
    storage_dir: Path = DEFAULT_STORAGE_DIR
    api_key: str | None = field(default=None, repr=False)
    device: str = "cuda"
    precompute_dtype: str = "float32"
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    batch_max_size: int = DEFAULT_BATCH_MAX_SIZE
    batch_max_aggregate_bytes: int = DEFAULT_BATCH_MAX_AGGREGATE_BYTES
    batch_processor_chunk: int = DEFAULT_BATCH_PROCESSOR_CHUNK
    cache_embeddings: bool = True
    inference_semaphore_enabled: bool = True
    city_limit_per_country: int = DEFAULT_CITY_LIMIT_PER_COUNTRY
    city_prompt_chunk: int = DEFAULT_CITY_PROMPT_CHUNK
    reference_index_path: Path | None = None

    @property
    def countries_path(self) -> Path:
        """Deprecated alias for the SQLite catalog path."""
        return self.catalog_db_path

    @property
    def cities_path(self) -> Path:
        """Deprecated alias for the SQLite catalog path."""
        return self.catalog_db_path

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        source = dict(os.environ) if env is None else {k: str(v) for k, v in env.items()}

        def get(*names: str) -> str | None:
            return _env_value(source, *names)

        catalog_db_value = get("STREETCLIP_CATALOG_DB")
        catalog_db_path = (
            Path(catalog_db_value).expanduser().resolve()
            if catalog_db_value
            else DEFAULT_CATALOG_DB_PATH
        )

        storage_value = get("STREETCLIP_STORAGE_DIR")
        storage_dir = (
            Path(storage_value).expanduser().resolve()
            if storage_value
            else DEFAULT_STORAGE_DIR
        )

        reference_value = get("STREETCLIP_REFERENCE_INDEX")
        reference_index_path = (
            Path(reference_value).expanduser().resolve() if reference_value else None
        )

        api_key = get("STREETCLIP_API_KEY")
        device = get("STREETCLIP_DEVICE") or "cuda"

        max_raw = get("STREETCLIP_MAX_UPLOAD_BYTES")
        max_upload_bytes = MAX_UPLOAD_BYTES
        if max_raw:
            try:
                parsed = int(max_raw)
            except ValueError as exc:
                raise ValueError("STREETCLIP_MAX_UPLOAD_BYTES must be an integer") from exc
            if parsed <= 0:
                raise ValueError("STREETCLIP_MAX_UPLOAD_BYTES must be positive")
            max_upload_bytes = parsed

        cache_raw = get("STREETCLIP_CACHE_EMBEDDINGS")
        cache_embeddings = cache_raw is None or cache_raw.lower() not in {"0", "false", "no", "off"}

        batch_max_size = _positive_int(
            get("STREETCLIP_BATCH_MAX_SIZE"),
            "STREETCLIP_BATCH_MAX_SIZE",
            DEFAULT_BATCH_MAX_SIZE,
        )
        if batch_max_size > MAX_BATCH_MAX_SIZE:
            raise ValueError(
                f"STREETCLIP_BATCH_MAX_SIZE must be <= {MAX_BATCH_MAX_SIZE}"
            )

        batch_processor_chunk = _positive_int(
            get("STREETCLIP_BATCH_PROCESSOR_CHUNK"),
            "STREETCLIP_BATCH_PROCESSOR_CHUNK",
            DEFAULT_BATCH_PROCESSOR_CHUNK,
        )
        if batch_processor_chunk > MAX_BATCH_PROCESSOR_CHUNK:
            raise ValueError(
                f"STREETCLIP_BATCH_PROCESSOR_CHUNK must be <= {MAX_BATCH_PROCESSOR_CHUNK}"
            )

        return cls(
            catalog_db_path=catalog_db_path,
            storage_dir=storage_dir,
            api_key=api_key,
            device=device,
            precompute_dtype=(get("STREETCLIP_PRECOMPUTE_DTYPE") or "float32").lower(),
            max_upload_bytes=max_upload_bytes,
            batch_max_size=batch_max_size,
            batch_max_aggregate_bytes=_positive_int(
                get("STREETCLIP_BATCH_MAX_AGGREGATE_BYTES"),
                "STREETCLIP_BATCH_MAX_AGGREGATE_BYTES",
                DEFAULT_BATCH_MAX_AGGREGATE_BYTES,
            ),
            batch_processor_chunk=batch_processor_chunk,
            cache_embeddings=cache_embeddings,
            inference_semaphore_enabled=True,
            city_limit_per_country=_positive_int(
                get("STREETCLIP_CITY_LIMIT_PER_COUNTRY"),
                "STREETCLIP_CITY_LIMIT_PER_COUNTRY",
                DEFAULT_CITY_LIMIT_PER_COUNTRY,
            ),
            reference_index_path=reference_index_path,
        )


def load_settings(env: dict[str, str] | None = None, *, load_dotenv: bool = True) -> Settings:
    """Resolve settings, optionally loading ``<repo>/.env`` first."""
    if load_dotenv and env is None:
        load_root_dotenv(REPO_ROOT)
    return Settings.from_env(env)


__all__ = [
    "CITY_PROMPT_TEMPLATES",
    "CITY_PROMPT_TEMPLATE_VERSION",
    "CITY_REFERENCE_FUSION_WEIGHT",
    "CITY_TEXT_FEATURE_CACHE_COUNTRIES",
    "CITY_TEXT_FUSION_WEIGHT",
    "DEFAULT_BATCH_MAX_AGGREGATE_BYTES",
    "DEFAULT_BATCH_MAX_SIZE",
    "DEFAULT_BATCH_PROCESSOR_CHUNK",
    "DEFAULT_CATALOG_DB_PATH",
    "DEFAULT_CITIES_PATH",
    "DEFAULT_CITY_LIMIT_PER_COUNTRY",
    "DEFAULT_CITY_PROMPT_CHUNK",
    "DEFAULT_COUNTRIES_PATH",
    "DEFAULT_STORAGE_DIR",
    "MAX_UPLOAD_BYTES",
    "MAX_BATCH_MAX_SIZE",
    "MAX_BATCH_PROCESSOR_CHUNK",
    "MODEL_ID",
    "MODEL_REVISION",
    "PROMPT_TEMPLATE_VERSION",
    "PROMPT_TEMPLATES",
    "REPO_ROOT",
    "Settings",
    "UNCERTAIN_MARGIN_THRESHOLD",
    "load_root_dotenv",
    "load_settings",
]
