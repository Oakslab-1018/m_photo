# m_photo backend (FastAPI modular monolith)

FastAPI rewrite of the legacy Flask backend. The analysis and reasoning code
now lives under `backend/app/domain/geolocation/`: the former `reasoning/`
modules (clue, engine, hypotheses, knowledge graph) are directly in that
package, and the former `ai/` modules are in its `analysis/` subpackage.
Repository-level `services/chat.py` remains the chat adapter dependency — no
algorithms were copied, and `easyocr` is **not** required (`GEO_AI_MODE`
defaults to `mock`; no real-OCR support is claimed in this stage).

Container/startup contract (owned by the deployment worker):

```bash
uv run --frozen uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

## Layout

```
backend/
  app/
    main.py              # ASGI app + create_app() (CORS, /media, JSON-safe 422s)
    config.py            # Settings.from_env(): M_PHOTO_DATA_DIR & aliases
    database.py          # SQLAlchemy 2.x Base/engine/session + GeoSession/ChatMessage
    deps.py              # per-request session & service dependencies
    schemas.py           # Pydantic v2 request/response models
    routers/health.py    # GET  /api/health          (real SELECT 1 -> 503 on failure)
    routers/countries.py # GET  /api/geo/countries
    routers/geo.py       # upload / feedback / messages / analyze-point
    domain/geolocation/  # clue / engine / hypotheses / knowledge_graph
      analysis/          # AnalysisPipeline + concrete analyzers
    services/            # thin adapters over domain packages
  alembic/               # baseline migration (0001_initial) + env.py
  alembic.ini
  tests/                 # pytest *and* `python -m unittest` compatible
```

HTTP handlers are **sync** `def` functions (threadpool) and every request gets
(and closes) its own SQLAlchemy `Session`; there is no shared session across
threads.

## Install / run

`uv` uses the repo `.uv-cache`; the project environment is `backend/.venv`
(the legacy root `.venv` has been removed; only gitignored `backend/.venv` remains):

```bash
# PowerShell: $env:UV_PROJECT_ENVIRONMENT="backend/.venv"
export UV_PROJECT_ENVIRONMENT=backend/.venv
uv sync --frozen
uv run --frozen uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Local development:

```bash
export M_PHOTO_DATA_DIR=./backend/storage
uv run --frozen uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Database & migrations

The authoritative schema is the Alembic baseline:

```bash
uv run --frozen alembic -c backend/alembic.ini upgrade head
```

`backend/app/database.py:init_db()` also calls `Base.metadata.create_all()` as
a **temporary** zero-setup convenience for local runs and tests
(`init_db(..., create_schema=False)` disables it). This is **not** a migration
substitute and the baseline is **not** yet wired into app startup or the
Docker entrypoint — that wiring is a follow-up stage. Legacy `database.db`
compatibility is intentionally not provided.

## API

| Method | Path                                  | Notes |
| ------ | ------------------------------------- | ----- |
| GET    | `/api/health`                         | 200 after a real `SELECT 1`; 503 if the DB fails |
| GET    | `/api/geo/countries`                  | country knowledge base (cached) |
| POST   | `/api/geo/upload`                     | multipart field `image`; 201 on success |
| POST   | `/api/geo/{session_id}/feedback`      | JSON `{type: confirm\|reject\|answer\|free_text, ...}` |
| GET    | `/api/geo/{session_id}/messages`      | chat history ordered by id |
| POST   | `/api/geo/{session_id}/analyze-point` | JSON `{x, y}` normalized to `[0, 1]` |

Errors are `{"error": ..., "detail": ...}`. Unknown sessions → **404**;
invalid/missing feedback fields and non-finite or out-of-range `x`/`y` → **422**
(the validation payload drops `input` and stringifies `ctx`, so it always
serializes as JSON). Uploaded images live in
`M_PHOTO_DATA_DIR/uploads` and are served at `/media/<filename>`.

## Configuration (env)

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `M_PHOTO_DATA_DIR` | `<repo>/backend/storage` | root for uploads + SQLite file (compose sets `/data`) |
| `M_PHOTO_UPLOAD_DIR` | `<data_dir>/uploads` | uploaded images |
| `M_PHOTO_DB_PATH` | `<data_dir>/m_photo.db` | SQLite file |
| `M_PHOTO_DB_URL` | derived | full SQLAlchemy URL (wins over path) |
| `M_PHOTO_CORS_ORIGINS` | dev origins | comma-separated allowed origins |
| `GEO_AI_MODE` | `mock` | passed to the reused AI pipeline; unknown values pass through unchanged |

The default data root deliberately avoids the repo-level `data/` directory,
which holds the read-only `data/countries.json`. Legacy aliases accepted:
`DATA_DIR`, `APP_DATA_DIR`, `UPLOAD_ROOT`, `M_PHOTO_DATA`, `UPLOAD_FOLDER`,
`DATABASE_PATH`, `DATABASE_URL`, `AI_MODE`, `CORS_ORIGINS`.

## Tests

```bash
uv run --frozen pytest backend/tests -q                              # pytest
uv run --frozen python -m unittest discover -s backend/tests -t . -q # unittest
```

Covered: health (incl. forced DB failure → 503), countries, upload
(happy path / invalid payload / no file), feedback (confirm, reject, answer,
free_text, missing fields, unknown session), messages, analyze-point
(404 / 422 / NaN / Infinity / 1×1 crop / temp-crop cleanup), JSON-safe 422
payloads, config env aliases and **storage persistence across app restart**.

> Deprecation warnings from the migrated analysis code (Pillow `getdata`,
> `datetime.utcnow`) are known and intentionally not addressed in this stage.
