# m_photo backend (FastAPI modular monolith)

FastAPI rewrite of the legacy Flask backend. The analysis and reasoning code
now lives under `backend/app/domain/geolocation/`: the former `reasoning/`
modules (clue, engine, hypotheses, knowledge graph) are directly in that
package, and the former `ai/` modules are in its `analysis/` subpackage.
Repository-level `services/chat.py` remains the chat adapter dependency — no
algorithms were copied, and `easyocr` is **not** required. OCR defaults to
`mock` in code for offline tests; real OCR uses pinned PP-OCRv5 server
det/cls/rec via RapidOCR + ONNX Runtime and is enabled with
`OCR_PROVIDER=rapidocr`, `OCR_MODEL_TYPE=server`. Mobile remains supported.
See [OCR documentation](../docs/ocr.md).

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
    database.py          # SQLAlchemy engine/session + models; init runs migrations
    migrations.py        # Alembic upgrades + validated pre-Alembic Geo adoption
    deps.py              # per-request session & service dependencies
    schemas.py           # Pydantic v2 request/response models
    routers/health.py    # GET  /api/health          (real SELECT 1 -> 503 on failure)
    routers/countries.py # GET  /api/geo/countries
    routers/geo.py       # upload / feedback / messages / cities / analyze-point
    domain/geolocation/  # clue / engine / hypotheses / knowledge_graph
      analysis/          # AnalysisPipeline + concrete analyzers
    services/            # thin adapters over domain packages
      model_gateway.py   # StreetCLIP HTTP adapter + evidence contract validation
      city_catalog.py    # read-only data/cities.json (city identity & coordinates)
  alembic/               # baseline migration (0001_initial) + env.py
  alembic.ini
  tests/                 # pytest *and* `python -m unittest` compatible
```

HTTP handlers are **sync** `def` functions (threadpool) and every request gets
(and closes) its own SQLAlchemy `Session`; there is no shared session across
threads.

Mutation methods in `GeoSessionService` flush without committing. Upload,
feedback and point analysis each use one operation-level transaction, validating
and checking JSON serialization of the response before committing. Failures roll
back all database writes; failed uploads also remove their image, and point crops
are cleaned up on success and failure. This does not add optimistic concurrency
control for simultaneous feedback or crash-proof filesystem/database atomicity.

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

Application lifespan now runs Alembic **before** the backend becomes ready.
`init_db(settings)` uses the runtime engine/connection, including injected test
settings, rather than rereading environment variables. `create_all()` is no
longer used during application startup; `create_schema=False` is only a
low-level opt-out for callers that manage schema themselves.

Startup behavior:

- **Empty database**: run `upgrade head`.
- **Alembic-managed database**: upgrade to the current head; repeated startup is
  idempotent. Unknown revision identifiers stop startup.
- **Existing unversioned Geo SQLite database**: compare schema metadata against
  a disposable database built from the immutable `0001_initial` migration.
  Check tables, columns/types/nullability/defaults, keys, indexes, constraints
  and SQLite table options; reject unexpected views or triggers. The known
  client-side defaults from the former `create_all()` schema are accepted.
  Only a matching schema is stamped **at the baseline**, then upgraded to head.
- **Incompatible unversioned database**: stop startup without stamping or altering schema.
  Inspect and resolve the mismatch manually; do not blindly stamp or delete data.

SQLite startup migration/adoption acquires `BEGIN IMMEDIATE`, so the schema
check, revision registration and migration DDL share one transaction and roll
back together on failure. This locking is specific to SQLite; no cross-database
deployment lock is provided. Back up storage before upgrading a deployment.

To run the same safe path without starting the server (from repo root):

```bash
uv run --frozen python -m backend.app.migrations
uv run --frozen alembic -c backend/alembic.ini current
```

The ordinary Alembic CLI remains available for an empty or already-versioned
DB, using the same environment configuration and SQLite PRAGMAs:

```bash
uv run --frozen alembic -c backend/alembic.ini upgrade head
```

For a pre-Alembic Geo database, use application startup or the safe module command
above **first**; a direct Alembic `upgrade head` does not perform automatic
adoption. Legacy Flask `database.db` is not supported or automatically migrated.
Docker already copies `backend/alembic.ini` and the scripts with `backend/`;
migrations run against the mounted volume at runtime, never during image build.

## API

| Method | Path                                  | Notes |
| ------ | ------------------------------------- | ----- |
| GET    | `/api/health`                         | 200 after a real `SELECT 1`; 503 if the DB fails |
| GET    | `/api/geo/countries`                  | country knowledge base (cached) |
| POST   | `/api/geo/upload`                     | multipart field `image`; 201 on success |
| POST   | `/api/geo/{session_id}/feedback`      | JSON `{type: confirm\|reject\|answer\|free_text, ...}` |
| GET    | `/api/geo/{session_id}/messages`      | chat history ordered by id |
| POST   | `/api/geo/{session_id}/cities`        | JSON `{iso}`; ranks that country's catalog cities for this session's image |
| POST   | `/api/geo/{session_id}/analyze-point` | JSON `{x, y}` normalized to `[0, 1]` |

Errors are `{"error": ..., "detail": ...}`. Unknown sessions → **404**;
invalid/missing feedback fields and non-finite or out-of-range `x`/`y` → **422**
(the validation payload drops `input` and stringifies `ctx`, so it always
serializes as JSON). Uploaded images live in
`M_PHOTO_DATA_DIR/uploads` and are served at `/media/<filename>`.

`cities` is read-only and stateless: it never writes a message, never changes the
country ranking and never changes a confirmed country, so browsing one country's
cities cannot alter the analysis. It answers with a `status` rather than an error
— `ok`, `no_cities` (catalog holds none for that ISO), or `unavailable` (the
model service is off or unreachable). A malformed `{iso}` → **422**; a session
whose stored image has disappeared → **404**. City rows are validated against the
backend's own `data/cities.json`: the service's `scope.city_catalog_revision`
must match that file's SHA-256, and every returned `city_id`, name and coordinate
must exist in it, so the catalog — not the model — owns city identity. `city_id`
is serialized as a **decimal string** (the UI keys markers and rows on it) and
coordinates are the catalog's representative points, not the photo's location.

## Configuration (env)

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `M_PHOTO_DATA_DIR` | `<repo>/backend/storage` | root for uploads + SQLite file (compose sets `/data`) |
| `M_PHOTO_UPLOAD_DIR` | `<data_dir>/uploads` | uploaded images |
| `M_PHOTO_DB_PATH` | `<data_dir>/m_photo.db` | SQLite file |
| `M_PHOTO_DB_URL` | derived | full SQLAlchemy URL (wins over path) |
| `M_PHOTO_CORS_ORIGINS` | dev origins | comma-separated allowed origins |
| `GEO_AI_MODE` | `mock` | passed to the reused AI pipeline; unknown values pass through unchanged |
| `OCR_PROVIDER` | `mock` (code); `rapidocr` in example/Compose | `mock` or real RapidOCR |
| `OCR_MODEL_TYPE` | `server` | validated `server` or `mobile` PP-OCRv5 profile |
| `M_PHOTO_OCR_MODEL_DIR` | `<data_dir>/ocr-models` | ONNX cache; Docker uses `/data/ocr-models` |
| `STREETCLIP_ENABLED` | `false` | call the local StreetCLIP service at all |
| `STREETCLIP_URL` | `http://127.0.0.1:8001` | inference service root |
| `STREETCLIP_API_KEY` | unset | bearer token; unset means no `Authorization` header |
| `STREETCLIP_TIMEOUT_SECONDS` | `45` | per-request HTTP timeout |

The default data root deliberately avoids the repo-level `data/` directory,
which holds the read-only catalogs `data/countries.json` (country directory and
ranking targets) and `data/cities.json` (GeoNames settlements used to resolve
city identity). Both are read straight from the repo/image, never written.
Legacy aliases accepted: `DATA_DIR`, `APP_DATA_DIR`, `UPLOAD_ROOT`,
`M_PHOTO_DATA`, `UPLOAD_FOLDER`, `DATABASE_PATH`, `DATABASE_URL`, `AI_MODE`,
`CORS_ORIGINS`.

## Tests

```bash
uv run --frozen pytest backend/tests -q                              # pytest
uv run --frozen python -m unittest discover -s backend/tests -t . -q # unittest
```

Covered: health (incl. forced DB failure → 503), countries, upload
(happy path / invalid payload / no file), feedback (confirm, reject, answer,
free_text, missing fields, unknown session), messages, analyze-point
(404 / 422 / NaN / Infinity / 1×1 crop / temp-crop cleanup), cities
(ok / no_cities / unavailable / 404 / 422, read-only, catalog-resolved identity,
per-row tamper rejection), JSON-safe 422 payloads, config env aliases and
**storage persistence across app restart**. Also covered: single-commit
transactions, injected analysis/message/commit/response failures, model-evidence
contract validation (corrupt evidence and crops fall back to rules; a corrupt
reference sidecar costs only the diagnostics), reference diagnostics riding along
without reordering countries, crops never duplicating the full-image sidecar,
safe migration adoption, incompatible schema rejection, future-revision upgrades,
migration DDL rollback, repeated startup, injected settings, logging
preservation, and CLI execution outside the repo.

> Deprecation warnings from the migrated analysis code (Pillow `getdata`,
> `datetime.utcnow`) are known and intentionally not addressed in this stage.

## 本地 StreetCLIP 与可选 DeepSeek 对话

模型环境独立在 `inference/streetclip/.venv`，使用同一张本机 GPU；业务容器只有 HTTP 适配器。
首次安装执行 `scripts/setup-streetclip.ps1`，启动执行 `scripts/start-streetclip.ps1`。
权重在 `inference/streetclip/storage/huggingface`，无需第三方 StreetCLIP API Key。

根目录 `.env` 已准备：`STREETCLIP_ENABLED=true` 启用本机模型，
`DEEPSEEK_API_KEY` 留空则使用模板对话。以后填写 Key，执行
`docker compose up -d --no-build --force-recreate --wait` 即可重新读取；不要提交 Key 或使用 `down -v`。
默认 DeepSeek 模型名 `deepseek-flash`，可配置；只发送有限文字上下文，不发送图片。

详细启动、缓存、认证、数据迁移、故障兜底和验收说明见 `docs/streetclip-deployment-plan.md`。

候选范围跟随 `data/countries.json`（当前 250 个国家/地区）；原始余弦不是概率，地图上的点是国家代表位置，不是照片 GPS。
选定国家后，`POST /api/geo/{session_id}/cities` 再从 `data/cities.json`（GeoNames，34156 个城市 / 244 个国家/地区）
按该会话原图检索城市候选，同样是代表点。城市行必须能在这份目录里逐条对上（id、名称、经纬度），
否则整份响应作废，因此城市身份由数据目录而不是模型说了算。

带标注的参考街景索引是可选的：推理服务加载后会在 `/predict` 响应里附带 `reference` sidecar，
后端校验其 `model_revision` / `catalog_revision` 与本次部署一致后，把命中结果作为**只读诊断信息**
挂在对应国家的候选上（`reference_matches` / `reference_regions` / `reference_cosine`）。
参考余弦只用于检索展示，**不参与排序**，也不会改变候选顺序；sidecar 损坏时只丢诊断，国家排名照常返回。
未配置索引时这三个字段整个不出现，前端据此区分"没有库"和"有库但没命中"。
