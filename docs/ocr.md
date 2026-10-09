# Real OCR (stage 2A): PP-OCRv5 via RapidOCR + ONNX Runtime (CPU)

The OCR analyzer has two providers, selected with `OCR_PROVIDER`:

| Provider | Value | Behavior |
| --- | --- | --- |
| Mock (default) | `mock` | No imports, no downloads, no clues. Tests and fresh installs stay offline. |
| Real | `rapidocr` | Lazy-loads PP-OCRv5 mobile det/cls/rec ONNX models and runs CPU inference. |

Legacy `GEO_AI_MODE=real` maps to `OCR_PROVIDER=rapidocr`; `GEO_OCR_PROVIDER` is
accepted as an alias. Real OCR is **never** enabled implicitly.

## What is pinned

- Python package: `rapidocr==3.10.0` (exact pin in `pyproject.toml` / `uv.lock`).
- Engine: `onnxruntime` (CPU only). No torch, no Paddle, no OpenVINO, no easyocr.
- Model family: `PP-OCRv5` (`ocr_version: PP-OCRv5`, `model_type: mobile`):
  - det: `ch_PP-OCRv5_det_mobile.onnx` (sha256 `4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae`)
  - cls: `ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx` (sha256 `54379ae5174d026780215fc748a7f31910dee36818e63d49e17dc598ecc82df7`)
  - rec: `ch_PP-OCRv5_rec_mobile.onnx` (sha256 `5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5`)
- URLs/hashes come from RapidOCR 3.10.0 `default_models.yaml`; downloads are
  verified by RapidOCR itself. Do not hand-copy weights.
- Detection resize policy: `limit_type=max`, `limit_side_len=1280` (verified to
  be required for reliable PP-OCRv5 det on small/wide signage crops).

## Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `OCR_PROVIDER` | `mock` | `mock` or `rapidocr` (aliases: `GEO_OCR_PROVIDER`; `real`/`true`/`1` → `rapidocr`). |
| `M_PHOTO_OCR_MODEL_DIR` / `OCR_MODEL_DIR` | `<data_dir>/ocr-models` | ONNX cache root; Docker resolves to `/data/ocr-models` on the named volume. |
| `OCR_DET_MODEL_PATH` | unset | Explicit det `.onnx` path (relative paths resolve under the model dir). |
| `OCR_REC_MODEL_PATH` | unset | Explicit rec `.onnx` path. |
| `OCR_CLS_MODEL_PATH` | unset | Explicit cls `.onnx` path. |
| `OCR_MIN_TEXT_SCORE` | `0.5` | RapidOCR line score floor before a line becomes a clue. |
| `OCR_LANGUAGE_MIN_CONFIDENCE` | `0.7` | Minimum OCR-line confidence before a language clue is emitted. |

Docker Compose sets `M_PHOTO_OCR_MODEL_DIR=/data/ocr-models` so weights persist
in the `backend-data` volume across container restarts. Models are never baked
into the image or committed to the repo (`backend/storage/` is git-ignored).

## Setup (real OCR)

```powershell
uv sync --frozen                    # install locked CPU-only dependencies
powershell -File scripts/setup-ocr.ps1   # download PP-OCRv5 models (git-ignored cache)
powershell -File scripts/verify-ocr.ps1  # REAL smoke on synthetic Windows-font signage
```

Enable at runtime (native):

```powershell
$env:OCR_PROVIDER = "rapidocr"
```

Docker: set `OCR_PROVIDER=rapidocr` in `.env` (Compose forwards it).

## Clue contract (unchanged shape)

- Each reliable OCR line becomes a `signage` clue: `source='ocr'`,
  `confidence=line score`, `extra.origin_source='ocr'`,
  `extra.provider='rapidocr'`, `extra.reliable=true`, `extra.bbox` / `extra.quad`
  (4×`[x,y]`), `extra.line_confidence`, and `extra.model_version='PP-OCRv5'`.
  The point-click router rewrites `source` to `point_click`, so fusion relies on
  `extra.origin_source` to keep OCR provenance.
- Language clues are derived **per reliable OCR line** with
  `source='ocr'`, `extra.origin_source='ocr'`, `extra.provider='rapidocr'`,
  `extra.reliable=true`, and `confidence ≤ OCR line confidence`.
- Conservative language policy (per line):
  - pure Han: **no** language clue (cannot distinguish Chinese from Japanese);
    signage is still emitted.
  - Han + kana, or any kana: `ja`.
  - Hangul: `ko`; Cyrillic: `cyrillic`; Thai: `thai`; Devanagari: `devanagari`.
  - Other Latin-script letters: generic `latin`.
  - Lines below `OCR_LANGUAGE_MIN_CONFIDENCE`, or mixed scripts without a
    60% dominant script, emit nothing.
- Coverage is limited to the configured models: the default `ch` rec model
  covers Simplified Chinese + English + digits + kana glyphs. It does **not**
  claim accurate Korean/Cyrillic/Thai/Devanagari transcription unless you point
  `OCR_REC_MODEL_PATH` at the corresponding PP-OCRv5 rec model.

## Failure handling

If RapidOCR is missing, model files cannot be downloaded, or inference fails:
the analyzer degrades to mock behavior for that call, returns no clues, logs a
sanitized warning, and exposes `OCRAnalyzer.last_status` /
`PipelineService.ocr_status()` as safe diagnostics (`state` =
`unavailable`/`failed`, provider, engine, model version). Paths and URLs are
scrubbed from diagnostics; nothing is silently presented as real OCR output.

## Verification

- Fake-runtime tests (offline, default): `uv run pytest backend/tests/test_ocr.py`
  — provider resolution, per-line confidence/quad/provider, language policy,
  confidence bounding, lazy single-engine init, serialized concurrent inference,
  safe unavailable/failed diagnostics, Settings/env clamps.
- Real smoke (manual, downloads models): `scripts/verify-ocr.ps1` renders Latin
  and Chinese/Japanese signage with Windows fonts and prints exact engine/model
  identity plus every recognized span, confidence, and quad bbox. It fails
  loudly (never pretends validated) if models are unavailable.

## Not used / not allowed

- No easyocr, no torch/Paddle backends, no GPU requirement.
- No TLS verification monkeypatching (`ssl._create_default_https_context`).
- No secrets and no private corpora; smoke images are synthetic and stay under
  git-ignored `backend/storage/ocr-smoke`.