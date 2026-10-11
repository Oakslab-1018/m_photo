# Real OCR: complete PP-OCRv5 profiles via RapidOCR + ONNX Runtime (CPU)

`OCR_PROVIDER` selects the OCR implementation:

| Provider | Value | Behavior |
| --- | --- | --- |
| Mock | `mock` | No import/download/inference; unit and offline setups stay deterministic. |
| Real | `rapidocr` | Lazily loads a complete PP-OCRv5 det + cls + rec profile on CPU. |

`OCR_MODEL_TYPE` selects the profile:

| Profile | Value | Use |
| --- | --- | --- |
| Server | `server` (default) | Accuracy-oriented full models; recommended for deployment. |
| Mobile | `mobile` | Lower-latency compatibility profile for constrained CPUs. |

Legacy `GEO_AI_MODE=real` maps to `OCR_PROVIDER=rapidocr`. `GEO_OCR_MODEL_TYPE`
is accepted as an alias for `OCR_MODEL_TYPE`. Unknown profile values fail fast.

## Pinned model trio

RapidOCR is pinned to `rapidocr==3.10.0`; ONNX Runtime is CPU-only. The server
profile maps to the following resolver keys and SHA-256 values from the pinned
`rapidocr.default_models.yaml`:

| Task | Model key | SHA-256 |
| --- | --- | --- |
| det | `ch_PP-OCRv5_det_server` | `0f8846b1d4bba223a2a2f9d9b44022fbc22cc019051a602b41a7fda9667e4cad` |
| cls | `ch_PP-LCNet_x1_0_textline_ori_cls_server` | `7d3c02ef6c7da8ae08b4347cc7695b2081aae68c325d64375724ecf39c99e743` |
| rec | `ch_PP-OCRv5_rec_server` | `e09385400eaaaef34ceff54aeb7c4f0f1fe014c27fa8b9905d4709b65746562a` |

Mobile keys remain `ch_PP-OCRv5_det_mobile`, `ch_PP-OCRv5_rec_mobile`, and
`ch_PP-LCNet_x0_25_textline_ori_cls_mobile`.

Official PaddleOCR benchmarks put PP-OCRv5 server rec at 86.38% average
accuracy vs 81.29% mobile, and server det at 83.8% vs 79.0% mobile. Server det
is substantially heavier on CPU (roughly 84 MB / 383 ms vs 4.7 MB / 58 ms in
the published benchmark); server rec is roughly 81 MB / 31 ms vs 16 MB / 21 ms.
Expect the full server profile to improve accuracy at a meaningful CPU latency
and cache-size cost. Detection still uses `limit_type=max`,
`limit_side_len=1280` for signage crops.

## Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `OCR_PROVIDER` | `mock` in code; `rapidocr` in `.env.example`/Compose | `mock` or `rapidocr`. |
| `OCR_MODEL_TYPE` | `server` | Validated profile: `server` or `mobile`. |
| `M_PHOTO_OCR_MODEL_DIR` / `OCR_MODEL_DIR` | `<data_dir>/ocr-models` | ONNX cache root; Docker uses `/data/ocr-models` on the named volume. |
| `OCR_DET_MODEL_PATH` | unset | Optional explicit det `.onnx` path. |
| `OCR_REC_MODEL_PATH` | unset | Optional explicit rec `.onnx` path. |
| `OCR_CLS_MODEL_PATH` | unset | Optional explicit cls `.onnx` path. |
| `OCR_MIN_TEXT_SCORE` | `0.5` | Minimum line score before a signage clue is emitted. |
| `OCR_LANGUAGE_MIN_CONFIDENCE` | `0.7` | Minimum OCR-line confidence before a language clue is emitted. |

Selected model IDs are reported by key. Explicit paths are reported by
**basename only**, so local paths never leak through clue provenance or status.

## Setup and real smoke

Host setup and verification:

```powershell
uv sync --frozen
powershell -File scripts/setup-ocr.ps1 -ModelType server
powershell -File scripts/verify-ocr.ps1 -ModelType server
```

The downloader uses RapidOCR's pinned resolver and verifies every selected file
against the resolver SHA-256 before reporting success. The smoke renders Latin,
Simplified Chinese, and Japanese signage, checks recognition plus exact profile
provenance, and prints per-case and total CPU inference time.

Container prefetch after the next image build (the downloader is copied into
`/app/scripts/`):

```powershell
docker compose run --rm backend python scripts/ocr_download_models.py --model-type server
docker compose up -d --no-build --force-recreate --wait
```

Do not bake weights into the image and do not run `docker compose down -v`;
weights must persist in the `backend-data` volume.

## Clue contract

Each recognized line still produces a `signage` clue with `source='ocr'`,
`extra.origin_source='ocr'`, `extra.provider='rapidocr'`, `extra.reliable=true`,
`extra.bbox` / `extra.quad`, `extra.line_confidence`, `extra.model_version`,
`extra.model_type`, and `extra.det_model` / `extra.rec_model` / `extra.cls_model`.
Language clues are derived per reliable line with confidence bounded by the OCR
line score.

After the base analyzers run, `AnalysisPipeline` appends semantic clues through
the required local `SemanticClueAnalyzer`; import failures are surfaced as
deployment bugs, while semantic runtime failures are logged with sanitized
type/message diagnostics and do not discard the OCR clues.

## Failure handling and verification

If RapidOCR, the selected weights, or inference is unavailable, the analyzer
returns no real-looking clues, logs a sanitized warning, and exposes
`OCRAnalyzer.last_status` / `PipelineService.ocr_status()` with provider,
state, engine, model version, profile, and basename-only model labels.

Offline tests:

```powershell
uv run --frozen pytest backend/tests/test_ocr.py backend/tests/test_config.py -q
```

They cover server default selection, explicit mobile compatibility, invalid
profile rejection, per-line provenance, explicit-path basename safety,
concurrency, lazy initialization, and sanitized failure diagnostics.
