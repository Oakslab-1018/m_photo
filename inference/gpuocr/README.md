# GPU OCR service

This directory is an isolated CUDA RapidOCR runtime. It must not be merged into
the root CPU backend venv or into the StreetCLIP predictor venv.

## Runtime

- Python 3.12
- RapidOCR 3.10.0
- ONNX Runtime GPU 1.26.0, CUDA 12.x build
- PP-OCRv5 server det/cls/rec weights from backend/storage/ocr-models
- CUDA/cuDNN DLLs reused from inference/streetclip/.venv/Lib/site-packages/torch/lib
- No duplicate Torch or CUDA toolkit install
- Port 8003 by default
- Raw image bytes accepted at POST /recognize
- Health at GET /health
- Bearer auth uses GPUOCR_API_KEY, falling back to STREETCLIP_API_KEY

The service verifies all three ONNX sessions after load. Each session must
report CUDAExecutionProvider first. A CPU fallback is rejected; the service
does not silently continue on CPU.

## Memory controls

- gpu_mem_limit: 3 GiB per ORT session
- arena_extend_strategy: kSameAsRequested
- cudnn_conv_algo_search: HEURISTIC
- cudnn_conv_use_max_workspace: false
- Rec.rec_batch_num: 4
- Cls.cls_batch_num: 4
- Det.limit_side_len: 1280
- Det.limit_type: max
- Run options: memory.enable_memory_arena_shrinkage=gpu:0

The run option is installed by wrapping RapidOCR's ONNX Runtime session call
with onnxruntime.RunOptions. It releases dynamic-shape arena blocks after a
run without changing image resize, model files, model profile, or OCR
semantics.

## Scripts

From the repository root:

    powershell -File scripts/setup-gpuocr.ps1
    powershell -File scripts/start-gpuocr.ps1
    powershell -File scripts/stop-gpuocr.ps1

start-gpuocr.ps1 refuses to create a duplicate listener and waits for the real
CUDA provider verification before reporting readiness. The root CPU venv is
never modified.

## Preflight evidence

Manifest index 1355 was run against the restarted 8003 service:

- HTTP 200
- 2.731 seconds after warm startup
- 0 OCR lines for that image, with successful CUDA execution
- det/cls/rec providers: CUDAExecutionProvider, CPUExecutionProvider
- model profile: PP-OCRv5 server
- runtime: RapidOCR 3.10.0, ONNX Runtime GPU 1.26.0, Python 3.12.10
- GPU after preflight: approximately 3.03 GiB used, 4.87 GiB free on the
  8.15 GiB device

The previous 2 GiB BFCArena OOM did not recur on the failing image.
