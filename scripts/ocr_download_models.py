"""Download the pinned PP-OCRv5 ONNX models into the backend storage cache.

Run via ``scripts/setup-ocr.ps1`` (or directly). Uses RapidOCR's own resolver so
the exact URLs and SHA256 hashes come from ``rapidocr.default_models.yaml`` for
the pinned package version -- no hand-copied weights.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "backend" / "storage" / "ocr-models"


def main() -> int:
    model_dir = Path(
        os.environ.get("M_PHOTO_OCR_MODEL_DIR")
        or os.environ.get("OCR_MODEL_DIR")
        or DEFAULT_MODEL_DIR
    ).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR
    except Exception as exc:  # pragma: no cover - setup environment issue
        print(f"rapidocr is not importable: {exc}", file=sys.stderr)
        print("Run: uv sync --frozen", file=sys.stderr)
        return 2

    # Only ``Global.model_root_dir`` matters here: constructing the engine
    # downloads det/cls/rec on first use through the verified model resolver.
    params = {
        "Global.model_root_dir": str(model_dir),
        "Global.log_level": "info",
        "Global.use_det": True,
        "Global.use_cls": True,
        "Global.use_rec": True,
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Det.lang_type": "ch",
        "Det.model_type": ModelType.MOBILE,
        "Cls.engine_type": EngineType.ONNXRUNTIME,
        "Cls.ocr_version": OCRVersion.PPOCRV5,
        "Cls.lang_type": "ch",
        "Cls.model_type": ModelType.MOBILE,
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Rec.lang_type": "ch",
        "Rec.model_type": ModelType.MOBILE,
    }

    try:
        engine = RapidOCR(params=params)
        # RapidOCR 3.10.0 lazily loads det/cls/rec on first use, but a blank
        # image short-circuits before cls/rec. Trigger each model explicitly so
        # all three PP-OCRv5 ONNX files are cached up front.
        engine._load_det_model()
        engine._load_cls_model()
        engine._load_rec_model()
    except Exception as exc:
        print(f"model download failed: {exc}", file=sys.stderr)
        return 1

    onnx_files = sorted(model_dir.glob("*.onnx"))
    if not onnx_files:
        print(f"no .onnx files found in {model_dir}", file=sys.stderr)
        return 1
    for path in onnx_files:
        print(f"{path.name}\t{path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())