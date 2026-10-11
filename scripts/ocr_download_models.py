"""Precache one complete pinned PP-OCRv5 det/cls/rec profile.

Run via ``scripts/setup-ocr.ps1`` (host) or from the backend image for a
container-volume prefetch. The server profile is the default:

    python scripts/ocr_download_models.py --model-type server

RapidOCR's own resolver supplies the exact URLs and SHA256 values from the
pinned ``rapidocr.default_models.yaml``. This script verifies every downloaded
file against that resolver before reporting success.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "backend" / "storage" / "ocr-models"
DEFAULT_MODEL_TYPE = "server"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-type",
        default=os.environ.get("OCR_MODEL_TYPE", DEFAULT_MODEL_TYPE),
        help="PP-OCRv5 profile: server (accuracy) or mobile (latency)",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("M_PHOTO_OCR_MODEL_DIR")
        or os.environ.get("OCR_MODEL_DIR")
        or str(DEFAULT_MODEL_DIR),
        help="ONNX cache directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, str(REPO_ROOT))

    try:
        from backend.app.config import normalize_ocr_model_type
    except Exception as exc:  # pragma: no cover - setup environment issue
        print(f"backend config is not importable: {exc}", file=sys.stderr)
        return 2

    try:
        model_type = normalize_ocr_model_type(args.model_type)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    model_dir = Path(args.model_dir).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR
        from rapidocr.inference_engine.base import FileInfo, InferSession
        from rapidocr.utils.typings import TaskType
    except Exception as exc:  # pragma: no cover - setup environment issue
        print(f"rapidocr is not importable: {exc}", file=sys.stderr)
        print("Run: uv sync --frozen", file=sys.stderr)
        return 2

    rapid_model_type = (
        ModelType.SERVER if model_type == "server" else ModelType.MOBILE
    )
    print(f"Preparing PP-OCRv5 {model_type} det/cls/rec ONNX models in {model_dir}")

    # Only ``Global.model_root_dir`` and the selected profile matter here:
    # constructing/loading each task uses RapidOCR's verified model resolver.
    params = {
        "Global.model_root_dir": str(model_dir),
        "Global.log_level": "info",
        "Global.use_det": True,
        "Global.use_cls": True,
        "Global.use_rec": True,
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Det.lang_type": "ch",
        "Det.model_type": rapid_model_type,
        "Cls.engine_type": EngineType.ONNXRUNTIME,
        "Cls.ocr_version": OCRVersion.PPOCRV5,
        "Cls.lang_type": "ch",
        "Cls.model_type": rapid_model_type,
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Rec.lang_type": "ch",
        "Rec.model_type": rapid_model_type,
    }

    try:
        engine = RapidOCR(params=params)
        # RapidOCR 3.10.0 lazily loads det/cls/rec on first use. Trigger each
        # task explicitly so a complete profile is cached before deployment.
        engine._load_det_model()
        engine._load_cls_model()
        engine._load_rec_model()
    except Exception as exc:
        print(f"model download or load failed: {exc}", file=sys.stderr)
        return 1

    failures = 0
    for task_type, label in (
        (TaskType.DET, "det"),
        (TaskType.CLS, "cls"),
        (TaskType.REC, "rec"),
    ):
        try:
            info = InferSession.get_model_url(
                FileInfo(
                    engine_type=EngineType.ONNXRUNTIME,
                    ocr_version=OCRVersion.PPOCRV5,
                    task_type=task_type,
                    lang_type="ch",
                    model_type=rapid_model_type,
                )
            )
            path = model_dir / Path(info["model_dir"]).name
            actual = sha256_file(path)
            expected = str(info.get("SHA256", "")).lower()
            if expected and actual.lower() != expected:
                print(
                    f"{label}: SHA256 mismatch for {path.name}: {actual}",
                    file=sys.stderr,
                )
                failures += 1
                continue
            print(f"{label}\t{path.name}\t{path.stat().st_size}\t{actual}")
        except Exception as exc:
            print(f"{label}: verification failed: {exc}", file=sys.stderr)
            failures += 1

    if failures:
        print(f"{failures} model verifications failed", file=sys.stderr)
        return 1
    print(f"OCR {model_type} profile ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
