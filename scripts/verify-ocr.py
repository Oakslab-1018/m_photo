"""REAL OCR smoke verification (manual, model dependent).

Renders synthetic Latin and Chinese/Japanese signage with Windows fonts, runs
the selected PP-OCRv5 det/cls/rec profile through RapidOCR + ONNX Runtime, and
prints exact model identity, hashes, every recognized span, and CPU timing.

The rendered images live under git-ignored ``backend/storage/ocr-smoke``. This
script intentionally does NOT run as part of pytest: it needs downloaded model
weights and real inference. Run it via ``scripts/verify-ocr.ps1`` or:

    uv run --frozen python scripts/verify-ocr.py --model-type server
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.app.config import Settings  # noqa: E402
from backend.app.domain.geolocation.analysis.ocr import (  # noqa: E402
    MODEL_PROFILES,
    PPOCR_VERSION,
    OCRAnalyzer,
)

FONT_CANDIDATES = {
    "latin": ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"],
    "cjk": [
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ],
    "ja": [
        "C:/Windows/Fonts/YuGothB.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
        "C:/Windows/Fonts/msjhbd.ttc",
    ],
}

# Tall, signage-like renderings (detector-friendly); keep the exact input text
# so the report can compare expected vs recognized spans.
CASES = [
    ("latin", "latin", ["OPEN", "MARKET"]),
    ("chinese", "cjk", ["北京市"]),
    ("japanese", "ja", ["東京駅"]),
]


def pick_font(kind: str) -> Path | None:
    for candidate in FONT_CANDIDATES[kind]:
        path = Path(candidate)
        if path.is_file():
            return path
    return None


def render(lines: list[str], font_path: Path, out_path: Path, size: int = 120) -> Path:
    """Stack text lines on a white board, similar to street signage."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(font_path), size=size)
    probe = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(probe)
    widths, heights = [], []
    for text in lines:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        widths.append(right - left)
        heights.append(bottom - top)
    pad = 40
    width = max(widths) + pad * 2
    height = sum(heights) + pad * (len(lines) + 1)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = pad
    for text, line_height in zip(lines, heights):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        draw.text((pad - left, y - top), text, font=font, fill=(15, 15, 15))
        y += line_height + pad
    img.save(out_path)
    return out_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label(explicit: Path | None, fallback: str) -> str:
    if explicit is None:
        return fallback
    return Path(str(explicit)).name or fallback


def verify_model_hashes(settings: Settings) -> int:
    """Verify selected models against the pinned RapidOCR resolver."""
    from rapidocr import EngineType, ModelType, OCRVersion
    from rapidocr.inference_engine.base import FileInfo, InferSession
    from rapidocr.utils.typings import TaskType

    profile = MODEL_PROFILES[settings.ocr_model_type]
    rapid_model_type = (
        ModelType.SERVER if settings.ocr_model_type == "server" else ModelType.MOBILE
    )
    model_dir = settings.ocr_model_dir or (settings.data_dir / "ocr-models")
    entries = (
        ("det", settings.ocr_det_model_path, profile.det_model, TaskType.DET),
        ("cls", settings.ocr_cls_model_path, profile.cls_model, TaskType.CLS),
        ("rec", settings.ocr_rec_model_path, profile.rec_model, TaskType.REC),
    )
    failures = 0
    for label, explicit, fallback, task_type in entries:
        path = Path(explicit) if explicit else None
        expected = None
        if path is None:
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
            expected = str(info.get("SHA256", "")).lower()
        if not path.is_file():
            print(f"{label}_hash : MISSING {path}", file=sys.stderr)
            failures += 1
            continue
        actual = _sha256(path)
        if expected and actual.lower() != expected:
            print(
                f"{label}_hash : MISMATCH {path.name} {actual} != {expected}",
                file=sys.stderr,
            )
            failures += 1
            continue
        source = "explicit" if explicit else "resolver"
        print(f"{label}_hash : {path.name} {actual} ({source})")
    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-type",
        default=os.environ.get("OCR_MODEL_TYPE", "server"),
        help="PP-OCRv5 profile: server or mobile",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ["OCR_MODEL_TYPE"] = args.model_type
    settings = Settings.from_env(os.environ)
    model_dir = settings.ocr_model_dir or (settings.data_dir / "ocr-models")
    work_dir = settings.data_dir / "ocr-smoke"
    work_dir.mkdir(parents=True, exist_ok=True)
    profile = MODEL_PROFILES[settings.ocr_model_type]

    print("== OCR REAL smoke ==")
    print("provider   : rapidocr")
    print("engine     : onnxruntime (CPU)")
    print(f"version    : {PPOCR_VERSION}")
    print(f"model_type : {settings.ocr_model_type}")
    print(f"det_model  : {_label(settings.ocr_det_model_path, profile.det_model)}")
    print(f"rec_model  : {_label(settings.ocr_rec_model_path, profile.rec_model)}")
    print(f"cls_model  : {_label(settings.ocr_cls_model_path, profile.cls_model)}")
    print(f"model_dir  : {model_dir}")
    print(f"work_dir   : {work_dir}")
    print()

    failures = verify_model_hashes(settings)
    analyzer = OCRAnalyzer(
        provider="rapidocr",
        model_type=settings.ocr_model_type,
        model_dir=model_dir,
        det_model_path=settings.ocr_det_model_path,
        rec_model_path=settings.ocr_rec_model_path,
        cls_model_path=settings.ocr_cls_model_path,
        min_text_score=settings.ocr_min_text_score,
    )

    total_elapsed = 0.0
    for name, kind, expected_lines in CASES:
        font_path = pick_font(kind)
        if font_path is None:
            print(f"[{name}] SKIP: no Windows font found for {kind}")
            failures += 1
            continue
        text = " ".join(expected_lines)
        image_path = render(expected_lines, font_path, work_dir / f"{name}.png")
        started = time.perf_counter()
        clues = analyzer.analyze(str(image_path))
        elapsed = time.perf_counter() - started
        total_elapsed += elapsed
        status = analyzer.status()
        recognized = [c.value for c in clues if c.dimension == "signage"]
        expected_cjk = any("一" <= ch <= "鿿" for ch in text)
        haystack = "".join(recognized)
        ok = any(fragment in haystack for fragment in expected_lines) and (
            not expected_cjk or any("一" <= ch <= "鿿" for ch in haystack)
        )
        expected_models = {
            "det": _label(settings.ocr_det_model_path, profile.det_model),
            "cls": _label(settings.ocr_cls_model_path, profile.cls_model),
            "rec": _label(settings.ocr_rec_model_path, profile.rec_model),
        }
        model_ok = (
            status["state"] == "ok"
            and status.get("model_type") == settings.ocr_model_type
            and status.get("models") == expected_models
        )
        print(
            f"[{name}] expected={text!r} font={font_path.name} "
            f"state={status['state']} lines={status['line_count']} "
            f"elapsed_ms={elapsed * 1000:.1f} model_ok={model_ok}"
        )
        for clue in clues:
            extra = clue.extra
            print(
                json.dumps(
                    {
                        "text": clue.value,
                        "confidence": round(clue.confidence, 4),
                        "provider": extra.get("provider"),
                        "origin_source": extra.get("origin_source"),
                        "model_type": extra.get("model_type"),
                        "model_version": extra.get("model_version"),
                        "det_model": extra.get("det_model"),
                        "rec_model": extra.get("rec_model"),
                        "cls_model": extra.get("cls_model"),
                        "bbox": extra.get("bbox"),
                    },
                    ensure_ascii=False,
                )
            )
        if not ok or not model_ok:
            failures += 1
            print(f"[{name}] FAIL: recognition={ok} model_provenance={model_ok}")
        print()

    print(
        f"smoke result: {'OK' if failures == 0 else f'{failures} case(s) failed'}; "
        f"total_inference_ms={total_elapsed * 1000:.1f}"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
