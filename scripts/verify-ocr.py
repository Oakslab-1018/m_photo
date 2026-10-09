"""REAL OCR smoke verification (manual, network/model dependent).

Renders synthetic Latin and Chinese/Japanese signage with Windows fonts, runs
the actual RapidOCR + ONNX Runtime PP-OCRv5 engine, and prints the exact
engine/model identity plus every recognized span with per-line confidence and
quad bbox. The rendered images live under git-ignored ``backend/storage``.

This script intentionally does NOT run as part of pytest: it needs downloaded
model weights and real inference. Run it via ``scripts/verify-ocr.ps1`` or:

    uv run --frozen python scripts/verify-ocr.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.app.config import Settings  # noqa: E402
from backend.app.domain.geolocation.analysis.ocr import (  # noqa: E402
    CLS_MODEL_KEY,
    DET_MODEL_KEY,
    PPOCR_VERSION,
    REC_MODEL_KEY,
    OCRAnalyzer,
)

FONT_CANDIDATES = {
    "latin": ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"],
    "cjk": ["C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"],
    "ja": ["C:/Windows/Fonts/YuGothB.ttc", "C:/Windows/Fonts/msgothic.ttc", "C:/Windows/Fonts/msjhbd.ttc"],
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


def main() -> int:
    settings = Settings.from_env(os.environ)
    model_dir = settings.ocr_model_dir or (settings.data_dir / "ocr-models")
    work_dir = settings.data_dir / "ocr-smoke"
    work_dir.mkdir(parents=True, exist_ok=True)

    print("== OCR REAL smoke ==")
    print(f"provider   : rapidocr")
    print(f"engine     : onnxruntime (CPU)")
    print(f"version    : {PPOCR_VERSION}")
    print(f"det_model  : {settings.ocr_det_model_path or DET_MODEL_KEY}")
    print(f"rec_model  : {settings.ocr_rec_model_path or REC_MODEL_KEY}")
    print(f"cls_model  : {settings.ocr_cls_model_path or CLS_MODEL_KEY}")
    print(f"model_dir  : {model_dir}")
    print(f"work_dir   : {work_dir}")
    print()

    analyzer = OCRAnalyzer(
        provider="rapidocr",
        model_dir=model_dir,
        det_model_path=settings.ocr_det_model_path,
        rec_model_path=settings.ocr_rec_model_path,
        cls_model_path=settings.ocr_cls_model_path,
        min_text_score=settings.ocr_min_text_score,
    )

    failures = 0
    for name, kind, expected_lines in CASES:
        font_path = pick_font(kind)
        if font_path is None:
            print(f"[{name}] SKIP: no Windows font found for {kind}")
            failures += 1
            continue
        text = " ".join(expected_lines)
        image_path = render(expected_lines, font_path, work_dir / f"{name}.png")
        clues = analyzer.analyze(str(image_path))
        status = analyzer.status()
        recognized = [c.value for c in clues if c.dimension == "signage"]
        expected_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in text)
        # A case passes when at least one expected fragment is recognized;
        # CJK must additionally yield CJK glyphs (model revisions vary).
        haystack = "".join(recognized)
        ok = any(fragment in haystack for fragment in expected_lines) and (
            not expected_cjk or any("\u4e00" <= ch <= "\u9fff" for ch in haystack)
        )
        print(f"[{name}] expected={text!r} font={font_path.name} state={status['state']} lines={status['line_count']}")
        for clue in clues:
            extra = clue.extra
            print(
                json.dumps(
                    {
                        "text": clue.value,
                        "confidence": round(clue.confidence, 4),
                        "provider": extra.get("provider"),
                        "origin_source": extra.get("origin_source"),
                        "model_version": extra.get("model_version"),
                        "bbox": extra.get("bbox"),
                    },
                    ensure_ascii=False,
                )
            )
        if not ok:
            failures += 1
            print(f"[{name}] FAIL: no usable recognized span")
        print()

    print(f"smoke result: {'OK' if failures == 0 else f'{failures} case(s) failed/skipped'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())