"""文字语言检测分析器。

两条路径：

``detect_language_from_text``
    向后兼容的纯启发式函数（既有测试依赖），仅返回 (语言, 置信度)。

``detect_ocr_language_line``
    保守的 OCR 专用路径：只有拿到**可靠 OCR 行**（高置信度 + 明确文字体系）
    才给出语言线索；纯汉字无法区分中/日，因此不产出地理语言线索。

语言取值受知识库约束（latin / cyrillic / thai / ja / ko / zh / arabic）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .base import BaseAnalyzer
from ..clue import Clue, VALUE_LABELS

# 默认最低 OCR 行置信度：低于此值不产出语言线索（可在 Settings 配置）。
DEFAULT_OCR_LANGUAGE_MIN_CONFIDENCE = 0.7


def detect_language_from_text(text: str) -> tuple[str, float]:
    """根据 Unicode 码点返回 (语言代码, 置信度)。（向后兼容）"""
    if not text:
        return ("unknown", 0.0)
    counts = {"zh": 0, "ja": 0, "ko": 0, "latin": 0, "cyrillic": 0, "arabic": 0, "thai": 0}
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:
            counts["zh"] += 1
        elif 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF:
            counts["ja"] += 1
        elif 0xAC00 <= cp <= 0xD7AF:
            counts["ko"] += 1
        elif 0x0400 <= cp <= 0x04FF:
            counts["cyrillic"] += 1
        elif 0x0600 <= cp <= 0x06FF:
            counts["arabic"] += 1
        elif 0x0E00 <= cp <= 0x0E7F:
            counts["thai"] += 1
        elif ch.isalpha() and cp < 0x0250:
            counts["latin"] += 1

    total = sum(counts.values())
    if total == 0:
        return ("unknown", 0.0)
    lang, cnt = max(counts.items(), key=lambda kv: kv[1])
    confidence = cnt / total
    # 日文含汉字，需用假名二次确认
    if lang == "zh" and counts["ja"] > 0:
        lang = "ja"
        confidence = (counts["ja"] + counts["zh"]) / total
    return (lang, round(confidence, 3))


@dataclass
class _ScriptCounts:
    han: int = 0
    kana: int = 0
    hangul: int = 0
    cyrillic: int = 0
    thai: int = 0
    devanagari: int = 0
    latin: int = 0

    @property
    def total(self) -> int:
        return (
            self.han + self.kana + self.hangul + self.cyrillic
            + self.thai + self.devanagari + self.latin
        )


def _count_scripts(text: str) -> _ScriptCounts:
    counts = _ScriptCounts()
    for ch in text:
        cp = ord(ch)
        if 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF or 0xFF66 <= cp <= 0xFF9F:
            counts.kana += 1
        elif 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0xF900 <= cp <= 0xFAFF:
            counts.han += 1
        elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            counts.hangul += 1
        elif 0x0400 <= cp <= 0x04FF:
            counts.cyrillic += 1
        elif 0x0E00 <= cp <= 0x0E7F:
            counts.thai += 1
        elif 0x0900 <= cp <= 0x097F:
            counts.devanagari += 1
        elif ch.isalpha() and cp < 0x0250:
            counts.latin += 1
    return counts


def detect_ocr_language_line(
    text: str,
    ocr_confidence: float,
    *,
    min_confidence: float = DEFAULT_OCR_LANGUAGE_MIN_CONFIDENCE,
) -> tuple[str | None, float]:
    """保守地从单条可靠 OCR 行判断语言。

    返回 ``(语言代码, 置信度)``；无法可靠判断时返回 ``(None, 0.0)``。
    置信度上限为 OCR 行置信度（不凭空放大）。
    """
    if not text:
        return (None, 0.0)
    ocr_confidence = max(0.0, min(1.0, float(ocr_confidence)))
    if ocr_confidence < min_confidence:
        return (None, 0.0)

    counts = _count_scripts(text)
    total = counts.total
    if total == 0:
        return (None, 0.0)

    # 假名是日语的强证据：即使混有汉字也判为 ja。
    if counts.kana > 0:
        ratio = (counts.kana + counts.han) / total
        return ("ja", min(ocr_confidence, max(0.6, ratio)))

    # 纯汉字无法区分中/日（甚至部分韩文语境）：不产出地理语言线索。
    if counts.han == total:
        return (None, 0.0)

    # 其余文字体系需在其 script 字符中占绝对多数，避免少量混排误判。
    dominant = max(
        ("ko", counts.hangul),
        ("cyrillic", counts.cyrillic),
        ("thai", counts.thai),
        ("devanagari", counts.devanagari),
        ("latin", counts.latin),
        key=lambda kv: kv[1],
    )
    lang, count = dominant
    if count == 0:
        return (None, 0.0)
    if count / total < 0.6:
        return (None, 0.0)

    ratio = count / total
    return (lang, min(ocr_confidence, round(ratio, 3)))


class TextLanguageAnalyzer(BaseAnalyzer):
    name = "ocr"

    def __init__(
        self,
        mode: str = "mock",
        *,
        min_confidence: float = DEFAULT_OCR_LANGUAGE_MIN_CONFIDENCE,
    ) -> None:
        self.mode = mode
        self.min_confidence = min_confidence

    def analyze_from_text(self, text: str) -> List[Clue]:
        """向后兼容入口：置信度 0.8 的“可靠”综合行。"""
        if self.mode != "rapidocr":
            return []
        lang, conf = detect_ocr_language_line(
            text, 0.8, min_confidence=self.min_confidence
        )
        return self._to_clue(lang, conf, text, ocr_confidence=0.8)

    def analyze_from_line(
        self, text: str, ocr_confidence: float, *, evidence: str | None = None
    ) -> List[Clue]:
        lang, conf = detect_ocr_language_line(
            text, ocr_confidence, min_confidence=self.min_confidence
        )
        return self._to_clue(lang, conf, text, ocr_confidence=ocr_confidence, evidence=evidence)

    def _to_clue(
        self,
        lang: str | None,
        confidence: float,
        text: str,
        *,
        ocr_confidence: float,
        evidence: str | None = None,
    ) -> List[Clue]:
        if lang is None or confidence <= 0.0:
            return []
        return [
            Clue(
                dimension="language",
                value=lang,
                confidence=round(min(confidence, 1.0), 3),
                evidence=evidence
                or f"OCR 行文字体系判断为 {VALUE_LABELS.get(lang, lang)}",
                source="ocr",
                extra={
                    "origin_source": "ocr",
                    "provider": "rapidocr",
                    "reliable": True,
                    "ocr_confidence": round(float(ocr_confidence), 3),
                    "threshold": self.min_confidence,
                    "text_sample": text[:50],
                },
            )
        ]

    def analyze(self, image_path: str) -> List[Clue]:
        # 语言检测依赖 OCR 结果，由 pipeline 注入；单独调用返回空
        return []


__all__ = [
    "DEFAULT_OCR_LANGUAGE_MIN_CONFIDENCE",
    "TextLanguageAnalyzer",
    "detect_language_from_text",
    "detect_ocr_language_line",
]