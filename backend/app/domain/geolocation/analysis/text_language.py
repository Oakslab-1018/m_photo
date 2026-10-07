"""文字语言检测分析器。

基于 OCR 输出的文字内容或 Unicode 码点范围，推断文字所属语言。
real 模式可接入 langdetect / fasttext；此处用 Unicode 区间启发式。
"""
from __future__ import annotations

from typing import List

from .base import BaseAnalyzer
from ..clue import Clue, VALUE_LABELS


def detect_language_from_text(text: str) -> tuple[str, float]:
    """根据 Unicode 码点返回 (语言代码, 置信度)。"""
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


class TextLanguageAnalyzer(BaseAnalyzer):
    name = "text_language"

    def __init__(self, mode: str = "mock"):
        self.mode = mode

    def analyze_from_text(self, text: str) -> List[Clue]:
        lang, conf = detect_language_from_text(text)
        if lang == "unknown" or conf < 0.3:
            return []
        return [
            Clue(
                dimension="language",
                value=lang,
                confidence=conf,
                evidence=f"基于文字 Unicode 区间判断为 {VALUE_LABELS.get(lang, lang)}",
                source=self.name,
                extra={"text_sample": text[:50]},
            )
        ]

    def analyze(self, image_path: str) -> List[Clue]:
        # 语言检测依赖 OCR 结果，由 pipeline 注入；单独调用返回空
        return []
