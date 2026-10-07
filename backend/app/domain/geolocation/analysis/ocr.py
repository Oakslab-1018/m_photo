"""文字识别分析器。

real 模式：使用 easyocr（CPU 可运行，支持中/英/日/韩/俄等多语言）。
  由于 easyocr 的语言组合限制（ch_sim 只能与 en 组合），
  采用多个 Reader 分别识别不同语言组合，再合并结果。
mock 模式：返回空线索（留给用户或其他分析器补充）。
"""
from __future__ import annotations

from typing import List

from .base import BaseAnalyzer
from ..clue import Clue


# easyocr 语言组合限制：ch_sim 只能与 en 同组；ja/ko/ru 可与 en 同组。
# 按语言组合拆分为多个 reader，逐一识别后合并去重。
LANG_GROUPS = [
    ["ch_sim", "en"],   # 简中 + 英文
    ["ja", "en"],       # 日文 + 英文
    ["ko", "en"],       # 韩文 + 英文
    ["ru", "en"],       # 俄文 + 英文
]


class OCRAnalyzer(BaseAnalyzer):
    name = "ocr"

    def __init__(self, mode: str = "mock"):
        self.mode = mode
        self._readers = []

    def _model_dir(self):
        """easyocr 模型存储目录：打包后指向随包发布的 model 目录。"""
        import sys
        import os
        if getattr(sys, "frozen", False):
            bundled = os.path.join(sys._MEIPASS, "easyocr_models")
            if os.path.isdir(bundled):
                return bundled
        return None  # 让 easyocr 使用默认目录 ~/.EasyOCR/model

    def _get_readers(self):
        if self._readers:
            return self._readers
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            import easyocr
        except Exception as e:
            print(f"[OCRAnalyzer] easyocr 不可用，降级为 mock: {e}")
            self.mode = "mock"
            return []
        model_dir = self._model_dir()
        for langs in LANG_GROUPS:
            try:
                kwargs = {"gpu": False}
                if model_dir:
                    kwargs["model_storage_directory"] = model_dir
                reader = easyocr.Reader(langs, **kwargs)
                self._readers.append(reader)
            except Exception as e:
                print(f"[OCRAnalyzer] 语言组 {langs} 加载失败，跳过: {e}")
        if not self._readers:
            self.mode = "mock"
        return self._readers

    def analyze(self, image_path: str) -> List[Clue]:
        if self.mode == "real":
            readers = self._get_readers()
            if not readers:
                return []
            all_texts: List[str] = []
            for reader in readers:
                try:
                    results = reader.readtext(image_path, detail=0)
                    if results:
                        all_texts.extend(results)
                except Exception as e:
                    print(f"[OCRAnalyzer] 识别失败: {e}")
            # 去重（保留顺序）
            seen = set()
            unique_texts = []
            for t in all_texts:
                t = t.strip()
                if t and t not in seen:
                    seen.add(t)
                    unique_texts.append(t)
            if not unique_texts:
                return []
            text = " ".join(unique_texts).strip()
            return [
                Clue(
                    dimension="signage",
                    value=text[:200],
                    confidence=0.85,
                    evidence=f"OCR 识别到 {len(unique_texts)} 段文字",
                    source=self.name,
                    extra={"raw_texts": unique_texts},
                )
            ]
        # mock 模式：不返回文字线索，保持流程可运行
        return []
