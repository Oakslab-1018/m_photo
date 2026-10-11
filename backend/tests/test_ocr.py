"""OCR provider contract tests using a fake RapidOCR runtime.

These tests never download models or import the real ``rapidocr`` package:
the adapter is exercised through an injected fake engine, so the suite stays
offline and deterministic (``OCR_PROVIDER=mock`` is the default everywhere).
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.config import (  # noqa: E402
    DEFAULT_OCR_LANGUAGE_MIN_CONFIDENCE,
    DEFAULT_OCR_MIN_TEXT_SCORE,
    Settings,
)
from backend.app.domain.geolocation.analysis.ocr import (  # noqa: E402
    CLS_MODEL_KEY,
    DET_MODEL_KEY,
    MOBILE_MODEL_PROFILE,
    MODEL_PROFILES,
    PPOCR_VERSION,
    REC_MODEL_KEY,
    SERVER_MODEL_PROFILE,
    OCRAnalyzer,
    RapidOCREngine,
    normalize_provider,
    sanitize_diagnostic,
)
from backend.app.domain.geolocation.analysis.pipeline import AnalysisPipeline  # noqa: E402
from backend.app.domain.geolocation.analysis.text_language import (  # noqa: E402
    TextLanguageAnalyzer,
    detect_language_from_text,
    detect_ocr_language_line,
)


class FakeOutput:
    def __init__(self, records):
        self._records = records

    def to_json(self):
        return [dict(r) for r in self._records]


class FakeEngine:
    """Minimal stand-in for ``rapidocr.RapidOCR``."""

    def __init__(self, records):
        self.records = records
        self.calls = []
        self.concurrent = 0
        self.max_concurrent = 0

    def __call__(self, image_path):
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            time.sleep(0.01)
            self.calls.append(str(image_path))
            return FakeOutput(self.records)
        finally:
            self.concurrent -= 1


def fake_analyzer(records, model_type="server"):
    engine = FakeEngine(records)
    analyzer = OCRAnalyzer(
        provider="rapidocr",
        model_type=model_type,
        engine=RapidOCREngine(model_type=model_type, loader=lambda params: engine),
    )
    return analyzer, engine


def sample_records():
    return [
        {"box": [[1, 1], [50, 1], [50, 10], [1, 10]], "txt": "HELLO WORLD", "score": 0.95},
        {"box": [[1, 20], [50, 20], [50, 30], [1, 30]], "txt": "東京都", "score": 0.9},
        {"box": [[1, 40], [50, 40], [50, 50], [1, 50]], "txt": "こんにちは", "score": 0.9},
        {"box": [[1, 60], [50, 60], [50, 70], [1, 70]], "txt": "Привет", "score": 0.8},
        {"box": [[1, 80], [50, 80], [50, 90], [1, 90]], "txt": "안녕하세요", "score": 0.6},
    ]


class OCRProviderTest(unittest.TestCase):
    def test_normalize_provider_aliases_and_unknown(self):
        self.assertEqual(normalize_provider(None), "mock")
        self.assertEqual(normalize_provider(""), "mock")
        self.assertEqual(normalize_provider("MOCK"), "mock")
        self.assertEqual(normalize_provider("rapidocr"), "rapidocr")
        self.assertEqual(normalize_provider("real"), "rapidocr")  # legacy alias
        # Unknown provider passes through and is rejected safely at analyze().
        self.assertEqual(normalize_provider("mystery"), "mystery")

    def test_mock_provider_returns_no_clues_and_never_calls_engine(self):
        analyzer, engine = fake_analyzer(sample_records())
        analyzer.provider = "mock"
        clues = analyzer.analyze("does-not-exist.png")
        self.assertEqual(clues, [])
        self.assertEqual(engine.calls, [])
        self.assertEqual(analyzer.status()["state"], "ok")
        self.assertEqual(analyzer.status()["line_count"], 0)

    def test_mock_pipeline_does_not_import_rapidocr(self):
        for name in [m for m in sys.modules if m.startswith("rapidocr")]:
            sys.modules.pop(name, None)
        pipeline = AnalysisPipeline(mode="mock")
        self.assertEqual(pipeline.ocr_provider, "mock")
        self.assertEqual(pipeline.ocr_status["provider"], "mock")
        self.assertFalse(
            any(m.startswith("rapidocr") for m in sys.modules),
            "mock mode must never import rapidocr",
        )

    def test_rapidocr_provider_preserves_per_line_confidence_bbox_provider(self):
        analyzer, engine = fake_analyzer(sample_records())
        clues = analyzer.analyze("street.png")
        self.assertEqual(len(clues), 5)
        self.assertEqual(engine.calls, ["street.png"])

        first = clues[0]
        self.assertEqual(first.dimension, "signage")
        self.assertEqual(first.value, "HELLO WORLD")
        self.assertEqual(first.source, "ocr")
        self.assertAlmostEqual(first.confidence, 0.95, places=3)
        extra = first.extra
        self.assertEqual(extra["origin_source"], "ocr")
        self.assertEqual(extra["provider"], "rapidocr")
        self.assertEqual(extra["bbox"], [[1.0, 1.0], [50.0, 1.0], [50.0, 10.0], [1.0, 10.0]])
        self.assertEqual(extra["quad"], extra["bbox"])
        self.assertAlmostEqual(extra["line_confidence"], 0.95, places=3)
        self.assertEqual(extra["model_version"], PPOCR_VERSION)
        self.assertTrue(extra["reliable"])
        self.assertEqual(extra["model_type"], "server")
        self.assertEqual(extra["det_model"], SERVER_MODEL_PROFILE.det_model)
        self.assertEqual(extra["rec_model"], SERVER_MODEL_PROFILE.rec_model)
        self.assertEqual(extra["cls_model"], SERVER_MODEL_PROFILE.cls_model)

        status = analyzer.status()
        self.assertEqual(status["state"], "ok")
        self.assertEqual(status["line_count"], 5)
        self.assertEqual(status["engine"], "onnxruntime")
        self.assertEqual(status["model_version"], PPOCR_VERSION)
        self.assertEqual(status["model_type"], "server")
        self.assertEqual(
            status["models"],
            {
                "det": SERVER_MODEL_PROFILE.det_model,
                "rec": SERVER_MODEL_PROFILE.rec_model,
                "cls": SERVER_MODEL_PROFILE.cls_model,
            },
        )

    def test_unknown_provider_is_safe_diagnostic(self):
        analyzer = OCRAnalyzer(provider="mystery")
        self.assertEqual(analyzer.analyze("street.png"), [])
        status = analyzer.status()
        self.assertEqual(status["state"], "unavailable")
        self.assertIn("unknown OCR provider", status["detail"])

    def test_engine_unavailable_reports_status_not_fake_clues(self):
        def broken_loader(params):
            raise ImportError("rapidocr is not installed")

        analyzer = OCRAnalyzer(
            provider="rapidocr", engine=RapidOCREngine(loader=broken_loader)
        )
        clues = analyzer.analyze("street.png")
        self.assertEqual(clues, [])
        status = analyzer.status()
        self.assertEqual(status["state"], "unavailable")
        self.assertIn("rapidocr", status["detail"])

    def test_inference_failure_reports_failed_status(self):
        class ExplodingEngine:
            def __call__(self, image_path):
                raise RuntimeError("boom")

        analyzer = OCRAnalyzer(provider="rapidocr", engine=ExplodingEngine())
        self.assertEqual(analyzer.analyze("street.png"), [])
        status = analyzer.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("boom", status["detail"])

    def test_engine_initialization_is_lazy_and_single_instance(self):
        calls = {"loader": 0}
        engine = FakeEngine(sample_records())

        def loader(params):
            calls["loader"] += 1
            return engine

        analyzer = OCRAnalyzer(
            provider="rapidocr", engine=RapidOCREngine(loader=loader)
        )
        self.assertEqual(calls["loader"], 0)  # lazy: nothing loaded at build time
        analyzer.analyze("a.png")
        analyzer.analyze("b.png")
        self.assertEqual(calls["loader"], 1)

    def test_concurrent_analyze_is_serialized(self):
        analyzer, engine = fake_analyzer(sample_records())
        barrier = threading.Barrier(4)
        errors = []

        def run(index):
            try:
                barrier.wait(timeout=5)
                analyzer.analyze(f"img-{index}.png")
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertEqual(errors, [])
        self.assertEqual(engine.max_concurrent, 1)
        self.assertEqual(len(engine.calls), 4)

    def test_diagnostics_scrub_paths_and_urls(self):
        dirty = (
            "download failed for "
            "https://example.test/models/secret.onnx at "
            "C:\\Users\\someone\\weights\\det.onnx and /home/user/models/rec.onnx"
        )
        clean = sanitize_diagnostic(dirty)
        self.assertNotIn("example.test", clean)
        self.assertNotIn("someone", clean)
        self.assertNotIn("/home/user", clean)
        self.assertIn("<url>", clean)
        self.assertIn("<path>", clean)

    def test_unavailable_detail_is_scrubbed(self):
        def broken_loader():
            raise FileNotFoundError("C:\\Users\\someone\\models\\det.onnx")

        analyzer = OCRAnalyzer(
            provider="rapidocr", engine=RapidOCREngine(loader=broken_loader)
        )
        analyzer.analyze("street.png")
        detail = analyzer.status()["detail"]
        self.assertNotIn("someone", detail)
        self.assertNotIn(".onnx", detail)

    def test_engine_build_params_default_to_ppocrv5_server_on_cpu(self):
        from rapidocr import EngineType, ModelType, OCRVersion

        engine = RapidOCREngine(model_dir="/data/ocr-models")
        params = engine._build_params()
        for task in ("Det", "Cls", "Rec"):
            self.assertIs(params[f"{task}.engine_type"], EngineType.ONNXRUNTIME)
            self.assertIs(params[f"{task}.ocr_version"], OCRVersion.PPOCRV5)
            self.assertIs(params[f"{task}.model_type"], ModelType.SERVER)
            value = params[f"{task}.ocr_version"].value
            self.assertEqual(value, "PP-OCRv5")
            self.assertEqual(params[f"{task}.engine_type"].value, "onnxruntime")
        self.assertEqual(params["Global.model_root_dir"], "/data/ocr-models")
        self.assertNotIn("Global.use_cuda", params)
        self.assertEqual(params["Det.limit_type"], "max")
        self.assertEqual(params["Det.limit_side_len"], 1280)
        self.assertEqual(engine.model_identity()["det_model"], SERVER_MODEL_PROFILE.det_model)

    def test_mobile_profile_remains_explicitly_supported(self):
        from rapidocr import ModelType

        engine = RapidOCREngine(model_type="mobile", model_dir="/data/ocr-models")
        params = engine._build_params()
        for task in ("Det", "Cls", "Rec"):
            self.assertIs(params[f"{task}.model_type"], ModelType.MOBILE)
        identity = engine.model_identity()
        self.assertEqual(identity["model_type"], "mobile")
        self.assertEqual(identity["det_model"], DET_MODEL_KEY)
        self.assertEqual(identity["rec_model"], REC_MODEL_KEY)
        self.assertEqual(identity["cls_model"], CLS_MODEL_KEY)

    def test_invalid_model_type_is_rejected(self):
        with self.assertRaises(ValueError):
            RapidOCREngine(model_type="large")
        with self.assertRaises(ValueError):
            OCRAnalyzer(provider="rapidocr", model_type="large")

    def test_explicit_model_paths_override_ids_with_safe_basename_provenance(self):
        engine = RapidOCREngine(
            model_type="server",
            det_model_path="C:\\weights\\custom_det.onnx",
            rec_model_path="/opt/models/custom_rec.onnx",
            cls_model_path="custom_cls.onnx",
        )
        params = engine._build_params()
        self.assertEqual(params["Det.model_path"], "C:\\weights\\custom_det.onnx")
        self.assertEqual(params["Rec.model_path"], "/opt/models/custom_rec.onnx")
        self.assertEqual(params["Cls.model_path"], "custom_cls.onnx")
        identity = engine.model_identity()
        self.assertEqual(identity["det_model"], "custom_det.onnx")
        self.assertEqual(identity["rec_model"], "custom_rec.onnx")
        self.assertEqual(identity["cls_model"], "custom_cls.onnx")
        self.assertNotIn("C:\\", identity["det_model"])
        self.assertNotIn("/opt/", identity["rec_model"])


class OCRLanguageEvidenceTest(unittest.TestCase):
    def test_pipeline_emits_reliable_language_clues_per_line(self):
        analyzer, _engine = fake_analyzer(sample_records())
        pipeline = AnalysisPipeline(
            mode="mock", ocr_provider="rapidocr", ocr_analyzer=analyzer
        )
        clues = pipeline.analyze("street.png")
        languages = [c for c in clues if c.dimension == "language"]
        by_value = {c.value: c for c in languages}

        self.assertEqual(set(by_value), {"latin", "ja", "cyrillic"})
        for clue in languages:
            self.assertEqual(clue.source, "ocr")
            self.assertTrue(clue.extra["reliable"])
            self.assertEqual(clue.extra["origin_source"], "ocr")
            self.assertEqual(clue.extra["provider"], "rapidocr")

        # confidence is bounded by the OCR line confidence
        self.assertLessEqual(by_value["latin"].confidence, 0.95)
        self.assertLessEqual(by_value["ja"].confidence, 0.9)
        self.assertLessEqual(by_value["cyrillic"].confidence, 0.8)
        # below threshold (0.6 < 0.7) -> no language clue for the Korean line
        self.assertNotIn("ko", by_value)

    def test_pure_han_never_claims_zh_and_keeps_signage(self):
        analyzer, _engine = fake_analyzer(
            [{"box": [[0, 0], [10, 0], [10, 10], [0, 10]], "txt": "北京市", "score": 0.99}]
        )
        pipeline = AnalysisPipeline(
            mode="mock", ocr_provider="rapidocr", ocr_analyzer=analyzer
        )
        clues = pipeline.analyze("sign.png")
        signage = [c for c in clues if c.dimension == "signage"]
        languages = [c for c in clues if c.dimension == "language"]
        self.assertEqual([c.value for c in signage], ["北京市"])
        self.assertEqual(languages, [])
        self.assertEqual(detect_ocr_language_line("北京市", 0.99), (None, 0.0))

    def test_mixed_han_and_kana_is_ja(self):
        lang, conf = detect_ocr_language_line("東京タワー", 0.9)
        self.assertEqual(lang, "ja")
        self.assertLessEqual(conf, 0.9)

    def test_language_confidence_bounded_by_ocr_confidence(self):
        self.assertEqual(detect_ocr_language_line("Hello World", 0.3), (None, 0.0))
        lang, conf = detect_ocr_language_line("Hello World", 0.82)
        self.assertEqual(lang, "latin")
        self.assertLessEqual(conf, 0.82)

    def test_configurable_language_threshold(self):
        analyzer = TextLanguageAnalyzer(mode="rapidocr", min_confidence=0.9)
        self.assertEqual(analyzer.analyze_from_line("Hello World", 0.8), [])
        clues = analyzer.analyze_from_line("Hello World", 0.95)
        self.assertEqual(len(clues), 1)
        self.assertEqual(clues[0].source, "ocr")
        self.assertTrue(clues[0].extra["reliable"])

    def test_default_threshold_constant(self):
        self.assertAlmostEqual(DEFAULT_OCR_LANGUAGE_MIN_CONFIDENCE, 0.7, places=3)
        self.assertAlmostEqual(DEFAULT_OCR_MIN_TEXT_SCORE, 0.5, places=3)

    def test_detect_language_from_text_backwards_compatible(self):
        self.assertEqual(detect_language_from_text("你好世界")[0], "zh")
        self.assertEqual(detect_language_from_text("こんにちは")[0], "ja")
        self.assertEqual(detect_language_from_text("안녕하세요")[0], "ko")
        self.assertEqual(detect_language_from_text("Привет")[0], "cyrillic")
        self.assertEqual(detect_language_from_text("Hello World")[0], "latin")
        self.assertEqual(detect_language_from_text(""), ("unknown", 0.0))

    def test_unprojected_language_analyzer_returns_nothing(self):
        analyzer = TextLanguageAnalyzer(mode="mock")
        self.assertEqual(analyzer.analyze_from_text("Hello World"), [])


class OCRSettingsTest(unittest.TestCase):
    def test_ocr_settings_defaults(self):
        settings = Settings.from_env({})
        self.assertEqual(settings.ocr_provider, "mock")
        self.assertEqual(settings.ocr_min_text_score, DEFAULT_OCR_MIN_TEXT_SCORE)
        self.assertEqual(
            settings.ocr_language_min_confidence, DEFAULT_OCR_LANGUAGE_MIN_CONFIDENCE
        )
        expected = (settings.data_dir / "ocr-models").resolve()
        self.assertEqual(settings.ocr_model_dir, expected)
        self.assertIsNone(settings.ocr_det_model_path)

    def test_ocr_provider_env_alias_and_real_alias(self):
        self.assertEqual(
            Settings.from_env({"OCR_PROVIDER": "RAPIDOCR"}).ocr_provider, "rapidocr"
        )
        self.assertEqual(Settings.from_env({"GEO_OCR_PROVIDER": "mock"}).ocr_provider, "mock")
        # legacy GEO_AI_MODE=real maps onto the rapidocr provider
        self.assertEqual(Settings.from_env({"GEO_AI_MODE": "real"}).ocr_provider, "rapidocr")
        self.assertEqual(Settings.from_env({}).ocr_provider, "mock")

    def test_ocr_model_paths_resolve_against_model_dir(self):
        settings = Settings.from_env(
            {
                "M_PHOTO_DATA_DIR": "/data",
                "OCR_DET_MODEL_PATH": "det.onnx",
                "OCR_REC_MODEL_PATH": "/abs/rec.onnx",
            }
        )
        self.assertEqual(settings.ocr_model_dir, Path("/data/ocr-models").resolve())
        self.assertEqual(settings.ocr_det_model_path, Path("/data/ocr-models/det.onnx").resolve())
        self.assertEqual(settings.ocr_rec_model_path, Path("/abs/rec.onnx").resolve())

    def test_ocr_score_settings_are_clamped(self):
        settings = Settings.from_env(
            {"OCR_MIN_TEXT_SCORE": "9", "OCR_LANGUAGE_MIN_CONFIDENCE": "not-a-number"}
        )
        self.assertEqual(settings.ocr_min_text_score, 1.0)
        self.assertEqual(
            settings.ocr_language_min_confidence, DEFAULT_OCR_LANGUAGE_MIN_CONFIDENCE
        )


if __name__ == "__main__":
    unittest.main()