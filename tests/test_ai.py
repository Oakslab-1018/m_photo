"""AI 分析流水线单元测试。"""
import sys
import os

from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai import AnalysisPipeline
from ai.text_language import detect_language_from_text


def make_test_image(path, color):
    img = Image.new("RGB", (100, 100), color)
    img.save(path)
    return path


def test_pipeline_runs_mock():
    tmp = os.path.join(os.path.dirname(__file__), "_tmp_green.png")
    make_test_image(tmp, (80, 180, 80))
    try:
        pipeline = AnalysisPipeline(mode="mock")
        clues = pipeline.analyze(tmp)
        assert len(clues) >= 4, "mock 模式至少应返回植被/车辆/建筑/气候 4 条线索"
        dims = {c.dimension for c in clues}
        assert "vegetation" in dims
        assert "climate" in dims
    finally:
        os.remove(tmp)


def test_green_image_vegetation_forest():
    tmp = os.path.join(os.path.dirname(__file__), "_tmp_forest.png")
    make_test_image(tmp, (60, 160, 60))
    try:
        pipeline = AnalysisPipeline(mode="mock")
        clues = pipeline.analyze(tmp)
        veg = [c for c in clues if c.dimension == "vegetation"][0]
        assert veg.value in ("temperate_forest", "grassland"), f"绿色图应判为森林/草原，实际={veg.value}"
    finally:
        os.remove(tmp)


def test_desert_image_climate_arid():
    tmp = os.path.join(os.path.dirname(__file__), "_tmp_desert.png")
    make_test_image(tmp, (200, 170, 90))
    try:
        pipeline = AnalysisPipeline(mode="mock")
        clues = pipeline.analyze(tmp)
        climate = [c for c in clues if c.dimension == "climate"][0]
        assert climate.value in ("arid", "mediterranean_sunny", "temperate"), \
            f"暖干色调应倾向干旱/地中海，实际={climate.value}"
    finally:
        os.remove(tmp)


def test_language_detection():
    assert detect_language_from_text("こんにちは")[0] == "ja"
    assert detect_language_from_text("你好世界")[0] == "zh"
    assert detect_language_from_text("안녕하세요")[0] == "ko"
    assert detect_language_from_text("Привет")[0] == "cyrillic"
    assert detect_language_from_text("Hello World")[0] == "latin"


if __name__ == "__main__":
    test_pipeline_runs_mock()
    print("✓ test_pipeline_runs_mock")
    test_green_image_vegetation_forest()
    print("✓ test_green_image_vegetation_forest")
    test_desert_image_climate_arid()
    print("✓ test_desert_image_climate_arid")
    test_language_detection()
    print("✓ test_language_detection")
    print("\n所有 AI 分析测试通过！")
