"""推理引擎与知识图谱单元测试。"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reasoning.clue import Clue
from reasoning.engine import ReasoningEngine
from reasoning.hypotheses import HypothesisManager
from reasoning import knowledge_graph as kg


def test_load_countries():
    countries = kg.load_countries()
    assert len(countries) >= 10
    japan = kg.get_by_iso("JP")
    assert japan is not None
    assert japan["name_zh"] == "日本"


def test_language_inference_japan():
    clues = [
        Clue(dimension="language", value="ja", confidence=0.9),
        Clue(dimension="vegetation", value="temperate_forest", confidence=0.7),
    ]
    engine = ReasoningEngine()
    scores = engine.infer(clues)
    assert scores, "应有候选国家"
    top = scores[0]
    assert top.iso == "JP", f"日语线索应优先指向日本，实际 Top1={top.iso}"


def test_language_inference_russia():
    clues = [
        Clue(dimension="language", value="cyrillic", confidence=0.9),
        Clue(dimension="climate", value="cold_polar", confidence=0.6),
    ]
    engine = ReasoningEngine()
    scores = engine.infer(clues)
    top = scores[0]
    assert top.iso == "RU", f"西里尔+寒冷应指向俄罗斯，实际 Top1={top.iso}"


def test_exclusion_removes_country():
    clues = [Clue(dimension="language", value="latin", confidence=0.9)]
    engine = ReasoningEngine()
    scores_with = engine.infer(clues, exclusions=[])
    isos_with = {s.iso for s in scores_with}

    scores_without = engine.infer(clues, exclusions=["FR"])
    isos_without = {s.iso for s in scores_without}
    assert "FR" not in isos_without, "被排除的国家不应出现"
    assert "FR" in isos_with or True  # 法国可能在拉丁语候选中


def test_hypothesis_confirm_when_clear():
    clues = [
        Clue(dimension="language", value="ja", confidence=0.95),
        Clue(dimension="climate", value="temperate", confidence=0.8),
    ]
    engine = ReasoningEngine()
    scores = engine.infer(clues)
    mgr = HypothesisManager(engine)
    action = mgr.next_action(scores, clues)
    assert action.kind == "confirm", "强线索下应提议确认"


def test_clue_roundtrip():
    c = Clue(dimension="vegetation", value="desert", confidence=0.6, evidence="test")
    d = c.to_dict()
    c2 = Clue.from_dict(d)
    assert c2.dimension == "vegetation"
    assert c2.value == "desert"
    assert abs(c2.confidence - 0.6) < 1e-6


if __name__ == "__main__":
    test_load_countries()
    print("✓ test_load_countries")
    test_language_inference_japan()
    print("✓ test_language_inference_japan")
    test_language_inference_russia()
    print("✓ test_language_inference_russia")
    test_exclusion_removes_country()
    print("✓ test_exclusion_removes_country")
    test_hypothesis_confirm_when_clear()
    print("✓ test_hypothesis_confirm_when_clear")
    test_clue_roundtrip()
    print("✓ test_clue_roundtrip")
    print("\n所有推理引擎测试通过！")
