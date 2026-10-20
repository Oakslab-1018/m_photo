"""Bounded regressions for persisted prior validation, preview mismatch and privacy."""
from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.database import GeoSession
from backend.app.domain.geolocation.engine import CountryScore
from backend.app.domain.geolocation.model_ranking import ModelAwareEngine
from backend.app.main import create_app
from backend.app.services.deepseek import DeepSeekClient
from backend.app.services.model_gateway import (
    MODEL_ID, MODEL_REVISION, StreetClipGateway, catalog_revision,
)
from backend.app.services.catalog_access import catalog_for_settings
from backend.tests.support import make_settings, make_test_image_bytes


def evidence(settings):
    """Complete SQLite-catalog prior with a medium margin for conversation tests."""
    countries = catalog_for_settings(settings).countries()
    rows = [
        {"iso": item["iso"], "raw_cosine": 0.40 if item["iso"] == "AU" else 0.39 if item["iso"] == "FR" else 0.10}
        for item in countries
    ]
    rows.sort(key=lambda row: (-row["raw_cosine"], row["iso"]))
    return {
        "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "catalog_revision": catalog_revision(countries), "rankings": rows,
        "margin": rows[0]["raw_cosine"] - rows[1]["raw_cosine"], "uncertain": True,
    }


def _upload(client: TestClient) -> dict:
    response = client.post(
        "/api/geo/upload",
        files={"image": ("x.png", make_test_image_bytes(), "image/png")},
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture
def settings(tmp_path):
    config = replace(
        make_settings(tmp_path),
        streetclip_enabled=True,
        streetclip_api_key="local-secret",
    )
    # Standalone provider tests also need the SQLite seed initialized.
    with TestClient(create_app(config)):
        yield config


@pytest.mark.parametrize("shape", ["incomplete", "nan", "corrupt"])
def test_corrupt_persisted_prior_falls_back_to_rules_across_restart(settings, shape, monkeypatch):
    monkeypatch.setattr(StreetClipGateway, "rank_image", lambda self, path: evidence(settings))
    app = create_app(settings)
    with TestClient(app) as client:
        payload = _upload(client)
        session_id = payload["session_id"]
        assert any(score["raw_cosine"] is not None for score in payload["scores"])

        good = evidence(settings)
        if shape == "incomplete":
            bad = {**good, "rankings": good["rankings"][:3]}
            bad["margin"] = round(bad["rankings"][0]["raw_cosine"] - bad["rankings"][1]["raw_cosine"], 6)
        elif shape == "nan":
            bad = {**good, "rankings": [{**row} for row in good["rankings"]]}
            bad["rankings"][0]["raw_cosine"] = float("nan")
        else:
            bad = {"unexpected": "shape"}

        with app.state.session_factory() as db:
            row = db.get(GeoSession, session_id)
            assert row is not None
            row.model_evidence_json = json.dumps(bad, allow_nan=True)
            db.commit()

        rejected = client.post(f"/api/geo/{session_id}/feedback", json={"type": "reject", "iso": "AU"})
        assert rejected.status_code == 200, rejected.text
        assert rejected.json()["scores"]
        assert all(score["score_type"] == "rule_weight" for score in rejected.json()["scores"])

    with TestClient(create_app(settings)) as restarted:
        response = restarted.post(f"/api/geo/{session_id}/feedback", json={"type": "free_text", "text": "继续分析"})
        assert response.status_code == 200, response.text
        assert response.json()["scores"]
        assert all(score["score_type"] == "rule_weight" for score in response.json()["scores"])


def test_preview_mismatch_discards_llm_text_and_persists_deterministic_reply(settings, monkeypatch):
    cfg = replace(settings, deepseek_api_key="unit-test-key")
    calls = []
    original_infer = ModelAwareEngine.infer

    def narrate(_client, _context):
        return "这段预览文字不应写入数据库。"

    def infer(self, clues, exclusions=None, top_n=5):
        calls.append(self)
        if len(calls) == 1:
            return [CountryScore(iso="AU", name="Australia", name_zh="澳大利亚", score=0.99, raw_cosine=0.99, score_type="preview-only", matched_clues=[], lat=0.0, lng=0.0)]
        return original_infer(self, clues, exclusions, top_n)

    monkeypatch.setattr(StreetClipGateway, "rank_image", lambda self, path: evidence(cfg))
    monkeypatch.setattr(DeepSeekClient, "narrate", narrate)
    monkeypatch.setattr(ModelAwareEngine, "infer", infer)

    with TestClient(create_app(cfg)) as client:
        payload = _upload(client)
        session_id = payload["session_id"]
        assert len(calls) >= 2
        message = payload["messages"][-1]
        assert message["content"] != "这段预览文字不应写入数据库。"
        assert message["meta"].get("conversation_provider") is None
        stored = client.get(f"/api/geo/{session_id}/messages").json()["messages"][-1]
        assert stored["content"] == message["content"]
        assert stored["meta"].get("conversation_provider") is None


def test_deepseek_redacts_ocr_user_and_history_secrets(settings):
    cfg = replace(settings, deepseek_api_key="ds-secret", streetclip_api_key="sc-secret")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"message": "请补充文字语言线索。"}, ensure_ascii=False)}}]})

    context = {
        "action": {
            "kind": "ask",
            "message": "请补充线索",
            "payload": {
                "dimensions": ["language"],
                "candidate_details": [{
                    "iso": "JP",
                    "name_zh": "日本",
                    "score": 0.5,
                    "matched_clues": [{"evidence": "nested-secret@example.com"}],
                }],
                "confidence": {
                    "tier": "medium",
                    "action": "confirm_candidates",
                    "top3": [{
                        "iso": "JP",
                        "name_zh": "日本",
                        "score": 0.5,
                        "matched_clues": [{"evidence": "nested-secret@example.com"}],
                    }],
                },
            },
        },
        "scores": [],
        "clues": [
            {"dimension": "signage", "value": "https://private.example/a?token=visible", "evidence": "user@example.com", "source": "ocr"},
            {"dimension": "language", "value": "ja", "evidence": "东京", "source": "user"},
            {"dimension": "signage", "value": "sk-abcdefgh12345678", "evidence": "Bearer abc.def", "source": "ocr"},
        ],
        "feedback": "ds-secret https://private.example/feedback",
        "history": [
            {"role": "user", "content": "hf_abcdefgh12345678"},
            {"role": "assistant", "content": "email me at user@example.com"},
        ],
    }
    result = DeepSeekClient(cfg, transport=httpx.MockTransport(handler)).narrate(context)
    assert result == "请补充文字语言线索。"
    assert "东京" not in json.dumps(seen["body"], ensure_ascii=False)
    for secret in ("https://private.example", "user@example.com", "sk-abcdefgh12345678", "Bearer abc.def", "ds-secret", "hf_abcdefgh12345678", "nested-secret@example.com"):
        assert secret not in json.dumps(seen["body"], ensure_ascii=False)
    assert "ja" in json.dumps(seen["body"], ensure_ascii=False)
    assert "[原始文字已隐藏]" in json.dumps(seen["body"], ensure_ascii=False)