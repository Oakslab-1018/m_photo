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
from backend.app.services.geo_session import GeoSessionService
from backend.app.services.model_gateway import (
    MODEL_ID, MODEL_REVISION, StreetClipGateway, catalog_revision,
)
from backend.app.services.catalog_access import catalog_for_settings
from backend.app.services.pipeline import PipelineService
from backend.tests.support import make_settings, make_test_image_bytes


TRUSTED_OCR_CLUE = {
    "dimension": "language",
    "value": "ja",
    "confidence": 0.9,
    "evidence": "recognized kana",
    "source": "ocr",
    "extra": {"origin_source": "ocr", "reliable": True},
}


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


def high_evidence(settings):
    data = evidence(settings)
    for row in data["rankings"]:
        if row["iso"] == "AU":
            row["raw_cosine"] = 0.40
        elif row["iso"] == "FR":
            row["raw_cosine"] = 0.37
    data["rankings"].sort(key=lambda row: (-row["raw_cosine"], row["iso"]))
    data["margin"] = round(
        data["rankings"][0]["raw_cosine"] - data["rankings"][1]["raw_cosine"], 6
    )
    data["uncertain"] = data["margin"] < 0.015
    return data


def _upload(client: TestClient) -> dict:
    response = client.post(
        "/api/geo/upload",
        files={"image": ("x.png", make_test_image_bytes(), "image/png")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _replace_persisted_message(settings, *, provider: str | None) -> dict:
    app = create_app(settings)
    with TestClient(app):
        with app.state.session_factory() as db:
            service = GeoSessionService(db)
            session = service.create("unused.png", "unused.png")
            service.add_message(
                session.id,
                "ai",
                "old template",
                meta={
                    "action": {"kind": "ask", "message": "old template"},
                    "scores": [{"iso": "JP"}],
                    "custom": {"keep": True},
                    "conversation_provider": "deepseek",
                },
            )
            service.replace_last_ai_message(
                session.id,
                "new template",
                {"kind": "result", "message": "new template"},
                provider=provider,
            )
            db.commit()
            return service.list_messages(session.id)[-1]


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


def test_replace_last_ai_message_clears_stale_provider_and_preserves_meta(settings):
    message = _replace_persisted_message(settings, provider=None)

    assert message["content"] == "new template"
    assert "conversation_provider" not in message["meta"]
    assert message["meta"]["action"] == {"kind": "result", "message": "new template"}
    assert message["meta"]["scores"] == [{"iso": "JP"}]
    assert message["meta"]["custom"] == {"keep": True}


def test_replace_last_ai_message_sets_explicit_provider_and_preserves_meta(settings):
    message = _replace_persisted_message(settings, provider="deepseek")

    assert message["content"] == "new template"
    assert message["meta"]["conversation_provider"] == "deepseek"
    assert message["meta"]["action"] == {"kind": "result", "message": "new template"}
    assert message["meta"]["scores"] == [{"iso": "JP"}]
    assert message["meta"]["custom"] == {"keep": True}


def test_model_only_high_template_without_key_has_no_provider_or_llm_call(settings, monkeypatch):
    cfg = replace(settings, deepseek_api_key="")
    calls = []

    def narrate(_client, context):
        calls.append(context)
        return "unexpected narration"

    monkeypatch.setattr(StreetClipGateway, "rank_image", lambda self, path: high_evidence(cfg))
    monkeypatch.setattr(DeepSeekClient, "narrate", narrate)
    app = create_app(cfg)
    with TestClient(app) as client:
        payload = _upload(client)
        assert payload["confidence"]["tier"] == "high"
        assert calls == []
        message = payload["messages"][-1]
        assert "conversation_provider" not in message["meta"]

        with app.state.session_factory() as db:
            persisted = GeoSessionService(db).list_messages(payload["session_id"])[-1]
        assert persisted["content"] == message["content"]
        assert "conversation_provider" not in persisted["meta"]
    assert calls == []


@pytest.mark.parametrize("shape", ["incomplete", "nan", "corrupt"])
def test_corrupt_persisted_prior_falls_back_to_rules_across_restart(settings, shape, monkeypatch):
    monkeypatch.setattr(StreetClipGateway, "rank_image", lambda self, path: evidence(settings))
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [dict(TRUSTED_OCR_CLUE)])
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


def test_accepted_llm_provider_persists_across_api_db_and_restart(settings, monkeypatch):
    cfg = replace(settings, deepseek_api_key="unit-test-key")
    narrated = []
    prefix = "\u89c4\u5219\u89e3\u91ca\uff1a"

    def narrate(_client, context):
        action_message = str((context.get("action") or {}).get("message") or "")
        message = f"{prefix}{action_message}"
        narrated.append(message)
        return message

    monkeypatch.setattr(StreetClipGateway, "rank_image", lambda self, path: evidence(cfg))
    monkeypatch.setattr(DeepSeekClient, "narrate", narrate)
    app = create_app(cfg)
    with TestClient(app) as client:
        payload = _upload(client)
        session_id = payload["session_id"]
        message = payload["messages"][-1]
        assert message["content"] == payload["action"]["message"] == narrated[-1]
        assert message["meta"]["conversation_provider"] == "deepseek"
        assert message["meta"]["scores"]

        with app.state.session_factory() as db:
            persisted = GeoSessionService(db).list_messages(session_id)[-1]
        assert persisted["content"] == message["content"]
        assert persisted["meta"]["conversation_provider"] == "deepseek"

    with TestClient(create_app(cfg)) as restarted:
        restarted_message = restarted.get(
            f"/api/geo/{session_id}/messages"
        ).json()["messages"][-1]
    assert restarted_message["content"] == narrated[-1]
    assert restarted_message["meta"]["conversation_provider"] == "deepseek"
    assert len(narrated) == 1


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

    app = create_app(cfg)
    with TestClient(app) as client:
        payload = _upload(client)
        session_id = payload["session_id"]
        assert len(calls) >= 2
        message = payload["messages"][-1]
        assert message["content"] != "这段预览文字不应写入数据库。"
        assert message["meta"].get("conversation_provider") is None
        stored = client.get(f"/api/geo/{session_id}/messages").json()["messages"][-1]
        assert stored["content"] == message["content"]
        assert stored["meta"].get("conversation_provider") is None
        with app.state.session_factory() as db:
            persisted = GeoSessionService(db).list_messages(session_id)[-1]
        assert persisted["content"] == message["content"]
        assert persisted["meta"].get("conversation_provider") is None

    with TestClient(create_app(cfg)) as restarted:
        restarted_message = restarted.get(
            f"/api/geo/{session_id}/messages"
        ).json()["messages"][-1]
    assert restarted_message["content"] == message["content"]
    assert restarted_message["meta"].get("conversation_provider") is None


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
