"""Provider contracts, full-prior persistence and transaction-safe conversation tests."""
import io
import json
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest import mock

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import create_app
from backend.app.database import GeoSession
from backend.app.services.catalog_access import catalog_for_settings
from backend.app.services.geo_session import GeoSessionService
from backend.app.services.model_gateway import StreetClipGateway, MODEL_ID, MODEL_REVISION, catalog_revision
from backend.app.services.deepseek import DeepSeekClient
from backend.tests.support import make_settings, make_test_image_bytes


HEAD_ROWS = 100  # rows that keep the historical 0.01 step (0.40 down to -0.59)


def evidence(settings):
    """Deterministic full-catalog ranking: 0.40 top row, margin 0.01.

    The top rows keep the historical shape so ``margin`` stays 0.01 and the
    uncertainty flag stays True. A fixed 0.01 step would drive the tail below
    the validator's -1 cosine bound once the dynamic catalog passes ~140
    entries, so the remainder steps down by a fraction sized to land just
    above -1 (see ``test_fusion.full_evidence`` for the same problem).
    """
    countries = sorted(catalog_for_settings(settings).countries(), key=lambda c: c["iso"])
    head = min(len(countries), HEAD_ROWS)
    tail = len(countries) - head
    tail_step = min(0.01, 0.40 / tail) if tail else 0.01
    rows = [{"iso": c["iso"],
             "raw_cosine": round(0.40 - i * 0.01 if i < head else -0.59 - (i - head + 1) * tail_step, 6)}
            for i, c in enumerate(countries)]
    return {"model": MODEL_ID, "model_revision": MODEL_REVISION,
            "catalog_revision": catalog_revision(countries), "rankings": rows,
            "margin": 0.01, "uncertain": True}


@pytest.fixture
def settings(tmp_path):
    config = replace(make_settings(tmp_path), streetclip_enabled=True, streetclip_api_key="local-secret")
    # Standalone gateway/provider tests also need the SQLite seed initialized.
    with TestClient(create_app(config)):
        yield config


def test_gateway_uploads_bytes_and_full_validated_ranking(settings, tmp_path):
    image = tmp_path / "test.png"
    image.write_bytes(make_test_image_bytes())
    calls = []
    def handler(request):
        assert request.url.path == "/predict"
        assert request.headers["Authorization"] == "Bearer local-secret"
        assert b"image/jpeg" in request.content
        calls.append(request)
        return httpx.Response(200, json=evidence(settings))
    result = StreetClipGateway(settings, transport=httpx.MockTransport(handler)).rank_image(image)
    assert len(result["rankings"]) == len(evidence(settings)["rankings"])
    assert result["provider_status"] == "ok"
    assert len(result["image_sha256"]) == 64
    assert len(calls) == 1


@pytest.mark.parametrize("corrupt", ["duplicate", "nan", "catalog", "revision", "margin", "sort", "uncertain"])
def test_gateway_rejects_bad_provider_contract(settings, corrupt):
    data = evidence(settings)
    if corrupt == "duplicate": data["rankings"][1]["iso"] = data["rankings"][0]["iso"]
    if corrupt == "nan": data["rankings"][0]["raw_cosine"] = float("nan")
    if corrupt == "catalog": data["catalog_revision"] = "wrong"
    if corrupt == "revision": data["model_revision"] = "main"
    if corrupt == "margin": data["margin"] = 0.4
    if corrupt == "sort": data["rankings"].reverse()
    if corrupt == "uncertain": data["uncertain"] = False
    with pytest.raises(ValueError): StreetClipGateway(settings).validate(data)


def test_gateway_off_or_timeout_falls_back(settings, tmp_path, caplog):
    image = tmp_path / "test.png"; image.write_bytes(make_test_image_bytes())
    def fail(request): raise httpx.ReadTimeout("local-secret must never be logged")
    gateway = StreetClipGateway(settings, transport=httpx.MockTransport(fail))
    assert gateway.rank_image(image) is None
    assert "local-secret" not in caplog.text
    assert StreetClipGateway(replace(settings, streetclip_enabled=False)).rank_image(image) is None


def test_deepseek_key_optional_and_request_sanitized(settings):
    calls = []
    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "deepseek-flash"
        assert body["thinking"] == {"type": "disabled"}
        assert body["max_tokens"] == 512
        assert request.headers["Authorization"] == "Bearer ds-secret"
        assert "private.jpg" not in body["messages"][1]["content"]
        calls.append(body)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": '{"message":"请补充招牌文字线索。"}'}}]})
    transport = httpx.MockTransport(handler)
    assert DeepSeekClient(settings, transport=transport).narrate({}) is None
    cfg = replace(settings, deepseek_api_key="ds-secret", deepseek_base_url="https://api.deepseek.com/v1/")
    assert "ds-secret" not in repr(cfg)
    assert DeepSeekClient(cfg, transport=transport).narrate({"image_path": "private.jpg", "action": {}, "scores": []}) == "请补充招牌文字线索。"
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["timeout", "401", "500", "malformed", "empty", "tool", "claimed_confirmed"])
def test_deepseek_failures_do_not_break_conversation(settings, mode, caplog):
    cfg = replace(settings, deepseek_api_key="ds-secret")
    def handler(request):
        if mode == "timeout": raise httpx.ReadTimeout("ds-secret")
        if mode in {"401", "500"}: return httpx.Response(int(mode), text="ds-secret")
        if mode == "malformed": return httpx.Response(200, json={"choices": []})
        content = '{"message":"已确认拍摄地。"}' if mode == "claimed_confirmed" else '{"message":""}'
        message = {"content": content}
        if mode == "tool": message["tool_calls"] = [{}]
        return httpx.Response(200, json={"choices": [{"message": message}]})
    assert DeepSeekClient(cfg, transport=httpx.MockTransport(handler)).narrate({}) is None
    assert "ds-secret" not in caplog.text


def test_full_model_prior_survives_rejections_restart_and_points(settings):
    data = evidence(settings)
    app = create_app(settings)
    with mock.patch.object(StreetClipGateway, "rank_image", return_value=data) as rank:
        with TestClient(app) as client:
            response = client.post("/api/geo/upload", files={"image": ("x.png", make_test_image_bytes(), "image/png")})
            assert response.status_code == 201, response.text
            initial = response.json(); sid = initial["session_id"]
            assert initial["scores"][0]["raw_cosine"] == 0.4
            assert initial["confidence"]["tier"] == "medium"
            assert initial["confidence"]["action"] == "confirm_candidates"
            assert initial["action"]["kind"] == "confirm"
            with app.state.session_factory() as db:
                assert db.get(GeoSession, sid).confirmed_iso is None
            assert not any(c["dimension"] == "country" for c in initial["clues"])
            for row in data["rankings"][:7]:
                response = client.post(f"/api/geo/{sid}/feedback", json={"type": "reject", "iso": row["iso"]})
                assert response.status_code == 200, response.text
            assert response.json()["scores"][0]["iso"] == data["rankings"][7]["iso"]
            assert client.post(f"/api/geo/{sid}/analyze-point", json={"x": 0.5, "y": 0.5}).status_code == 200
            with app.state.session_factory() as db:
                stored = db.get(GeoSession, sid).model_evidence
                # The full-image prior stays immutable; the point click only
                # appends unique crop evidence (see test_fusion for details).
                assert {k: v for k, v in stored.items() if k != "crops"} == data
                assert stored["rankings"] == data["rankings"]
                assert len(stored["crops"]) == 1
                assert stored["crops"][0]["region"] == {"x": 0.5, "y": 0.5, "size": 22}
        with TestClient(create_app(settings)) as restarted:
            response = restarted.post(f"/api/geo/{sid}/feedback", json={"type": "free_text", "text": "继续分析"})
            assert response.status_code == 200
            assert response.json()["scores"][0]["iso"] == data["rankings"][7]["iso"]
        # Upload ranks the full image once; the point click ranks exactly one
        # crop. Restart/feedback never re-contacts the model.
        assert rank.call_count == 2


def test_external_calls_precede_write_transaction_and_confirmation_skips_llm(settings):
    cfg = replace(settings, deepseek_api_key="test-not-real")
    inside = False
    calls = []
    original = GeoSessionService.transaction
    @contextmanager
    def track(geo):
        nonlocal inside
        assert not inside
        with original(geo):
            inside = True
            try: yield
            finally: inside = False
    def rank(gateway, path):
        assert not inside
        calls.append("image")
        return evidence(cfg)
    narrated = []
    def narrate(provider, context):
        assert not inside
        calls.append("chat")
        action_message = str((context.get("action") or {}).get("message") or "")
        message = f"规则解释：{action_message}"
        narrated.append(message)
        return message
    with mock.patch.object(GeoSessionService, "transaction", track), \
         mock.patch.object(StreetClipGateway, "rank_image", autospec=True, side_effect=rank), \
         mock.patch.object(DeepSeekClient, "narrate", autospec=True, side_effect=narrate):
        with TestClient(create_app(cfg)) as client:
            payload = client.post("/api/geo/upload", files={"image": ("x.png", make_test_image_bytes(), "image/png")}).json()
            sid = payload["session_id"]
            assert payload["messages"][-1]["content"] == payload["action"]["message"] == narrated[-1]
            assert "规则解释：" in payload["messages"][-1]["content"]
            assert payload["messages"][-1]["meta"]["conversation_provider"] == "deepseek"
            for path, body in [("feedback", {"type":"reject", "iso":"AU"}), ("analyze-point", {"x":0.5, "y":0.5})]:
                response = client.post(f"/api/geo/{sid}/{path}", json=body)
                assert response.status_code == 200, response.text
                result = response.json()
                message = result["messages"][-1]
                assert message["content"] == result["action"]["message"]
                assert narrated[-1] in message["content"]
                assert message["meta"]["conversation_provider"] == "deepseek"
            count = calls.count("chat")
            assert client.post(f"/api/geo/{sid}/feedback", json={"type":"confirm", "iso":"JP"}).json()["done"]
            assert calls.count("chat") == count == 3
