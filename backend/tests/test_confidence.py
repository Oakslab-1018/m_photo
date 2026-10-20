"""Deterministic confidence routing and upload/guided integration tests."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.app.database import GuidedDialogue
from backend.app.domain.geolocation.confidence import route_confidence
from backend.app.main import create_app
from backend.app.services.chat import ChatService
from backend.app.services.deepseek import DeepSeekClient, _safe_action_payload, narration_matches_action
from backend.app.services.model_gateway import StreetClipGateway, catalog_revision
from backend.app.services.pipeline import PipelineService
from catalog_db import CatalogDatabase
from backend.tests.support import make_settings, make_test_image_bytes


def _score(iso: str, score: float, name_zh: str | None = None) -> dict:
    return {
        "iso": iso,
        "name": iso,
        "name_zh": name_zh or iso,
        "score": score,
        "raw_cosine": score,
        "score_type": "cosine",
    }


def _evidence(prior: str, second: str, margin: float) -> dict:
    return {
        "rankings": [
            {"iso": prior, "raw_cosine": 0.5},
            {"iso": second, "raw_cosine": 0.5 - margin},
        ],
        "margin": margin,
    }


def test_route_high_medium_low_and_leader_change() -> None:
    high = route_confidence(
        [_score("JP", 0.60), _score("CN", 0.50)],
        _evidence("JP", "CN", 0.03),
    )
    assert high.tier == "high"
    assert high.action == "direct_top1"
    assert high.effective_margin == 0.03

    medium = route_confidence(
        [_score("JP", 0.506), _score("CN", 0.50)],
        _evidence("JP", "CN", 0.01),
    )
    assert medium.tier == "medium"
    assert medium.action == "confirm_candidates"

    low = route_confidence(
        [_score("JP", 0.502), _score("CN", 0.50)],
        _evidence("JP", "CN", 0.002),
    )
    assert low.tier == "low"
    assert low.action == "ask_clue"

    changed = route_confidence(
        [_score("US", 0.60), _score("JP", 0.50)],
        _evidence("JP", "US", 0.03),
    )
    assert changed.leader_changed is True
    assert changed.tier == "medium"


def _ranking(settings, prior: str, second: str, margin: float) -> dict:
    countries = sorted(
        CatalogDatabase(settings.catalog_db_path).countries(),
        key=lambda item: item["iso"],
    )
    values = {}
    for index, country in enumerate(countries):
        iso = country["iso"]
        if iso == prior:
            values[iso] = 0.40
        elif iso == second:
            values[iso] = 0.40 - margin
        else:
            values[iso] = max(-0.50, 0.02 - index * 0.002)
    rows = [{"iso": iso, "raw_cosine": round(value, 6)} for iso, value in values.items()]
    rows.sort(key=lambda row: (-row["raw_cosine"], row["iso"]))
    actual_margin = round(rows[0]["raw_cosine"] - rows[1]["raw_cosine"], 6)
    return {
        "model": "geolocal/StreetCLIP",
        "model_revision": "e3561ba2ad9bf14c9efd6b0092607b8497efbfea",
        "catalog_revision": catalog_revision(countries),
        "rankings": rows,
        "margin": actual_margin,
        "uncertain": actual_margin < 0.015,
    }


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


def _upload_with_margin(
    settings,
    monkeypatch,
    margin: float,
    *,
    deepseek: bool = False,
    pipeline_clues: list[dict] | None = None,
    prior: str = "JP",
    second: str = "CN",
):
    config = settings
    if deepseek:
        from dataclasses import replace

        config = replace(settings, deepseek_api_key="unit-test-key")
    clues = [dict(item) for item in (pipeline_clues or [])]
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [dict(item) for item in clues])
    with TestClient(create_app(config)) as client:
        evidence = _ranking(settings, prior, second, margin)
        monkeypatch.setattr(
            StreetClipGateway,
            "rank_image",
            lambda self, path: json.loads(json.dumps(evidence)),
        )
        response = client.post(
            "/api/geo/upload",
            files={"image": ("route.png", make_test_image_bytes(), "image/png")},
        )
        assert response.status_code == 201, response.text
        body = response.json()
        return client, body


def test_upload_routes_high_to_direct_result(settings, monkeypatch) -> None:
    _, body = _upload_with_margin(settings, monkeypatch, 0.03)
    assert body["confidence"]["tier"] == "high"
    assert body["confidence"]["action"] == "direct_top1"
    assert body["action"]["kind"] == "result"
    assert body["action"]["payload"]["iso"] == "JP"
    assert len(body["confidence"]["top3"]) == 3


def test_upload_routes_medium_to_confirmation(settings, monkeypatch) -> None:
    _, body = _upload_with_margin(settings, monkeypatch, 0.01)
    assert body["confidence"]["tier"] == "medium"
    assert body["action"]["kind"] == "confirm"
    assert body["action"]["payload"]["iso"] == "JP"
    assert "candidate_details" in body["action"]["payload"]


def test_upload_routes_low_to_clue_request(settings, monkeypatch) -> None:
    _, body = _upload_with_margin(settings, monkeypatch, 0.002)
    assert body["confidence"]["tier"] == "low"
    assert body["action"]["kind"] in {"ask", "request_clue"}


def test_high_confidence_never_calls_llm(settings, monkeypatch) -> None:
    contexts: list[dict] = []

    def narrate(self, context):
        contexts.append(context)
        return "高置信预测，但这仍然不是已确认地点。"

    monkeypatch.setattr(DeepSeekClient, "narrate", narrate)
    client, body = _upload_with_margin(settings, monkeypatch, 0.03, deepseek=True)
    guided = client.post(f"/api/geo/{body['session_id']}/dialogue/start")
    assert guided.status_code == 200, guided.text
    assert contexts == []


def test_guided_low_confidence_blocks_country_confirmation(settings, monkeypatch) -> None:
    client, body = _upload_with_margin(settings, monkeypatch, 0.002)
    response = client.post(f"/api/geo/{body['session_id']}/dialogue/start")
    assert response.status_code == 200, response.text
    guided = response.json()
    assert guided["confidence"]["tier"] == "low"
    assert guided["dialogue"]["pending_question"]["kind"] != "confirm_location"

OCR_CLUE = {
    "dimension": "language",
    "value": "ja",
    "confidence": 0.9,
    "evidence": "recognized kana",
    "source": "ocr",
    "extra": {"origin_source": "ocr", "reliable": True},
}


def _score_by_iso(body: dict) -> dict[str, float]:
    return {str(item["iso"]): float(item["score"]) for item in body.get("scores") or []}


def test_route_requires_both_valid_margins_and_leader() -> None:
    fused = [_score("JP", 0.60), _score("CN", 0.50)]
    valid = _evidence("JP", "CN", 0.03)
    variants = []
    missing = _evidence("JP", "CN", 0.03)
    missing.pop("margin")
    variants.append(missing)
    negative = _evidence("JP", "CN", 0.03)
    negative["margin"] = -0.03
    variants.append(negative)
    boolean = _evidence("JP", "CN", 0.03)
    boolean["margin"] = True
    variants.append(boolean)
    inconsistent = _evidence("JP", "CN", 0.03)
    inconsistent["margin"] = 0.04
    variants.append(inconsistent)
    no_leader = {"rankings": [{"iso": "JP", "raw_cosine": 0.5}], "margin": 0.0}
    variants.append(no_leader)
    bad_leader = {"rankings": [{"iso": "??", "raw_cosine": 0.5}, {"iso": "CN", "raw_cosine": 0.4}], "margin": 0.1}
    variants.append(bad_leader)
    for evidence in variants:
        route = route_confidence(fused, evidence)
        assert route.tier == "low"
        assert route.action == "ask_clue"
        assert route.model_only_margin is None
    assert route_confidence([_score("JP", 0.60)], valid).tier == "low"
    assert route_confidence(fused, None).tier == "low"


def test_route_canonicalizes_model_grouping_and_preserves_provenance() -> None:
    evidence = {
        "rankings": [
            {"iso": "TW", "raw_cosine": 0.90},
            {"iso": "HK", "raw_cosine": 0.89},
            {"iso": "JP", "raw_cosine": 0.50},
        ],
        "margin": 0.01,
    }
    fused = [
        {
            "iso": "CN",
            "name": "China",
            "name_zh": "中国",
            "score": 0.60,
            "raw_cosine": 0.60,
            "score_type": "cosine",
            "source_regions": [{"source_iso": "TW"}, {"source_iso": "HK"}],
            "is_grouped": True,
        },
        _score("JP", 0.50),
        _score("FR", 0.40),
    ]
    route = route_confidence(fused, evidence)
    assert route.tier == "high"
    assert route.prior_top_iso == "CN"
    assert route.top1_iso == "CN"
    assert route.model_only_margin == pytest.approx(0.40)
    assert route.top3[0]["source_regions"] == [{"source_iso": "TW"}, {"source_iso": "HK"}]
    assert route.top3[0]["is_grouped"] is True


def _guided_event(client, session_id: int, body: dict, payload: dict) -> dict:
    response = client.post(f"/api/geo/{session_id}/dialogue/events", json={
        **payload,
        "expected_revision": body["dialogue"]["revision"],
    })
    assert response.status_code == 200, response.text
    return response.json()


def test_guided_high_is_prediction_requires_explicit_confirmation(settings, monkeypatch) -> None:
    client, body = _upload_with_margin(
        settings, monkeypatch, 0.03, pipeline_clues=[OCR_CLUE]
    )
    session_id = int(body["session_id"])
    guided = client.post(f"/api/geo/{session_id}/dialogue/start").json()
    assert guided["dialogue"]["stage"] == "country"
    assert guided["confidence"]["tier"] == "high"
    assert guided["confidence"]["action"] == "direct_top1"
    assert guided["action"]["kind"] == "result"
    assert guided["dialogue"]["pending_question"] is None
    assert guided["action"]["payload"]["requires_explicit_confirmation"] is True
    assert "日语" in guided["action"]["message"]
    assert "你可以直接确认" in guided["action"]["message"]
    assert guided["action"]["payload"]["evidence_rationale"] == "可信线索：日语"
    target_iso = guided["action"]["payload"]["iso"]

    confirmed = _guided_event(client, session_id, guided, {
        "type": "confirm_location",
        "event_id": "high-explicit-confirm",
        "target": {"level": "country", "id": target_iso},
    })
    assert confirmed["dialogue"]["stage"] == "admin1"
    assert confirmed["confidence"] is None
    assert confirmed["action"]["payload"].get("confidence") is None


def test_guided_medium_bypasses_routine_review_with_top3_confirmation(settings, monkeypatch) -> None:
    client, body = _upload_with_margin(
        settings, monkeypatch, 0.01, pipeline_clues=[OCR_CLUE]
    )
    session_id = int(body["session_id"])
    guided = client.post(f"/api/geo/{session_id}/dialogue/start").json()
    assert guided["dialogue"]["stage"] == "country"
    assert guided["confidence"]["tier"] == "medium"
    assert guided["action"]["kind"] == "confirm"
    question = guided["dialogue"]["pending_question"]
    assert question["kind"] == "confirm_location"
    assert len(question["options"]) >= 2
    assert guided["action"]["payload"]["requires_explicit_confirmation"] is True
    assert "日语" in guided["action"]["message"]
    assert "下方三个候选" in guided["action"]["message"]

    confirmed = _guided_event(client, session_id, guided, {
        "type": "confirm_location",
        "event_id": "medium-explicit-confirm",
        "target": {"level": "country", "id": question["target"]["id"]},
    })
    assert confirmed["dialogue"]["stage"] == "admin1"
    assert confirmed["confidence"] is None


def test_guided_low_question_budget_exhausts_to_manual_top3(settings, monkeypatch) -> None:
    client, body = _upload_with_margin(
        settings, monkeypatch, 0.002, pipeline_clues=[OCR_CLUE]
    )
    session_id = int(body["session_id"])
    guided = client.post(f"/api/geo/{session_id}/dialogue/start").json()
    assert guided["dialogue"]["pending_question"] is not None
    assert guided["dialogue"]["pending_question"]["id"].startswith("clue:country:")
    for index in range(3):
        question = guided["dialogue"]["pending_question"]
        assert question is not None
        assert question["id"]
        guided = _guided_event(client, session_id, guided, {
            "type": "unknown",
            "event_id": f"low-unknown-{index}",
            "question_id": question["id"],
        })
    assert guided["confidence"]["tier"] == "low"
    assert guided["confidence"]["action"] == "confirm_candidates"
    assert guided["action"]["kind"] == "confirm"
    assert guided["dialogue"]["pending_question"]["kind"] == "confirm_location"
    assert "不会自动猜测" in guided["action"]["message"]
    assert "下方三个候选" in guided["action"]["message"]


def test_guided_answer_adds_user_clue_and_rescores(settings, monkeypatch) -> None:
    client, body = _upload_with_margin(
        settings, monkeypatch, 0.002, pipeline_clues=[OCR_CLUE]
    )
    session_id = int(body["session_id"])
    guided = client.post(f"/api/geo/{session_id}/dialogue/start").json()
    before = _score_by_iso(guided)
    question = guided["dialogue"]["pending_question"]
    assert question["dimension"] != "language"
    assert question["options"]
    option = question["options"][0]
    answered = _guided_event(client, session_id, guided, {
        "type": "answer",
        "event_id": "low-answer",
        "question_id": question["id"],
        "value": option["value"],
    })
    assert any(
        clue["source"] == "user"
        and clue["dimension"] == question["dimension"]
        and clue["value"] == option["value"]
        for clue in answered["dialogue"]["clues"]
    )
    public_user = [
        clue for clue in answered["clues"]
        if clue.get("source") == "user"
        and clue.get("dimension") == question["dimension"]
        and clue.get("value") == option["value"]
    ]
    assert len(public_user) == 1
    assert public_user[0]["extra"]["guided_status"] == "pending"
    after = _score_by_iso(answered)
    assert after != before
    assert after["JP"] > before["JP"]
    assert max(after, key=after.get) == "JP"
    fetched = client.get(f"/api/geo/{session_id}/dialogue")
    assert fetched.status_code == 200, fetched.text
    reloaded_user = [
        clue for clue in fetched.json()["clues"]
        if clue.get("source") == "user"
        and clue.get("dimension") == question["dimension"]
        and clue.get("value") == option["value"]
    ]
    assert len(reloaded_user) == 1
    assert _score_by_iso(fetched.json()) == after


def test_guided_medium_and_low_narrate_on_post_but_not_get(settings, monkeypatch) -> None:
    contexts: list[dict] = []

    def narrate(self, context):
        contexts.append(context)
        action = context.get("action") or {}
        if action.get("kind") == "confirm":
            return "请确认日本（JP）是否是你的拍摄地？"
        return "请补充语言或文字线索？"

    monkeypatch.setattr(DeepSeekClient, "narrate", narrate)
    for margin, event_id in ((0.01, "medium-narration"), (0.002, "low-narration")):
        client, body = _upload_with_margin(
            settings, monkeypatch, margin, deepseek=True, pipeline_clues=[OCR_CLUE]
        )
        before_start = len(contexts)
        response = client.post(f"/api/geo/{body['session_id']}/dialogue/start")
        assert response.status_code == 200, response.text
        started = response.json()
        assert len(contexts) > before_start, event_id
        if started["confidence"]["tier"] == "low":
            pending_question = started["dialogue"]["pending_question"]
            assert started["action"]["payload"]["question_id"] == pending_question["id"]
            assert started["action"]["payload"]["options"] == pending_question["options"]
            assert started["action"]["message"] == pending_question["message"]
        else:
            pending_question = started["dialogue"]["pending_question"]
            assert started["action"]["payload"]["question_id"] == pending_question["id"]
            assert started["action"]["payload"]["options"] == pending_question["options"]
        after_start = len(contexts)
        fetched = client.get(f"/api/geo/{body['session_id']}/dialogue")
        assert fetched.status_code == 200, fetched.text
        assert len(contexts) == after_start
        assert fetched.json()["dialogue"]["revision"] == started["dialogue"]["revision"]


def test_legacy_confirm_remains_done(settings, monkeypatch) -> None:
    client, body = _upload_with_margin(settings, monkeypatch, 0.03)
    response = client.post(
        f"/api/geo/{body['session_id']}/feedback",
        json={"type": "confirm", "iso": "JP"},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["done"] is True
    assert result["action"]["kind"] == "done"
    assert result["confidence"] is None

def test_guided_llm_extracted_user_clue_persists_and_rescores(settings, monkeypatch) -> None:
    extracted: list[dict] = []

    def extract(self, context):
        extracted.append(context)
        return {
            "intent": "assert",
            "clues": [{"dimension": "driving_side", "value": "left", "quote": "left"}],
            "reject": [],
        }

    monkeypatch.setattr(DeepSeekClient, "extract", extract)
    monkeypatch.setattr(DeepSeekClient, "narrate", lambda self, context: None)
    client, body = _upload_with_margin(
        settings, monkeypatch, 0.002, deepseek=True, pipeline_clues=[OCR_CLUE]
    )
    session_id = int(body["session_id"])
    guided = client.post(f"/api/geo/{session_id}/dialogue/start").json()
    before = _score_by_iso(guided)
    answered = _guided_event(client, session_id, guided, {
        "type": "free_text",
        "event_id": "remote-clue",
        "text": "The traffic keeps left.",
    })
    assert extracted
    assert any(
        clue["source"] == "user"
        and clue["dimension"] == "driving_side"
        and clue["value"] == "left"
        for clue in answered["dialogue"]["clues"]
    )
    assert any(
        clue["source"] == "user"
        and clue["dimension"] == "driving_side"
        and clue["value"] == "left"
        for clue in answered["clues"]
    )
    after = _score_by_iso(answered)
    assert after["JP"] > before["JP"]
    assert max(after, key=after.get) == "JP"
    with client.app.state.session_factory() as db:
        state = db.get(GuidedDialogue, session_id).state
        persisted = [
            item for item in state["clues"]
            if item.get("dimension") == "driving_side" and item.get("value") == "left"
        ]
        assert persisted
        assert persisted[-1]["_extra"]["parser"] == "deepseek"

def test_guided_free_text_consumes_low_question_budget_once(settings, monkeypatch) -> None:
    client, body = _upload_with_margin(
        settings, monkeypatch, 0.002, pipeline_clues=[OCR_CLUE]
    )
    session_id = int(body["session_id"])
    guided = client.post(f"/api/geo/{session_id}/dialogue/start").json()
    question = guided["dialogue"]["pending_question"]
    assert question["id"].startswith("clue:country:")

    non_answer = client.post(f"/api/geo/{session_id}/dialogue/events", json={
        "type": "free_text",
        "event_id": "free-text-non-answer",
        "expected_revision": guided["dialogue"]["revision"],
        "text": "为什么问这个？",
    })
    assert non_answer.status_code == 200, non_answer.text
    assert non_answer.json()["dialogue"]["pending_question"]["id"] == question["id"]
    with client.app.state.session_factory() as db:
        history = db.get(GuidedDialogue, session_id).state["asked_clue_questions"]
        assert history == []

    first_event = {
        "type": "free_text",
        "event_id": "free-text-answer-1",
        "expected_revision": guided["dialogue"]["revision"],
        "text": "靠左行驶",
    }
    first = client.post(f"/api/geo/{session_id}/dialogue/events", json=first_event)
    assert first.status_code == 200, first.text
    answered = first.json()
    with client.app.state.session_factory() as db:
        assert len(db.get(GuidedDialogue, session_id).state["asked_clue_questions"]) == 1
    replay = client.post(f"/api/geo/{session_id}/dialogue/events", json=first_event)
    assert replay.status_code == 200, replay.text
    with client.app.state.session_factory() as db:
        assert len(db.get(GuidedDialogue, session_id).state["asked_clue_questions"]) == 1

    reloaded = client.get(f"/api/geo/{session_id}/dialogue")
    assert reloaded.status_code == 200, reloaded.text
    current = reloaded.json()
    assert current["dialogue"]["revision"] == answered["dialogue"]["revision"]
    for index, text_value in ((2, "看不清"), (3, "不确定")):
        question = current["dialogue"]["pending_question"]
        assert question["id"].startswith("clue:country:")
        response = client.post(f"/api/geo/{session_id}/dialogue/events", json={
            "type": "free_text",
            "event_id": f"free-text-uncertain-{index}",
            "expected_revision": current["dialogue"]["revision"],
            "text": text_value,
        })
        assert response.status_code == 200, response.text
        current = response.json()
    with client.app.state.session_factory() as db:
        history = db.get(GuidedDialogue, session_id).state["asked_clue_questions"]
        assert len(history) == 3
    assert current["confidence"]["tier"] == "low"
    assert current["confidence"]["action"] == "confirm_candidates"
    assert current["dialogue"]["pending_question"]["kind"] == "confirm_location"

def test_deepseek_route_payload_sanitizes_nested_evidence() -> None:
    payload = {
        "dimensions": ["language"],
        "candidate_details": [{
            "iso": "JP",
            "score": 0.5,
            "matched_clues": [{"evidence": "nested-secret@example.com"}],
        }],
        "confidence": {
            "tier": "medium",
            "action": "confirm_candidates",
            "top3": [{
                "iso": "JP",
                "score": 0.5,
                "matched_clues": [{"evidence": "nested-secret@example.com"}],
            }],
        },
    }
    safe = _safe_action_payload(payload)
    encoded = json.dumps(safe, ensure_ascii=False)
    assert "nested-secret@example.com" not in encoded
    assert safe["confidence"]["top3"][0]["iso"] == "JP"


def test_narration_must_match_deterministic_action() -> None:
    language_action = {"kind": "ask", "payload": {"dimensions": ["language"]}}
    driving_action = {
        "kind": "ask",
        "payload": {
            "dimension": "driving_side",
            "options": [
                {"value": "left", "label": "靠左行驶"},
                {"value": "right", "label": "靠右行驶"},
            ],
        },
    }
    assert narration_matches_action("请补充语言线索？", language_action)
    assert not narration_matches_action("这是一个普通说明。", language_action)
    assert not narration_matches_action("你喜欢什么颜色？", driving_action)
    assert narration_matches_action("请说明车辆靠左还是靠右行驶？", driving_action)
    assert narration_matches_action(
        "请确认为日本（JP）？",
        {"kind": "confirm", "payload": {"target": {"id": "JP", "label": "日本"}}},
    )
    assert not narration_matches_action(
        "请选择一个地点。",
        {"kind": "confirm", "payload": {"target": {"id": "JP", "label": "日本"}}},
    )

def test_high_rationale_uses_catalog_country_label_for_place_name(settings, monkeypatch) -> None:
    place_clue = {
        "dimension": "place_name",
        "value": "FR",
        "confidence": 0.9,
        "evidence": "user supplied country",
        "source": "user",
    }
    client, body = _upload_with_margin(
        settings,
        monkeypatch,
        0.03,
        pipeline_clues=[place_clue],
        prior="FR",
        second="CN",
    )
    guided = client.post(f"/api/geo/{body['session_id']}/dialogue/start").json()
    assert guided["confidence"]["tier"] == "high"
    assert "可信线索：法国" in guided["action"]["message"]
    assert "可信线索：FR" not in guided["action"]["message"]

def test_rejected_guided_user_clue_is_visible_but_not_scored(settings, monkeypatch) -> None:
    client, body = _upload_with_margin(
        settings, monkeypatch, 0.002, pipeline_clues=[OCR_CLUE]
    )
    session_id = int(body["session_id"])
    guided = client.post(f"/api/geo/{session_id}/dialogue/start").json()
    question = guided["dialogue"]["pending_question"]
    option = question["options"][0]
    answered = _guided_event(client, session_id, guided, {
        "type": "answer",
        "event_id": "rejectable-user-clue",
        "question_id": question["id"],
        "value": option["value"],
    })
    scored = _score_by_iso(answered)
    with client.app.state.session_factory() as db:
        row = db.get(GuidedDialogue, session_id)
        state = dict(row.state)
        changed = False
        records = []
        for item in state.get("clues") or []:
            record = dict(item) if isinstance(item, dict) else item
            if (
                isinstance(record, dict)
                and record.get("_origin") == "guided_user"
                and record.get("dimension") == question["dimension"]
                and record.get("value") == option["value"]
            ):
                record["status"] = "rejected"
                changed = True
            records.append(record)
        assert changed
        state["clues"] = records
        row.state = state
        db.commit()

    response = client.get(f"/api/geo/{session_id}/dialogue")
    assert response.status_code == 200, response.text
    current = response.json()
    rejected = [
        clue for clue in current["clues"]
        if clue.get("dimension") == question["dimension"]
        and clue.get("value") == option["value"]
    ]
    assert len(rejected) == 1
    assert rejected[0]["extra"]["guided_status"] == "rejected"
    rescored = _score_by_iso(current)
    assert rescored["JP"] < scored["JP"]

