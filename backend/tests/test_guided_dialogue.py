"""Focused guided-dialogue API/state tests.

The tests use the committed country/city/admin1 catalogs. Only the optional
image model boundary is stubbed, so no network calls are made.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.database import GeoSession, GuidedDialogue, GuidedEvent
from backend.app.main import create_app
from backend.app.services.admin1_catalog import Admin1CatalogError
from backend.app.services.city_catalog import load_city_index
from backend.app.services.model_gateway import StreetClipGateway
from backend.app.services.pipeline import PipelineService
from backend.tests.support import make_settings, make_test_image_bytes

CLUES = [
    {
        "dimension": "language",
        "value": "ja",
        "confidence": 0.9,
        "evidence": "test",
        "source": "ocr",
    },
    {
        "dimension": "vegetation",
        "value": "temperate_forest",
        "confidence": 0.8,
        "evidence": "test",
        "source": "pipeline",
    },
]


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: list(CLUES))
    monkeypatch.setattr(StreetClipGateway, "rank_image", lambda self, path: None)
    monkeypatch.setattr(StreetClipGateway, "rank_cities", lambda self, path, iso: None)
    settings = make_settings(tmp_path)
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, settings, app


def upload(client: TestClient) -> int:
    response = client.post(
        "/api/geo/upload",
        files={"image": ("guided.png", make_test_image_bytes(), "image/png")},
    )
    assert response.status_code == 201, response.text
    return int(response.json()["session_id"])


def start(client: TestClient, session_id: int) -> dict:
    response = client.post(f"/api/geo/{session_id}/dialogue/start")
    assert response.status_code == 200, response.text
    return response.json()


def event(client: TestClient, session_id: int, payload: dict) -> dict:
    response = client.post(f"/api/geo/{session_id}/dialogue/events", json=payload)
    return response


def confirm_country(
    client: TestClient,
    session_id: int,
    body: dict,
    event_id: str = "country",
    iso: str = "JP",
) -> dict:
    response = event(client, session_id, {
        "type": "confirm_location",
        "event_id": event_id,
        "expected_revision": body["dialogue"]["revision"],
        "target": {"level": "country", "id": iso},
    })
    assert response.status_code == 200, response.text
    return response.json()


def confirm_admin1(client: TestClient, session_id: int, body: dict, event_id: str = "admin1") -> dict:
    response = event(client, session_id, {
        "type": "confirm_location",
        "event_id": event_id,
        "expected_revision": body["dialogue"]["revision"],
        "target": {"level": "admin1", "id": "JP.40"},
    })
    assert response.status_code == 200, response.text
    return response.json()


def test_start_is_idempotent_and_get_is_read_only(api):
    client, _, _ = api
    session_id = upload(client)
    first = start(client, session_id)
    messages = first["messages"]
    second = start(client, session_id)
    assert second["dialogue"] == first["dialogue"]
    assert second["messages"] == messages

    current = client.get(f"/api/geo/{session_id}/dialogue")
    assert current.status_code == 200
    assert current.json()["dialogue"] == first["dialogue"]


def test_explicit_path_and_legacy_status_are_coherent(api):
    client, settings, app = api
    session_id = upload(client)
    body = start(client, session_id)
    assert body["dialogue"]["stage"] == "country"

    body = confirm_country(client, session_id, body)
    assert body["dialogue"]["stage"] == "admin1"
    assert body["dialogue"]["confirmed"]["country"]["id"] == "JP"

    with app.state.session_factory() as db:
        row = db.get(GeoSession, session_id)
        assert (row.confirmed_iso, row.status) == ("JP", "active")

    body = confirm_admin1(client, session_id, body)
    assert body["dialogue"]["stage"] == "city"
    assert body["dialogue"]["confirmed"]["admin1"]["source_iso"] == "JP"

    city_id = next(
        row.city_id
        for row in load_city_index(settings.cities_path)["JP"].values()
        if row.region == "40"
    )
    response = event(client, session_id, {
        "type": "confirm_location",
        "event_id": "city",
        "expected_revision": body["dialogue"]["revision"],
        "target": {"level": "city", "id": str(city_id)},
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["dialogue"]["stage"] == "resolved"
    assert body["dialogue"]["confirmed"]["city"]["id"] == str(city_id)

    with app.state.session_factory() as db:
        row = db.get(GeoSession, session_id)
        assert (row.confirmed_iso, row.status) == ("JP", "done")


def test_clue_review_reject_and_retract(api):
    client, _, _ = api
    session_id = upload(client)
    body = start(client, session_id)
    clue = body["dialogue"]["clues"][0]

    rejected = event(client, session_id, {
        "type": "reject_clue",
        "event_id": "reject-clue",
        "expected_revision": body["dialogue"]["revision"],
        "clue_id": clue["id"],
    })
    assert rejected.status_code == 200, rejected.text
    rejected_body = rejected.json()
    assert next(item for item in rejected_body["dialogue"]["clues"] if item["id"] == clue["id"])["status"] == "rejected"

    restored = event(client, session_id, {
        "type": "retract",
        "event_id": "restore-clue",
        "expected_revision": rejected_body["dialogue"]["revision"],
        "clue_id": clue["id"],
    })
    assert restored.status_code == 200, restored.text
    assert next(
        item for item in restored.json()["dialogue"]["clues"] if item["id"] == clue["id"]
    )["status"] == "pending"


def test_unknown_requires_question_and_does_not_advance(api):
    client, _, _ = api
    session_id = upload(client)
    body = start(client, session_id)
    question = body["dialogue"]["pending_question"]
    assert question is not None
    response = event(client, session_id, {
        "type": "unknown",
        "event_id": "unknown",
        "expected_revision": body["dialogue"]["revision"],
        "question_id": question["id"],
    })
    assert response.status_code == 200, response.text
    unknown = response.json()
    assert unknown["dialogue"]["stage"] == "country"
    assert unknown["dialogue"]["revision"] == body["dialogue"]["revision"] + 1

    stale_question = event(client, session_id, {
        "type": "unknown",
        "event_id": "bad-question",
        "expected_revision": unknown["dialogue"]["revision"],
        "question_id": question["id"],
    })
    assert stale_question.status_code == 409


def test_scoped_exclusion_filters_only_its_scope_and_retract_restores(api):
    client, _, _ = api
    session_id = upload(client)
    body = confirm_country(client, session_id, start(client, session_id))
    excluded = event(client, session_id, {
        "type": "reject_location",
        "event_id": "exclude-tokyo",
        "expected_revision": body["dialogue"]["revision"],
        "target": {"level": "admin1", "id": "JP.40"},
    })
    assert excluded.status_code == 200, excluded.text
    excluded_body = excluded.json()
    assert all(item["target"]["id"] != "JP.40" for item in excluded_body["candidates"])
    assert any(item["target"]["id"] == "JP.32" for item in excluded_body["candidates"])

    restored = event(client, session_id, {
        "type": "retract",
        "event_id": "restore-tokyo",
        "expected_revision": excluded_body["dialogue"]["revision"],
        "target": {"level": "admin1", "id": "JP.40"},
    })
    assert restored.status_code == 200, restored.text
    assert any(item["target"]["id"] == "JP.40" for item in restored.json()["candidates"])


def test_backtrack_clears_children_but_keeps_exclusions(api):
    client, _, _ = api
    session_id = upload(client)
    body = confirm_country(client, session_id, start(client, session_id))
    body = confirm_admin1(client, session_id, body)
    excluded = event(client, session_id, {
        "type": "reject_location",
        "event_id": "exclude-city",
        "expected_revision": body["dialogue"]["revision"],
        "target": {"level": "city", "id": "1850147"},
    }).json()
    backtracked = event(client, session_id, {
        "type": "backtrack",
        "event_id": "backtrack-admin1",
        "expected_revision": excluded["dialogue"]["revision"],
        "level": "admin1",
    })
    assert backtracked.status_code == 200, backtracked.text
    result = backtracked.json()
    assert result["dialogue"]["stage"] == "admin1"
    assert result["dialogue"]["confirmed"]["country"]["id"] == "JP"
    assert result["dialogue"]["confirmed"]["admin1"] is None
    assert any(item["id"] == "1850147" for item in result["dialogue"]["exclusions"])


def test_duplicate_event_id_replays_receipt_and_stale_revision_conflicts(api):
    client, _, _ = api
    session_id = upload(client)
    body = start(client, session_id)
    payload = {
        "type": "confirm_location",
        "event_id": "idempotent-country",
        "expected_revision": body["dialogue"]["revision"],
        "target": {"level": "country", "id": "JP"},
    }
    first = event(client, session_id, payload)
    assert first.status_code == 200, first.text
    first_body = first.json()
    duplicate = event(client, session_id, {**payload, "expected_revision": 999})
    assert duplicate.status_code == 200, duplicate.text
    duplicate_body = duplicate.json()
    assert duplicate_body["receipt"]["event_id"] == "idempotent-country"
    assert duplicate_body["receipt"]["status"] == first_body["receipt"]["status"]
    assert duplicate_body["dialogue"]["revision"] == first_body["dialogue"]["revision"]

    stale = event(client, session_id, {
        "type": "backtrack",
        "event_id": "stale-backtrack",
        "expected_revision": body["dialogue"]["revision"],
        "level": "country",
    })
    assert stale.status_code == 409


def test_free_text_proposes_city_outside_national_top10_without_confirming(api):
    client, settings, app = api
    session_id = upload(client)
    body = confirm_country(client, session_id, start(client, session_id))
    body = confirm_admin1(client, session_id, body)
    scoped = sorted(
        (
            row for row in load_city_index(settings.cities_path)["JP"].values()
            if row.region == "40"
        ),
        key=lambda row: (-row.population, row.city_id),
    )
    omitted = next(
        row for row in scoped[10:]
        if row.name.replace(" ", "").isalpha() and len(row.name) >= 5
    )
    response = event(client, session_id, {
        "type": "free_text",
        "event_id": "free-city",
        "expected_revision": body["dialogue"]["revision"],
        "text": omitted.name,
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["dialogue"]["stage"] == "city"
    assert any(item["target"]["id"] == str(omitted.city_id) for item in result["candidates"])
    assert result["dialogue"]["confirmed"]["city"] is None

    confirmed = event(client, session_id, {
        "type": "confirm_location",
        "event_id": "confirm-omitted-city",
        "expected_revision": result["dialogue"]["revision"],
        "target": {"level": "city", "id": str(omitted.city_id)},
    })
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["dialogue"]["stage"] == "resolved"


def test_reject_location_enforces_active_scope(api):
    client, _, _ = api
    session_id = upload(client)
    body = confirm_country(client, session_id, start(client, session_id))
    response = event(client, session_id, {
        "type": "reject_location",
        "event_id": "unrelated-admin1",
        "expected_revision": body["dialogue"]["revision"],
        "target": {"level": "admin1", "id": "US.CA"},
    })
    assert response.status_code == 422, response.text
    current = client.get(f"/api/geo/{session_id}/dialogue").json()
    assert current["dialogue"]["exclusions"] == []


def test_chinese_city_alias_is_proposal_only_and_audited(api):
    client, _, app = api
    session_id = upload(client)
    body = confirm_country(client, session_id, start(client, session_id), event_id="cn-country", iso="CN")
    response = event(client, session_id, {
        "type": "confirm_location",
        "event_id": "zhejiang",
        "expected_revision": body["dialogue"]["revision"],
        "target": {"level": "admin1", "id": "CN.02"},
    })
    assert response.status_code == 200, response.text
    body = response.json()
    response = event(client, session_id, {
        "type": "free_text",
        "event_id": "hangzhou-alias",
        "expected_revision": body["dialogue"]["revision"],
        "text": "杭州",
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["dialogue"]["confirmed"]["city"] is None
    assert any(item["target"]["id"] == "1808926" for item in result["candidates"])
    assert result["messages"][-2]["content"] == "杭州"
    assert result["messages"][-2]["meta"]["event_payload"]["text"] == "杭州"
    assert any(item["id"] == "1808926" for item in result["messages"][-2]["meta"]["proposals"])

    with app.state.session_factory() as db:
        ledger = db.query(GuidedEvent).filter_by(session_id=session_id, event_id="hangzhou-alias").one()
        assert ledger.receipt["audit"]["payload"]["text"] == "杭州"

    queried = client.get(f"/api/geo/{session_id}/dialogue", params={"query": "杭州"})
    assert queried.status_code == 200
    assert queried.json()["candidates"][0]["target"]["id"] == "1808926"


def test_free_text_conflict_prompts_clarification_without_path_change(api):
    client, _, _ = api
    session_id = upload(client)
    body = confirm_country(client, session_id, start(client, session_id))
    body = confirm_admin1(client, session_id, body)
    response = event(client, session_id, {
        "type": "free_text",
        "event_id": "conflicting-city",
        "expected_revision": body["dialogue"]["revision"],
        "text": "Osaka",
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["dialogue"]["stage"] == "city"
    assert result["dialogue"]["confirmed"]["admin1"]["id"] == "JP.40"
    assert result["dialogue"]["confirmed"]["city"] is None
    assert result["dialogue"]["pending_question"]["kind"] == "ask"
    assert result["action"]["kind"] == "ask"


def test_city_model_ranking_is_scoped_before_limit(api, monkeypatch):
    client, _, _ = api
    session_id = upload(client)
    body = confirm_country(client, session_id, start(client, session_id))
    monkeypatch.setattr(
        StreetClipGateway,
        "rank_cities",
        lambda self, path, iso: {
            "city_scores": [
                {
                    "city_id": "1853909",
                    "iso": "JP",
                    "name": "Osaka",
                    "region": "32",
                    "lat": 34.69379,
                    "lng": 135.50107,
                    "raw_cosine": 0.9,
                    "score": 0.9,
                    "score_type": "city_text_cosine",
                    "reference_cosine": None,
                },
                {
                    "city_id": "1850147",
                    "iso": "JP",
                    "name": "Tokyo",
                    "region": "40",
                    "lat": 35.6895,
                    "lng": 139.69171,
                    "raw_cosine": 0.4,
                    "score": 0.4,
                    "score_type": "city_text_cosine",
                    "reference_cosine": None,
                },
            ]
        },
    )
    body = confirm_admin1(client, session_id, body)
    targets = {item["target"]["id"]: item for item in body["candidates"]}
    assert "1850147" in targets
    assert "1853909" not in targets
    assert targets["1850147"]["score"] == pytest.approx(0.4)


def test_missing_admin1_catalog_is_explicit_unavailable(api, monkeypatch):
    client, _, _ = api
    session_id = upload(client)
    confirm_country(client, session_id, start(client, session_id))

    import backend.app.services.guided_dialogue as guided_module

    def missing():
        raise Admin1CatalogError("missing catalog")

    monkeypatch.setattr(guided_module, "load_admin1_index", missing)
    response = client.get(f"/api/geo/{session_id}/dialogue")
    assert response.status_code == 503
    assert response.json()["error"] == "一级行政区目录不可用"


def test_guided_state_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: list(CLUES))
    monkeypatch.setattr(StreetClipGateway, "rank_image", lambda self, path: None)
    monkeypatch.setattr(StreetClipGateway, "rank_cities", lambda self, path, iso: None)
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        session_id = upload(client)
        body = confirm_country(client, session_id, start(client, session_id))
        revision = body["dialogue"]["revision"]

    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        body = client.get(f"/api/geo/{session_id}/dialogue").json()
        assert body["dialogue"]["revision"] == revision
        assert body["dialogue"]["confirmed"]["country"]["id"] == "JP"
        assert body["dialogue"]["stage"] == "admin1"


def test_rollback_when_strict_serialization_fails(api, monkeypatch):
    client, _, app = api
    session_id = upload(client)
    body = start(client, session_id)
    revision = body["dialogue"]["revision"]

    import backend.app.services.guided_dialogue as guided_module

    def fail(_response):
        raise ValueError("serialize failure")

    monkeypatch.setattr(guided_module.GuidedDialogueService, "_strict_response", staticmethod(fail))
    response = event(client, session_id, {
        "type": "confirm_location",
        "event_id": "rollback-country",
        "expected_revision": revision,
        "target": {"level": "country", "id": "JP"},
    })
    assert response.status_code == 500
    with app.state.session_factory() as db:
        row = db.get(GuidedDialogue, session_id)
        assert row.revision == revision
        assert db.get(GuidedEvent, 1) is None
        session = db.get(GeoSession, session_id)
        assert (session.confirmed_iso, session.status) == (None, "active")
