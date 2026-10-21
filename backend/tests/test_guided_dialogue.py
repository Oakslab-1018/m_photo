"""Focused guided-dialogue API/state tests.

The tests use the committed country/city/admin1 catalogs. Only the optional
image model boundary is stubbed, so no network calls are made.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.database import GeoSession, GuidedDialogue, GuidedEvent
from backend.app.main import create_app
from backend.app.services.catalog_access import catalog_for_settings
from backend.app.services.city_catalog import load_city_index
from backend.app.services.model_gateway import (
    MODEL_ID,
    MODEL_REVISION,
    StreetClipGateway,
    catalog_revision,
)
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


@pytest.fixture
def model_api(tmp_path, monkeypatch):
    settings = replace(make_settings(tmp_path), streetclip_enabled=True)
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: list(CLUES))
    monkeypatch.setattr(StreetClipGateway, "rank_image", lambda self, path: None)
    monkeypatch.setattr(StreetClipGateway, "rank_cities", lambda self, path, iso: None)
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, settings, app


def full_model_evidence(settings, leader: str = "FR") -> dict:
    countries = catalog_for_settings(settings).countries()
    rows = [
        {"iso": country["iso"], "raw_cosine": 0.40 if country["iso"] == leader else 0.10}
        for country in countries
    ]
    rows.sort(key=lambda row: (-row["raw_cosine"], row["iso"]))
    return {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "catalog_revision": catalog_revision(countries),
        "rankings": rows,
        "margin": round(rows[0]["raw_cosine"] - rows[1]["raw_cosine"], 6),
        "uncertain": False,
    }


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


def review_pending_clues(
    client: TestClient,
    session_id: int,
    body: dict,
    *,
    decision: str = "confirmed",
    tag: str = "review",
) -> dict:
    """Answer every pending review_clue question through the normal flow."""
    index = 0
    while True:
        question = body["dialogue"]["pending_question"]
        if not isinstance(question, dict) or question.get("kind") != "review_clue":
            return body
        response = event(client, session_id, {
            "type": "confirm_clue" if decision == "confirmed" else "reject_clue",
            "event_id": f"{tag}-{index}",
            "expected_revision": body["dialogue"]["revision"],
            "question_id": question["id"],
            "clue_id": question["clue_id"],
        })
        assert response.status_code == 200, response.text
        body = response.json()
        index += 1


def confirm_country(
    client: TestClient,
    session_id: int,
    body: dict,
    event_id: str = "country",
    iso: str = "JP",
) -> dict:
    body = review_pending_clues(client, session_id, body, tag=f"{event_id}-review")
    response = event(client, session_id, {
        "type": "confirm_location",
        "event_id": event_id,
        "expected_revision": body["dialogue"]["revision"],
        "target": {"level": "country", "id": iso},
    })
    assert response.status_code == 200, response.text
    return response.json()


def confirm_city(
    client: TestClient,
    session_id: int,
    body: dict,
    event_id: str = "city",
    city_id: str = "1850147",
) -> dict:
    body = review_pending_clues(client, session_id, body, tag=f"{event_id}-review")
    response = event(client, session_id, {
        "type": "confirm_location",
        "event_id": event_id,
        "expected_revision": body["dialogue"]["revision"],
        "target": {"level": "city", "id": city_id},
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
    client, _, app = api
    session_id = upload(client)
    body = start(client, session_id)
    assert body["dialogue"]["stage"] == "country"

    body = confirm_country(client, session_id, body)
    assert body["dialogue"]["stage"] == "city"
    assert body["dialogue"]["confirmed"]["country"]["id"] == "JP"
    assert body["dialogue"]["confirmed"]["admin1"] is None
    assert body["dialogue"]["confirmed"]["city"] is None

    with app.state.session_factory() as db:
        row = db.get(GeoSession, session_id)
        assert (row.confirmed_iso, row.status) == ("JP", "active")

    city_id = "1850147"
    response = event(client, session_id, {
        "type": "confirm_location",
        "event_id": "city",
        "expected_revision": body["dialogue"]["revision"],
        "target": {"level": "city", "id": city_id},
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["dialogue"]["stage"] == "resolved"
    assert body["dialogue"]["confirmed"]["city"]["id"] == city_id
    assert body["dialogue"]["confirmed"]["admin1"] is None

    with app.state.session_factory() as db:
        row = db.get(GeoSession, session_id)
        assert (row.confirmed_iso, row.status) == ("JP", "done")

    direct_admin1 = event(client, session_id, {
        "type": "confirm_location",
        "event_id": "reject-admin1-step",
        "expected_revision": body["dialogue"]["revision"],
        "target": {"level": "admin1", "id": "JP.40"},
    })
    assert direct_admin1.status_code == 422, direct_admin1.text


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


def test_unknown_then_retract_requeues_review_before_location_confirmation(api):
    client, _, _ = api
    session_id = upload(client)
    body = start(client, session_id)
    question = body["dialogue"]["pending_question"]
    assert question is not None and question["kind"] == "review_clue"
    clue_id = question["clue_id"]

    unknown = event(client, session_id, {
        "type": "unknown",
        "event_id": "unknown-review",
        "expected_revision": body["dialogue"]["revision"],
        "question_id": question["id"],
    })
    assert unknown.status_code == 200, unknown.text
    body = unknown.json()
    assert next(
        item for item in body["dialogue"]["clues"] if item["id"] == clue_id
    )["status"] == "uncertain"

    restored = event(client, session_id, {
        "type": "retract",
        "event_id": "retract-unknown-review",
        "expected_revision": body["dialogue"]["revision"],
        "clue_id": clue_id,
    })
    assert restored.status_code == 200, restored.text
    body = restored.json()
    restored_question = body["dialogue"]["pending_question"]
    assert restored_question is not None
    assert restored_question["kind"] == "review_clue"
    assert restored_question["clue_id"] == clue_id

    reviewed = review_pending_clues(
        client, session_id, body, tag="unknown-retract-review"
    )
    assert reviewed["dialogue"]["pending_question"]["kind"] == "confirm_location"
    confirmed = event(client, session_id, {
        "type": "confirm_location",
        "event_id": "confirm-after-unknown-retract",
        "expected_revision": reviewed["dialogue"]["revision"],
        "target": {"level": "country", "id": "JP"},
    })
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["dialogue"]["stage"] == "city"


def test_get_repairs_persisted_pending_clue_with_stale_asked_review(api):
    client, _, app = api
    session_id = upload(client)
    initial = start(client, session_id)
    question = initial["dialogue"]["pending_question"]
    assert question is not None and question["kind"] == "review_clue"
    clue_id = question["clue_id"]

    with app.state.session_factory() as db:
        row = db.get(GuidedDialogue, session_id)
        state = row.state
        for item in state["clues"]:
            if isinstance(item, dict):
                item["status"] = "pending" if item.get("id") == clue_id else "confirmed"
        state["asked_question_ids"] = [f"review:{clue_id}"]
        state["pending_question"] = {
            "id": "confirm:country:JP",
            "kind": "confirm_location",
            "message": "stale confirm question",
            "target": {"level": "country", "id": "JP"},
        }
        row.state = state
        db.commit()

    repaired = client.get(f"/api/geo/{session_id}/dialogue")
    assert repaired.status_code == 200, repaired.text
    body = repaired.json()
    assert body["dialogue"]["revision"] == initial["dialogue"]["revision"] + 1
    repaired_question = body["dialogue"]["pending_question"]
    assert repaired_question is not None
    assert repaired_question["kind"] == "review_clue"
    assert repaired_question["clue_id"] == clue_id
    assert body["messages"] == initial["messages"]

    with app.state.session_factory() as db:
        row = db.get(GuidedDialogue, session_id)
        assert row.revision == body["dialogue"]["revision"]
        assert f"review:{clue_id}" not in row.state["asked_question_ids"]
        assert row.state["pending_question"]["kind"] == "review_clue"

    repeated = client.get(f"/api/geo/{session_id}/dialogue")
    assert repeated.status_code == 200, repeated.text
    repeated_body = repeated.json()
    assert repeated_body["dialogue"]["revision"] == body["dialogue"]["revision"]
    assert repeated_body["messages"] == body["messages"]


def test_scoped_exclusion_filters_only_its_scope_and_retract_restores(api):
    client, _, _ = api
    session_id = upload(client)
    body = confirm_country(client, session_id, start(client, session_id))
    excluded = event(client, session_id, {
        "type": "reject_location",
        "event_id": "exclude-tokyo",
        "expected_revision": body["dialogue"]["revision"],
        "target": {"level": "city", "id": "1850147"},
    })
    assert excluded.status_code == 200, excluded.text
    excluded_body = excluded.json()
    assert all(item["target"]["id"] != "1850147" for item in excluded_body["candidates"])
    assert any(item["target"]["id"] == "1853909" for item in excluded_body["candidates"])

    restored = event(client, session_id, {
        "type": "retract",
        "event_id": "restore-tokyo",
        "expected_revision": excluded_body["dialogue"]["revision"],
        "target": {"level": "city", "id": "1850147"},
    })
    assert restored.status_code == 200, restored.text
    assert any(item["target"]["id"] == "1850147" for item in restored.json()["candidates"])


def test_backtrack_clears_children_but_keeps_exclusions(api):
    client, _, _ = api
    session_id = upload(client)
    body = confirm_country(client, session_id, start(client, session_id))
    excluded = event(client, session_id, {
        "type": "reject_location",
        "event_id": "exclude-city",
        "expected_revision": body["dialogue"]["revision"],
        "target": {"level": "city", "id": "1853909"},
    }).json()
    assert excluded["dialogue"]["stage"] == "city"

    # The old city-stage parent was admin1. It now aliases country.
    backtracked = event(client, session_id, {
        "type": "backtrack",
        "event_id": "backtrack-city-to-country",
        "expected_revision": excluded["dialogue"]["revision"],
        "level": "admin1",
    })
    assert backtracked.status_code == 200, backtracked.text
    result = backtracked.json()
    assert result["dialogue"]["stage"] == "country"
    assert result["dialogue"]["confirmed"]["country"] is None
    assert result["dialogue"]["confirmed"]["admin1"] is None
    assert result["dialogue"]["confirmed"]["city"] is None
    assert any(item["id"] == "1853909" for item in result["dialogue"]["exclusions"])


def test_duplicate_event_id_replays_receipt_and_stale_revision_conflicts(api):
    client, _, _ = api
    session_id = upload(client)
    body = review_pending_clues(client, session_id, start(client, session_id), tag="idempotent-review")
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
    client, settings, _ = api
    session_id = upload(client)
    body = confirm_country(client, session_id, start(client, session_id))
    assert body["dialogue"]["stage"] == "city"
    scoped = sorted(
        load_city_index(settings.cities_path)["JP"].values(),
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
    matched = next(
        item for item in result["candidates"]
        if item["target"]["id"] == str(omitted.city_id)
    )
    assert matched["selection_basis"] == "user_text"
    assert "score" not in matched
    assert result["dialogue"]["confirmed"]["city"] is None

    reviewed = review_pending_clues(client, session_id, result, tag="omitted-city-review")
    confirmed = event(client, session_id, {
        "type": "confirm_location",
        "event_id": "confirm-omitted-city",
        "expected_revision": reviewed["dialogue"]["revision"],
        "target": {"level": "city", "id": str(omitted.city_id)},
    })
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["dialogue"]["stage"] == "resolved"


def test_reject_location_enforces_active_scope(api):
    client, settings, _ = api
    session_id = upload(client)
    body = confirm_country(client, session_id, start(client, session_id))
    foreign_city = next(iter(load_city_index(settings.cities_path)["US"].values())).city_id
    response = event(client, session_id, {
        "type": "reject_location",
        "event_id": "unrelated-city",
        "expected_revision": body["dialogue"]["revision"],
        "target": {"level": "city", "id": str(foreign_city)},
    })
    assert response.status_code == 422, response.text
    current = client.get(f"/api/geo/{session_id}/dialogue").json()
    assert current["dialogue"]["exclusions"] == []


def test_chinese_city_alias_is_proposal_only_and_audited(api):
    client, _, app = api
    session_id = upload(client)
    body = confirm_country(client, session_id, start(client, session_id), event_id="cn-country", iso="CN")
    assert body["dialogue"]["stage"] == "city"
    response = event(client, session_id, {
        "type": "free_text",
        "event_id": "hangzhou-alias",
        "expected_revision": body["dialogue"]["revision"],
        "text": "杭州",
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["dialogue"]["confirmed"]["city"] is None
    matched = next(item for item in result["candidates"] if item["target"]["id"] == "1808926")
    assert matched["selection_basis"] == "user_text"
    assert "score" not in matched
    assert result["messages"][-2]["content"] == "杭州"
    assert result["messages"][-2]["meta"]["event_payload"]["text"] == "杭州"
    assert any(item["id"] == "1808926" for item in result["messages"][-2]["meta"]["proposals"])

    with app.state.session_factory() as db:
        ledger = db.query(GuidedEvent).filter_by(session_id=session_id, event_id="hangzhou-alias").one()
        assert ledger.receipt["audit"]["payload"]["text"] == "杭州"

    queried = client.get(f"/api/geo/{session_id}/dialogue", params={"query": "杭州"})
    assert queried.status_code == 200
    assert queried.json()["candidates"][0]["target"]["id"] == "1808926"

    confirmed = event(client, session_id, {
        "type": "confirm_location",
        "event_id": "confirm-hangzhou",
        "expected_revision": result["dialogue"]["revision"],
        "target": {"level": "city", "id": "1808926"},
    })
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["dialogue"]["stage"] == "resolved"


def test_free_text_conflict_prompts_clarification_without_path_change(api):
    client, _, _ = api
    session_id = upload(client)
    body = confirm_country(client, session_id, start(client, session_id))
    response = event(client, session_id, {
        "type": "free_text",
        "event_id": "conflicting-country",
        "expected_revision": body["dialogue"]["revision"],
        "text": "这是法国",
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["dialogue"]["stage"] == "city"
    assert result["dialogue"]["confirmed"]["country"]["id"] == "JP"
    assert result["dialogue"]["confirmed"]["city"] is None
    assert result["dialogue"]["pending_question"]["kind"] == "ask"
    assert result["action"]["kind"] == "ask"


def test_city_model_ranking_is_country_wide_before_limit(api, monkeypatch):
    client, _, _ = api
    session_id = upload(client)
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
    body = confirm_country(client, session_id, start(client, session_id))
    assert body["dialogue"]["stage"] == "city"
    targets = {item["target"]["id"]: item for item in body["candidates"]}
    assert "1850147" in targets
    assert "1853909" in targets
    assert targets["1850147"]["score"] == pytest.approx(0.4)
    assert targets["1853909"]["score"] == pytest.approx(0.9)


def test_missing_admin1_catalog_does_not_block_country_city_flow(api, monkeypatch):
    client, _, _ = api
    session_id = upload(client)
    confirm_country(client, session_id, start(client, session_id))

    import backend.app.services.geo_catalog_store as store_module

    def missing(self, source_iso):
        raise RuntimeError("missing admin1 catalog")

    monkeypatch.setattr(store_module.GeoCatalogStore, "admin1_rows", missing)
    response = client.get(f"/api/geo/{session_id}/dialogue")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["dialogue"]["stage"] == "city"
    assert body["candidates"]
    assert body["dialogue"]["confirmed"]["admin1"] is None


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
        assert body["dialogue"]["stage"] == "city"


def test_confirm_location_cannot_skip_pending_clue_review(api):
    client, _, app = api
    session_id = upload(client)
    body = start(client, session_id)
    question = body["dialogue"]["pending_question"]
    assert question is not None and question["kind"] == "review_clue"

    blocked = event(client, session_id, {
        "type": "confirm_location",
        "event_id": "skip-clue-review",
        "expected_revision": body["dialogue"]["revision"],
        "target": {"level": "country", "id": "JP"},
    })
    assert blocked.status_code == 409, blocked.text

    current = client.get(f"/api/geo/{session_id}/dialogue").json()
    assert current["dialogue"]["revision"] == body["dialogue"]["revision"]
    assert current["dialogue"]["stage"] == "country"
    assert current["dialogue"]["confirmed"]["country"] is None
    assert current["dialogue"]["pending_question"]["id"] == question["id"]
    with app.state.session_factory() as db:
        assert db.query(GuidedEvent).filter_by(
            session_id=session_id, event_id="skip-clue-review"
        ).first() is None

    reviewed = review_pending_clues(client, session_id, current, tag="skip-review")
    confirmed = event(client, session_id, {
        "type": "confirm_location",
        "event_id": "confirm-after-review",
        "expected_revision": reviewed["dialogue"]["revision"],
        "target": {"level": "country", "id": "JP"},
    })
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["dialogue"]["stage"] == "city"


def test_free_text_proposal_cannot_confirm_past_pending_review(api):
    client, _, _ = api
    session_id = upload(client)
    proposed = event(client, session_id, {
        "type": "free_text",
        "event_id": "free-text-proposal",
        "expected_revision": start(client, session_id)["dialogue"]["revision"],
        "text": "这是日本",
    })
    assert proposed.status_code == 200, proposed.text
    body = proposed.json()
    question = body["dialogue"]["pending_question"]
    assert question is not None and question["kind"] == "review_clue"

    blocked = event(client, session_id, {
        "type": "confirm_location",
        "event_id": "free-text-skip-review",
        "expected_revision": body["dialogue"]["revision"],
        "target": {"level": "country", "id": "JP"},
    })
    assert blocked.status_code == 409, blocked.text
    current = client.get(f"/api/geo/{session_id}/dialogue").json()
    assert current["dialogue"]["confirmed"]["country"] is None
    assert current["dialogue"]["revision"] == body["dialogue"]["revision"]


def test_confirm_location_blocked_while_correction_ask_has_pending_clues(api):
    client, _, _ = api
    session_id = upload(client)
    body = start(client, session_id)
    corrected = event(client, session_id, {
        "type": "free_text",
        "event_id": "correction-ask",
        "expected_revision": body["dialogue"]["revision"],
        "text": "不是日本",
    })
    assert corrected.status_code == 200, corrected.text
    body = corrected.json()
    question = body["dialogue"]["pending_question"]
    assert question is not None and question["kind"] == "ask"
    assert any(item["status"] == "pending" for item in body["dialogue"]["clues"])

    blocked = event(client, session_id, {
        "type": "confirm_location",
        "event_id": "confirm-during-correction",
        "expected_revision": body["dialogue"]["revision"],
        "target": {"level": "country", "id": "JP"},
    })
    assert blocked.status_code == 409, blocked.text
    current = client.get(f"/api/geo/{session_id}/dialogue").json()
    assert current["dialogue"]["stage"] == "country"
    assert current["dialogue"]["confirmed"]["country"] is None
    assert current["dialogue"]["revision"] == body["dialogue"]["revision"]
    assert current["dialogue"]["pending_question"]["id"] == question["id"]

    resolved = event(client, session_id, {
        "type": "answer",
        "event_id": "keep-correction",
        "expected_revision": current["dialogue"]["revision"],
        "question_id": question["id"],
        "value": "keep",
    })
    assert resolved.status_code == 200, resolved.text
    reviewed = review_pending_clues(client, session_id, resolved.json(), tag="post-correction")
    confirmed = event(client, session_id, {
        "type": "confirm_location",
        "event_id": "confirm-after-correction",
        "expected_revision": reviewed["dialogue"]["revision"],
        "target": {"level": "country", "id": "JP"},
    })
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["dialogue"]["stage"] == "city"


def test_confirm_location_blocked_while_conflict_ask_has_pending_clues(api):
    client, _, _ = api
    session_id = upload(client)
    body = confirm_country(client, session_id, start(client, session_id))
    body = confirm_city(client, session_id, body)
    conflicted = event(client, session_id, {
        "type": "free_text",
        "event_id": "conflict-ask",
        "expected_revision": body["dialogue"]["revision"],
        "text": "这是Osaka",
    })
    assert conflicted.status_code == 200, conflicted.text
    body = conflicted.json()
    question = body["dialogue"]["pending_question"]
    assert question is not None and question["kind"] == "ask"
    assert any(item["status"] == "pending" for item in body["dialogue"]["clues"])

    blocked = event(client, session_id, {
        "type": "confirm_location",
        "event_id": "confirm-during-conflict",
        "expected_revision": body["dialogue"]["revision"],
        "target": {"level": "city", "id": "1853909"},
    })
    assert blocked.status_code == 409, blocked.text
    current = client.get(f"/api/geo/{session_id}/dialogue").json()
    assert current["dialogue"]["stage"] == "city"
    assert current["dialogue"]["confirmed"]["city"] is None
    assert current["dialogue"]["revision"] == body["dialogue"]["revision"]

    resolved = event(client, session_id, {
        "type": "answer",
        "event_id": "resolve-conflict",
        "expected_revision": current["dialogue"]["revision"],
        "question_id": question["id"],
        "value": "东京",
    })
    assert resolved.status_code == 200, resolved.text
    reviewed = review_pending_clues(client, session_id, resolved.json(), tag="post-conflict")
    confirmed = event(client, session_id, {
        "type": "confirm_location",
        "event_id": "confirm-after-conflict",
        "expected_revision": reviewed["dialogue"]["revision"],
        "target": {"level": "city", "id": "1850147"},
    })
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["dialogue"]["stage"] == "resolved"


def test_review_question_uses_natural_chinese_labels(api, monkeypatch):
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [{
        "dimension": "building",
        "value": "generic",
        "confidence": 0.9,
        "evidence": "test",
        "source": "building",
    }])
    client, _, _ = api
    session_id = upload(client)
    body = start(client, session_id)
    question = body["dialogue"]["pending_question"]
    assert question is not None and question["kind"] == "review_clue"
    assert "建筑风格" in question["message"]
    assert "普通 / 不明显" in question["message"]
    assert "您觉得准确吗" in question["message"]
    assert "generic" not in question["message"]
    assert "building" not in question["message"]


def test_rollback_when_strict_serialization_fails(api, monkeypatch):
    client, _, app = api
    session_id = upload(client)
    body = review_pending_clues(client, session_id, start(client, session_id), tag="rollback-review")
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
        assert db.query(GuidedEvent).filter_by(
            session_id=session_id, event_id="rollback-country"
        ).first() is None
        session = db.get(GeoSession, session_id)
        assert (session.confirmed_iso, session.status) == (None, "active")


HEURISTIC_COLOR_CLUES = [
    {
        "dimension": "climate",
        "value": "cold_polar",
        "confidence": 0.45,
        "evidence": "dark scene",
        "source": "climate",
    },
    {
        "dimension": "vegetation",
        "value": "urban_sparse",
        "confidence": 0.3,
        "evidence": "sparse green",
        "source": "vegetation",
    },
    {
        "dimension": "vehicle",
        "value": "dark_vehicle",
        "confidence": 0.4,
        "evidence": "dark vehicle-ish region",
        "source": "vehicle",
    },
    {
        "dimension": "building",
        "value": "generic",
        "confidence": 0.25,
        "evidence": "no clear style",
        "source": "building",
    },
]


def test_color_heuristics_never_rank_even_when_confirmed(model_api, monkeypatch):
    client, _, _ = model_api
    monkeypatch.setattr(
        PipelineService,
        "analyze",
        lambda self, path: [dict(item) for item in HEURISTIC_COLOR_CLUES],
    )
    response = client.post(
        "/api/geo/upload",
        files={"image": ("heuristic.png", make_test_image_bytes(), "image/png")},
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["scores"] == []
    assert "暂无可确认的文字线索" in payload["inference_notice"]

    session_id = int(payload["session_id"])
    body = review_pending_clues(
        client, session_id, start(client, session_id), tag="color-policy"
    )
    assert body["scores"] == []
    assert body["candidates"] == []
    assert "暂无可确认的文字线索" in body["inference_notice"]
    question = body["dialogue"]["pending_question"]
    assert question is not None and question["kind"] == "request_clue"


def test_trusted_ocr_clue_still_rule_ranks_when_model_is_missing(model_api, monkeypatch):
    client, _, _ = model_api
    clue = {
        "dimension": "language",
        "value": "ja",
        "confidence": 0.9,
        "evidence": "recognized kana",
        "source": "ocr",
        "extra": {"origin_source": "ocr", "reliable": True},
    }
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [dict(clue)])
    response = client.post(
        "/api/geo/upload",
        files={"image": ("ocr.png", make_test_image_bytes(), "image/png")},
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    assert "本会话未取得可用的本地模型结果" in payload["inference_notice"]
    assert "暂无可确认的文字线索" not in payload["inference_notice"]
    assert payload["scores"][0]["iso"] == "JP"
    assert payload["scores"][0]["score_type"] == "rule_weight"


def test_valid_model_evidence_survives_guided_review(model_api, monkeypatch):
    client, settings, app = model_api
    heuristic = {
        "dimension": "climate",
        "value": "temperate",
        "confidence": 0.3,
        "evidence": "display only",
        "source": "climate",
    }
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [dict(heuristic)])
    monkeypatch.setattr(
        StreetClipGateway,
        "rank_image",
        lambda self, path: full_model_evidence(settings, leader="FR"),
    )
    response = client.post(
        "/api/geo/upload",
        files={"image": ("model.png", make_test_image_bytes(), "image/png")},
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["inference_notice"] is None
    assert payload["scores"][0]["iso"] == "FR"
    assert payload["scores"][0]["score_type"] == "cosine"
    assert payload["scores"][0]["raw_cosine"] == pytest.approx(0.40)

    session_id = int(payload["session_id"])
    body = start(client, session_id)
    assert body["inference_notice"] is None
    assert body["scores"][0]["iso"] == "FR"
    body = review_pending_clues(client, session_id, body, tag="model-review")
    assert body["inference_notice"] is None
    assert body["scores"][0]["iso"] == "FR"
    assert body["scores"][0]["raw_cosine"] == pytest.approx(0.40)

    with app.state.session_factory() as db:
        stored = db.get(GeoSession, session_id).model_evidence
        assert isinstance(stored, dict)
        assert len(stored["rankings"]) == 250
        assert stored["rankings"][0]["iso"] == "FR"


def test_invalid_persisted_model_evidence_is_explicit(model_api, monkeypatch):
    client, settings, app = model_api
    trusted = {
        "dimension": "language",
        "value": "ja",
        "confidence": 0.9,
        "evidence": "recognized kana",
        "source": "user",
    }
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [dict(trusted)])
    monkeypatch.setattr(
        StreetClipGateway,
        "rank_image",
        lambda self, path: full_model_evidence(settings, leader="FR"),
    )
    response = client.post(
        "/api/geo/upload",
        files={"image": ("invalid.png", make_test_image_bytes(), "image/png")},
    )
    assert response.status_code == 201, response.text
    session_id = int(response.json()["session_id"])
    body = start(client, session_id)
    assert body["inference_notice"] is None
    assert body["scores"][0]["iso"] == "FR"

    with app.state.session_factory() as db:
        row = db.get(GeoSession, session_id)
        evidence = dict(row.model_evidence)
        evidence["catalog_revision"] = "0" * 64
        row.model_evidence = evidence
        db.commit()

    body = client.get(f"/api/geo/{session_id}/dialogue").json()
    assert "无效或与当前地点目录不一致" in body["inference_notice"]
    assert body["scores"][0]["iso"] == "JP"
    assert body["scores"][0]["score_type"] == "rule_weight"
    assert body["scores"][0]["raw_cosine"] is None
    with app.state.session_factory() as db:
        assert len(db.get(GeoSession, session_id).model_evidence["rankings"]) == 250


def test_retract_pins_restored_clue_and_unknown_uses_current_question(api, monkeypatch):
    client, _, _ = api
    low_priority = {
        "dimension": "vegetation",
        "value": "urban_sparse",
        "confidence": 0.3,
        "evidence": "display only",
        "source": "vegetation",
    }
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [dict(low_priority)])
    session_id = upload(client)
    body = start(client, session_id)
    question = body["dialogue"]["pending_question"]
    assert question is not None and question["kind"] == "review_clue"
    low_clue_id = question["clue_id"]

    rejected = event(client, session_id, {
        "type": "reject_clue",
        "event_id": "reject-low",
        "expected_revision": body["dialogue"]["revision"],
        "question_id": question["id"],
        "clue_id": low_clue_id,
    })
    assert rejected.status_code == 200, rejected.text
    body = rejected.json()

    proposed = event(client, session_id, {
        "type": "free_text",
        "event_id": "add-place",
        "expected_revision": body["dialogue"]["revision"],
        "text": "这是日本",
    })
    assert proposed.status_code == 200, proposed.text
    body = proposed.json()
    assert body["dialogue"]["pending_question"]["clue_id"] != low_clue_id

    restored = event(client, session_id, {
        "type": "retract",
        "event_id": "restore-low",
        "expected_revision": body["dialogue"]["revision"],
        "clue_id": low_clue_id,
    })
    assert restored.status_code == 200, restored.text
    body = restored.json()
    pinned = body["dialogue"]["pending_question"]
    assert pinned["kind"] == "review_clue"
    assert pinned["clue_id"] == low_clue_id

    unknown = event(client, session_id, {
        "type": "unknown",
        "event_id": "unknown-pinned",
        "expected_revision": body["dialogue"]["revision"],
        "question_id": pinned["id"],
    })
    assert unknown.status_code == 200, unknown.text
    next_question = unknown.json()["dialogue"]["pending_question"]
    assert next_question["kind"] == "review_clue"
    assert next_question["clue_id"] != low_clue_id


def test_confirm_location_blocked_by_unresolved_correction_without_pending_clues(api, monkeypatch):
    client, _, _ = api
    clue = {
        "dimension": "place_name",
        "value": "JP",
        "confidence": 0.9,
        "evidence": "user",
        "source": "user",
    }
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [dict(clue)])
    session_id = upload(client)
    body = review_pending_clues(client, session_id, start(client, session_id), tag="correction-clean")
    assert body["dialogue"]["pending_question"]["kind"] == "confirm_location"

    corrected = event(client, session_id, {
        "type": "free_text",
        "event_id": "correction-with-no-clues",
        "expected_revision": body["dialogue"]["revision"],
        "text": "不是日本",
    })
    assert corrected.status_code == 200, corrected.text
    body = corrected.json()
    assert body["dialogue"]["pending_question"]["kind"] == "ask"
    assert all(clue["status"] != "pending" for clue in body["dialogue"]["clues"])

    blocked = event(client, session_id, {
        "type": "confirm_location",
        "event_id": "confirm-during-correction",
        "expected_revision": body["dialogue"]["revision"],
        "target": {"level": "country", "id": "JP"},
    })
    assert blocked.status_code == 409, blocked.text
    current = client.get(f"/api/geo/{session_id}/dialogue").json()
    assert current["dialogue"]["pending_question"]["kind"] == "ask"
    assert current["dialogue"]["confirmed"]["country"] is None


def test_pending_question_exposes_target_and_confirm_without_qid_stays_valid(api):
    client, _, _ = api
    session_id = upload(client)
    body = review_pending_clues(client, session_id, start(client, session_id), tag="target-field")
    question = body["dialogue"]["pending_question"]
    assert question["kind"] == "confirm_location"
    assert question["target"]["level"] == "country"
    target_id = question["target"]["id"]

    confirmed = event(client, session_id, {
        "type": "confirm_location",
        "event_id": "target-no-qid",
        "expected_revision": body["dialogue"]["revision"],
        "target": {"level": "country", "id": target_id},
    })
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["dialogue"]["stage"] == "city"


def test_degraded_session_drops_stale_confirm_question(model_api, monkeypatch):
    client, _, app = model_api
    heuristic = {
        "dimension": "climate",
        "value": "cold_polar",
        "confidence": 0.45,
        "evidence": "display only",
        "source": "climate",
        "extra": {"display_only": True, "affects_ranking": False, "reliable": False},
    }
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [dict(heuristic)])
    session_id = upload(client)
    body = review_pending_clues(
        client, session_id, start(client, session_id), decision="rejected", tag="stale-confirm"
    )
    assert body["dialogue"]["pending_question"]["kind"] == "request_clue"

    with app.state.session_factory() as db:
        row = db.get(GuidedDialogue, session_id)
        state = dict(row.state)
        state["pending_question"] = {
            "id": "confirm:country:FR",
            "kind": "confirm_location",
            "message": "stale catalog-only confirmation",
            "target": {
                "level": "country",
                "id": "FR",
                "label": "法国",
                "country_iso": "FR",
            },
        }
        row.state = state
        db.commit()

    body = client.get(f"/api/geo/{session_id}/dialogue").json()
    assert body["candidates"] == []
    assert "本会话未取得可用的本地模型结果" in body["inference_notice"]
    assert body["dialogue"]["pending_question"]["kind"] == "request_clue"


def test_retract_pin_keeps_pending_correction_after_review(api, monkeypatch):
    client, _, _ = api
    low_priority = {
        "dimension": "vegetation",
        "value": "urban_sparse",
        "confidence": 0.3,
        "evidence": "display only",
        "source": "vegetation",
    }
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [dict(low_priority)])
    session_id = upload(client)
    body = start(client, session_id)
    question = body["dialogue"]["pending_question"]
    low_clue_id = question["clue_id"]

    rejected = event(client, session_id, {
        "type": "reject_clue",
        "event_id": "correction-pin-reject",
        "expected_revision": body["dialogue"]["revision"],
        "question_id": question["id"],
        "clue_id": low_clue_id,
    })
    assert rejected.status_code == 200, rejected.text
    body = rejected.json()

    added = event(client, session_id, {
        "type": "free_text",
        "event_id": "correction-pin-add",
        "expected_revision": body["dialogue"]["revision"],
        "text": "这是日本",
    })
    assert added.status_code == 200, added.text
    body = added.json()

    corrected = event(client, session_id, {
        "type": "free_text",
        "event_id": "correction-pin-plan",
        "expected_revision": body["dialogue"]["revision"],
        "text": "不是日本",
    })
    assert corrected.status_code == 200, corrected.text
    body = corrected.json()
    assert body["dialogue"]["pending_question"]["kind"] == "ask"

    restored = event(client, session_id, {
        "type": "retract",
        "event_id": "correction-pin-restore",
        "expected_revision": body["dialogue"]["revision"],
        "clue_id": low_clue_id,
    })
    assert restored.status_code == 200, restored.text
    body = restored.json()
    pinned = body["dialogue"]["pending_question"]
    assert pinned["kind"] == "review_clue"
    assert pinned["clue_id"] == low_clue_id

    resolved = event(client, session_id, {
        "type": "unknown",
        "event_id": "correction-pin-resolve",
        "expected_revision": body["dialogue"]["revision"],
        "question_id": pinned["id"],
    })
    assert resolved.status_code == 200, resolved.text
    next_question = resolved.json()["dialogue"]["pending_question"]
    assert next_question["kind"] == "ask"
