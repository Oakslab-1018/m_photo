"""Stage2B API tests: crop fusion, provenance, dedup, failure, restart, atomicity.

All inference (StreetCLIP + OCR pipeline) is mocked; no model service is contacted.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from backend.app.database import ChatMessage, GeoSession
from backend.app.main import create_app
from backend.app.services.deepseek import DeepSeekClient
from backend.app.services.geo_session import GeoSessionService
from backend.app.services.model_gateway import MAX_CROPS, StreetClipGateway
from backend.app.services.pipeline import PipelineService
from backend.tests.support import make_settings, make_test_image_bytes
from backend.tests.test_fusion import biased_evidence, full_evidence

FAKE_POINT_CLUES = [
    {
        "dimension": "language",
        "value": "ja",
        "confidence": 0.9,
        "evidence": "fake OCR line",
        "source": "ocr",
        "extra": {"origin_source": "ocr", "provider": "rapidocr", "reliable": True},
    },
    {
        "dimension": "vegetation",
        "value": "temperate_forest",
        "confidence": 0.8,
        "evidence": "fake vegetation heuristic",
        "source": "vegetation",
        "extra": {},
    },
]


@pytest.fixture
def settings(tmp_path):
    return replace(make_settings(tmp_path), streetclip_enabled=True)


def make_gradient_image_bytes(size=(256, 256)) -> bytes:
    """Non-uniform image so different crop regions produce different bytes."""
    import io

    from PIL import Image

    image = Image.new("RGB", size)
    pixels = image.load()
    for y in range(size[1]):
        for x in range(size[0]):
            pixels[x, y] = (x % 256, y % 256, (x * 7 + y * 13) % 256)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def upload(client: TestClient, image_bytes: bytes | None = None):
    response = client.post(
        "/api/geo/upload",
        files={"image": ("x.png", image_bytes or make_test_image_bytes(size=(256, 256)), "image/png")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def upload_gradient(client: TestClient):
    return upload(client, make_gradient_image_bytes())


def crop_only_pipeline(upload_clues=None):
    """Pipeline mock: first call (upload) as given, later calls (crops) fixed clues."""
    state = {"calls": 0}

    def analyze(self, path):
        state["calls"] += 1
        if state["calls"] > 1:
            return json.loads(json.dumps(FAKE_POINT_CLUES))
        return json.loads(json.dumps(upload_clues or []))

    return analyze


def make_ranker(settings, calls):
    """Full-image evidence on the upload call, crop-specific evidence after."""
    full = full_evidence(settings)

    def rank(self, path):
        calls.append(str(path))
        if len(calls) == 1:
            return json.loads(json.dumps(full))
        return biased_evidence(settings, "CN")

    return rank


def test_api_crop_fusion_changes_ranking_and_preserves_full_prior(settings, monkeypatch):
    monkeypatch.setattr(PipelineService, "analyze", crop_only_pipeline())
    calls: list[str] = []
    monkeypatch.setattr(StreetClipGateway, "rank_image", make_ranker(settings, calls))

    app = create_app(settings)
    with TestClient(app) as client:
        payload = upload(client)
        session_id = payload["session_id"]
        assert payload["scores"][0]["iso"] == "JP"
        assert payload["scores"][0]["score_type"] == "cosine"
        assert all("crops" not in score for score in payload["scores"])

        response = client.post(f"/api/geo/{session_id}/analyze-point", json={"x": 0.5, "y": 0.5})
        assert response.status_code == 200, response.text
        result = response.json()
        assert len(calls) == 2

        top = result["scores"][0]
        assert top["iso"] == "CN"  # crop was ranked before the temp crop was cleaned up
        assert top["score_type"] == "hybrid_rank"
        assert top["fusion"]["crop_weight"] == 0.25
        assert top["fusion"]["crop_mean"] == pytest.approx(0.90)
        assert top["raw_cosine"] == pytest.approx(0.30)  # still the full-image cosine

        # Provenance retained for the crop clue: source is point_click but the
        # originating analyzer is recorded, so trusted fusion still applies.
        ja = next(clue for clue in result["new_clues"] if clue["dimension"] == "language")
        assert ja["source"] == "point_click"
        assert ja["extra"]["origin_source"] == "ocr"
        assert ja["extra"]["reliable"] is True
        assert ja["extra"]["provider"] == "rapidocr"
        assert "点击区域" in ja["evidence"]

        # OCR/rules clues are persisted alongside the untouched full prior.
        with app.state.session_factory() as db:
            row = db.get(GeoSession, session_id)
            evidence = json.loads(row.model_evidence_json)
            assert [r["iso"] for r in evidence["rankings"]][:2] == ["JP", "CN"]
            assert len(evidence["crops"]) == 1
            assert evidence["crops"][0]["region"] == {"x": 0.5, "y": 0.5, "size": 89}
            assert len(evidence["crops"][0]["rankings"]) == len(evidence["rankings"])
            assert row.clues[-1]["source"] == "point_click"


def test_repeat_clicks_do_not_inflate_and_second_region_is_stored(settings, monkeypatch):
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [])
    calls: list[str] = []
    monkeypatch.setattr(StreetClipGateway, "rank_image", make_ranker(settings, calls))

    app = create_app(settings)
    with TestClient(app) as client:
        session_id = upload_gradient(client)["session_id"]
        first = client.post(f"/api/geo/{session_id}/analyze-point", json={"x": 0.5, "y": 0.5}).json()
        second = client.post(f"/api/geo/{session_id}/analyze-point", json={"x": 0.5, "y": 0.5}).json()
        # Same region + identical crop bytes: no new crop, identical ranking.
        assert [s["iso"] for s in first["scores"]] == [s["iso"] for s in second["scores"]]
        assert [s["score"] for s in first["scores"]] == [s["score"] for s in second["scores"]]
        assert second["scores"][0]["fusion"]["crop_count"] == 1

        third = client.post(f"/api/geo/{session_id}/analyze-point", json={"x": 0.1, "y": 0.9}).json()
        assert third["scores"][0]["fusion"]["crop_count"] == 2

        with app.state.session_factory() as db:
            evidence = json.loads(db.get(GeoSession, session_id).model_evidence_json)
            assert len(evidence["crops"]) == 2
            regions = [(c["region"]["x"], c["region"]["y"]) for c in evidence["crops"]]
            assert regions == [(0.5, 0.5), (0.1, 0.9)]


def test_crop_model_failure_keeps_full_prior_and_ocr_still_works(settings, monkeypatch):
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: json.loads(json.dumps(FAKE_POINT_CLUES)))
    calls: list[str] = []

    def rank(self, path):
        calls.append(str(path))
        return full_evidence(settings) if len(calls) == 1 else None

    monkeypatch.setattr(StreetClipGateway, "rank_image", rank)

    app = create_app(settings)
    with TestClient(app) as client:
        session_id = upload(client)["session_id"]
        response = client.post(f"/api/geo/{session_id}/analyze-point", json={"x": 0.5, "y": 0.5})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["new_clues"], "OCR/rules must still function when the crop model fails"
        assert result["scores"][0]["iso"] == "JP"
        assert result["scores"][0]["score_type"] == "hybrid_rank"  # reliable OCR still boosts
        assert result["scores"][0]["fusion"]["crop_count"] == 0
        assert result["scores"][0]["fusion"]["crop_mean"] is None
        with app.state.session_factory() as db:
            evidence = json.loads(db.get(GeoSession, session_id).model_evidence_json)
            assert "crops" not in evidence


def test_disabled_model_makes_no_requests_and_rules_still_work(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)  # streetclip_enabled defaults to False
    calls: list[str] = []

    from backend.app.services import model_gateway

    def forbidden_clean(path):
        calls.append(str(path))
        raise AssertionError("disabled model must not read or upload the image")

    monkeypatch.setattr(model_gateway, "clean_image", forbidden_clean)
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [])

    with TestClient(create_app(settings)) as client:
        session_id = upload(client)["session_id"]
        response = client.post(f"/api/geo/{session_id}/analyze-point", json={"x": 0.5, "y": 0.5})
        assert response.status_code == 200, response.text
        assert calls == []
        assert response.json()["scores"] == [] or all(
            score["score_type"] == "rule_weight" for score in response.json()["scores"]
        )


def test_crop_fusion_survives_restart_without_rerun(settings, monkeypatch):
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [])
    calls: list[str] = []
    monkeypatch.setattr(StreetClipGateway, "rank_image", make_ranker(settings, calls))

    app = create_app(settings)
    with TestClient(app) as client:
        session_id = upload_gradient(client)["session_id"]
        point = client.post(f"/api/geo/{session_id}/analyze-point", json={"x": 0.4, "y": 0.6}).json()
        assert point["scores"][0]["iso"] == "CN"

    assert len(calls) == 2  # upload + crop only; no restart inference

    with TestClient(create_app(settings)) as restarted:
        response = restarted.post(f"/api/geo/{session_id}/feedback", json={"type": "free_text", "text": "继续分析"})
        assert response.status_code == 200, response.text
        scores = response.json()["scores"]
        assert scores[0]["iso"] == "CN"
        assert scores[0]["score_type"] == "hybrid_rank"
        assert scores[0]["fusion"]["crop_count"] == 1
        assert scores[0]["fusion"]["crop_mean"] == pytest.approx(0.90)
        assert scores[0]["raw_cosine"] == pytest.approx(0.30)


@pytest.mark.parametrize("corrupt", ["not-list", "too-many", "bad-region", "bad-cosine", "duplicate-region"])
def test_corrupt_persisted_crops_fall_back_to_rules(settings, monkeypatch, corrupt):
    # Crop click finds a real reliable OCR language clue, so the rules fallback
    # is exercised on understood evidence (never an empty, weakened assertion).
    monkeypatch.setattr(PipelineService, "analyze", crop_only_pipeline())
    calls: list[str] = []
    monkeypatch.setattr(StreetClipGateway, "rank_image", make_ranker(settings, calls))

    app = create_app(settings)
    with TestClient(app) as client:
        payload = upload_gradient(client)
        session_id = payload["session_id"]
        client.post(f"/api/geo/{session_id}/analyze-point", json={"x": 0.5, "y": 0.5})

        with app.state.session_factory() as db:
            row = db.get(GeoSession, session_id)
            evidence = json.loads(row.model_evidence_json)
            crop = evidence["crops"][0]
            if corrupt == "not-list":
                evidence["crops"] = {"unexpected": True}
            elif corrupt == "too-many":
                evidence["crops"] = [json.loads(json.dumps(crop)) for _ in range(MAX_CROPS + 1)]
            elif corrupt == "bad-region":
                crop["region"] = {"x": 5.0, "y": 0.5, "size": 10}
            elif corrupt == "bad-cosine":
                crop["rankings"][0]["raw_cosine"] = float("nan")
            else:
                evidence["crops"].append(json.loads(json.dumps(crop)))
            row.model_evidence_json = json.dumps(evidence, allow_nan=True)
            db.commit()

        response = client.post(f"/api/geo/{session_id}/feedback", json={"type": "free_text", "text": "继续分析"})
        assert response.status_code == 200, response.text
        assert response.json()["scores"]
        assert all(score["score_type"] == "rule_weight" for score in response.json()["scores"])


def test_preview_and_persisted_scores_match_with_crop_fusion(settings, monkeypatch):
    cfg = replace(settings, deepseek_api_key="unit-test-key")
    previews: list[dict] = []

    def narrate(self, context):
        previews.append(context)
        return None

    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: json.loads(json.dumps(FAKE_POINT_CLUES)))
    monkeypatch.setattr(StreetClipGateway, "rank_image", make_ranker(settings, []))
    monkeypatch.setattr(DeepSeekClient, "narrate", narrate)

    with TestClient(create_app(cfg)) as client:
        payload = upload(client)
        session_id = payload["session_id"]
        point = client.post(f"/api/geo/{session_id}/analyze-point", json={"x": 0.5, "y": 0.5}).json()

    assert previews, "preview must run when DeepSeek is configured"
    preview_scores = previews[-1]["scores"]
    assert [(s["iso"], s["score"]) for s in preview_scores] == [(s["iso"], s["score"]) for s in point["scores"]]
    assert preview_scores[0]["score_type"] == "hybrid_rank"


def test_point_crop_write_is_atomic_when_commit_fails(settings, monkeypatch):
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: json.loads(json.dumps(FAKE_POINT_CLUES)))
    calls: list[str] = []
    monkeypatch.setattr(StreetClipGateway, "rank_image", make_ranker(settings, calls))

    app = create_app(settings)
    with TestClient(app) as client:
        payload = upload_gradient(client)
        session_id = payload["session_id"]
        # First click commits one crop + clues.
        assert client.post(f"/api/geo/{session_id}/analyze-point", json={"x": 0.5, "y": 0.5}).status_code == 200

        def snapshot():
            with app.state.session_factory() as db:
                row = db.get(GeoSession, session_id)
                messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).count()
                return json.loads(row.model_evidence_json), json.loads(row.clues_json), messages

        before = snapshot()
        files_before = sorted(p.name for p in settings.upload_dir.iterdir())

        original_commit = GeoSessionService.commit

        def fail_commit(self):
            raise RuntimeError("commit failed")

        monkeypatch.setattr(GeoSessionService, "commit", fail_commit)
        with pytest.raises(RuntimeError, match="commit failed"):
            client.post(f"/api/geo/{session_id}/analyze-point", json={"x": 0.1, "y": 0.9})
        monkeypatch.setattr(GeoSessionService, "commit", original_commit)

        assert snapshot() == before
        assert sorted(p.name for p in settings.upload_dir.iterdir()) == files_before


def test_crops_are_capped_through_the_api(settings, monkeypatch):
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [])
    calls: list[str] = []

    counter = {"n": -1}

    def rank(self, path):
        calls.append(str(path))
        counter["n"] += 1
        return full_evidence(settings) if counter["n"] == 0 else biased_evidence(settings, "CN")

    monkeypatch.setattr(StreetClipGateway, "rank_image", rank)

    app = create_app(settings)
    with TestClient(app) as client:
        session_id = upload_gradient(client)["session_id"]
        for index in range(MAX_CROPS + 3):
            x = (index + 1) / (MAX_CROPS + 4)
            response = client.post(f"/api/geo/{session_id}/analyze-point", json={"x": x, "y": 0.5})
            assert response.status_code == 200, response.text
        with app.state.session_factory() as db:
            evidence = json.loads(db.get(GeoSession, session_id).model_evidence_json)
            assert len(evidence["crops"]) == MAX_CROPS
            assert len(evidence["rankings"]) == 20  # full prior untouched
