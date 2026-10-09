"""City-candidate endpoint, catalog validation and reference-sidecar tests.

The country ranking -> city lookup -> reference diagnostics chain is exercised
against the *real* GeoNames catalog (``data/cities.json``), so the tests prove
the shipped data wiring, not a fixture. Only the HTTP boundary to the inference
service is stubbed: ``httpx.MockTransport`` for the gateway contract, and a
patched ``rank_cities`` for the endpoint behaviour.
"""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.database import GeoSession
from backend.app.main import create_app
from backend.app.services.city_catalog import catalog_digest, load_city_index
from backend.app.services.model_gateway import (
    MAX_CITY_ROWS,
    MODEL_ID,
    MODEL_REVISION,
    StreetClipGateway,
    catalog_revision,
)
from backend.app.services.pipeline import PipelineService
from backend.tests.support import make_settings, make_test_image_bytes

JP = "JP"


@pytest.fixture
def settings(tmp_path):
    return replace(make_settings(tmp_path), streetclip_enabled=True, streetclip_api_key="local-secret")


def catalog_countries(settings):
    return sorted(json.loads(settings.countries_path.read_text(encoding="utf-8")), key=lambda c: c["iso"])


def top_cities(settings, iso=JP, limit=3):
    """Highest-population catalog rows for ``iso``, in catalog selection order."""
    rows = load_city_index(settings.cities_path).get(iso, {})
    return sorted(rows.values(), key=lambda row: (-row.population, row.city_id))[:limit]


def city_payload(settings, iso=JP, *, cosines=None, considered=None, score_type="city_text_cosine"):
    """A valid ``/predict-cities`` payload built from the real catalog."""
    cities = top_cities(settings, iso)
    cosines = cosines if cosines is not None else [0.5 - index * 0.1 for index in range(len(cities))]
    return {
        "country_iso": iso,
        "scope": {
            "city_limit_per_country": 100,
            "city_catalog_revision": catalog_digest(settings.cities_path),
            "cities_considered": len(cities) if considered is None else considered,
        },
        "status": "ok",
        "city_scores": [
            {
                "city_id": city.city_id,
                "iso": iso,
                "name": city.name,
                "region": city.region,
                "lat": city.lat,
                "lng": city.lng,
                "raw_cosine": cosine,
                "score": cosine,
                "score_type": score_type,
                "reference_cosine": None,
            }
            for city, cosine in zip(cities, cosines)
        ],
    }


def gateway_city_result(settings, payload, iso=JP):
    """What a real ``rank_cities`` call returns: validated and normalised.

    Endpoint tests patch ``rank_cities``, so they must patch it with the
    *validated* contract (string ids, catalog identity); feeding the raw
    provider payload would test a path the production code never takes.
    """
    return {
        **StreetClipGateway(settings).validate_city_ranking(payload, iso),
        "provider_status": "ok",
        "schema_version": 1,
        "image_sha256": "d" * 64,
    }


def reference_sidecar(settings, *, iso=JP, cosine=0.62, with_region=True):
    """A valid labeled-reference sidecar matching the pinned revisions."""
    coverage = sorted({iso})
    sidecar = {
        "index_revision": "a" * 64,
        "model_revision": MODEL_REVISION,
        "catalog_revision": catalog_revision(catalog_countries(settings)),
        "aggregation": "max",
        "matches": [
            {
                "reference_id": "ref-0001",
                "iso": iso,
                "region": "kanto",
                "city": "tokyo",
                "city_id": "1850147",
                "place": "labeled street corner",
                "viewpoint": "north",
                "lat": 35.68,
                "lng": 139.76,
                "raw_cosine": cosine,
            }
        ],
        "regions": [],
        "coverage": coverage,
    }
    if with_region:
        sidecar["regions"] = [
            {
                "iso": iso,
                "region": "kanto",
                "city": "tokyo",
                "raw_cosine": cosine,
                "prototype_count": 2,
                "support": 5,
            }
        ]
    return sidecar


def country_evidence(settings, *, reference=None, leader=JP):
    """Complete ranking led decisively by ``leader``, optionally with a sidecar.

    The leader is singled out so it survives the ``TOP_N_HYPOTHESES`` slice and
    per-country diagnostics can actually be asserted; every other country steps
    down inside the cosine bounds for a 250-entry catalog.
    """
    countries = catalog_countries(settings)
    rest = [country["iso"] for country in countries if country["iso"] != leader]
    step = 1.20 / len(rest)
    rows = [{"iso": leader, "raw_cosine": 0.40}]
    rows += [{"iso": iso, "raw_cosine": round(0.30 - index * step, 6)} for index, iso in enumerate(rest)]
    rows.sort(key=lambda row: (-row["raw_cosine"], row["iso"]))
    payload = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "catalog_revision": catalog_revision(countries),
        "rankings": rows,
        "margin": round(rows[0]["raw_cosine"] - rows[1]["raw_cosine"], 6),
        "uncertain": False,
    }
    if reference is not None:
        payload["reference"] = reference
    return payload


def upload(client: TestClient, name: str = "street.png"):
    response = client.post(
        "/api/geo/upload",
        files={"image": (name, make_test_image_bytes(size=(96, 96)), "image/png")},
    )
    assert response.status_code == 201, response.text
    return response.json()


# ---------------------------------------------------------------- gateway HTTP
def test_rank_cities_posts_multipart_with_bearer_and_returns_validated_rows(settings, tmp_path):
    image = tmp_path / "street.png"
    image.write_bytes(make_test_image_bytes())
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = request.content
        return httpx.Response(200, json=city_payload(settings))

    result = StreetClipGateway(settings, transport=httpx.MockTransport(handler)).rank_cities(image, "jp")

    assert seen["path"] == "/predict-cities"
    assert seen["auth"] == "Bearer local-secret"
    assert b'name="country_iso"' in seen["body"] and b"JP" in seen["body"]
    assert result["provider_status"] == "ok"
    assert len(result["image_sha256"]) == 64
    assert result["country_iso"] == JP
    # The public contract carries decimal-string ids so markers can key on them.
    assert [row["city_id"] for row in result["city_scores"]] == [
        str(city.city_id) for city in top_cities(settings)
    ]
    # Coordinates/names come from the catalog, not from the model service.
    expected = top_cities(settings)[0]
    assert result["city_scores"][0]["lat"] == expected.lat
    assert result["city_scores"][0]["region"] == expected.region


def test_rank_cities_disabled_or_unavailable_returns_none(settings, tmp_path, caplog):
    image = tmp_path / "street.png"
    image.write_bytes(make_test_image_bytes())

    def fail(request):
        raise httpx.ReadTimeout("local-secret must never be logged")

    gateway = StreetClipGateway(settings, transport=httpx.MockTransport(fail))
    assert gateway.rank_cities(image, JP) is None
    assert "local-secret" not in caplog.text
    assert StreetClipGateway(replace(settings, streetclip_enabled=False)).rank_cities(image, JP) is None


@pytest.mark.parametrize(
    "corrupt",
    [
        "foreign_city",
        "unknown_city",
        "duplicate_city",
        "name_mismatch",
        "coordinate_mismatch",
        "unsorted",
        "bad_cosine",
        "bad_score",
        "catalog_mismatch",
        "empty_ok",
        "rows_with_no_cities",
        "considered_below_rows",
        "unknown_score_type",
        "hybrid_without_reference",
        "text_with_reference",
        "too_many_rows",
        "bad_country",
        "bad_status",
    ],
)
def test_validate_city_ranking_rejects_out_of_contract_payloads(settings, corrupt):
    payload = city_payload(settings)
    rows = payload["city_scores"]
    if corrupt == "foreign_city":
        rows[0]["iso"] = "FR"
    elif corrupt == "unknown_city":
        rows[0]["city_id"] = 999_999_999
    elif corrupt == "duplicate_city":
        rows[1] = copy.deepcopy(rows[0])
    elif corrupt == "name_mismatch":
        rows[0]["name"] = "Fabricated"
    elif corrupt == "coordinate_mismatch":
        rows[0]["lat"] = rows[0]["lat"] + 1.0
    elif corrupt == "unsorted":
        rows.reverse()
    elif corrupt == "bad_cosine":
        rows[0]["raw_cosine"] = 1.4
    elif corrupt == "bad_score":
        rows[0]["score"] = float("nan")
    elif corrupt == "catalog_mismatch":
        payload["scope"]["city_catalog_revision"] = "b" * 64
    elif corrupt == "empty_ok":
        payload["city_scores"] = []
        payload["scope"]["cities_considered"] = 0
    elif corrupt == "rows_with_no_cities":
        payload["status"] = "no_cities"
    elif corrupt == "considered_below_rows":
        payload["scope"]["cities_considered"] = 1
    elif corrupt == "unknown_score_type":
        rows[0]["score_type"] = "made_up"
    elif corrupt == "hybrid_without_reference":
        rows[0]["score_type"] = "city_reference_hybrid"
    elif corrupt == "text_with_reference":
        rows[0]["reference_cosine"] = 0.5
    elif corrupt == "too_many_rows":
        payload["city_scores"] = [copy.deepcopy(rows[0]) for _ in range(MAX_CITY_ROWS + 1)]
        payload["scope"]["cities_considered"] = MAX_CITY_ROWS + 1
    elif corrupt == "bad_country":
        payload["country_iso"] = "FR"
    else:
        payload["status"] = "unavailable"

    with pytest.raises(ValueError):
        StreetClipGateway(settings).validate_city_ranking(payload, JP)


def test_validate_city_ranking_accepts_the_degraded_no_cities_payload(settings):
    payload = {
        "country_iso": JP,
        "scope": {"city_limit_per_country": 100, "city_catalog_revision": None, "cities_considered": 0},
        "status": "no_cities",
        "city_scores": [],
    }
    validated = StreetClipGateway(settings).validate_city_ranking(payload, JP)
    assert validated["status"] == "no_cities"
    assert validated["city_scores"] == []


# ------------------------------------------------------------------ endpoint
def test_cities_endpoint_returns_catalog_candidates_for_the_session(settings, monkeypatch):
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [])
    monkeypatch.setattr(StreetClipGateway, "rank_image", lambda self, path: None)
    result = gateway_city_result(settings, city_payload(settings))
    monkeypatch.setattr(StreetClipGateway, "rank_cities", lambda self, path, iso: result)

    with TestClient(create_app(settings)) as client:
        session_id = upload(client)["session_id"]
        response = client.post(f"/api/geo/{session_id}/cities", json={"iso": "jp"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["country_iso"] == JP
    assert body["status"] == "ok"
    assert body["scope"]["cities_considered"] == 3
    assert [row["city_id"] for row in body["city_scores"]] == [
        str(city.city_id) for city in top_cities(settings)
    ]
    for row in body["city_scores"]:
        assert isinstance(row["city_id"], str)
        assert row["iso"] == JP
        assert -1.0 <= row["raw_cosine"] <= 1.0
        assert 0.0 <= row["score"] <= 1.0


def test_cities_endpoint_is_read_only(settings, monkeypatch):
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [])
    monkeypatch.setattr(StreetClipGateway, "rank_image", lambda self, path: None)
    result = gateway_city_result(settings, city_payload(settings))
    monkeypatch.setattr(StreetClipGateway, "rank_cities", lambda self, path, iso: result)

    app = create_app(settings)
    with TestClient(app) as client:
        session_id = upload(client)["session_id"]
        before = client.get(f"/api/geo/{session_id}/messages").json()["messages"]
        scores_before = client.post(
            f"/api/geo/{session_id}/feedback", json={"type": "free_text", "text": "继续"}
        ).json()
        messages_before = scores_before["messages"]

        assert client.post(f"/api/geo/{session_id}/cities", json={"iso": JP}).status_code == 200

        after = client.get(f"/api/geo/{session_id}/messages").json()["messages"]
        assert after == messages_before
        # Browsing a country's cities must never change the ranking or exclusions.
        again = client.post(f"/api/geo/{session_id}/feedback", json={"type": "free_text", "text": "继续"})
        assert [row["iso"] for row in again.json()["scores"]] == [
            row["iso"] for row in scores_before["scores"]
        ]
        assert again.json()["exclusions"] == scores_before["exclusions"]
        assert before != messages_before  # the session really did advance


def test_cities_endpoint_reports_unavailable_without_a_model(settings, monkeypatch):
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [])
    monkeypatch.setattr(StreetClipGateway, "rank_image", lambda self, path: None)
    calls = []
    monkeypatch.setattr(
        StreetClipGateway, "rank_cities",
        lambda self, path, iso: calls.append(iso) or None,
    )

    with TestClient(create_app(settings)) as client:
        session_id = upload(client)["session_id"]
        response = client.post(f"/api/geo/{session_id}/cities", json={"iso": JP})

    assert calls == [JP]
    assert response.status_code == 200
    assert response.json() == {
        "city_scores": [], "country_iso": JP, "scope": None, "status": "unavailable",
    }


def test_cities_endpoint_passes_through_no_cities(settings, monkeypatch):
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [])
    monkeypatch.setattr(StreetClipGateway, "rank_image", lambda self, path: None)
    # Tokelau has a catalog ISO but no settlement above the cities15000
    # population threshold the catalog is built from.
    assert not load_city_index(settings.cities_path).get("TK"), "pick an empty country"
    result = gateway_city_result(settings, {
        "country_iso": "TK",
        "scope": {
            "city_limit_per_country": 100,
            "city_catalog_revision": catalog_digest(settings.cities_path),
            "cities_considered": 0,
        },
        "status": "no_cities",
        "city_scores": [],
    }, iso="TK")
    monkeypatch.setattr(StreetClipGateway, "rank_cities", lambda self, path, iso: result)

    with TestClient(create_app(settings)) as client:
        session_id = upload(client)["session_id"]
        response = client.post(f"/api/geo/{session_id}/cities", json={"iso": "TK"})

    assert response.status_code == 200
    assert response.json()["status"] == "no_cities"
    assert response.json()["city_scores"] == []


def test_cities_endpoint_rejects_unknown_session_image_and_country(settings, monkeypatch):
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [])
    monkeypatch.setattr(StreetClipGateway, "rank_image", lambda self, path: None)

    with TestClient(create_app(settings)) as client:
        session_id = upload(client)["session_id"]
        assert client.post("/api/geo/424242/cities", json={"iso": JP}).status_code == 404
        assert client.post(f"/api/geo/{session_id}/cities", json={"iso": "ZZ"}).status_code == 422
        for bad in [{"iso": "J"}, {"iso": "12"}, {"iso": ""}, {}]:
            assert client.post(f"/api/geo/{session_id}/cities", json=bad).status_code == 422

        # The stored image disappearing is the only way this 404 can trigger.
        for stored in Path(settings.upload_dir).glob("*.png"):
            stored.unlink()
        assert client.post(f"/api/geo/{session_id}/cities", json={"iso": JP}).status_code == 404


# ------------------------------------------------------- reference-sidecar chain
def test_country_scores_carry_reference_diagnostics_when_an_index_is_loaded(settings, monkeypatch):
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [])
    evidence = country_evidence(settings, reference=reference_sidecar(settings))
    monkeypatch.setattr(StreetClipGateway, "rank_image", lambda self, path: evidence)

    with TestClient(create_app(settings)) as client:
        payload = upload(client)

    jp = next(row for row in payload["scores"] if row["iso"] == JP)
    assert jp["reference_cosine"] == pytest.approx(0.62)
    assert [row["reference_id"] for row in jp["reference_matches"]] == ["ref-0001"]
    assert [row["region"] for row in jp["reference_regions"]] == ["kanto"]
    # Diagnostics ride along on every returned candidate so the UI can tell
    # "library configured, nothing matched here" from "no library at all".
    for row in payload["scores"]:
        if row["iso"] == JP:
            continue
        assert row["reference_matches"] == []
        assert row["reference_regions"] == []
        assert row["reference_cosine"] is None


def test_country_scores_omit_reference_fields_without_an_index(settings, monkeypatch):
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [])
    monkeypatch.setattr(StreetClipGateway, "rank_image", lambda self, path: country_evidence(settings))

    with TestClient(create_app(settings)) as client:
        payload = upload(client)

    for row in payload["scores"]:
        assert "reference_matches" not in row
        assert "reference_regions" not in row
        assert "reference_cosine" not in row


def test_reference_diagnostics_never_reorder_countries(settings, monkeypatch):
    """A near-perfect labeled reference must not lift its own country's rank."""
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [])
    base = country_evidence(settings)
    # Target the runner-up: it is inside the returned slice, so the sidecar is
    # actually attached and the equality below proves "cannot reorder" rather
    # than proving nothing at all (a country outside the slice has no row).
    runner_up = base["rankings"][1]["iso"]
    with_reference = country_evidence(
        settings, reference=reference_sidecar(settings, iso=runner_up, cosine=0.99)
    )
    observed = []
    for evidence in (base, with_reference):
        monkeypatch.setattr(StreetClipGateway, "rank_image", lambda self, path, e=evidence: e)
        with TestClient(create_app(settings)) as client:
            observed.append(upload(client)["scores"])

    assert [(row["iso"], row["score"]) for row in observed[0]] == [
        (row["iso"], row["score"]) for row in observed[1]
    ]
    assert observed[1][1]["iso"] == runner_up
    assert observed[1][1]["reference_cosine"] == pytest.approx(0.99)


@pytest.mark.parametrize(
    "corrupt",
    ["bad-model-revision", "bad-catalog", "bad-aggregation", "unsorted-matches",
     "unsorted-regions", "unknown-iso", "bad-cosine", "not-a-dict", "bad-revision-hash"],
)
def test_corrupt_reference_sidecar_only_loses_the_diagnostics(settings, monkeypatch, corrupt):
    """A broken sidecar costs the diagnostics; the country ranking survives it.

    Only the prior is independently validated, so a malformed *diagnostic* must
    never downgrade the user to the rule engine, which here has no clues at all.
    """
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [])
    sidecar = reference_sidecar(settings)
    if corrupt == "bad-model-revision":
        sidecar["model_revision"] = "main"
    elif corrupt == "bad-catalog":
        sidecar["catalog_revision"] = "c" * 64
    elif corrupt == "bad-aggregation":
        sidecar["aggregation"] = "sum"
    elif corrupt == "unsorted-matches":
        sidecar["matches"].append({**sidecar["matches"][0], "reference_id": "ref-0002", "raw_cosine": 0.9})
    elif corrupt == "unsorted-regions":
        sidecar["regions"].append({**sidecar["regions"][0], "region": "kansai", "raw_cosine": 0.9})
    elif corrupt == "unknown-iso":
        sidecar["matches"][0]["iso"] = "ZZ"
    elif corrupt == "bad-cosine":
        sidecar["matches"][0]["raw_cosine"] = 2.0
    elif corrupt == "bad-revision-hash":
        sidecar["index_revision"] = "short"
    else:
        sidecar = "not-an-object"
    monkeypatch.setattr(
        StreetClipGateway, "rank_image",
        lambda self, path: country_evidence(settings, reference=sidecar),
    )

    with TestClient(create_app(settings)) as client:
        payload = upload(client)

    assert payload["scores"]
    assert payload["scores"][0]["iso"] == JP
    assert payload["scores"][0]["score_type"] == "cosine"
    assert all("reference_matches" not in row for row in payload["scores"])
    assert all("reference_cosine" not in row for row in payload["scores"])


def test_crop_evidence_never_persists_the_reference_sidecar(settings, monkeypatch):
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [])
    evidence = country_evidence(settings, reference=reference_sidecar(settings))
    monkeypatch.setattr(StreetClipGateway, "rank_image", lambda self, path: copy.deepcopy(evidence))

    app = create_app(settings)
    with TestClient(app) as client:
        session_id = upload(client)["session_id"]
        response = client.post(f"/api/geo/{session_id}/analyze-point", json={"x": 0.5, "y": 0.5})
        assert response.status_code == 200, response.text
        # The sidecar describes the full image; it is not evidence per crop.
        top = response.json()["scores"][0]
        assert top["reference_matches"] and top["reference_cosine"] == pytest.approx(0.62)

        with app.state.session_factory() as db:
            stored = db.get(GeoSession, session_id).model_evidence
    assert stored["reference"]["matches"][0]["reference_id"] == "ref-0001"
    assert len(stored["crops"]) == 1
    assert "reference" not in stored["crops"][0]


def test_city_hybrid_fusion_rows_pass_through_the_backend(settings, monkeypatch):
    monkeypatch.setattr(PipelineService, "analyze", lambda self, path: [])
    monkeypatch.setattr(StreetClipGateway, "rank_image", lambda self, path: None)
    payload = city_payload(settings, cosines=[0.4, 0.3, 0.2])
    # The inference service fuses labeled-reference cosines into the city score;
    # the backend must preserve the composite and its provenance unchanged.
    payload["city_scores"][0].update(
        {"score_type": "city_reference_hybrid", "reference_cosine": 0.8, "score": 0.4 * 0.4 + 0.6 * 0.8}
    )
    payload["city_scores"].sort(key=lambda row: (-row["score"], row["city_id"]))
    result = gateway_city_result(settings, payload)
    monkeypatch.setattr(StreetClipGateway, "rank_cities", lambda self, path, iso: result)

    with TestClient(create_app(settings)) as client:
        session_id = upload(client)["session_id"]
        body = client.post(f"/api/geo/{session_id}/cities", json={"iso": JP}).json()

    top = body["city_scores"][0]
    assert top["score_type"] == "city_reference_hybrid"
    assert top["reference_cosine"] == pytest.approx(0.8)
    assert top["raw_cosine"] == pytest.approx(0.4)
    assert top["score"] == pytest.approx(0.64)
