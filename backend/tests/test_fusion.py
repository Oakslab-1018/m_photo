"""Stage2B: trusted-clue + local-crop fusion over the immutable full-image prior.

All inference is mocked; no StreetCLIP/RapidOCR service is contacted.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.database import GeoSession
from backend.app.domain.geolocation.clue import Clue
from backend.app.domain.geolocation.model_ranking import (
    CLUE_BOOST_CAP,
    CLUE_BOOST_SCALE,
    CROP_WEIGHT,
    ModelAwareEngine,
)
from backend.app.main import create_app
from backend.app.services.chat import ChatService
from backend.app.services.deepseek import DeepSeekClient
from backend.app.services.geo_session import GeoSessionService
from backend.app.services.model_gateway import (
    MAX_CROPS,
    MODEL_ID,
    MODEL_REVISION,
    StreetClipGateway,
    catalog_revision,
)
from backend.app.services.pipeline import PipelineService
from backend.tests.support import make_settings, make_test_image_bytes

FAKE_OCR_CLUES = [
    {
        "dimension": "language",
        "value": "ja",
        "confidence": 0.9,
        "evidence": "fake OCR line",
        "source": "ocr",
        "extra": {"origin_source": "ocr", "provider": "rapidocr", "reliable": True},
    }
]


def countries_of(settings: Settings) -> list[dict]:
    return json.loads(settings.countries_path.read_text(encoding="utf-8"))


def _ranking_payload(settings: Settings, values: dict[str, float]) -> dict:
    countries = sorted(countries_of(settings), key=lambda c: c["iso"])
    rows = [{"iso": country["iso"], "raw_cosine": round(values[country["iso"]], 6)}
            for country in countries]
    rows.sort(key=lambda row: (-row["raw_cosine"], row["iso"]))
    margin = round(rows[0]["raw_cosine"] - rows[1]["raw_cosine"], 6)
    return {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "catalog_revision": catalog_revision(countries_of(settings)),
        "rankings": rows,
        "margin": margin,
        "uncertain": margin < 0.015,
    }


def full_evidence(settings: Settings) -> dict:
    """Deterministic full-image prior: JP first, CN second, others well below."""
    countries = sorted(countries_of(settings), key=lambda c: c["iso"])
    values = {}
    for index, country in enumerate(countries):
        iso = country["iso"]
        values[iso] = {"JP": 0.40, "CN": 0.30}.get(
            iso, max(-0.50, 0.02 - index * 0.002)
        )
    return _ranking_payload(settings, values)


def biased_evidence(settings: Settings, iso: str) -> dict:
    """A complete crop ranking where ``iso`` wins clearly."""
    countries = sorted(countries_of(settings), key=lambda c: c["iso"])
    values = {}
    for index, country in enumerate(countries):
        current = country["iso"]
        values[current] = 0.90 if current == iso else max(-0.50, 0.04 - index * 0.002)
    return _ranking_payload(settings, values)


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


def _engine(settings: Settings, evidence: dict | None, clues: list[Clue]) -> ModelAwareEngine:
    engine = ModelAwareEngine()
    engine.evidence = evidence
    return engine


def _clue(**kwargs) -> Clue:
    return Clue(**kwargs)


def crop_record(settings: Settings, region: dict, iso: str, digest: str) -> dict:
    gateway = StreetClipGateway(settings)
    return gateway.validate_crop(
        {**biased_evidence(settings, iso), "image_sha256": digest},
        region,
    )


# --------------------------------------------------------------------------- engine


def test_full_prior_without_secondary_evidence_stays_cosine(settings):
    evidence = StreetClipGateway(settings).validate(full_evidence(settings))
    engine = _engine(settings, evidence, [])
    scores = engine.infer([_clue(dimension="building", value="gothic", confidence=0.9, source="building")])
    assert scores
    assert all(score.score_type == "cosine" for score in scores)
    assert all(not score.fusion for score in scores)
    assert scores[0].iso == "JP"
    assert scores[0].score == pytest.approx(scores[0].raw_cosine, abs=1e-9)


def test_crop_mean_uses_modest_weight_and_keeps_full_image_cosine(settings):
    evidence = StreetClipGateway(settings).validate(full_evidence(settings))
    base = {s.iso: s.score for s in _engine(settings, evidence, []).infer([])}
    crop = crop_record(settings, {"x": 0.5, "y": 0.5, "size": 32}, "CN", "a" * 64)
    with_crop = StreetClipGateway(settings).with_crop(evidence, crop)
    assert with_crop is not None and evidence.get("crops") is None
    fused = _engine(settings, StreetClipGateway(settings).validate(with_crop), []).infer([])

    latin = {row["iso"]: row["raw_cosine"] for row in evidence["rankings"]}
    cn = next(score for score in fused if score.iso == "CN")
    expected_crop_mean = 0.90
    expected = (latin["CN"] + CROP_WEIGHT * expected_crop_mean) / (1.0 + CROP_WEIGHT)
    assert cn.score == pytest.approx(expected, abs=1e-6)
    assert cn.raw_cosine == pytest.approx(latin["CN"], abs=1e-9)
    assert cn.score_type == "hybrid_rank"
    assert cn.fusion["crop_weight"] == CROP_WEIGHT
    assert cn.fusion["crop_mean"] == pytest.approx(expected_crop_mean)
    # Modest: the crop moves CN over JP but the immutable full ranking is intact.
    assert fused[0].iso == "CN"
    assert [row["iso"] for row in evidence["rankings"]][:2] == ["JP", "CN"]


def test_repeat_crop_and_repeat_clues_do_not_inflate(settings):
    evidence = StreetClipGateway(settings).validate(full_evidence(settings))
    crop = crop_record(settings, {"x": 0.5, "y": 0.5, "size": 32}, "CN", "b" * 64)
    gateway = StreetClipGateway(settings)
    once = gateway.with_crop(evidence, crop)
    assert gateway.with_crop(once, crop) is None  # same region + hash
    assert gateway.with_crop(once, {**crop, "image_sha256": "c" * 64}) is None  # same region
    assert gateway.with_crop(once, {**crop, "region": {"x": 0.7, "y": 0.7, "size": 32}}) is None  # same hash

    clue = _clue(dimension="language", value="ja", confidence=0.9, source="ocr",
                 extra={"origin_source": "ocr", "reliable": True})
    once_scores = _engine(settings, gateway.validate(once), [clue]).infer([clue])
    twice_scores = _engine(settings, gateway.validate(once), [clue, clue]).infer([clue, clue])
    assert [(s.iso, s.score) for s in once_scores] == [(s.iso, s.score) for s in twice_scores]


def test_strongest_confidence_wins_and_user_wins_ties(settings):
    evidence = StreetClipGateway(settings).validate(full_evidence(settings))
    weak_ocr = _clue(dimension="language", value="ja", confidence=0.4, source="ocr",
                     extra={"origin_source": "ocr", "reliable": True})
    strong_user = _clue(dimension="language", value="ja", confidence=0.95, source="user")
    tie_user = _clue(dimension="language", value="ja", confidence=0.4, source="user")

    def jp_boost(clues):
        scores = _engine(settings, evidence, clues).infer(clues)
        jp = next(score for score in scores if score.iso == "JP")
        return jp.fusion.get("clue_score", 0.0)

    assert jp_boost([weak_ocr, strong_user]) == pytest.approx(0.95 * 2.0)
    assert jp_boost([weak_ocr, tie_user]) == pytest.approx(0.4 * 2.0)  # user tie wins
    assert jp_boost([strong_user, weak_ocr]) == pytest.approx(0.95 * 2.0)  # order independent
    # Reliable OCR alone still boosts; plain heuristics never do.
    heuristic = _clue(dimension="language", value="ja", confidence=1.0, source="vegetation")
    assert jp_boost([weak_ocr]) == pytest.approx(0.4 * 2.0)
    assert jp_boost([heuristic]) == 0.0


def test_reliable_ocr_language_boosts_only_matching_country_and_han_is_not_cn(settings):
    evidence = StreetClipGateway(settings).validate(full_evidence(settings))
    ja = _clue(dimension="language", value="ja", confidence=0.9, source="ocr",
               extra={"origin_source": "ocr", "reliable": True})
    scores = _engine(settings, evidence, [ja]).infer([ja])
    jp = next(score for score in scores if score.iso == "JP")
    cn = next(score for score in scores if score.iso == "CN")
    assert jp.fusion["clue_boost"] > 0
    assert cn.fusion.get("clue_boost", 0.0) == 0.0
    # Pure Han signage text without an explicit reliable language clue is not CN.
    han = _clue(dimension="signage", value="北京路 58 号", confidence=0.99, source="ocr",
                extra={"origin_source": "ocr", "reliable": True})
    assert _engine(settings, evidence, [han]).infer([han])[0].iso == "JP"


def test_point_click_clues_keep_ocr_provenance_and_exclusions_apply_after_fusion(settings):
    evidence = StreetClipGateway(settings).validate(full_evidence(settings))
    crop = crop_record(settings, {"x": 0.2, "y": 0.2, "size": 16}, "CN", "d" * 64)
    merged = StreetClipGateway(settings).with_crop(evidence, crop)
    point_clue = _clue(dimension="language", value="ja", confidence=0.9, source="point_click",
                       extra={"origin_source": "ocr", "reliable": True})
    scores = _engine(settings, StreetClipGateway(settings).validate(merged), [point_clue]).infer(
        [point_clue], exclusions=["CN"]
    )
    assert all(score.iso != "CN" for score in scores)
    assert next(score for score in scores if score.iso == "JP").fusion["clue_boost"] > 0
    assert scores[0].iso == "JP"


def test_clue_boost_is_capped(settings):
    evidence = StreetClipGateway(settings).validate(full_evidence(settings))
    clue = _clue(dimension="language", value="ja", confidence=1.0, source="user")
    jp = next(score for score in _engine(settings, evidence, [clue]).infer([clue]) if score.iso == "JP")
    assert jp.fusion["clue_boost"] == pytest.approx(min(CLUE_BOOST_CAP, 1.0 * 2.0 * CLUE_BOOST_SCALE))


# --------------------------------------------------------------------------- gateway


def test_gateway_validates_optional_bounded_crop_contract(settings):
    gateway = StreetClipGateway(settings)
    evidence = gateway.validate(full_evidence(settings))  # backwards compatible: no crops
    assert "crops" not in evidence

    crops = [
        crop_record(settings, {"x": index / 10, "y": 0.5, "size": 16}, "CN", f"{index:064x}")
        for index in range(MAX_CROPS)
    ]
    payload = {**full_evidence(settings), "crops": crops}
    validated = gateway.validate(payload)
    assert len(validated["crops"]) == MAX_CROPS
    assert validated["crops"][0]["region"] == {"x": 0.0, "y": 0.5, "size": 16}

    for mutate in (
        lambda data: data["crops"].append(data["crops"][0]),
        lambda data: data["crops"][0].update(region={"x": 2.0, "y": 0.5, "size": 16}),
        lambda data: data["crops"][0].update(region={"x": 0.5, "y": 0.5, "size": True}),
        lambda data: data["crops"][0]["rankings"].__setitem__(0, {"iso": "JP", "raw_cosine": float("nan")}),
        lambda data: data["crops"][0].update(image_sha256="zz"),
        lambda data: data.update(crops=data["crops"] * 2),
    ):
        corrupt = json.loads(json.dumps(payload, ensure_ascii=False))
        mutate(corrupt)
        with pytest.raises(ValueError):
            gateway.validate(corrupt)


def test_gateway_uses_configured_country_set_dynamically(settings, tmp_path, monkeypatch):
    custom = [{"iso": iso, "name": f"Name {iso}"} for iso in ("AA", "BB", "CC")]
    path = tmp_path / "custom_countries.json"
    path.write_text(json.dumps(custom), encoding="utf-8")
    monkeypatch.setattr(Settings, "countries_path", property(lambda self: path))

    def revision():
        return catalog_revision(custom)

    def payload(rows):
        rows = sorted(rows, key=lambda row: (-row["raw_cosine"], row["iso"]))
        margin = round(rows[0]["raw_cosine"] - rows[1]["raw_cosine"], 6)
        return {"model": MODEL_ID, "model_revision": MODEL_REVISION, "catalog_revision": revision(),
                "rankings": rows, "margin": margin, "uncertain": margin < 0.015}

    complete = payload([{"iso": "AA", "raw_cosine": 0.3}, {"iso": "BB", "raw_cosine": 0.2}, {"iso": "CC", "raw_cosine": 0.1}])
    assert len(StreetClipGateway(settings).validate(complete)["rankings"]) == 3
    with pytest.raises(ValueError):
        StreetClipGateway(settings).validate(payload([{"iso": "AA", "raw_cosine": 0.3}, {"iso": "BB", "raw_cosine": 0.2}]))
    with pytest.raises(ValueError):
        StreetClipGateway(settings).validate({**complete, "catalog_revision": catalog_revision(custom[:2])})


def test_with_crop_is_immutable_and_bounded(settings):
    gateway = StreetClipGateway(settings)
    evidence = gateway.validate(full_evidence(settings))
    before = json.dumps(evidence, sort_keys=True)
    crops = []
    for index in range(MAX_CROPS):
        crops.append(crop_record(settings, {"x": index / 20, "y": 0.4, "size": 16}, "CN", f"{index + 100:064x}"))
    current = evidence
    for crop in crops:
        updated = gateway.with_crop(current, crop)
        assert updated is not None
        current = updated
    assert json.dumps(evidence, sort_keys=True) == before
    assert len(current["crops"]) == MAX_CROPS
    extra = crop_record(settings, {"x": 0.9, "y": 0.9, "size": 16}, "CN", "9" * 64)
    assert gateway.with_crop(current, extra) is None


def test_engine_tolerates_unvalidated_legacy_evidence(settings):
    evidence = full_evidence(settings)
    engine = _engine(settings, evidence, [])
    scores = engine.infer([])
    assert scores and scores[0].iso == "JP"
    assert all(score.score_type == "cosine" for score in scores)
