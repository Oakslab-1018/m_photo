"""City-level API and optional reference-index contract tests.

All predictors here are torch-free stubs: they prove the FastAPI/service wiring,
the dynamic catalogs, the exact response contract and the bounded reference
fusion without loading real weights. Real hierarchical CLIP behavior is
smoke-tested separately with the pinned model.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from inference.streetclip.app import create_app
from inference.streetclip.config import Settings
from inference.streetclip.service import StreetClipService


def _image_bytes(size=(16, 16), color=(200, 100, 50), exif=None) -> bytes:
    image = Image.new("RGB", size, color=color)
    buffer = io.BytesIO()
    kwargs = {"format": "PNG"}
    if exif is not None:
        kwargs["format"] = "JPEG"
        kwargs["exif"] = exif
    image.save(buffer, **kwargs)
    return buffer.getvalue()


def _city(city_id, name, *, iso="JP", region="", lat=35.0, lng=139.0, population=1000) -> dict:
    return {
        "city_id": str(city_id),
        "iso": iso,
        "name": name,
        "region": region,
        "lat": lat,
        "lng": lng,
        "population": population,
    }


def _write_rows(tmp_path: Path, rows: list[dict], name: str = "cities.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StubPredictor:
    """Deterministic torch-free predictor with the shared-encoder surface."""

    def __init__(self, *, ready: bool = True, scores: dict[int, float] | None = None) -> None:
        self._ready = ready
        self._device = "cpu"
        self._scores = scores or {}
        self.image_calls = 0
        self.city_calls = 0
        self.active = 0
        self.max_active = 0
        self.delay = 0.0

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def device(self) -> str:
        return self._device

    @property
    def catalog_revision(self) -> str:
        return "c" * 64

    def load(self) -> None:
        self._ready = True

    def encode_image(self, image) -> list[float]:
        self.image_calls += 1
        return [1.0, 0.0, 0.0]

    def predict_from_vector(self, vector) -> dict:
        self._enter()
        try:
            from inference.streetclip.schema import build_prediction

            return build_prediction(
                catalog_revision="c" * 64,
                rankings=[("JP", 0.5), ("FR", 0.4)],
            ).as_dict()
        finally:
            self._exit()

    def predict(self, payload: bytes) -> dict:
        self._enter()
        try:
            from inference.streetclip.schema import build_prediction

            return build_prediction(
                catalog_revision="c" * 64,
                rankings=[("JP", 0.5), ("FR", 0.4)],
            ).as_dict()
        finally:
            self._exit()

    def city_text_scores(self, vector, country_iso, cities, country_name):
        self.city_calls += 1
        self._enter()
        try:
            if self.delay:
                time.sleep(self.delay)
            return [
                (city.city_id, self._scores.get(city.city_id, 0.5))
                for city in cities
            ]
        finally:
            self._exit()

    def _enter(self) -> None:
        self.active += 1
        self.max_active = max(self.max_active, self.active)

    def _exit(self) -> None:
        self.active -= 1


class FakeReferenceIndex:
    """Minimal stand-in for the separate pure ReferenceIndex library."""

    def __init__(self, matches: list[dict] | None = None) -> None:
        self.matches = matches or []
        self.calls = 0

    def search(self, vector, top_k=5) -> dict:
        self.calls += 1
        return {
            "index_revision": "i" * 64,
            "model_revision": "e3561ba2ad9bf14c9efd6b0092607b8497efbfea",
            "catalog_revision": "c" * 64,
            "aggregation": "max",
            "matches": self.matches,
            "regions": [],
            "coverage": sorted({match["iso"] for match in self.matches}),
        }


def _settings(tmp_path: Path, cities_path: Path | None, **overrides) -> Settings:
    options = {
        "device": "cpu",
        "cities_path": cities_path or (tmp_path / "missing-cities.json"),
    }
    options.update(overrides)
    return Settings(**options)


TOKYO = _city(1850147, "Tokyo", region="Tokyo", lat=35.6895, lng=139.6917, population=9733276)
OSAKA = _city(1853909, "Osaka", region="Osaka", lat=34.6937, lng=135.5023, population=2753862)


def test_health_reports_city_capability_and_load_count_without_paths(tmp_path):
    path = _write_rows(tmp_path, [TOKYO, OSAKA])
    with TestClient(create_app(_settings(tmp_path, path), predictor=StubPredictor())) as client:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["ready"] is True
        assert payload["city_capability"] is True
        assert payload["cities_loaded"] == 2
        assert payload["city_catalog_revision"] == _file_hash(path)
        assert payload["reference_configured"] is False
        assert payload["reference_loaded"] is False
        serialized = json.dumps(payload)
        assert "cities.json" not in serialized
        assert "reference-index" not in serialized


def test_predict_cities_returns_real_catalog_identity_and_text_scores(tmp_path):
    path = _write_rows(tmp_path, [TOKYO, OSAKA])
    stub = StubPredictor(scores={1850147: 0.9, 1853909: 0.4})
    with TestClient(create_app(_settings(tmp_path, path), predictor=stub)) as client:
        response = client.post(
            "/predict-cities",
            files={"file": ("sample.png", _image_bytes(), "image/png")},
            data={"country_iso": "JP"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"city_scores", "country_iso", "scope", "status"}
    assert payload["country_iso"] == "JP"
    assert payload["status"] == "ok"
    assert set(payload["scope"].keys()) == {
        "city_limit_per_country", "city_catalog_revision", "cities_considered",
    }
    assert payload["scope"]["city_limit_per_country"] == 100
    assert payload["scope"]["cities_considered"] == 2
    assert payload["scope"]["city_catalog_revision"] == _file_hash(path)

    top = payload["city_scores"][0]
    assert set(top.keys()) == {
        "city_id", "iso", "name", "region", "lat", "lng",
        "raw_cosine", "score", "score_type", "reference_cosine",
    }
    assert top["city_id"] == 1850147
    assert top["name"] == "Tokyo"
    assert top["region"] == "Tokyo"
    assert top["lat"] == TOKYO["lat"]
    assert top["lng"] == TOKYO["lng"]
    assert top["raw_cosine"] == 0.9
    assert top["score"] == 0.9
    assert top["score_type"] == "city_text_cosine"
    assert top["reference_cosine"] is None
    assert stub.image_calls == 1


def test_predict_cities_top10_tie_break_and_configured_limit(tmp_path):
    rows = [
        _city(1000 + index, f"City{index}", lat=30.0 + index, lng=130.0 + index, population=1000 - index)
        for index in range(12)
    ]
    path = _write_rows(tmp_path, rows)
    stub = StubPredictor(scores={1000 + index: 0.5 for index in range(12)})
    settings = _settings(tmp_path, path, city_limit_per_country=100)
    with TestClient(create_app(settings, predictor=stub)) as client:
        response = client.post(
            "/predict-cities",
            files={"file": ("sample.png", _image_bytes(), "image/png")},
            data={"country_iso": "JP"},
        )
    payload = response.json()
    assert payload["status"] == "ok"
    assert len(payload["city_scores"]) == 10
    assert payload["scope"]["cities_considered"] == 12
    assert [row["city_id"] for row in payload["city_scores"]] == [
        1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009,
    ]

    capped = _settings(tmp_path, path, city_limit_per_country=3)
    with TestClient(create_app(capped, predictor=StubPredictor())) as client:
        response = client.post(
            "/predict-cities",
            files={"file": ("sample.png", _image_bytes(), "image/png")},
            data={"country_iso": "JP"},
        )
    payload = response.json()
    assert payload["scope"]["city_limit_per_country"] == 3
    assert payload["scope"]["cities_considered"] == 3
    assert len(payload["city_scores"]) == 3


def test_city_coordinates_come_from_catalog_and_file_is_not_mutated(tmp_path):
    path = _write_rows(tmp_path, [OSAKA, TOKYO])
    before_bytes = path.read_bytes()
    stub = StubPredictor(scores={1850147: 0.3, 1853909: 0.7})
    with TestClient(create_app(_settings(tmp_path, path), predictor=stub)) as client:
        first = client.post(
            "/predict-cities",
            files={"file": ("a.png", _image_bytes(), "image/png")},
            data={"country_iso": "JP"},
        ).json()
        second = client.post(
            "/predict-cities",
            files={"file": ("b.png", _image_bytes(), "image/png")},
            data={"country_iso": "JP"},
        ).json()
    assert first == second
    assert first["city_scores"][0]["name"] == "Osaka"
    assert first["city_scores"][0]["lat"] == OSAKA["lat"]
    assert first["city_scores"][0]["lng"] == OSAKA["lng"]
    assert first["city_scores"][0]["raw_cosine"] == 0.7
    assert path.read_bytes() == before_bytes


def test_predict_cities_auth_rejected_before_image_parsing(tmp_path):
    path = _write_rows(tmp_path, [TOKYO])
    stub = StubPredictor()
    settings = _settings(tmp_path, path, api_key="s" * 64)
    with TestClient(create_app(settings, predictor=stub)) as client:
        response = client.post(
            "/predict-cities",
            files={"file": ("bad.png", b"not an image", "image/png")},
            data={"country_iso": "JP"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert response.status_code == 401
        assert stub.image_calls == 0

        ok = client.post(
            "/predict-cities",
            files={"file": ("sample.png", _image_bytes(), "image/png")},
            data={"country_iso": "JP"},
            headers={"Authorization": "Bearer " + "s" * 64},
        )
        assert ok.status_code == 200


def test_predict_cities_invalid_country_returns_clear_400_or_422(tmp_path):
    path = _write_rows(tmp_path, [TOKYO])
    with TestClient(create_app(_settings(tmp_path, path), predictor=StubPredictor())) as client:
        for country in ("ZZ", "J", "?"):
            response = client.post(
                "/predict-cities",
                files={"file": ("sample.png", _image_bytes(), "image/png")},
                data={"country_iso": country},
            )
            assert response.status_code == 400
            assert "country_iso" in response.json()["detail"]
        missing = client.post(
            "/predict-cities",
            files={"file": ("sample.png", _image_bytes(), "image/png")},
        )
        assert missing.status_code == 422


def test_predict_cities_empty_and_missing_catalog_report_no_cities(tmp_path):
    paris = _city(2988507, "Paris", iso="FR", region="Ile-de-France", lat=48.8566, lng=2.3522, population=2138551)
    empty_path = _write_rows(tmp_path, [paris], name="fr-only.json")
    with TestClient(create_app(_settings(tmp_path, empty_path), predictor=StubPredictor())) as client:
        response = client.post(
            "/predict-cities",
            files={"file": ("sample.png", _image_bytes(), "image/png")},
            data={"country_iso": "JP"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "no_cities"
        assert payload["city_scores"] == []
        assert payload["scope"]["cities_considered"] == 0
        assert payload["scope"]["city_catalog_revision"] is None

    missing_settings = _settings(tmp_path, None)
    with TestClient(create_app(missing_settings, predictor=StubPredictor())) as client:
        health = client.get("/health").json()
        assert health["city_capability"] is False
        assert health["cities_loaded"] == 0
        response = client.post(
            "/predict-cities",
            files={"file": ("sample.png", _image_bytes(), "image/png")},
            data={"country_iso": "JP"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "no_cities"


def test_city_image_validation_limits_and_exif_gps(tmp_path):
    path = _write_rows(tmp_path, [TOKYO])
    stub = StubPredictor()
    limited = _settings(tmp_path, path, max_upload_bytes=32)
    with TestClient(create_app(limited, predictor=stub)) as client:
        oversize = client.post(
            "/predict-cities",
            files={"file": ("big.png", b"x" * 64, "image/png")},
            data={"country_iso": "JP"},
        )
        assert oversize.status_code == 413

    exif = Image.Exif()
    exif[34853] = {1: "N", 2: (1.0, 2.0, 3.0)}
    with TestClient(create_app(_settings(tmp_path, path), predictor=stub)) as client:
        gps = client.post(
            "/predict-cities",
            files={"file": ("gps.jpg", _image_bytes(exif=exif), "image/jpeg")},
            data={"country_iso": "JP"},
        )
        assert gps.status_code == 422
    assert stub.image_calls == 0


def test_city_endpoint_serializes_inference_calls(tmp_path):
    path = _write_rows(tmp_path, [TOKYO, OSAKA])
    stub = StubPredictor()
    stub.delay = 0.05
    with TestClient(create_app(_settings(tmp_path, path), predictor=stub)) as client:
        def call() -> int:
            return client.post(
                "/predict-cities",
                files={"file": ("sample.png", _image_bytes(), "image/png")},
                data={"country_iso": "JP"},
            ).status_code

        with ThreadPoolExecutor(max_workers=2) as pool:
            statuses = list(pool.map(lambda _: call(), range(2)))
    assert statuses == [200, 200]
    assert stub.max_active == 1


def test_predict_without_reference_preserves_exact_six_keys(tmp_path):
    path = _write_rows(tmp_path, [TOKYO])
    with TestClient(create_app(_settings(tmp_path, path), predictor=StubPredictor())) as client:
        response = client.post(
            "/predict",
            files={"file": ("sample.png", _image_bytes(), "image/png")},
        )
        assert response.status_code == 200
        assert set(response.json().keys()) == {
            "model", "model_revision", "catalog_revision", "rankings", "margin", "uncertain",
        }


def test_reference_index_sidecar_and_city_hybrid_fusion(tmp_path):
    path = _write_rows(tmp_path, [TOKYO, OSAKA])
    reference = FakeReferenceIndex(
        matches=[
            {
                "reference_id": "osaka-street-1",
                "iso": "JP",
                "region": "Osaka",
                "city": "Osaka",
                "city_id": "1853909",
                "place": "street",
                "viewpoint": "north",
                "lat": 34.7,
                "lng": 135.5,
                "raw_cosine": 0.8,
            }
        ]
    )
    stub = StubPredictor(scores={1850147: 0.9, 1853909: 0.4})
    settings = _settings(tmp_path, path, reference_index_path=(tmp_path / "ref-index"))
    service = StreetClipService(settings, predictor=stub, reference_index=reference)
    with TestClient(create_app(settings, service=service)) as client:
        health = client.get("/health").json()
        assert health["reference_configured"] is True
        assert health["reference_loaded"] is True
        assert "ref-index" not in json.dumps(health)

        cities = client.post(
            "/predict-cities",
            files={"file": ("sample.png", _image_bytes(), "image/png")},
            data={"country_iso": "JP"},
        ).json()
        by_name = {row["name"]: row for row in cities["city_scores"]}
        assert by_name["Tokyo"]["score_type"] == "city_text_cosine"
        assert by_name["Tokyo"]["reference_cosine"] is None
        osaka = by_name["Osaka"]
        assert osaka["score_type"] == "city_reference_hybrid"
        assert osaka["reference_cosine"] == 0.8
        assert osaka["raw_cosine"] == 0.4
        assert osaka["score"] == 0.4 * 0.4 + 0.6 * 0.8
        assert cities["city_scores"][0]["name"] == "Tokyo"
        assert reference.calls >= 1

        country = client.post(
            "/predict",
            files={"file": ("sample.png", _image_bytes(), "image/png")},
        ).json()
    assert set(country.keys()) == {
        "model", "model_revision", "catalog_revision", "rankings", "margin", "uncertain", "reference",
    }
    assert country["reference"]["aggregation"] == "max"
    assert country["reference"]["matches"][0]["reference_id"] == "osaka-street-1"


def test_broken_configured_reference_index_keeps_country_service_operable(tmp_path):
    path = _write_rows(tmp_path, [TOKYO])
    settings = _settings(
        tmp_path,
        path,
        reference_index_path=(tmp_path / "does-not-exist"),
    )
    stub = StubPredictor()
    with TestClient(create_app(settings, predictor=stub)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        payload = health.json()
        assert payload["ready"] is True
        assert payload["reference_configured"] is True
        assert payload["reference_loaded"] is False
        assert payload["city_capability"] is True
        assert "does-not-exist" not in json.dumps(payload)

        response = client.post(
            "/predict",
            files={"file": ("sample.png", _image_bytes(), "image/png")},
        )
        assert response.status_code == 200
        assert set(response.json().keys()) == {
            "model", "model_revision", "catalog_revision", "rankings", "margin", "uncertain",
        }

        cities = client.post(
            "/predict-cities",
            files={"file": ("sample.png", _image_bytes(), "image/png")},
            data={"country_iso": "JP"},
        )
        assert cities.status_code == 200
        assert cities.json()["city_scores"][0]["score_type"] == "city_text_cosine"


