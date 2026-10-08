"""Injected-predictor tests for the FastAPI application.

The stub predictor proves the application cannot silently score random/fake
images: when no real predictor is injected, ``/predict`` returns 503, while an
injected stub returns the contract and reports ready/device through ``/health``.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from inference.streetclip.app import create_app
from inference.streetclip.config import Settings


def _png_bytes() -> bytes:
    image = Image.new("RGB", (16, 16), color=(200, 100, 50))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class StubPredictor:
    def __init__(self, *, ready: bool = True, device: str = "cpu") -> None:
        self._ready = ready
        self._device = device
        self.calls = 0

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        self._ready = True

    def predict(self, payload: bytes) -> dict[str, object]:
        from inference.streetclip.schema import build_prediction

        self.calls += 1
        rankings = [
            (iso, 0.5 - (index * 0.01))
            for index, iso in enumerate(
                ["JP", "FR", "US", "CN", "DE", "IT", "GB", "ES", "GR", "RU",
                 "CA", "BR", "AU", "IN", "TH", "EG", "MX", "NL", "SE", "KR"]
            )
        ]
        if not payload:
            raise ValueError("empty")
        return build_prediction(catalog_revision="r" * 64, rankings=rankings).as_dict()


def _settings(api_key: str | None = None) -> Settings:
    return Settings(api_key=api_key, device="cpu")


def test_health_reports_required_fields_for_injected_predictor():
    with TestClient(create_app(_settings(), predictor=StubPredictor())) as client:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["ready"] is True
        assert payload["model"] == "geolocal/StreetCLIP"
        assert payload["model_revision"] == "e3561ba2ad9bf14c9efd6b0092607b8497efbfea"
        assert payload["device"] == "cpu"
        assert payload["countries"] == 20


def test_predict_requires_file_field():
    with TestClient(create_app(_settings(), predictor=StubPredictor())) as client:
        response = client.post("/predict")
        assert response.status_code == 422


def test_predict_returns_exact_contract_with_20_sorted_rankings():
    with TestClient(create_app(_settings(), predictor=StubPredictor())) as client:
        response = client.post(
            "/predict",
            files={"file": ("sample.png", _png_bytes(), "image/png")},
        )
        assert response.status_code == 200
        payload = response.json()
        assert set(payload.keys()) == {
            "model", "model_revision", "catalog_revision", "rankings", "margin", "uncertain",
        }
        assert len(payload["rankings"]) == 20
        cosines = [item["raw_cosine"] for item in payload["rankings"]]
        assert cosines == sorted(cosines, reverse=True)
        assert payload["margin"] == pytest.approx(cosines[0] - cosines[1])
        assert payload["uncertain"] == (payload["margin"] < 0.015)
        assert all(set(item.keys()) == {"iso", "raw_cosine"} for item in payload["rankings"])


def test_predict_does_not_silently_score_without_model():
    app = create_app(_settings(), predictor=StubPredictor(ready=False))
    # ``service.predict`` must reject until an explicit predictor is ready;
    # StubPredictor without load() is a deliberately unavailable model.
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("sample.png", _png_bytes(), "image/png")},
        )
        assert response.status_code == 503


def test_auth_rejected_before_image_parsing():
    with TestClient(create_app(_settings(api_key="s" * 64), predictor=StubPredictor())) as client:
        response = client.post(
            "/predict",
            files={"file": ("bad.png", b"not an image", "image/png")},
            headers={"Authorization": "Bearer wrong"},
        )
        assert response.status_code == 401

        ok = client.post(
            "/predict",
            files={"file": ("sample.png", _png_bytes(), "image/png")},
            headers={"Authorization": "Bearer " + "s" * 64},
        )
        assert ok.status_code == 200


def test_invalid_image_body_returns_400_and_does_not_call_predictor():
    stub = StubPredictor()
    with TestClient(create_app(_settings(), predictor=stub)) as client:
        response = client.post(
            "/predict",
            files={"file": ("bad.png", b"definitely not an image", "image/png")},
        )
        assert response.status_code == 400
        assert stub.calls == 0
