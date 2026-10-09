"""Tests for request validation and ranking schema.

These tests must pass in a torch-free interpreter: the app and schema modules
are imported normally, while the real predictor is never loaded. Catalog
assertions derive the expected set from ``data/countries.json`` because the
geodata worker owns the dynamic global catalog.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import pytest
from PIL import Image

from inference.streetclip.catalog import Country, compute_catalog_revision
from inference.streetclip.schema import (
    DecodeError,
    EmptyImageError,
    ExifGpsPresentError,
    ImageTooLargeError,
    build_prediction,
    decode_image_bytes,
    validate_image_bytes,
)

COUNTRIES_PATH = Path("data/countries.json")


def _image_bytes(size=(36, 24), fmt="PNG", exif=None) -> bytes:
    image = Image.new("RGB", size, color=(12, 34, 56))
    buffer = io.BytesIO()
    kwargs = {"format": fmt}
    if exif is not None:
        kwargs["exif"] = exif
    image.save(buffer, **kwargs)
    return buffer.getvalue()


def test_catalog_revision_is_deterministic_for_dynamic_global_catalog():
    payload = json.loads(COUNTRIES_PATH.read_text(encoding="utf-8"))
    countries = [
        Country(iso=str(item["iso"]).strip().upper(), name=str(item["name"]).strip())
        for item in payload
    ]
    assert len(countries) == len(payload)
    assert len(countries) >= 20
    assert len({item.iso for item in countries}) == len(countries)
    first = compute_catalog_revision(countries)
    second = compute_catalog_revision(list(reversed(countries)))
    assert first == second
    assert len(first) == 64


def test_empty_upload_rejected():
    with pytest.raises(EmptyImageError):
        validate_image_bytes(b"")


def test_oversize_upload_rejected_before_decode():
    with pytest.raises(ImageTooLargeError):
        validate_image_bytes(b"x" * 17, max_bytes=16)


def test_random_bytes_rejected():
    with pytest.raises(DecodeError):
        validate_image_bytes(b"definitely not an image")


def test_decode_applies_exif_orientation_and_strips_metadata():
    image = Image.new("RGB", (40, 20), color=(1, 2, 3))
    exif = Image.Exif()
    exif[274] = 6  # orientation: rotate 90 degrees CW
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    decoded = decode_image_bytes(buffer.getvalue())
    assert decoded.size == (20, 40)
    assert decoded.mode == "RGB"
    assert not decoded.getexif()


def test_exif_gps_rejected():
    exif = Image.Exif()
    exif[34853] = {1: "N", 2: (1.0, 2.0, 3.0)}
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="JPEG", exif=exif)
    with pytest.raises(ExifGpsPresentError):
        decode_image_bytes(buffer.getvalue())


def test_prediction_sorted_descending_with_margin_and_uncertain():
    raw = [
        ("JP", 0.20),
        ("FR", 0.40),
        ("US", 0.10),
    ]
    prediction = build_prediction(catalog_revision="c" * 64, rankings=raw)
    assert [item.iso for item in prediction.rankings] == ["FR", "JP", "US"]
    assert prediction.rankings[0].raw_cosine == 0.40
    assert prediction.margin == pytest.approx(0.20)
    assert prediction.uncertain is False


def test_prediction_uncertain_when_margin_below_threshold():
    prediction = build_prediction(
        catalog_revision="c" * 64,
        rankings=[("JP", 0.30), ("FR", 0.29)],
    )
    assert prediction.margin == pytest.approx(0.01)
    assert prediction.uncertain is True


def test_prediction_rejects_duplicate_and_non_finite_cosine():
    with pytest.raises(ValueError):
        build_prediction(catalog_revision="c" * 64, rankings=[("JP", 0.1), ("JP", 0.2)])
    with pytest.raises(ValueError):
        build_prediction(catalog_revision="c" * 64, rankings=[("JP", math.nan)])


def test_prediction_as_dict_is_exact_contract_shape():
    prediction = build_prediction(
        catalog_revision="c" * 64,
        rankings=[("JP", 0.30), ("FR", 0.20)],
    )
    payload = prediction.as_dict()
    assert set(payload.keys()) == {
        "model", "model_revision", "catalog_revision", "rankings", "margin", "uncertain",
    }
    assert payload["model"] == "geolocal/StreetCLIP"
    assert payload["model_revision"] == "e3561ba2ad9bf14c9efd6b0092607b8497efbfea"
    assert isinstance(payload["margin"], float)
    assert isinstance(payload["uncertain"], bool)
    assert all(set(item.keys()) == {"iso", "raw_cosine"} for item in payload["rankings"])
