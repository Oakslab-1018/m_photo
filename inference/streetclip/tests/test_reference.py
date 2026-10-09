"""Focused synthetic tests for the labeled reference index and builder.

All vectors/images here are deterministic synthetic fixtures used only to
exercise the index math and validation. They are NOT a production corpus and
must never be treated as labeled reference data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from inference.streetclip.build_reference import build_from_manifest
from inference.streetclip.config import MODEL_ID, MODEL_REVISION
from inference.streetclip.reference import (
    ARTIFACT_NAMES,
    PREPROCESSING_VERSION,
    SCHEMA_VERSION,
    ReferenceIndex,
    ReferenceIndexError,
    build_index,
    load_manifest,
    normalize_vector,
)

ALLOWED = {"JP", "US", "FR"}


def _record(ref_id: str, *, iso="JP", region="kanto", city="", city_id=None, viewpoint="", lat=35.6, lng=139.7):
    item = {
        "reference_id": ref_id, "iso": iso, "region": region, "place": "street",
        "lat": lat, "lng": lng, "source": "synthetic-test-fixture", "license": "test-only",
        "image_path": f"images/{ref_id}.png",
    }
    if city:
        item["city"] = city
    if city_id:
        item["city_id"] = city_id
    if viewpoint:
        item["viewpoint"] = viewpoint
    return item


def _unit(vector) -> np.ndarray:
    return normalize_vector(vector)


def _axes(*indices, d=4) -> np.ndarray:
    vector = np.zeros(d, dtype=np.float32)
    for index in indices:
        vector[index] = 1.0
    return vector


def _write_manifest(tmp_path: Path, items: list[dict], name="manifest.json") -> Path:
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    for item in items:
        Image.new("RGB", (8, 8), color=(10, 20, 30)).save(images / (item["reference_id"] + ".png"))
    path = tmp_path / name
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "references": items}), encoding="utf-8")
    return path


class _StubEncoder:
    """Deterministic torch-free encoder keyed by image size only."""

    def __init__(self):
        self.calls = 0

    def encode_image(self, image: Image.Image) -> np.ndarray:
        self.calls += 1
        return _axes(self.calls % 3)


def test_manifest_requires_references_list_and_known_schema(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 999, "references": []}), encoding="utf-8")
    with pytest.raises(ReferenceIndexError):
        load_manifest(bad, allowed_isos=ALLOWED)


def test_manifest_rejects_missing_labels_license_source_and_duplicates(tmp_path):
    items = [_record("r1")]
    path = _write_manifest(tmp_path, items)
    for field in ("region", "place", "source", "license"):
        broken = json.loads(path.read_text(encoding="utf-8"))
        del broken["references"][0][field]
        broken_path = tmp_path / "broken.json"
        broken_path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ReferenceIndexError):
            load_manifest(broken_path, allowed_isos=ALLOWED)

    dup = _write_manifest(tmp_path, [_record("r1"), _record("r1")], name="dup.json")
    with pytest.raises(ReferenceIndexError):
        load_manifest(dup, allowed_isos=ALLOWED)


def test_manifest_coordinates_must_be_bounded_and_finite(tmp_path):
    for lat, lng in ((91.0, 0.0), (0.0, -181.0), (math.nan, 0.0), ("35", 0.0)):
        item = _record("r1")
        item["lat"], item["lng"] = lat, lng
        path = _write_manifest(tmp_path, [item], name="coord.json")
        with pytest.raises(ReferenceIndexError):
            load_manifest(path, allowed_isos=ALLOWED)


def test_manifest_iso_validated_against_dynamic_catalog_membership(tmp_path):
    path = _write_manifest(tmp_path, [_record("r1", iso="ZZ")], name="iso.json")
    with pytest.raises(ReferenceIndexError):
        load_manifest(path, allowed_isos=ALLOWED)
    records, _ = load_manifest(path, allowed_isos=ALLOWED | {"ZZ"})
    assert records[0].iso == "ZZ"


def test_manifest_rejects_traversal_and_symlink_escape(tmp_path):
    item = _record("r1")
    item["image_path"] = "../outside.png"
    path = _write_manifest(tmp_path, [item], name="traversal.json")
    with pytest.raises(ReferenceIndexError):
        load_manifest(path, allowed_isos=ALLOWED)

    outside = tmp_path.parent / "outside-target.png"
    Image.new("RGB", (4, 4)).save(outside)
    link = tmp_path / "images" / "r2.png"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation not permitted on this platform")
    item = _record("r2")
    path = _write_manifest(tmp_path, [item], name="symlink.json")
    with pytest.raises(ReferenceIndexError):
        load_manifest(path, allowed_isos=ALLOWED)
    outside.unlink(missing_ok=True)


def test_normalize_rejects_non_finite_and_zero_vectors():
    for value in ([0.0, 0.0], [math.nan, 1.0], [math.inf, 1.0], []):
        with pytest.raises(ReferenceIndexError):
            normalize_vector(value)


def test_search_returns_nearest_neighbor_and_exact_sidecar_shape():
    records = (
        _record_from("jp-tokyo", iso="JP", region="kanto", city="tokyo", city_id="1392685764"),
        _record_from("us-nyc", iso="US", region="new-york", city="new york"),
    )
    index = build_index(records, [_unit(_axes(0)), _unit(_axes(1))], catalog_revision="c" * 64)
    sidecar = index.search(_unit(_axes(0)), top_k=5)
    assert set(sidecar) == {
        "index_revision", "model_revision", "catalog_revision", "aggregation",
        "matches", "regions", "coverage",
    }
    assert sidecar["matches"][0]["reference_id"] == "jp-tokyo"
    assert sidecar["matches"][0]["iso"] == "JP"
    assert sidecar["matches"][0]["city"] == "tokyo"
    assert sidecar["matches"][0]["city_id"] == "1392685764"
    assert sidecar["matches"][0]["raw_cosine"] == pytest.approx(1.0)
    assert sidecar["aggregation"] == "max"
    assert sidecar["coverage"] == ["JP", "US"]
    assert sidecar["index_revision"] == index.index_revision
    assert sidecar["model_revision"] == MODEL_REVISION


def _record_from(ref_id, **kwargs):
    from inference.streetclip.reference import ReferenceRecord

    return ReferenceRecord.from_dict(_record(ref_id, **kwargs), index=0)


def test_single_country_string_is_not_a_city_and_city_groups_stay_separate():
    records = (
        _record_from("jp-a", iso="JP", region="kanto", city="tokyo"),
        _record_from("jp-b", iso="JP", region="kanto", city="osaka"),
        _record_from("jp-c", iso="JP", region="kanto"),  # country-level, no city label
    )
    index = build_index(records, [_unit(_axes(0)), _unit(_axes(1)), _unit(_axes(2))], catalog_revision="c" * 64)
    sidecar = index.search(_unit(_axes(0)))
    # ties at 0.0 are broken deterministically by city string ("" sorts first)
    assert [row["city"] for row in sidecar["regions"]] == ["tokyo", "", "osaka"]
    tokyo = index.city_scores(_unit(_axes(0)), "JP")[0]
    assert tokyo["city"] == "tokyo"
    assert tokyo["support"] == 1 and tokyo["prototype_count"] == 1
    assert all(row["iso"] == "JP" for row in index.city_scores(_unit(_axes(0)), "JP"))


def test_multiple_prototypes_are_deterministic_and_beats_single_mean():
    vectors = [_unit(_axes(0)), _unit(_axes(1)), _unit(_axes(2)), _unit(np.ones(4))]
    records = tuple(_record_from(f"jp-{i}", iso="JP", region="kanto", city="tokyo") for i in range(4))
    first = build_index(records, vectors, catalog_revision="c" * 64, max_prototypes=3)
    second = build_index(
        tuple(reversed(records)), list(reversed(vectors)), catalog_revision="c" * 64, max_prototypes=3
    )
    assert first.prototypes.tobytes() == second.prototypes.tobytes()
    assert first.index_revision == second.index_revision
    assert first.groups[0].prototype_count == 3
    sidecar = first.search(_unit(_axes(0)))
    assert sidecar["regions"][0]["prototype_count"] == 3
    assert sidecar["regions"][0]["support"] == 4
    single = build_index(records, vectors, catalog_revision="c" * 64, max_prototypes=1)
    assert single.search(_unit(_axes(0)))["regions"][0]["prototype_count"] == 1
    assert sidecar["regions"][0]["raw_cosine"] >= single.search(_unit(_axes(0)))["regions"][0]["raw_cosine"]


def test_topk_and_response_bounds_are_enforced():
    records = tuple(_record_from(f"jp-{i:02d}", iso="JP", region="kanto", city=f"c{i}") for i in range(25))
    vectors = [_unit(np.eye(25, dtype=np.float32)[i]) for i in range(25)]
    index = build_index(records, vectors, catalog_revision="c" * 64)
    sidecar = index.search(_unit(np.eye(25, dtype=np.float32)[0]), top_k=100)
    assert len(sidecar["matches"]) == 10
    assert len(sidecar["regions"]) == 20
    with pytest.raises(ReferenceIndexError):
        index.search(_unit(np.eye(25, dtype=np.float32)[0]), aggregation="mean")


def test_search_rejects_query_dimension_mismatch():
    index = build_index((_record_from("jp-tokyo"),), [_unit(_axes(0))], catalog_revision="c" * 64)
    with pytest.raises(ReferenceIndexError):
        index.search(_unit(_axes(0, d=5)))


def test_save_load_round_trip_and_no_overwrite_without_force(tmp_path):
    records = (
        _record_from("jp-a", iso="JP", region="kanto", city="tokyo"),
        _record_from("jp-b", iso="JP", region="kanto", city="tokyo", viewpoint="north"),
    )
    index = build_index(records, [_unit(_axes(0)), _unit(_axes(1))], catalog_revision="c" * 64)
    target = tmp_path / "index"
    index.save(target)
    assert sorted(path.name for path in target.iterdir()) == sorted(ARTIFACT_NAMES)
    with pytest.raises(ReferenceIndexError):
        index.save(target)
    index.save(target, force=True)  # safe overwrite of a validated artifact-only dir

    loaded = ReferenceIndex.load(target, MODEL_REVISION, "c" * 64, allowed_isos=ALLOWED, expected_dimension=4)
    assert loaded.index_revision == index.index_revision
    assert loaded.vectors.tobytes() == index.vectors.tobytes()
    assert loaded.prototypes.tobytes() == index.prototypes.tobytes()
    assert loaded.search(_unit(_axes(0)), top_k=1)["matches"][0]["reference_id"] == "jp-a"


def test_save_refuses_unexpected_directory_contents_even_with_force(tmp_path):
    records = (_record_from("jp-a"),)
    index = build_index(records, [_unit(_axes(0))], catalog_revision="c" * 64)
    target = tmp_path / "index"
    target.mkdir()
    (target / "user-data.txt").write_text("do not delete", encoding="utf-8")
    with pytest.raises(ReferenceIndexError):
        index.save(target, force=True)
    assert (target / "user-data.txt").exists()


def test_load_rejects_revision_dimension_preprocessing_and_catalog_mismatch(tmp_path):
    index = build_index((_record_from("jp-a"),), [_unit(_axes(0))], catalog_revision="c" * 64)
    target = tmp_path / "index"
    index.save(target)
    with pytest.raises(ReferenceIndexError):
        ReferenceIndex.load(target, "0" * 40, "c" * 64)
    with pytest.raises(ReferenceIndexError):
        ReferenceIndex.load(target, MODEL_REVISION, "d" * 64)
    with pytest.raises(ReferenceIndexError):
        ReferenceIndex.load(target, MODEL_REVISION, "c" * 64, expected_dimension=8)
    with pytest.raises(ReferenceIndexError):
        ReferenceIndex.load(target, MODEL_REVISION, "c" * 64, expected_preprocessing="other")
    metadata = json.loads((target / ARTIFACT_NAMES[0]).read_text(encoding="utf-8"))
    assert metadata["preprocessing"] == PREPROCESSING_VERSION
    assert metadata["model"] == MODEL_ID


def test_load_detects_tampered_metadata_and_arrays(tmp_path):
    records = (_record_from("jp-a"), _record_from("jp-b"))
    index = build_index(records, [_unit(_axes(0)), _unit(_axes(1))], catalog_revision="c" * 64)
    target = tmp_path / "index"
    index.save(target)

    metadata_path = target / ARTIFACT_NAMES[0]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["index_revision"] = "f" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ReferenceIndexError):
        ReferenceIndex.load(target, MODEL_REVISION, "c" * 64)
    index.save(target, force=True)

    with np.load(target / ARTIFACT_NAMES[1], allow_pickle=False) as data:
        protos, vectors = data["prototypes"], data["vectors"]
    flipped = vectors.copy()
    flipped[0] = -flipped[0]  # still unit norm, different hash
    np.savez(target / ARTIFACT_NAMES[1], vectors=flipped, prototypes=protos)
    with pytest.raises(ReferenceIndexError):
        ReferenceIndex.load(target, MODEL_REVISION, "c" * 64)


def test_load_rejects_object_arrays_without_pickle(tmp_path):
    index = build_index((_record_from("jp-a"),), [_unit(_axes(0))], catalog_revision="c" * 64)
    target = tmp_path / "index"
    index.save(target)
    np.savez(
        target / ARTIFACT_NAMES[1],
        vectors=np.array([object()], dtype=object),
        prototypes=index.prototypes,
    )
    with pytest.raises(ReferenceIndexError):
        ReferenceIndex.load(target, MODEL_REVISION, "c" * 64)


def test_builder_encodes_manifest_images_and_writes_artifacts(tmp_path):
    items = [
        _record("jp-tokyo-a", iso="JP", region="kanto", city="tokyo", city_id="1392685764"),
        _record("jp-tokyo-b", iso="JP", region="kanto", city="tokyo"),
        _record("us-nyc", iso="US", region="new-york", city="new york"),
    ]
    path = _write_manifest(tmp_path, list(reversed(items)))
    encoder = _StubEncoder()
    index = build_from_manifest(
        path, tmp_path / "out", predictor=encoder,
        allowed_isos=ALLOWED, catalog_revision="c" * 64,
    )
    assert encoder.calls == 3
    assert index.reference_count == 3
    assert index.coverage == ("JP", "US")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(ARTIFACT_NAMES)
    loaded = ReferenceIndex.load(tmp_path / "out", MODEL_REVISION, "c" * 64)
    assert loaded.reference_count == 3


def test_builder_rejects_empty_manifest_and_encoder_dimension_drift(tmp_path):
    empty = _write_manifest(tmp_path, [], name="empty.json")
    with pytest.raises(ReferenceIndexError):
        build_from_manifest(empty, tmp_path / "out", predictor=_StubEncoder(), allowed_isos=ALLOWED, catalog_revision="c" * 64)

    items = [_record("jp-a"), _record("jp-b")]
    path = _write_manifest(tmp_path, items, name="two.json")

    class Flaky:
        def __init__(self):
            self.calls = 0

        def encode_image(self, image):
            self.calls += 1
            return _axes(0, d=4 if self.calls == 1 else 5)

    with pytest.raises(ReferenceIndexError):
        build_from_manifest(path, tmp_path / "out2", predictor=Flaky(), allowed_isos=ALLOWED, catalog_revision="c" * 64)


def test_repository_example_manifest_is_empty_and_valid():
    example = Path(__file__).resolve().parents[3] / "data" / "reference-manifest.example.json"
    records, paths = load_manifest(example, allowed_isos=ALLOWED)
    assert records == ()
    assert paths == ()