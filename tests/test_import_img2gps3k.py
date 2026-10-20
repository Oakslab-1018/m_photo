from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image
from shapely.geometry import Polygon, mapping

from scripts import import_img2gps3k as importer
from scripts.eval_manifest import load_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_image(path: Path, *, gps: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (12, 8), (120, 80, 40))
    if not gps:
        image.save(path, format="JPEG")
        return
    exif = Image.Exif()
    gps_ifd = exif.get_ifd(34853)
    gps_ifd[1] = "N"
    gps_ifd[2] = (10.0, 20.0, 30.0)
    image.save(path, format="JPEG", exif=exif)


def _polygon(
    bounds: tuple[float, float, float, float],
    *,
    hole: tuple[float, float, float, float] | None = None,
) -> Polygon:
    min_x, min_y, max_x, max_y = bounds
    exterior = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
        (min_x, min_y),
    ]
    holes = None
    if hole is not None:
        h_min_x, h_min_y, h_max_x, h_max_y = hole
        holes = [[
            (h_min_x, h_min_y),
            (h_min_x, h_max_y),
            (h_max_x, h_max_y),
            (h_max_x, h_min_y),
            (h_min_x, h_min_y),
        ]]
    return Polygon(exterior, holes)


def _feature(geometry: Any, **properties: Any) -> dict[str, Any]:
    return {"type": "Feature", "geometry": mapping(geometry), "properties": properties}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "lat", "lon"])
        writer.writeheader()
        writer.writerows(rows)


def _write_catalog(path: Path, isos: Iterable[str]) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE geo_place (place_id TEXT, level TEXT)")
        connection.executemany(
            "INSERT INTO geo_place (place_id, level) VALUES (?, 'country')",
            [(iso,) for iso in isos],
        )


def _prepare(
    tmp_path: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
    images: Mapping[str, str] | None = None,
    catalog_isos: Iterable[str] = ("AA", "BB", "CC", "DD"),
) -> dict[str, Path]:
    image_root = tmp_path / "images"
    image_root.mkdir(parents=True)
    for name, kind in (images or {}).items():
        target = image_root / name
        if kind == "bad":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"not a decodable image")
        else:
            _write_image(target, gps=kind == "gps")

    csv_path = tmp_path / "im2gps3k.csv"
    _write_csv(csv_path, rows)
    boundary_path = tmp_path / "countries.geojson"
    boundary_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": list(features)}),
        encoding="utf-8",
    )
    catalog_path = tmp_path / "catalog.sqlite3"
    _write_catalog(catalog_path, catalog_isos)
    return {
        "image_root": image_root,
        "csv": csv_path,
        "boundaries": boundary_path,
        "catalog": catalog_path,
        "output": tmp_path / "manifest.jsonl",
        "audit": tmp_path / "reports",
    }


def _run(context: Mapping[str, Path], *extra: str) -> int:
    return importer.main(
        [
            "--csv",
            str(context["csv"]),
            "--image-root",
            str(context["image_root"]),
            "--boundaries",
            str(context["boundaries"]),
            "--output",
            str(context["output"]),
            "--audit-dir",
            str(context["audit"]),
            "--catalog-db",
            str(context["catalog"]),
            *extra,
        ]
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _audit(context: Mapping[str, Path]) -> dict[str, Any]:
    return json.loads((context["audit"] / "audit.json").read_text(encoding="utf-8"))


def test_mapping_hole_boundary_multi_match_and_explicit_code_resolution(tmp_path: Path) -> None:
    features = [
        _feature(
            _polygon((-10, -10, 10, 10), hole=(1, 1, 2, 2)),
            NAME="Alpha",
            ISO_A2_EH="AA",
            ISO_A2="ZZ",
            ADM0_A3="AAA",
        ),
        _feature(
            _polygon((10, -10, 20, 10)),
            NAME="Beta",
            ISO_A2_EH="-99",
            ISO_A2="BB",
            ADM0_A3="BBB",
        ),
        _feature(
            _polygon((4, 4, 6, 6)),
            NAME="Overlap",
            ISO_A2_EH="CC",
            ISO_A2="CC",
            ADM0_A3="CCC",
        ),
        _feature(
            _polygon((50, -10, 60, 10)),
            NAME="Somaliland-like",
            ISO_A2_EH="-99",
            ISO_A2="-99",
            ADM0_A3="SOL",
        ),
        _feature(
            _polygon((70, -10, 80, 10)),
            NAME="Needs fallback",
            ISO_A2_EH="YY",
            ISO_A2="AA",
            ADM0_A3="YYY",
        ),
    ]
    rows = [
        {"path": "hole.jpg", "lat": 1.5, "lon": 1.5},
        {"path": "edge.jpg", "lat": 0, "lon": -10},
        {"path": "primary.jpg", "lat": 0, "lon": -5},
        {"path": "fallback-invalid.jpg", "lat": 0, "lon": 15},
        {"path": "multi.jpg", "lat": 5, "lon": 5},
        {"path": "unknown.jpg", "lat": 0, "lon": 55},
        {"path": "fallback-not-catalog.jpg", "lat": 0, "lon": 75},
    ]
    context = _prepare(
        tmp_path,
        rows=rows,
        features=features,
        images={row["path"]: "ok" for row in rows},
    )

    assert _run(context) == 0

    manifest = _read_jsonl(context["output"])
    assert [row["country_iso"] for row in manifest] == ["AA", "AA", "BB", "AA"]
    assert [row["image"] for row in manifest] == [
        "images/edge.jpg",
        "images/primary.jpg",
        "images/fallback-invalid.jpg",
        "images/fallback-not-catalog.jpg",
    ]
    assert all(row["city_id"] is None for row in manifest)
    entries = load_manifest(context["output"], valid_isos={"AA", "BB", "CC", "DD"})
    assert [entry.country_iso for entry in entries] == ["AA", "AA", "BB", "AA"]

    audit = _audit(context)
    assert audit["inputs"]["boundaries"]["feature_count"] == 5
    assert audit["inputs"]["boundaries"]["sha256"] == _sha256(context["boundaries"])
    assert audit["inputs"]["csv"]["sha256"] == _sha256(context["csv"])
    assert audit["manifest"]["sha256"] == _sha256(context["output"])
    assert audit["summary"]["manifest_samples"] == 4
    assert audit["summary"]["country_labeled_samples"] == 4
    assert audit["summary"]["coordinate_only_samples"] == 0
    assert audit["summary"]["country_unresolved_samples"] == 3
    assert audit["summary"]["unresolved_samples"] == 3
    assert audit["summary"]["unresolved_reason_counts"] == {
        "multiple_boundary_matches": 1,
        "no_boundary_match": 1,
        "unknown_iso": 1,
    }

    by_path = {sample["path"]: sample for sample in audit["samples"]}
    assert by_path["hole.jpg"]["reasons"] == ["no_boundary_match"]
    assert by_path["multi.jpg"]["reasons"] == ["multiple_boundary_matches"]
    assert by_path["multi.jpg"]["boundary"]["match_count"] == 2
    assert {item["name"] for item in by_path["multi.jpg"]["boundary"]["candidates"]} == {
        "Alpha",
        "Overlap",
    }
    assert by_path["unknown.jpg"]["reasons"] == ["unknown_iso"]
    assert by_path["fallback-invalid.jpg"]["boundary"]["selected"]["iso_field"] == "ISO_A2"
    assert (
        by_path["fallback-not-catalog.jpg"]["boundary"]["selected"]["iso_resolution"]
        == "fallback_in_catalog"
    )
    assert by_path["fallback-not-catalog.jpg"]["boundary"]["selected"]["iso_attempts"][0] == {
        "field": "ISO_A2_EH",
        "value": "YY",
        "outcome": "not_in_catalog",
    }
    assert by_path["edge.jpg"]["boundary"]["match_count"] == 1


def test_paths_bad_images_gps_swap_duplicates_and_extra_images(tmp_path: Path) -> None:
    features = [
        _feature(
            _polygon((-10, -10, 10, 10)),
            NAME="Alpha",
            ISO_A2_EH="AA",
            ISO_A2="AA",
            ADM0_A3="AAA",
        )
    ]
    rows = [
        {"path": "../outside.jpg", "lat": 0, "lon": 0},
        {"path": "good.jpg", "lat": 0, "lon": 0},
        {"path": "good.jpg", "lat": 0, "lon": 0},
        {"path": "bad.jpg", "lat": 0, "lon": 0},
        {"path": "gps.jpg", "lat": 0, "lon": 0},
        {"path": "swap.jpg", "lat": 120, "lon": 40},
        {"path": "missing.jpg", "lat": 0, "lon": 0},
    ]
    context = _prepare(
        tmp_path,
        rows=rows,
        features=features,
        images={
            "good.jpg": "ok",
            "bad.jpg": "bad",
            "gps.jpg": "gps",
            "swap.jpg": "ok",
            "extra-1.jpg": "ok",
            "extra-2.jpg": "ok",
            "extra-3.jpg": "ok",
        },
    )
    _write_image(tmp_path / "outside.jpg")
    csv_before = _sha256(context["csv"])
    good_before = _sha256(context["image_root"] / "good.jpg")

    assert _run(context) == 0

    manifest = _read_jsonl(context["output"])
    assert len(manifest) == 1
    assert manifest[0]["image"] == "images/good.jpg"
    assert manifest[0]["country_iso"] == "AA"
    assert manifest[0]["city_id"] is None
    assert _sha256(context["csv"]) == csv_before
    assert _sha256(context["image_root"] / "good.jpg") == good_before

    audit = _audit(context)
    assert audit["summary"]["manifest_samples"] == 1
    assert audit["summary"]["country_labeled_samples"] == 1
    assert audit["summary"]["coordinate_only_samples"] == 0
    assert audit["summary"]["country_unresolved_samples"] == 1
    assert audit["summary"]["unresolved_samples"] == 6
    assert audit["summary"]["images"]["discovered"] == 7
    assert audit["summary"]["images"]["extra_count"] == 3
    assert {row["path"] for row in audit["summary"]["images"]["extra_images"]} == {
        "extra-1.jpg",
        "extra-2.jpg",
        "extra-3.jpg",
    }
    reasons = audit["summary"]["unresolved_reason_counts"]
    assert reasons["path_escapes_image_root"] == 1
    assert reasons["duplicate_csv_path"] == 1
    assert reasons["image_unreadable"] == 1
    assert reasons["image_exif_gps"] == 1
    assert reasons["coordinates_may_be_swapped"] == 1
    assert reasons["image_missing"] == 1

    by_path = {sample["path"]: sample for sample in audit["samples"]}
    assert by_path["bad.jpg"]["image"]["status"] == "unreadable"
    assert by_path["gps.jpg"]["image"]["status"] == "exif_gps_present"
    assert by_path["swap.jpg"]["coordinates"]["raw_lat"] == "120"
    assert by_path["swap.jpg"]["coordinates"]["raw_lon"] == "40"
    good_statuses = [
        sample["status"] for sample in audit["samples"] if sample["path"] == "good.jpg"
    ]
    assert good_statuses == ["resolved", "unresolved"]

    extra_rows = list(csv.DictReader((context["audit"] / "extra-images.csv").open(encoding="utf-8")))
    assert {row["path"] for row in extra_rows} == {"extra-1.jpg", "extra-2.jpg", "extra-3.jpg"}
    issue_rows = list(csv.DictReader((context["audit"] / "image-issues.csv").open(encoding="utf-8")))
    assert {row["path"] for row in issue_rows} == {
        "../outside.jpg",
        "bad.jpg",
        "gps.jpg",
        "missing.jpg",
    }


def test_retain_unlabeled_country_is_opt_in_and_excludes_bad_samples(tmp_path: Path) -> None:
    features = [
        _feature(
            _polygon((-10, -10, 10, 10)),
            NAME="Alpha",
            ISO_A2_EH="AA",
            ISO_A2="AA",
            ADM0_A3="AAA",
        )
    ]
    rows = [
        {"path": "labeled.jpg", "lat": 0, "lon": 0},
        {"path": "unlabeled.jpg", "lat": 20, "lon": 20},
        {"path": "bad.jpg", "lat": 0, "lon": 0},
        {"path": "swapped.jpg", "lat": 120, "lon": 40},
    ]
    context = _prepare(
        tmp_path,
        rows=rows,
        features=features,
        images={
            "labeled.jpg": "ok",
            "unlabeled.jpg": "ok",
            "bad.jpg": "bad",
            "swapped.jpg": "ok",
        },
    )

    assert _run(context) == 0
    assert len(_read_jsonl(context["output"])) == 1

    assert _run(context, "--retain-unlabeled-country", "--force") == 0
    manifest = _read_jsonl(context["output"])
    assert [(row["image"], row["country_iso"]) for row in manifest] == [
        ("images/labeled.jpg", "AA"),
        ("images/unlabeled.jpg", None),
    ]
    assert all(row["city_id"] is None for row in manifest)
    assert all(row["lat"] is not None and row["lng"] is not None for row in manifest)

    audit = _audit(context)
    assert audit["manifest"]["policy"] == "retain_unlabeled_country_coordinate_only"
    assert audit["summary"]["manifest_samples"] == 2
    assert audit["summary"]["country_labeled_samples"] == 1
    assert audit["summary"]["coordinate_only_samples"] == 1
    assert audit["summary"]["country_unresolved_samples"] == 2
    assert audit["summary"]["excluded_from_manifest_samples"] == 2

    by_path = {sample["path"]: sample for sample in audit["samples"]}
    assert by_path["unlabeled.jpg"]["status"] == "unresolved"
    assert by_path["unlabeled.jpg"]["manifest_included"] is True
    assert by_path["unlabeled.jpg"]["manifest_inclusion"] == "coordinate_only"
    assert by_path["bad.jpg"]["manifest_included"] is False
    assert by_path["bad.jpg"]["excluded_reasons"] == ["image_unreadable"]
    assert by_path["swapped.jpg"]["manifest_included"] is False
    assert by_path["swapped.jpg"]["excluded_reasons"] == ["coordinates_may_be_swapped"]


def test_manual_override_records_required_basis_and_never_creates_city_id(tmp_path: Path) -> None:
    features = [
        _feature(
            _polygon((-10, -10, 10, 10)),
            NAME="Alpha",
            ISO_A2_EH="AA",
            ISO_A2="AA",
            ADM0_A3="AAA",
        )
    ]
    rows = [{"path": "review.jpg", "lat": 20, "lon": 20}]
    context = _prepare(tmp_path, rows=rows, features=features, images={"review.jpg": "ok"})
    override_path = tmp_path / "overrides.json"
    override_path.write_text(
        json.dumps(
            {
                "overrides": [
                    {
                        "path": "review.jpg",
                        "country_iso": "AA",
                        "reason": "visible route shield and place name",
                        "source": "manual review 2026-10-20",
                        "reviewed_by": "reviewer",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _run(context, "--overrides", str(override_path)) == 0

    manifest = _read_jsonl(context["output"])
    assert len(manifest) == 1
    row = manifest[0]
    assert row["country_iso"] == "AA"
    assert row["lat"] == 20.0
    assert row["lng"] == 20.0
    assert row["city_id"] is None
    assert "manual_override" in row["source"]
    assert "override_reason=visible route shield and place name" in row["notes"]

    audit = _audit(context)
    assert audit["inputs"]["overrides"]["sha256"] == _sha256(override_path)
    sample = audit["samples"][0]
    assert sample["status"] == "resolved"
    assert sample["resolution"] == "manual_override"
    assert sample["boundary"]["match_count"] == 0
    assert sample["override"]["reason"] == "visible route shield and place name"
    assert sample["label_provenance"]["city_id"] == {"kind": "not_derived", "value": None}


def test_override_without_reason_is_rejected(tmp_path: Path) -> None:
    context = _prepare(
        tmp_path,
        rows=[{"path": "photo.jpg", "lat": 0, "lon": 0}],
        features=[_feature(_polygon((-10, -10, 10, 10)), ISO_A2_EH="AA")],
        images={"photo.jpg": "ok"},
    )
    override_path = tmp_path / "overrides.json"
    override_path.write_text(
        json.dumps({"overrides": [{"path": "photo.jpg", "country_iso": "AA"}]}),
        encoding="utf-8",
    )

    assert _run(context, "--overrides", str(override_path)) == 2
    assert not context["output"].exists()
    assert not (context["audit"] / "audit.json").exists()


def test_stable_ids_csv_order_force_and_boundary_sha_check(tmp_path: Path) -> None:
    features = [
        _feature(
            _polygon((-10, -10, 10, 10)),
            NAME="Alpha",
            ISO_A2_EH="AA",
            ISO_A2="AA",
            ADM0_A3="AAA",
        )
    ]
    rows = [
        {"path": "one.jpg", "lat": 0, "lon": 0},
        {"path": "two.jpg", "lat": 1, "lon": 1},
    ]
    context = _prepare(
        tmp_path,
        rows=rows,
        features=features,
        images={"one.jpg": "ok", "two.jpg": "ok"},
    )
    boundary_sha = _sha256(context["boundaries"])
    expected_ids = [
        "img2gps3k-" + hashlib.sha256(name.encode()).hexdigest()[:16]
        for name in ("one.jpg", "two.jpg")
    ]

    assert _run(context, "--boundary-sha256", boundary_sha) == 0
    first = _read_jsonl(context["output"])
    assert [row["id"] for row in first] == expected_ids
    assert all(re.fullmatch(r"[A-Za-z0-9._-]+", row["id"]) for row in first)

    assert _run(context, "--boundary-sha256", boundary_sha, "--force") == 0
    second = _read_jsonl(context["output"])
    assert [row["id"] for row in second] == expected_ids

    assert _run(context, "--boundary-sha256", "0" * 64, "--force") == 2
    assert [row["id"] for row in _read_jsonl(context["output"])] == expected_ids
