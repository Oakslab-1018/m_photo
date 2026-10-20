from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import make_eval_manifest
from scripts.eval_manifest import (
    ManifestError,
    load_env_file,
    load_manifest,
)


def _write_image(path: Path, payload: bytes = b"not-a-real-jpeg") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "sample-1",
        "image": "images/a.jpg",
        "country_iso": "JP",
        "lat": 35.68,
        "lng": 139.76,
    }
    row.update(overrides)
    return row


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_load_manifest_normalizes_optional_fields(tmp_path: Path) -> None:
    _write_image(tmp_path / "images" / "a.jpg")
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        [
            _row(
                city_id=1850147,
                city="Tokyo",
                city_source_iso="jp",
                admin1_code="40",
                tags=["urban", " signage "],
                license="CC BY-SA 4.0",
                source="https://example.org/photo",
                notes="Shibuya",
            )
        ],
    )

    entries = load_manifest(manifest, valid_isos={"JP", "KR"})
    assert len(entries) == 1
    entry = entries[0]
    assert entry.id == "sample-1"
    assert entry.image_path == (tmp_path / "images" / "a.jpg").resolve()
    assert entry.country_iso == "JP"
    assert entry.city_id == "1850147"
    assert entry.city_source_iso == "JP"
    assert entry.tags == ("urban", "signage")
    assert entry.lat == 35.68


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    _write_image(tmp_path / "images" / "a.jpg")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n" + json.dumps(_row()) + "\n\n", encoding="utf-8")
    assert len(load_manifest(manifest)) == 1


def test_duplicate_id_is_rejected(tmp_path: Path) -> None:
    _write_image(tmp_path / "images" / "a.jpg")
    _write_image(tmp_path / "images" / "b.jpg")
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        [_row(), _row(image="images/b.jpg")],
    )
    with pytest.raises(ManifestError, match="duplicate id"):
        load_manifest(manifest)


def test_absolute_image_path_is_rejected(tmp_path: Path) -> None:
    absolute = tmp_path / "images" / "a.jpg"
    _write_image(absolute)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [_row(image=str(absolute))])
    with pytest.raises(ManifestError, match="must be relative"):
        load_manifest(manifest)


def test_parent_escape_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.jpg"
    _write_image(outside)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [_row(image="../outside.jpg")])
    with pytest.raises(ManifestError, match="must not escape"):
        load_manifest(manifest)


def test_missing_image_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [_row(image="images/missing.jpg")])
    with pytest.raises(ManifestError, match="image not found"):
        load_manifest(manifest)


def test_unknown_country_is_rejected_when_catalog_is_supplied(tmp_path: Path) -> None:
    _write_image(tmp_path / "images" / "a.jpg")
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [_row(country_iso="ZZ")])
    with pytest.raises(ManifestError, match="not in the loaded catalog"):
        load_manifest(manifest, valid_isos={"JP"})


def test_non_finite_coordinate_is_rejected(tmp_path: Path) -> None:
    _write_image(tmp_path / "images" / "a.jpg")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"id":"sample-1","image":"images/a.jpg","country_iso":"JP",'
        '"lat":NaN,"lng":139.76}\n',
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="lat"):
        load_manifest(manifest)


def test_city_id_must_be_positive(tmp_path: Path) -> None:
    _write_image(tmp_path / "images" / "a.jpg")
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [_row(city_id=0)])
    with pytest.raises(ManifestError, match="city_id"):
        load_manifest(manifest)


def test_load_env_file_parses_quotes_and_export(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "STREETCLIP_ENABLED=true\n"
        'export STREETCLIP_URL="http://127.0.0.1:8001"\n'
        "DEEPSEEK_API_KEY='secret'\n"
        "NO_EQUALS_LINE\n"
        "\n",
        encoding="utf-8",
    )
    values = load_env_file(env)
    assert values == {
        "STREETCLIP_ENABLED": "true",
        "STREETCLIP_URL": "http://127.0.0.1:8001",
        "DEEPSEEK_API_KEY": "secret",
    }


def test_missing_env_file_returns_empty(tmp_path: Path) -> None:
    assert load_env_file(tmp_path / "missing.env") == {}


def test_fractional_city_id_is_rejected(tmp_path: Path) -> None:
    _write_image(tmp_path / "images" / "a.jpg")
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [_row(city_id=1.5)])
    with pytest.raises(ManifestError, match="city_id"):
        load_manifest(manifest)


def test_empty_valid_isos_rejects_every_country(tmp_path: Path) -> None:
    _write_image(tmp_path / "images" / "a.jpg")
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [_row()])
    with pytest.raises(ManifestError, match="not in the loaded catalog"):
        load_manifest(manifest, valid_isos=set())


def test_duplicate_image_path_is_rejected(tmp_path: Path) -> None:
    _write_image(tmp_path / "images" / "a.jpg")
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        [
            _row(),
            _row(id="sample-2", image="images/./a.jpg"),
        ],
    )
    with pytest.raises(ManifestError, match="duplicate image path"):
        load_manifest(manifest)


def test_generated_manifest_round_trips_with_default_image_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "photos"
    _write_image(root / "a.jpg")
    output = root / "manifest.jsonl"
    assert make_eval_manifest.main(
        ["--image-root", str(root), "--output", str(output)]
    ) == 0

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    rows[0].update(country_iso="JP", lat=35.68, lng=139.76)
    _write_manifest(output, rows)

    entries = load_manifest(output)
    assert entries[0].image_ref == "a.jpg"
    assert entries[0].image_path == (root / "a.jpg").resolve()


def test_generated_external_manifest_prints_image_root_override(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "photos"
    _write_image(root / "a.jpg")
    output = tmp_path / "metadata" / "manifest.jsonl"
    assert make_eval_manifest.main(
        ["--image-root", str(root), "--output", str(output)]
    ) == 0
    assert "--image-root" in capsys.readouterr().out

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    rows[0].update(country_iso="JP", lat=35.68, lng=139.76)
    _write_manifest(output, rows)
    assert load_manifest(output, image_root=root)[0].image_ref == "a.jpg"
