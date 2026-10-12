"""Tests for the authoritative GeoNames admin1 catalog and strict loader.

The tests exercise two distinct layers:

* small, explicit fixtures prove every validation path without network access;
* the committed ``data/admin1.json`` is checked against the real city catalog so
  coverage gaps are visible and cannot silently turn into fabricated admin1
  entries.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from backend.app.services.admin1_catalog import (
    Admin1CatalogError,
    admin1_from_mapping,
    load_admin1_index,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ADMIN1_PATH = REPO_ROOT / "data" / "admin1.json"
SOURCES_PATH = REPO_ROOT / "data" / "admin1-sources.json"
CITIES_PATH = REPO_ROOT / "data" / "cities.json"


def _row(
    admin1_id: str,
    *,
    iso: str | None = None,
    code: str | None = None,
    name: str = "Example",
    ascii_name: str = "Example",
    geoname_id: int = 1,
    **extra,
) -> dict:
    if iso is None:
        iso = admin1_id.split(".", 1)[0]
    if code is None:
        code = admin1_id.split(".", 1)[1]
    return {
        "admin1_id": admin1_id,
        "iso": iso,
        "code": code,
        "name": name,
        "ascii_name": ascii_name,
        "geoname_id": geoname_id,
        **extra,
    }


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_default_loader_returns_the_full_committed_directory():
    index = load_admin1_index()

    assert len(index.by_id) == 3865
    assert len(index.by_iso) == 228
    assert list(index.by_id) == sorted(index.by_id)
    assert all(
        [row.admin1_id for row in rows] == sorted(row.admin1_id for row in rows)
        for rows in index.by_iso.values()
    )

    tokyo = index.by_id["JP.40"]
    assert tokyo.iso == "JP"
    assert tokyo.code == "40"
    assert tokyo.name == "Tokyo"
    assert tokyo.ascii_name == "Tokyo"
    assert tokyo.geoname_id == 1850144
    assert tokyo.label == "Tokyo"
    assert tokyo.matches("Tokyo")
    assert tokyo.matches("JP.40")

    assert index.by_iso["JP"] == tuple(index.by_id[key] for key in sorted(
        key for key in index.by_id if key.startswith("JP.")
    ))
    assert index.revision == hashlib.sha256(ADMIN1_PATH.read_bytes()).hexdigest()


def test_leading_zero_codes_are_preserved_as_strings():
    index = load_admin1_index()

    for admin1_id, code in (("CA.01", "01"), ("CN.01", "01"), ("JP.09", "09")):
        row = index.by_id[admin1_id]
        assert row.code == code
        assert row.code != code.lstrip("0")
        assert row.admin1_id == f"{row.iso}.{row.code}"


def test_cn_provinces_have_verified_chinese_labels_and_aliases():
    index = load_admin1_index()
    expected = {
        "CN.01": ("Anhui", "安徽省", "安徽"),
        "CN.02": ("Zhejiang", "浙江省", "浙江"),
        "CN.03": ("Jiangxi", "江西省", "江西"),
        "CN.04": ("Jiangsu", "江苏省", "江苏"),
        "CN.05": ("Jilin", "吉林省", "吉林"),
        "CN.06": ("Qinghai", "青海省", "青海"),
        "CN.07": ("Fujian", "福建省", "福建"),
        "CN.08": ("Heilongjiang", "黑龙江省", "黑龙江"),
        "CN.09": ("Henan", "河南省", "河南"),
        "CN.10": ("Hebei", "河北省", "河北"),
        "CN.11": ("Hunan", "湖南省", "湖南"),
        "CN.12": ("Hubei", "湖北省", "湖北"),
        "CN.13": ("Xinjiang", "新疆维吾尔自治区", "新疆"),
        "CN.14": ("Tibet", "西藏自治区", "西藏"),
        "CN.15": ("Gansu", "甘肃省", "甘肃"),
        "CN.16": ("Guangxi", "广西壮族自治区", "广西"),
        "CN.18": ("Guizhou", "贵州省", "贵州"),
        "CN.19": ("Liaoning", "辽宁省", "辽宁"),
        "CN.20": ("Inner Mongolia", "内蒙古自治区", "内蒙古"),
        "CN.21": ("Ningxia", "宁夏回族自治区", "宁夏"),
        "CN.22": ("Beijing", "北京市", "北京"),
        "CN.23": ("Shanghai", "上海市", "上海"),
        "CN.24": ("Shanxi", "山西省", "山西"),
        "CN.25": ("Shandong", "山东省", "山东"),
        "CN.26": ("Shaanxi", "陕西省", "陕西"),
        "CN.28": ("Tianjin", "天津市", "天津"),
        "CN.29": ("Yunnan", "云南省", "云南"),
        "CN.30": ("Guangdong", "广东省", "广东"),
        "CN.31": ("Hainan", "海南省", "海南"),
        "CN.32": ("Sichuan", "四川省", "四川"),
        "CN.33": ("Chongqing", "重庆市", "重庆"),
    }

    cn_rows = [row for row in index.by_id.values() if row.iso == "CN"]
    assert {row.admin1_id for row in cn_rows} == set(expected)
    for admin1_id, (source_name, name_zh, alias) in expected.items():
        row = index.by_id[admin1_id]
        assert row.name == source_name
        assert row.name_zh == name_zh
        assert row.aliases == (alias,)
        assert row.label == name_zh
        assert row.matches(name_zh)
        assert row.matches(alias)


def test_city_crosswalk_explicitly_marks_every_unmatched_and_absent_code():
    index = load_admin1_index()
    metadata = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    crosswalk = metadata["city_catalog_crosswalk"]
    cities = json.loads(CITIES_PATH.read_text(encoding="utf-8"))

    pairs: Counter[tuple[str, str]] = Counter()
    absent: Counter[str] = Counter()
    for city in cities:
        region = city.get("region")
        if region is None or (isinstance(region, str) and not region.strip()):
            absent[city["iso"]] += 1
        else:
            pairs[(city["iso"], region)] += 1

    expected_unmatched = {
        f"{iso}.{code}": count
        for (iso, code), count in pairs.items()
        if f"{iso}.{code}" not in index.by_id
    }
    actual_unmatched = {
        item["admin1_id"]: item["city_rows"] for item in crosswalk["unmatched"]
    }
    assert actual_unmatched == expected_unmatched
    assert crosswalk["unmatched_admin1_codes"] == len(expected_unmatched)
    assert crosswalk["city_rows_with_admin1_code"] == sum(pairs.values())
    assert crosswalk["city_rows_without_admin1_code"] == sum(absent.values())
    assert {
        item["iso"]: item["city_rows"] for item in crosswalk["absent"]
    } == dict(absent)

    # The unmatched list documents gaps; it must never be materialised as a
    # fake admin1 row in the authoritative directory.
    assert not (set(expected_unmatched) & set(index.by_id))

    # CN is the key guided-dialogue scope: every city with a region resolves,
    # and the two region-less rows are explicitly recorded as absent.
    cn_city_pairs = {code for (iso, code) in pairs if iso == "CN"}
    assert cn_city_pairs
    assert all(f"CN.{code}" in index.by_id for code in cn_city_pairs)
    assert crosswalk["absent"][1] == {"iso": "CN", "code": None, "city_rows": 2}


def test_source_metadata_records_provenance_and_output_revision():
    index = load_admin1_index()
    metadata = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    source = metadata["sources"]["admin1"]

    assert source["download_url"] == (
        "https://download.geonames.org/export/dump/admin1CodesASCII.txt"
    )
    assert source["retrieved"] == "2026-10-12"
    assert source["license"] == "Creative Commons Attribution 4.0 (CC BY 4.0)"
    assert "GeoNames" in source["attribution"]
    assert source["sha256"] == (
        "1da92a6323a5fec3176f3f743bf4cf4040fd56a876da55e46fbca23c863aa60a"
    )
    assert source["records"] == len(index.by_id)
    assert metadata["source_sha256"] == source["sha256"]
    assert metadata["source_retrieved"] == "2026-10-12"
    assert metadata["outputs"]["data/admin1.json"]["sha256"] == index.revision
    assert metadata["cn_translations"]["translated"] == 31
    assert metadata["cn_translations"]["untranslated_ids"] == []


def test_fixture_loader_builds_sorted_indexes_and_optional_fields(tmp_path):
    path = tmp_path / "admin1.json"
    _write(path, [
        _row("JP.40", name="Tokyo", ascii_name="Tokyo", geoname_id=1850144),
        _row("JP.01", name="Aichi", ascii_name="Aichi", geoname_id=1865694),
        _row(
            "CN.22",
            name="Beijing",
            ascii_name="Beijing",
            geoname_id=2038349,
            name_zh="北京市",
            aliases=["北京", "Peking"],
        ),
    ])

    index = load_admin1_index(path)

    assert list(index.by_id) == ["CN.22", "JP.01", "JP.40"]
    assert [row.admin1_id for row in index.by_iso["JP"]] == ["JP.01", "JP.40"]
    beijing = index.by_id["CN.22"]
    assert beijing.aliases == ("北京", "Peking")
    assert beijing.label == "北京市"
    assert beijing.matches("北京")
    assert beijing.matches("Peking")
    assert beijing.matches("BEIJING")
    assert not beijing.matches("Shanghai")


def test_loader_accepts_utf8_bom_and_preserves_unicode_names(tmp_path):
    path = tmp_path / "bom.json"
    row = _row("JP.13", name="Hyōgo", ascii_name="Hyogo", geoname_id=1862047)
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps([row], ensure_ascii=False).encode("utf-8"))

    index = load_admin1_index(path)

    hyogo = index.by_id["JP.13"]
    assert hyogo.name == "Hyōgo"
    assert hyogo.ascii_name == "Hyogo"
    assert hyogo.matches("HYOGO")


def test_loader_accepts_a_wrapped_rows_payload(tmp_path):
    path = tmp_path / "admin1.json"
    _write(path, {"rows": [_row("JP.40", name="Tokyo", ascii_name="Tokyo")]})

    index = load_admin1_index(path)

    assert list(index.by_id) == ["JP.40"]


def test_loader_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "admin1.json"
    _write(path, [
        _row("JP.40", name="Tokyo", ascii_name="Tokyo"),
        _row("JP.40", name="Duplicate", ascii_name="Duplicate", geoname_id=2),
    ])

    with pytest.raises(Admin1CatalogError, match="duplicate admin1_id"):
        load_admin1_index(path)


def test_loader_rejects_inconsistent_id_and_code_fields(tmp_path):
    path = tmp_path / "admin1.json"
    _write(path, [_row("JP.40", iso="JP", code="4")])

    with pytest.raises(Admin1CatalogError, match="inconsistent"):
        load_admin1_index(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "   "),
        ("ascii_name", ""),
        ("code", ""),
        ("code", "4-0"),
        ("geoname_id", 0),
        ("geoname_id", "not-a-number"),
        ("aliases", "not-an-array"),
    ],
)
def test_loader_rejects_malformed_fields(tmp_path, field, value):
    path = tmp_path / "admin1.json"
    row = _row("JP.40", name="Tokyo", ascii_name="Tokyo")
    row[field] = value
    _write(path, [row])

    with pytest.raises(Admin1CatalogError):
        load_admin1_index(path)


def test_loader_rejects_malformed_json_invalid_utf8_and_missing_files(tmp_path):
    malformed = tmp_path / "malformed.json"
    malformed.write_bytes(b'[{"admin1_id": "JP.40"}')

    invalid_utf8 = tmp_path / "invalid.json"
    invalid_utf8.write_bytes(b'[{"admin1_id": "\xff"}]')

    with pytest.raises(Admin1CatalogError, match="not valid JSON"):
        load_admin1_index(malformed)
    with pytest.raises(Admin1CatalogError, match="not valid UTF-8"):
        load_admin1_index(invalid_utf8)
    with pytest.raises(Admin1CatalogError, match="unreadable"):
        load_admin1_index(tmp_path / "missing.json")


def test_loader_rejects_empty_catalog_payloads(tmp_path):
    path = tmp_path / "empty.json"
    _write(path, [])

    with pytest.raises(Admin1CatalogError, match="non-empty"):
        load_admin1_index(path)


def test_mapping_validation_rejects_invalid_admin1_id_shape():
    with pytest.raises(Admin1CatalogError, match="ISO.CODE"):
        admin1_from_mapping(_row("JP-40", iso="JP", code="40"), 0)
