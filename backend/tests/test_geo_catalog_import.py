"""DB geographic catalog: import, fingerprint gate, store and CLI acceptance.

Every test runs against a temporary SQLite database; the workspace database is
never touched. Real-catalog tests assert the committed files' exact counts and
raw SHA-256 digests, the fixed ``admin1-largest-city-v1`` derivation policy and
the 2764/1101 coordinate coverage. Small fixtures prove re-derivation on a
changed city file, transactional rollback, leading-zero codes, CN grouping and
source scoping.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, select
from sqlalchemy.exc import IntegrityError

from backend.app.database import init_db
from backend.app.main import create_app
from backend.app.domain.geolocation.country_policy import POLICY_VERSION
from backend.app.services.admin1_catalog import load_admin1_index
from backend.app.services.city_catalog import catalog_digest, load_city_index
from backend.app.services.geo_catalog_import import (
    IMPORT_VERSION,
    GeoCatalogImportError,
    catalog_import_reasons,
    import_geo_catalog,
)
from backend.app.services.geo_catalog_models import GeoCatalogSource, GeoPlace
from backend.app.services.geo_catalog_store import (
    QUERY_DEFAULT_LIMIT,
    QUERY_MAX_LIMIT,
    GeoCatalogStore,
)
from backend.tests.support import make_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
COUNTRIES_PATH = DATA_DIR / "countries.json"
ADMIN1_PATH = DATA_DIR / "admin1.json"
CITIES_PATH = DATA_DIR / "cities.json"

COUNTRIES_FIXTURE = [
    {"iso": "JP", "iso3": "JPN", "name": "Japan", "name_zh": "日本", "lat": 36.0, "lng": 138.0},
    {"iso": "TW", "iso3": "TWN", "name": "Taiwan", "name_zh": "台湾", "lat": 23.7, "lng": 121.0},
]

ADMIN1_FIXTURE = [
    {
        "admin1_id": "JP.01",
        "iso": "JP",
        "code": "01",
        "name": "Aichi",
        "ascii_name": "Aichi",
        "geoname_id": 1865694,
        "name_zh": "爱知县",
        "aliases": ["愛知県"],
    },
    {
        "admin1_id": "JP.02",
        "iso": "JP",
        "code": "02",
        "name": "Empty",
        "ascii_name": "Empty",
        "geoname_id": 2,
    },
    {
        "admin1_id": "TW.01",
        "iso": "TW",
        "code": "01",
        "name": "Taipei",
        "ascii_name": "Taipei",
        "geoname_id": 3,
    },
]

CITIES_FIXTURE_V1 = [
    {"city_id": "1001", "iso": "JP", "name": "Nagoya", "ascii_name": "Nagoya", "region": "01", "lat": 35.18, "lng": 136.9, "population": 100000},
    {"city_id": "1002", "iso": "JP", "name": "Small", "ascii_name": "Small", "region": "01", "lat": 35.0, "lng": 137.0, "population": 500},
    {"city_id": "1003", "iso": "JP", "name": "Nowhere", "ascii_name": "Nowhere", "region": "99", "lat": 40.0, "lng": 140.0, "population": 999999},
    {"city_id": "1004", "iso": "TW", "name": "Taipei City", "ascii_name": "Taipei City", "region": "01", "lat": 25.03, "lng": 121.56, "population": 2000000},
]

CITIES_FIXTURE_V2 = [
    {"city_id": "1001", "iso": "JP", "name": "Nagoya", "ascii_name": "Nagoya", "region": "01", "lat": 35.18, "lng": 136.9, "population": 400},
    {"city_id": "1002", "iso": "JP", "name": "Small", "ascii_name": "Small", "region": "01", "lat": 35.0, "lng": 137.0, "population": 2000000},
    {"city_id": "1003", "iso": "JP", "name": "Nowhere", "ascii_name": "Nowhere", "region": "99", "lat": 40.0, "lng": 140.0, "population": 999999},
    {"city_id": "1004", "iso": "TW", "name": "Taipei City", "ascii_name": "Taipei City", "region": "01", "lat": 25.03, "lng": 121.56, "population": 2000000},
]


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _fixture_settings(tmp_path: Path, *, cities=None):
    root = tmp_path / "catalog"
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "countries": root / "countries.json",
        "admin1": root / "admin1.json",
        "cities": root / "cities.json",
    }
    _write_json(paths["countries"], COUNTRIES_FIXTURE)
    _write_json(paths["admin1"], ADMIN1_FIXTURE)
    _write_json(paths["cities"], CITIES_FIXTURE_V1 if cities is None else cities)
    settings = replace(
        make_settings(tmp_path / "db-root"),
        geo_catalog_mode="db",
        countries_path_override=paths["countries"],
        admin1_path_override=paths["admin1"],
        cities_path_override=paths["cities"],
    )
    return settings, paths


def _query_all(store: GeoCatalogStore, level: str, **kwargs):
    rows = []
    offset = 0
    total = None
    while True:
        page = store.query_places(level, limit=QUERY_MAX_LIMIT, offset=offset, **kwargs)
        total = page["total"]
        rows.extend(page["rows"])
        offset += QUERY_MAX_LIMIT
        if not page["rows"] or len(rows) >= total:
            break
    return total, rows


def _expected_admin1_winners() -> dict[str, tuple[float, float, int]]:
    """Independent re-derivation of the admin1-largest-city-v1 policy."""

    admin1_ids = {
        row["admin1_id"]
        for row in json.loads(ADMIN1_PATH.read_text(encoding="utf-8"))
    }
    winners: dict[str, tuple[int, int, float, float]] = {}
    for city in json.loads(CITIES_PATH.read_text(encoding="utf-8")):
        region = city.get("region")
        if not isinstance(region, str) or not region.strip():
            continue
        key = f"{city['iso']}.{region}"
        if key not in admin1_ids:
            continue
        candidate = (int(city["population"]), -int(city["city_id"]))
        current = winners.get(key)
        if current is None or candidate > current[:2]:
            winners[key] = (candidate[0], candidate[1], float(city["lat"]), float(city["lng"]))
    return {
        key: (lat, lng, -negated_city_id)
        for key, (_population, negated_city_id, lat, lng) in winners.items()
    }


@pytest.fixture(scope="module")
def real_catalog(tmp_path_factory):
    root = tmp_path_factory.mktemp("geo-real")
    settings = replace(make_settings(root), geo_catalog_mode="db")
    engine, session_factory = init_db(settings)
    result = import_geo_catalog(engine, settings)
    namespace = SimpleNamespace(
        settings=settings, engine=engine, session_factory=session_factory, result=result
    )
    yield namespace
    engine.dispose()


@pytest.fixture()
def real_store(real_catalog):
    with real_catalog.session_factory() as session:
        yield GeoCatalogStore(session)


@pytest.fixture()
def fixture_catalog(tmp_path):
    settings, paths = _fixture_settings(tmp_path)
    engine, session_factory = init_db(settings)
    result = import_geo_catalog(engine, settings)
    namespace = SimpleNamespace(
        settings=settings,
        paths=paths,
        engine=engine,
        session_factory=session_factory,
        result=result,
    )
    yield namespace
    engine.dispose()


@pytest.fixture()
def fixture_store(fixture_catalog):
    with fixture_catalog.session_factory() as session:
        yield GeoCatalogStore(session)


# --------------------------------------------------------------------------
# real committed catalogs
# --------------------------------------------------------------------------


def test_real_counts_and_fingerprints_match_pinned_metadata(real_catalog, real_store):
    assert real_catalog.result.imported is True
    assert real_catalog.result.counts == {"countries": 250, "admin1": 3865, "cities": 34156}
    assert real_catalog.result.admin1_with_coordinates == 2764
    assert real_catalog.result.admin1_missing_coordinates == 1101

    revisions = real_store.revisions()
    assert revisions["import_version"] == IMPORT_VERSION
    geo_meta = json.loads((DATA_DIR / "geo-catalog-sources.json").read_text(encoding="utf-8"))
    admin1_meta = json.loads((DATA_DIR / "admin1-sources.json").read_text(encoding="utf-8"))
    assert revisions["countries"] == geo_meta["outputs"]["data/countries.json"]["sha256"]
    assert revisions["cities"] == geo_meta["outputs"]["data/cities.json"]["sha256"]
    assert revisions["admin1"] == admin1_meta["outputs"]["data/admin1.json"]["sha256"]
    assert revisions["countries"] == hashlib.sha256(COUNTRIES_PATH.read_bytes()).hexdigest()
    assert revisions["admin1"] == hashlib.sha256(ADMIN1_PATH.read_bytes()).hexdigest()
    assert revisions["cities"] == catalog_digest(CITIES_PATH)
    assert geo_meta["outputs"]["data/countries.json"]["records"] == 250
    assert geo_meta["outputs"]["data/cities.json"]["records"] == 34156
    assert admin1_meta["outputs"]["data/admin1.json"]["records"] == 3865


def test_real_coordinate_coverage_follows_the_fixed_policy(real_catalog, real_store):
    countries = real_store.query_places("country", limit=QUERY_MAX_LIMIT, coordinates_only=False)
    assert (countries["total"], countries["missing_coordinates"]) == (250, 0)
    assert all(row.lat is not None and row.lng is not None for row in countries["rows"])

    cities = real_store.query_places("city", limit=QUERY_MAX_LIMIT, coordinates_only=False)
    assert (cities["total"], cities["missing_coordinates"]) == (34156, 0)
    assert all(row.lat is not None and row.lng is not None for row in cities["rows"])

    admin1 = real_store.query_places("admin1", limit=QUERY_MAX_LIMIT, coordinates_only=False)
    assert (admin1["total"], admin1["missing_coordinates"]) == (3865, 1101)

    # Every stored admin1 point is exactly the highest-population matching city
    # (tie break: smallest numeric city id) - never an average/placeholder.
    expected = _expected_admin1_winners()
    assert len(expected) == 2764
    with real_catalog.session_factory() as session:
        places = session.execute(
            select(GeoPlace).where(GeoPlace.level == "admin1")
        ).scalars().all()
    covered = {
        place.place_id: (place.lat, place.lng, place.coordinate_city_id)
        for place in places
        if place.lat is not None
    }
    assert set(covered) == set(expected)
    for place_id, (lat, lng, city_id) in expected.items():
        assert covered[place_id] == (lat, lng, city_id)
    for place in places:
        if place.lat is None:
            assert place.lng is None
            assert place.coordinate_source == "unavailable"
            assert place.coordinate_city_id is None
        else:
            assert place.coordinate_source == "admin1-largest-city-v1"
            assert place.coordinate_quality == "derived"

    jp40 = real_store.get_place("admin1", "JP.40")
    assert (jp40.lat, jp40.lng, jp40.coordinate_city_id) == (35.6895, 139.69171, 1850147)
    assert jp40.coordinate_quality == "derived"
    assert jp40.label == "Tokyo"
    ad02 = real_store.get_place("admin1", "AD.02")
    assert (ad02.lat, ad02.lng, ad02.coordinate_city_id) == (None, None, None)
    assert ad02.coordinate_source == "unavailable"
    assert ad02.coordinate_quality == "unavailable"
    assert ad02.label == "Canillo"
    assert real_store.get_place("country", "JP").coordinate_quality == "catalog"
    assert real_store.get_place("city", "1850147").coordinate_quality == "catalog"


def test_import_version_binds_every_derivation_policy():
    assert IMPORT_VERSION.startswith("geo-catalog-v1+")
    assert "admin1-largest-city-v1" in IMPORT_VERSION
    assert POLICY_VERSION in IMPORT_VERSION


def test_unchanged_restart_performs_no_writes(real_catalog):
    writes: list[str] = []

    def _record(_conn, _cursor, statement, _parameters, _context, _executemany):
        head = statement.lstrip().split(None, 1)[0].upper() if statement else ""
        if head in {"INSERT", "UPDATE", "DELETE"}:
            writes.append(statement)

    with real_catalog.session_factory() as session:
        before = {
            row.source: row.imported_at
            for row in session.execute(select(GeoCatalogSource)).scalars()
        }
        assert before

    event.listen(real_catalog.engine, "before_cursor_execute", _record)
    try:
        result = import_geo_catalog(real_catalog.engine, real_catalog.settings)
        checked = catalog_import_reasons(real_catalog.engine, real_catalog.settings)
    finally:
        event.remove(real_catalog.engine, "before_cursor_execute", _record)

    with real_catalog.session_factory() as session:
        after = {
            row.source: row.imported_at
            for row in session.execute(select(GeoCatalogSource)).scalars()
        }

    assert result.imported is False
    assert result.reasons == ("unchanged",)
    assert checked.reasons == ("unchanged",)
    assert after == before
    assert writes == []


def test_startup_import_runs_in_db_mode_and_json_mode_skips_store(tmp_path):
    settings, _paths = _fixture_settings(tmp_path / "db-mode")
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").status_code == 200
        with client.app.state.session_factory() as session:
            store = GeoCatalogStore(session)
            assert store.revisions()["import_version"] == IMPORT_VERSION
            assert store.query_places("city", limit=10)["total"] == 4

    json_settings, _paths = _fixture_settings(tmp_path / "json-mode")
    json_settings = replace(json_settings, geo_catalog_mode="json")
    with TestClient(create_app(json_settings)) as client:
        assert client.get("/api/health").status_code == 200
        with client.app.state.session_factory() as session:
            store = GeoCatalogStore(session)
            assert store.revisions() == {
                "countries": None,
                "admin1": None,
                "cities": None,
                "import_version": None,
            }
            assert store.query_places("city", limit=10)["total"] == 0


# --------------------------------------------------------------------------
# CN grouping / source scoping
# --------------------------------------------------------------------------


def test_cn_grouping_groups_canonical_and_keeps_raw_scopes(real_store):
    total, rows = _query_all(real_store, "city", country="CN")
    assert total == 2336
    assert {row.source_iso for row in rows} == {"CN", "HK", "MO", "TW"}
    assert all(row.country_iso == "CN" for row in rows)

    cn_total, cn_rows = _query_all(real_store, "city", country="CN", source_iso="CN")
    assert cn_total == 2122 and {row.source_iso for row in cn_rows} == {"CN"}

    tw_total, tw_rows = _query_all(real_store, "city", country="TW")
    assert tw_total == 67 and {row.source_iso for row in tw_rows} == {"TW"}

    hk_total, hk_rows = _query_all(real_store, "city", source_iso="HK")
    assert hk_total == 141 and {row.source_iso for row in hk_rows} == {"HK"}

    admin1_total, admin1_rows = _query_all(
        real_store, "admin1", country="CN", coordinates_only=False
    )
    assert admin1_total == 61
    assert {row.source_iso for row in admin1_rows} == {"CN", "HK", "MO", "TW"}

    tw_country = real_store.get_place("country", "TW")
    assert (tw_country.source_iso, tw_country.country_iso) == ("TW", "CN")
    nl_cities = real_store.query_places("city", country="NL", limit=5)
    assert all(row.country_iso == "NL" for row in nl_cities["rows"])


def test_query_places_reports_missing_coordinates_and_stable_paging(real_store):
    first = real_store.query_places("admin1", limit=QUERY_MAX_LIMIT, coordinates_only=False)
    assert first["total"] == 3865 and first["missing_coordinates"] == 1101
    assert len(first["rows"]) == QUERY_MAX_LIMIT
    second = real_store.query_places(
        "admin1", limit=QUERY_MAX_LIMIT, offset=QUERY_MAX_LIMIT, coordinates_only=False
    )
    assert len(second["rows"]) == 3865 - QUERY_MAX_LIMIT
    ids = [row.place_id for row in first["rows"]] + [row.place_id for row in second["rows"]]
    assert len(ids) == len(set(ids)) == 3865

    # Paging is stable and the default page size stays bounded.
    repeat = real_store.query_places("admin1", limit=10, offset=5, coordinates_only=False)
    assert [row.place_id for row in repeat["rows"]] == [
        row.place_id for row in first["rows"][5:15]
    ]
    assert QUERY_DEFAULT_LIMIT == 500

    scoped = real_store.query_places(
        "admin1", country="CN", coordinates_only=False, limit=QUERY_MAX_LIMIT
    )
    keys = [(row.source_iso, row.place_id) for row in scoped["rows"]]
    assert keys == sorted(keys)
    assert scoped["missing_coordinates"] == 6
    assert scoped["total"] == 61


# --------------------------------------------------------------------------
# store API surface
# --------------------------------------------------------------------------


def test_query_places_q_is_unicode_normalized_and_parameterized(real_store):
    tokyo = real_store.query_places("admin1", q="\uff34\uff2f\uff2b\uff39\uff2f")  # full-width TOKYO
    assert tokyo["total"] == 1 and tokyo["rows"][0].place_id == "JP.40"
    assert real_store.query_places("admin1", q="tokyo")["total"] == 1
    shanghai = real_store.query_places("admin1", q="\u4e0a\u6d77")
    assert [row.place_id for row in shanghai["rows"]] == ["CN.23"]
    assert shanghai["rows"][0].label == "\u4e0a\u6d77\u5e02"
    injected = real_store.query_places("admin1", q="Tok'yo OR '1'='1")
    assert injected["total"] == 0
    assert real_store.query_places("country", q="   ")["total"] == 250
    assert real_store.query_places("country", q="japan")["total"] == 1


def test_query_places_bbox_matches_an_independent_scan(real_store):
    bbox = (136.0, 34.0, 137.5, 36.0)
    page = real_store.query_places("city", source_iso="JP", bbox=bbox, limit=QUERY_MAX_LIMIT)
    expected = [
        city
        for city in json.loads(CITIES_PATH.read_text(encoding="utf-8"))
        if city["iso"] == "JP"
        and bbox[0] <= city["lng"] <= bbox[2]
        and bbox[1] <= city["lat"] <= bbox[3]
    ]
    expected.sort(key=lambda city: (-int(city["population"]), int(city["city_id"])))
    assert page["total"] == len(expected) == 123
    assert page["missing_coordinates"] == 0
    assert [row.place_id for row in page["rows"]] == [
        str(city["city_id"]) for city in expected
    ]
    assert all(
        bbox[0] <= row.lng <= bbox[2] and bbox[1] <= row.lat <= bbox[3]
        for row in page["rows"]
    )


def test_query_places_validates_arguments(real_store):
    with pytest.raises(ValueError):
        real_store.query_places("continent")
    with pytest.raises(ValueError):
        real_store.query_places("city", limit=0)
    with pytest.raises(ValueError):
        real_store.query_places("city", limit=QUERY_MAX_LIMIT + 1)
    with pytest.raises(ValueError):
        real_store.query_places("city", offset=-1)
    with pytest.raises(ValueError):
        real_store.query_places("city", country="C1")
    with pytest.raises(ValueError):
        real_store.query_places("city", source_iso="C1")
    with pytest.raises(ValueError):
        real_store.query_places("city", bbox=(10.0, 10.0, 9.0, 11.0))
    with pytest.raises(ValueError):
        real_store.query_places("city", bbox=(10.0, 10.0, 11.0))
    with pytest.raises(ValueError):
        real_store.get_place("city", "")
    with pytest.raises(ValueError):
        real_store.get_place("continent", "JP")


def test_store_adapters_match_the_existing_catalogs(real_store):
    admin1_index = load_admin1_index(ADMIN1_PATH)
    assert real_store.admin1_rows("JP") == admin1_index.by_iso["JP"]
    assert len(real_store.admin1_rows("JP")) == 47

    city_index = load_city_index(CITIES_PATH)
    assert real_store.city_rows("JP") == city_index["JP"]
    assert real_store.city_rows("TW") == city_index["TW"]
    assert real_store.city_rows("JP", admin1_id="JP.09") == {
        city_id: row for city_id, row in city_index["JP"].items() if row.region == "09"
    }

    raw_countries = json.loads(COUNTRIES_PATH.read_text(encoding="utf-8"))
    assert {row["iso"]: row for row in real_store.countries()} == {
        row["iso"]: row for row in raw_countries
    }
    assert real_store.get_place("country", "XX") is None


def test_store_on_an_empty_migrated_database_returns_nothing(tmp_path):
    settings = replace(make_settings(tmp_path), geo_catalog_mode="json")
    engine, session_factory = init_db(settings)
    try:
        with session_factory() as session:
            store = GeoCatalogStore(session)
            assert store.revisions() == {
                "countries": None,
                "admin1": None,
                "cities": None,
                "import_version": None,
            }
            assert store.query_places("city", country="JP") == {
                "rows": [],
                "total": 0,
                "missing_coordinates": 0,
            }
            assert store.get_place("country", "JP") is None
            assert store.countries() == []
            assert store.admin1_rows("JP") == ()
            assert store.city_rows("JP") == {}
    finally:
        engine.dispose()


# --------------------------------------------------------------------------
# fixture catalogs: re-derivation, rollback, leading zeros
# --------------------------------------------------------------------------


def test_fixture_derives_admin1_coordinates_and_unavailable_rows(fixture_store):
    jp01 = fixture_store.get_place("admin1", "JP.01")
    assert (jp01.lat, jp01.lng) == (35.18, 136.9)
    assert jp01.coordinate_source == "admin1-largest-city-v1"
    assert jp01.coordinate_city_id == 1001
    assert jp01.coordinate_quality == "derived"
    assert list(jp01.aliases) == ["\u611b\u77e5\u770c"]
    assert jp01.payload["code"] == "01"

    jp02 = fixture_store.get_place("admin1", "JP.02")
    assert (jp02.lat, jp02.lng) == (None, None)
    assert jp02.coordinate_source == "unavailable"
    assert jp02.coordinate_quality == "unavailable"
    assert fixture_store.get_place("country", "JP").coordinate_quality == "catalog"
    assert fixture_store.get_place("city", "1001").coordinate_quality == "catalog"

    tw01 = fixture_store.get_place("admin1", "TW.01")
    assert (tw01.source_iso, tw01.country_iso) == ("TW", "CN")
    assert (tw01.lat, tw01.lng, tw01.coordinate_city_id) == (25.03, 121.56, 1004)

    matched = fixture_store.get_place("city", "1001")
    assert (matched.admin1_id, matched.parent_id, matched.code) == ("JP.01", "JP.01", "01")
    unmatched = fixture_store.get_place("city", "1003")
    assert (unmatched.admin1_id, unmatched.parent_id, unmatched.code) == (None, "JP", "99")
    assert fixture_store.get_place("country", "JP").label == "\u65e5\u672c"


def test_changed_city_file_rederives_admin1_coordinates(fixture_catalog):
    _write_json(fixture_catalog.paths["cities"], CITIES_FIXTURE_V2)
    assert fixture_catalog.paths["cities"].exists()  # JSON stays the pinned source

    result = import_geo_catalog(fixture_catalog.engine, fixture_catalog.settings)
    assert result.imported is True
    assert "cities:fingerprint" in result.reasons

    with fixture_catalog.session_factory() as session:
        store = GeoCatalogStore(session)
        jp01 = store.get_place("admin1", "JP.01")
        assert (jp01.lat, jp01.lng) == (35.0, 137.0)
        assert jp01.coordinate_city_id == 1002
        assert store.get_place("admin1", "TW.01").coordinate_city_id == 1004
        assert store.revisions()["cities"] == catalog_digest(fixture_catalog.paths["cities"])
        assert store.get_place("city", "1002").population == 2000000


def test_policy_or_import_version_mismatch_forces_a_rebuild(fixture_catalog):
    with fixture_catalog.engine.begin() as connection:
        connection.exec_driver_sql(
            "UPDATE geo_catalog_source SET import_version = 'geo-catalog-v1'"
        )

    result = import_geo_catalog(fixture_catalog.engine, fixture_catalog.settings)
    assert result.imported is True
    assert set(result.reasons) == {
        "countries:import_version",
        "admin1:import_version",
        "cities:import_version",
    }
    with fixture_catalog.session_factory() as session:
        store = GeoCatalogStore(session)
        assert store.revisions()["import_version"] == IMPORT_VERSION
        assert store.get_place("admin1", "JP.01").coordinate_city_id == 1001


def test_invalid_input_rolls_back_the_intact_previous_catalog(fixture_catalog):
    with fixture_catalog.session_factory() as session:
        store = GeoCatalogStore(session)
        before = SimpleNamespace(
            revisions=store.revisions(),
            counts=(store.query_places("country", limit=10)["total"],
                    store.query_places("admin1", limit=10)["total"],
                    store.query_places("city", limit=10)["total"]),
            jp01=(store.get_place("admin1", "JP.01").lat, store.get_place("admin1", "JP.01").lng),
        )

    broken = json.loads(json.dumps(CITIES_FIXTURE_V1))
    broken[0]["lat"] = 999.0
    _write_json(fixture_catalog.paths["cities"], broken)
    with pytest.raises(GeoCatalogImportError):
        import_geo_catalog(fixture_catalog.engine, fixture_catalog.settings)
    with pytest.raises(GeoCatalogImportError):
        catalog_import_reasons(fixture_catalog.engine, fixture_catalog.settings)

    with fixture_catalog.session_factory() as session:
        store = GeoCatalogStore(session)
        assert store.revisions() == before.revisions
        assert (store.query_places("country", limit=10)["total"],
                store.query_places("admin1", limit=10)["total"],
                store.query_places("city", limit=10)["total"]) == before.counts
        assert (store.get_place("admin1", "JP.01").lat,
                store.get_place("admin1", "JP.01").lng) == before.jp01


def test_zero_zero_placeholder_is_rejected_and_changes_nothing(fixture_catalog):
    with fixture_catalog.session_factory() as session:
        before = GeoCatalogStore(session).revisions()

    broken = json.loads(json.dumps(CITIES_FIXTURE_V1))
    broken[0]["lat"] = 0.0
    broken[0]["lng"] = 0.0
    _write_json(fixture_catalog.paths["cities"], broken)
    with pytest.raises(GeoCatalogImportError, match=r"\(0, 0\)"):
        import_geo_catalog(fixture_catalog.engine, fixture_catalog.settings)

    with fixture_catalog.session_factory() as session:
        store = GeoCatalogStore(session)
        assert store.revisions() == before
        assert store.get_place("city", "1001").lat == 35.18


def test_missing_source_file_is_reported_and_changes_nothing(fixture_catalog):
    with fixture_catalog.session_factory() as session:
        before = GeoCatalogStore(session).revisions()
    fixture_catalog.paths["cities"].unlink()
    with pytest.raises(GeoCatalogImportError, match="unreadable"):
        import_geo_catalog(fixture_catalog.engine, fixture_catalog.settings)
    with fixture_catalog.session_factory() as session:
        assert GeoCatalogStore(session).revisions() == before


def test_mid_transaction_failure_rolls_back_the_replacement(fixture_catalog):
    # A trigger aborts the bulk insert half way through the new catalog; the
    # delete + inserts must roll back together leaving the previous catalog.
    with fixture_catalog.engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TRIGGER geo_place_block BEFORE INSERT ON geo_place "
            "WHEN NEW.place_id = 'XX' BEGIN SELECT RAISE(ABORT, 'blocked import'); END"
        )
    with fixture_catalog.session_factory() as session:
        before = GeoCatalogStore(session).revisions()
    _write_json(
        fixture_catalog.paths["countries"],
        COUNTRIES_FIXTURE + [{"iso": "XX", "iso3": "XXX", "name": "Blocked", "lat": 1.0, "lng": 2.0}],
    )

    with pytest.raises(IntegrityError):
        import_geo_catalog(fixture_catalog.engine, fixture_catalog.settings)

    with fixture_catalog.session_factory() as session:
        store = GeoCatalogStore(session)
        assert store.revisions()["countries"] == before["countries"]
        assert store.revisions()["import_version"] == IMPORT_VERSION
        assert store.query_places("country", limit=10)["total"] == 2
        assert store.get_place("country", "XX") is None
        assert store.get_place("admin1", "JP.01").coordinate_city_id == 1001


def test_leading_zero_admin1_codes_survive_the_round_trip(fixture_store):
    jp01 = fixture_store.get_place("admin1", "JP.01")
    assert jp01.place_id == "JP.01"
    assert jp01.code == "01"
    assert jp01.coordinate_city_id == 1001

    city = fixture_store.get_place("city", "1001")
    assert city.code == "01"

    admin1_rows = fixture_store.admin1_rows("JP")
    assert [row.admin1_id for row in admin1_rows] == ["JP.01", "JP.02"]
    assert admin1_rows[0].code == "01"

    cities = fixture_store.city_rows("JP", admin1_id="JP.01")
    assert set(cities) == {1001, 1002}
    assert cities[1001].region == "01"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_check_import_and_force_against_a_temp_database(tmp_path):
    settings = make_settings(tmp_path)
    env = dict(
        os.environ,
        M_PHOTO_DATA_DIR=str(settings.data_dir),
        M_PHOTO_DB_URL=settings.database_url,
        # The CLI is an explicit operator action and runs regardless of the
        # runtime mode; this also proves the module never consults the network.
        M_PHOTO_GEO_CATALOG_MODE="json",
        PYTHONPATH=str(REPO_ROOT),
    )
    command = [sys.executable, "-m", "backend.app.services.geo_catalog_import"]

    def run(*args):
        return subprocess.run(
            command + list(args), cwd=tmp_path, env=env, text=True,
            capture_output=True, timeout=120,
        )

    check = run("--check")
    assert check.returncode == 1, check.stdout + check.stderr
    assert "import required" in check.stdout

    imported = run()
    assert imported.returncode == 0, imported.stdout + imported.stderr
    assert "imported" in imported.stdout

    fresh = run("--check")
    assert fresh.returncode == 0, fresh.stdout + fresh.stderr
    assert "up to date" in fresh.stdout

    forced = run("--force")
    assert forced.returncode == 0, forced.stdout + forced.stderr
    assert "imported" in forced.stdout

    conflict = run("--check", "--force")
    assert conflict.returncode == 2

    engine, session_factory = init_db(settings)
    try:
        with session_factory() as session:
            store = GeoCatalogStore(session)
            assert store.revisions()["cities"] == catalog_digest(CITIES_PATH)
            assert store.query_places("country", limit=QUERY_MAX_LIMIT)["total"] == 250
    finally:
        engine.dispose()
