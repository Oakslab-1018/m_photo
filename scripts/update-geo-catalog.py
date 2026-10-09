#!/usr/bin/env python3
"""Build the reproducible country/territory and city metadata catalogs.

Pinned inputs (recorded in ``data/geo-catalog-sources.json``):

* ``mledoze/countries`` @ commit ``c2ac0049c14edcf2436c7aa1b2493222a020b462``
  (ODbL-1.0, verified from the repository ``LICENSE`` file), file
  ``countries.json`` (250 entries, including the user-assigned code ``XK``).
* GeoNames ``cities15000.zip`` retrieved 2026-10-09 (CC BY 4.0, verified from
  the GeoNames ``readme.txt``). GeoNames refreshes the dump daily, so the zip
  is pinned by sha256: a checksum mismatch fails the build instead of silently
  producing different data.

The script downloads missing inputs into the git-ignored cache
``.cache/geo-catalog``, never extracts an archive to disk (it reads the single
``cities15000.txt`` member in memory) and writes deterministic
one-record-per-line JSON. Use ``--offline`` to build strictly from the cache
and ``--check`` to prove the committed catalogs match a clean rebuild.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "geo-catalog"
DEFAULT_COUNTRIES_PATH = REPO_ROOT / "data" / "countries.json"
DEFAULT_CITIES_PATH = REPO_ROOT / "data" / "cities.json"
DEFAULT_SOURCES_PATH = REPO_ROOT / "data" / "geo-catalog-sources.json"

RETRIEVED_DATE = "2026-10-09"

MLEDOZE_SLUG = "mledoze/countries"
MLEDOZE_COMMIT = "c2ac0049c14edcf2436c7aa1b2493222a020b462"
MLEDOZE_URL = (
    f"https://raw.githubusercontent.com/{MLEDOZE_SLUG}/{MLEDOZE_COMMIT}/countries.json"
)
MLEDOZE_LICENSE_URL = (
    f"https://raw.githubusercontent.com/{MLEDOZE_SLUG}/{MLEDOZE_COMMIT}/LICENSE"
)
MLEDOZE_SHA256 = "913e5d716f9dc6b59881ee23488d47f5fda52070d1a34cac5fa327c9e61d8f7c"
MLEDOZE_LICENSE_SHA256 = "a221a27af5e4ef5a9cecce097c718c28f1aaf7d368cb57c994aa447c68d9defc"
MLEDOZE_LICENSE_NAME = "ODC Open Database License (ODbL) v1.0"

GEONAMES_URL = "https://download.geonames.org/export/dump/cities15000.zip"
GEONAMES_README_URL = "https://download.geonames.org/export/dump/readme.txt"
GEONAMES_SHA256 = "0ea3b760f5b0b026a4b5ecba0f6582bb93c064a60d59c9ad88545e1aba47b7a0"
GEONAMES_README_SHA256 = "b1957379b6c1242c700c98ac9a8aa0a09f56c3c0a50ee72175527005f48ef2c5"
GEONAMES_LICENSE_NAME = "Creative Commons Attribution 4.0 (CC BY 4.0)"
GEONAMES_ZIP_MEMBER = "cities15000.txt"
GEONAMES_MAX_MEMBER_BYTES = 64 * 1024 * 1024

CACHE_MLEDOZE = "mledoze-countries.json"
CACHE_MLEDOZE_LICENSE = "mledoze-LICENSE.txt"
CACHE_GEONAMES_ZIP = "cities15000.zip"
CACHE_GEONAMES_README = "geonames-readme.txt"

COUNTRY_KEYS = (
    "name", "name_zh", "iso", "iso3", "lat", "lng", "continent", "region",
    "subregion", "driving_side", "independent", "un_member", "status",
    "languages_iso639_3", "languages_spoken", "features",
)
CITY_KEYS = ("city_id", "iso", "name", "ascii_name", "region", "lat", "lng", "population")

# Preserved curated knowledge for the original 20-country feature catalog:
# corrected Google representative coordinates (retrieved 2026-10-09), the
# original feature vocabulary, driving side and reviewer-curated language
# tokens. New countries never receive invented features or coordinates.
CURATED20 = json.loads(r"""{"JP":{"iso":"JP","name":"Japan","name_zh":"\u65e5\u672c","lat":36.2048,"lng":138.2529,"continent":"Asia","driving_side":"left","languages_spoken":["ja"],"features":{"language":["ja"],"vegetation":["temperate_forest"],"building":["modern_glass","generic"],"vehicle":["white_silver_vehicle","dark_vehicle"],"climate":["temperate","temperate_maritime"],"culture":["shrine","sakura"]}},"CN":{"iso":"CN","name":"China","name_zh":"\u4e2d\u56fd","lat":35.8617,"lng":104.1954,"continent":"Asia","driving_side":"right","languages_spoken":["zh"],"features":{"language":["zh"],"vegetation":["temperate_forest","grassland","desert"],"building":["modern_glass","colonial","generic"],"vehicle":["white_silver_vehicle","dark_vehicle"],"climate":["temperate","arid","mediterranean_sunny"],"culture":["pagoda","lantern"]}},"KR":{"iso":"KR","name":"South Korea","name_zh":"\u97e9\u56fd","lat":35.9078,"lng":127.7669,"continent":"Asia","driving_side":"right","languages_spoken":["ko"],"features":{"language":["ko"],"vegetation":["temperate_forest"],"building":["modern_glass","generic"],"vehicle":["white_silver_vehicle","dark_vehicle"],"climate":["temperate","temperate_maritime"],"culture":["hanbok"]}},"FR":{"iso":"FR","name":"France","name_zh":"\u6cd5\u56fd","lat":46.2276,"lng":2.2137,"continent":"Europe","driving_side":"right","languages_spoken":["latin"],"features":{"language":["latin"],"vegetation":["temperate_forest","grassland"],"building":["gothic","mediterranean","generic"],"vehicle":["white_silver_vehicle","dark_vehicle"],"climate":["temperate_maritime","mediterranean_sunny","temperate"],"culture":["eiffel","cafe"]}},"DE":{"iso":"DE","name":"Germany","name_zh":"\u5fb7\u56fd","lat":51.1657,"lng":10.4515,"continent":"Europe","driving_side":"right","languages_spoken":["latin"],"features":{"language":["latin"],"vegetation":["temperate_forest"],"building":["gothic","modern_glass","generic"],"vehicle":["dark_vehicle","white_silver_vehicle"],"climate":["temperate","temperate_maritime"],"culture":["castle"]}},"IT":{"iso":"IT","name":"Italy","name_zh":"\u610f\u5927\u5229","lat":41.8719,"lng":12.5674,"continent":"Europe","driving_side":"right","languages_spoken":["latin"],"features":{"language":["latin"],"vegetation":["grassland","savanna"],"building":["mediterranean","gothic","generic"],"vehicle":["white_silver_vehicle","dark_vehicle"],"climate":["mediterranean_sunny","arid"],"culture":["colosseum","pizza"]}},"GB":{"iso":"GB","name":"United Kingdom","name_zh":"\u82f1\u56fd","lat":55.3781,"lng":-3.436,"continent":"Europe","driving_side":"left","languages_spoken":["latin"],"features":{"language":["latin"],"vegetation":["temperate_forest","grassland"],"building":["gothic","generic"],"vehicle":["dark_vehicle","white_silver_vehicle"],"climate":["temperate_maritime","temperate"],"culture":["big_ben","red_bus"]}},"ES":{"iso":"ES","name":"Spain","name_zh":"\u897f\u73ed\u7259","lat":40.4637,"lng":-3.7492,"continent":"Europe","driving_side":"right","languages_spoken":["latin"],"features":{"language":["latin"],"vegetation":["grassland","savanna"],"building":["mediterranean","generic"],"vehicle":["white_silver_vehicle","dark_vehicle"],"climate":["mediterranean_sunny","arid"],"culture":["flamenco"]}},"GR":{"iso":"GR","name":"Greece","name_zh":"\u5e0c\u814a","lat":39.0742,"lng":21.8243,"continent":"Europe","driving_side":"right","languages_spoken":["latin"],"features":{"language":["latin"],"vegetation":["savanna","grassland"],"building":["mediterranean","generic"],"vehicle":["white_silver_vehicle","dark_vehicle"],"climate":["mediterranean_sunny"],"culture":["acropolis"]}},"RU":{"iso":"RU","name":"Russia","name_zh":"\u4fc4\u7f57\u65af","lat":61.524,"lng":105.3188,"continent":"Europe","driving_side":"right","languages_spoken":["cyrillic"],"features":{"language":["cyrillic"],"vegetation":["temperate_forest","urban_sparse"],"building":["generic","modern_glass","gothic"],"vehicle":["dark_vehicle","white_silver_vehicle"],"climate":["cold_polar","temperate"],"culture":["kremlin"]}},"US":{"iso":"US","name":"United States","name_zh":"\u7f8e\u56fd","lat":37.0902,"lng":-95.7129,"continent":"North America","driving_side":"right","languages_spoken":["latin"],"features":{"language":["latin"],"vegetation":["temperate_forest","grassland","desert"],"building":["modern_glass","colonial","generic"],"vehicle":["dark_vehicle","white_silver_vehicle"],"climate":["temperate","arid","mediterranean_sunny","cold_polar"],"culture":["statue_of_liberty"]}},"CA":{"iso":"CA","name":"Canada","name_zh":"\u52a0\u62ff\u5927","lat":56.1304,"lng":-106.3468,"continent":"North America","driving_side":"right","languages_spoken":["latin"],"features":{"language":["latin"],"vegetation":["temperate_forest","urban_sparse"],"building":["modern_glass","generic"],"vehicle":["dark_vehicle","white_silver_vehicle"],"climate":["cold_polar","temperate"],"culture":["maple"]}},"BR":{"iso":"BR","name":"Brazil","name_zh":"\u5df4\u897f","lat":-14.235,"lng":-51.9253,"continent":"South America","driving_side":"right","languages_spoken":["latin"],"features":{"language":["latin"],"vegetation":["tropical_rainforest","savanna"],"building":["modern_glass","colonial","generic"],"vehicle":["white_silver_vehicle","dark_vehicle"],"climate":["temperate","arid"],"culture":["samba"]}},"AU":{"iso":"AU","name":"Australia","name_zh":"\u6fb3\u5927\u5229\u4e9a","lat":-25.2744,"lng":133.7751,"continent":"Oceania","driving_side":"left","languages_spoken":["latin"],"features":{"language":["latin"],"vegetation":["desert","savanna","grassland"],"building":["modern_glass","generic"],"vehicle":["white_silver_vehicle","dark_vehicle"],"climate":["arid","mediterranean_sunny","temperate"],"culture":["opera_house"]}},"IN":{"iso":"IN","name":"India","name_zh":"\u5370\u5ea6","lat":20.5937,"lng":78.9629,"continent":"Asia","driving_side":"left","languages_spoken":["latin"],"features":{"language":["latin"],"vegetation":["tropical_rainforest","grassland","desert"],"building":["generic","colonial"],"vehicle":["white_silver_vehicle","dark_vehicle"],"climate":["temperate","arid"],"culture":["temple","sari"]}},"TH":{"iso":"TH","name":"Thailand","name_zh":"\u6cf0\u56fd","lat":15.87,"lng":100.9925,"continent":"Asia","driving_side":"left","languages_spoken":["thai"],"features":{"language":["thai"],"vegetation":["tropical_rainforest","savanna"],"building":["generic","mediterranean"],"vehicle":["white_silver_vehicle","dark_vehicle"],"climate":["temperate","arid"],"culture":["temple","elephant"]}},"EG":{"iso":"EG","name":"Egypt","name_zh":"\u57c3\u53ca","lat":26.8206,"lng":30.8025,"continent":"Africa","driving_side":"right","languages_spoken":["arabic"],"features":{"language":["arabic"],"vegetation":["desert","savanna"],"building":["generic","mediterranean"],"vehicle":["white_silver_vehicle","dark_vehicle"],"climate":["arid","mediterranean_sunny"],"culture":["pyramid"]}},"MX":{"iso":"MX","name":"Mexico","name_zh":"\u58a8\u897f\u54e5","lat":23.6345,"lng":-102.5528,"continent":"North America","driving_side":"right","languages_spoken":["latin"],"features":{"language":["latin"],"vegetation":["desert","savanna","tropical_rainforest"],"building":["colonial","generic"],"vehicle":["white_silver_vehicle","dark_vehicle"],"climate":["arid","mediterranean_sunny"],"culture":["aztec"]}},"NL":{"iso":"NL","name":"Netherlands","name_zh":"\u8377\u5170","lat":52.1326,"lng":5.2913,"continent":"Europe","driving_side":"right","languages_spoken":["latin"],"features":{"language":["latin"],"vegetation":["grassland","temperate_forest"],"building":["generic","gothic"],"vehicle":["dark_vehicle","white_silver_vehicle"],"climate":["temperate_maritime","temperate"],"culture":["windmill"]}},"SE":{"iso":"SE","name":"Sweden","name_zh":"\u745e\u5178","lat":60.1282,"lng":18.6435,"continent":"Europe","driving_side":"right","languages_spoken":["latin"],"features":{"language":["latin"],"vegetation":["temperate_forest","urban_sparse"],"building":["modern_glass","generic"],"vehicle":["dark_vehicle","white_silver_vehicle"],"climate":["cold_polar","temperate"],"culture":["viking"]}}}""")

SCRIPT_TOKEN_ORDER = ("zh", "ja", "ko", "cyrillic", "arabic", "thai", "devanagari", "latin")

# ISO 639-3 -> conservative writing-system tokens. Only tokens the OCR/script
# analyzers actually detect are used; languages written in scripts outside that
# vocabulary map to no token instead of a wrong one.
_SCRIPT_TOKENS: dict[str, tuple[str, ...]] = {
    "ara": ("arabic",), "ber": ("arabic",), "ckb": ("arabic",), "fas": ("arabic",),
    "mey": ("arabic",), "prs": ("arabic",), "pus": ("arabic",), "urd": ("arabic",),
    "zho": ("zh",), "jpn": ("ja",), "kor": ("ko",),
    "bel": ("cyrillic",), "bul": ("cyrillic",), "kaz": ("cyrillic",),
    "kir": ("cyrillic",), "mkd": ("cyrillic",), "mon": ("cyrillic",),
    "rus": ("cyrillic",), "srp": ("cyrillic", "latin"), "tgk": ("cyrillic",),
    "ukr": ("cyrillic",), "cnr": ("cyrillic", "latin"), "bos": ("latin", "cyrillic"),
    "hin": ("devanagari",), "nep": ("devanagari",), "tha": ("thai",), "lao": ("thai",),
}
_NON_LATIN_NO_TOKEN = {
    "amh", "div", "ell", "heb", "hye", "kat", "nzs", "sin", "tam", "tir", "zib",
}
_OCEANIA_SUBREGIONS = {"Australia and New Zealand", "Melanesia", "Micronesia", "Polynesia"}
_REGION_CONTINENT = {
    "Africa": "Africa", "Americas": "North America", "Asia": "Asia",
    "Europe": "Europe", "Oceania": "Oceania", "Antarctic": "Antarctica",
}


class BuildError(RuntimeError):
    """Raised when a source input or a generated record fails a hard check."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _download(url: str, destination: Path) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "m-photo-geo-catalog/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - pinned https inputs
        payload = response.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return payload


def _cached_bytes(url: str, cache_dir: Path, filename: str, sha256: str, *, offline: bool) -> bytes:
    path = cache_dir / filename
    if path.is_file():
        payload = path.read_bytes()
        if sha256_bytes(payload) == sha256:
            return payload
        if offline:
            raise BuildError(f"cached input hash mismatch: {path}")
    elif offline:
        raise BuildError(f"missing cached input for offline build: {path}")
    payload = _download(url, path)
    if sha256_bytes(payload) != sha256:
        raise BuildError(f"downloaded input hash mismatch: {filename}")
    return payload


def _demonym_side(record: dict) -> str | None:
    """Use an explicit driving-side label only; never infer it from demonyms."""
    value = (record.get("car") or {}).get("side")
    return value if value in {"left", "right"} else None


def _continent(record: dict) -> str | None:
    region = record.get("region")
    if region == "Americas":
        return "South America" if record.get("subregion") == "South America" else "North America"
    return _REGION_CONTINENT.get(region)


def _language_tokens(codes: list[str]) -> list[str]:
    tokens: list[str] = []
    for code in codes:
        mapped = _SCRIPT_TOKENS.get(code)
        if mapped:
            for token in mapped:
                if token not in tokens:
                    tokens.append(token)
        elif code in _NON_LATIN_NO_TOKEN:
            continue
        elif code not in tokens:
            tokens.append("latin")
    return [token for token in SCRIPT_TOKEN_ORDER if token in tokens]


def _clean_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text.lower() in {"n/a", "null", "none"}:
        return None
    return text


def _clean_coord(value: object, *, low: float, high: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not low <= number <= high:
        return None
    return number


def build_countries(raw_payload: bytes) -> list[dict]:
    records = json.loads(raw_payload.decode("utf-8"))
    if not isinstance(records, list) or not records:
        raise BuildError("mledoze payload must be a non-empty JSON array")

    curated = {iso: record for iso, record in CURATED20.items()}
    countries: list[dict] = []
    seen: set[str] = set()
    for record in records:
        iso = _clean_text(record.get("cca2"))
        name = _clean_text((record.get("name") or {}).get("common"))
        if not iso or not name:
            raise BuildError("mledoze record without cca2/common name")
        iso = iso.upper()
        if iso in seen:
            raise BuildError(f"duplicate cca2 code: {iso}")
        seen.add(iso)

        names = record.get("name") or {}
        native = names.get("native") or {}
        translations = record.get("translations") or {}
        name_zh = _clean_text((translations.get("zho") or {}).get("common"))
        if name_zh is None and native:
            first = native.get(sorted(native)[0]) or {}
            name_zh = _clean_text(first.get("common"))
        if name_zh is None:
            name_zh = name

        languages = record.get("languages") or {}
        codes = sorted(code for code in languages if isinstance(code, str) and code.strip())
        tokens = _language_tokens(codes)
        latlng = record.get("latlng") or []
        if len(latlng) != 2:
            raise BuildError(f"country {iso} has no representative coordinate pair")
        lat = _clean_coord(latlng[0], low=-90.0, high=90.0)
        lng = _clean_coord(latlng[1], low=-180.0, high=180.0)
        if lat is None or lng is None:
            raise BuildError(f"country {iso} has invalid representative coordinates")

        match = curated.get(iso)
        if match:
            entry = {
                "name": name, "name_zh": match["name_zh"], "iso": iso,
                "iso3": _clean_text(record.get("cca3")), "lat": match["lat"], "lng": match["lng"],
                "continent": match["continent"], "region": _clean_text(record.get("region")),
                "subregion": _clean_text(record.get("subregion")),
                "driving_side": match["driving_side"], "independent": record.get("independent"),
                "un_member": record.get("unMember"), "status": _clean_text(record.get("status")),
                "languages_iso639_3": codes, "languages_spoken": match["languages_spoken"],
                "features": match["features"],
            }
        else:
            entry = {
                "name": name, "name_zh": name_zh, "iso": iso, "iso3": _clean_text(record.get("cca3")),
                "lat": lat, "lng": lng, "continent": _continent(record),
                "region": _clean_text(record.get("region")),
                "subregion": _clean_text(record.get("subregion")),
                "driving_side": _demonym_side(record), "independent": record.get("independent"),
                "un_member": record.get("unMember"), "status": _clean_text(record.get("status")),
                "languages_iso639_3": codes, "languages_spoken": tokens, "features": {},
            }
        countries.append(entry)

    if len(countries) != len(seen):
        raise BuildError("duplicate country records")
    return sorted(countries, key=lambda item: item["iso"])

def build_cities(zip_payload: bytes) -> list[dict]:
    """Read ``cities15000.txt`` in memory; never extract the archive to disk."""
    with zipfile.ZipFile(io.BytesIO(zip_payload)) as archive:
        info = archive.getinfo(GEONAMES_ZIP_MEMBER)
        if info.file_size > GEONAMES_MAX_MEMBER_BYTES:
            raise BuildError(f"unexpected {GEONAMES_ZIP_MEMBER} size: {info.file_size}")
        with archive.open(info) as handle:
            text = handle.read().decode("utf-8").splitlines()

    cities: dict[str, dict] = {}
    for line_number, line in enumerate(text, start=1):
        if not line.strip():
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 15:
            raise BuildError(f"cities15000 line {line_number} has {len(fields)} columns")
        city_id = fields[0].strip()
        name = fields[1].strip()
        ascii_name = fields[2].strip()
        lat = _clean_coord(fields[4], low=-90.0, high=90.0)
        lng = _clean_coord(fields[5], low=-180.0, high=180.0)
        feature_class = fields[6].strip()
        country = fields[8].strip().upper()
        admin1 = fields[10].strip()
        population_raw = fields[14].strip()
        if feature_class != "P":
            raise BuildError(f"cities15000 line {line_number} is not a populated place")
        if not city_id.isdigit() or not name or len(country) != 2 or not country.isalpha():
            raise BuildError(f"cities15000 line {line_number} has an invalid identity")
        if lat is None or lng is None:
            raise BuildError(f"city {city_id} has invalid coordinates")
        try:
            population = int(population_raw)
        except ValueError as exc:
            raise BuildError(f"city {city_id} has a non-integer population") from exc
        if population < 0:
            raise BuildError(f"city {city_id} has a negative population")
        if city_id in cities:
            raise BuildError(f"duplicate geoname id: {city_id}")
        cities[city_id] = {
            "city_id": city_id, "iso": country,
            "name": name, "ascii_name": ascii_name or name,
            "region": admin1 or None, "lat": lat, "lng": lng, "population": population,
        }

    if not cities:
        raise BuildError("cities15000 contained no cities")
    ordered = sorted(cities.values(), key=lambda item: int(item["city_id"]))
    if len(ordered) != len(cities):
        raise BuildError("geoname ids are not unique")
    return ordered


def _dumps_records(records: list[dict]) -> str:
    lines = [json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records]
    return "[\n" + ",\n".join(lines) + "\n]\n"


def _dumps_sources(metadata: dict) -> str:
    return json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"


def _sources_metadata(mledoze_sha: str, geonames_sha: str, countries: list[dict], cities: list[dict]) -> dict:
    countries_sha = sha256_bytes(_dumps_records(countries).encode("utf-8"))
    cities_sha = sha256_bytes(_dumps_records(cities).encode("utf-8"))
    scopes = sorted({city["iso"] for city in cities})
    return {
        "schema_version": 1,
        "generated": "2026-10-09",
        "scope_note": (
            "Country/territory directory follows the ISO-style codes present in the "
            "pinned mledoze/countries dataset: 249 officially assigned codes plus the "
            "user-assigned code XK (Kosovo). This is not a claim that every entry is a "
            "sovereign state; 194 UN member states are marked with un_member=true and "
            "dependent territories are flagged with independent=false. Political "
            "status is reported as-is from the pinned source and is not adjudicated."
        ),
        "sources": {
            "countries": {
                "name": "mledoze/countries",
                "url": f"https://github.com/{MLEDOZE_SLUG}",
                "retrieved": RETRIEVED_DATE,
                "commit": MLEDOZE_COMMIT,
                "path": "countries.json",
                "license": MLEDOZE_LICENSE_NAME,
                "license_url": MLEDOZE_LICENSE_URL,
                "license_sha256": MLEDOZE_LICENSE_SHA256,
                "sha256": mledoze_sha,
                "records": len(countries),
                "note": "250 country/territory records incl. XK; coordinates are source representative points except for the preserved 20-country curated values.",
            },
            "cities": {
                "name": "GeoNames cities15000",
                "url": "https://download.geonames.org/export/dump/",
                "retrieved": RETRIEVED_DATE,
                "download_url": GEONAMES_URL,
                "license": GEONAMES_LICENSE_NAME,
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "attribution": "Contains data from GeoNames (https://www.geonames.org/), licensed CC BY 4.0.",
                "readme_sha256": GEONAMES_README_SHA256,
                "sha256": geonames_sha,
                "records": len(cities),
                "coverage_country_codes": scopes,
                "note": "Populated places with population > 15000 or capitals; region is the GeoNames admin1 code when no province names are embedded.",
            },
        },
        "outputs": {
            "data/countries.json": {"sha256": countries_sha, "records": len(countries)},
            "data/cities.json": {"sha256": cities_sha, "records": len(cities)},
        },
        "build_command": "python scripts/update-geo-catalog.py",
        "verify_command": "python scripts/update-geo-catalog.py --check",
    }

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--countries", type=Path, default=DEFAULT_COUNTRIES_PATH)
    parser.add_argument("--cities", type=Path, default=DEFAULT_CITIES_PATH)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES_PATH)
    parser.add_argument("--offline", action="store_true", help="fail instead of downloading")
    parser.add_argument("--check", action="store_true", help="verify outputs match a clean rebuild")
    args = parser.parse_args(argv)

    mledoze_payload = _cached_bytes(
        MLEDOZE_URL, args.cache_dir, CACHE_MLEDOZE, MLEDOZE_SHA256, offline=args.offline
    )
    geonames_payload = _cached_bytes(
        GEONAMES_URL, args.cache_dir, CACHE_GEONAMES_ZIP, GEONAMES_SHA256, offline=args.offline
    )
    _cached_bytes(
        MLEDOZE_LICENSE_URL, args.cache_dir, CACHE_MLEDOZE_LICENSE,
        MLEDOZE_LICENSE_SHA256, offline=args.offline,
    )
    _cached_bytes(
        GEONAMES_README_URL, args.cache_dir, CACHE_GEONAMES_README,
        GEONAMES_README_SHA256, offline=args.offline,
    )

    countries = build_countries(mledoze_payload)
    cities = build_cities(geonames_payload)

    countries_text = _dumps_records(countries)
    cities_text = _dumps_records(cities)
    sources_text = _dumps_sources(
        _sources_metadata(
            sha256_bytes(mledoze_payload), sha256_bytes(geonames_payload), countries, cities
        )
    )

    if args.check:
        failures = []
        for path, expected in (
            (args.countries, countries_text),
            (args.cities, cities_text),
            (args.sources, sources_text),
        ):
            actual = path.read_text(encoding="utf-8") if path.is_file() else ""
            if actual != expected:
                failures.append(str(path))
        if failures:
            print("catalog check failed; regenerate: " + ", ".join(failures), file=sys.stderr)
            return 1
        print(f"catalog check ok: {len(countries)} countries, {len(cities)} cities")
        return 0

    for path, text in (
        (args.countries, countries_text),
        (args.cities, cities_text),
        (args.sources, sources_text),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {path} ({len(text)} bytes)")

    feature_count = sum(1 for country in countries if country["features"])
    missing_coords = sum(1 for country in countries if country["lat"] is None or country["lng"] is None)
    print(
        f"countries={len(countries)} with_features={feature_count} "
        f"missing_coords={missing_coords} cities={len(cities)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
