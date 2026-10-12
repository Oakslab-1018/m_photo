#!/usr/bin/env python3
"""Build the authoritative GeoNames admin1 catalog in SQLite.

The only remote input is the small official GeoNames file::

    https://download.geonames.org/export/dump/admin1CodesASCII.txt

Use ``--input`` to build from a local copy of that exact file (offline or
reviewable builds).  The default source SHA-256 is pinned so a changed GeoNames
dump cannot silently replace the catalog.  The generated rows preserve the full
directory and every code exactly as supplied by GeoNames, including leading-zero
codes such as ``JP.09``.

Rows are written to the portable SQLite catalog through the atomic maintenance
writer.  The retired generated ``data/admin1.json`` is never read, required or
recreated.  The crosswalk is computed from the ``cities`` rows already stored in
the same database, never from ``data/cities.json``.  City region codes absent
from the authoritative admin1 directory are reported explicitly; no placeholder
admin1 row is invented.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_PATH = REPO_ROOT / "data" / "geo_catalog.sqlite3"
DEFAULT_SOURCES_PATH = REPO_ROOT / "data" / "admin1-sources.json"
SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from catalog_db import CatalogDatabase, CatalogDatabaseError  # noqa: E402
from geo_catalog_maintenance import (  # noqa: E402
    CatalogMaintenanceError,
    check_catalog_levels,
    replace_catalog_levels,
)

ADMIN1_URL = "https://download.geonames.org/export/dump/admin1CodesASCII.txt"
RETRIEVED_DATE = "2026-10-12"
ADMIN1_SHA256 = "1da92a6323a5fec3176f3f743bf4cf4040fd56a876da55e46fbca23c863aa60a"
LICENSE_NAME = "Creative Commons Attribution 4.0 (CC BY 4.0)"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
ATTRIBUTION = "Contains data from GeoNames (https://www.geonames.org/), licensed CC BY 4.0."

_ADMIN1_ID_RE = re.compile(r"^(?P<iso>[A-Z]{2})\.(?P<code>[A-Z0-9]+)$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

# Verified Chinese labels for the CN province-level rows present in the
# pinned GeoNames file.  The expected English name is checked against the
# source before a translation is attached, so a future source change cannot
# silently apply a translation to the wrong code.  Rows or codes not listed
# here remain authoritative English GeoNames rows with no fabricated Chinese
# label.
CN_PROVINCE_TRANSLATIONS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "CN.01": ("Anhui", "安徽省", ("安徽",)),
    "CN.02": ("Zhejiang", "浙江省", ("浙江",)),
    "CN.03": ("Jiangxi", "江西省", ("江西",)),
    "CN.04": ("Jiangsu", "江苏省", ("江苏",)),
    "CN.05": ("Jilin", "吉林省", ("吉林",)),
    "CN.06": ("Qinghai", "青海省", ("青海",)),
    "CN.07": ("Fujian", "福建省", ("福建",)),
    "CN.08": ("Heilongjiang", "黑龙江省", ("黑龙江",)),
    "CN.09": ("Henan", "河南省", ("河南",)),
    "CN.10": ("Hebei", "河北省", ("河北",)),
    "CN.11": ("Hunan", "湖南省", ("湖南",)),
    "CN.12": ("Hubei", "湖北省", ("湖北",)),
    "CN.13": ("Xinjiang", "新疆维吾尔自治区", ("新疆",)),
    "CN.14": ("Tibet", "西藏自治区", ("西藏",)),
    "CN.15": ("Gansu", "甘肃省", ("甘肃",)),
    "CN.16": ("Guangxi", "广西壮族自治区", ("广西",)),
    "CN.18": ("Guizhou", "贵州省", ("贵州",)),
    "CN.19": ("Liaoning", "辽宁省", ("辽宁",)),
    "CN.20": ("Inner Mongolia", "内蒙古自治区", ("内蒙古",)),
    "CN.21": ("Ningxia", "宁夏回族自治区", ("宁夏",)),
    "CN.22": ("Beijing", "北京市", ("北京",)),
    "CN.23": ("Shanghai", "上海市", ("上海",)),
    "CN.24": ("Shanxi", "山西省", ("山西",)),
    "CN.25": ("Shandong", "山东省", ("山东",)),
    "CN.26": ("Shaanxi", "陕西省", ("陕西",)),
    "CN.28": ("Tianjin", "天津市", ("天津",)),
    "CN.29": ("Yunnan", "云南省", ("云南",)),
    "CN.30": ("Guangdong", "广东省", ("广东",)),
    "CN.31": ("Hainan", "海南省", ("海南",)),
    "CN.32": ("Sichuan", "四川省", ("四川",)),
    "CN.33": ("Chongqing", "重庆市", ("重庆",)),
}


class BuildError(RuntimeError):
    """Raised when an input or a generated catalog fails a hard check."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_source() -> bytes:
    request = urllib.request.Request(
        ADMIN1_URL,
        headers={"User-Agent": "m-photo-admin1-catalog/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - pinned official HTTPS URL
            return response.read()
    except OSError as exc:
        raise BuildError(f"could not download the official GeoNames input: {exc}") from exc


def _read_source(input_path: Path | None, expected_sha256: str) -> tuple[bytes, str]:
    if not _SHA256_RE.fullmatch(expected_sha256):
        raise BuildError("expected SHA-256 must be exactly 64 hexadecimal characters")

    if input_path is None:
        payload = _download_source()
    else:
        try:
            payload = Path(input_path).read_bytes()
        except OSError as exc:
            raise BuildError(f"could not read local GeoNames input: {input_path}") from exc

    actual_sha256 = _sha256_bytes(payload)
    if actual_sha256.lower() != expected_sha256.lower():
        raise BuildError(
            "GeoNames input SHA-256 mismatch: "
            f"expected {expected_sha256.lower()}, got {actual_sha256}"
        )
    return payload, actual_sha256


def _normalize_name(value: str, line_number: int, field: str) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        raise BuildError(f"line {line_number}: {field} must not be blank")
    return normalized


def build_admin1_rows(payload: bytes) -> list[dict[str, Any]]:
    """Parse the pinned GeoNames text into deterministic, validated rows."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BuildError("GeoNames input is not valid UTF-8") from exc
    if text.startswith("\ufeff"):
        text = text[1:]

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            raise BuildError(f"line {line_number}: blank lines are not allowed")
        fields = raw_line.split("\t")
        if len(fields) != 4:
            raise BuildError(
                f"line {line_number}: expected 4 tab-separated columns, got {len(fields)}"
            )

        admin1_id, name_raw, ascii_name_raw, geoname_raw = fields
        match = _ADMIN1_ID_RE.fullmatch(admin1_id)
        if match is None:
            raise BuildError(f"line {line_number}: invalid admin1 id {admin1_id!r}")
        iso = match.group("iso")
        code = match.group("code")
        if admin1_id in seen_ids:
            raise BuildError(f"line {line_number}: duplicate admin1 id {admin1_id!r}")

        name = _normalize_name(name_raw, line_number, "name")
        ascii_name = _normalize_name(ascii_name_raw, line_number, "ascii_name")
        if not geoname_raw.isascii() or not geoname_raw.isdigit():
            raise BuildError(f"line {line_number}: geoname_id must be a positive integer")
        geoname_id = int(geoname_raw)
        if geoname_id <= 0:
            raise BuildError(f"line {line_number}: geoname_id must be a positive integer")

        row: dict[str, Any] = {
            "admin1_id": admin1_id,
            "iso": iso,
            "code": code,
            "name": name,
            "ascii_name": ascii_name,
            "geoname_id": geoname_id,
        }

        translation = CN_PROVINCE_TRANSLATIONS.get(admin1_id)
        if translation is not None:
            expected_name, name_zh, aliases = translation
            if name != expected_name:
                raise BuildError(
                    f"line {line_number}: CN translation for {admin1_id} expected "
                    f"name {expected_name!r} from the verified source mapping, got {name!r}"
                )
            row["name_zh"] = name_zh
            row["aliases"] = list(aliases)

        seen_ids.add(admin1_id)
        rows.append(row)

    if not rows:
        raise BuildError("GeoNames input contained no admin1 rows")
    rows.sort(key=lambda row: row["admin1_id"])
    return rows


def _relative_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _catalog_cities(
    source: CatalogDatabase | Path | str | Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, str]:
    """Return city rows, city logical revision and database provenance label."""

    if isinstance(source, CatalogDatabase):
        reader = source
    elif isinstance(source, (list, tuple)):
        return list(source), "", "cities rows"
    else:
        reader = CatalogDatabase(Path(source))
    return reader.cities(), reader.revision("cities"), _relative_label(reader.path)


def build_city_crosswalk(
    database: CatalogDatabase | Path | str | Sequence[dict[str, Any]],
    admin1_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Record matched city/admin1 code pairs using cities stored in SQLite."""

    cities, city_revision, database_label = _catalog_cities(database)
    admin1_ids = {row["admin1_id"] for row in admin1_rows}
    pairs: Counter[tuple[str, str]] = Counter()
    absent: Counter[str] = Counter()

    for position, city in enumerate(cities):
        iso = city.get("iso")
        region = city.get("region")
        if not isinstance(iso, str) or not iso:
            raise BuildError(f"city catalog item {position} has an invalid iso")
        if region is None or (isinstance(region, str) and not region.strip()):
            absent[iso] += 1
        elif isinstance(region, str):
            pairs[(iso, region.strip())] += 1
        else:
            raise BuildError(f"city catalog item {position} has an invalid region")

    matched_count = 0
    unmatched: list[dict[str, Any]] = []
    for (iso, code), city_rows in sorted(pairs.items()):
        admin1_id = f"{iso}.{code}"
        item = {
            "admin1_id": admin1_id,
            "iso": iso,
            "code": code,
            "city_rows": city_rows,
        }
        if admin1_id in admin1_ids:
            matched_count += 1
        else:
            unmatched.append(item)

    absent_rows = [
        {"iso": iso, "code": None, "city_rows": city_rows}
        for iso, city_rows in sorted(absent.items())
    ]
    return {
        "catalog": database_label,
        "database": database_label,
        "source": "cities",
        "logical_revision": city_revision,
        "city_revision": city_revision,
        "city_rows": len(cities),
        "city_rows_with_admin1_code": sum(pairs.values()),
        "city_rows_without_admin1_code": sum(absent.values()),
        "distinct_city_admin1_codes": len(pairs),
        "matched_admin1_codes": matched_count,
        "unmatched_admin1_codes": len(unmatched),
        "unmatched": unmatched,
        "absent": absent_rows,
        "note": (
            "Entries in unmatched are non-null city region codes absent from the "
            "authoritative GeoNames admin1 directory. Entries in absent are city "
            "rows with no region code. Neither case creates a placeholder admin1 row. "
            "The city logical revision is read from SQLite metadata; the database file "
            "itself is never hashed."
        ),
    }


def _dumps_sources(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"


def _database_outputs(
    database_path: Path,
    *,
    revisions: dict[str, str],
    counts: dict[str, int],
) -> dict[str, Any]:
    logical_revisions = {
        name: str(revisions[name]) for name in ("countries", "admin1", "cities")
    }
    record_counts = {
        name: int(counts[name]) for name in ("countries", "admin1", "cities")
    }
    label = _relative_label(database_path)
    detail = {
        "path": label,
        "format": "sqlite",
        "tables": ["geo_catalog_source", "geo_place"],
        "logical_revisions": logical_revisions,
        "record_counts": record_counts,
        "note": (
            "Per-level fingerprints are the stored logical source revisions; "
            "they are never a hash of the SQLite database file."
        ),
    }
    return {
        label: detail,
        "logical_revisions": logical_revisions,
        "record_counts": record_counts,
    }


def build_sources_metadata(
    *,
    source_sha256: str,
    admin1_rows: list[dict[str, Any]],
    crosswalk: dict[str, Any],
    revisions: dict[str, str],
    counts: dict[str, int],
    database_path: Path,
) -> dict[str, Any]:
    iso_codes = sorted({row["iso"] for row in admin1_rows})
    cn_ids = sorted(row["admin1_id"] for row in admin1_rows if row["iso"] == "CN")
    cn_translated_ids = sorted(
        row["admin1_id"] for row in admin1_rows if row["iso"] == "CN" and row.get("name_zh")
    )
    return {
        "schema_version": 1,
        "generated": RETRIEVED_DATE,
        "scope_note": (
            "Full admin1CodesASCII.txt directory. Codes are kept as strings, "
            "including leading zeros. No row is added to satisfy a city region "
            "code that is absent from the authoritative source."
        ),
        "sources": {
            "admin1": {
                "name": "GeoNames admin1CodesASCII",
                "url": "https://download.geonames.org/export/dump/",
                "download_url": ADMIN1_URL,
                "retrieved": RETRIEVED_DATE,
                "license": LICENSE_NAME,
                "license_url": LICENSE_URL,
                "attribution": ATTRIBUTION,
                "sha256": source_sha256,
                "records": len(admin1_rows),
                "coverage_country_codes": iso_codes,
                "note": (
                    "Official GeoNames admin1 code/name/ASCII-name/geoname-id rows. "
                    "CN name_zh values are verified against the code/name pairs in this "
                    "pinned file before being attached."
                ),
            }
        },
        "outputs": _database_outputs(
            database_path,
            revisions=revisions,
            counts=counts,
        ),
        "cn_translations": {
            "admin1_rows": len(cn_ids),
            "translated": len(cn_translated_ids),
            "translated_ids": cn_translated_ids,
            "untranslated_ids": sorted(set(cn_ids) - set(cn_translated_ids)),
        },
        "city_catalog_crosswalk": crosswalk,
        "source_sha256": source_sha256,
        "source_url": ADMIN1_URL,
        "source_retrieved": RETRIEVED_DATE,
        "build_command": (
            "python scripts/update-admin1-catalog.py --database data/geo_catalog.sqlite3"
        ),
        "verify_command": (
            "python scripts/update-admin1-catalog.py --database data/geo_catalog.sqlite3 --check"
        ),
    }


def _write_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = text.encode("utf-8")
    if path.is_file() and path.read_bytes() == encoded:
        print(f"unchanged {path}")
        return
    path.write_bytes(encoded)
    print(f"wrote {path} ({len(encoded)} bytes)")


def _json_text_equal(path: Path, expected_text: str) -> bool:
    if not path.is_file():
        return False
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
        expected = json.loads(expected_text)
    except (OSError, ValueError):
        return False
    return actual == expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="local copy of the official admin1CodesASCII.txt (no download is made)",
    )
    parser.add_argument(
        "--expected-sha256",
        default=ADMIN1_SHA256,
        help="expected SHA-256 of the input; defaults to the pinned 2026-10-12 file",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="require --input and never contact the network",
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare logical admin1 rows/revisions without writing",
    )
    args = parser.parse_args(argv)

    try:
        if args.offline and args.input is None:
            raise BuildError("--offline requires --input")
        payload, source_sha256 = _read_source(args.input, args.expected_sha256)
        admin1_rows = build_admin1_rows(payload)
    except BuildError as exc:
        print(f"admin1 catalog update failed: {exc}", file=sys.stderr)
        return 2

    if args.check:
        try:
            check = check_catalog_levels(args.database, admin1=admin1_rows)
            crosswalk = build_city_crosswalk(args.database, admin1_rows)
            sources_text = _dumps_sources(
                build_sources_metadata(
                    source_sha256=source_sha256,
                    admin1_rows=admin1_rows,
                    crosswalk=crosswalk,
                    revisions=dict(check.plan.revisions),
                    counts=dict(check.plan.counts),
                    database_path=args.database,
                )
            )
        except (CatalogMaintenanceError, CatalogDatabaseError, BuildError) as exc:
            print(f"admin1 catalog check failed: {exc}", file=sys.stderr)
            return 2
        failures = list(check.mismatches)
        if not _json_text_equal(args.sources, sources_text):
            failures.append(str(args.sources))
        if failures:
            print(
                "admin1 catalog check failed; regenerate: " + ", ".join(failures),
                file=sys.stderr,
            )
            return 1
        print(
            "admin1 catalog check ok: "
            f"{check.plan.counts['admin1']} rows, "
            f"matched_city_codes={crosswalk['matched_admin1_codes']}, "
            f"unmatched_city_codes={crosswalk['unmatched_admin1_codes']}"
        )
        return 0

    try:
        result = replace_catalog_levels(args.database, admin1=admin1_rows)
        crosswalk = build_city_crosswalk(args.database, admin1_rows)
        sources_text = _dumps_sources(
            build_sources_metadata(
                source_sha256=source_sha256,
                admin1_rows=admin1_rows,
                crosswalk=crosswalk,
                revisions=dict(result.revisions),
                counts=dict(result.counts),
                database_path=args.database,
            )
        )
    except (CatalogMaintenanceError, CatalogDatabaseError, BuildError) as exc:
        print(f"admin1 catalog update failed: {exc}", file=sys.stderr)
        return 2

    _write_if_changed(args.sources, sources_text)
    print(
        "admin1 catalog database updated: "
        f"rows={result.counts['admin1']} "
        f"cn_translated={sum(1 for row in admin1_rows if row.get('name_zh'))} "
        f"matched_city_codes={crosswalk['matched_admin1_codes']} "
        f"unmatched_city_codes={crosswalk['unmatched_admin1_codes']} "
        f"changed={','.join(result.changed_sources) or 'none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
