"""Build offline Chinese display labels without changing the geographic catalog.

Usage: python scripts/import_city_names_zh.py --archive PATH/alternateNamesV2.zip
Input is the official GeoNames alternateNamesV2 dump (CC BY 4.0). No download
or model inference occurs here. Source SHA256 and coverage are recorded beside
the generated labels for review. City IDs must already exist in our catalog.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCALES = {name: index for index, name in enumerate((
    "zh-cn", "zh-hans", "zh", "zh-sg", "zh-tw", "zh-hant", "zh-hk", "zh-mo", "zh-my",
))}
HAN = re.compile(r"[\u3400-\u9fff]")


def build(archive: Path, catalog: Path, output: Path, retrieved: str) -> dict:
    with sqlite3.connect(catalog.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        ids = {row[0] for row in connection.execute(
            "SELECT place_id FROM geo_place WHERE level='city'"
        )}
    ranked: dict[str, tuple[tuple, str]] = {}
    with zipfile.ZipFile(archive) as bundle, bundle.open("alternateNamesV2.txt") as stream:
        for line in stream:
            raw = line.rstrip(b"\r\n").split(b"\t")
            if len(raw) < 8 or not raw[2].lower().startswith(b"zh"):
                continue
            city_id = raw[1].decode("ascii")
            if city_id not in ids:
                continue
            fields = [value.decode("utf-8") for value in raw]
            locale, name = fields[2].lower(), fields[3].strip()
            if (fields[6] == "1" or fields[7] == "1"
                    or (len(fields) > 9 and fields[9]) or not HAN.search(name)
                    or re.search(r"[A-Za-z\u3040-\u30ff\uac00-\ud7af]", name)):
                continue
            # Language tags are authoritative; never infer Chinese from Kanji.
            rank = (LOCALES.get(locale, len(LOCALES)), fields[4] != "1",
                    fields[5] != "1", len(name), int(fields[0]))
            if city_id not in ranked or rank < ranked[city_id][0]:
                ranked[city_id] = (rank, name)
    names = {key: ranked[key][1] for key in sorted(ranked, key=int)}
    with archive.open("rb") as source:
        source_hash = hashlib.file_digest(source, "sha256").hexdigest()
    metadata = {
        "source": "GeoNames alternateNamesV2",
        "url": "https://download.geonames.org/export/dump/alternateNamesV2.zip",
        "retrieved": retrieved,
        "sha256": source_hash,
        "license": "CC BY 4.0", "attribution": "GeoNames",
        "catalog_city_count": len(ids), "translated_city_count": len(names),
        "selection": "Match catalog GeoNames IDs; exclude historic, colloquial and ended names; "
                     "prefer zh-CN, zh-Hans, zh, then other zh locales; prefer preferred/short names "
                     "then length and alternate-name ID. Display only; catalog identity, coordinates, "
                     "scores and revisions are unchanged.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(names, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".sources.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/geo_catalog.sqlite3")
    parser.add_argument("--output", type=Path, default=ROOT / "data/city-names.zh-CN.json")
    parser.add_argument("--retrieved", default=date.today().isoformat())
    args = parser.parse_args()
    metadata = build(args.archive, args.catalog, args.output, args.retrieved)
    print(f"Chinese names: {metadata['translated_city_count']} / {metadata['catalog_city_count']}")


if __name__ == "__main__":
    main()
