"""Supplement missing city labels from Wikidata P1566; never machine-translate.

Reads public city IDs only, caches successful query batches, and writes a local
presentation dictionary. Existing GeoNames Chinese labels retain priority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://query.wikidata.org/sparql"
LANGUAGES = ("zh-hans", "zh-cn", "zh", "zh-hant", "zh-tw", "zh-hk", "zh-sg", "zh-mo")
USER_AGENT = "m-photo-city-labels/1.0 (offline GeoNames city label import)"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_query(ids: list[str]) -> str:
    values = " ".join(json.dumps(city_id) for city_id in ids)
    languages = ", ".join(json.dumps(language) for language in LANGUAGES)
    return ("SELECT ?city ?id ?label WHERE { VALUES ?id { " + values + " } "
            "?city wdt:P1566 ?id; rdfs:label ?label. FILTER(LANG(?label) IN (" + languages + ")) }")


class RateLimited(urllib.error.URLError):
    pass


def fetch_batch(ids: list[str], cache_dir: Path) -> tuple[dict, str]:
    query = make_query(ids)
    fingerprint = hashlib.sha256(query.encode()).hexdigest()
    cached = cache_dir / (fingerprint + ".json")
    if cached.exists():
        record = json.loads(cached.read_text(encoding="utf-8"))
        if record.get("query") == query and record.get("ids") == ids:
            return record, fingerprint
    cooldown_file = cache_dir / "cooldown.json"
    if cooldown_file.exists():
        cooldown = json.loads(cooldown_file.read_text(encoding="utf-8"))
        if float(cooldown.get("retry_at_unix", 0)) > time.time():
            raise RateLimited("Wikidata Retry-After cooldown is still active")
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={
                "User-Agent": USER_AGENT, "Accept": "application/sparql-results+json",
            })
            with urllib.request.urlopen(request, timeout=45) as response:
                data = json.load(response)
            if not isinstance(data.get("results", {}).get("bindings"), list):
                raise ValueError("Invalid Wikidata query response")
            record = {"ids": ids, "query": query, "response": data,
                      "fetched_at": datetime.now(timezone.utc).isoformat()}
            write_json(cached, record)
            return record, fingerprint
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
                try:
                    retry_after = max(60, int(exc.headers.get("Retry-After", "60")))
                except ValueError:
                    retry_after = 60
                write_json(cooldown_file, {"retry_at_unix": time.time() + retry_after,
                                          "retry_after_seconds": retry_after,
                                          "reason": "Wikidata HTTP 429; stop and resume later"})
                raise RateLimited(f"Wikidata rate limited; retry after {retry_after}s") from exc
            if attempt == 2:
                raise
            delay = 3 * (attempt + 1)
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    delay = min(30, max(delay, int(exc.headers.get("Retry-After", delay))))
                except ValueError:
                    pass
            print(f"Retry {attempt + 1}: {type(exc).__name__}; backoff {delay}s", flush=True)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def select_names(bindings: list[dict], requested: set[str]) -> tuple[dict, set[str]]:
    grouped: dict[str, list[tuple]] = {}
    for row in bindings:
        try:
            city_id = row["id"]["value"]
            entity = row["city"]["value"].rsplit("/", 1)[-1]
            label, language = row["label"]["value"].strip(), row["label"]["xml:lang"].lower()
        except (KeyError, TypeError, AttributeError):
            continue
        if (city_id not in requested or not re.fullmatch(r"Q[1-9][0-9]*", entity)
                or language not in LANGUAGES or not re.search(r"[\u3400-\u9fff]", label)):
            continue
        grouped.setdefault(city_id, []).append((LANGUAGES.index(language), label, entity, language))
    selected, ambiguous = {}, set()
    for city_id, candidates in grouped.items():
        if len({row[2] for row in candidates}) != 1:
            ambiguous.add(city_id)
            continue
        _, label, entity, language = min(candidates)
        selected[city_id] = {"label": label, "entity_id": entity, "language": language}
    return selected, ambiguous


def run(args: argparse.Namespace) -> None:
    existing_path = ROOT / "data/city-names.zh-CN.json"
    existing = json.loads(existing_path.read_text(encoding="utf-8"))
    with sqlite3.connect(args.catalog.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        all_ids = [str(row[0]) for row in connection.execute(
            "SELECT place_id FROM geo_place WHERE level='city' ORDER BY population DESC, geoname_id ASC"
        )]
    missing = [city_id for city_id in all_ids if city_id not in existing]
    selected, ambiguous, failed, batches = {}, set(), [], []
    checked = 0
    rate_limited = False
    total_batches = (len(missing) + args.batch_size - 1) // args.batch_size
    output = ROOT / "data/city-names.wikidata.zh.json"
    sources = ROOT / "data/city-names.wikidata.zh.sources.json"
    for offset in range(0, len(missing), args.batch_size):
        ids = missing[offset:offset + args.batch_size]
        fingerprint = hashlib.sha256(make_query(ids).encode()).hexdigest()
        if args.offline and not (args.cache_dir / (fingerprint + ".json")).exists():
            continue
        try:
            record, fingerprint = fetch_batch(ids, args.cache_dir)
            found, duplicate_ids = select_names(record["response"]["results"]["bindings"], set(ids))
            selected.update(found)
            ambiguous.update(duplicate_ids)
            checked += len(ids)
            batches.append({"query_sha256": fingerprint, "fetched_at": record.get("fetched_at")})
            print(f"Batch {offset // args.batch_size + 1}/{total_batches}: "
                  f"checked={checked}, added={len(selected)}, ambiguous={len(ambiguous)}", flush=True)
        except RateLimited as exc:
            rate_limited = True
            print(str(exc), flush=True)
            break
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            failed.append({"batch": offset // args.batch_size + 1, "error": type(exc).__name__})
        if not args.offline and offset + len(ids) < len(missing):
            time.sleep(1)
    names = {key: selected[key]["label"] for key in sorted(selected, key=int)}
    # A resumed/limited run must never discard labels from successful cached batches.
    if output.exists():
        previous_names = json.loads(output.read_text(encoding="utf-8"))
        if not set(previous_names).issubset(names):
            raise SystemExit("Refusing to discard existing labels; rebuild all cached batches with --offline")
    write_json(output, names)
    cooldown_file = args.cache_dir / "cooldown.json"
    cooldown = json.loads(cooldown_file.read_text(encoding="utf-8")) if cooldown_file.exists() else {}
    cooldown_active = float(cooldown.get("retry_at_unix", 0)) > time.time()
    write_json(sources, {
        "source": "Wikidata", "endpoint": ENDPOINT, "property": "P1566", "license": "CC0",
        "generated_at": datetime.now(timezone.utc).isoformat(), "catalog_city_count": len(all_ids),
        "existing_chinese_count": len(existing), "queried_missing_city_count": checked,
        "pending_city_count": len(missing) - checked,
        "supplemental_chinese_count": len(names), "ambiguous_ids": sorted(ambiguous, key=int),
        "failed_batches": failed, "rate_limited": rate_limited or cooldown_active,
        "retry_after_unix": cooldown.get("retry_at_unix") if cooldown_active else None,
        "complete": checked == len(missing) and not failed,
        "fallback": "original catalog name; no machine translation",
        "languages": LANGUAGES, "batches": batches, "labels": selected,
    })
    print(f"Saved {len(names)} supplemental Chinese labels; pending={len(missing) - checked}", flush=True)
    if failed or rate_limited:
        raise SystemExit("Partial import saved. Resume later; successful queries are cached.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/geo_catalog.sqlite3")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/geo-catalog/wikidata-zh")
    parser.add_argument("--batch-size", type=int, choices=range(1, 501), default=500)
    parser.add_argument("--offline", action="store_true", help="Rebuild outputs from cached successful queries only")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
