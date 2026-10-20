#!/usr/bin/env python3
"""Offline geolocation accuracy evaluation for m-photo.

Runs the same chain used by the backend upload flow:

    OCR + observation analyzers -> StreetCLIP country prior -> clue fusion
    -> optional city ranking (true country and end-to-end predicted country)

against a labeled JSONL manifest and writes a reproducible report. The backend
HTTP service is not required; the local StreetCLIP inference process is
required when ``STREETCLIP_ENABLED=true`` (and can be waived with
``--allow-degraded`` to measure the rules-only fallback).

Examples
--------
Host run (StreetCLIP started with ``scripts/start-streetclip.ps1``)::

    .venv\\Scripts\\python.exe scripts\\eval_geolocation.py ^
        --manifest data\\eval\\manifest.jsonl

Quick plumbing check on a subset, without either city path::

    ... --limit 5 --no-cities --no-e2e-cities

Rules-only degradation baseline::

    ... --allow-degraded --no-cities --no-e2e-cities

Exit codes: 0 report written; 1 gate failed or ``--fail-fast`` stopped;
2 configuration, manifest or preflight failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402
from PIL import Image  # noqa: E402

from backend.app.config import Settings  # noqa: E402
from backend.app.domain.geolocation import knowledge_graph as kg  # noqa: E402
from backend.app.domain.geolocation.clue import dicts_to_clues  # noqa: E402
from backend.app.domain.geolocation.country_policy import canonical_country_iso  # noqa: E402
from backend.app.domain.geolocation.model_ranking import ModelAwareEngine  # noqa: E402
from backend.app.services.city_catalog import load_city_index  # noqa: E402
from backend.app.services.city_reasoning import CityClueReasoner  # noqa: E402
from backend.app.services.model_gateway import (  # noqa: E402
    PROMPT_VERSION,
    StreetClipGateway,
    catalog_revision as semantic_catalog_revision,
    clean_image,
)
from backend.app.services.pipeline import PipelineService  # noqa: E402
from catalog_db import CatalogDatabase  # noqa: E402

try:  # script execution vs. ``import scripts.eval_geolocation`` in tests
    from .eval_manifest import ManifestEntry, ManifestError, load_env_file, load_manifest
    from .eval_metrics import (
        haversine_km,
        rank_of,
        summarize_city,
        summarize_country,
        summarize_latency,
    )
except ImportError:  # pragma: no cover - exercised by direct CLI runs
    from eval_manifest import ManifestEntry, ManifestError, load_env_file, load_manifest
    from eval_metrics import (
        haversine_km,
        rank_of,
        summarize_city,
        summarize_country,
        summarize_latency,
    )

MAX_COUNTRY_RANK = 1000
SCHEMA_VERSION = 1


class EvalError(RuntimeError):
    """A configuration or preflight problem that makes the run meaningless."""


# ---------------------------------------------------------------------------
# Configuration and preflight
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rewrite_container_host(url: str) -> str:
    """Prefer loopback when a host-side run inherits a container URL."""

    parts = urlsplit(url)
    if parts.hostname != "host.docker.internal":
        return url
    netloc = "127.0.0.1" + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _build_settings(args: argparse.Namespace, catalog_path: Path) -> Settings:
    dotenv = load_env_file(Path(args.env_file)) if args.env_file else {}
    env = {**dotenv, **os.environ}
    # The evaluation is read-only: point the gateway at the immutable seed (or
    # an explicitly supplied catalog) and drop aliases that could silently
    # select a different database.
    for key in (
        "M_PHOTO_DB_PATH",
        "DATABASE_PATH",
        "M_PHOTO_DATABASE_PATH",
        "M_PHOTO_DATABASE_URL",
        "DATABASE_URL",
    ):
        env.pop(key, None)
    env["M_PHOTO_DB_URL"] = f"sqlite:///{catalog_path.as_posix()}"

    if args.ocr_provider:
        env["OCR_PROVIDER"] = args.ocr_provider
    if args.ocr_model_type:
        env["OCR_MODEL_TYPE"] = args.ocr_model_type
    if args.ai_mode:
        env["GEO_AI_MODE"] = args.ai_mode

    url = args.streetclip_url or env.get("STREETCLIP_URL") or "http://127.0.0.1:8001"
    if not args.container and not args.streetclip_url:
        url = _rewrite_container_host(url)
    env["STREETCLIP_URL"] = url.rstrip("/")
    return Settings.from_env(env)


def _headers(settings: Settings) -> dict[str, str]:
    return (
        {"Authorization": f"Bearer {settings.streetclip_api_key}"}
        if settings.streetclip_api_key
        else {}
    )


def _probe_streetclip(settings: Settings, *, require_city: bool) -> dict[str, Any]:
    """Verify readiness, auth and the backend contract before scoring samples.

    ``StreetClipGateway.rank_image`` deliberately degrades every failure to
    ``None``. For evaluation that silence is unacceptable: a misconfigured run
    would look like a bad model. This probe therefore reports the concrete
    failure (HTTP status or contract error) once, up front.
    """

    if not settings.streetclip_enabled:
        raise EvalError(
            "STREETCLIP_ENABLED is false; a model evaluation would silently "
            "measure the rules-only fallback. Start StreetCLIP and set "
            "STREETCLIP_ENABLED=true, or pass --allow-degraded to record that "
            "baseline explicitly."
        )

    base = settings.streetclip_url.rstrip("/")
    timeout = max(5.0, min(float(settings.streetclip_timeout_seconds), 30.0))
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False) as client:
            health_response = client.get(base + "/health", headers=_headers(settings))
            if health_response.status_code != 200:
                raise EvalError(
                    f"StreetCLIP /health returned HTTP {health_response.status_code}: "
                    f"{health_response.text[:200]}"
                )
            health = health_response.json()
            if not isinstance(health, dict) or not health.get("ready"):
                raise EvalError(
                    "StreetCLIP is reachable but reports ready=false; inspect "
                    "inference/streetclip/storage/server.stderr.log"
                )
            with tempfile.TemporaryDirectory(prefix="m-photo-eval-") as tmp:
                probe = Path(tmp) / "probe.jpg"
                Image.new("RGB", (64, 64), (127, 127, 127)).save(probe, format="JPEG")
                response = client.post(
                    base + "/predict",
                    headers=_headers(settings),
                    files={"file": ("probe.jpg", clean_image(probe), "image/jpeg")},
                )
            if response.status_code != 200:
                raise EvalError(
                    f"StreetCLIP /predict returned HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
            gateway = StreetClipGateway(settings)
            gateway.validate(response.json())
            return health
    except EvalError:
        raise
    except httpx.HTTPError as exc:
        raise EvalError(
            f"cannot reach StreetCLIP at {base} ({type(exc).__name__}). "
            "Start it with scripts/start-streetclip.ps1, or pass --allow-degraded."
        ) from exc
    except Exception as exc:  # validation errors from gateway.validate
        raise EvalError(
            f"StreetCLIP /predict response failed backend validation: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------


def _canonical_gt(entry: ManifestEntry) -> str | None:
    if entry.country_iso is None:
        return None
    return canonical_country_iso(entry.country_iso) or entry.country_iso


def _predicted_source_iso(scores: Sequence[Any]) -> str | None:
    """Raw catalog ISO to use for end-to-end city ranking.

    Grouped candidates (CN/HK/MO/TW) expose their members in
    ``source_regions``; choose the highest-scoring member so the city call asks
    the same source scope that produced the winning evidence.
    """

    if not scores:
        return None
    top = scores[0]
    regions = getattr(top, "source_regions", None)
    if regions:
        best = max(
            regions,
            key=lambda row: (float(row.get("score") or 0.0), str(row.get("source_iso") or "")),
        )
        return str(best.get("source_iso") or top.iso)
    return str(top.iso)


def _country_result(
    scores: Sequence[Any], gt_iso: str | None, evidence: dict | None
) -> dict[str, Any]:
    predicted = [str(score.iso) for score in scores]
    margin = None
    if len(scores) >= 2:
        margin = round(float(scores[0].score) - float(scores[1].score), 6)
    return {
        "labeled": gt_iso is not None,
        "rank": rank_of(predicted, gt_iso) if gt_iso is not None else None,
        "predicted_iso": predicted,
        "top1_score": round(float(scores[0].score), 6) if scores else None,
        "margin": margin,
        "model_margin": evidence.get("margin") if evidence else None,
        "model_uncertain": evidence.get("uncertain") if evidence else None,
        "degraded": evidence is None,
        "top5": [score.to_dict() for score in scores[:5]],
    }


def _city_row_summary(row: dict, gt_lat: float, gt_lng: float) -> dict[str, Any]:
    lat = float(row["lat"])
    lng = float(row["lng"])
    return {
        "city_id": str(row.get("city_id")),
        "name": row.get("name"),
        "region": row.get("region"),
        "score": round(float(row.get("score") or 0.0), 6),
        "raw_cosine": row.get("raw_cosine"),
        "km": round(haversine_km(gt_lat, gt_lng, lat, lng), 3),
    }


def _evaluate_city(
    payload: dict | None,
    entry: ManifestEntry,
    clues: Sequence[Any],
    reasoner: CityClueReasoner,
    *,
    iso: str,
) -> dict[str, Any]:
    if payload is None:
        return {"source_iso": iso, "available": False, "error": "city ranking unavailable"}
    rows = list(payload.get("city_scores") or [])
    if not rows:
        return {
            "source_iso": iso,
            "available": False,
            "status": payload.get("status"),
            "error": "model returned no city candidates",
        }
    try:
        reranked = reasoner.rerank(rows, clues, iso=iso)
    except Exception as exc:
        return {
            "source_iso": iso,
            "available": False,
            "status": payload.get("status"),
            "error": f"clue rerank failed: {type(exc).__name__}: {exc}",
        }
    rows = list(reranked.city_scores)
    if not rows:
        return {
            "source_iso": iso,
            "available": False,
            "status": payload.get("status"),
            "error": "no rows after clue rerank",
        }

    summaries = [_city_row_summary(row, entry.lat, entry.lng) for row in rows]
    distances = [row["km"] for row in summaries]
    gt_city_id = str(entry.city_id) if entry.city_id else None
    rank = None
    if gt_city_id:
        for index, row in enumerate(summaries, start=1):
            if str(row.get("city_id")) == gt_city_id:
                rank = index
                break
    return {
        "source_iso": iso,
        "available": True,
        "status": payload.get("status"),
        "rank": rank,
        "exact_top1": (rank == 1) if gt_city_id else None,
        "top1_km": distances[0],
        "top3_min_km": min(distances[:3]),
        "top1_city": summaries[0],
        "top3": summaries[:3],
        "clue_dimensions": list(reranked.clue_dimensions),
        "coverage_note": reranked.coverage_note,
    }


def _evaluate_sample(
    entry: ManifestEntry,
    *,
    pipeline: PipelineService,
    gateway: StreetClipGateway,
    engine: ModelAwareEngine,
    reasoner: CityClueReasoner | None,
    evaluate_cities: bool,
    evaluate_e2e_cities: bool,
    city_path_error: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": entry.id,
        "image": entry.image_ref,
        "ground_truth": {
            "country_iso": entry.country_iso,
            "canonical_country_iso": _canonical_gt(entry),
            "city_id": entry.city_id,
            "city": entry.city,
            "admin1_code": entry.admin1_code,
            "lat": entry.lat,
            "lng": entry.lng,
        },
        "timing_ms": {},
    }
    timings: dict[str, float] = record["timing_ms"]
    started = time.perf_counter()
    oracle_skipped = evaluate_cities and entry.country_iso is None
    if oracle_skipped:
        record["city_true_country"] = {
            "available": False,
            "skipped": True,
            "reason": "no_country_label",
        }

    try:
        tick = time.perf_counter()
        clue_dicts = pipeline.analyze(str(entry.image_path))
        timings["analyze_ms"] = round((time.perf_counter() - tick) * 1000.0, 1)
        ocr_status = pipeline.ocr_status()
        clues = dicts_to_clues(clue_dicts)
        record["ocr"] = {
            "provider": ocr_status.get("provider"),
            "state": ocr_status.get("state"),
            "available": bool(ocr_status.get("available")),
            "line_count": ocr_status.get("line_count"),
        }
        record["clues"] = {
            "count": len(clue_dicts),
            "dimensions": dict(
                sorted(Counter(str(clue.get("dimension") or "unknown") for clue in clue_dicts).items())
            ),
        }

        tick = time.perf_counter()
        evidence = gateway.rank_image(entry.image_path)
        timings["country_model_ms"] = round((time.perf_counter() - tick) * 1000.0, 1)

        tick = time.perf_counter()
        engine.evidence = evidence
        scores = engine.infer(clues, top_n=MAX_COUNTRY_RANK)
        timings["fusion_ms"] = round((time.perf_counter() - tick) * 1000.0, 1)
        record["country"] = _country_result(scores, _canonical_gt(entry), evidence)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        timings["total_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        return record

    needs_city_path = evaluate_e2e_cities or (evaluate_cities and not oracle_skipped)
    if reasoner is None and needs_city_path:
        record["city_error"] = city_path_error or "city catalog unavailable"
        timings["total_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        return record

    if reasoner is not None and (evaluate_cities or evaluate_e2e_cities):
        true_city: dict[str, Any] | None = None
        true_iso: str | None = None
        if evaluate_cities and not oracle_skipped:
            true_iso = entry.city_source_iso or entry.country_iso
            tick = time.perf_counter()
            try:
                payload = gateway.rank_cities(entry.image_path, true_iso)
                true_city = _evaluate_city(payload, entry, clues, reasoner, iso=true_iso)
            except Exception as exc:
                true_city = {
                    "source_iso": true_iso,
                    "available": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            timings["cities_true_ms"] = round((time.perf_counter() - tick) * 1000.0, 1)
            record["city_true_country"] = true_city

        if evaluate_e2e_cities:
            predicted_iso = _predicted_source_iso(scores)
            # ``predicted_iso`` may be a raw member scope (TW/HK/MO) that is
            # not the manifest's country; the city call always uses it as-is.
            if not predicted_iso:
                record["city_e2e"] = {"available": False, "error": "no country prediction"}
            elif true_city is not None and predicted_iso == true_iso:
                record["city_e2e"] = {**true_city, "reused_true_country_ranking": True}
            else:
                tick = time.perf_counter()
                try:
                    payload = gateway.rank_cities(entry.image_path, predicted_iso)
                    record["city_e2e"] = _evaluate_city(
                        payload, entry, clues, reasoner, iso=predicted_iso
                    )
                except Exception as exc:
                    record["city_e2e"] = {
                        "source_iso": predicted_iso,
                        "available": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                timings["cities_e2e_ms"] = round((time.perf_counter() - tick) * 1000.0, 1)

    timings["total_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    return record


# ---------------------------------------------------------------------------
# Aggregation and rendering
# ---------------------------------------------------------------------------


def _ocr_summary(records: Sequence[dict]) -> dict[str, Any]:
    ocr_rows = [record["ocr"] for record in records if isinstance(record.get("ocr"), dict)]
    clue_counts = [
        int(record["clues"]["count"])
        for record in records
        if isinstance(record.get("clues"), dict)
    ]
    dimensions: Counter[str] = Counter()
    for record in records:
        clues = record.get("clues")
        if isinstance(clues, dict) and isinstance(clues.get("dimensions"), dict):
            dimensions.update(clues["dimensions"])
    available = [bool(row.get("available")) for row in ocr_rows]
    return {
        "samples": len(ocr_rows),
        "available_rate": (sum(available) / len(available)) if available else None,
        "states": dict(sorted(Counter(str(row.get("state")) for row in ocr_rows).items())),
        "mean_lines": (sum(row.get("line_count") or 0 for row in ocr_rows) / len(ocr_rows))
        if ocr_rows
        else None,
        "mean_clues": (sum(clue_counts) / len(clue_counts)) if clue_counts else None,
        "clue_dimensions": dict(sorted(dimensions.items())),
    }


def _latency_summary(records: Sequence[dict], stage: str) -> dict[str, Any]:
    return summarize_latency(
        [
            record["timing_ms"][stage]
            for record in records
            if isinstance(record.get("timing_ms"), dict)
            and isinstance(record["timing_ms"].get(stage), (int, float))
        ]
    )


def _aggregate(
    records: Sequence[dict],
    *,
    evaluate_cities: bool | None = None,
    evaluate_e2e_cities: bool | None = None,
) -> dict[str, Any]:
    """Aggregate attempted records without hiding failed samples.

    Country metrics use only attempted samples with a country label. Country
    errors still count as misses there; unlabeled samples remain in e2e city,
    global error/degraded, and strict checks. City metrics are
    available-conditional, with an overall block over attempted city paths.
    """

    if evaluate_cities is None:
        evaluate_cities = any("city_true_country" in record for record in records)
    if evaluate_e2e_cities is None:
        evaluate_e2e_cities = any("city_e2e" in record for record in records)

    def country_labeled(record: dict) -> bool:
        ground_truth = record.get("ground_truth")
        if isinstance(ground_truth, dict) and "country_iso" in ground_truth:
            return ground_truth.get("country_iso") not in (None, "")
        country_row = record.get("country")
        return not (isinstance(country_row, dict) and country_row.get("labeled") is False)

    labeled_records = [record for record in records if country_labeled(record)]
    unlabeled_records = [record for record in records if not country_labeled(record)]
    country_records = [
        record["country"] for record in labeled_records if isinstance(record.get("country"), dict)
    ]
    country = summarize_country(country_records)
    country["overall"] = summarize_country(
        [
            record["country"]
            if isinstance(record.get("country"), dict)
            else {"rank": None, "degraded": False}
            for record in labeled_records
        ]
    )
    country["labeled_samples"] = len(labeled_records)
    country["unlabeled_samples"] = len(unlabeled_records)
    country["label_coverage"] = (
        round(len(labeled_records) / len(records), 6) if records else None
    )

    def city_path_skipped(record: dict, field: str) -> bool:
        row = record.get(field)
        return (
            field == "city_true_country"
            and isinstance(row, dict)
            and bool(row.get("skipped"))
            and row.get("reason") == "no_country_label"
        )

    def attempted_city_rows(field: str, enabled: bool) -> list[dict]:
        if not enabled:
            return []
        rows: list[dict] = []
        for record in records:
            if city_path_skipped(record, field):
                continue
            row = record.get(field)
            if isinstance(row, dict):
                rows.append(row)
            else:
                rows.append(
                    {
                        "available": False,
                        "error": record.get("city_error") or "city path not evaluated",
                    }
                )
        return rows

    true_city_records = attempted_city_rows("city_true_country", evaluate_cities)
    e2e_records = attempted_city_rows("city_e2e", evaluate_e2e_cities)

    country_top1_records: list[dict] = []
    if evaluate_e2e_cities:
        for record in records:
            country_row = record.get("country")
            if not isinstance(country_row, dict) or country_row.get("rank") != 1:
                continue
            city_row = record.get("city_e2e")
            if isinstance(city_row, dict):
                country_top1_records.append(city_row)
            else:
                country_top1_records.append(
                    {
                        "available": False,
                        "error": record.get("city_error") or "city path not evaluated",
                    }
                )

    city_true = summarize_city(true_city_records) if evaluate_cities else None
    city_e2e = summarize_city(e2e_records) if evaluate_e2e_cities else None
    conditional = (
        summarize_city(country_top1_records)
        if evaluate_e2e_cities and country_top1_records
        else None
    )

    latencies = {
        stage: _latency_summary(records, stage)
        for stage in (
            "analyze_ms",
            "country_model_ms",
            "fusion_ms",
            "cities_true_ms",
            "cities_e2e_ms",
            "total_ms",
        )
    }
    errors = [record for record in records if record.get("error")]
    degraded = [record for record in records if (record.get("country") or {}).get("degraded")]

    enabled_city_paths = (
        (("true_country", "city_true_country"),) if evaluate_cities else ()
    ) + ((("e2e", "city_e2e"),) if evaluate_e2e_cities else ())

    def path_unavailable(record: dict, field: str) -> bool:
        if city_path_skipped(record, field):
            return False
        row = record.get(field)
        return not isinstance(row, dict) or not row.get("available")

    unavailable_city = [
        record
        for record in records
        if any(path_unavailable(record, field) for _, field in enabled_city_paths)
    ]
    city_errors: list[dict[str, str]] = []
    for record in records:
        if not enabled_city_paths:
            continue
        setup_error = record.get("city_error")
        if setup_error:
            active_paths = [
                (path, field)
                for path, field in enabled_city_paths
                if not city_path_skipped(record, field)
            ]
            if not active_paths:
                continue
            city_errors.append(
                {"id": str(record.get("id")), "path": "setup", "error": str(setup_error)}
            )
            continue
        for path, field in enabled_city_paths:
            if city_path_skipped(record, field):
                continue
            row = record.get(field)
            if isinstance(row, dict) and row.get("available"):
                continue
            error = (
                row.get("error")
                if isinstance(row, dict) and row.get("error")
                else "city path unavailable"
            )
            city_errors.append(
                {"id": str(record.get("id")), "path": path, "error": str(error)}
            )

    return {
        "summary": {
            "samples": len(records),
            "errors": len(errors),
            "degraded": len(degraded),
            "country": country,
            "city": {
                "true_country": city_true,
                "e2e": city_e2e,
                "e2e_given_country_top1": conditional,
            },
            "ocr": _ocr_summary(records),
            "latency_ms": latencies,
        },
        "errors": errors,
        "degraded_samples": degraded,
        "unavailable_city_samples": unavailable_city,
        "city_errors": city_errors,
    }


def _fmt_rate(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100.0:.1f}%"


def _fmt_interval(value: Any) -> str:
    if not value or len(value) != 2:
        return "—"
    return f"{float(value[0]) * 100.0:.1f}%–{float(value[1]) * 100.0:.1f}%"


def _fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    country = summary["country"]
    city = summary["city"]
    lines: list[str] = []
    lines.append("# Geolocation evaluation")
    lines.append("")
    lines.append(f"- Generated: `{report['generated_at']}`")
    if report.get("label"):
        lines.append(f"- Label: `{report['label']}`")
    lines.append(f"- Manifest: `{report['manifest']['path']}` "
                 f"({report['manifest']['samples']} samples, sha256 `{report['manifest']['sha256'][:12]}…`)")
    lines.append(f"- Image root: `{report['manifest']['image_root']}`")
    run = report.get("run") or {}
    if run.get("partial"):
        lines.append(
            f"- Partial run: `{run.get('completed_samples')}` of "
            f"`{run.get('selected_samples')}` selected samples; "
            f"stopped at `{run.get('fail_fast_sample')}`"
        )
    lines.append("")

    lines.append("## Configuration")
    lines.append("")
    config = report["configuration"]
    lines.append("| item | value |")
    lines.append("| --- | --- |")
    for key in (
        "streetclip_enabled",
        "streetclip_url",
        "model_revision",
        "prompt_version",
        "catalog_revision",
        "city_catalog_revision",
        "allow_unlabeled_country",
        "ocr_provider",
        "ocr_model_type",
        "geo_ai_mode",
    ):
        lines.append(f"| {key} | `{config.get(key)}` |")
    lines.append("")

    gates = report.get("gates") or {}
    lines.append("## Gates")
    lines.append("")
    lines.append("| gate | observed | threshold | result |")
    lines.append("| --- | --- | --- | --- |")
    failures = gates.get("failures") or []
    strict_failures = gates.get("strict_failures") or []
    strict_result = (
        "off"
        if not gates.get("strict")
        else ("pass" if not strict_failures else "fail")
    )
    lines.append(
        f"| strict | {len(strict_failures)} failure(s) | enabled/disabled | "
        f"{strict_result} |"
    )
    threshold = gates.get("min_country_top1")
    observed = gates.get("country_top1_overall")
    if threshold is None:
        minimum_result = "off"
        threshold_display = "—"
    else:
        minimum_result = (
            "pass" if gates.get("min_country_top1_passed") else "fail"
        )
        threshold_display = _fmt_rate(threshold)
    lines.append(
        f"| min-country-top1 (all attempted) | {_fmt_rate(observed)} | "
        f"{threshold_display} | {minimum_result} |"
    )
    if failures:
        lines.append("")
        lines.append("Failures: " + "; ".join(str(item) for item in failures))
    lines.append("")

    lines.append("## Country ranking")
    lines.append("")
    overall = country.get("overall") or {}
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append(
        f"| country labels / unlabeled | {country.get('labeled_samples')} / "
        f"{country.get('unlabeled_samples')} |"
    )
    lines.append(f"| country label coverage | {_fmt_rate(country.get('label_coverage'))} |")
    lines.append(f"| country-labeled attempted samples | {overall.get('samples')} |")
    lines.append(f"| country ranking available | {country.get('samples')} |")
    lines.append(f"| top-1 (available) | {_fmt_rate(country.get('top1'))} |")
    lines.append(f"| top-1 (all attempted) | {_fmt_rate(overall.get('top1'))} |")
    lines.append(f"| top-1 95% CI (all attempted) | {_fmt_interval(overall.get('top1_ci95'))} |")
    lines.append(f"| top-3 (available) | {_fmt_rate(country.get('top3'))} |")
    lines.append(f"| top-3 (all attempted) | {_fmt_rate(overall.get('top3'))} |")
    lines.append(f"| top-5 (available) | {_fmt_rate(country.get('top5'))} |")
    lines.append(f"| top-5 (all attempted) | {_fmt_rate(overall.get('top5'))} |")
    lines.append(f"| MRR (available) | {_fmt_num(country.get('mrr'))} |")
    lines.append(f"| MRR (all attempted) | {_fmt_num(overall.get('mrr'))} |")
    lines.append(f"| unranked (available denominator) | {country.get('unranked')} |")
    lines.append(f"| unranked (all attempted denominator) | {overall.get('unranked')} |")
    lines.append(f"| degraded (rules only, available) | {_fmt_rate(country.get('degraded_rate'))} |")
    lines.append(
        f"| degraded (rules only, country-labeled attempted) | "
        f"{_fmt_rate(overall.get('degraded_rate'))} |"
    )
    lines.append(f"| mean top-1 score (available) | {_fmt_num(country.get('mean_top1_score'))} |")
    lines.append(f"| mean top1-top2 margin (available) | {_fmt_num(country.get('mean_margin'))} |")
    lines.append("")
    lines.append("### Accuracy at coverage")
    lines.append("")
    lines.append(
        "| minimum margin | available coverage | available accuracy | "
        "all-attempted coverage | all-attempted accuracy | available samples |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    overall_coverage = {
        round(float(row.get("margin_min") or 0.0), 6): row
        for row in overall.get("coverage") or []
    }
    for row in country.get("coverage") or []:
        threshold = round(float(row.get("margin_min") or 0.0), 6)
        attempted_row = overall_coverage.get(threshold) or {}
        lines.append(
            f"| {_fmt_num(row.get('margin_min'), 4)} | {_fmt_rate(row.get('coverage'))} | "
            f"{_fmt_rate(row.get('accuracy'))} | {_fmt_rate(attempted_row.get('coverage'))} | "
            f"{_fmt_rate(attempted_row.get('accuracy'))} | {row.get('samples')} |"
        )
    lines.append("")

    lines.append("## City ranking")
    lines.append("")
    lines.append("| metric | true country | end-to-end | e2e \\| country top-1 correct |")
    lines.append("| --- | ---: | ---: | ---: |")
    true_city = city.get("true_country") or {}
    e2e_city = city.get("e2e") or {}
    conditional = city.get("e2e_given_country_top1") or {}

    def row(
        label: str,
        key: str,
        formatter=_fmt_rate,
        *,
        overall: bool = False,
    ) -> str:
        def value(summary_row: dict) -> Any:
            source = (summary_row.get("overall") or {}) if overall else summary_row
            return source.get(key)

        return (
            f"| {label} | {formatter(value(true_city))} | "
            f"{formatter(value(e2e_city))} | {formatter(value(conditional))} |"
        )

    lines.append(row("attempted samples", "samples", lambda value: _fmt_num(value, 0)))
    lines.append(row("available", "available", lambda value: _fmt_num(value, 0)))
    lines.append(row("available rate", "available_rate"))
    lines.append(row("top-1 exact city id (available)", "top1_exact"))
    lines.append(row("top-1 ≤ 25 km (available)", "top1_at_25km"))
    lines.append(row("top-1 ≤ 50 km (available)", "top1_at_50km"))
    lines.append(row("top-1 ≤ 50 km 95% CI (available)", "top1_at_50km_ci95", _fmt_interval))
    lines.append(row("top-1 ≤ 100 km (available)", "top1_at_100km"))
    lines.append(row("top-1 ≤ 25 km (all attempted)", "top1_at_25km", overall=True))
    lines.append(row("top-1 ≤ 50 km (all attempted)", "top1_at_50km", overall=True))
    lines.append(
        row("top-1 ≤ 50 km 95% CI (all attempted)", "top1_at_50km_ci95", _fmt_interval, overall=True)
    )
    lines.append(row("top-1 ≤ 100 km (all attempted)", "top1_at_100km", overall=True))
    lines.append(row("top-3 ≤ 50 km (available)", "top3_at_50km"))
    lines.append(row("top-3 ≤ 50 km 95% CI (available)", "top3_at_50km_ci95", _fmt_interval))
    lines.append(row("top-3 ≤ 50 km (all attempted)", "top3_at_50km", overall=True))
    lines.append(row("median top-1 distance (km, available)", "median_top1_km",
                     lambda value: _fmt_num(value, 1)))
    lines.append(row("p90 top-1 distance (km, available)", "p90_top1_km",
                     lambda value: _fmt_num(value, 1)))
    lines.append("")
    if true_city.get("distance_buckets"):
        lines.append("Top-1 distance distribution (true country):")
        lines.append("")
        lines.append("| bucket | count |")
        lines.append("| --- | ---: |")
        for bucket, count in true_city["distance_buckets"].items():
            lines.append(f"| {bucket} | {count} |")
        lines.append("")

    lines.append("## Latency (ms)")
    lines.append("")
    lines.append("| stage | p50 | p95 | mean | samples |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for stage, values in summary["latency_ms"].items():
        lines.append(
            f"| {stage} | {_fmt_num(values.get('p50'), 1)} | "
            f"{_fmt_num(values.get('p95'), 1)} | {_fmt_num(values.get('mean'), 1)} | "
            f"{values.get('samples')} |"
        )
    lines.append("")

    ocr = summary["ocr"]
    lines.append("## OCR / clue coverage")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append(f"| OCR available | {_fmt_rate(ocr.get('available_rate'))} |")
    lines.append(f"| mean OCR lines | {_fmt_num(ocr.get('mean_lines'), 2)} |")
    lines.append(f"| mean clues | {_fmt_num(ocr.get('mean_clues'), 2)} |")
    lines.append(f"| states | `{json.dumps(ocr.get('states'), ensure_ascii=False)}` |")
    lines.append(f"| dimensions | `{json.dumps(ocr.get('clue_dimensions'), ensure_ascii=False)}` |")
    lines.append("")

    errors = report.get("errors") or []
    if errors:
        lines.append("## Failed samples")
        lines.append("")
        for record in errors[:20]:
            lines.append(f"- `{record['id']}`: {record['error']}")
        if len(errors) > 20:
            lines.append(f"- … {len(errors) - 20} more")
        lines.append("")

    degraded_ids = report.get("degraded_sample_ids") or []
    if degraded_ids:
        lines.append("## Rules-only samples")
        lines.append("")
        for sample_id in degraded_ids[:20]:
            lines.append(f"- `{sample_id}`: StreetCLIP evidence missing")
        if len(degraded_ids) > 20:
            lines.append(f"- … {len(degraded_ids) - 20} more")
        lines.append("")

    city_errors = report.get("city_errors") or []
    if city_errors:
        lines.append("## City ranking errors")
        lines.append("")
        for row in city_errors[:20]:
            path = row.get("path")
            prefix = f" [{path}]" if path else ""
            lines.append(f"- `{row['id']}`{prefix}: {row['error']}")
        if len(city_errors) > 20:
            lines.append(f"- … {len(city_errors) - 20} more")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Scores are cosine-like similarities, not probabilities. "
                 "Country calibration is reported as accuracy at margin-based coverage.")
    lines.append("- City distances compare the labeled photo point with the catalog's "
                 "representative city point, so they include catalog discretisation error.")
    lines.append("- `true country` is an oracle-country upper bound; `end-to-end` uses "
                 "the fused top-1 country and its highest-scoring source region.")
    lines.append("- Samples with `degraded=true` ran without StreetCLIP evidence and "
                 "must not be compared against model-enabled baselines.")
    lines.append("- Existing country/city rates are conditioned on available rankings. "
                 "Country `overall` rates use country-labeled attempted samples; "
                 "unlabeled samples are excluded from country accuracy and oracle "
                 "city metrics but remain in end-to-end city and global error/degraded "
                 "checks. Errors and unavailable paths count as misses where attempted; "
                 "country mean score/margin remain available-only, and optional "
                 "exact-city IDs remain available-conditional only.")
    lines.append("")
    return "\n".join(lines)


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_render_markdown(report), encoding="utf-8")


def _print_summary(report: dict[str, Any], output_dir: Path) -> None:
    summary = report["summary"]
    country = summary["country"]
    overall = country.get("overall") or {}
    print("")
    print(f"Country top-1 (available): {_fmt_rate(country.get('top1'))}   "
          f"top-3: {_fmt_rate(country.get('top3'))}   "
          f"MRR: {_fmt_num(country.get('mrr'))}")
    print(f"Country top-1 (all attempted): {_fmt_rate(overall.get('top1'))}   "
          f"top-3: {_fmt_rate(overall.get('top3'))}   "
          f"MRR: {_fmt_num(overall.get('mrr'))}")
    print(f"Country label coverage: {_fmt_rate(country.get('label_coverage'))} "
          f"({country.get('labeled_samples')}/{summary['samples']}; "
          f"{country.get('unlabeled_samples')} unlabeled)")
    city_true = summary["city"].get("true_country") or {}
    city_e2e = summary["city"].get("e2e") or {}
    if city_true:
        city_true_overall = city_true.get("overall") or {}
        print(f"City (true country) top-1 <=50 km: "
              f"{_fmt_rate(city_true.get('top1_at_50km'))} available / "
              f"{_fmt_rate(city_true_overall.get('top1_at_50km'))} all attempted   "
              f"available: {_fmt_rate(city_true.get('available_rate'))}")
    if city_e2e:
        city_e2e_overall = city_e2e.get("overall") or {}
        print(f"City (end-to-end)   top-1 <=50 km: "
              f"{_fmt_rate(city_e2e.get('top1_at_50km'))} available / "
              f"{_fmt_rate(city_e2e_overall.get('top1_at_50km'))} all attempted   "
              f"available: {_fmt_rate(city_e2e.get('available_rate'))}")
    degraded = int(summary.get("degraded") or 0)
    degraded_rate = degraded / summary["samples"] if summary.get("samples") else None
    print(f"Degraded (all samples): {_fmt_rate(degraded_rate)}   "
          f"errors: {summary['errors']}/{summary['samples']}")
    print(f"Report: {output_dir / 'report.md'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _country_top1_threshold(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number in [0, 1]") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number in [0, 1]")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the production geolocation chain over a labeled manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="labeled JSONL manifest")
    parser.add_argument("--image-root", help="image root (default: manifest directory)")
    parser.add_argument(
        "--catalog-db",
        default=str(REPO_ROOT / "data" / "geo_catalog.sqlite3"),
        help="read-only SQLite catalog used for identities and scoring",
    )
    parser.add_argument("--output", help="report directory (default: reports/eval-<UTC>)")
    parser.add_argument("--env-file", default=str(REPO_ROOT / ".env"), help="env file to load")
    parser.add_argument("--limit", type=int, help="evaluate only the first N samples")
    parser.add_argument("--ids", help="comma-separated sample ids to evaluate")
    parser.add_argument("--no-cities", action="store_true", help="skip oracle-country city ranking")
    parser.add_argument("--no-e2e-cities", action="store_true", help="skip end-to-end city ranking")
    parser.add_argument(
        "--allow-unlabeled-country",
        action="store_true",
        help="allow coordinate-only samples without a country label",
    )
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help="permit a rules-only run when StreetCLIP is disabled or unreachable",
    )
    parser.add_argument("--fail-fast", action="store_true", help="abort on the first sample error")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 on sample errors, country degradation or any enabled city-path failure",
    )
    parser.add_argument(
        "--min-country-top1",
        type=_country_top1_threshold,
        help="exit 1 when all-attempted country top-1 accuracy is below this fraction",
    )
    parser.add_argument("--label", help="experiment label stored in the report")
    parser.add_argument("--streetclip-url", help="override STREETCLIP_URL")
    parser.add_argument(
        "--container",
        action="store_true",
        help="keep host.docker.internal URLs (run inside a container)",
    )
    parser.add_argument("--ocr-provider", choices=("mock", "rapidocr"))
    parser.add_argument("--ocr-model-type", choices=("server", "mobile"))
    parser.add_argument("--ai-mode", choices=("mock", "real"))
    parser.add_argument("--force", action="store_true", help="write into a non-empty output dir")
    parser.add_argument("--quiet", action="store_true", help="suppress per-sample progress")
    return parser.parse_args(argv)


def _select_entries(entries: Sequence[ManifestEntry], args: argparse.Namespace) -> list[ManifestEntry]:
    selected = list(entries)
    if args.ids:
        wanted = [item.strip() for item in args.ids.split(",") if item.strip()]
        wanted_set = set(wanted)
        selected = [entry for entry in selected if entry.id in wanted_set]
        missing = sorted(wanted_set - {entry.id for entry in selected})
        if missing:
            raise ManifestError("ids not found in manifest: " + ", ".join(missing))
    if args.limit is not None:
        if args.limit <= 0:
            raise EvalError("--limit must be positive")
        selected = selected[: args.limit]
    if not selected:
        raise ManifestError("no samples selected")
    return selected


def _git_info() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except Exception:
        return {"commit": None, "dirty": None}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gate_failures(
    report: dict[str, Any],
    args: argparse.Namespace,
    *,
    evaluate_cities: bool,
    evaluate_e2e_cities: bool,
    fail_fast_sample: str | None = None,
) -> list[str]:
    summary = report["summary"]
    country = summary["country"]
    overall = country.get("overall") or {}
    failures: list[str] = []

    if fail_fast_sample is not None:
        failures.append(f"fail-fast stopped on sample {fail_fast_sample}")
    if args.strict:
        if summary.get("errors"):
            failures.append(f"{summary['errors']} sample error(s)")
        if summary.get("degraded") or overall.get("degraded_rate"):
            failures.append("some samples ran without StreetCLIP evidence")
        enabled_city_errors = len(report.get("city_errors") or [])
        if not enabled_city_errors:
            city = summary.get("city") or {}
            for enabled, key in (
                (evaluate_cities, "true_country"),
                (evaluate_e2e_cities, "e2e"),
            ):
                if not enabled:
                    continue
                row = city.get(key) or {}
                enabled_city_errors += int(row.get("unavailable") or 0)
        if enabled_city_errors:
            failures.append(f"{enabled_city_errors} enabled city-path failure(s)")
    if args.min_country_top1 is not None:
        observed = overall.get("top1")
        if observed is None or observed < args.min_country_top1:
            failures.append(
                "country top-1 (all attempted) "
                f"{observed if observed is not None else 'n/a'} "
                f"< {args.min_country_top1}"
            )
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    catalog_path = Path(args.catalog_db).expanduser().resolve()
    if not catalog_path.is_file():
        print(f"error: catalog not found: {catalog_path}", file=sys.stderr)
        return 2

    try:
        settings = _build_settings(args, catalog_path)
        kg.configure_catalog(catalog_path)
        catalog = CatalogDatabase(catalog_path)
        countries = catalog.countries()
        valid_isos = {str(row["iso"]).upper() for row in countries}
        entries = load_manifest(
            args.manifest,
            image_root=args.image_root,
            valid_isos=valid_isos,
            allow_unlabeled_country=args.allow_unlabeled_country,
        )
        entries = _select_entries(entries, args)
    except (ManifestError, EvalError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_dir = (
        Path(args.output)
        if args.output
        else REPO_ROOT / "reports" / f"eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        print(f"error: output directory is not empty: {output_dir} (use --force)", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluate_cities = not args.no_cities
    evaluate_e2e_cities = not args.no_e2e_cities
    require_city = evaluate_e2e_cities or (
        evaluate_cities and any(entry.country_iso is not None for entry in entries)
    )

    runtime_settings = settings
    city_path_error: str | None = None
    model_health: dict[str, Any]
    try:
        model_health = _probe_streetclip(settings, require_city=require_city)
        model_health = {**model_health, "country_capability": True}
    except EvalError as exc:
        if not args.allow_degraded:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        # A failed country preflight means the whole model path is unreliable.
        # Disable it for this run instead of allowing opportunistic per-sample
        # retries that would silently mix model and rules-only records.
        runtime_settings = replace(settings, streetclip_enabled=False)
        city_path_error = "country model unavailable; rules-only degraded run"
        model_health = {
            "ready": False,
            "country_capability": False,
            "city_capability": False,
            "detail": str(exc),
        }
        print(f"warning: {exc}", file=sys.stderr)
        print(
            "warning: continuing with a rules-only baseline; "
            "country/city model calls are disabled for this run.",
            file=sys.stderr,
        )
    else:
        if require_city and not model_health.get("city_capability"):
            message = (
                "StreetCLIP country prediction is available, but city ranking is "
                "unavailable (city_capability=false)"
            )
            if not args.allow_degraded:
                print(f"error: {message}; rerun with --no-cities --no-e2e-cities "
                      "for a country-only run", file=sys.stderr)
                return 2
            city_path_error = "StreetCLIP city capability unavailable"
            print(f"warning: {message}; continuing country-only with city paths unavailable.",
                  file=sys.stderr)

    pipeline = PipelineService(runtime_settings)
    gateway = StreetClipGateway(runtime_settings)
    engine = ModelAwareEngine()
    reasoner = None
    if require_city and city_path_error is None:
        try:
            city_index = load_city_index(catalog_path)
            reasoner = CityClueReasoner(city_index)
        except Exception as exc:
            if not args.allow_degraded:
                print(f"error: city catalog unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 2
            city_path_error = f"city catalog unavailable: {type(exc).__name__}"
            print(f"warning: {city_path_error}; city metrics will be unavailable",
                  file=sys.stderr)

    semantic_revision = semantic_catalog_revision(countries)
    stored_country_revision = catalog.revision("countries")
    city_revision = catalog.revision("cities")

    records: list[dict[str, Any]] = []
    fail_fast_sample: str | None = None
    samples_path = output_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as samples_handle:
        for index, entry in enumerate(entries, start=1):
            record = _evaluate_sample(
                entry,
                pipeline=pipeline,
                gateway=gateway,
                engine=engine,
                reasoner=reasoner,
                evaluate_cities=evaluate_cities,
                evaluate_e2e_cities=evaluate_e2e_cities,
                city_path_error=city_path_error,
            )
            samples_handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            samples_handle.flush()
            records.append(record)
            if not args.quiet:
                country = record.get("country") or {}
                city_true = record.get("city_true_country") or {}
                city_e2e = record.get("city_e2e") or {}
                parts = [
                    f"[{index}/{len(entries)}] {entry.id}",
                    f"country_rank={country.get('rank')}",
                ]
                if city_true:
                    parts.append(f"city@true={_fmt_num(city_true.get('top1_km'), 1)}km")
                if city_e2e:
                    parts.append(f"city@e2e={_fmt_num(city_e2e.get('top1_km'), 1)}km")
                if record.get("error"):
                    parts.append(f"error={record['error'][:80]}")
                print("  ".join(parts), flush=True)
            if args.fail_fast and record.get("error"):
                fail_fast_sample = entry.id
                print(f"error: aborting on sample {entry.id}", file=sys.stderr)
                break

    aggregated = _aggregate(
        records,
        evaluate_cities=evaluate_cities,
        evaluate_e2e_cities=evaluate_e2e_cities,
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "label": args.label,
        "run": {
            "selected_samples": len(entries),
            "completed_samples": len(records),
            "partial": fail_fast_sample is not None,
            "fail_fast_sample": fail_fast_sample,
        },
        "manifest": {
            "path": str(Path(args.manifest).resolve()),
            "sha256": _sha256_file(Path(args.manifest)),
            "samples": len(entries),
            "image_root": str(
                Path(args.image_root).resolve()
                if args.image_root
                else Path(args.manifest).resolve().parent
            ),
        },
        "environment": {
            "git": _git_info(),
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "command": [sys.executable, *sys.argv],
        },
        "configuration": {
            "streetclip_enabled": runtime_settings.streetclip_enabled,
            "streetclip_requested": settings.streetclip_enabled,
            "streetclip_url": settings.streetclip_url,
            "streetclip_api_key_configured": bool(settings.streetclip_api_key),
            "streetclip_timeout_seconds": settings.streetclip_timeout_seconds,
            "model_health": model_health,
            "model_revision": model_health.get("model_revision"),
            "prompt_version": PROMPT_VERSION,
            "catalog_db": str(catalog_path),
            "catalog_revision": semantic_revision,
            "catalog_revision_stored": stored_country_revision,
            "city_catalog_revision": city_revision,
            "city_paths": {
                "true_country": evaluate_cities,
                "e2e": evaluate_e2e_cities,
            },
            "city_path_error": city_path_error,
            "allow_unlabeled_country": args.allow_unlabeled_country,
            "ocr_provider": runtime_settings.ocr_provider,
            "ocr_model_type": runtime_settings.ocr_model_type,
            "geo_ai_mode": runtime_settings.geo_ai_mode,
        },
        "summary": aggregated["summary"],
        "errors": [
            {"id": record["id"], "error": record["error"]}
            for record in aggregated["errors"]
        ],
        "degraded_sample_ids": [record["id"] for record in aggregated["degraded_samples"]],
        "unavailable_city_sample_ids": [
            record["id"] for record in aggregated["unavailable_city_samples"]
        ],
        "city_errors": aggregated["city_errors"],
        "samples": records,
    }

    failures = _gate_failures(
        report,
        args,
        evaluate_cities=evaluate_cities,
        evaluate_e2e_cities=evaluate_e2e_cities,
        fail_fast_sample=fail_fast_sample,
    )
    country = report["summary"]["country"]
    overall = country.get("overall") or {}
    threshold = args.min_country_top1
    observed_top1 = overall.get("top1")
    minimum_passed = (
        None
        if threshold is None
        else observed_top1 is not None and float(observed_top1) >= float(threshold)
    )
    strict_failures = [
        failure
        for failure in failures
        if not failure.startswith("country top-1 (all attempted) ")
        and not failure.startswith("fail-fast stopped on sample ")
    ]
    report["gates"] = {
        "strict": args.strict,
        "strict_failures": strict_failures,
        "min_country_top1": threshold,
        "min_country_top1_passed": minimum_passed,
        "country_top1_available": country.get("top1"),
        "country_top1_overall": observed_top1,
        "city_path_failures": len(report["city_errors"]),
        "failures": failures,
        "passed": not failures,
    }
    _write_report(output_dir, report)
    _print_summary(report, output_dir)

    if failures:
        print("gate failed: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
