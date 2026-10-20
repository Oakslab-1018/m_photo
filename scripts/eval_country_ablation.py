#!/usr/bin/env python3
"""Country-only paired ablation for StreetCLIP and OCR clues.

Capture imports the evaluator, pipeline and gateway from ``--source-root`` so
the cached clues retain the source semantic analyzer. Replay imports only the
current worktree scoring modules, performs no OCR/model/network calls, and
evaluates three arms from the same cached full-image prior:

* model_only: no clues
* legacy_ocr: the complete clue list captured by the source root
* revised_ocr: captured non-semantic clues plus semantic clues re-derived by
  the current worktree SemanticClueAnalyzer

Ground truth is never written to the cache and is never passed to capture,
inference, or evidence construction.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

SCHEMA_VERSION = 1
CACHE_FILENAME = "cache.jsonl"
META_FILENAME = "meta.json"
REPORT_FILENAME = "report.json"
EVIDENCE_VIEW = "full_image_only"
TOP_N = 1000
ARM_NAMES = ("model_only", "legacy_ocr", "revised_ocr")
COMPARISONS = (
    ("legacy_ocr_vs_model_only", "legacy_ocr", "model_only"),
    ("revised_ocr_vs_model_only", "revised_ocr", "model_only"),
    ("revised_ocr_vs_legacy_ocr", "revised_ocr", "legacy_ocr"),
)
SOURCE_SCORER = Path("backend/app/domain/geolocation/model_ranking.py")
SOURCE_ENGINE = Path("backend/app/domain/geolocation/engine.py")
SOURCE_SEMANTIC = Path("backend/app/domain/geolocation/analysis/semantic.py")
SOURCE_EVAL = Path("scripts/eval_geolocation.py")
WORKTREE_ROOT = Path(__file__).resolve().parents[1]


class AblationError(RuntimeError):
    """A configuration, cache, or replay invariant failed closed."""


@dataclass
class CaptureContext:
    entries: list[Any]
    pipeline: Any
    gateway: Any
    version_info: dict[str, Any]
    provenance: dict[str, Any]
    manifest_info: dict[str, Any]
    config_info: dict[str, Any]


@dataclass
class ReplayContext:
    entries: list[Any]
    countries: list[dict[str, Any]]
    engine_factory: Callable[[], Any]
    rederive_semantic: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
    dicts_to_clues: Callable[[list[dict[str, Any]]], Any]
    canonical_country_iso: Callable[[Any], str | None]
    version_info: dict[str, Any]
    provenance: dict[str, Any]
    weights: dict[str, Any]
    manifest_info: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sealed_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    payload.pop("record_sha256", None)
    return payload


def _record_sha256(record: dict[str, Any]) -> str:
    return _sha256_bytes(_json_text(_sealed_payload(record)).encode("utf-8"))


def seal_record(record: dict[str, Any]) -> dict[str, Any]:
    sealed = copy.deepcopy(record)
    sealed.pop("record_sha256", None)
    sealed["record_sha256"] = _record_sha256(sealed)
    return sealed


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise AblationError(f"invalid {label}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise AblationError(f"invalid {label}") from exc
    return value


def _ensure_new_dir(path: str | Path, *, label: str) -> Path:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise AblationError(f"{label} directory already exists: {target}")
    target.mkdir(parents=True)
    return target


def _assert_module_from(module: Any, root: Path, label: str) -> None:
    filename = getattr(module, "__file__", None)
    if not filename:
        raise AblationError(f"{label} has no source file")
    try:
        Path(filename).resolve().relative_to(root)
    except ValueError as exc:
        raise AblationError(f"{label} was imported outside {root}") from exc


def _source_hashes(root: Path) -> dict[str, str]:
    paths = {
        "scorer": root / SOURCE_SCORER,
        "engine": root / SOURCE_ENGINE,
        "semantic": root / SOURCE_SEMANTIC,
        "eval_helper": root / SOURCE_EVAL,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise AblationError("missing source provenance file: " + ", ".join(missing))
    return {name: _sha256_file(path) for name, path in paths.items()}


def _load_capture_context(args: argparse.Namespace) -> CaptureContext:
    source_root = Path(args.source_root).expanduser().resolve()
    if not source_root.is_dir():
        raise AblationError(f"source root not found: {source_root}")

    # This must happen before the first project import. Capture must use the
    # source-root pipeline/semantic implementation, not this worktree's copy.
    sys.path.insert(0, str(source_root))
    eg = importlib.import_module("scripts.eval_geolocation")
    pipeline_module = importlib.import_module("backend.app.services.pipeline")
    gateway_module = importlib.import_module("backend.app.services.model_gateway")
    catalog_module = importlib.import_module("catalog_db")
    kg_module = importlib.import_module("backend.app.domain.geolocation.knowledge_graph")
    for module, label in (
        (eg, "source evaluator"),
        (pipeline_module, "source pipeline"),
        (gateway_module, "source gateway"),
        (catalog_module, "source catalog"),
        (kg_module, "source knowledge graph"),
    ):
        _assert_module_from(module, source_root, label)

    catalog_path = Path(args.catalog_db).expanduser().resolve()
    if not catalog_path.is_file():
        raise AblationError(f"catalog not found: {catalog_path}")

    settings = eg._build_settings(args, catalog_path)
    if not settings.streetclip_enabled:
        raise AblationError("capture requires STREETCLIP_ENABLED=true")
    if str(settings.ocr_provider).strip().lower() != "rapidocr":
        raise AblationError("capture requires OCR_PROVIDER=rapidocr")
    health = eg._probe_streetclip(settings, require_city=False)
    if not isinstance(health, dict) or not health.get("ready"):
        raise AblationError("StreetCLIP preflight did not report ready=true")

    kg_module.configure_catalog(catalog_path)
    catalog = catalog_module.CatalogDatabase(catalog_path)
    countries = catalog.countries()
    valid_isos = {str(row["iso"]).upper() for row in countries}
    entries = eg.load_manifest(
        args.manifest,
        image_root=args.image_root,
        valid_isos=valid_isos,
        allow_unlabeled_country=True,
    )
    entries = eg._select_entries(entries, SimpleNamespace(ids=args.ids, limit=args.limit))

    model_revision = str(health.get("model_revision") or gateway_module.MODEL_REVISION)
    if model_revision != gateway_module.MODEL_REVISION:
        raise AblationError("StreetCLIP health model revision does not match the gateway")
    version_info = {
        "model_id": gateway_module.MODEL_ID,
        "model_revision": model_revision,
        "prompt_version": gateway_module.PROMPT_VERSION,
        "prompts": list(gateway_module.PROMPTS),
        "catalog_revision": eg.semantic_catalog_revision(countries),
        "catalog_revision_stored": catalog.revision("countries"),
    }
    source_hashes = _source_hashes(source_root)
    manifest_path = Path(args.manifest).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve() if args.image_root else manifest_path.parent
    provenance = {
        "source_root": str(source_root),
        "legacy_semantic_sha256": source_hashes["semantic"],
        "legacy_scorer_sha256": source_hashes["scorer"],
        "legacy_engine_sha256": source_hashes["engine"],
        "legacy_eval_helper_sha256": source_hashes["eval_helper"],
        "capture_script_sha256": _sha256_file(Path(__file__).resolve()),
    }
    manifest_info = {
        "path": str(manifest_path),
        "sha256": _sha256_file(manifest_path),
        "image_root": str(image_root),
        "selected_ids": [entry.id for entry in entries],
    }
    config_info = {
        "ocr_provider": settings.ocr_provider,
        "ocr_model_type": settings.ocr_model_type,
        "geo_ai_mode": settings.geo_ai_mode,
        "streetclip_enabled": bool(settings.streetclip_enabled),
        "allow_unlabeled_country": True,
    }
    return CaptureContext(
        entries=list(entries),
        pipeline=pipeline_module.PipelineService(settings),
        gateway=gateway_module.StreetClipGateway(settings),
        version_info=version_info,
        provenance=provenance,
        manifest_info=manifest_info,
        config_info=config_info,
    )


def _validate_capture_evidence(evidence: Any, version_info: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(evidence, dict):
        raise AblationError("StreetCLIP returned no evidence")
    rankings = evidence.get("rankings")
    if not isinstance(rankings, list) or not rankings:
        raise AblationError("StreetCLIP evidence has no full rankings")
    for field in ("model_revision", "prompt_version", "catalog_revision"):
        if evidence.get(field) != version_info.get(field):
            raise AblationError(f"StreetCLIP evidence {field} mismatch")
    return evidence


def capture_records(
    entries: Sequence[Any],
    pipeline: Any,
    gateway: Any,
    version_info: dict[str, Any],
    provenance: dict[str, Any],
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Capture one OCR pass and one model pass per image without ground truth."""

    records: list[dict[str, Any]] = []
    for entry in entries:
        raw_image_sha256 = _sha256_file(entry.image_path)
        clues = pipeline.analyze(str(entry.image_path))
        ocr_status = pipeline.ocr_status()
        if not isinstance(clues, list) or not all(isinstance(clue, dict) for clue in clues):
            raise AblationError(f"pipeline returned invalid clues for {entry.id}")
        if not isinstance(ocr_status, dict) or ocr_status.get("state") != "ok":
            state = ocr_status.get("state") if isinstance(ocr_status, dict) else "invalid"
            raise AblationError(f"OCR was not ok for {entry.id}: {state}")

        evidence = _validate_capture_evidence(
            gateway.rank_image(entry.image_path), version_info
        )
        record = {
            "schema_version": SCHEMA_VERSION,
            "kind": "country_ablation_sample",
            "id": str(entry.id),
            "image_ref": str(entry.image_ref),
            "raw_image_sha256": raw_image_sha256,
            "model_image_sha256": evidence.get("image_sha256"),
            "model_id": version_info["model_id"],
            "model_revision": version_info["model_revision"],
            "prompt_version": version_info["prompt_version"],
            "prompt_templates": copy.deepcopy(version_info["prompts"]),
            "catalog_revision": version_info["catalog_revision"],
            "catalog_revision_stored": version_info["catalog_revision_stored"],
            "legacy_semantic_sha256": provenance["legacy_semantic_sha256"],
            "legacy_scorer_sha256": provenance["legacy_scorer_sha256"],
            "legacy_engine_sha256": provenance["legacy_engine_sha256"],
            "ocr_status": copy.deepcopy(ocr_status),
            "evidence": copy.deepcopy(evidence),
            "clues": copy.deepcopy(clues),
        }
        sealed = seal_record(record)
        if on_record is not None:
            on_record(sealed)
        records.append(sealed)
    return records


def _build_capture_meta(
    context: CaptureContext,
    records: Sequence[dict[str, Any]],
    cache_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "country_ablation_capture",
        "generated_at": _utc_now(),
        "source_root": context.provenance["source_root"],
        "manifest": context.manifest_info,
        "config": context.config_info,
        "versions": context.version_info,
        "provenance": context.provenance,
        "sample_count": len(records),
        "cache_file": CACHE_FILENAME,
        "cache_sha256": cache_sha256,
    }


def _capture_command(args: argparse.Namespace) -> int:
    output_dir = _ensure_new_dir(args.output, label="capture output")
    partial_path = output_dir / (CACHE_FILENAME + ".partial")
    final_cache_path = output_dir / CACHE_FILENAME
    meta_tmp = output_dir / (META_FILENAME + ".tmp")
    written = 0
    try:
        context = _load_capture_context(args)
        with partial_path.open("w", encoding="utf-8") as handle:
            def write_record(record: dict[str, Any]) -> None:
                nonlocal written
                handle.write(
                    json.dumps(
                        record, ensure_ascii=False, sort_keys=True, allow_nan=False
                    )
                    + "\n"
                )
                handle.flush()
                written += 1

            records = capture_records(
                context.entries,
                context.pipeline,
                context.gateway,
                context.version_info,
                context.provenance,
                on_record=write_record,
            )
        cache_sha256 = _sha256_file(partial_path)
        os.replace(partial_path, final_cache_path)
        meta = _build_capture_meta(context, records, cache_sha256)
        meta_tmp.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(meta_tmp, output_dir / META_FILENAME)
    except Exception:
        if meta_tmp.exists():
            meta_tmp.unlink()
        if partial_path.exists():
            print(
                f"capture stopped: {written} completed samples preserved at {partial_path}",
                file=sys.stderr,
            )
        elif final_cache_path.exists():
            print(
                f"capture stopped after cache publish: {final_cache_path} has no final meta",
                file=sys.stderr,
            )
        else:
            print("capture stopped before any completed sample was written", file=sys.stderr)
        raise
    print(f"capture: {output_dir}")
    return 0


def load_cache(cache_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(cache_dir).expanduser().resolve()
    meta_path = root / META_FILENAME
    cache_path = root / CACHE_FILENAME
    if not meta_path.is_file() or not cache_path.is_file():
        raise AblationError("cache directory must contain meta.json and cache.jsonl")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AblationError("cache meta.json is invalid JSON") from exc
    if meta.get("schema_version") != SCHEMA_VERSION or meta.get("kind") != "country_ablation_capture":
        raise AblationError("unsupported capture cache schema")
    expected_cache_sha256 = _require_sha256(meta.get("cache_sha256"), "cache sha256")
    if _sha256_file(cache_path) != expected_cache_sha256:
        raise AblationError("cache.jsonl hash mismatch")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with cache_path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                raise AblationError(f"blank cache record at line {lineno}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AblationError(f"invalid cache JSON at line {lineno}") from exc
            if not isinstance(record, dict):
                raise AblationError(f"cache record at line {lineno} is not an object")
            if record.get("schema_version") != SCHEMA_VERSION or record.get("kind") != "country_ablation_sample":
                raise AblationError(f"unsupported cache record at line {lineno}")
            if record.get("record_sha256") != _record_sha256(record):
                raise AblationError(f"cache record checksum mismatch at line {lineno}")
            sample_id = record.get("id")
            if not isinstance(sample_id, str) or not sample_id:
                raise AblationError(f"cache record at line {lineno} has no id")
            if sample_id in seen:
                raise AblationError(f"duplicate cache sample id: {sample_id}")
            seen.add(sample_id)
            for forbidden in ("country_iso", "canonical_country_iso", "ground_truth", "gt"):
                if forbidden in record:
                    raise AblationError(f"cache sample {sample_id} contains ground truth")
            _validate_record_shape(record)
            records.append(record)

    manifest = meta.get("manifest")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("selected_ids"), list):
        raise AblationError("cache meta has no selected ids")
    expected_ids = [str(item) for item in manifest["selected_ids"]]
    actual_ids = [record["id"] for record in records]
    if len(actual_ids) != len(expected_ids):
        raise AblationError("cache record count does not match meta selected ids")
    if actual_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(actual_ids))
        extra = sorted(set(actual_ids) - set(expected_ids))
        if missing or extra:
            raise AblationError(
                "cache selected ids mismatch: "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
        raise AblationError("cache record order does not match meta selected ids")
    if meta.get("sample_count") != len(records):
        raise AblationError("cache sample_count mismatch")
    _validate_cache_versions(records, meta)
    return records, meta


def _validate_record_shape(record: dict[str, Any]) -> None:
    for key in (
        "raw_image_sha256",
        "model_revision",
        "prompt_version",
        "catalog_revision",
        "catalog_revision_stored",
        "legacy_semantic_sha256",
        "legacy_scorer_sha256",
    ):
        if not isinstance(record.get(key), str) or not record[key]:
            raise AblationError(f"cache sample {record.get('id')} has no {key}")
    _require_sha256(record["raw_image_sha256"], f"{record['id']} raw image sha256")
    _require_sha256(record["legacy_semantic_sha256"], f"{record['id']} legacy semantic sha256")
    _require_sha256(record["legacy_scorer_sha256"], f"{record['id']} legacy scorer sha256")
    if not isinstance(record.get("prompt_templates"), list) or not record["prompt_templates"]:
        raise AblationError(f"cache sample {record['id']} has no prompt templates")
    if not isinstance(record.get("evidence"), dict):
        raise AblationError(f"cache sample {record['id']} has invalid evidence")
    if not isinstance(record["evidence"].get("rankings"), list) or not record["evidence"]["rankings"]:
        raise AblationError(f"cache sample {record['id']} has no full prior")
    if not isinstance(record.get("clues"), list):
        raise AblationError(f"cache sample {record['id']} has invalid clues")
    if not isinstance(record.get("ocr_status"), dict):
        raise AblationError(f"cache sample {record['id']} has invalid OCR status")


def _validate_cache_versions(records: Sequence[dict[str, Any]], meta: dict[str, Any]) -> None:
    versions = meta.get("versions")
    provenance = meta.get("provenance")
    if not isinstance(versions, dict) or not isinstance(provenance, dict):
        raise AblationError("cache meta has no versions or provenance")
    for record in records:
        for field in (
            "model_id",
            "model_revision",
            "prompt_version",
            "catalog_revision",
            "catalog_revision_stored",
        ):
            if record.get(field) != versions.get(field):
                raise AblationError(f"cache {field} mismatch for {record['id']}")
        if record.get("prompt_templates") != versions.get("prompts"):
            raise AblationError(f"cache prompt templates mismatch for {record['id']}")
        for field in ("model_revision", "prompt_version", "catalog_revision"):
            if record["evidence"].get(field) != record.get(field):
                raise AblationError(f"evidence {field} mismatch for {record['id']}")
        if record["legacy_semantic_sha256"] != provenance.get("legacy_semantic_sha256"):
            raise AblationError(f"legacy semantic version mismatch for {record['id']}")
        if record["legacy_scorer_sha256"] != provenance.get("legacy_scorer_sha256"):
            raise AblationError(f"legacy scorer version mismatch for {record['id']}")
        if record["legacy_engine_sha256"] != provenance.get("legacy_engine_sha256"):
            raise AblationError(f"legacy engine version mismatch for {record['id']}")


def _full_image_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    view = copy.deepcopy(evidence)
    view.pop("crops", None)
    view.pop("reference", None)
    return view


def _revised_clue_payloads(
    legacy_clues: Sequence[dict[str, Any]],
    rederive_semantic: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = [
        copy.deepcopy(clue)
        for clue in legacy_clues
        if isinstance(clue, dict) and clue.get("source") != "semantic"
    ]
    derived = rederive_semantic(copy.deepcopy(base))
    if not isinstance(derived, list) or not all(isinstance(clue, dict) for clue in derived):
        raise AblationError("SemanticClueAnalyzer returned invalid clues")
    return base, copy.deepcopy(base) + copy.deepcopy(derived)


def _rank_of(predicted: Sequence[Any], target: Any, canonicalizer: Callable[[Any], str | None]) -> int | None:
    canonical_target = canonicalizer(target)
    if not canonical_target:
        return None
    for index, iso in enumerate(predicted, start=1):
        if canonicalizer(iso) == canonical_target:
            return index
    return None


def _score_arm(
    engine_factory: Callable[[], Any],
    dicts_to_clues: Callable[[list[dict[str, Any]]], Any],
    evidence: dict[str, Any],
    clues: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    try:
        engine = engine_factory()
        engine.evidence = copy.deepcopy(evidence)
        scores = engine.infer(
            dicts_to_clues(copy.deepcopy(list(clues))),
            top_n=max(TOP_N, len(evidence.get("rankings") or [])),
        )
        predicted = [str(score.iso) for score in scores]
        if not predicted:
            raise AblationError("engine returned no country scores")
        return {"predicted": predicted, "top5": predicted[:5], "error": None}
    except Exception as exc:  # per-arm failure is counted as a miss and reported
        return {"predicted": [], "top5": [], "error": type(exc).__name__}


def _validate_replay_versions(
    records: Sequence[dict[str, Any]],
    meta: dict[str, Any],
    context: ReplayContext,
) -> None:
    for field in ("model_revision", "prompt_version", "catalog_revision"):
        expected = meta["versions"].get(field)
        actual = context.version_info.get(field)
        if expected != actual:
            raise AblationError(f"replay {field} mismatch")
    if meta["versions"].get("model_id") != context.version_info.get("model_id"):
        raise AblationError("replay model_id mismatch")
    if meta["versions"].get("catalog_revision_stored") != context.version_info.get("catalog_revision_stored"):
        raise AblationError("replay catalog_revision_stored mismatch")
    if meta["versions"].get("prompts") != context.version_info.get("prompts"):
        raise AblationError("replay prompt templates mismatch")
    if meta["manifest"].get("sha256") != context.manifest_info.get("sha256"):
        raise AblationError("manifest sha256 mismatch")
    expected_ids = [str(item) for item in meta["manifest"]["selected_ids"]]
    if [entry.id for entry in context.entries] != expected_ids:
        raise AblationError("replay manifest ids do not match cache ids")
    _validate_cache_versions(records, meta)
    capture_provenance = meta["provenance"]
    if capture_provenance.get("legacy_scorer_sha256") != context.provenance.get("revised_scorer_sha256"):
        raise AblationError("replay scorer changed from capture")
    if capture_provenance.get("legacy_engine_sha256") != context.provenance.get("revised_engine_sha256"):
        raise AblationError("replay engine changed from capture")


def _metric(rows: Sequence[dict[str, Any]], arm: str, k: int) -> float | None:
    if not rows:
        return None
    hits = sum(1 for row in rows if (row["arms"][arm]["rank"] or 10**9) <= k)
    return round(hits / len(rows), 6)


def _rank_key(value: int | None) -> str:
    return "miss" if value is None else str(value)


def _comparison(
    rows: Sequence[dict[str, Any]], arm: str, baseline: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k in (1, 3, 5):
        corrected: list[str] = []
        harmed: list[str] = []
        unchanged: list[str] = []
        transitions: dict[str, list[str]] = {}
        for row in rows:
            baseline_rank = row["arms"][baseline]["rank"]
            candidate_rank = row["arms"][arm]["rank"]
            baseline_hit = baseline_rank is not None and baseline_rank <= k
            candidate_hit = candidate_rank is not None and candidate_rank <= k
            if not baseline_hit and candidate_hit:
                corrected.append(row["id"])
            elif baseline_hit and not candidate_hit:
                harmed.append(row["id"])
            else:
                unchanged.append(row["id"])
            transition = f"{_rank_key(baseline_rank)}->{_rank_key(candidate_rank)}"
            transitions.setdefault(transition, []).append(row["id"])
        result[f"top{k}"] = {
            "corrected_ids": corrected,
            "harmed_ids": harmed,
            "unchanged_ids": unchanged,
            "rank_transitions": transitions,
        }
    return result


def replay_records(
    records: Sequence[dict[str, Any]],
    meta: dict[str, Any],
    context: ReplayContext,
) -> dict[str, Any]:
    """Evaluate all three arms from cache without OCR, model, or network calls."""

    _validate_replay_versions(records, meta, context)
    entries_by_id = {entry.id: entry for entry in context.entries}
    if set(entries_by_id) != {record["id"] for record in records}:
        raise AblationError("manifest entries do not match cache records")

    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    sample_reports: list[dict[str, Any]] = []

    for record in records:
        sample_id = record["id"]
        entry = entries_by_id[sample_id]
        actual_raw_sha256 = _sha256_file(entry.image_path)
        if actual_raw_sha256 != record["raw_image_sha256"]:
            raise AblationError(f"raw image sha256 mismatch for {sample_id}")

        country_iso = getattr(entry, "country_iso", None)
        if country_iso is None:
            excluded.append({"id": sample_id, "reason": "unknown_label"})
            sample_reports.append(
                {"id": sample_id, "included": False, "exclude_reason": "unknown_label"}
            )
            continue

        evidence = _full_image_evidence(record["evidence"])
        legacy_clues = copy.deepcopy(record["clues"])
        arm_errors: dict[str, str] = {}
        try:
            base_clues, revised_clues = _revised_clue_payloads(
                legacy_clues, context.rederive_semantic
            )
        except Exception as exc:
            base_clues = [
                copy.deepcopy(clue)
                for clue in legacy_clues
                if isinstance(clue, dict) and clue.get("source") != "semantic"
            ]
            revised_clues = copy.deepcopy(base_clues)
            arm_errors["revised_ocr"] = type(exc).__name__

        payloads = {
            "model_only": [],
            "legacy_ocr": legacy_clues,
            "revised_ocr": revised_clues,
        }
        arms: dict[str, Any] = {}
        for arm in ARM_NAMES:
            if arm == "revised_ocr" and arm in arm_errors:
                result = {"predicted": [], "top5": [], "error": arm_errors[arm]}
            else:
                result = _score_arm(
                    context.engine_factory,
                    context.dicts_to_clues,
                    evidence,
                    payloads[arm],
                )
            rank = _rank_of(
                result["predicted"], country_iso, context.canonical_country_iso
            )
            result["rank"] = rank
            arms[arm] = result
            if result["error"]:
                failures.append(
                    {"id": sample_id, "arm": arm, "reason": result["error"]}
                )

        row = {
            "id": sample_id,
            "country_iso": context.canonical_country_iso(country_iso),
            "arms": arms,
            "clue_counts": {
                "legacy_ocr": len(legacy_clues),
                "revised_base": len(base_clues),
                "revised_ocr": len(revised_clues),
            },
        }
        rows.append(row)
        sample_reports.append(
            {
                "id": sample_id,
                "included": True,
                "country_iso": row["country_iso"],
                "arms": {
                    arm: {
                        "rank": arms[arm]["rank"],
                        "top5": arms[arm]["top5"],
                        "error": arms[arm]["error"],
                    }
                    for arm in ARM_NAMES
                },
                "clue_counts": row["clue_counts"],
            }
        )

    metrics = {
        arm: {
            "samples": len(rows),
            "top1": _metric(rows, arm, 1),
            "top3": _metric(rows, arm, 3),
            "top5": _metric(rows, arm, 5),
        }
        for arm in ARM_NAMES
    }
    included_ids = [row["id"] for row in rows]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "country_ablation_replay",
        "generated_at": _utc_now(),
        "cache": {
            "path": str(Path(meta["_cache_dir"]).resolve()) if meta.get("_cache_dir") else None,
            "sha256": meta["cache_sha256"],
            "sample_count": meta["sample_count"],
        },
        "manifest": context.manifest_info,
        "versions": context.version_info,
        "provenance": {
            "capture": meta["provenance"],
            "replay": context.provenance,
        },
        "evidence_view": EVIDENCE_VIEW,
        "weights": context.weights,
        "arms": {
            "model_only": {"clues": "none"},
            "legacy_ocr": {"clues": "captured_full"},
            "revised_ocr": {
                "clues": "captured_non_semantic_plus_current_semantic"
            },
        },
        "denominator": {
            "attempted_labelled": len(rows),
            "successful_all_arms": sum(
                1 for row in rows if all(row["arms"][arm]["error"] is None for arm in ARM_NAMES)
            ),
            "unknown_label_excluded": len(excluded),
            "included_ids": included_ids,
        },
        "metrics": metrics,
        "comparisons": {
            name: _comparison(rows, arm, baseline)
            for name, arm, baseline in COMPARISONS
        },
        "excluded": excluded,
        "failures": failures,
        "samples": sample_reports,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Country ablation",
        "",
        f"- Evidence view: `{report['evidence_view']}`",
        f"- Labelled attempted: `{report['denominator']['attempted_labelled']}`",
        f"- Unknown label excluded: `{report['denominator']['unknown_label_excluded']}`",
        "",
        "| arm | top1 | top3 | top5 |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARM_NAMES:
        row = report["metrics"][arm]
        lines.append(
            f"| {arm} | {row['top1']} | {row['top3']} | {row['top5']} |"
        )
    lines.extend(
        [
            "",
            "| comparison | metric | corrected | harmed | unchanged |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for comparison, data in report["comparisons"].items():
        for k in (1, 3, 5):
            row = data[f"top{k}"]
            lines.append(
                f"| {comparison} | top{k} | {len(row['corrected_ids'])} | "
                f"{len(row['harmed_ids'])} | {len(row['unchanged_ids'])} |"
            )
    if report["excluded"]:
        lines.extend(["", "## Excluded", ""])
        lines.extend(
            f"- `{item['id']}`: {item['reason']}" for item in report["excluded"]
        )
    if report["failures"]:
        lines.extend(["", "## Arm failures", ""])
        lines.extend(
            f"- `{item['id']}` / `{item['arm']}`: {item['reason']}"
            for item in report["failures"]
        )
    lines.append("")
    return "\n".join(lines)


def _write_replay_outputs(output_dir: Path, report: dict[str, Any]) -> None:
    (output_dir / REPORT_FILENAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_render_markdown(report), encoding="utf-8")
    meta = {
        "schema_version": SCHEMA_VERSION,
        "kind": "country_ablation_replay_meta",
        "generated_at": report["generated_at"],
        "cache": report["cache"],
        "versions": report["versions"],
        "provenance": report["provenance"],
        "weights": report["weights"],
        "evidence_view": report["evidence_view"],
    }
    (output_dir / META_FILENAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _replay_command(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if output_dir == cache_dir or output_dir.is_relative_to(cache_dir) or cache_dir.is_relative_to(output_dir):
        raise AblationError("replay output must be separate from the cache directory")
    _ensure_new_dir(output_dir, label="replay output")
    records, meta = load_cache(cache_dir)
    meta = {**meta, "_cache_dir": str(cache_dir)}
    context = _load_replay_context(args, meta)
    report = replay_records(records, meta, context)
    _write_replay_outputs(output_dir, report)
    print(f"replay: {output_dir}")
    return 0


def _load_replay_context(args: argparse.Namespace, meta: dict[str, Any]) -> ReplayContext:
    sys.path.insert(0, str(WORKTREE_ROOT))
    eval_manifest = importlib.import_module("scripts.eval_manifest")
    model_ranking = importlib.import_module("backend.app.domain.geolocation.model_ranking")
    semantic_module = importlib.import_module(
        "backend.app.domain.geolocation.analysis.semantic"
    )
    clue_module = importlib.import_module("backend.app.domain.geolocation.clue")
    country_policy = importlib.import_module(
        "backend.app.domain.geolocation.country_policy"
    )
    gateway_module = importlib.import_module("backend.app.services.model_gateway")
    catalog_module = importlib.import_module("catalog_db")
    kg_module = importlib.import_module("backend.app.domain.geolocation.knowledge_graph")
    for module, label in (
        (eval_manifest, "worktree eval manifest"),
        (model_ranking, "worktree scorer"),
        (semantic_module, "worktree semantic analyzer"),
        (clue_module, "worktree clue module"),
        (country_policy, "worktree country policy"),
        (gateway_module, "worktree model gateway constants"),
        (catalog_module, "worktree catalog"),
        (kg_module, "worktree knowledge graph"),
    ):
        _assert_module_from(module, WORKTREE_ROOT, label)

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise AblationError(f"manifest not found: {manifest_path}")
    manifest_sha256 = _sha256_file(manifest_path)
    if manifest_sha256 != meta["manifest"]["sha256"]:
        raise AblationError("replay manifest sha256 does not match capture")
    catalog_path = Path(args.catalog_db).expanduser().resolve()
    if not catalog_path.is_file():
        raise AblationError(f"catalog not found: {catalog_path}")
    kg_module.configure_catalog(catalog_path)
    catalog = catalog_module.CatalogDatabase(catalog_path)
    countries = catalog.countries()
    valid_isos = {str(row["iso"]).upper() for row in countries}
    image_root = (
        Path(args.image_root).expanduser().resolve()
        if args.image_root
        else Path(meta["manifest"]["image_root"]).expanduser().resolve()
    )
    entries = eval_manifest.load_manifest(
        manifest_path,
        image_root=image_root,
        valid_isos=valid_isos,
        allow_unlabeled_country=True,
    )
    entries_by_id = {entry.id: entry for entry in entries}
    selected_ids = [str(item) for item in meta["manifest"]["selected_ids"]]
    missing = [sample_id for sample_id in selected_ids if sample_id not in entries_by_id]
    if missing:
        raise AblationError("manifest is missing cached ids: " + ", ".join(missing[:5]))
    selected_entries = [entries_by_id[sample_id] for sample_id in selected_ids]

    current_version_info = {
        "model_id": gateway_module.MODEL_ID,
        "model_revision": gateway_module.MODEL_REVISION,
        "prompt_version": gateway_module.PROMPT_VERSION,
        "prompts": list(gateway_module.PROMPTS),
        "catalog_revision": gateway_module.catalog_revision(countries),
        "catalog_revision_stored": catalog.revision("countries"),
    }
    replay_hashes = _source_hashes(WORKTREE_ROOT)
    provenance = {
        "worktree_root": str(WORKTREE_ROOT),
        "revised_scorer_sha256": replay_hashes["scorer"],
        "revised_engine_sha256": replay_hashes["engine"],
        "revised_semantic_sha256": replay_hashes["semantic"],
        "replay_script_sha256": _sha256_file(Path(__file__).resolve()),
    }
    weights = {
        "crops_disabled": True,
        "reference_disabled": True,
        "crop_weight": model_ranking.CROP_WEIGHT,
        "clue_boost_scale": model_ranking.CLUE_BOOST_SCALE,
        "clue_boost_cap": model_ranking.CLUE_BOOST_CAP,
        "semantic_boost_scales": dict(model_ranking.SEMANTIC_BOOST_SCALES),
        "semantic_boost_cap": model_ranking.SEMANTIC_BOOST_CAP,
    }
    analyzer = semantic_module.SemanticClueAnalyzer(countries=countries)

    def rederive_semantic(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
        clues = clue_module.dicts_to_clues(payload)
        return clue_module.clues_to_dicts(analyzer.analyze(clues))

    return ReplayContext(
        entries=selected_entries,
        countries=countries,
        engine_factory=model_ranking.ModelAwareEngine,
        rederive_semantic=rederive_semantic,
        dicts_to_clues=clue_module.dicts_to_clues,
        canonical_country_iso=country_policy.canonical_country_iso,
        version_info=current_version_info,
        provenance=provenance,
        weights=weights,
        manifest_info={
            "path": str(manifest_path),
            "sha256": manifest_sha256,
            "image_root": str(image_root),
            "selected_ids": selected_ids,
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture and replay a country-only StreetCLIP/OCR ablation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="capture full prior and clues")
    capture.add_argument("--source-root", required=True)
    capture.add_argument("--manifest", required=True)
    capture.add_argument("--image-root")
    capture.add_argument("--catalog-db", required=True)
    capture.add_argument("--env-file")
    capture.add_argument("--ids", help="comma-separated sample ids")
    capture.add_argument("--limit", type=int)
    capture.add_argument("--ocr-provider", choices=("mock", "rapidocr"))
    capture.add_argument("--ocr-model-type", choices=("server", "mobile"))
    capture.add_argument("--ai-mode", choices=("mock", "real"))
    capture.add_argument("--streetclip-url")
    capture.add_argument("--container", action="store_true")
    capture.add_argument("--output", required=True)

    replay = subparsers.add_parser("replay", help="score cached arms without model calls")
    replay.add_argument("--cache", required=True)
    replay.add_argument("--manifest", required=True)
    replay.add_argument("--image-root")
    replay.add_argument("--catalog-db", required=True)
    replay.add_argument("--output", required=True)
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "capture":
            return _capture_command(args)
        if args.command == "replay":
            return _replay_command(args)
        raise AblationError(f"unknown command: {args.command}")
    except AblationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
