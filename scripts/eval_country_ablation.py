#!/usr/bin/env python3
"""Country-only paired ablation for StreetCLIP and OCR clues.

Capture imports the evaluator, pipeline and gateway from ``--source-root`` so
the cached clues retain the source semantic analyzer. A partial capture can be
recovered only when every sealed record, current image, source hash and version
still agree; recovery writes a new directory and marks the derived metadata so
it can never be mistaken for an originally published capture. Replay imports
only the current worktree scoring modules, performs no OCR/model/network calls,
and evaluates three arms from the same cached full-image prior:

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
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

SCHEMA_VERSION = 1
CACHE_FILENAME = "cache.jsonl"
META_FILENAME = "meta.json"
RECOVERY_AUDIT_FILENAME = "recovery-audit.json"
RESUME_CHECKPOINT_FILENAME = "resume-checkpoint.json"
PROGRESS_FILENAME = "progress.json"
CAPTURE_SNAPSHOT_FILENAME = "capture-snapshot.json"
PROGRESS_WRITE_ATTEMPTS = 3
PROGRESS_RETRY_DELAY_SECONDS = 0.05
REPORT_FILENAME = "report.json"
EVIDENCE_VIEW = "full_image_only"
TOP_N = 1000
RECOVERED_META_KIND = "country_ablation_recovered_capture"
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
    preflight: Callable[[], dict[str, Any]] | None = None


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
    catalog_db_path: Path | None = None


@dataclass
class RecoveryContext:
    versions: dict[str, Any]
    provenance: dict[str, Any]
    manifest_info: dict[str, Any]
    config: dict[str, Any]
    verification: dict[str, Any]


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


def _ids_sha256(ids: Sequence[str]) -> str:
    payload = ("\n".join(str(sample_id) for sample_id in ids) + "\n").encode("utf-8")
    return _sha256_bytes(payload)


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


def _source_snapshot(root: Path) -> dict[str, str]:
    root = root.expanduser().resolve()
    snapshot: dict[str, str] = {}
    roots = (
        root / "backend" / "app",
        root / "scripts",
        root / "services",
    )
    for base in roots:
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            snapshot[path.relative_to(root).as_posix()] = _sha256_file(path)
    for relative in ("catalog_db.py", "config.py", "backend/__init__.py"):
        path = root / relative
        if path.is_file():
            snapshot[relative] = _sha256_file(path)
    required = {
        "scripts/eval_geolocation.py",
        "scripts/eval_country_ablation.py",
        "backend/app/services/pipeline.py",
        "backend/app/services/model_gateway.py",
        "backend/app/domain/geolocation/model_ranking.py",
        "backend/app/domain/geolocation/engine.py",
        "backend/app/domain/geolocation/analysis/semantic.py",
        "catalog_db.py",
    }
    missing = sorted(required - set(snapshot))
    if missing:
        raise AblationError("source snapshot is missing: " + ", ".join(missing))
    return snapshot


def _validate_capture_settings(settings: Any) -> None:
    if str(getattr(settings, "geo_ai_mode", "")).strip().lower() != "mock":
        raise AblationError("capture requires GEO_AI_MODE=mock")
    if str(getattr(settings, "deepseek_api_key", "") or "").strip():
        raise AblationError("capture requires a blank DEEPSEEK_API_KEY")


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
    _validate_capture_settings(settings)
    if not settings.streetclip_enabled:
        raise AblationError("capture requires STREETCLIP_ENABLED=true")
    if str(settings.ocr_provider).strip().lower() != "rapidocr":
        raise AblationError("capture requires OCR_PROVIDER=rapidocr")

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

    model_revision = gateway_module.MODEL_REVISION
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
        "deepseek_api_key_blank": True,
    }
    return CaptureContext(
        entries=list(entries),
        pipeline=pipeline_module.PipelineService(settings),
        gateway=gateway_module.StreetClipGateway(settings),
        version_info=version_info,
        provenance=provenance,
        manifest_info=manifest_info,
        config_info=config_info,
        preflight=lambda: eg._probe_streetclip(settings, require_city=False),
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
    *,
    resume: dict[str, Any] | None = None,
    snapshot_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(context.config_info)
    provenance = copy.deepcopy(context.provenance)
    if resume is not None:
        config["applies_to_records"] = "new_records_only"
        config["reused_original_llm_mode"] = "unknown"
        provenance["capture_script_applies_to_records"] = "new_records_only"
        provenance["reused_capture_script_sha256"] = "unknown"
    meta = {
        "schema_version": SCHEMA_VERSION,
        "kind": "country_ablation_capture",
        "generated_at": _utc_now(),
        "source_root": context.provenance["source_root"],
        "manifest": context.manifest_info,
        "config": config,
        "versions": context.version_info,
        "provenance": provenance,
        "sample_count": len(records),
        "cache_file": CACHE_FILENAME,
        "cache_sha256": cache_sha256,
    }
    if resume is not None:
        meta["resume"] = copy.deepcopy(resume)
    if snapshot_info is not None:
        meta["capture_snapshot"] = copy.deepcopy(snapshot_info)
    return meta


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        tmp_path.write_bytes(
            (
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            ).encode("utf-8")
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _progress_json_write(path: Path, payload: dict[str, Any]) -> tuple[bool, str | None]:
    last_error: str | None = None
    for attempt in range(PROGRESS_WRITE_ATTEMPTS):
        try:
            _atomic_json_write(path, payload)
            return True, None
        except OSError as exc:
            last_error = type(exc).__name__
            if attempt + 1 < PROGRESS_WRITE_ATTEMPTS:
                time.sleep(PROGRESS_RETRY_DELAY_SECONDS)
        except Exception as exc:
            return False, type(exc).__name__
    return False, last_error or "OSError"


def _capture_input_snapshot(args: argparse.Namespace, source_root: Path) -> dict[str, Any]:
    catalog_path = Path(args.catalog_db).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    return {
        "captured_at": _utc_now(),
        "source_root": str(source_root),
        "source_files": _source_snapshot(source_root),
        "catalog": {"path": str(catalog_path), "sha256": _sha256_file(catalog_path)},
        "manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path)},
    }


def _compare_capture_snapshots(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    before_sources = before.get("source_files") or {}
    after_sources = after.get("source_files") or {}
    changed = sorted(
        path
        for path in set(before_sources) | set(after_sources)
        if before_sources.get(path) != after_sources.get(path)
    )
    catalog_unchanged = (before.get("catalog") or {}).get("sha256") == (
        after.get("catalog") or {}
    ).get("sha256")
    manifest_unchanged = (before.get("manifest") or {}).get("sha256") == (
        after.get("manifest") or {}
    ).get("sha256")
    return {
        "source_unchanged": not changed,
        "catalog_unchanged": catalog_unchanged,
        "manifest_unchanged": manifest_unchanged,
        "changed_source_files": changed,
        "unchanged": not changed and catalog_unchanged and manifest_unchanged,
    }


def _validate_resume_records(
    context: CaptureContext, records: Sequence[dict[str, Any]]
) -> None:
    expected_ids = [str(entry.id) for entry in context.entries]
    actual_ids = [str(record["id"]) for record in records]
    if not actual_ids:
        raise AblationError("resume source cache contains no records")
    if len(set(actual_ids)) != len(actual_ids):
        raise AblationError("resume source cache contains duplicate ids")
    if len(actual_ids) >= len(expected_ids):
        raise AblationError("resume source cache leaves no samples to capture")
    if actual_ids != expected_ids[: len(actual_ids)]:
        raise AblationError("resume source ids are not an exact manifest-order prefix")

    for record, entry in zip(records, context.entries[: len(records)], strict=True):
        sample_id = record["id"]
        if record.get("image_ref") != entry.image_ref:
            raise AblationError(f"resume image reference mismatch for {sample_id}")
        if _sha256_file(entry.image_path) != record.get("raw_image_sha256"):
            raise AblationError(f"resume raw image sha256 mismatch for {sample_id}")
        for field in (
            "model_id",
            "model_revision",
            "prompt_version",
            "catalog_revision",
            "catalog_revision_stored",
        ):
            if record.get(field) != context.version_info.get(field):
                raise AblationError(f"resume {field} mismatch for {sample_id}")
        if record.get("prompt_templates") != context.version_info.get("prompts"):
            raise AblationError(f"resume prompt templates mismatch for {sample_id}")
        for field in (
            "legacy_semantic_sha256",
            "legacy_scorer_sha256",
            "legacy_engine_sha256",
        ):
            if record.get(field) != context.provenance.get(field):
                raise AblationError(f"resume {field} mismatch for {sample_id}")
        evidence = record.get("evidence")
        if not isinstance(evidence, dict):
            raise AblationError(f"resume model evidence is invalid for {sample_id}")
        for field in ("model_revision", "prompt_version", "catalog_revision"):
            if evidence.get(field) != record.get(field):
                raise AblationError(f"resume evidence {field} mismatch for {sample_id}")
        if evidence.get("provider_status") != "ok":
            raise AblationError(f"resume model evidence is not ok for {sample_id}")
        if evidence.get("image_sha256") != record.get("model_image_sha256"):
            raise AblationError(f"resume model image sha256 mismatch for {sample_id}")
        ocr_status = record.get("ocr_status")
        if not isinstance(ocr_status, dict) or ocr_status.get("state") != "ok":
            raise AblationError(f"resume OCR status is not ok for {sample_id}")
        if ocr_status.get("provider") != "rapidocr":
            raise AblationError(f"resume OCR provider mismatch for {sample_id}")
        if ocr_status.get("model_type") != context.config_info.get("ocr_model_type"):
            raise AblationError(f"resume OCR model type mismatch for {sample_id}")


def _resume_info(
    *,
    source_dir: Path,
    source_sha256: str,
    source_meta: dict[str, Any],
    context: CaptureContext,
    reused_count: int,
    total_count: int,
) -> dict[str, Any]:
    source_config = source_meta.get("config") if isinstance(source_meta.get("config"), dict) else {}
    source_provenance = (
        source_meta.get("provenance") if isinstance(source_meta.get("provenance"), dict) else {}
    )
    original_llm_mode = source_config.get("geo_ai_mode")
    if original_llm_mode not in {"mock", "real"}:
        original_llm_mode = "unknown"
    original_script = source_provenance.get("capture_script_sha256")
    if not isinstance(original_script, str) or not original_script:
        original_script = "unknown"
    recovery = source_meta.get("recovery") if isinstance(source_meta.get("recovery"), dict) else None
    return {
        "kind": "country_ablation_resume",
        "source_cache_dir": str(source_dir),
        "source_cache_sha256": source_sha256,
        "source_cache_kind": source_meta.get("kind"),
        "source_sample_count": source_meta.get("sample_count"),
        "source_manifest_sha256": (source_meta.get("manifest") or {}).get("sha256"),
        "source_config": copy.deepcopy(source_config),
        "source_provenance": copy.deepcopy(source_provenance),
        "source_recovery": copy.deepcopy(recovery),
        "original_meta_available": source_meta.get("kind") != RECOVERED_META_KIND,
        "reused_original_llm_mode": original_llm_mode,
        "reused_original_capture_script_sha256": original_script,
        "reused_count": reused_count,
        "new_record_count": total_count - reused_count,
        "total_count": total_count,
        "reused_ids_sha256": _ids_sha256(
            [str(entry.id) for entry in context.entries[:reused_count]]
        ),
        "reuse_mode": "exact_source_cache_bytes_then_append",
        "reused_records_inference_calls": 0,
        "new_capture_llm_mode": context.config_info.get("geo_ai_mode"),
        "new_capture_script_sha256": context.provenance.get("capture_script_sha256"),
        "notice": (
            "Reused records retain unknown original LLM mode and unknown original capture-script hash; "
            "current script/config provenance applies only to newly captured records."
        ),
    }


def _capture_command(args: argparse.Namespace) -> int:
    output_dir = _ensure_new_dir(args.output, label="capture output")
    partial_path = output_dir / (CACHE_FILENAME + ".partial")
    final_cache_path = output_dir / CACHE_FILENAME
    meta_tmp = output_dir / (META_FILENAME + ".tmp")
    progress_path = Path(
        getattr(args, "progress_json", None) or output_dir / PROGRESS_FILENAME
    ).expanduser().resolve()
    resume_from = getattr(args, "resume_from", None)
    checkpoint_path = output_dir / RESUME_CHECKPOINT_FILENAME
    snapshot_path = output_dir / CAPTURE_SNAPSHOT_FILENAME
    started_at = _utc_now()
    written = 0
    reused_records: list[dict[str, Any]] = []
    total_count = 0
    stage = "prepare"
    current_id: str | None = None
    resume_data: dict[str, Any] | None = None
    snapshot_before: dict[str, Any] | None = None
    snapshot_after: dict[str, Any] | None = None
    snapshot_comparison: dict[str, Any] | None = None
    progress_failure_announced = False

    def write_progress(
        status: str,
        *,
        completed: int,
        partial_bytes: int | None = None,
        error_type: str | None = None,
    ) -> None:
        nonlocal progress_failure_announced
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": "country_ablation_capture_progress",
            "status": status,
            "started_at": started_at,
            "updated_at": _utc_now(),
            "stage": stage,
            "output": str(output_dir),
            "partial": str(partial_path),
            "total": total_count,
            "reused": len(reused_records),
            "completed": completed,
            "remaining": max(total_count - completed, 0),
            "new_completed": written,
            "current_id": current_id,
            "partial_bytes": partial_bytes,
            "error_type": error_type,
        }
        ok, error_type_name = _progress_json_write(progress_path, payload)
        if not ok and not progress_failure_announced:
            progress_failure_announced = True
            print(
                f"progress warning: telemetry write failed ({error_type_name}); capture continues",
                file=sys.stderr,
                flush=True,
            )

    def write_checkpoint(
        status: str, *, completed: int, error_type: str | None = None
    ) -> None:
        if resume_data is None:
            return
        _atomic_json_write(
            checkpoint_path,
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "country_ablation_resume_checkpoint",
                "status": status,
                "started_at": started_at,
                "updated_at": _utc_now(),
                "stage": stage,
                "output": str(output_dir),
                "partial": str(partial_path),
                "progress": str(progress_path),
                "total": total_count,
                "reused": len(reused_records),
                "completed": completed,
                "new_completed": written,
                "current_id": current_id,
                "resume": resume_data,
                "snapshot_before": snapshot_before,
                "snapshot_after": snapshot_after,
                "snapshot_comparison": snapshot_comparison,
                "error_type": error_type,
            },
        )

    try:
        context = _load_capture_context(args)
        total_count = len(context.entries)
        if total_count <= 0:
            raise AblationError("capture selection contains no samples")

        if resume_from:
            if getattr(args, "ids", None) or getattr(args, "limit", None):
                raise AblationError("resume capture does not accept --ids or --limit")
            source_dir = Path(resume_from).expanduser().resolve()
            source_records, source_meta = load_cache(source_dir)
            source_cache_path = source_dir / CACHE_FILENAME
            source_sha256 = _sha256_file(source_cache_path)
            if source_sha256 != source_meta.get("cache_sha256"):
                raise AblationError("resume source cache hash mismatch")
            _validate_resume_records(context, source_records)
            reused_records = copy.deepcopy(source_records)
            resume_data = _resume_info(
                source_dir=source_dir,
                source_sha256=source_sha256,
                source_meta=source_meta,
                context=context,
                reused_count=len(reused_records),
                total_count=total_count,
            )
            snapshot_before = _capture_input_snapshot(
                args, Path(args.source_root).expanduser().resolve()
            )
            shutil.copyfile(source_cache_path, partial_path)
            if _sha256_file(partial_path) != source_sha256:
                raise AblationError("resume source bytes were not preserved")
            write_checkpoint("running", completed=len(reused_records))
            write_progress(
                "running",
                completed=len(reused_records),
                partial_bytes=partial_path.stat().st_size,
            )
        else:
            write_progress("running", completed=0)

        stage = "preflight"
        if context.preflight is not None:
            health = context.preflight()
            if not isinstance(health, dict) or not health.get("ready"):
                raise AblationError("StreetCLIP preflight did not report ready=true")
            if health.get("model_revision") != context.version_info.get("model_revision"):
                raise AblationError("StreetCLIP health model revision does not match the gateway")

        stage = "capture"
        remaining_entries = context.entries[len(reused_records):]
        mode = "a" if resume_from else "w"
        with partial_path.open(mode, encoding="utf-8", newline="\n") as handle:
            def write_record(record: dict[str, Any]) -> None:
                nonlocal written, current_id
                handle.write(
                    json.dumps(
                        record, ensure_ascii=False, sort_keys=True, allow_nan=False
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
                written += 1
                current_id = str(record["id"])
                completed = len(reused_records) + written
                write_progress(
                    "running",
                    completed=completed,
                    partial_bytes=handle.tell(),
                )
                print(
                    f"capture progress: {completed}/{total_count} {current_id}",
                    flush=True,
                )

            new_records = capture_records(
                remaining_entries,
                context.pipeline,
                context.gateway,
                context.version_info,
                context.provenance,
                on_record=write_record,
            )
        records = reused_records + new_records

        stage = "finalize"
        expected_ids = [str(entry.id) for entry in context.entries]
        actual_ids = [str(record["id"]) for record in records]
        if len(records) != total_count:
            raise AblationError(
                f"capture produced {len(records)} records; expected exactly {total_count}"
            )
        if len(set(actual_ids)) != len(actual_ids):
            raise AblationError("capture produced duplicate sample ids")
        if actual_ids != expected_ids:
            raise AblationError("capture records do not match manifest order")
        if resume_from:
            snapshot_after = _capture_input_snapshot(
                args, Path(args.source_root).expanduser().resolve()
            )
            snapshot_comparison = _compare_capture_snapshots(
                snapshot_before or {}, snapshot_after
            )
            _atomic_json_write(
                snapshot_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "country_ablation_capture_snapshot",
                    "started_at": started_at,
                    "finished_at": _utc_now(),
                    "before": snapshot_before,
                    "after": snapshot_after,
                    "comparison": snapshot_comparison,
                },
            )
            if not snapshot_comparison["unchanged"]:
                raise AblationError("source/catalog/manifest changed during resume capture")

        cache_sha256 = _sha256_file(partial_path)
        os.replace(partial_path, final_cache_path)
        snapshot_info = None
        if resume_from:
            snapshot_info = {
                "file": CAPTURE_SNAPSHOT_FILENAME,
                "sha256": _sha256_file(snapshot_path),
            }
        meta = _build_capture_meta(
            context,
            records,
            cache_sha256,
            resume=resume_data,
            snapshot_info=snapshot_info,
        )
        meta_tmp.write_bytes(
            (
                json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            ).encode("utf-8")
        )
        os.replace(meta_tmp, output_dir / META_FILENAME)
        write_checkpoint("completed", completed=total_count)
        write_progress(
            "completed",
            completed=total_count,
            partial_bytes=final_cache_path.stat().st_size,
        )
    except BaseException as exc:
        if meta_tmp.exists():
            meta_tmp.unlink()
        completed = len(reused_records) + written
        if resume_from and snapshot_before is not None:
            try:
                snapshot_after = _capture_input_snapshot(
                    args, Path(args.source_root).expanduser().resolve()
                )
                snapshot_comparison = _compare_capture_snapshots(
                    snapshot_before, snapshot_after
                )
                _atomic_json_write(
                    snapshot_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": "country_ablation_capture_snapshot",
                        "started_at": started_at,
                        "finished_at": _utc_now(),
                        "before": snapshot_before,
                        "after": snapshot_after,
                        "comparison": snapshot_comparison,
                    },
                )
            except Exception:
                pass
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        error_type = type(exc).__name__
        try:
            write_checkpoint(status, completed=completed, error_type=error_type)
        except Exception:
            pass
        try:
            write_progress(
                status,
                completed=completed,
                partial_bytes=partial_path.stat().st_size if partial_path.exists() else None,
                error_type=error_type,
            )
        except Exception:
            pass
        if partial_path.exists():
            print(
                f"capture stopped: {completed} completed samples preserved at {partial_path}",
                file=sys.stderr,
                flush=True,
            )
        elif final_cache_path.exists():
            print(
                f"capture stopped after cache publish: {final_cache_path} has no final meta",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                "capture stopped before any completed sample was written",
                file=sys.stderr,
                flush=True,
            )
        raise
    print(f"capture: {output_dir}")
    return 0


def _read_cache_records(cache_path: str | Path) -> list[dict[str, Any]]:
    path = Path(cache_path)
    if not path.is_file():
        raise AblationError(f"cache file not found: {path}")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
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
    if not records:
        raise AblationError("cache contains no records")
    return records


def _validate_recovered_meta(
    meta: dict[str, Any],
    root: Path,
    *,
    cache_sha256: str,
    selected_ids: Sequence[str],
) -> None:
    recovery = meta.get("recovery")
    if not isinstance(recovery, dict):
        raise AblationError("recovered cache meta has no recovery provenance")
    if recovery.get("derived_meta") is not True:
        raise AblationError("recovered cache meta is not marked as derived")
    if recovery.get("source_sha256") != cache_sha256:
        raise AblationError("recovered cache source sha256 mismatch")
    if recovery.get("source_line_count") != meta.get("sample_count"):
        raise AblationError("recovered cache source line count mismatch")
    if recovery.get("selected_ids_sha256") != _ids_sha256(selected_ids):
        raise AblationError("recovered cache selected ids mismatch")
    audit_file = recovery.get("audit_file")
    if audit_file != RECOVERY_AUDIT_FILENAME:
        raise AblationError("recovered cache audit filename is invalid")
    audit_path = root / RECOVERY_AUDIT_FILENAME
    expected_audit_sha = _require_sha256(
        recovery.get("audit_sha256"), "recovery audit sha256"
    )
    if not audit_path.is_file() or _sha256_file(audit_path) != expected_audit_sha:
        raise AblationError("recovery audit hash mismatch")
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AblationError("recovery audit is invalid JSON") from exc
    source = audit.get("source")
    if (
        not isinstance(source, dict)
        or source.get("sha256") != cache_sha256
        or source.get("line_count") != meta.get("sample_count")
    ):
        raise AblationError("recovery audit does not match cache meta")


def _validate_resume_meta(meta: dict[str, Any], root: Path) -> None:
    resume = meta.get("resume")
    if not isinstance(resume, dict):
        raise AblationError("resume capture meta has no resume provenance")
    reused = resume.get("reused_count")
    new = resume.get("new_record_count")
    total = resume.get("total_count")
    if (
        not isinstance(reused, int)
        or not isinstance(new, int)
        or reused <= 0
        or new <= 0
        or total != reused + new
        or total != meta.get("sample_count")
    ):
        raise AblationError("resume capture counts are inconsistent")
    manifest_sha = (meta.get("manifest") or {}).get("sha256")
    if resume.get("source_manifest_sha256") != manifest_sha:
        raise AblationError("resume source manifest does not match final manifest")
    if resume.get("new_capture_llm_mode") != "mock":
        raise AblationError("resume new records were not captured with mock mode")
    if resume.get("reused_original_llm_mode") not in {"mock", "real", "unknown"}:
        raise AblationError("resume reused LLM mode is invalid")
    if meta.get("config", {}).get("applies_to_records") != "new_records_only":
        raise AblationError("resume config scope is not explicit")
    snapshot = meta.get("capture_snapshot")
    if not isinstance(snapshot, dict):
        raise AblationError("resume capture meta has no capture snapshot")
    snapshot_file = snapshot.get("file")
    if snapshot_file != CAPTURE_SNAPSHOT_FILENAME:
        raise AblationError("capture snapshot filename is invalid")
    snapshot_sha = _require_sha256(snapshot.get("sha256"), "capture snapshot sha256")
    snapshot_path = root / CAPTURE_SNAPSHOT_FILENAME
    if not snapshot_path.is_file() or _sha256_file(snapshot_path) != snapshot_sha:
        raise AblationError("capture snapshot hash mismatch")


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
    supported_kinds = {"country_ablation_capture", RECOVERED_META_KIND}
    if meta.get("schema_version") != SCHEMA_VERSION or meta.get("kind") not in supported_kinds:
        raise AblationError("unsupported capture cache schema")
    expected_cache_sha256 = _require_sha256(meta.get("cache_sha256"), "cache sha256")
    if _sha256_file(cache_path) != expected_cache_sha256:
        raise AblationError("cache.jsonl hash mismatch")
    if meta.get("cache_file") != CACHE_FILENAME:
        raise AblationError("cache meta has an unsupported cache filename")

    records = _read_cache_records(cache_path)

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
    if meta.get("kind") == RECOVERED_META_KIND:
        _validate_recovered_meta(
            meta,
            root,
            cache_sha256=expected_cache_sha256,
            selected_ids=expected_ids,
        )
    elif meta.get("resume") is not None:
        _validate_resume_meta(meta, root)
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


def _consistent_record_value(records: Sequence[dict[str, Any]], field: str) -> Any:
    values = {_json_text(record.get(field)) for record in records}
    if len(values) != 1:
        raise AblationError(f"partial cache has inconsistent {field}")
    return copy.deepcopy(records[0].get(field))


def _observed_ocr_config(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    statuses = [record.get("ocr_status") for record in records]
    if any(not isinstance(status, dict) for status in statuses):
        raise AblationError("partial cache has invalid OCR status")
    observed = {
        field: {
            _json_text(status.get(field)): status.get(field) for status in statuses
        }
        for field in ("provider", "state", "engine", "model_type", "model_version", "models")
    }
    for field, values in observed.items():
        if len(values) != 1:
            raise AblationError(f"partial cache has inconsistent OCR {field}")
    first = statuses[0]
    if first.get("provider") != "rapidocr":
        raise AblationError("recovery requires rapidocr records")
    if first.get("state") != "ok" or first.get("available") is not True:
        raise AblationError("recovery requires complete successful OCR status")
    if not isinstance(first.get("model_type"), str) or not first["model_type"]:
        raise AblationError("partial cache has no OCR model type")
    if not isinstance(first.get("model_version"), str) or not first["model_version"]:
        raise AblationError("partial cache has no OCR model version")
    if not isinstance(first.get("models"), dict) or not first["models"]:
        raise AblationError("partial cache has no OCR model inventory")
    return {
        "ocr_provider": first["provider"],
        "ocr_state": first["state"],
        "ocr_engine": first.get("engine"),
        "ocr_model_type": first["model_type"],
        "ocr_model_version": first["model_version"],
        "ocr_models": copy.deepcopy(first["models"]),
    }


def _load_recovery_context(
    args: argparse.Namespace, records: Sequence[dict[str, Any]]
) -> RecoveryContext:
    source_root = Path(args.source_root).expanduser().resolve()
    if not source_root.is_dir():
        raise AblationError(f"source root not found: {source_root}")
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise AblationError(f"manifest not found: {manifest_path}")
    catalog_path = Path(args.catalog_db).expanduser().resolve()
    if not catalog_path.is_file():
        raise AblationError(f"catalog not found: {catalog_path}")

    sys.path.insert(0, str(WORKTREE_ROOT))
    eval_manifest = importlib.import_module("scripts.eval_manifest")
    gateway_module = importlib.import_module("backend.app.services.model_gateway")
    catalog_module = importlib.import_module("catalog_db")
    kg_module = importlib.import_module("backend.app.domain.geolocation.knowledge_graph")
    for module, label in (
        (eval_manifest, "worktree eval manifest"),
        (gateway_module, "worktree model gateway constants"),
        (catalog_module, "worktree catalog"),
        (kg_module, "worktree knowledge graph"),
    ):
        _assert_module_from(module, WORKTREE_ROOT, label)

    manifest_sha256 = _sha256_file(manifest_path)
    kg_module.configure_catalog(catalog_path)
    catalog = catalog_module.CatalogDatabase(catalog_path)
    countries = catalog.countries()
    valid_isos = {str(row["iso"]).upper() for row in countries}
    image_root = (
        Path(args.image_root).expanduser().resolve()
        if args.image_root
        else manifest_path.parent
    )
    entries = eval_manifest.load_manifest(
        manifest_path,
        image_root=image_root,
        valid_isos=valid_isos,
        allow_unlabeled_country=True,
    )
    entries_by_id = {entry.id: entry for entry in entries}
    selected_ids = [str(record["id"]) for record in records]
    missing = [sample_id for sample_id in selected_ids if sample_id not in entries_by_id]
    if missing:
        raise AblationError("manifest is missing cached ids: " + ", ".join(missing[:5]))
    selected_entries = [entries_by_id[sample_id] for sample_id in selected_ids]

    versions = {
        "model_id": _consistent_record_value(records, "model_id"),
        "model_revision": _consistent_record_value(records, "model_revision"),
        "prompt_version": _consistent_record_value(records, "prompt_version"),
        "prompts": _consistent_record_value(records, "prompt_templates"),
        "catalog_revision": _consistent_record_value(records, "catalog_revision"),
        "catalog_revision_stored": _consistent_record_value(
            records, "catalog_revision_stored"
        ),
    }
    current_versions = {
        "model_id": gateway_module.MODEL_ID,
        "model_revision": gateway_module.MODEL_REVISION,
        "prompt_version": gateway_module.PROMPT_VERSION,
        "prompts": list(gateway_module.PROMPTS),
        "catalog_revision": gateway_module.catalog_revision(countries),
        "catalog_revision_stored": catalog.revision("countries"),
    }
    for field, expected in current_versions.items():
        if versions.get(field) != expected:
            raise AblationError(f"partial cache {field} does not match current inputs")

    source_hashes = _source_hashes(source_root)
    source_fields = {
        "legacy_semantic_sha256": "semantic",
        "legacy_scorer_sha256": "scorer",
        "legacy_engine_sha256": "engine",
    }
    for record_field, source_key in source_fields.items():
        expected = _consistent_record_value(records, record_field)
        if expected != source_hashes[source_key]:
            raise AblationError(f"partial cache {record_field} does not match source root")

    raw_hash_matches = 0
    evidence_matches = 0
    for record, entry in zip(records, selected_entries, strict=True):
        if record.get("image_ref") != entry.image_ref:
            raise AblationError(f"image reference mismatch for {record['id']}")
        if _sha256_file(entry.image_path) != record["raw_image_sha256"]:
            raise AblationError(f"raw image sha256 mismatch for {record['id']}")
        raw_hash_matches += 1
        evidence = record["evidence"]
        if (
            evidence.get("provider_status") != "ok"
            or evidence.get("image_sha256") != record.get("model_image_sha256")
        ):
            raise AblationError(f"model evidence mismatch for {record['id']}")
        evidence_matches += 1

    _validate_cache_versions(
        records,
        {
            "versions": versions,
            "provenance": {
                "legacy_semantic_sha256": source_hashes["semantic"],
                "legacy_scorer_sha256": source_hashes["scorer"],
                "legacy_engine_sha256": source_hashes["engine"],
            },
        },
    )
    observed_config = _observed_ocr_config(records)
    manifest_ids = [entry.id for entry in entries]
    manifest_order_prefix = manifest_ids[: len(selected_ids)] == selected_ids
    catalog_file_sha256 = _sha256_file(catalog_path)
    provenance = {
        "origin": "derived_from_sealed_partial_records",
        "legacy_semantic_sha256": source_hashes["semantic"],
        "legacy_scorer_sha256": source_hashes["scorer"],
        "legacy_engine_sha256": source_hashes["engine"],
        "verified_source_root": str(source_root),
        "verified_eval_helper_sha256": source_hashes["eval_helper"],
    }
    manifest_info = {
        "path": str(manifest_path),
        "sha256": manifest_sha256,
        "image_root": str(image_root),
        "selected_ids": selected_ids,
        "selected_ids_origin": "derived_from_validated_record_order",
        "selected_ids_sha256": _ids_sha256(selected_ids),
        "manifest_order_prefix": manifest_order_prefix,
    }
    verification = {
        "record_count": len(records),
        "unique_ids": len({record["id"] for record in records}),
        "sealed_records": len(records),
        "raw_image_hashes_verified": raw_hash_matches,
        "model_evidence_verified": evidence_matches,
        "manifest_order_prefix": manifest_order_prefix,
        "manifest_sha256": manifest_sha256,
        "catalog_file_sha256": catalog_file_sha256,
        "source_root": str(source_root),
        "source_hashes": source_hashes,
        "current_versions": current_versions,
        "observed_config": observed_config,
        "limitations": [
            "No original capture meta.json was present; meta.json is explicitly derived.",
            "Record call counts and the original invocation's LLM mode were not recorded per sample.",
            "Sealed record evidence is trusted only together with matching image, manifest, source and catalog hashes.",
            "A manifest-order prefix is not a random or blind holdout sample.",
        ],
    }
    return RecoveryContext(
        versions=versions,
        provenance=provenance,
        manifest_info=manifest_info,
        config=observed_config,
        verification=verification,
    )


def _recover_command(args: argparse.Namespace) -> int:
    partial_path = Path(args.partial).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if output_dir.exists():
        raise AblationError(f"recovery output directory already exists: {output_dir}")
    if not partial_path.is_file():
        raise AblationError(f"partial cache not found: {partial_path}")
    expected_sha256 = _require_sha256(args.expected_sha256, "expected partial sha256")
    if args.expected_count is not None and args.expected_count <= 0:
        raise AblationError("expected count must be positive")

    source_stat = partial_path.stat()
    source_sha256 = _sha256_file(partial_path)
    if source_sha256 != expected_sha256:
        raise AblationError("partial cache sha256 does not match expected value")
    records = _read_cache_records(partial_path)
    if args.expected_count is not None and len(records) != args.expected_count:
        raise AblationError(
            f"partial cache count {len(records)} does not match expected {args.expected_count}"
        )

    context = _load_recovery_context(args, records)
    selected_ids = [str(record["id"]) for record in records]
    if _sha256_file(partial_path) != source_sha256:
        raise AblationError("partial cache changed during recovery")
    if _sha256_file(Path(args.manifest)) != context.manifest_info["sha256"]:
        raise AblationError("manifest changed during recovery")
    catalog_path = Path(args.catalog_db).expanduser().resolve()
    if _sha256_file(catalog_path) != context.verification["catalog_file_sha256"]:
        raise AblationError("catalog changed during recovery")
    if _source_hashes(Path(args.source_root).expanduser().resolve()) != context.verification["source_hashes"]:
        raise AblationError("source files changed during recovery")

    recovered_at = _utc_now()
    audit = {
        "schema_version": SCHEMA_VERSION,
        "kind": "country_ablation_recovery_audit",
        "recovered_at": recovered_at,
        "source": {
            "path": str(partial_path),
            "sha256": source_sha256,
            "size_bytes": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
            "line_count": len(records),
            "expected_sha256": expected_sha256,
            "expected_count": args.expected_count,
            "original_meta_available": False,
        },
        "validation": {
            key: value
            for key, value in context.verification.items()
            if not key.startswith("_")
        },
        "manifest": context.manifest_info,
        "versions": context.versions,
        "provenance": context.provenance,
    }
    audit_payload = (
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    audit_sha256 = _sha256_bytes(audit_payload.encode("utf-8"))
    recovery = {
        "derived_meta": True,
        "source_path": str(partial_path),
        "source_sha256": source_sha256,
        "source_size_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_line_count": len(records),
        "selected_ids_sha256": _ids_sha256(selected_ids),
        "manifest_order_prefix": context.verification["manifest_order_prefix"],
        "audit_file": RECOVERY_AUDIT_FILENAME,
        "audit_sha256": audit_sha256,
        "original_meta_available": False,
        "limitations": context.verification["limitations"],
    }
    meta = {
        "schema_version": SCHEMA_VERSION,
        "kind": RECOVERED_META_KIND,
        "generated_at": recovered_at,
        "manifest": context.manifest_info,
        "config": context.config,
        "versions": context.versions,
        "provenance": context.provenance,
        "sample_count": len(records),
        "cache_file": CACHE_FILENAME,
        "cache_sha256": source_sha256,
        "recovery": recovery,
    }

    output_dir.mkdir(parents=True)
    recovered_cache_path = output_dir / CACHE_FILENAME
    shutil.copyfile(partial_path, recovered_cache_path)
    if _sha256_file(recovered_cache_path) != source_sha256:
        raise AblationError("recovered cache copy hash mismatch")
    (output_dir / RECOVERY_AUDIT_FILENAME).write_bytes(audit_payload.encode("utf-8"))
    (output_dir / META_FILENAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    load_cache(output_dir)
    print(f"recovered: {output_dir}")
    return 0


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


def _score_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


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
        top1_score = _score_number(getattr(scores[0], "score", None))
        top2_score = _score_number(getattr(scores[1], "score", None)) if len(scores) > 1 else None
        margin = (
            round(top1_score - top2_score, 6)
            if top1_score is not None and top2_score is not None
            else None
        )
        return {
            "predicted": predicted,
            "top5": predicted[:5],
            "top1_score": top1_score,
            "top2_score": top2_score,
            "margin": margin,
            "error": None,
        }
    except Exception as exc:  # per-arm failure is counted as a miss and reported
        return {
            "predicted": [],
            "top5": [],
            "top1_score": None,
            "top2_score": None,
            "margin": None,
            "error": type(exc).__name__,
        }


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


def _margin_diagnostics(
    rows: Sequence[dict[str, Any]], *, high: float, medium: float
) -> dict[str, Any]:
    if high <= medium or medium < 0:
        raise AblationError("margin thresholds must satisfy high > medium >= 0")
    arms: dict[str, Any] = {}
    for arm in ARM_NAMES:
        margin_available = 0
        high_eligible = 0
        false_high_ids: list[str] = []
        high_correct = 0
        medium_eligible = 0
        false_medium_ids: list[str] = []
        leader_changed_excluded = 0
        for row in rows:
            arm_result = row["arms"][arm]
            model_result = row["arms"]["model_only"]
            arm_margin = _score_number(arm_result.get("margin"))
            model_margin = _score_number(model_result.get("margin"))
            if arm_margin is None or model_margin is None:
                continue
            margin_available += 1
            effective_margin = min(arm_margin, model_margin)
            model_top = (model_result.get("top5") or [None])[0]
            arm_top = (arm_result.get("top5") or [None])[0]
            leader_changed = model_top != arm_top
            correct_top1 = arm_result.get("rank") == 1
            if effective_margin >= medium:
                medium_eligible += 1
                if not correct_top1:
                    false_medium_ids.append(row["id"])
            if effective_margin >= high:
                if leader_changed:
                    leader_changed_excluded += 1
                    continue
                high_eligible += 1
                if correct_top1:
                    high_correct += 1
                else:
                    false_high_ids.append(row["id"])
        arms[arm] = {
            "margin_available": margin_available,
            "high_eligible": high_eligible,
            "high_correct": high_correct,
            "false_high": len(false_high_ids),
            "false_high_rate": (
                round(len(false_high_ids) / high_eligible, 6) if high_eligible else None
            ),
            "false_high_ids": false_high_ids,
            "high_leader_changed_excluded": leader_changed_excluded,
            "medium_or_better": medium_eligible,
            "medium_or_better_false_top1": len(false_medium_ids),
            "medium_or_better_false_top1_rate": (
                round(len(false_medium_ids) / medium_eligible, 6)
                if medium_eligible
                else None
            ),
            "medium_or_better_false_top1_ids": false_medium_ids,
        }
    return {
        "thresholds": {"high": high, "medium": medium},
        "effective_margin": "min(model_only_margin, arm_margin)",
        "high_extra_rule": "arm top1 must equal model-only top1",
        "interpretation": "descriptive margin diagnostic, not a probability or validated policy",
        "arms": arms,
    }


def replay_records(
    records: Sequence[dict[str, Any]],
    meta: dict[str, Any],
    context: ReplayContext,
    *,
    margin_thresholds: dict[str, float] | None = None,
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
                        "top1_score": arms[arm]["top1_score"],
                        "margin": arms[arm]["margin"],
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
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "country_ablation_replay",
        "generated_at": _utc_now(),
        "cache": {
            "path": str(Path(meta["_cache_dir"]).resolve()) if meta.get("_cache_dir") else None,
            "kind": meta["kind"],
            "sha256": meta["cache_sha256"],
            "sample_count": meta["sample_count"],
            "recovered": meta["kind"] == RECOVERED_META_KIND,
            "recovery": copy.deepcopy(meta.get("recovery")),
            "resume": copy.deepcopy(meta.get("resume")),
            "capture_snapshot": copy.deepcopy(meta.get("capture_snapshot")),
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
    if margin_thresholds is not None:
        report["margin_thresholds"] = copy.deepcopy(margin_thresholds)
        report["margin_diagnostics"] = _margin_diagnostics(
            rows,
            high=float(margin_thresholds["high"]),
            medium=float(margin_thresholds["medium"]),
        )
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Country ablation",
        "",
        f"- Evidence view: `{report['evidence_view']}`",
        f"- Labelled attempted: `{report['denominator']['attempted_labelled']}`",
        f"- Unknown label excluded: `{report['denominator']['unknown_label_excluded']}`",
    ]
    if report["cache"].get("recovered"):
        lines.extend(
            [
                "- Cache provenance: **recovered validated partial**, not an original completed capture.",
                f"- Partial rows: `{report['cache']['sample_count']}`",
            ]
        )
        if report["cache"]["recovery"].get("manifest_order_prefix"):
            lines.append(
                "- Selection: manifest-order prefix; not a random or blind holdout sample."
            )
    lines.extend(["", "| arm | top1 | top3 | top5 |", "|---|---:|---:|---:|"])
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
    diagnostics = report.get("margin_diagnostics")
    if diagnostics:
        thresholds = diagnostics["thresholds"]
        lines.extend(
            [
                "",
                "## Margin diagnostic",
                "",
                f"- High `>= {thresholds['high']}`; medium `>= {thresholds['medium']}`.",
                f"- Effective margin: `{diagnostics['effective_margin']}`.",
                "- Descriptive only; thresholds were supplied, not tuned on this subset.",
                "",
                "| arm | margin available | high eligible | false high | false-high rate | medium+ false top1 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for arm in ARM_NAMES:
            item = diagnostics["arms"][arm]
            lines.append(
                f"| {arm} | {item['margin_available']} | {item['high_eligible']} | "
                f"{item['false_high']} | {item['false_high_rate']} | "
                f"{item['medium_or_better_false_top1']} |"
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
        "margin_thresholds": report.get("margin_thresholds"),
    }
    (output_dir / META_FILENAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _margin_thresholds_from_args(args: argparse.Namespace) -> dict[str, float] | None:
    high = getattr(args, "high_margin", None)
    medium = getattr(args, "medium_margin", None)
    if high is None and medium is None:
        return None
    if high is None or medium is None:
        raise AblationError("both high and medium margin thresholds are required")
    if medium < 0 or high <= medium:
        raise AblationError("margin thresholds must satisfy high > medium >= 0")
    return {"high": float(high), "medium": float(medium)}


def _assert_replay_inputs_unchanged(context: ReplayContext) -> None:
    if context.provenance.get("worktree_root") != str(WORKTREE_ROOT):
        return
    hashes = _source_hashes(WORKTREE_ROOT)
    expected = {
        "scorer": context.provenance.get("revised_scorer_sha256"),
        "engine": context.provenance.get("revised_engine_sha256"),
        "semantic": context.provenance.get("revised_semantic_sha256"),
    }
    for name, value in expected.items():
        if hashes[name] != value:
            raise AblationError(f"replay {name} source changed during execution")
    if _sha256_file(Path(__file__).resolve()) != context.provenance.get(
        "replay_script_sha256"
    ):
        raise AblationError("replay script changed during execution")
    manifest_path = Path(context.manifest_info["path"])
    if _sha256_file(manifest_path) != context.manifest_info["sha256"]:
        raise AblationError("manifest changed during replay")
    catalog_sha256 = context.provenance.get("catalog_file_sha256")
    if (
        catalog_sha256
        and context.catalog_db_path is not None
        and _sha256_file(context.catalog_db_path) != catalog_sha256
    ):
        raise AblationError("catalog changed during replay")


def _replay_command(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if output_dir == cache_dir or output_dir.is_relative_to(cache_dir) or cache_dir.is_relative_to(output_dir):
        raise AblationError("replay output must be separate from the cache directory")
    _ensure_new_dir(output_dir, label="replay output")
    records, meta = load_cache(cache_dir)
    meta = {**meta, "_cache_dir": str(cache_dir)}
    context = _load_replay_context(args, meta)
    report = replay_records(
        records,
        meta,
        context,
        margin_thresholds=_margin_thresholds_from_args(args),
    )
    _assert_replay_inputs_unchanged(context)
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
        "catalog_file_sha256": _sha256_file(catalog_path),
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
        catalog_db_path=catalog_path,
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
    capture.add_argument(
        "--resume-from",
        help="validated capture directory whose complete prefix must be reused byte-for-byte",
    )
    capture.add_argument(
        "--progress-json",
        help="atomic progress JSON path (defaults to <output>/progress.json)",
    )
    capture.add_argument("--output", required=True)

    replay = subparsers.add_parser("replay", help="score cached arms without model calls")
    replay.add_argument("--cache", required=True)
    replay.add_argument("--manifest", required=True)
    replay.add_argument("--image-root")
    replay.add_argument("--catalog-db", required=True)
    replay.add_argument(
        "--high-margin",
        type=float,
        help="explicit high-margin threshold for descriptive false-high diagnostics",
    )
    replay.add_argument(
        "--medium-margin",
        type=float,
        help="explicit medium-margin threshold for descriptive diagnostics",
    )
    replay.add_argument("--output", required=True)

    recover = subparsers.add_parser(
        "recover-partial",
        help="derive a replayable cache from a fully validated partial capture",
    )
    recover.add_argument("--partial", required=True)
    recover.add_argument("--expected-sha256", required=True)
    recover.add_argument("--expected-count", type=int)
    recover.add_argument("--source-root", required=True)
    recover.add_argument("--manifest", required=True)
    recover.add_argument("--image-root")
    recover.add_argument("--catalog-db", required=True)
    recover.add_argument("--output", required=True)
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
        if args.command == "recover-partial":
            return _recover_command(args)
        raise AblationError(f"unknown command: {args.command}")
    except AblationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
