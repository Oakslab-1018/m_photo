from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import eval_country_ablation as ea


SHA_A = "a" * 64
SHA_B = "b" * 64


@dataclass
class Entry:
    id: str
    image_path: Path
    image_ref: str
    country_iso: str | None


@dataclass
class FakeScore:
    iso: str
    score: float | None = None


class FakePipeline:
    def __init__(self, clues: list[dict[str, Any]]) -> None:
        self.clues = clues
        self.analyze_calls = 0
        self.status_calls = 0

    def analyze(self, image_path: str) -> list[dict[str, Any]]:
        self.analyze_calls += 1
        return copy.deepcopy(self.clues)

    def ocr_status(self) -> dict[str, Any]:
        self.status_calls += 1
        return {
            "provider": "rapidocr",
            "state": "ok",
            "available": True,
            "engine": "onnxruntime",
            "model_type": "server",
            "model_version": "PP-OCRv5",
            "models": {
                "det": "det-server",
                "rec": "rec-server",
                "cls": "cls-server",
            },
            "line_count": 1,
        }


class FakeGateway:
    def __init__(self, evidence: dict[str, Any] | Any) -> None:
        self.evidence = evidence
        self.calls = 0

    def rank_image(self, image_path: str | Path) -> dict[str, Any]:
        self.calls += 1
        value = self.evidence(Path(image_path)) if callable(self.evidence) else self.evidence
        return copy.deepcopy(value)


class FakeEngine:
    calls: list[dict[str, Any]] = []

    def __init__(self, plan: dict[str, dict[str, Any]]) -> None:
        self.plan = plan
        self.evidence: dict[str, Any] | None = None

    def infer(self, clues: list[dict[str, Any]], top_n: int) -> list[FakeScore]:
        assert self.evidence is not None
        sample_id = self.evidence["sample_id"]
        revised = any(
            isinstance(clue.get("extra"), dict) and clue["extra"].get("revised")
            for clue in clues
        )
        legacy = any(clue.get("source") == "semantic" for clue in clues)
        arm = "model_only" if not clues else ("revised_ocr" if revised else "legacy_ocr" if legacy else "model_only")
        self.calls.append(
            {
                "sample_id": sample_id,
                "arm": arm,
                "evidence": copy.deepcopy(self.evidence),
                "clues": copy.deepcopy(clues),
                "top_n": top_n,
            }
        )
        predicted = self.plan[sample_id][arm]
        if predicted == "FAIL":
            raise RuntimeError("synthetic failure")
        return [
            FakeScore(iso, round(0.9 - index * 0.1, 6))
            for index, iso in enumerate(predicted)
        ]


def _versions(**overrides: Any) -> dict[str, Any]:
    values = {
        "model_id": "geolocal/StreetCLIP",
        "model_revision": "model-r1",
        "prompt_version": "prompt-v1",
        "prompts": ["a photo from {country}"],
        "catalog_revision": "catalog-r1",
        "catalog_revision_stored": "stored-r1",
    }
    values.update(overrides)
    return values


def _evidence(versions: dict[str, Any], sample_id: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "model_revision": versions["model_revision"],
        "prompt_version": versions["prompt_version"],
        "catalog_revision": versions["catalog_revision"],
        "image_sha256": SHA_A,
        "provider_status": "ok",
        "rankings": [
            {"iso": "AA", "raw_cosine": 0.9},
            {"iso": "BB", "raw_cosine": 0.8},
            {"iso": "CC", "raw_cosine": 0.7},
            {"iso": "DD", "raw_cosine": 0.6},
            {"iso": "EE", "raw_cosine": 0.5},
        ],
        "margin": 0.1,
        "uncertain": False,
        "crops": [
            {
                "region": {"region_id": "crop-1"},
                "rankings": [{"iso": "AA", "raw_cosine": 1.0}],
            }
        ],
        "reference": {"matches": [], "regions": [], "coverage": []},
    }


def _provenance() -> dict[str, Any]:
    return {
        "source_root": "D:/source",
        "legacy_semantic_sha256": SHA_A,
        "legacy_scorer_sha256": SHA_B,
        "legacy_engine_sha256": SHA_A,
        "legacy_eval_helper_sha256": SHA_B,
        "capture_script_sha256": SHA_A,
    }


def _make_entry(tmp_path: Path, sample_id: str, country_iso: str | None = "US") -> Entry:
    image = tmp_path / f"{sample_id}.jpg"
    image.write_bytes(f"image-{sample_id}".encode())
    return Entry(sample_id, image, image.name, country_iso)


def _legacy_clues() -> list[dict[str, Any]]:
    return [
        {
            "dimension": "language",
            "value": "latin",
            "confidence": 0.9,
            "source": "text_language",
            "extra": {"origin_source": "ocr"},
        },
        {
            "dimension": "place_name",
            "value": "old-semantic",
            "confidence": 0.9,
            "source": "semantic",
            "extra": {"validated": True},
        },
    ]


def _revised_semantic(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assert all(clue.get("source") != "semantic" for clue in payload)
    return [
        {
            "dimension": "place_name",
            "value": "new-semantic",
            "confidence": 0.95,
            "source": "semantic",
            "extra": {"validated": True, "revised": True},
        }
    ]


def _capture(tmp_path: Path, entries: list[Entry], clues: list[dict[str, Any]] | None = None):
    versions = _versions()
    pipeline = FakePipeline(_legacy_clues() if clues is None else clues)
    gateway = FakeGateway(lambda image_path: _evidence(versions, image_path.stem))
    records = ea.capture_records(entries, pipeline, gateway, versions, _provenance())
    return records, pipeline, gateway, versions


def _cache_context(
    entries: list[Entry],
    versions: dict[str, Any],
    manifest_sha256: str,
) -> ea.CaptureContext:
    return ea.CaptureContext(
        entries=entries,
        pipeline=SimpleNamespace(),
        gateway=SimpleNamespace(),
        version_info=versions,
        provenance=_provenance(),
        manifest_info={
            "path": "manifest.jsonl",
            "sha256": manifest_sha256,
            "image_root": str(entries[0].image_path.parent),
            "selected_ids": [entry.id for entry in entries],
        },
        config_info={},
    )


def _write_cache(
    root: Path,
    entries: list[Entry],
    records: list[dict[str, Any]],
    versions: dict[str, Any],
) -> dict[str, Any]:
    root.mkdir()
    cache_text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    cache_path = root / ea.CACHE_FILENAME
    cache_path.write_text(cache_text, encoding="utf-8")
    manifest_sha = "c" * 64
    context = _cache_context(entries, versions, manifest_sha)
    meta = ea._build_capture_meta(context, records, ea._sha256_file(cache_path))
    (root / ea.META_FILENAME).write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return meta


def _replay_context(
    entries: list[Entry],
    versions: dict[str, Any],
    plan: dict[str, dict[str, Any]],
    *,
    manifest_sha256: str = "c" * 64,
) -> ea.ReplayContext:
    def engine_factory():
        return FakeEngine(plan)

    return ea.ReplayContext(
        entries=entries,
        countries=[],
        engine_factory=engine_factory,
        rederive_semantic=_revised_semantic,
        dicts_to_clues=lambda payload: payload,
        canonical_country_iso=lambda value: str(value).upper() if value else None,
        version_info=versions,
        provenance={
            "worktree_root": "D:/worktree",
            "revised_scorer_sha256": SHA_B,
            "revised_engine_sha256": SHA_A,
            "revised_semantic_sha256": SHA_B,
            "replay_script_sha256": SHA_A,
        },
        weights={"crops_disabled": True, "reference_disabled": True},
        manifest_info={
            "path": "manifest.jsonl",
            "sha256": manifest_sha256,
            "image_root": str(entries[0].image_path.parent),
            "selected_ids": [entry.id for entry in entries],
        },
    )


def test_capture_calls_ocr_and_model_once_and_omits_ground_truth(tmp_path: Path) -> None:
    entry = _make_entry(tmp_path, "s1", "US")
    records, pipeline, gateway, versions = _capture(tmp_path, [entry])

    assert len(records) == 1
    assert pipeline.analyze_calls == 1
    assert pipeline.status_calls == 1
    assert gateway.calls == 1
    record = records[0]
    assert record["raw_image_sha256"] == hashlib.sha256(entry.image_path.read_bytes()).hexdigest()
    assert record["model_revision"] == versions["model_revision"]
    assert record["record_sha256"] == ea._record_sha256(record)
    assert "country_iso" not in record
    assert "ground_truth" not in record


def test_capture_inputs_do_not_change_with_ground_truth(tmp_path: Path) -> None:
    image = tmp_path / "same.jpg"
    image.write_bytes(b"same-image")
    jp = Entry("same", image, "same.jpg", "JP")
    us = Entry("same", image, "same.jpg", "US")

    first, _, _, _ = _capture(tmp_path, [jp])
    second, _, _, _ = _capture(tmp_path, [us])

    assert first == second


def test_replay_strips_crops_and_rederives_only_semantic_clues(tmp_path: Path) -> None:
    entry = _make_entry(tmp_path, "s1", "US")
    records, pipeline, gateway, versions = _capture(tmp_path, [entry])
    plan = {
        "s1": {
            "model_only": ["US"],
            "legacy_ocr": ["AA", "US"],
            "revised_ocr": ["US"],
        }
    }
    context = _replay_context([entry], versions, plan)
    seen_bases: list[list[dict[str, Any]]] = []

    original_rederive = context.rederive_semantic

    def rederive(payload):
        seen_bases.append(copy.deepcopy(payload))
        return original_rederive(payload)

    context.rederive_semantic = rederive
    FakeEngine.calls = []
    report = ea.replay_records(records, _load_meta_for(records, [entry], versions), context)

    assert report["evidence_view"] == "full_image_only"
    assert FakeEngine.calls
    assert all("crops" not in call["evidence"] for call in FakeEngine.calls)
    assert all("reference" not in call["evidence"] for call in FakeEngine.calls)
    assert seen_bases and all(
        all(clue.get("source") != "semantic" for clue in base) for base in seen_bases
    )
    language = next(call for call in FakeEngine.calls if call["arm"] == "revised_ocr")
    assert any(clue.get("source") == "text_language" for clue in language["clues"])
    assert any(clue.get("extra", {}).get("revised") for clue in language["clues"])
    assert pipeline.analyze_calls == 1
    assert gateway.calls == 1


def test_replay_ground_truth_change_does_not_change_inference_inputs(tmp_path: Path) -> None:
    image = tmp_path / "same.jpg"
    image.write_bytes(b"same-image")
    us = Entry("same", image, "same.jpg", "US")
    jp = Entry("same", image, "same.jpg", "JP")
    records, _, _, versions = _capture(tmp_path, [us])
    meta = _load_meta_for(records, [us], versions)
    plan = {"same": {arm: ["US"] for arm in ea.ARM_NAMES}}

    FakeEngine.calls = []
    ea.replay_records(records, meta, _replay_context([us], versions, plan))
    first_calls = copy.deepcopy(FakeEngine.calls)
    FakeEngine.calls = []
    ea.replay_records(
        records, {**meta, "manifest": {**meta["manifest"], "selected_ids": ["same"]}},
        _replay_context([jp], versions, plan),
    )

    assert FakeEngine.calls == first_calls


def _load_meta_for(
    records: list[dict[str, Any]],
    entries: list[Entry],
    versions: dict[str, Any],
) -> dict[str, Any]:
    cache_text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    return {
        "schema_version": ea.SCHEMA_VERSION,
        "kind": "country_ablation_capture",
        "manifest": {
            "path": "manifest.jsonl",
            "sha256": "c" * 64,
            "image_root": str(entries[0].image_path.parent),
            "selected_ids": [entry.id for entry in entries],
        },
        "versions": versions,
        "provenance": _provenance(),
        "sample_count": len(records),
        "cache_sha256": hashlib.sha256(cache_text.encode()).hexdigest(),
    }


def test_replay_uses_same_labelled_denominator_and_reports_transitions(tmp_path: Path) -> None:
    entries = [
        _make_entry(tmp_path, "s1", "US"),
        _make_entry(tmp_path, "s2", "JP"),
        _make_entry(tmp_path, "s3", "FR"),
        _make_entry(tmp_path, "s4", None),
        _make_entry(tmp_path, "s5", "DE"),
    ]
    records, _, _, versions = _capture(tmp_path, entries)
    plan = {
        "s1": {
            "model_only": ["US"],
            "legacy_ocr": ["AA", "BB", "CC", "DD", "US"],
            "revised_ocr": ["AA", "BB", "CC", "DD", "US"],
        },
        "s2": {
            "model_only": ["AA", "BB", "CC", "DD", "JP"],
            "legacy_ocr": ["JP"],
            "revised_ocr": ["AA", "JP"],
        },
        "s3": {
            "model_only": ["AA", "FR"],
            "legacy_ocr": ["AA", "BB", "CC", "FR"],
            "revised_ocr": ["FR"],
        },
        "s5": {
            "model_only": ["DE"],
            "legacy_ocr": "FAIL",
            "revised_ocr": ["AA", "BB", "DE"],
        },
    }
    context = _replay_context(entries, versions, plan)
    FakeEngine.calls = []
    report = ea.replay_records(records, _load_meta_for(records, entries, versions), context)

    assert report["denominator"]["attempted_labelled"] == 4
    assert report["denominator"]["unknown_label_excluded"] == 1
    assert report["denominator"]["included_ids"] == ["s1", "s2", "s3", "s5"]
    assert {arm: report["metrics"][arm]["samples"] for arm in ea.ARM_NAMES} == {
        "model_only": 4,
        "legacy_ocr": 4,
        "revised_ocr": 4,
    }
    assert report["metrics"]["model_only"]["top1"] == 0.5
    assert report["metrics"]["legacy_ocr"]["top1"] == 0.25
    assert report["metrics"]["revised_ocr"]["top1"] == 0.25
    assert report["denominator"]["successful_all_arms"] == 3
    assert report["excluded"] == [{"id": "s4", "reason": "unknown_label"}]
    assert report["failures"] == [
        {"id": "s5", "arm": "legacy_ocr", "reason": "RuntimeError"}
    ]

    legacy_top1 = report["comparisons"]["legacy_ocr_vs_model_only"]["top1"]
    assert legacy_top1["corrected_ids"] == ["s2"]
    assert legacy_top1["harmed_ids"] == ["s1", "s5"]
    assert legacy_top1["unchanged_ids"] == ["s3"]
    assert legacy_top1["rank_transitions"]["5->1"] == ["s2"]
    assert legacy_top1["rank_transitions"]["1->5"] == ["s1"]
    assert legacy_top1["rank_transitions"]["1->miss"] == ["s5"]

    revised_top1 = report["comparisons"]["revised_ocr_vs_model_only"]["top1"]
    assert revised_top1["corrected_ids"] == ["s3"]
    assert revised_top1["harmed_ids"] == ["s1", "s5"]
    assert revised_top1["unchanged_ids"] == ["s2"]
    assert revised_top1["rank_transitions"]["2->1"] == ["s3"]

    repair_top1 = report["comparisons"]["revised_ocr_vs_legacy_ocr"]["top1"]
    assert repair_top1["corrected_ids"] == ["s3"]
    assert repair_top1["harmed_ids"] == ["s2"]
    assert repair_top1["unchanged_ids"] == ["s1", "s5"]
    assert repair_top1["rank_transitions"]["4->1"] == ["s3"]
    assert repair_top1["rank_transitions"]["miss->3"] == ["s5"]
    repair_top3 = report["comparisons"]["revised_ocr_vs_legacy_ocr"]["top3"]
    assert repair_top3["corrected_ids"] == ["s3", "s5"]


def test_replay_rejects_model_catalog_image_and_scorer_mismatch(tmp_path: Path) -> None:
    entry = _make_entry(tmp_path, "s1", "US")
    records, _, _, versions = _capture(tmp_path, [entry])
    meta = _load_meta_for(records, [entry], versions)
    plan = {"s1": {arm: ["US"] for arm in ea.ARM_NAMES}}

    wrong_model = _versions(model_revision="model-r2")
    with pytest.raises(ea.AblationError, match="model_revision"):
        ea.replay_records(records, meta, _replay_context([entry], wrong_model, plan))

    wrong_catalog = _versions(catalog_revision="catalog-r2")
    with pytest.raises(ea.AblationError, match="catalog_revision"):
        ea.replay_records(records, meta, _replay_context([entry], wrong_catalog, plan))

    wrong_prompts = _versions(prompts=["different prompt"])
    with pytest.raises(ea.AblationError, match="prompt templates"):
        ea.replay_records(records, meta, _replay_context([entry], wrong_prompts, plan))

    scorer_tampered = copy.deepcopy(records)
    scorer_tampered[0]["legacy_scorer_sha256"] = SHA_A
    scorer_tampered[0] = ea.seal_record(scorer_tampered[0])
    with pytest.raises(ea.AblationError, match="legacy scorer version"):
        ea.replay_records(
            scorer_tampered, meta, _replay_context([entry], versions, plan)
        )

    entry.image_path.write_bytes(b"changed-image")
    with pytest.raises(ea.AblationError, match="raw image sha256"):
        ea.replay_records(records, meta, _replay_context([entry], versions, plan))

    changed_scorer = _replay_context([entry], versions, plan)
    changed_scorer.provenance["revised_scorer_sha256"] = SHA_A
    with pytest.raises(ea.AblationError, match="scorer changed"):
        ea.replay_records(records, meta, changed_scorer)

    changed_engine = _replay_context([entry], versions, plan)
    changed_engine.provenance["revised_engine_sha256"] = SHA_B
    with pytest.raises(ea.AblationError, match="engine changed"):
        ea.replay_records(records, meta, changed_engine)


def _refresh_cache_hash(root: Path) -> None:
    meta_path = root / ea.META_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["cache_sha256"] = ea._sha256_file(root / ea.CACHE_FILENAME)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_load_cache_rejects_duplicate_missing_and_tampered_records(tmp_path: Path) -> None:
    entries = [_make_entry(tmp_path, "s1"), _make_entry(tmp_path, "s2")]
    records, _, _, versions = _capture(tmp_path, entries)
    cache_root = tmp_path / "cache"
    _write_cache(cache_root, entries, records, versions)
    cache_path = cache_root / ea.CACHE_FILENAME
    original_lines = cache_path.read_text(encoding="utf-8").splitlines()

    cache_path.write_text(original_lines[0] + "\n" + original_lines[0] + "\n", encoding="utf-8")
    _refresh_cache_hash(cache_root)
    with pytest.raises(ea.AblationError, match="duplicate cache sample id"):
        ea.load_cache(cache_root)

    cache_path.write_text(original_lines[0] + "\n", encoding="utf-8")
    _refresh_cache_hash(cache_root)
    with pytest.raises(ea.AblationError, match="record count"):
        ea.load_cache(cache_root)

    tampered = copy.deepcopy(records[0])
    tampered["clues"][0]["value"] = "tampered"
    cache_path.write_text(
        json.dumps(records[0], sort_keys=True) + "\n" + json.dumps(tampered, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_cache_hash(cache_root)
    with pytest.raises(ea.AblationError, match="checksum"):
        ea.load_cache(cache_root)


def test_replay_command_does_not_call_capture_or_write_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _make_entry(tmp_path, "s1", "US")
    records, pipeline, gateway, versions = _capture(tmp_path, [entry])
    cache_root = tmp_path / "cache"
    _write_cache(cache_root, [entry], records, versions)
    cache_before = (cache_root / ea.CACHE_FILENAME).read_bytes()
    plan = {"s1": {arm: ["US"] for arm in ea.ARM_NAMES}}
    context = _replay_context([entry], versions, plan)
    output = tmp_path / "report"
    monkeypatch.setattr(ea, "_load_replay_context", lambda args, meta: context)
    args = SimpleNamespace(
        cache=str(cache_root),
        manifest="manifest.jsonl",
        image_root=None,
        catalog_db="catalog.sqlite3",
        output=str(output),
    )

    FakeEngine.calls = []
    assert ea._replay_command(args) == 0
    assert (cache_root / ea.CACHE_FILENAME).read_bytes() == cache_before
    assert pipeline.analyze_calls == 1
    assert gateway.calls == 1
    assert len(FakeEngine.calls) == 3
    assert (output / ea.REPORT_FILENAME).is_file()


def test_output_directories_refuse_overwrite(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ea.AblationError, match="already exists"):
        ea._ensure_new_dir(existing, label="capture output")


def test_capture_accepts_ids_and_replay_has_no_source_root() -> None:
    args = ea._parse_args(
        [
            "capture",
            "--source-root",
            "D:/source",
            "--manifest",
            "manifest.jsonl",
            "--catalog-db",
            "catalog.sqlite3",
            "--ids",
            "s1,s2",
            "--output",
            "capture",
        ]
    )
    assert args.ids == "s1,s2"
    with pytest.raises(SystemExit):
        ea._parse_args(
            [
                "replay",
                "--cache",
                "cache",
                "--manifest",
                "manifest.jsonl",
                "--catalog-db",
                "catalog.sqlite3",
                "--source-root",
                "D:/source",
                "--output",
                "report",
            ]
        )


def test_capture_failure_keeps_completed_partial_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    entries = [_make_entry(tmp_path, "s1", "US"), _make_entry(tmp_path, "s2", "JP")]
    versions = _versions()

    class FailSecondGateway(FakeGateway):
        def rank_image(self, image_path: str | Path) -> dict[str, Any]:
            if Path(image_path).stem == "s2":
                raise RuntimeError("stop for test")
            return super().rank_image(image_path)

    pipeline = FakePipeline(_legacy_clues())
    gateway = FailSecondGateway(
        lambda image_path: _evidence(versions, image_path.stem)
    )
    context = ea.CaptureContext(
        entries=entries,
        pipeline=pipeline,
        gateway=gateway,
        version_info=versions,
        provenance=_provenance(),
        manifest_info={
            "path": "manifest.jsonl",
            "sha256": "c" * 64,
            "image_root": str(tmp_path),
            "selected_ids": ["s1", "s2"],
        },
        config_info={},
    )
    monkeypatch.setattr(ea, "_load_capture_context", lambda args: context)
    output = tmp_path / "capture"
    result = ea.main(
        [
            "capture",
            "--source-root",
            "D:/source",
            "--manifest",
            "manifest.jsonl",
            "--catalog-db",
            "catalog.sqlite3",
            "--ids",
            "s1,s2",
            "--output",
            str(output),
        ]
    )

    assert result == 2
    partial = output / (ea.CACHE_FILENAME + ".partial")
    assert partial.is_file()
    assert not (output / ea.CACHE_FILENAME).exists()
    assert not (output / ea.META_FILENAME).exists()
    lines = partial.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["id"] == "s1"
    assert record["record_sha256"] == ea._record_sha256(record)
    assert "1 completed samples preserved" in capsys.readouterr().err
    assert pipeline.analyze_calls == 2
    assert gateway.calls == 1


def _recovery_context(
    entries: list[Entry],
    versions: dict[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
    catalog_sha256: str,
) -> ea.RecoveryContext:
    provenance = {
        "origin": "derived_from_sealed_partial_records",
        "legacy_semantic_sha256": SHA_A,
        "legacy_scorer_sha256": SHA_B,
        "legacy_engine_sha256": SHA_A,
        "verified_source_root": "D:/source",
        "verified_eval_helper_sha256": SHA_B,
    }
    return ea.RecoveryContext(
        versions=versions,
        provenance=provenance,
        manifest_info={
            "path": str(manifest_path),
            "sha256": manifest_sha256,
            "image_root": str(entries[0].image_path.parent),
            "selected_ids": [entry.id for entry in entries],
            "selected_ids_origin": "derived_from_validated_record_order",
            "selected_ids_sha256": ea._ids_sha256([entry.id for entry in entries]),
            "manifest_order_prefix": True,
        },
        config={
            "ocr_provider": "rapidocr",
            "ocr_state": "ok",
            "ocr_model_type": "server",
            "ocr_model_version": "PP-OCRv5",
        },
        verification={
            "record_count": len(entries),
            "unique_ids": len(entries),
            "sealed_records": len(entries),
            "raw_image_hashes_verified": len(entries),
            "model_evidence_verified": len(entries),
            "manifest_order_prefix": True,
            "manifest_sha256": manifest_sha256,
            "catalog_file_sha256": catalog_sha256,
            "source_root": "D:/source",
            "source_hashes": {
                "semantic": SHA_A,
                "scorer": SHA_B,
                "engine": SHA_A,
                "eval_helper": SHA_B,
            },
            "current_versions": versions,
            "observed_config": {},
            "limitations": ["derived test cache"],
        },
    )


def test_recover_partial_writes_derived_cache_and_preserves_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = [_make_entry(tmp_path, "s1", "US"), _make_entry(tmp_path, "s2", "JP")]
    records, _, _, versions = _capture(tmp_path, entries)
    partial = tmp_path / "cache.jsonl.partial"
    partial.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    source_before = partial.read_bytes()
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text('{"id":"s1"}\n', encoding="utf-8")
    catalog_path = tmp_path / "catalog.sqlite3"
    catalog_path.write_bytes(b"catalog")
    context = _recovery_context(
        entries,
        versions,
        manifest_path=manifest_path,
        manifest_sha256=ea._sha256_file(manifest_path),
        catalog_sha256=ea._sha256_file(catalog_path),
    )
    monkeypatch.setattr(ea, "_load_recovery_context", lambda args, rows: context)
    monkeypatch.setattr(
        ea,
        "_source_hashes",
        lambda root: context.verification["source_hashes"],
    )
    output = tmp_path / "recovered"
    args = SimpleNamespace(
        partial=str(partial),
        expected_sha256=ea._sha256_file(partial),
        expected_count=2,
        source_root="D:/source",
        manifest=str(manifest_path),
        image_root=None,
        catalog_db=str(catalog_path),
        output=str(output),
    )

    assert ea._recover_command(args) == 0
    assert partial.read_bytes() == source_before
    assert (output / ea.CACHE_FILENAME).read_bytes() == source_before
    recovered, meta = ea.load_cache(output)
    assert len(recovered) == 2
    assert meta["kind"] == ea.RECOVERED_META_KIND
    assert meta["recovery"]["derived_meta"] is True
    assert meta["recovery"]["original_meta_available"] is False
    assert "source_root" not in meta
    audit = json.loads((output / ea.RECOVERY_AUDIT_FILENAME).read_text(encoding="utf-8"))
    assert audit["source"]["line_count"] == 2
    assert audit["source"]["original_meta_available"] is False


def test_recovered_cache_rejects_tampered_audit_and_bad_expectations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _make_entry(tmp_path, "s1", "US")
    records, _, _, versions = _capture(tmp_path, [entry])
    partial = tmp_path / "cache.jsonl.partial"
    partial.write_text(json.dumps(records[0], sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text('{"id":"s1"}\n', encoding="utf-8")
    catalog_path = tmp_path / "catalog.sqlite3"
    catalog_path.write_bytes(b"catalog")
    context = _recovery_context(
        [entry],
        versions,
        manifest_path=manifest_path,
        manifest_sha256=ea._sha256_file(manifest_path),
        catalog_sha256=ea._sha256_file(catalog_path),
    )
    monkeypatch.setattr(ea, "_load_recovery_context", lambda args, rows: context)
    monkeypatch.setattr(ea, "_source_hashes", lambda root: context.verification["source_hashes"])
    args = SimpleNamespace(
        partial=str(partial),
        expected_sha256=ea._sha256_file(partial),
        expected_count=1,
        source_root="D:/source",
        manifest=str(manifest_path),
        image_root=None,
        catalog_db=str(catalog_path),
        output=str(tmp_path / "recovered"),
    )
    assert ea._recover_command(args) == 0
    audit_path = tmp_path / "recovered" / ea.RECOVERY_AUDIT_FILENAME
    audit_path.write_text(audit_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ea.AblationError, match="audit hash"):
        ea.load_cache(tmp_path / "recovered")

    bad = SimpleNamespace(**vars(args))
    bad.output = str(tmp_path / "wrong-hash")
    bad.expected_sha256 = "0" * 64
    with pytest.raises(ea.AblationError, match="sha256 does not match"):
        ea._recover_command(bad)
    assert not (tmp_path / "wrong-hash").exists()

    bad_count = SimpleNamespace(**vars(args))
    bad_count.output = str(tmp_path / "wrong-count")
    bad_count.expected_count = 2
    with pytest.raises(ea.AblationError, match="does not match expected"):
        ea._recover_command(bad_count)
    assert not (tmp_path / "wrong-count").exists()


def test_margin_diagnostics_reports_false_high_and_leader_change() -> None:
    def arm(rank: int, top: str, margin: float) -> dict[str, Any]:
        return {"rank": rank, "top5": [top], "margin": margin}

    rows = [
        {
            "id": "correct",
            "arms": {
                "model_only": arm(1, "US", 0.05),
                "legacy_ocr": arm(1, "US", 0.04),
                "revised_ocr": arm(1, "US", 0.04),
            },
        },
        {
            "id": "false-high",
            "arms": {
                "model_only": arm(2, "CA", 0.05),
                "legacy_ocr": arm(2, "CA", 0.03),
                "revised_ocr": arm(2, "CA", 0.03),
            },
        },
        {
            "id": "leader-change",
            "arms": {
                "model_only": arm(1, "US", 0.05),
                "legacy_ocr": arm(2, "CA", 0.05),
                "revised_ocr": arm(2, "CA", 0.05),
            },
        },
    ]

    diagnostics = ea._margin_diagnostics(rows, high=0.02, medium=0.005)
    legacy = diagnostics["arms"]["legacy_ocr"]
    assert legacy["high_eligible"] == 2
    assert legacy["false_high"] == 1
    assert legacy["false_high_ids"] == ["false-high"]
    assert legacy["high_leader_changed_excluded"] == 1
    assert legacy["medium_or_better"] == 3


def _resume_capture_context(
    entries: list[Entry],
    versions: dict[str, Any],
    pipeline: FakePipeline,
    gateway: FakeGateway,
) -> ea.CaptureContext:
    return ea.CaptureContext(
        entries=entries,
        pipeline=pipeline,
        gateway=gateway,
        version_info=versions,
        provenance=_provenance(),
        manifest_info={
            "path": "manifest.jsonl",
            "sha256": "c" * 64,
            "image_root": str(entries[0].image_path.parent),
            "selected_ids": [entry.id for entry in entries],
        },
        config_info={
            "ocr_provider": "rapidocr",
            "ocr_model_type": "server",
            "geo_ai_mode": "mock",
            "streetclip_enabled": True,
            "allow_unlabeled_country": True,
            "deepseek_api_key_blank": True,
        },
        preflight=None,
    )


def _stable_capture_snapshot() -> dict[str, Any]:
    return {
        "captured_at": "2026-10-20T00:00:00+00:00",
        "source_root": "D:/source",
        "source_files": {"scripts/eval_country_ablation.py": SHA_A},
        "catalog": {"path": "catalog.sqlite3", "sha256": SHA_B},
        "manifest": {"path": "manifest.jsonl", "sha256": "c" * 64},
    }


def test_resume_reuses_prefix_without_calls_and_final_cache_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = [_make_entry(tmp_path, "s1", "US"), _make_entry(tmp_path, "s2", "JP")]
    prefix_records, _, _, versions = _capture(tmp_path, entries[:1])
    source_cache = tmp_path / "source-cache"
    _write_cache(source_cache, entries[:1], prefix_records, versions)
    source_before = (source_cache / ea.CACHE_FILENAME).read_bytes()

    pipeline = FakePipeline(_legacy_clues())
    gateway = FakeGateway(lambda image_path: _evidence(versions, image_path.stem))
    context = _resume_capture_context(entries, versions, pipeline, gateway)
    monkeypatch.setattr(ea, "_load_capture_context", lambda args: context)
    monkeypatch.setattr(ea, "_capture_input_snapshot", lambda args, root: _stable_capture_snapshot())
    output = tmp_path / "resumed"
    args = SimpleNamespace(
        output=str(output),
        progress_json=str(output / ea.PROGRESS_FILENAME),
        resume_from=str(source_cache),
        source_root="D:/source",
        manifest="manifest.jsonl",
        catalog_db="catalog.sqlite3",
        ids=None,
        limit=None,
    )

    assert ea._capture_command(args) == 0
    assert pipeline.analyze_calls == 1
    assert pipeline.status_calls == 1
    assert gateway.calls == 1
    assert (source_cache / ea.CACHE_FILENAME).read_bytes() == source_before
    records, meta = ea.load_cache(output)
    assert [record["id"] for record in records] == ["s1", "s2"]
    assert records[0] == prefix_records[0]
    assert meta["resume"]["reused_original_llm_mode"] == "unknown"
    assert meta["config"]["applies_to_records"] == "new_records_only"
    assert meta["provenance"]["capture_script_applies_to_records"] == "new_records_only"
    progress = json.loads((output / ea.PROGRESS_FILENAME).read_text(encoding="utf-8"))
    assert progress["status"] == "completed"
    assert progress["completed"] == 2
    checkpoint = json.loads((output / ea.RESUME_CHECKPOINT_FILENAME).read_text(encoding="utf-8"))
    assert checkpoint["snapshot_comparison"]["unchanged"] is True

    plan = {
        "s1": {arm: [entries[0].country_iso or "US"] for arm in ea.ARM_NAMES},
        "s2": {arm: [entries[1].country_iso or "JP"] for arm in ea.ARM_NAMES},
    }
    report = ea.replay_records(records, meta, _replay_context(entries, versions, plan))
    assert report["denominator"]["attempted_labelled"] == 2
    assert report["metrics"]["model_only"]["top1"] == 1.0


def test_resume_bad_prefix_fails_before_inference(
    tmp_path: Path,
) -> None:
    entries = [_make_entry(tmp_path, "s1", "US"), _make_entry(tmp_path, "s2", "JP")]
    records, _, _, versions = _capture(tmp_path, entries[:1])
    pipeline = FakePipeline(_legacy_clues())
    gateway = FakeGateway(lambda image_path: _evidence(versions, image_path.stem))
    context = _resume_capture_context(entries, versions, pipeline, gateway)

    duplicate = [records[0], copy.deepcopy(records[0])]
    with pytest.raises(ea.AblationError, match="duplicate ids"):
        ea._validate_resume_records(context, duplicate)

    wrong_model = copy.deepcopy(records)
    wrong_model[0]["model_revision"] = "different"
    with pytest.raises(ea.AblationError, match="model_revision"):
        ea._validate_resume_records(context, wrong_model)

    wrong_ocr = copy.deepcopy(records)
    wrong_ocr[0]["ocr_status"]["model_type"] = "mobile"
    with pytest.raises(ea.AblationError, match="OCR model type"):
        ea._validate_resume_records(context, wrong_ocr)

    wrong_source = copy.deepcopy(records)
    wrong_source[0]["legacy_semantic_sha256"] = SHA_B
    with pytest.raises(ea.AblationError, match="legacy_semantic_sha256"):
        ea._validate_resume_records(context, wrong_source)

    assert pipeline.analyze_calls == 0
    assert pipeline.status_calls == 0
    assert gateway.calls == 0


def test_resume_failure_retains_partial_progress_and_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = [
        _make_entry(tmp_path, "s1", "US"),
        _make_entry(tmp_path, "s2", "JP"),
        _make_entry(tmp_path, "s3", "FR"),
    ]
    prefix_records, _, _, versions = _capture(tmp_path, entries[:1])
    source_cache = tmp_path / "source-cache"
    _write_cache(source_cache, entries[:1], prefix_records, versions)
    source_before = (source_cache / ea.CACHE_FILENAME).read_bytes()

    class FailThirdGateway(FakeGateway):
        def rank_image(self, image_path: str | Path) -> dict[str, Any]:
            if Path(image_path).stem == "s3":
                raise RuntimeError("stop for resume test")
            return super().rank_image(image_path)

    pipeline = FakePipeline(_legacy_clues())
    gateway = FailThirdGateway(lambda image_path: _evidence(versions, image_path.stem))
    context = _resume_capture_context(entries, versions, pipeline, gateway)
    monkeypatch.setattr(ea, "_load_capture_context", lambda args: context)
    monkeypatch.setattr(ea, "_capture_input_snapshot", lambda args, root: _stable_capture_snapshot())
    output = tmp_path / "interrupted"
    args = SimpleNamespace(
        output=str(output),
        progress_json=str(output / ea.PROGRESS_FILENAME),
        resume_from=str(source_cache),
        source_root="D:/source",
        manifest="manifest.jsonl",
        catalog_db="catalog.sqlite3",
        ids=None,
        limit=None,
    )

    with pytest.raises(RuntimeError, match="stop for resume test"):
        ea._capture_command(args)
    partial = output / (ea.CACHE_FILENAME + ".partial")
    assert partial.is_file()
    assert len(partial.read_text(encoding="utf-8").splitlines()) == 2
    assert not (output / ea.CACHE_FILENAME).exists()
    assert not (output / ea.META_FILENAME).exists()
    checkpoint = json.loads((output / ea.RESUME_CHECKPOINT_FILENAME).read_text(encoding="utf-8"))
    progress = json.loads((output / ea.PROGRESS_FILENAME).read_text(encoding="utf-8"))
    assert checkpoint["status"] == "failed"
    assert checkpoint["completed"] == 2
    assert checkpoint["snapshot_before"] is not None
    assert progress["status"] == "failed"
    assert progress["completed"] == 2
    assert (source_cache / ea.CACHE_FILENAME).read_bytes() == source_before


def test_capture_settings_require_mock_and_blank_deepseek() -> None:
    ea._validate_capture_settings(SimpleNamespace(geo_ai_mode="mock", deepseek_api_key=""))
    with pytest.raises(ea.AblationError, match="GEO_AI_MODE=mock"):
        ea._validate_capture_settings(SimpleNamespace(geo_ai_mode="real", deepseek_api_key=""))
    with pytest.raises(ea.AblationError, match="blank DEEPSEEK_API_KEY"):
        ea._validate_capture_settings(SimpleNamespace(geo_ai_mode="mock", deepseek_api_key="set"))


def test_resume_keyboard_interrupt_marks_checkpoint_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = [_make_entry(tmp_path, "s1", "US"), _make_entry(tmp_path, "s2", "JP")]
    prefix_records, _, _, versions = _capture(tmp_path, entries[:1])
    source_cache = tmp_path / "source-cache"
    _write_cache(source_cache, entries[:1], prefix_records, versions)

    class InterruptingGateway(FakeGateway):
        def rank_image(self, image_path: str | Path) -> dict[str, Any]:
            raise KeyboardInterrupt

    pipeline = FakePipeline(_legacy_clues())
    gateway = InterruptingGateway(lambda image_path: _evidence(versions, image_path.stem))
    context = _resume_capture_context(entries, versions, pipeline, gateway)
    monkeypatch.setattr(ea, "_load_capture_context", lambda args: context)
    monkeypatch.setattr(ea, "_capture_input_snapshot", lambda args, root: _stable_capture_snapshot())
    output = tmp_path / "interrupted-signal"
    args = SimpleNamespace(
        output=str(output),
        progress_json=str(output / ea.PROGRESS_FILENAME),
        resume_from=str(source_cache),
        source_root="D:/source",
        manifest="manifest.jsonl",
        catalog_db="catalog.sqlite3",
        ids=None,
        limit=None,
    )

    with pytest.raises(KeyboardInterrupt):
        ea._capture_command(args)
    assert len((output / (ea.CACHE_FILENAME + ".partial")).read_text(encoding="utf-8").splitlines()) == 1
    checkpoint = json.loads((output / ea.RESUME_CHECKPOINT_FILENAME).read_text(encoding="utf-8"))
    progress = json.loads((output / ea.PROGRESS_FILENAME).read_text(encoding="utf-8"))
    assert checkpoint["status"] == "interrupted"
    assert checkpoint["snapshot_before"] is not None
    assert progress["status"] == "interrupted"


def test_source_snapshot_uses_stable_posix_keys(tmp_path: Path) -> None:
    required = (
        "scripts/eval_geolocation.py",
        "scripts/eval_country_ablation.py",
        "backend/app/services/pipeline.py",
        "backend/app/services/model_gateway.py",
        "backend/app/domain/geolocation/model_ranking.py",
        "backend/app/domain/geolocation/engine.py",
        "backend/app/domain/geolocation/analysis/semantic.py",
        "catalog_db.py",
    )
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pass\n", encoding="utf-8")

    snapshot = ea._source_snapshot(tmp_path)
    assert "backend/app/domain/geolocation/analysis/semantic.py" in snapshot
    assert "scripts/eval_country_ablation.py" in snapshot
    assert all("\\" not in key for key in snapshot)


def test_progress_replace_permission_error_retries_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "progress.json"
    real_replace = ea.os.replace
    failures = 0

    def flaky_replace(source: str | Path, target: str | Path) -> None:
        nonlocal failures
        if Path(target) == path and failures == 0:
            failures += 1
            raise PermissionError("temporary lock")
        real_replace(source, target)

    monkeypatch.setattr(ea.os, "replace", flaky_replace)
    ok, error_type = ea._progress_json_write(path, {"status": "running", "completed": 1})

    assert ok is True
    assert error_type is None
    assert failures == 1
    assert json.loads(path.read_text(encoding="utf-8"))["completed"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_capture_continues_when_progress_replace_permanently_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    entries = [_make_entry(tmp_path, "s1", "US"), _make_entry(tmp_path, "s2", "JP")]
    prefix_records, _, _, versions = _capture(tmp_path, entries[:1])
    source_cache = tmp_path / "source-cache"
    _write_cache(source_cache, entries[:1], prefix_records, versions)
    pipeline = FakePipeline(_legacy_clues())
    gateway = FakeGateway(lambda image_path: _evidence(versions, image_path.stem))
    context = _resume_capture_context(entries, versions, pipeline, gateway)
    monkeypatch.setattr(ea, "_load_capture_context", lambda args: context)
    monkeypatch.setattr(ea, "_capture_input_snapshot", lambda args, root: _stable_capture_snapshot())
    real_replace = ea.os.replace

    def progress_locked(source: str | Path, target: str | Path) -> None:
        if Path(target).name == ea.PROGRESS_FILENAME:
            raise PermissionError("reader lock")
        real_replace(source, target)

    monkeypatch.setattr(ea.os, "replace", progress_locked)
    output = tmp_path / "telemetry-failure"
    args = SimpleNamespace(
        output=str(output),
        progress_json=str(output / ea.PROGRESS_FILENAME),
        resume_from=str(source_cache),
        source_root="D:/source",
        manifest="manifest.jsonl",
        catalog_db="catalog.sqlite3",
        ids=None,
        limit=None,
    )

    assert ea._capture_command(args) == 0
    records, _ = ea.load_cache(output)
    assert [record["id"] for record in records] == ["s1", "s2"]
    assert gateway.calls == 1
    assert "progress warning: telemetry write failed" in capsys.readouterr().err
