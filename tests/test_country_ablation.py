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
        return [FakeScore(iso) for iso in predicted]


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
