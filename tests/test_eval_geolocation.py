from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from backend.app.domain.geolocation import knowledge_graph as kg
from scripts import eval_geolocation as eg
from scripts.eval_manifest import ManifestEntry


@pytest.fixture
def restore_knowledge_graph():
    original = kg._CONFIGURED_CATALOG_PATH  # type: ignore[attr-defined]
    kg.load_countries.cache_clear()
    try:
        yield
    finally:
        kg._CONFIGURED_CATALOG_PATH = original  # type: ignore[attr-defined]
        kg.load_countries.cache_clear()


@dataclass
class FakeScore:
    iso: str
    score: float
    source_regions: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"iso": self.iso, "score": self.score}


class FakeEngine:
    evidence: dict[str, Any] | None = None

    def __init__(
        self,
        scores: list[FakeScore] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.scores = scores or []
        self.error = error

    def infer(self, clues, top_n: int):
        if self.error is not None:
            raise self.error
        return self.scores


class FakePipeline:
    def analyze(self, image_path: str) -> list[dict[str, Any]]:
        return []

    def ocr_status(self) -> dict[str, Any]:
        return {
            "provider": "mock",
            "state": "available",
            "available": True,
            "line_count": 0,
        }


class FakeGateway:
    def __init__(
        self,
        *,
        evidence: dict[str, Any] | None = None,
        city_rows: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.evidence = evidence or {"margin": 0.1, "uncertain": False}
        self.city_rows = city_rows or {}
        self.city_calls: list[str] = []

    def rank_image(self, path: str | Path) -> dict[str, Any]:
        return self.evidence

    def rank_cities(self, path: str | Path, iso: str) -> dict[str, Any]:
        self.city_calls.append(iso)
        return {
            "status": "ok",
            "city_scores": self.city_rows.get(iso, []),
        }


class FakeReasoner:
    def rerank(self, rows, clues, *, iso: str):
        return SimpleNamespace(
            city_scores=list(rows),
            clue_dimensions=(),
            coverage_note=None,
        )


def _entry(tmp_path: Path, **overrides: Any) -> ManifestEntry:
    image = tmp_path / f"{overrides.get('id', 'sample-1')}.jpg"
    image.write_bytes(b"not-a-real-jpeg")
    values = {
        "id": "sample-1",
        "image_path": image,
        "image_ref": image.name,
        "country_iso": "JP",
        "lat": 35.6895,
        "lng": 139.6917,
        "city_id": "1",
        "city": "Tokyo",
        "city_source_iso": "JP",
    }
    values.update(overrides)
    return ManifestEntry(**values)


def _city_payload(iso: str) -> list[dict[str, Any]]:
    if iso == "JP":
        return [
            {
                "city_id": "1",
                "name": "Tokyo",
                "region": "13",
                "score": 0.9,
                "raw_cosine": 0.8,
                "lat": 35.6895,
                "lng": 139.6917,
            }
        ]
    return [
        {
            "city_id": "2",
            "name": "New York",
            "region": "NY",
            "score": 0.8,
            "raw_cosine": 0.7,
            "lat": 40.7128,
            "lng": -74.0060,
        }
    ]


def test_oracle_city_uses_gt_while_e2e_uses_prediction(tmp_path: Path) -> None:
    entry = _entry(tmp_path)
    gateway = FakeGateway(
        city_rows={"JP": _city_payload("JP"), "US": _city_payload("US")}
    )
    engine = FakeEngine([FakeScore("US", 0.9), FakeScore("JP", 0.1)])

    record = eg._evaluate_sample(
        entry,
        pipeline=FakePipeline(),
        gateway=gateway,
        engine=engine,
        reasoner=FakeReasoner(),
        evaluate_cities=True,
        evaluate_e2e_cities=True,
    )

    assert gateway.city_calls == ["JP", "US"]
    assert record["country"]["rank"] == 2
    assert record["city_true_country"]["source_iso"] == "JP"
    assert record["city_true_country"]["exact_top1"] is True
    assert record["city_e2e"]["source_iso"] == "US"
    assert record["city_e2e"]["top1_city"]["city_id"] == "2"
    assert record["city_e2e"]["exact_top1"] is False
    assert "reused_true_country_ranking" not in record["city_e2e"]


def test_country_prediction_failure_stops_before_city_calls(tmp_path: Path) -> None:
    gateway = FakeGateway()
    record = eg._evaluate_sample(
        _entry(tmp_path),
        pipeline=FakePipeline(),
        gateway=gateway,
        engine=FakeEngine(error=RuntimeError("boom")),
        reasoner=FakeReasoner(),
        evaluate_cities=True,
        evaluate_e2e_cities=True,
    )

    assert record["error"] == "RuntimeError: boom"
    assert "country" not in record
    assert gateway.city_calls == []
    assert "city_true_country" not in record
    assert "city_e2e" not in record


def test_aggregate_reports_available_and_all_attempted_metrics() -> None:
    city_ok = {
        "available": True,
        "exact_top1": True,
        "top1_km": 10.0,
        "top3_min_km": 10.0,
    }
    records = [
        {
            "id": "ok",
            "country": {
                "rank": 1,
                "top1_score": 0.8,
                "margin": 0.1,
                "degraded": False,
            },
            "city_true_country": dict(city_ok),
            "city_e2e": dict(city_ok),
        },
        {"id": "failed", "error": "RuntimeError: boom"},
    ]
    aggregated = eg._aggregate(
        records, evaluate_cities=True, evaluate_e2e_cities=True
    )
    country = aggregated["summary"]["country"]
    city_e2e = aggregated["summary"]["city"]["e2e"]

    assert country["samples"] == 1
    assert country["top1"] == 1.0
    assert country["overall"]["samples"] == 2
    assert country["overall"]["top1"] == 0.5
    assert country["overall"]["coverage"][0]["coverage"] == 0.5
    assert city_e2e["samples"] == 2
    assert city_e2e["available_rate"] == 0.5
    assert city_e2e["top1_at_50km"] == 1.0
    assert city_e2e["overall"]["samples"] == 2
    assert city_e2e["overall"]["top1_at_50km"] == 0.5
    assert city_e2e["overall"]["top3_at_50km"] == 0.5
    assert any(row["path"] == "e2e" for row in aggregated["city_errors"])


@pytest.mark.parametrize("path", ["e2e", "setup"])
def test_strict_fails_any_enabled_city_path_error(path: str) -> None:
    report = {
        "summary": {
            "errors": 0,
            "country": {"overall": {"top1": 1.0, "degraded_rate": 0.0}},
            "city": {
                "true_country": {"unavailable": 0},
                "e2e": {"unavailable": 1},
            },
        },
        "city_errors": [{"id": "sample-1", "path": path, "error": "unavailable"}],
    }
    args = SimpleNamespace(strict=True, min_country_top1=None)

    failures = eg._gate_failures(
        report,
        args,
        evaluate_cities=True,
        evaluate_e2e_cities=True,
    )

    assert failures == ["1 enabled city-path failure(s)"]


def test_min_country_gate_uses_all_attempted_accuracy() -> None:
    report = {
        "summary": {
            "errors": 1,
            "country": {
                "overall": {"top1": 0.5, "degraded_rate": 0.0},
            },
            "city": {},
        },
        "city_errors": [],
    }
    args = SimpleNamespace(strict=False, min_country_top1=0.75)

    failures = eg._gate_failures(
        report,
        args,
        evaluate_cities=False,
        evaluate_e2e_cities=False,
    )

    assert failures == ["country top-1 (all attempted) 0.5 < 0.75"]


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1", "1.1"])
def test_min_country_top1_rejects_invalid_values(value: str) -> None:
    with pytest.raises(SystemExit) as exc:
        eg._parse_args(["--manifest", "manifest.jsonl", "--min-country-top1", value])
    assert exc.value.code == 2


def test_invalid_settings_return_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    restore_knowledge_graph,
) -> None:
    catalog = tmp_path / "catalog.sqlite3"
    catalog.write_bytes(b"catalog")

    def invalid_settings(*args, **kwargs):
        raise ValueError("bad configuration")

    monkeypatch.setattr(eg, "_build_settings", invalid_settings)
    result = eg.main(
        ["--manifest", str(tmp_path / "manifest.jsonl"), "--catalog-db", str(catalog)]
    )

    assert result == 2
    assert "bad configuration" in capsys.readouterr().err


def _install_fake_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entries: list[ManifestEntry],
    settings: FakeSettings,
    probe,
) -> None:
    catalog = tmp_path / "catalog.sqlite3"
    catalog.write_bytes(b"catalog")
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({"id": entries[0].id}) + "\n", encoding="utf-8"
    )

    class FakeCatalog:
        def __init__(self, path: Path) -> None:
            self.path = path

        def countries(self) -> list[dict[str, str]]:
            return [{"iso": "JP"}, {"iso": "US"}]

        def revision(self, source: str) -> str:
            return f"revision-{source}"

    monkeypatch.setattr(eg, "_build_settings", lambda args, path: settings)
    monkeypatch.setattr(eg.kg, "configure_catalog", lambda path: None)
    monkeypatch.setattr(eg, "CatalogDatabase", FakeCatalog)
    monkeypatch.setattr(eg, "load_manifest", lambda *args, **kwargs: entries)
    monkeypatch.setattr(eg, "_probe_streetclip", probe)
    monkeypatch.setattr(eg, "semantic_catalog_revision", lambda countries: "semantic")
    monkeypatch.setattr(eg, "_git_info", lambda: {"commit": None, "dirty": None})
    monkeypatch.setattr(eg, "PipelineService", lambda value: SimpleNamespace(settings=value))
    monkeypatch.setattr(eg, "StreetClipGateway", lambda value: SimpleNamespace(settings=value))
    monkeypatch.setattr(eg, "ModelAwareEngine", lambda: SimpleNamespace())
    monkeypatch.setattr(
        eg,
        "load_city_index",
        lambda path: pytest.fail("city catalog should not load for an unavailable city path"),
    )


@dataclass(frozen=True)
class FakeSettings:
    streetclip_enabled: bool = True
    streetclip_url: str = "http://127.0.0.1:8001"
    streetclip_api_key: str = ""
    streetclip_timeout_seconds: float = 45.0
    ocr_provider: str = "mock"
    ocr_model_type: str = "server"
    geo_ai_mode: str = "mock"


def _fake_settings() -> FakeSettings:
    return FakeSettings()


def test_allow_degraded_country_preflight_failure_disables_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_knowledge_graph,
) -> None:
    entry = _entry(tmp_path)
    _install_fake_main(
        tmp_path,
        monkeypatch,
        [entry],
        _fake_settings(),
        lambda settings, require_city: (_ for _ in ()).throw(eg.EvalError("down")),
    )
    captured: dict[str, Any] = {}

    def fake_evaluate(entry, **kwargs):
        captured.update(kwargs)
        return {
            "id": entry.id,
            "country": {"rank": None, "degraded": True},
            "timing_ms": {},
        }

    monkeypatch.setattr(eg, "_evaluate_sample", fake_evaluate)
    result = eg.main(
        [
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--catalog-db",
            str(tmp_path / "catalog.sqlite3"),
            "--output",
            str(tmp_path / "report"),
            "--allow-degraded",
            "--quiet",
        ]
    )

    report = json.loads((tmp_path / "report" / "report.json").read_text(encoding="utf-8"))
    assert result == 0
    assert captured["gateway"].settings.streetclip_enabled is False
    assert captured["reasoner"] is None
    assert captured["city_path_error"] == "country model unavailable; rules-only degraded run"
    assert report["configuration"]["streetclip_enabled"] is False
    assert report["configuration"]["streetclip_requested"] is True
    assert report["configuration"]["allow_unlabeled_country"] is False
    assert report["configuration"]["model_health"]["country_capability"] is False
    markdown = (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    assert "Rules-only samples" in markdown
    assert entry.id in markdown


def test_city_capability_failure_keeps_country_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_knowledge_graph,
) -> None:
    entry = _entry(tmp_path)
    _install_fake_main(
        tmp_path,
        monkeypatch,
        [entry],
        _fake_settings(),
        lambda settings, require_city: {"ready": True, "city_capability": False},
    )
    captured: dict[str, Any] = {}

    def fake_evaluate(entry, **kwargs):
        captured.update(kwargs)
        return {
            "id": entry.id,
            "country": {"rank": 1, "degraded": False},
            "timing_ms": {},
        }

    monkeypatch.setattr(eg, "_evaluate_sample", fake_evaluate)
    result = eg.main(
        [
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--catalog-db",
            str(tmp_path / "catalog.sqlite3"),
            "--output",
            str(tmp_path / "report"),
            "--allow-degraded",
            "--quiet",
        ]
    )

    report = json.loads((tmp_path / "report" / "report.json").read_text(encoding="utf-8"))
    assert result == 0
    assert captured["gateway"].settings.streetclip_enabled is True
    assert captured["reasoner"] is None
    assert captured["city_path_error"] == "StreetCLIP city capability unavailable"
    assert report["configuration"]["streetclip_enabled"] is True
    assert report["configuration"]["model_health"]["city_capability"] is False


def test_fail_fast_writes_partial_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_knowledge_graph,
) -> None:
    entry = _entry(tmp_path)
    _install_fake_main(
        tmp_path,
        monkeypatch,
        [entry],
        _fake_settings(),
        lambda settings, require_city: {"ready": True, "city_capability": True},
    )
    monkeypatch.setattr(
        eg,
        "_evaluate_sample",
        lambda entry, **kwargs: {
            "id": entry.id,
            "error": "RuntimeError: boom",
            "timing_ms": {},
        },
    )
    result = eg.main(
        [
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--catalog-db",
            str(tmp_path / "catalog.sqlite3"),
            "--output",
            str(tmp_path / "report"),
            "--no-cities",
            "--no-e2e-cities",
            "--fail-fast",
            "--quiet",
        ]
    )

    report = json.loads((tmp_path / "report" / "report.json").read_text(encoding="utf-8"))
    samples = (tmp_path / "report" / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    assert result == 1
    assert report["run"]["partial"] is True
    assert report["run"]["fail_fast_sample"] == entry.id
    assert report["gates"]["passed"] is False
    assert len(samples) == 1
    assert (tmp_path / "report" / "report.md").is_file()

def test_unlabeled_country_still_runs_e2e_prediction_and_distance(tmp_path: Path) -> None:
    entry = _entry(tmp_path, country_iso=None)
    gateway = FakeGateway(city_rows={"US": _city_payload("US")})
    engine = FakeEngine([FakeScore("US", 0.9), FakeScore("JP", 0.1)])

    record = eg._evaluate_sample(
        entry,
        pipeline=FakePipeline(),
        gateway=gateway,
        engine=engine,
        reasoner=FakeReasoner(),
        evaluate_cities=True,
        evaluate_e2e_cities=True,
    )

    assert record["ground_truth"]["canonical_country_iso"] is None
    assert record["country"]["labeled"] is False
    assert record["country"]["rank"] is None
    assert gateway.city_calls == ["US"]
    assert record["city_true_country"] == {
        "available": False,
        "skipped": True,
        "reason": "no_country_label",
    }
    assert record["city_e2e"]["available"] is True
    assert record["city_e2e"]["source_iso"] == "US"
    assert record["city_e2e"]["top1_km"] > 0
    assert record["city_e2e"]["top1_city"]["city_id"] == "2"
    assert "reused_true_country_ranking" not in record["city_e2e"]


def test_country_denominator_uses_only_labeled_attempts() -> None:
    records = [
        {
            "id": "labeled-ok",
            "ground_truth": {"country_iso": "JP"},
            "country": {"labeled": True, "rank": 1, "degraded": False},
        },
        {
            "id": "unlabeled-ok",
            "ground_truth": {"country_iso": None},
            "country": {"labeled": False, "rank": None, "degraded": False},
        },
        {
            "id": "labeled-error",
            "ground_truth": {"country_iso": "JP"},
            "error": "RuntimeError: boom",
        },
        {
            "id": "unlabeled-error",
            "ground_truth": {"country_iso": None},
            "error": "RuntimeError: boom",
        },
    ]

    aggregated = eg._aggregate(
        records,
        evaluate_cities=False,
        evaluate_e2e_cities=False,
    )
    country = aggregated["summary"]["country"]

    assert country["labeled_samples"] == 2
    assert country["unlabeled_samples"] == 2
    assert country["label_coverage"] == 0.5
    assert country["overall"]["samples"] == 2
    assert country["overall"]["top1"] == 0.5
    assert aggregated["summary"]["errors"] == 2


def test_unlabeled_oracle_skip_does_not_fail_strict_or_bias_city_denominator(
    tmp_path: Path,
) -> None:
    entry = _entry(tmp_path, country_iso=None)
    record = eg._evaluate_sample(
        entry,
        pipeline=FakePipeline(),
        gateway=FakeGateway(city_rows={"US": _city_payload("US")}),
        engine=FakeEngine([FakeScore("US", 0.9)]),
        reasoner=FakeReasoner(),
        evaluate_cities=True,
        evaluate_e2e_cities=True,
    )
    aggregated = eg._aggregate(
        [record],
        evaluate_cities=True,
        evaluate_e2e_cities=True,
    )
    city = aggregated["summary"]["city"]

    assert city["true_country"]["samples"] == 0
    assert city["true_country"]["unavailable"] == 0
    assert city["e2e"]["samples"] == 1
    assert aggregated["unavailable_city_samples"] == []
    assert aggregated["city_errors"] == []

    report = {
        "summary": aggregated["summary"],
        "city_errors": aggregated["city_errors"],
    }
    args = SimpleNamespace(strict=True, min_country_top1=None)
    failures = eg._gate_failures(
        report,
        args,
        evaluate_cities=True,
        evaluate_e2e_cities=True,
    )
    assert failures == []


def test_unlabeled_degraded_is_global_and_fails_strict() -> None:
    record = {
        "id": "unlabeled-degraded",
        "ground_truth": {"country_iso": None},
        "country": {"labeled": False, "rank": None, "degraded": True},
    }
    aggregated = eg._aggregate(
        [record],
        evaluate_cities=False,
        evaluate_e2e_cities=False,
    )

    assert aggregated["summary"]["country"]["overall"]["samples"] == 0
    assert aggregated["summary"]["degraded"] == 1

    report = {
        "summary": aggregated["summary"],
        "city_errors": aggregated["city_errors"],
    }
    args = SimpleNamespace(strict=True, min_country_top1=None)
    failures = eg._gate_failures(
        report,
        args,
        evaluate_cities=False,
        evaluate_e2e_cities=False,
    )
    assert failures == ["some samples ran without StreetCLIP evidence"]


def test_min_country_gate_cannot_pass_without_country_labels() -> None:
    record = {
        "id": "unlabeled",
        "ground_truth": {"country_iso": None},
        "country": {"labeled": False, "rank": None, "degraded": False},
    }
    aggregated = eg._aggregate(
        [record],
        evaluate_cities=False,
        evaluate_e2e_cities=False,
    )
    report = {
        "summary": aggregated["summary"],
        "city_errors": aggregated["city_errors"],
    }
    args = SimpleNamespace(strict=False, min_country_top1=0.5)

    failures = eg._gate_failures(
        report,
        args,
        evaluate_cities=False,
        evaluate_e2e_cities=False,
    )

    assert failures == ["country top-1 (all attempted) n/a < 0.5"]


def test_allow_unlabeled_country_cli_is_opt_in() -> None:
    strict_args = eg._parse_args(["--manifest", "manifest.jsonl"])
    opt_in_args = eg._parse_args(
        ["--manifest", "manifest.jsonl", "--allow-unlabeled-country"]
    )

    assert strict_args.allow_unlabeled_country is False
    assert opt_in_args.allow_unlabeled_country is True

def test_unlabeled_country_error_keeps_oracle_skip_but_counts_e2e(tmp_path: Path) -> None:
    entry = _entry(tmp_path, country_iso=None)
    record = eg._evaluate_sample(
        entry,
        pipeline=FakePipeline(),
        gateway=FakeGateway(),
        engine=FakeEngine(error=RuntimeError("boom")),
        reasoner=FakeReasoner(),
        evaluate_cities=True,
        evaluate_e2e_cities=True,
    )
    aggregated = eg._aggregate(
        [record],
        evaluate_cities=True,
        evaluate_e2e_cities=True,
    )

    assert record["city_true_country"]["reason"] == "no_country_label"
    assert record["error"] == "RuntimeError: boom"
    assert aggregated["summary"]["city"]["true_country"]["samples"] == 0
    assert aggregated["summary"]["city"]["e2e"]["unavailable"] == 1
    assert aggregated["summary"]["errors"] == 1
    assert [row["path"] for row in aggregated["city_errors"]] == ["e2e"]
