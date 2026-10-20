from __future__ import annotations

import math

import pytest

from scripts.eval_metrics import (
    accuracy_at_coverage,
    distance_bucket,
    haversine_km,
    percentile,
    rank_of,
    summarize_city,
    summarize_country,
    summarize_latency,
    wilson_interval,
)


def test_haversine_known_distance() -> None:
    assert haversine_km(35.6895, 139.6917, 35.6895, 139.6917) == pytest.approx(0.0)
    tokyo_osaka = haversine_km(35.6895, 139.6917, 34.6937, 135.5023)
    assert tokyo_osaka == pytest.approx(396.44, abs=0.5)


def test_rank_of_is_case_insensitive_and_bounded() -> None:
    assert rank_of(["jp", "KR", "cn"], "kr") == 2
    assert rank_of(["JP"], "CN") is None
    assert rank_of([], "JP") is None


def test_percentile_interpolates() -> None:
    assert percentile([], 50) is None
    assert percentile([7.0], 95) == 7.0
    assert percentile([0.0, 10.0], 50) == pytest.approx(5.0)
    assert percentile([0.0, 10.0, 20.0, 30.0], 90) == pytest.approx(27.0)


def test_distance_bucket_boundaries() -> None:
    assert distance_bucket(0.0) == "<=1 km"
    assert distance_bucket(1.0) == "<=1 km"
    assert distance_bucket(1.01) == "<=25 km"
    assert distance_bucket(2500.0) == "<=2500 km"
    assert distance_bucket(2500.1) == ">2500 km"


def test_summarize_country() -> None:
    records = [
        {"rank": 1, "top1_score": 0.30, "margin": 0.10, "degraded": False},
        {"rank": 2, "top1_score": 0.20, "margin": 0.01, "degraded": False},
        {"rank": None, "top1_score": 0.10, "margin": 0.001, "degraded": True},
        {"rank": 1, "top1_score": 0.40, "margin": 0.0, "degraded": False},
    ]
    summary = summarize_country(records)
    assert summary["samples"] == 4
    assert summary["top1"] == 0.5
    assert summary["top3"] == 0.75
    assert summary["top5"] == 0.75
    assert summary["mrr"] == pytest.approx(0.625)
    assert summary["unranked"] == 1
    assert summary["degraded_rate"] == 0.25

    coverage = {row["margin_min"]: row for row in summary["coverage"]}
    assert coverage[0.01]["samples"] == 2
    assert coverage[0.01]["coverage"] == 0.5
    assert coverage[0.01]["accuracy"] == 0.5


def test_accuracy_at_coverage_reports_abstention() -> None:
    records = [
        {"rank": 1, "margin": 0.10},
        {"rank": 1, "margin": 0.02},
        {"rank": 2, "margin": 0.015},
        {"rank": 2, "margin": 0.0},
    ]
    rows = accuracy_at_coverage(records, thresholds=(0.0, 0.05))
    assert rows[0]["coverage"] == 1.0
    assert rows[0]["accuracy"] == 0.5
    assert rows[1]["samples"] == 1
    assert rows[1]["coverage"] == 0.25
    assert rows[1]["accuracy"] == 1.0


def test_summarize_city_distance_and_exact_metrics() -> None:
    records = [
        {
            "available": True,
            "exact_top1": True,
            "top1_km": 10.0,
            "top3_min_km": 10.0,
        },
        {
            "available": True,
            "exact_top1": False,
            "top1_km": 60.0,
            "top3_min_km": 40.0,
        },
        {
            "available": True,
            "exact_top1": False,
            "top1_km": 3000.0,
            "top3_min_km": 2000.0,
        },
        {"available": False, "error": "unavailable"},
    ]
    summary = summarize_city(records)
    assert summary["samples"] == 4
    assert summary["available"] == 3
    assert summary["unavailable"] == 1
    assert summary["available_rate"] == 0.75
    assert summary["overall"]["samples"] == 4
    assert summary["overall"]["top1_at_25km"] == 0.25
    assert summary["overall"]["top1_at_50km"] == 0.25
    assert summary["overall"]["top3_at_50km"] == 0.5
    assert "top1_exact" not in summary["overall"]
    assert summary["top1_exact"] == pytest.approx(1 / 3)
    assert summary["top1_at_25km"] == pytest.approx(1 / 3)
    assert summary["top1_at_50km"] == pytest.approx(1 / 3)
    assert summary["top3_at_50km"] == pytest.approx(2 / 3)
    assert summary["median_top1_km"] == 60.0
    assert summary["distance_buckets"] == {
        "<=1 km": 0,
        "<=25 km": 1,
        "<=200 km": 1,
        "<=750 km": 0,
        "<=2500 km": 0,
        ">2500 km": 1,
    }


def test_summarize_city_without_exact_ids_skips_exact_rate() -> None:
    summary = summarize_city(
        [{"available": True, "exact_top1": None, "top1_km": 5.0, "top3_min_km": 5.0}]
    )
    assert summary["top1_exact"] is None
    assert summary["top1_exact_samples"] == 0
    assert summary["top1_at_25km"] == 1.0


def test_wilson_interval() -> None:
    assert wilson_interval(0, 0) is None
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=0.001)
    assert high == pytest.approx(0.5962, abs=0.001)
    low, high = wilson_interval(1, 1)
    assert low == pytest.approx(0.2065, abs=0.001)
    assert high == 1.0


def test_summarize_latency() -> None:
    assert summarize_latency([]) == {"samples": 0, "p50": None, "p95": None, "mean": None}
    summary = summarize_latency([10.0, 20.0, 30.0, 40.0])
    assert summary["samples"] == 4
    assert summary["p50"] == 25.0
    assert summary["mean"] == 25.0
    assert math.isfinite(summary["p95"])
