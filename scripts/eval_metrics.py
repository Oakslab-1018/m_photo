"""Pure metric helpers for offline geolocation evaluation.

All functions take plain dictionaries/lists and return JSON-serializable
values. Nothing here imports the backend, so the definitions can be tested
independently and reused by notebooks.

Terminology
-----------
* country rank: 1-based position of the ground-truth canonical country in the
  fused ranking; ``None`` means it was not returned.
* exact city hit: the ground-truth ``city_id`` is the top-1 row. Only computed
  for samples that carry a city id.
* distance metrics always use the labeled photo coordinates against the
  catalog city's representative point, so even an unmatched city id can still
  be scored at 25/50/100 km.
"""
from __future__ import annotations

import math
from statistics import mean, median
from typing import Iterable, Sequence

EARTH_RADIUS_KM = 6371.0088

DISTANCE_BUCKETS: tuple[tuple[str, float], ...] = (
    ("<=1 km", 1.0),
    ("<=25 km", 25.0),
    ("<=200 km", 200.0),
    ("<=750 km", 750.0),
    ("<=2500 km", 2500.0),
)
FAR_BUCKET = ">2500 km"

DEFAULT_COVERAGE_THRESHOLDS: tuple[float, ...] = (
    0.0,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
)


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in kilometres between two WGS84 points."""

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def rank_of(predicted: Sequence[object], target: object) -> int | None:
    """1-based rank of ``target`` in ``predicted`` or ``None`` when absent."""

    wanted = str(target).strip().upper()
    for index, value in enumerate(predicted, start=1):
        if str(value).strip().upper() == wanted:
            return index
    return None


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if not rank or rank < 1 else 1.0 / float(rank)


def percentile(values: Sequence[float], percent: float) -> float | None:
    """Linear-interpolated percentile; ``None`` for an empty input."""

    cleaned = sorted(float(value) for value in values if value is not None and math.isfinite(value))
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    position = (len(cleaned) - 1) * (max(0.0, min(100.0, percent)) / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return cleaned[lower]
    weight = position - lower
    return cleaned[lower] * (1.0 - weight) + cleaned[upper] * weight


def distance_bucket(km: float) -> str:
    for label, bound in DISTANCE_BUCKETS:
        if km <= bound:
            return label
    return FAR_BUCKET


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> list[float] | None:
    """Two-sided Wilson score interval for a binomial proportion.

    Unlike a normal approximation this stays inside [0, 1] and behaves
    sensibly for the small, stratified samples an offline evaluation starts
    with. ``None`` when there are no trials.
    """

    if total <= 0:
        return None
    successes = max(0, min(int(successes), int(total)))
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [round(max(0.0, center - half), 6), round(min(1.0, center + half), 6)]


def _rate(flags: Iterable[bool]) -> float | None:
    values = [bool(flag) for flag in flags]
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _mean(values: Iterable[float | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not cleaned:
        return None
    return round(mean(cleaned), 6)


def accuracy_at_coverage(
    records: Sequence[dict],
    thresholds: Sequence[float] = DEFAULT_COVERAGE_THRESHOLDS,
) -> list[dict]:
    """Top-1 accuracy restricted to samples whose top1-top2 margin clears a floor.

    Coverage is the fraction of all evaluated samples that would be answered at
    that threshold. This is an honest alternative to treating a cosine score as
    a probability: it reports how much accuracy is bought by abstaining.
    """

    total = len(records)
    rows: list[dict] = []
    for threshold in thresholds:
        selected = [
            record
            for record in records
            if isinstance(record.get("margin"), (int, float))
            and record["margin"] >= threshold
        ]
        hits = sum(1 for record in selected if record.get("rank") == 1)
        rows.append(
            {
                "margin_min": round(float(threshold), 6),
                "coverage": round(len(selected) / total, 6) if total else 0.0,
                "accuracy": round(hits / len(selected), 6) if selected else None,
                "samples": len(selected),
            }
        )
    return rows


def summarize_country(records: Sequence[dict]) -> dict:
    """Aggregate country-level sample records.

    Each record may contain ``rank`` (int or None), ``top1_score``, ``margin``
    and ``degraded`` (no StreetCLIP evidence). Records that failed before a
    country ranking was produced must be excluded by the caller.
    """

    ranks = [record.get("rank") for record in records]
    scores = [record.get("top1_score") for record in records]
    margins = [record.get("margin") for record in records]
    top1_hits = sum(1 for rank in ranks if rank == 1)
    top3_hits = sum(1 for rank in ranks if rank is not None and rank <= 3)
    return {
        "samples": len(records),
        "top1": _rate(rank == 1 for rank in ranks),
        "top1_ci95": wilson_interval(top1_hits, len(records)),
        "top3": _rate(rank is not None and rank <= 3 for rank in ranks),
        "top3_ci95": wilson_interval(top3_hits, len(records)),
        "top5": _rate(rank is not None and rank <= 5 for rank in ranks),
        "mrr": round(
            sum(reciprocal_rank(rank) for rank in ranks) / len(records), 6
        )
        if records
        else None,
        "unranked": sum(1 for rank in ranks if rank is None),
        "mean_top1_score": _mean(scores),
        "mean_margin": _mean(margins),
        "degraded_rate": _rate(bool(record.get("degraded")) for record in records),
        "coverage": accuracy_at_coverage(records),
    }


def summarize_city(records: Sequence[dict]) -> dict:
    """Aggregate city-level records from one source (true or end-to-end).

    The existing flat accuracy/distance fields remain *available-conditional*:
    their denominators are records with an available city ranking and the
    required distance/identity. overall uses all attempted records as its
    denominator, so unavailable or failed city paths count as misses. This
    keeps the historical fields comparable while making end-to-end coverage
    explicit.
    """

    available = [record for record in records if record.get("available")]
    distances = [
        float(record["top1_km"])
        for record in available
        if isinstance(record.get("top1_km"), (int, float))
    ]
    top3_distances = [
        float(record["top3_min_km"])
        for record in available
        if isinstance(record.get("top3_min_km"), (int, float))
    ]
    exact_flags = [
        bool(record["exact_top1"])
        for record in available
        if record.get("exact_top1") is not None
    ]
    bucket_counts = {label: 0 for label, _ in DISTANCE_BUCKETS}
    bucket_counts[FAR_BUCKET] = 0
    for km in distances:
        bucket_counts[distance_bucket(km)] += 1

    def within(bound: float) -> float | None:
        return _rate(km <= bound for km in distances)

    def overall_within(record: dict, key: str, bound: float) -> bool:
        value = record.get(key)
        return (
            bool(record.get("available"))
            and isinstance(value, (int, float))
            and float(value) <= bound
        )

    top1_50_hits = sum(1 for km in distances if km <= 50.0)
    top3_50_hits = sum(1 for km in top3_distances if km <= 50.0)
    overall_top1_50_hits = sum(
        1 for record in records if overall_within(record, "top1_km", 50.0)
    )
    return {
        "samples": len(records),
        "available": len(available),
        "unavailable": len(records) - len(available),
        "available_rate": _rate(bool(record.get("available")) for record in records),
        "top1_exact": _rate(exact_flags),
        "top1_exact_samples": len(exact_flags),
        "top1_at_1km": within(1.0),
        "top1_at_25km": within(25.0),
        "top1_at_50km": within(50.0),
        "top1_at_50km_ci95": wilson_interval(top1_50_hits, len(distances)),
        "top1_at_100km": within(100.0),
        "top1_at_200km": within(200.0),
        "top1_at_750km": within(750.0),
        "top1_at_2500km": within(2500.0),
        "top3_at_25km": _rate(km <= 25.0 for km in top3_distances),
        "top3_at_50km": _rate(km <= 50.0 for km in top3_distances),
        "top3_at_50km_ci95": wilson_interval(top3_50_hits, len(top3_distances)),
        "top3_at_100km": _rate(km <= 100.0 for km in top3_distances),
        "mean_top1_km": round(mean(distances), 3) if distances else None,
        "median_top1_km": round(median(distances), 3) if distances else None,
        "p90_top1_km": round(percentile(distances, 90.0), 3) if distances else None,
        "distance_buckets": bucket_counts,
        "overall": {
            "samples": len(records),
            "top1_at_25km": _rate(
                overall_within(record, "top1_km", 25.0) for record in records
            ),
            "top1_at_50km": _rate(
                overall_within(record, "top1_km", 50.0) for record in records
            ),
            "top1_at_50km_ci95": wilson_interval(overall_top1_50_hits, len(records)),
            "top1_at_100km": _rate(
                overall_within(record, "top1_km", 100.0) for record in records
            ),
            "top3_at_50km": _rate(
                overall_within(record, "top3_min_km", 50.0) for record in records
            ),
        },
    }


def summarize_latency(values: Sequence[float]) -> dict:
    """p50/p95/mean for millisecond timings."""

    cleaned = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not cleaned:
        return {"samples": 0, "p50": None, "p95": None, "mean": None}
    return {
        "samples": len(cleaned),
        "p50": round(percentile(cleaned, 50.0) or 0.0, 1),
        "p95": round(percentile(cleaned, 95.0) or 0.0, 1),
        "mean": round(mean(cleaned), 1),
    }


__all__ = [
    "DEFAULT_COVERAGE_THRESHOLDS",
    "DISTANCE_BUCKETS",
    "EARTH_RADIUS_KM",
    "FAR_BUCKET",
    "accuracy_at_coverage",
    "distance_bucket",
    "haversine_km",
    "percentile",
    "rank_of",
    "reciprocal_rank",
    "summarize_city",
    "summarize_country",
    "summarize_latency",
    "wilson_interval",
]
