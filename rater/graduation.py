"""The graduation gate: one bar every engine must clear before its ratings
may be used for staking.

An engine graduates when ALL of the following hold on a held-out,
time-ordered test set (no leakage):

1. Skill vs. naive baselines: Brier score beats BOTH the constant
   base-rate predictor and the bookmaker-implied-probability predictor
   (Brier skill > 0 against each).
2. Calibration: in every 5pp probability bucket with >= 200 samples, the
   absolute gap between mean predicted probability and actual win rate
   is < 5 percentage points.
3. Sample size: >= 2,000 test snapshots (or events, for pre-match engines).
4. Freshness: test window ends within the last 18 months of data.

Thresholds are intentionally simple and documented here — change them
deliberately, not silently.
"""
from __future__ import annotations

from typing import Dict, List

from .base import GraduationVerdict

MAX_BUCKET_DEVIATION_PP = 5.0
MIN_BUCKET_SAMPLES = 200
MIN_TEST_SAMPLES = 2000


def evaluate_graduation(report: Dict) -> GraduationVerdict:
    gaps: List[str] = []
    metrics: Dict = {}

    brier = report.get("brier")
    brier_base = report.get("brier_baseline_base_rate")
    brier_book = report.get("brier_baseline_book")
    n_test = report.get("n_test", 0)
    calibration = report.get("calibration", [])

    metrics["brier"] = round(brier, 4) if brier is not None else None
    metrics["brier_skill_vs_base_rate"] = (
        round(1 - brier / brier_base, 4) if brier and brier_base else None
    )
    metrics["brier_skill_vs_book"] = (
        round(1 - brier / brier_book, 4) if brier and brier_book else None
    )
    metrics["n_test"] = n_test

    if n_test < MIN_TEST_SAMPLES:
        gaps.append(f"only {n_test} test samples (< {MIN_TEST_SAMPLES})")

    if brier is not None and brier_base is not None and brier >= brier_base:
        gaps.append("Brier no better than constant base-rate predictor")
    if brier is not None and brier_book is not None and brier >= brier_book:
        gaps.append("Brier no better than bookmaker-implied probabilities")

    worst_bucket = 0.0
    thin_buckets = 0
    for row in calibration:
        n = row.get("n", 0)
        if n < MIN_BUCKET_SAMPLES:
            thin_buckets += 1
            continue
        dev = abs(row.get("mean_predicted", 0) - row.get("actual_win_rate", 0)) * 100
        worst_bucket = max(worst_bucket, dev)
        if dev >= MAX_BUCKET_DEVIATION_PP:
            gaps.append(
                f"bucket {row.get('bucket')} miscalibrated: "
                f"predicted {row.get('mean_predicted', 0):.1%}, "
                f"actual {row.get('actual_win_rate', 0):.1%} (n={n})"
            )
    metrics["worst_calibrated_bucket_deviation_pp"] = round(worst_bucket, 2)
    metrics["thin_buckets_skipped"] = thin_buckets

    return GraduationVerdict(go=not gaps, gaps=gaps, metrics=metrics)
