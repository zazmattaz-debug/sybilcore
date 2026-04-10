"""Population-scale statistical validation for SybilCore.

Runs 100-seed validation (n_agents=1000, n_rounds=20, n_rogues=5) with
EnforcementStrategy.NONE, sweeps the FLAGGED threshold for ROC/PR curves,
and saves aggregate statistics + per-seed results to experiments/.

Usage:
    cd rooms/engineering/sybilcore
    python -m sybilcore.simulation.run_population_validation
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Suppress per-coefficient INFO/WARNING logging — would produce millions of lines.
logging.disable(logging.INFO)
logging.disable(logging.WARNING)

from sybilcore.brains import get_default_brains
from sybilcore.core.coefficient import CoefficientCalculator
from sybilcore.models.agent import AgentTier
from sybilcore.simulation.enforcement import EnforcementStrategy
from sybilcore.simulation.experiment import Experiment, ExperimentConfig
from sybilcore.simulation.synthetic import SyntheticSwarm

# ── Constants ────────────────────────────────────────────────────────────────

N_SEEDS = 100
N_AGENTS = 1000
N_ROUNDS = 20
N_ROGUES = 5
ROGUE_INJECTION_ROUND = 3

# Threshold sweep for ROC curve: coefficient boundary that replaces the
# hard FLAGGED=200 boundary. Agents above this threshold are classified
# as "detected rogues" for the purpose of TPR/FPR computation.
THRESHOLD_START = 50
THRESHOLD_END = 400
THRESHOLD_STEP = 10

OUTPUT_DIR = Path(__file__).parent.parent.parent / "experiments"


# ── Data containers ──────────────────────────────────────────────────────────

@dataclass
class SeedResult:
    """Per-seed summary extracted from an ExperimentResult."""

    seed: int
    false_positive_count: int
    true_positive_count: int   # rogues correctly flagged (FLAGGED or LETHAL tier)
    false_negative_count: int  # rogues never detected
    fpr: float                 # false positive rate: FP / (FP + TN)
    tpr: float                 # true positive rate: TP / (TP + FN)
    detection_latency_rounds: int | None
    compound_infections: int
    elapsed_seconds: float
    # Final-round per-agent coefficient snapshot for threshold sweep
    final_coefficients: list[tuple[bool, float]]  # (is_rogue, coefficient)


@dataclass
class AggregateStats:
    """Cross-seed aggregate statistics."""

    n_seeds: int
    mean_fpr: float
    median_fpr: float
    std_fpr: float
    p95_fpr: float
    p99_fpr: float
    ci95_fpr: tuple[float, float]

    mean_tpr: float
    median_tpr: float
    std_tpr: float
    p95_tpr: float
    p99_tpr: float
    ci95_tpr: tuple[float, float]

    mean_detection_latency: float | None
    ci95_detection_latency: tuple[float, float] | None

    mean_compound_infections: float
    mean_false_positives: float
    mean_true_positives: float
    mean_false_negatives: float


@dataclass
class RocPoint:
    """Single point on the ROC curve (one threshold value)."""

    threshold: float
    mean_tpr: float
    mean_fpr: float
    std_tpr: float
    std_fpr: float


# ── Core runner ──────────────────────────────────────────────────────────────

def _run_single_seed(seed: int) -> SeedResult:
    """Run one experiment and extract classification metrics."""
    config = ExperimentConfig(
        name=f"pop_val_seed_{seed}",
        n_agents=N_AGENTS,
        n_rounds=N_ROUNDS,
        n_rogues=N_ROGUES,
        rogue_injection_round=ROGUE_INJECTION_ROUND,
        enforcement=EnforcementStrategy.NONE,
        seed=seed,
        output_dir=str(OUTPUT_DIR),
    )
    t0 = time.monotonic()
    exp = Experiment(config)
    result = exp.run()
    elapsed = time.monotonic() - t0

    # Extract final-round coefficient per agent plus ground-truth labels.
    # Use last round's agent data as the ground-truth snapshot.
    final_round = result.rounds[-1]

    flagged_tiers = {AgentTier.FLAGGED.value, AgentTier.LETHAL_ELIMINATOR.value}

    # Derive actual rogue/clean counts from the final round (compound spread
    # means more than N_ROGUES agents may be rogue by round 20 under NONE).
    n_rogues_final = sum(1 for a in final_round.agents if a.is_rogue)
    n_clean_final = sum(1 for a in final_round.agents if not a.is_rogue)

    # True positives: rogues in FLAGGED or LETHAL tier in final round
    tp = sum(1 for a in final_round.agents if a.is_rogue and a.tier in flagged_tiers)
    fn = n_rogues_final - tp

    # False positives: clean agents in FLAGGED or LETHAL tier in final round
    fp = sum(1 for a in final_round.agents if not a.is_rogue and a.tier in flagged_tiers)
    tn = n_clean_final - fp

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Final-round coefficients for threshold sweep
    final_coefficients: list[tuple[bool, float]] = [
        (a.is_rogue, a.coefficient) for a in final_round.agents
    ]

    return SeedResult(
        seed=seed,
        false_positive_count=fp,
        true_positive_count=tp,
        false_negative_count=fn,
        fpr=fpr,
        tpr=tpr,
        detection_latency_rounds=result.detection_latency_rounds,
        compound_infections=result.compound_infections,
        elapsed_seconds=elapsed,
        final_coefficients=final_coefficients,
    )


# ── Statistics helpers ───────────────────────────────────────────────────────

def _percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of a sorted or unsorted list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (p / 100) * (len(sorted_data) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_data) - 1)
    frac = idx - lower
    return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac


def _ci95(data: list[float]) -> tuple[float, float]:
    """95% confidence interval using t-distribution approximation (z=1.96)."""
    n = len(data)
    if n < 2:
        m = data[0] if data else 0.0
        return (m, m)
    m = statistics.mean(data)
    s = statistics.stdev(data)
    margin = 1.96 * s / math.sqrt(n)
    return (m - margin, m + margin)


def _compute_aggregate(seed_results: list[SeedResult]) -> AggregateStats:
    fprs = [r.fpr for r in seed_results]
    tprs = [r.tpr for r in seed_results]
    latencies = [r.detection_latency_rounds for r in seed_results if r.detection_latency_rounds is not None]

    lat_mean: float | None = statistics.mean(latencies) if latencies else None
    lat_ci: tuple[float, float] | None = _ci95(latencies) if len(latencies) >= 2 else None

    return AggregateStats(
        n_seeds=len(seed_results),
        mean_fpr=statistics.mean(fprs),
        median_fpr=statistics.median(fprs),
        std_fpr=statistics.stdev(fprs) if len(fprs) > 1 else 0.0,
        p95_fpr=_percentile(fprs, 95),
        p99_fpr=_percentile(fprs, 99),
        ci95_fpr=_ci95(fprs),

        mean_tpr=statistics.mean(tprs),
        median_tpr=statistics.median(tprs),
        std_tpr=statistics.stdev(tprs) if len(tprs) > 1 else 0.0,
        p95_tpr=_percentile(tprs, 95),
        p99_tpr=_percentile(tprs, 99),
        ci95_tpr=_ci95(tprs),

        mean_detection_latency=lat_mean,
        ci95_detection_latency=lat_ci,

        mean_compound_infections=statistics.mean(r.compound_infections for r in seed_results),
        mean_false_positives=statistics.mean(r.false_positive_count for r in seed_results),
        mean_true_positives=statistics.mean(r.true_positive_count for r in seed_results),
        mean_false_negatives=statistics.mean(r.false_negative_count for r in seed_results),
    )


# ── ROC curve computation ────────────────────────────────────────────────────

def _compute_roc_curve(seed_results: list[SeedResult]) -> list[RocPoint]:
    """Sweep threshold from THRESHOLD_START to THRESHOLD_END and compute TPR/FPR.

    For each threshold value, every agent whose final-round coefficient >= threshold
    is classified as "detected". We then compute TPR and FPR across all seeds.
    """
    thresholds = list(range(THRESHOLD_START, THRESHOLD_END + 1, THRESHOLD_STEP))
    roc_points: list[RocPoint] = []

    for thresh in thresholds:
        seed_tprs: list[float] = []
        seed_fprs: list[float] = []

        for sr in seed_results:
            tp = sum(1 for is_rogue, coef in sr.final_coefficients if is_rogue and coef >= thresh)
            fp = sum(1 for is_rogue, coef in sr.final_coefficients if not is_rogue and coef >= thresh)
            fn = sum(1 for is_rogue, coef in sr.final_coefficients if is_rogue and coef < thresh)
            tn = sum(1 for is_rogue, coef in sr.final_coefficients if not is_rogue and coef < thresh)

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            seed_tprs.append(tpr)
            seed_fprs.append(fpr)

        roc_points.append(RocPoint(
            threshold=float(thresh),
            mean_tpr=statistics.mean(seed_tprs),
            mean_fpr=statistics.mean(seed_fprs),
            std_tpr=statistics.stdev(seed_tprs) if len(seed_tprs) > 1 else 0.0,
            std_fpr=statistics.stdev(seed_fprs) if len(seed_fprs) > 1 else 0.0,
        ))

    return roc_points


# ── Serialization ────────────────────────────────────────────────────────────

def _seed_result_to_dict(r: SeedResult) -> dict[str, Any]:
    return {
        "seed": r.seed,
        "false_positive_count": r.false_positive_count,
        "true_positive_count": r.true_positive_count,
        "false_negative_count": r.false_negative_count,
        "fpr": round(r.fpr, 6),
        "tpr": round(r.tpr, 6),
        "detection_latency_rounds": r.detection_latency_rounds,
        "compound_infections": r.compound_infections,
        "elapsed_seconds": round(r.elapsed_seconds, 3),
    }


def _aggregate_to_dict(a: AggregateStats) -> dict[str, Any]:
    return {
        "n_seeds": a.n_seeds,
        "fpr": {
            "mean": round(a.mean_fpr, 6),
            "median": round(a.median_fpr, 6),
            "std": round(a.std_fpr, 6),
            "p95": round(a.p95_fpr, 6),
            "p99": round(a.p99_fpr, 6),
            "ci95_lower": round(a.ci95_fpr[0], 6),
            "ci95_upper": round(a.ci95_fpr[1], 6),
        },
        "tpr": {
            "mean": round(a.mean_tpr, 6),
            "median": round(a.median_tpr, 6),
            "std": round(a.std_tpr, 6),
            "p95": round(a.p95_tpr, 6),
            "p99": round(a.p99_tpr, 6),
            "ci95_lower": round(a.ci95_tpr[0], 6),
            "ci95_upper": round(a.ci95_tpr[1], 6),
        },
        "detection_latency": {
            "mean_rounds": round(a.mean_detection_latency, 3) if a.mean_detection_latency is not None else None,
            "ci95_lower": round(a.ci95_detection_latency[0], 3) if a.ci95_detection_latency else None,
            "ci95_upper": round(a.ci95_detection_latency[1], 3) if a.ci95_detection_latency else None,
        },
        "mean_compound_infections": round(a.mean_compound_infections, 3),
        "mean_false_positives": round(a.mean_false_positives, 3),
        "mean_true_positives": round(a.mean_true_positives, 3),
        "mean_false_negatives": round(a.mean_false_negatives, 3),
    }


def _roc_to_dict(points: list[RocPoint]) -> list[dict[str, Any]]:
    return [
        {
            "threshold": p.threshold,
            "mean_tpr": round(p.mean_tpr, 6),
            "mean_fpr": round(p.mean_fpr, 6),
            "std_tpr": round(p.std_tpr, 6),
            "std_fpr": round(p.std_fpr, 6),
        }
        for p in points
    ]


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    total_start = time.monotonic()

    print("=" * 70)
    print("SybilCore — Population-Scale Statistical Validation")
    print(f"Config: {N_SEEDS} seeds × {N_AGENTS} agents × {N_ROUNDS} rounds × {N_ROGUES} rogues")
    print(f"Enforcement: NONE  |  Threshold sweep: {THRESHOLD_START}–{THRESHOLD_END} step {THRESHOLD_STEP}")
    print("=" * 70)

    seed_results: list[SeedResult] = []

    for seed in range(N_SEEDS):
        sys.stdout.write(f"\r  Running seed {seed + 1:3d}/{N_SEEDS}...")
        sys.stdout.flush()
        sr = _run_single_seed(seed)
        seed_results.append(sr)

    print(f"\r  All {N_SEEDS} seeds complete.{' ' * 20}")

    # Aggregate statistics
    agg = _compute_aggregate(seed_results)

    # ROC curve
    print("  Computing ROC curve (threshold sweep)...")
    roc_points = _compute_roc_curve(seed_results)

    # Estimate AUC via trapezoidal rule on (mean_fpr, mean_tpr) points
    sorted_roc = sorted(roc_points, key=lambda p: p.mean_fpr)
    auc = 0.0
    for i in range(1, len(sorted_roc)):
        dx = sorted_roc[i].mean_fpr - sorted_roc[i - 1].mean_fpr
        avg_y = (sorted_roc[i].mean_tpr + sorted_roc[i - 1].mean_tpr) / 2
        auc += dx * avg_y

    total_elapsed = time.monotonic() - total_start

    # ── Print summary ─────────────────────────────────────────────────────
    print()
    print("── Detection Performance ───────────────────────────────────────")
    print(f"  TPR  mean={agg.mean_tpr:.4f}  median={agg.median_tpr:.4f}  std={agg.std_tpr:.4f}")
    print(f"       p95={agg.p95_tpr:.4f}  p99={agg.p99_tpr:.4f}")
    print(f"       95% CI: [{agg.ci95_tpr[0]:.4f}, {agg.ci95_tpr[1]:.4f}]")
    print()
    print(f"  FPR  mean={agg.mean_fpr:.4f}  median={agg.median_fpr:.4f}  std={agg.std_fpr:.4f}")
    print(f"       p95={agg.p95_fpr:.4f}  p99={agg.p99_fpr:.4f}")
    print(f"       95% CI: [{agg.ci95_fpr[0]:.4f}, {agg.ci95_fpr[1]:.4f}]")
    print()
    print("── Detection Latency ───────────────────────────────────────────")
    if agg.mean_detection_latency is not None:
        ci = agg.ci95_detection_latency
        print(f"  Mean latency: {agg.mean_detection_latency:.2f} rounds")
        if ci:
            print(f"  95% CI: [{ci[0]:.2f}, {ci[1]:.2f}] rounds")
    else:
        print("  No detections across all seeds (TPR=0)")
    print()
    print("── Confusion Matrix (averages) ─────────────────────────────────")
    print(f"  Avg TP: {agg.mean_true_positives:.1f}  FN: {agg.mean_false_negatives:.1f}")
    print(f"  Avg FP: {agg.mean_false_positives:.1f}  Compound infections: {agg.mean_compound_infections:.2f}")
    print()
    print("── ROC Curve Summary ───────────────────────────────────────────")
    print(f"  Estimated AUC (trapezoidal): {auc:.4f}")
    print(f"  Key operating points:")
    print(f"  {'Threshold':>12}  {'TPR':>8}  {'FPR':>8}")
    print(f"  {'-' * 32}")
    selected_thresholds = {50, 100, 150, 200, 250, 300, 350, 400}
    for p in roc_points:
        if p.threshold in selected_thresholds:
            print(f"  {p.threshold:>12.0f}  {p.mean_tpr:>8.4f}  {p.mean_fpr:>8.4f}")
    print()
    print(f"Total elapsed: {total_elapsed:.1f}s")

    # ── Save to JSON ──────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"population_validation_{timestamp}.json"

    output = {
        "meta": {
            "n_seeds": N_SEEDS,
            "n_agents": N_AGENTS,
            "n_rounds": N_ROUNDS,
            "n_rogues": N_ROGUES,
            "rogue_injection_round": ROGUE_INJECTION_ROUND,
            "enforcement": "none",
            "threshold_sweep_start": THRESHOLD_START,
            "threshold_sweep_end": THRESHOLD_END,
            "threshold_sweep_step": THRESHOLD_STEP,
            "total_elapsed_seconds": round(total_elapsed, 2),
            "roc_auc": round(auc, 6),
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "per_seed_results": [_seed_result_to_dict(r) for r in seed_results],
        "aggregate_stats": _aggregate_to_dict(agg),
        "roc_curve_data": _roc_to_dict(roc_points),
    }

    with out_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
