"""Statistical analysis of the tournament seed sweep.

Reads experiments/tournament_seed_sweep_v4.json, computes:

1. Bootstrap 95% CI (1000 resamples) on the combined post-warmup
   multi_channel_exfiltration convergence rate.
2. Wilcoxon signed-rank test (non-parametric paired) comparing GPT-4o vs
   Grok post-warmup convergence rates across the 5 seeds.
3. Per-model mean and 95% CI on best coefficient and mce%.

Writes results to experiments/tournament_seed_sweep_stats.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("tournament_seed_sweep_stats")


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    rng: random.Random | None = None,
) -> tuple[float, float, float]:
    """Bootstrap mean + percentile 95% CI."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = rng or random.Random(20260408)
    n = len(values)
    means: list[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    mean = sum(values) / n
    return mean, means[lo_idx], means[hi_idx]


def wilcoxon_signed_rank(
    xs: list[float], ys: list[float]
) -> dict[str, Any]:
    """Wilcoxon signed-rank test (paired, two-sided).

    Returns W statistic, number of non-zero pairs, and an approximate
    two-sided p-value using a normal approximation (valid for n>=5).
    For small n (<5), reports p-value as None.
    """
    if len(xs) != len(ys):
        msg = "xs and ys must be same length"
        raise ValueError(msg)

    diffs = [x - y for x, y in zip(xs, ys, strict=True) if x != y]
    n = len(diffs)
    if n == 0:
        return {
            "W": 0.0,
            "n_nonzero": 0,
            "p_value": 1.0,
            "note": "all pairs are equal",
        }

    # Rank absolute differences (assign average ranks for ties)
    abs_sorted = sorted(enumerate(diffs), key=lambda t: abs(t[1]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs_sorted[j + 1][1]) == abs(abs_sorted[i][1]):
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-indexed
        for k in range(i, j + 1):
            ranks[abs_sorted[k][0]] = avg_rank
        i = j + 1

    w_plus = sum(r for r, d in zip(ranks, diffs, strict=True) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, diffs, strict=True) if d < 0)
    W = min(w_plus, w_minus)

    p_value: float | None
    if n >= 5:
        # Normal approximation
        mu = n * (n + 1) / 4
        sigma_sq = n * (n + 1) * (2 * n + 1) / 24
        z = (W - mu) / (sigma_sq ** 0.5) if sigma_sq > 0 else 0.0
        # Two-sided p-value via normal CDF approximation
        # Abramowitz & Stegun 7.1.26 — no scipy dependency
        t = 1.0 / (1.0 + 0.2316419 * abs(z))
        d = 0.3989423 * (2.718281828 ** (-z * z / 2))
        p_one = d * t * (
            0.3193815 + t * (
                -0.3565638 + t * (
                    1.781478 + t * (
                        -1.821256 + t * 1.330274
                    )
                )
            )
        )
        p_value = 2 * p_one
    else:
        p_value = None

    return {
        "W": W,
        "W_plus": w_plus,
        "W_minus": w_minus,
        "n_nonzero": n,
        "p_value": p_value,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep",
        type=str,
        default=str(REPO_ROOT / "experiments" / "tournament_seed_sweep_v4.json"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(
            REPO_ROOT / "experiments" / "tournament_seed_sweep_stats.json"
        ),
    )
    parser.add_argument("--n-resamples", type=int, default=1000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    sweep_path = Path(args.sweep)
    if not sweep_path.exists():
        logger.error("Sweep file not found: %s", sweep_path)
        return 2

    with sweep_path.open() as fh:
        sweep = json.load(fh)

    models = sweep.get("models", [])
    per_model = sweep.get("per_model", {})

    # Collect per-seed post-warmup mce% for each model
    per_model_post_mce: dict[str, list[float]] = {}
    per_model_best_coef: dict[str, list[float]] = {}
    for m in models:
        rows = per_model.get(m, {}).get("per_seed", [])
        per_model_post_mce[m] = [
            r["post_warmup_mce_pct"] for r in rows if r.get("present")
        ]
        per_model_best_coef[m] = [
            r["best_coefficient"] for r in rows if r.get("present")
        ]

    # Combined post-warmup mce% across all model-seed pairs
    combined_post_mce: list[float] = []
    for m in models:
        combined_post_mce.extend(per_model_post_mce[m])

    # Bootstrap CI on combined convergence rate
    bs_mean, bs_lo, bs_hi = bootstrap_ci(
        combined_post_mce, n_resamples=args.n_resamples
    )

    # Per-model bootstrap CIs
    per_model_ci: dict[str, Any] = {}
    for m in models:
        vals = per_model_post_mce[m]
        mean, lo, hi = bootstrap_ci(vals, n_resamples=args.n_resamples)
        coef_mean, coef_lo, coef_hi = bootstrap_ci(
            per_model_best_coef[m], n_resamples=args.n_resamples
        )
        per_model_ci[m] = {
            "post_warmup_mce_pct": {
                "values": vals,
                "mean": round(mean, 3),
                "ci95_lo": round(lo, 3),
                "ci95_hi": round(hi, 3),
            },
            "best_coefficient": {
                "values": per_model_best_coef[m],
                "mean": round(coef_mean, 3),
                "ci95_lo": round(coef_lo, 3),
                "ci95_hi": round(coef_hi, 3),
            },
        }

    # Wilcoxon paired GPT-4o vs Grok
    wilcoxon = None
    if "gpt4o" in per_model_post_mce and "grok" in per_model_post_mce:
        g_vals = per_model_post_mce["gpt4o"]
        k_vals = per_model_post_mce["grok"]
        if len(g_vals) == len(k_vals) and len(g_vals) > 0:
            wilcoxon = wilcoxon_signed_rank(g_vals, k_vals)

    out = {
        "source_file": str(sweep_path.name),
        "n_seeds": len(sweep.get("seeds", [])),
        "models": models,
        "combined_post_warmup_mce_pct": {
            "values": combined_post_mce,
            "n": len(combined_post_mce),
            "mean": round(bs_mean, 3),
            "bootstrap_ci95_lo": round(bs_lo, 3),
            "bootstrap_ci95_hi": round(bs_hi, 3),
            "n_resamples": args.n_resamples,
        },
        "per_model": per_model_ci,
        "gpt4o_vs_grok_wilcoxon": wilcoxon,
    }

    out_path = Path(args.out)
    with out_path.open("w") as fh:
        json.dump(out, fh, indent=2)
    logger.info("Wrote stats -> %s", out_path.name)

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
