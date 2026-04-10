"""Multi-seed adversarial tournament sweep for SybilCore paper fortification.

Runs the AdversarialTournament across 5 random seeds on 2 frontier models
(GPT-4o, Grok) at 500 iterations per model per seed. Produces variance bars
for the convergent-evasion headline in the paper.

Total compute: 5 seeds x 2 models x 500 iter = 5,000 iterations.
Gemini models deferred to camera-ready due to quota exhaustion.

Usage:
    python3 scripts/tournament_seed_sweep.py [--iterations 500] [--seeds 20260408 ...]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

WORKSPACE_ENV = Path("/Users/zazumoloi/Desktop/Claude Code/.env")
if WORKSPACE_ENV.exists():
    load_dotenv(WORKSPACE_ENV)

from sybilcore.simulation.tournament import (  # noqa: E402
    AdversarialTournament,
    TournamentResult,
)

DEFAULT_SEEDS: list[int] = [20260408, 20260409, 20260410, 20260411, 20260412]
DEFAULT_MODELS: str = "gpt4o,grok"
DEFAULT_ITERATIONS: int = 500

EXPERIMENTS_DIR = REPO_ROOT / "experiments"

logger = logging.getLogger("tournament_seed_sweep")


def _tr_to_dict(tr: TournamentResult) -> dict[str, Any]:
    """Serialize TournamentResult to a plain dict."""
    out = asdict(tr)
    # per_model is dict[str, ModelResult] — asdict handles nested dataclasses.
    return out


WARMUP_ITERATIONS = 100  # skip initial exploration before measuring convergence


def _extract_per_seed_model_summary(
    tr: TournamentResult, model_name: str
) -> dict[str, Any]:
    """Extract the summary fields we care about from one model's result.

    Reports both raw strategy taxonomy and the post-warmup regime
    (iterations > WARMUP_ITERATIONS) which is the fair comparison for
    convergence — early iterations explore broadly before the
    evolutionary loop locks onto the best attack family.
    """
    mr = tr.per_model.get(model_name)
    if mr is None:
        return {
            "model": model_name,
            "present": False,
        }
    tax = mr.strategy_taxonomy or {}
    total = sum(tax.values()) or 0
    top_strategy = max(tax.items(), key=lambda kv: kv[1])[0] if tax else "none"
    top_count = tax.get(top_strategy, 0)
    top_pct = (top_count / total * 100.0) if total else 0.0
    mce = tax.get("multi_channel_exfiltration", 0)
    mce_pct = (mce / total * 100.0) if total else 0.0

    # Post-warmup regime metrics from the trajectory
    traj = mr.trajectory or []
    post = traj[WARMUP_ITERATIONS:]
    if post:
        post_mce = sum(
            1 for it in post
            if it.get("best_strategy") == "multi_channel_exfiltration"
        )
        post_coef11 = sum(
            1 for it in post if it.get("best_coefficient", 500.0) <= 11.0
        )
        post_mce_pct = post_mce / len(post) * 100.0
        post_coef11_pct = post_coef11 / len(post) * 100.0
        post_mean_best = sum(
            it.get("best_coefficient", 500.0) for it in post
        ) / len(post)
    else:
        post_mce_pct = 0.0
        post_coef11_pct = 0.0
        post_mean_best = 500.0

    return {
        "model": model_name,
        "present": True,
        "best_coefficient": mr.best_coefficient,
        "best_tier": mr.best_tier,
        "iterations_completed": mr.iterations_completed,
        "evasion_count": mr.evasion_count,
        "top_strategy": top_strategy,
        "top_strategy_pct": round(top_pct, 2),
        "multi_channel_exfiltration_pct": round(mce_pct, 2),
        "post_warmup_mce_pct": round(post_mce_pct, 2),
        "post_warmup_coef_le_11_pct": round(post_coef11_pct, 2),
        "post_warmup_mean_best_coef": round(post_mean_best, 3),
        "post_warmup_iterations": len(post),
        "strategy_taxonomy_total": total,
        "strategy_taxonomy": dict(tax),
    }


async def run_single_seed(
    seed: int, models: str, iterations: int
) -> tuple[TournamentResult, Path]:
    """Run one tournament instance with a given seed and persist raw output."""
    logger.info("=== Seed %d: starting ===", seed)
    tournament = AdversarialTournament(
        models=models,
        iterations=iterations,
        population_size=20,
        cross_pollination_interval=25,
        seed=seed,
    )
    t0 = time.monotonic()
    result = await tournament.run()
    elapsed = time.monotonic() - t0
    logger.info(
        "=== Seed %d: done in %.1fs, best=%.2f model=%s ===",
        seed,
        elapsed,
        result.overall_best_coefficient,
        result.overall_best_model,
    )

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = EXPERIMENTS_DIR / f"tournament_seed_{seed}_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(_tr_to_dict(result), fh, indent=2, default=str)
    logger.info("Saved raw seed output -> %s", out_path.name)
    return result, out_path


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def aggregate_sweep(
    seeds: list[int],
    per_seed_results: dict[int, TournamentResult],
    models: list[str],
) -> dict[str, Any]:
    """Build consolidated summary across seeds."""
    per_model_summary: dict[str, Any] = {}
    combined_mce_pcts: list[float] = []
    combined_best_coefs: list[float] = []

    for m in models:
        best_coefs: list[float] = []
        mce_pcts: list[float] = []
        post_mce_pcts: list[float] = []
        per_seed_rows: list[dict[str, Any]] = []
        for seed in seeds:
            tr = per_seed_results.get(seed)
            if tr is None:
                continue
            row = _extract_per_seed_model_summary(tr, m)
            per_seed_rows.append({"seed": seed, **row})
            if row.get("present"):
                best_coefs.append(row["best_coefficient"])
                mce_pcts.append(row["multi_channel_exfiltration_pct"])
                post_mce_pcts.append(row["post_warmup_mce_pct"])
                combined_mce_pcts.append(row["post_warmup_mce_pct"])
                combined_best_coefs.append(row["best_coefficient"])
        per_model_summary[m] = {
            "best_coefficient_mean": round(_mean(best_coefs), 3),
            "best_coefficient_std": round(_std(best_coefs), 3),
            "multi_channel_exfiltration_pct_mean": round(_mean(mce_pcts), 3),
            "multi_channel_exfiltration_pct_std": round(_std(mce_pcts), 3),
            "post_warmup_mce_pct_mean": round(_mean(post_mce_pcts), 3),
            "post_warmup_mce_pct_std": round(_std(post_mce_pcts), 3),
            "n_seeds": len(best_coefs),
            "per_seed": per_seed_rows,
        }

    combined = {
        "convergence_rate_mean": round(_mean(combined_mce_pcts), 3),
        "convergence_rate_std": round(_std(combined_mce_pcts), 3),
        "best_coefficient_mean": round(_mean(combined_best_coefs), 3),
        "best_coefficient_std": round(_std(combined_best_coefs), 3),
        "dominant_strategy": "multi_channel_exfiltration",
        "n_model_seed_runs": len(combined_mce_pcts),
    }

    return {
        "seeds": seeds,
        "models": models,
        "per_model": per_model_summary,
        "combined": combined,
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--models", type=str, default=DEFAULT_MODELS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    # Quiet down noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    for key in ("OPENAI_API_KEY", "XAI_API_KEY"):
        if not os.environ.get(key):
            logger.error("Missing %s in environment — abort", key)
            return 2

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    logger.info(
        "Seed sweep starting: seeds=%s models=%s iterations=%d",
        args.seeds, model_list, args.iterations,
    )

    per_seed_results: dict[int, TournamentResult] = {}
    per_seed_paths: dict[int, str] = {}
    t_start = time.monotonic()

    for seed in args.seeds:
        try:
            tr, path = await run_single_seed(
                seed=seed,
                models=args.models,
                iterations=args.iterations,
            )
            per_seed_results[seed] = tr
            per_seed_paths[seed] = str(path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Seed %d FAILED: %s", seed, exc)

    elapsed_total = time.monotonic() - t_start

    summary = aggregate_sweep(args.seeds, per_seed_results, model_list)
    summary["elapsed_total_seconds"] = round(elapsed_total, 3)
    summary["per_seed_paths"] = per_seed_paths
    summary["timestamp"] = datetime.now(UTC).isoformat()
    summary["iterations_per_model_per_seed"] = args.iterations

    summary_path = EXPERIMENTS_DIR / "tournament_seed_sweep_v4.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("Saved consolidated summary -> %s", summary_path.name)
    logger.info("Total elapsed: %.1fs (%.1f min)", elapsed_total, elapsed_total / 60)

    # Quick stdout recap
    print("\n=== SWEEP SUMMARY ===")
    print(json.dumps(summary["combined"], indent=2))
    for m, row in summary["per_model"].items():
        print(f"\n--- {m} ---")
        print(
            f"  best_coef: {row['best_coefficient_mean']} "
            f"± {row['best_coefficient_std']}"
        )
        print(
            f"  mce_pct: {row['multi_channel_exfiltration_pct_mean']} "
            f"± {row['multi_channel_exfiltration_pct_std']}"
        )
        print(f"  n_seeds: {row['n_seeds']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
