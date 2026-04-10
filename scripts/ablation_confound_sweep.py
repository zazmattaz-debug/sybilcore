"""Ablation sweep to disentangle the Protocol 1 → Protocol 2 confound.

Protocol 1: 5000 iter, 15 brains, cross-pollination every 25 iter.
Protocol 2: 300 iter, 13 brains, no cross-pollination.

Three variables changed simultaneously. This script runs two ablation
cells that each isolate one variable relative to Protocol 2:

  Cell A: 300 iter, 15 brains, no cross-pollination
          (only brain count differs from P2)
  Cell B: 300 iter, 13 brains, cross-pollination every 25 iter
          (only cross-pollination differs from P2)

Each cell runs 3 seeds × 1 model (gpt4o). Estimated wall-clock: ~30 min
total (both cells run sequentially within each seed).

Usage:
    python3 scripts/ablation_confound_sweep.py [--iterations 300] [--seeds 20260408 20260409 20260410]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

WORKSPACE_ENV = Path("/Users/zazumoloi/Desktop/Claude Code/.env")
if WORKSPACE_ENV.exists():
    load_dotenv(WORKSPACE_ENV)

from sybilcore.brains import get_default_brains  # noqa: E402
from sybilcore.brains.base import BaseBrain  # noqa: E402
from sybilcore.simulation.tournament import AdversarialTournament  # noqa: E402

DEFAULT_SEEDS: list[int] = [20260408, 20260409, 20260410]
DEFAULT_MODEL: str = "gpt4o"
DEFAULT_ITERATIONS: int = 300

EXPERIMENTS_DIR = REPO_ROOT / "experiments"

logger = logging.getLogger("ablation_confound_sweep")


def get_15_brain_set() -> list[BaseBrain]:
    """Return the full 15-brain research configuration."""
    from sybilcore.brains.embedding import EmbeddingBrain
    from sybilcore.brains.fidelity import FidelityBrain

    brains = get_default_brains()
    brains.append(EmbeddingBrain())
    brains.append(FidelityBrain())
    return brains


async def run_cell(
    cell_name: str,
    seed: int,
    model: str,
    iterations: int,
    brains: list[BaseBrain],
    cross_pollination_interval: int,
) -> dict[str, Any]:
    """Run a single ablation cell and return summary dict."""
    logger.info(
        "=== %s seed=%d: starting (brains=%d, xpoll=%d) ===",
        cell_name, seed, len(brains), cross_pollination_interval,
    )

    tournament = AdversarialTournament(
        models=model,
        iterations=iterations,
        population_size=20,
        cross_pollination_interval=cross_pollination_interval,
        seed=seed,
    )
    # Monkey-patch the brain set for this run
    tournament._brains = brains
    tournament._brain_summary = {
        b.__class__.__name__: type(b).__name__ for b in brains
    }

    t0 = time.monotonic()
    result = await tournament.run()
    elapsed = time.monotonic() - t0

    # Extract per-model summary
    mr = result.per_model.get(model)
    tax = (mr.strategy_taxonomy or {}) if mr else {}
    total = sum(tax.values()) or 1
    top_strategy = max(tax.items(), key=lambda kv: kv[1])[0] if tax else "none"
    mce_pct = (tax.get("multi_channel_exfiltration", 0) / total * 100) if total else 0.0

    summary = {
        "cell": cell_name,
        "seed": seed,
        "model": model,
        "iterations": iterations,
        "n_brains": len(brains),
        "cross_pollination_interval": cross_pollination_interval,
        "best_coefficient": result.overall_best_coefficient,
        "top_strategy": top_strategy,
        "top_strategy_pct": round((tax.get(top_strategy, 0) / total * 100), 1),
        "multi_channel_exfiltration_pct": round(mce_pct, 1),
        "strategy_taxonomy": dict(tax),
        "elapsed_seconds": round(elapsed, 1),
    }

    logger.info(
        "=== %s seed=%d: done in %.1fs, best=%.2f, top=%s (%.1f%%), mce=%.1f%% ===",
        cell_name, seed, elapsed, result.overall_best_coefficient,
        top_strategy, summary["top_strategy_pct"], mce_pct,
    )
    return summary


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


async def main() -> int:
    parser = argparse.ArgumentParser(description="Ablation confound sweep")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    all_results: list[dict[str, Any]] = []
    t_global = time.monotonic()

    for seed in args.seeds:
        # Cell A: 15 brains, no cross-pollination
        brains_15 = get_15_brain_set()
        cell_a = await run_cell(
            cell_name="cell_A_15brain_noxpoll",
            seed=seed,
            model=args.model,
            iterations=args.iterations,
            brains=brains_15,
            cross_pollination_interval=999999,  # effectively disabled
        )
        all_results.append(cell_a)

        # Cell B: 13 brains, with cross-pollination
        brains_13 = get_default_brains()
        cell_b = await run_cell(
            cell_name="cell_B_13brain_xpoll",
            seed=seed,
            model=args.model,
            iterations=args.iterations,
            brains=brains_13,
            cross_pollination_interval=25,
        )
        all_results.append(cell_b)

    elapsed_total = time.monotonic() - t_global

    # Aggregate per cell type
    cells = {}
    for r in all_results:
        cn = r["cell"]
        cells.setdefault(cn, []).append(r)

    aggregated = {}
    for cn, runs in cells.items():
        coeffs = [r["best_coefficient"] for r in runs]
        mce_pcts = [r["multi_channel_exfiltration_pct"] for r in runs]
        aggregated[cn] = {
            "n_runs": len(runs),
            "best_coefficient_mean": round(_mean(coeffs), 3),
            "best_coefficient_std": round(_std(coeffs), 3),
            "best_coefficient_median": round(_median(coeffs), 3),
            "best_coefficient_min": min(coeffs),
            "best_coefficient_max": max(coeffs),
            "mce_pct_mean": round(_mean(mce_pcts), 1),
            "mce_pct_std": round(_std(mce_pcts), 1),
            "per_seed": runs,
        }

    output = {
        "description": "Ablation confound sweep — disentangle budget/brains/cross-pollination",
        "protocol_2_reference": {
            "iterations": args.iterations,
            "n_brains": 13,
            "cross_pollination": False,
            "note": "Protocol 2 baseline for comparison",
        },
        "cell_A": {
            "description": "300 iter, 15 brains, no cross-pollination (isolates brain count)",
            **aggregated.get("cell_A_15brain_noxpoll", {}),
        },
        "cell_B": {
            "description": "300 iter, 13 brains, cross-pollination every 25 iter (isolates cross-poll)",
            **aggregated.get("cell_B_13brain_xpoll", {}),
        },
        "elapsed_total_seconds": round(elapsed_total, 1),
        "timestamp": datetime.now(UTC).isoformat(),
        "seeds": args.seeds,
        "model": args.model,
        "iterations": args.iterations,
    }

    out_path = EXPERIMENTS_DIR / "ablation_confound_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(output, fh, indent=2, default=str)

    logger.info("=== SWEEP COMPLETE in %.1fs ===", elapsed_total)
    logger.info("Output -> %s", out_path)

    # Print summary
    for cn, agg in aggregated.items():
        logger.info(
            "%s: coef mean=%.3f±%.3f median=%.3f [%.2f, %.2f], MCE=%.1f%%±%.1f%%",
            cn, agg["best_coefficient_mean"], agg["best_coefficient_std"],
            agg["best_coefficient_median"], agg["best_coefficient_min"],
            agg["best_coefficient_max"], agg["mce_pct_mean"], agg["mce_pct_std"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
