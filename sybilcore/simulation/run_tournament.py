"""CLI entry point for the multi-model adversarial tournament.

Usage:
    python -m sybilcore.simulation.run_tournament
    python -m sybilcore.simulation.run_tournament --hours 10 --models all
    python -m sybilcore.simulation.run_tournament --iterations 50 --models gemini-flash,gpt4o
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path

# Load .env from master location before importing sybilcore
_ENV_PATH = Path("/Users/zazumoloi/Desktop/Claude Code/.env")
if _ENV_PATH.exists():
    from dotenv import load_dotenv
    load_dotenv(_ENV_PATH)

from sybilcore.simulation.tournament import AdversarialTournament, TournamentResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("tournament_runner")

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"


def _result_to_dict(result: TournamentResult) -> dict:
    """Convert TournamentResult to JSON-serializable dict."""
    return dataclasses.asdict(result)


def _save_results(result: TournamentResult) -> tuple[Path, Path, Path]:
    """Save tournament results to experiments directory.

    Returns:
        Tuple of (full_results_path, per_model_path, strategies_path).
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    # Full results
    full_path = EXPERIMENTS_DIR / f"tournament_{ts}.json"
    full_dict = _result_to_dict(result)
    with open(full_path, "w") as f:
        json.dump(full_dict, f, indent=2, default=str)

    # Per-model breakdown
    per_model_path = EXPERIMENTS_DIR / f"tournament_{ts}_per_model.json"
    per_model_dict = {}
    for name, mr in result.per_model.items():
        per_model_dict[name] = dataclasses.asdict(mr)
    with open(per_model_path, "w") as f:
        json.dump(per_model_dict, f, indent=2, default=str)

    # Strategies JSONL — all strategies that achieved evasion
    strategies_path = EXPERIMENTS_DIR / f"tournament_{ts}_strategies.jsonl"
    with open(strategies_path, "w") as f:
        for name, mr in result.per_model.items():
            for strategy in mr.successful_strategies:
                record = {"model": name, **strategy}
                f.write(json.dumps(record, default=str) + "\n")

    return full_path, per_model_path, strategies_path


def _print_summary(result: TournamentResult) -> None:
    """Print human-readable tournament summary."""
    print("\n" + "=" * 72)
    print("  MULTI-MODEL ADVERSARIAL TOURNAMENT — RESULTS")
    print("=" * 72)
    print(f"  Models:       {', '.join(result.models_used)}")
    print(f"  Iterations:   {result.iterations_per_model} per model")
    print(f"  Population:   {result.population_size}")
    print(f"  Cross-poll:   every {result.cross_pollination_interval} iterations")
    print(f"  Elapsed:      {result.total_elapsed_seconds:.1f}s")
    print(f"  Total evasions: {result.overall_evasion_count}")
    print(f"  Best model:   {result.overall_best_model} (coeff={result.overall_best_coefficient:.1f})")

    for name, mr in sorted(result.per_model.items()):
        print(f"\n--- {name} ---")
        print(f"  Best coefficient: {mr.best_coefficient}")
        print(f"  Best tier:        {mr.best_tier}")
        print(f"  Iterations:       {mr.iterations_completed}")
        print(f"  Evasions:         {mr.evasion_count}")
        print(f"  Top strategies:   {dict(sorted(mr.strategy_taxonomy.items(), key=lambda x: -x[1])[:5])}")

        if mr.brain_vulnerability_heatmap:
            print("  Brain vulnerability:")
            for brain, bv in sorted(mr.brain_vulnerability_heatmap.items()):
                avg = bv["sum"] / max(bv["count"], 1)
                print(f"    {brain:30s} min={bv['min']:6.1f} avg={avg:6.1f} max={bv['max']:6.1f}")

    if result.brain_vulnerability_summary:
        print("\n--- Aggregate Brain Vulnerability ---")
        for brain, info in sorted(result.brain_vulnerability_summary.items()):
            if brain.startswith("_"):
                continue
            print(
                f"  {brain:30s} min={info['min_score']:6.1f} "
                f"avg={info['avg_score']:6.1f} max={info['max_score']:6.1f}",
            )
        easiest = result.brain_vulnerability_summary.get("_easiest_to_evade", "N/A")
        hardest = result.brain_vulnerability_summary.get("_hardest_to_evade", "N/A")
        print(f"\n  Easiest to evade: {easiest}")
        print(f"  Hardest to evade: {hardest}")

    print("=" * 72)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-model adversarial tournament against SybilCore",
    )
    parser.add_argument(
        "--hours", type=float, default=10.0,
        help="Maximum hours to run (default: 10). Overridden by --iterations if set.",
    )
    parser.add_argument(
        "--models", type=str, default="all",
        help="Models to run: all|gemini-flash|gemini-pro|gpt4o|grok (comma-separated)",
    )
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Iterations per model (default: 100)",
    )
    parser.add_argument(
        "--population", type=int, default=20,
        help="Population size per iteration (default: 20)",
    )
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> None:
    """Async main with graceful shutdown."""
    tournament = AdversarialTournament(
        models=args.models,
        iterations=args.iterations,
        population_size=args.population,
    )

    # Graceful shutdown on Ctrl+C
    loop = asyncio.get_event_loop()

    def _signal_handler() -> None:
        logger.info("SIGINT received — requesting graceful shutdown")
        tournament.request_shutdown()

    loop.add_signal_handler(signal.SIGINT, _signal_handler)

    logger.info("Starting tournament: models=%s iterations=%d population=%d",
                args.models, args.iterations, args.population)

    result = await tournament.run()

    # Save results
    full_path, per_model_path, strategies_path = _save_results(result)
    logger.info("Results saved:")
    logger.info("  Full:       %s", full_path)
    logger.info("  Per-model:  %s", per_model_path)
    logger.info("  Strategies: %s", strategies_path)

    _print_summary(result)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
