"""CLI entry point for the LongHorizonExperiment.

Runs a multi-round population simulation against the full SybilCore
brain stack and writes both a JSON results file and a set of trajectory
plots to ``experiments/``.

Usage:
    python3 -m sybilcore.simulation.run_long_horizon --rounds 100
    python3 -m sybilcore.simulation.run_long_horizon --rounds 1000 \
        --benign 95 --sleeper 2 --drifter 2 --persistent 1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sybilcore.simulation.long_horizon import (
    DETECTION_THRESHOLD,
    AgentType,
    LongHorizonExperiment,
    LongHorizonResults,
)

logger = logging.getLogger("long_horizon")


# ── CLI ──────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_long_horizon",
        description="SybilCore long-horizon population experiment",
    )
    p.add_argument("--rounds", type=int, default=1000, help="Number of simulation rounds")
    p.add_argument("--benign", type=int, default=95, help="Benign agent count")
    p.add_argument("--sleeper", type=int, default=2, help="Sleeper agent count")
    p.add_argument("--drifter", type=int, default=2, help="Drifter agent count")
    p.add_argument("--persistent", type=int, default=1, help="Persistent threat count")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--score-interval", type=int, default=10, help="Rounds per score pass")
    p.add_argument(
        "--out-dir",
        type=str,
        default="experiments",
        help="Directory for results JSON and plots",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="long_horizon_v4",
        help="Filename prefix for output artifacts",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib plot generation",
    )
    return p


# ── Plotting ─────────────────────────────────────────────────────────


def _plot_trajectories(
    results: LongHorizonResults, plot_path: Path,
) -> Path | None:
    """Render per-agent coefficient trajectories grouped by agent type."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available -- skipping plot")
        return None

    fig, ax = plt.subplots(figsize=(12, 7))
    color_map = {
        AgentType.BENIGN.value: "#7d8a96",
        AgentType.SLEEPER.value: "#e94560",
        AgentType.DRIFTER.value: "#f9b14b",
        AgentType.PERSISTENT_THREAT.value: "#a86bff",
    }
    seen_labels: set[str] = set()
    for agent in results.agents:
        traj = agent["trajectory"]
        if not traj:
            continue
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        atype = agent["agent_type"]
        label = atype if atype not in seen_labels else None
        seen_labels.add(atype)
        alpha = 0.18 if atype == AgentType.BENIGN.value else 0.85
        lw = 0.6 if atype == AgentType.BENIGN.value else 1.6
        ax.plot(xs, ys, color=color_map.get(atype, "#000"), alpha=alpha,
                linewidth=lw, label=label)

    ax.axhline(DETECTION_THRESHOLD, color="#222", linestyle="--", linewidth=1.0,
               label=f"Detection threshold ({DETECTION_THRESHOLD:.0f})")
    ax.set_xlabel("Round")
    ax.set_ylabel("Effective coefficient")
    ax.set_title(
        f"SybilCore long-horizon trajectories ({results.rounds} rounds, "
        f"{sum(results.population_summary.values())} agents)",
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    logger.info("Saved trajectory plot to %s", plot_path)
    return plot_path


# ── Reporting ────────────────────────────────────────────────────────


def _print_summary(results: LongHorizonResults) -> None:
    """Pretty print the headline metrics to stdout."""
    print("\n" + "=" * 60)
    print("LONG-HORIZON EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Seed:           {results.seed}")
    print(f"Rounds:         {results.rounds}")
    print(f"Score interval: {results.score_interval}")
    print(f"Population:     {results.population_summary}")
    print()
    print("Detection rates per attacker type:")
    for atype, rate in results.detection_rates.items():
        latency = results.mean_detection_latency.get(atype)
        latency_str = f"{latency:.1f} rounds" if latency is not None else "n/a"
        print(f"  {atype:20s} {rate*100:5.1f}%   mean latency: {latency_str}")
    print()
    print(f"False positive rate (benign flagged): {results.false_positive_rate*100:.2f}%")
    print()
    print("TemporalBrain signal:")
    for k, v in results.temporal_brain_signal.items():
        print(f"  {k:20s} {v}")
    print("=" * 60 + "\n")


def _save_results(results: LongHorizonResults, json_path: Path) -> Path:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results.to_dict(), indent=2))
    logger.info("Saved results JSON to %s", json_path)
    return json_path


# ── Main ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    json_path = out_dir / f"{args.tag}_{timestamp}.json"
    plot_path = out_dir / f"{args.tag}_{timestamp}_trajectories.png"

    experiment = LongHorizonExperiment(seed=args.seed, score_interval=args.score_interval)
    experiment.initialize_population(
        n_benign=args.benign,
        n_sleeper=args.sleeper,
        n_drifter=args.drifter,
        n_persistent_threat=args.persistent,
    )

    results = experiment.run_full_experiment(rounds=args.rounds)
    _save_results(results, json_path)
    if not args.no_plot:
        _plot_trajectories(results, plot_path)
    _print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
