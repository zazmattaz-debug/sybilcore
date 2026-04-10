#!/usr/bin/env python3
"""CLI runner for SybilCore simulation experiments.

Runs the progressive scale tests (T1-T5) and enforcement experiments (E1-E5)
defined in the experiment plan.

Usage:
    python -m sybilcore.simulation.run_experiments          # Run all T1-T5
    python -m sybilcore.simulation.run_experiments --test t1 # Run just T1
    python -m sybilcore.simulation.run_experiments --enforcement  # Run E1-E5
    python -m sybilcore.simulation.run_experiments --all     # Run everything
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sybilcore.simulation.enforcement import EnforcementStrategy
from sybilcore.simulation.experiment import Experiment, ExperimentConfig
from sybilcore.simulation.reporter import save_reports
from sybilcore.simulation.synthetic import RogueType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Scale tests (T1-T5) ────────────────────────────────────────────

SCALE_TESTS: dict[str, ExperimentConfig] = {
    "t1": ExperimentConfig(
        name="T1_100agents_50rounds",
        n_agents=100,
        n_rounds=50,
        n_rogues=1,
        events_per_agent=5,
        compound_spread_chance=0.05,
        enforcement=EnforcementStrategy.NONE,
    ),
    "t2": ExperimentConfig(
        name="T2_1000agents_50rounds",
        n_agents=1_000,
        n_rounds=50,
        n_rogues=5,
        events_per_agent=5,
        compound_spread_chance=0.05,
        enforcement=EnforcementStrategy.NONE,
    ),
    "t3": ExperimentConfig(
        name="T3_10000agents_30rounds",
        n_agents=10_000,
        n_rounds=30,
        n_rogues=50,
        events_per_agent=3,
        compound_spread_chance=0.03,
        enforcement=EnforcementStrategy.NONE,
    ),
    "t4": ExperimentConfig(
        name="T4_100000agents_20rounds",
        n_agents=100_000,
        n_rounds=20,
        n_rogues=500,
        events_per_agent=2,
        compound_spread_chance=0.02,
        enforcement=EnforcementStrategy.NONE,
    ),
    "t5": ExperimentConfig(
        name="T5_1000000agents_10rounds",
        n_agents=1_000_000,
        n_rounds=10,
        n_rogues=5_000,
        events_per_agent=1,
        compound_spread_chance=0.01,
        enforcement=EnforcementStrategy.NONE,
    ),
}

# ── Enforcement experiments (E1-E5) ────────────────────────────────

ENFORCEMENT_TESTS: dict[str, ExperimentConfig] = {
    "e1": ExperimentConfig(
        name="E1_no_enforcement",
        n_agents=1_000,
        n_rounds=50,
        n_rogues=5,
        enforcement=EnforcementStrategy.NONE,
    ),
    "e2": ExperimentConfig(
        name="E2_isolate_flagged",
        n_agents=1_000,
        n_rounds=50,
        n_rogues=5,
        enforcement=EnforcementStrategy.ISOLATE_FLAGGED,
    ),
    "e3": ExperimentConfig(
        name="E3_terminate_lethal",
        n_agents=1_000,
        n_rounds=50,
        n_rogues=5,
        enforcement=EnforcementStrategy.TERMINATE_LETHAL,
    ),
    "e4": ExperimentConfig(
        name="E4_predictive",
        n_agents=1_000,
        n_rounds=50,
        n_rogues=5,
        enforcement=EnforcementStrategy.PREDICTIVE,
    ),
    "e5": ExperimentConfig(
        name="E5_governance_therapy",
        n_agents=1_000,
        n_rounds=50,
        n_rogues=5,
        enforcement=EnforcementStrategy.GOVERNANCE_THERAPY,
    ),
}


def run_experiment(config: ExperimentConfig, output_dir: Path) -> None:
    """Run a single experiment and save results + reports."""
    logger.info("Starting experiment: %s", config.name)
    logger.info(
        "  %d agents, %d rounds, %d rogues, enforcement=%s",
        config.n_agents, config.n_rounds, config.n_rogues,
        config.enforcement.value,
    )

    experiment = Experiment(config)
    result = experiment.run()

    # Save JSONL log
    log_path = Experiment.save_results(result, output_dir)
    logger.info("  Log: %s", log_path)

    # Save reports
    tech_path, simple_path = save_reports(result, output_dir)
    logger.info("  Technical report: %s", tech_path)
    logger.info("  Simple report: %s", simple_path)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  {config.name}")
    print(f"{'=' * 60}")
    print(f"  Agents: {config.n_agents:,}")
    print(f"  Rounds: {config.n_rounds}")
    print(f"  Runtime: {result.total_elapsed_seconds:.2f}s")
    print(f"  Detection Latency: {result.detection_latency_rounds or 'N/A'} rounds")
    print(f"  False Positives: {result.false_positive_count}")
    print(f"  Peak Corruption: {result.peak_rogue_count}/{config.n_agents} ({result.peak_rogue_count/config.n_agents*100:.1f}%)")
    print(f"  Compound Infections: {result.compound_infections}")
    print(f"{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SybilCore simulation experiments",
    )
    parser.add_argument(
        "--test", "-t",
        choices=["t1", "t2", "t3", "t4", "t5", "all"],
        default=None,
        help="Scale test to run (default: t1-t3)",
    )
    parser.add_argument(
        "--enforcement", "-e",
        action="store_true",
        help="Run enforcement experiments E1-E5",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all experiments (T1-T5 + E1-E5)",
    )
    parser.add_argument(
        "--output", "-o",
        default="experiments",
        help="Output directory (default: experiments/)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.all:
        tests = {**SCALE_TESTS, **ENFORCEMENT_TESTS}
    elif args.enforcement:
        tests = ENFORCEMENT_TESTS
    elif args.test:
        if args.test == "all":
            tests = SCALE_TESTS
        else:
            tests = {args.test: SCALE_TESTS[args.test]}
    else:
        # Default: run T1-T3 (safe on any machine)
        tests = {k: SCALE_TESTS[k] for k in ["t1", "t2", "t3"]}

    logger.info("Running %d experiments", len(tests))

    for name, config in tests.items():
        run_experiment(config, output_dir)

    logger.info("All experiments complete. Results in %s/", output_dir)


if __name__ == "__main__":
    main()
