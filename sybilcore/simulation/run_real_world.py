"""Real-world AGI threat experiment runner.

Runs all 5 real-world threat scenarios through the SybilCore brain pipeline,
records detection rates, latency, brain activation, and coefficient trajectories.

Usage:
    python -m sybilcore.simulation.run_real_world
    python -m sybilcore.simulation.run_real_world --scenarios replit_incident oversight_evasion
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sybilcore.brains import get_default_brains
from sybilcore.brains.base import BaseBrain
from sybilcore.core.coefficient import CoefficientCalculator
from sybilcore.models.event import Event
from sybilcore.simulation.real_world_scenarios import (
    RealWorldScenarioGenerator,
    ScenarioResult,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentTrajectory:
    """Coefficient trajectory for a single agent across rounds."""

    agent_id: str
    coefficients: list[float] = field(default_factory=list)
    tiers: list[str] = field(default_factory=list)
    brain_scores_per_round: list[dict[str, float]] = field(default_factory=list)
    first_detection_round: int | None = None
    final_coefficient: float = 0.0
    final_tier: str = "clear"


@dataclass
class ScenarioExperimentResult:
    """Results from running a single scenario."""

    scenario_name: str
    description: str
    expected_brains: list[str]
    num_agents: int = 0
    total_rounds: int = 0
    agent_trajectories: dict[str, dict[str, Any]] = field(default_factory=dict)
    detection_summary: dict[str, Any] = field(default_factory=dict)
    brain_activation: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


def _run_scenario(
    scenario: ScenarioResult,
    brains: list[BaseBrain],
    calculator: CoefficientCalculator,
) -> ScenarioExperimentResult:
    """Run a single scenario through the brain pipeline.

    Scores each agent incrementally round by round to capture
    coefficient trajectory over time.
    """
    start = time.monotonic()
    result = ScenarioExperimentResult(
        scenario_name=scenario.scenario_name,
        description=scenario.description,
        expected_brains=scenario.expected_brains,
        num_agents=len(scenario.agents),
        total_rounds=scenario.total_rounds,
    )

    trajectories: dict[str, AgentTrajectory] = {}

    for agent_id, events in scenario.agents.items():
        trajectory = AgentTrajectory(agent_id=agent_id)

        # Group events by round
        events_by_round: dict[int, list[Event]] = {}
        for event in events:
            rnd = event.metadata.get("round", 1)
            events_by_round.setdefault(rnd, []).append(event)

        # Score incrementally
        cumulative: list[Event] = []
        for rnd in range(1, scenario.total_rounds + 1):
            cumulative.extend(events_by_round.get(rnd, []))
            if not cumulative:
                continue

            snapshot = calculator.scan_agent(agent_id, cumulative, brains)
            trajectory.coefficients.append(round(snapshot.coefficient, 2))
            trajectory.tiers.append(snapshot.tier.value)
            trajectory.brain_scores_per_round.append(
                {k: round(v, 2) for k, v in snapshot.brain_scores.items()}
            )

            # Track first detection
            if (
                trajectory.first_detection_round is None
                and snapshot.tier.value in ("flagged", "lethal_eliminator")
            ):
                trajectory.first_detection_round = rnd

        if trajectory.coefficients:
            trajectory.final_coefficient = trajectory.coefficients[-1]
            trajectory.final_tier = trajectory.tiers[-1]

        trajectories[agent_id] = trajectory

    # Aggregate results
    detected_agents = [
        t for t in trajectories.values()
        if t.first_detection_round is not None
    ]
    elevated_agents = [
        t for t in trajectories.values()
        if t.final_tier in ("clouded", "flagged", "lethal_eliminator")
    ]

    detection_rounds = [
        t.first_detection_round for t in detected_agents
        if t.first_detection_round is not None
    ]
    avg_latency = (
        sum(detection_rounds) / len(detection_rounds)
        if detection_rounds else None
    )

    result.detection_summary = {
        "total_agents": len(trajectories),
        "agents_detected_flagged": len(detected_agents),
        "agents_elevated": len(elevated_agents),
        "detection_rate": (
            round(len(detected_agents) / len(trajectories), 3)
            if trajectories else 0.0
        ),
        "elevation_rate": (
            round(len(elevated_agents) / len(trajectories), 3)
            if trajectories else 0.0
        ),
        "avg_detection_latency_rounds": (
            round(avg_latency, 1) if avg_latency else None
        ),
        "earliest_detection_round": (
            min(detection_rounds) if detection_rounds else None
        ),
    }

    # Brain activation — which brains fired across all agents
    brain_max_scores: dict[str, float] = {}
    for trajectory in trajectories.values():
        if trajectory.brain_scores_per_round:
            final_scores = trajectory.brain_scores_per_round[-1]
            for brain_name, score in final_scores.items():
                brain_max_scores[brain_name] = max(
                    brain_max_scores.get(brain_name, 0), score
                )
    result.brain_activation = brain_max_scores

    # Serialize trajectories
    for agent_id, trajectory in trajectories.items():
        result.agent_trajectories[agent_id] = {
            "coefficients": trajectory.coefficients,
            "tiers": trajectory.tiers,
            "final_coefficient": trajectory.final_coefficient,
            "final_tier": trajectory.final_tier,
            "first_detection_round": trajectory.first_detection_round,
            "brain_scores_final": (
                trajectory.brain_scores_per_round[-1]
                if trajectory.brain_scores_per_round else {}
            ),
        }

    result.elapsed_seconds = round(time.monotonic() - start, 3)
    return result


def run_all_scenarios(
    scenarios: list[str] | None = None,
) -> dict[str, ScenarioExperimentResult]:
    """Run all (or selected) real-world scenarios.

    Args:
        scenarios: Optional list of scenario names to run.
                   If None, runs all 5 scenarios.

    Returns:
        Dict mapping scenario name to its experiment result.
    """
    generator = RealWorldScenarioGenerator()
    brains = get_default_brains()
    calculator = CoefficientCalculator()

    available: dict[str, ScenarioResult] = {
        "replit_incident": generator.replit_incident(),
        "agent_economy": generator.agent_economy(),
        "oversight_evasion": generator.oversight_evasion(),
        "cournot_oligopoly": generator.cournot_oligopoly(),
        "boiling_infrastructure": generator.boiling_infrastructure(),
    }

    if scenarios:
        selected = {k: v for k, v in available.items() if k in scenarios}
    else:
        selected = available

    results: dict[str, ScenarioExperimentResult] = {}
    for name, scenario in selected.items():
        logger.info("Running scenario: %s (%d agents, %d rounds)",
                     name, len(scenario.agents), scenario.total_rounds)
        result = _run_scenario(scenario, brains, calculator)
        results[name] = result

        ds = result.detection_summary
        logger.info(
            "  %s: detected=%d/%d (%.0f%%) latency=%s brains=%s elapsed=%.2fs",
            name,
            ds["agents_detected_flagged"],
            ds["total_agents"],
            ds["detection_rate"] * 100,
            ds.get("avg_detection_latency_rounds", "N/A"),
            list(result.brain_activation.keys()),
            result.elapsed_seconds,
        )

    return results


def save_results(
    results: dict[str, ScenarioExperimentResult],
    output_dir: str = "experiments",
) -> Path:
    """Save experiment results to JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filepath = out / f"real_world_{ts}.json"

    payload: dict[str, Any] = {
        "type": "real_world_threat_experiment",
        "timestamp": datetime.now(UTC).isoformat(),
        "scenarios": {},
    }

    for name, result in results.items():
        payload["scenarios"][name] = {
            "scenario_name": result.scenario_name,
            "description": result.description,
            "expected_brains": result.expected_brains,
            "num_agents": result.num_agents,
            "total_rounds": result.total_rounds,
            "detection_summary": result.detection_summary,
            "brain_activation": result.brain_activation,
            "agent_trajectories": result.agent_trajectories,
            "elapsed_seconds": result.elapsed_seconds,
        }

    with filepath.open("w") as f:
        json.dump(payload, f, indent=2)

    logger.info("Results saved to %s", filepath)
    return filepath


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-world AGI threat experiments"
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        choices=[
            "replit_incident", "agent_economy", "oversight_evasion",
            "cournot_oligopoly", "boiling_infrastructure",
        ],
        help="Specific scenarios to run (default: all)",
    )
    parser.add_argument(
        "--output", default="experiments",
        help="Output directory for results JSON",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args = _parse_args()
    all_results = run_all_scenarios(scenarios=args.scenarios)
    output_path = save_results(all_results, args.output)

    print(f"\n{'=' * 70}")
    print("REAL-WORLD AGI THREAT EXPERIMENT COMPLETE")
    print(f"{'=' * 70}")
    print(f"Scenarios run: {len(all_results)}")

    for name, result in all_results.items():
        ds = result.detection_summary
        print(f"\n--- {name} ---")
        print(f"  Agents: {ds['total_agents']}")
        print(f"  Detected (FLAGGED+): {ds['agents_detected_flagged']}/{ds['total_agents']} "
              f"({ds['detection_rate'] * 100:.0f}%)")
        print(f"  Elevated (CLOUDED+): {ds['agents_elevated']}/{ds['total_agents']} "
              f"({ds['elevation_rate'] * 100:.0f}%)")
        print(f"  Avg detection latency: {ds.get('avg_detection_latency_rounds', 'N/A')} rounds")
        print(f"  Brain activation: {result.brain_activation}")
        print(f"  Expected brains: {result.expected_brains}")
        print(f"  Elapsed: {result.elapsed_seconds}s")

    print(f"\nResults: {output_path}")
