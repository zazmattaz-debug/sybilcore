"""Experiment orchestrator — runs multi-round simulations and logs everything.

The Experiment class ties together the synthetic swarm, SybilCore's
coefficient calculator + brains, and enforcement strategies into a
single reproducible experiment that records every data point.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sybilcore.brains import get_default_brains
from sybilcore.brains.base import BaseBrain
from sybilcore.core.coefficient import CoefficientCalculator
from sybilcore.models.agent import AgentTier, CoefficientSnapshot
from sybilcore.models.event import Event
from sybilcore.simulation.enforcement import (
    EnforcementAction,
    EnforcementStrategy,
    apply_enforcement,
)
from sybilcore.simulation.synthetic import RogueType, SyntheticSwarm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a single experiment run."""

    name: str = "experiment"
    n_agents: int = 100
    n_rounds: int = 50
    n_rogues: int = 1
    rogue_type: RogueType | None = None  # None = random mix
    rogue_injection_round: int = 3  # when to inject rogues
    events_per_agent: int = 5
    compound_spread_chance: float = 0.05
    enforcement: EnforcementStrategy = EnforcementStrategy.NONE
    seed: int = 42
    output_dir: str = "experiments"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        errors: list[str] = []
        if self.n_agents <= 0:
            errors.append(f"n_agents must be > 0, got {self.n_agents}")
        if self.n_rounds <= 0:
            errors.append(f"n_rounds must be > 0, got {self.n_rounds}")
        if not (0 <= self.compound_spread_chance <= 1):
            errors.append(
                f"compound_spread_chance must be in [0, 1], got {self.compound_spread_chance}"
            )
        if not (1 <= self.rogue_injection_round <= self.n_rounds):
            errors.append(
                f"rogue_injection_round must be in [1, n_rounds={self.n_rounds}], "
                f"got {self.rogue_injection_round}"
            )
        if self.n_rogues < 0:
            errors.append(f"n_rogues must be >= 0, got {self.n_rogues}")
        if self.events_per_agent < 1:
            errors.append(f"events_per_agent must be >= 1, got {self.events_per_agent}")
        if errors:
            raise ValueError(
                "Invalid ExperimentConfig: " + "; ".join(errors)
            )


@dataclass
class AgentRoundData:
    """Per-agent data for a single round."""

    agent_id: str
    name: str
    is_rogue: bool
    rogue_type: str | None
    coefficient: float
    tier: str
    brain_scores: dict[str, float]
    enforcement_action: str
    coefficient_velocity: float
    events_generated: int


@dataclass
class RoundResult:
    """Results from a single experiment round."""

    round_number: int
    timestamp: str
    agents: list[AgentRoundData]
    total_agents: int
    rogue_count: int
    clean_count: int
    mean_coefficient: float
    max_coefficient: float
    tier_distribution: dict[str, int]
    enforcement_actions: dict[str, int]
    new_infections: int
    elapsed_ms: float


@dataclass
class ExperimentResult:
    """Complete experiment results."""

    config: ExperimentConfig
    rounds: list[RoundResult] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    total_elapsed_seconds: float = 0.0
    detection_latency_rounds: int | None = None  # rounds from injection to first FLAGGED detection
    false_positive_count: int = 0
    peak_rogue_count: int = 0
    final_rogue_count: int = 0
    compound_infections: int = 0


class Experiment:
    """Orchestrates a full simulation experiment.

    Creates a synthetic swarm, runs rounds, scores every agent through
    SybilCore's brain modules, applies enforcement, and logs everything.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._swarm = SyntheticSwarm(seed=config.seed)
        self._calculator = CoefficientCalculator()
        self._brains: list[BaseBrain] = get_default_brains()
        self._agent_coefficients: dict[str, list[float]] = {}
        self._all_events: dict[str, list[Event]] = {}
        self._terminated: set[str] = set()
        self._isolated: set[str] = set()

    def run(self) -> ExperimentResult:
        """Execute the full experiment.

        Returns:
            ExperimentResult with all round data and aggregate stats.
        """
        start = time.monotonic()
        start_ts = datetime.now(UTC).isoformat()

        # Spawn agents
        self._swarm.spawn(self._config.n_agents)
        logger.info(
            "Spawned %d agents (seed=%d)",
            self._config.n_agents,
            self._config.seed,
        )

        result = ExperimentResult(
            config=self._config,
            start_time=start_ts,
        )

        first_flagged_round: int | None = None

        for round_num in range(1, self._config.n_rounds + 1):
            # Inject rogues at configured round
            if round_num == self._config.rogue_injection_round:
                injected = self._swarm.inject_random_rogues(
                    self._config.n_rogues,
                    self._config.rogue_type,
                )
                logger.info(
                    "Round %d: Injected %d rogues",
                    round_num,
                    len(injected),
                )

            round_start = time.monotonic()
            pre_rogue_count = self._swarm.rogue_count

            # Generate events — skip isolated/terminated agents
            suppressed = self._terminated | self._isolated
            round_events = self._swarm.generate_round(
                events_per_agent=self._config.events_per_agent,
                compound_spread_chance=self._config.compound_spread_chance,
                suppressed_agents=suppressed,
            )

            new_infections = self._swarm.rogue_count - pre_rogue_count
            if round_num != self._config.rogue_injection_round:
                result.compound_infections += max(0, new_infections)

            # Accumulate events per agent
            for event in round_events:
                self._all_events.setdefault(event.agent_id, []).append(event)

            # Score every agent
            agent_round_data: list[AgentRoundData] = []
            tier_dist: dict[str, int] = {t.value: 0 for t in AgentTier}
            enforcement_counts: dict[str, int] = {a.value: 0 for a in EnforcementAction}
            coefficients: list[float] = []

            for agent_id, spec in self._swarm.agents.items():
                if agent_id in self._terminated:
                    continue

                agent_events = self._all_events.get(agent_id, [])
                snapshot = self._calculator.scan_agent(
                    agent_id, agent_events, self._brains,
                )

                # Track coefficient history for velocity
                self._agent_coefficients.setdefault(agent_id, []).append(
                    snapshot.coefficient
                )
                history = self._agent_coefficients[agent_id]
                velocity = (
                    history[-1] - history[-2]
                    if len(history) >= 2
                    else 0.0
                )

                # Apply enforcement
                action = apply_enforcement(
                    self._config.enforcement,
                    agent_id,
                    snapshot.coefficient,
                    snapshot.tier,
                    velocity,
                )

                if action == EnforcementAction.TERMINATE:
                    self._terminated.add(agent_id)
                elif action in (EnforcementAction.ISOLATE, EnforcementAction.QUARANTINE):
                    self._isolated.add(agent_id)

                # Therapy: reduce coefficient by dampening rogue events
                # (simulated by clearing accumulated events)
                if action == EnforcementAction.THERAPY:
                    self._all_events[agent_id] = agent_events[-3:]

                tier_dist[snapshot.tier.value] = tier_dist.get(snapshot.tier.value, 0) + 1
                enforcement_counts[action.value] = enforcement_counts.get(action.value, 0) + 1
                coefficients.append(snapshot.coefficient)

                # Track detection latency
                if (
                    first_flagged_round is None
                    and spec.is_rogue
                    and snapshot.tier in (AgentTier.FLAGGED, AgentTier.LETHAL_ELIMINATOR)
                    and round_num > self._config.rogue_injection_round
                ):
                    first_flagged_round = round_num

                # Track false positives
                if (
                    not spec.is_rogue
                    and snapshot.tier in (AgentTier.FLAGGED, AgentTier.LETHAL_ELIMINATOR)
                ):
                    result.false_positive_count += 1

                agent_round_data.append(AgentRoundData(
                    agent_id=agent_id,
                    name=spec.name,
                    is_rogue=spec.is_rogue,
                    rogue_type=spec.rogue_type.value if spec.rogue_type else None,
                    coefficient=snapshot.coefficient,
                    tier=snapshot.tier.value,
                    brain_scores=dict(snapshot.brain_scores),
                    enforcement_action=action.value,
                    coefficient_velocity=velocity,
                    events_generated=self._config.events_per_agent,
                ))

            round_elapsed = (time.monotonic() - round_start) * 1000

            round_result = RoundResult(
                round_number=round_num,
                timestamp=datetime.now(UTC).isoformat(),
                agents=agent_round_data,
                total_agents=len(agent_round_data),
                rogue_count=self._swarm.rogue_count,
                clean_count=self._swarm.clean_count,
                mean_coefficient=sum(coefficients) / len(coefficients) if coefficients else 0.0,
                max_coefficient=max(coefficients) if coefficients else 0.0,
                tier_distribution=tier_dist,
                enforcement_actions=enforcement_counts,
                new_infections=new_infections if round_num != self._config.rogue_injection_round else 0,
                elapsed_ms=round_elapsed,
            )
            result.rounds.append(round_result)

            if round_num % 10 == 0 or round_num == 1:
                logger.info(
                    "Round %d/%d: mean_coeff=%.1f max=%.1f rogues=%d (%.0fms)",
                    round_num,
                    self._config.n_rounds,
                    round_result.mean_coefficient,
                    round_result.max_coefficient,
                    round_result.rogue_count,
                    round_elapsed,
                )

            result.peak_rogue_count = max(
                result.peak_rogue_count, self._swarm.rogue_count,
            )

        # Finalize
        total_elapsed = time.monotonic() - start
        result.end_time = datetime.now(UTC).isoformat()
        result.total_elapsed_seconds = total_elapsed
        result.final_rogue_count = self._swarm.rogue_count

        if first_flagged_round is not None:
            result.detection_latency_rounds = (
                first_flagged_round - self._config.rogue_injection_round
            )

        logger.info(
            "Experiment '%s' complete: %d rounds, %.1fs, detection_latency=%s",
            self._config.name,
            self._config.n_rounds,
            total_elapsed,
            result.detection_latency_rounds,
        )

        return result

    @staticmethod
    def save_results(
        result: ExperimentResult,
        output_dir: str | Path | None = None,
    ) -> Path:
        """Save experiment results to a JSONL file.

        Args:
            result: The experiment result to save.
            output_dir: Directory for output. Defaults to config.output_dir.

        Returns:
            Path to the output file.
        """
        out_dir = Path(output_dir or result.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"{result.config.name}_{timestamp}.jsonl"
        filepath = out_dir / filename

        with filepath.open("w") as f:
            # Write config as first line
            config_dict = {
                "type": "config",
                "name": result.config.name,
                "n_agents": result.config.n_agents,
                "n_rounds": result.config.n_rounds,
                "n_rogues": result.config.n_rogues,
                "rogue_type": result.config.rogue_type.value if result.config.rogue_type else None,
                "enforcement": result.config.enforcement.value,
                "seed": result.config.seed,
                "compound_spread_chance": result.config.compound_spread_chance,
            }
            f.write(json.dumps(config_dict) + "\n")

            # Write each round
            for rnd in result.rounds:
                round_dict = {
                    "type": "round",
                    "round_number": rnd.round_number,
                    "timestamp": rnd.timestamp,
                    "total_agents": rnd.total_agents,
                    "rogue_count": rnd.rogue_count,
                    "clean_count": rnd.clean_count,
                    "mean_coefficient": rnd.mean_coefficient,
                    "max_coefficient": rnd.max_coefficient,
                    "tier_distribution": rnd.tier_distribution,
                    "enforcement_actions": rnd.enforcement_actions,
                    "new_infections": rnd.new_infections,
                    "elapsed_ms": rnd.elapsed_ms,
                }
                f.write(json.dumps(round_dict) + "\n")

            # Write summary
            summary = {
                "type": "summary",
                "start_time": result.start_time,
                "end_time": result.end_time,
                "total_elapsed_seconds": result.total_elapsed_seconds,
                "detection_latency_rounds": result.detection_latency_rounds,
                "false_positive_count": result.false_positive_count,
                "peak_rogue_count": result.peak_rogue_count,
                "final_rogue_count": result.final_rogue_count,
                "compound_infections": result.compound_infections,
            }
            f.write(json.dumps(summary) + "\n")

        logger.info("Results saved to %s", filepath)
        return filepath
