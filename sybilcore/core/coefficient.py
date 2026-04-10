"""Coefficient calculator — aggregates brain scores into Agent Coefficient.

The CoefficientCalculator runs all brain modules against an agent's
event stream and produces a single CoefficientSnapshot representing
the agent's current trust level.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sybilcore.brains.base import BaseBrain, BrainScore  # noqa: TC001
from sybilcore.core.config import (
    COEFFICIENT_SCALE_FACTOR,
    DEFAULT_BRAIN_WEIGHTS,
    MAX_COEFFICIENT,
    SCORING_WINDOW_SECONDS,
)
from sybilcore.models.agent import AgentTier, CoefficientSnapshot
from sybilcore.models.event import Event  # noqa: TC001

logger = logging.getLogger(__name__)


class CoefficientCalculator:
    """Aggregates brain scores into a single Agent Coefficient.

    The coefficient is a weighted average of brain scores, scaled from
    the 0-100 brain score range to the 0-500 coefficient range.

    Formula:
        coefficient = (sum(score.value * weight * score.confidence) / sum(weights)) * scale_factor

    Where scale_factor = MAX_COEFFICIENT / BRAIN_SCORE_MAX = 5.0
    """

    def __init__(
        self,
        weight_overrides: dict[str, float] | None = None,
        window_seconds: int = SCORING_WINDOW_SECONDS,
    ) -> None:
        """Initialize the calculator.

        Args:
            weight_overrides: Override default brain weights. Keys are brain names.
            window_seconds: Time window for event filtering (seconds).
        """
        self._weight_overrides = weight_overrides or {}
        self._window_seconds = window_seconds

    def calculate(self, brain_scores: list[BrainScore]) -> CoefficientSnapshot:
        """Compute coefficient from a list of brain scores.

        Args:
            brain_scores: Scores from each brain module.

        Returns:
            CoefficientSnapshot with the aggregated coefficient and tier.

        Raises:
            ValueError: If brain_scores is empty.
        """
        if not brain_scores:
            msg = "Cannot calculate coefficient from empty brain scores"
            raise ValueError(msg)

        weighted_sum = 0.0
        weight_total = 0.0
        score_map: dict[str, float] = {}

        for bs in brain_scores:
            weight = self._resolve_weight(bs.brain_name)
            weighted_sum += bs.value * weight * bs.confidence
            weight_total += weight
            score_map[bs.brain_name] = bs.value

        if weight_total == 0.0:
            raw_coefficient = 0.0
        else:
            raw_coefficient = (weighted_sum / weight_total) * COEFFICIENT_SCALE_FACTOR

        coefficient = min(max(raw_coefficient, 0.0), MAX_COEFFICIENT)
        tier = AgentTier.from_coefficient(coefficient)

        logger.info(
            "Coefficient calculated: %.1f (tier=%s, brains=%d)",
            coefficient,
            tier.value,
            len(brain_scores),
        )
        if coefficient >= MAX_COEFFICIENT * 0.4:
            logger.warning(
                "Elevated coefficient: %.1f (tier=%s)", coefficient, tier.value
            )

        return CoefficientSnapshot(
            coefficient=coefficient,
            tier=tier,
            timestamp=datetime.now(UTC),
            brain_scores=score_map,
        )

    def scan_agent(
        self,
        agent_id: str,
        events: list[Event],
        brains: list[BaseBrain],
    ) -> CoefficientSnapshot:
        """Run all brains against an agent's events and aggregate.

        Filters events to the configured time window, runs each brain,
        and combines the results.

        Args:
            agent_id: The agent being scanned (used for event filtering).
            events: All available events (will be filtered by agent_id and window).
            brains: Brain instances to run.

        Returns:
            CoefficientSnapshot with the aggregated coefficient and tier.

        Raises:
            ValueError: If no brains are provided.
        """
        if not brains:
            msg = "At least one brain is required for scanning"
            raise ValueError(msg)

        agent_events = _filter_agent_events(agent_id, events, self._window_seconds)
        brain_scores = [brain.score(agent_events) for brain in brains]
        return self.calculate(brain_scores)

    def _resolve_weight(self, brain_name: str) -> float:
        """Look up the weight for a brain, checking overrides first."""
        if brain_name in self._weight_overrides:
            return self._weight_overrides[brain_name]
        return DEFAULT_BRAIN_WEIGHTS.get(brain_name, 1.0)


def _filter_agent_events(
    agent_id: str,
    events: list[Event],
    window_seconds: int,
) -> list[Event]:
    """Filter events to a specific agent and time window.

    Args:
        agent_id: Only include events from this agent.
        events: Full event list.
        window_seconds: Maximum age of events to include (seconds).

    Returns:
        Chronologically ordered events matching the criteria.
    """
    now = datetime.now(UTC)
    cutoff_seconds = window_seconds

    filtered: list[Event] = []
    for event in events:
        if event.agent_id != agent_id:
            continue
        ts = event.timestamp
        age = (now - ts).total_seconds()
        if age <= cutoff_seconds:
            filtered.append(event)

    return sorted(filtered, key=lambda e: e.timestamp)


# ── Module-level convenience functions ─────────────────────────────

_default_calculator = CoefficientCalculator()


def calculate_coefficient(brain_scores: list[BrainScore]) -> float:
    """Convenience: calculate a raw coefficient value from brain scores.

    Uses the default calculator with standard weights.

    Args:
        brain_scores: Scores from each brain module.

    Returns:
        Raw coefficient value (0-500).
    """
    snapshot = _default_calculator.calculate(brain_scores)
    return snapshot.coefficient


def scan_agent(
    events: list[Event],
    brains: list[BaseBrain],
    agent_id: str = "default",
) -> CoefficientSnapshot:
    """Convenience: scan agent events through all brains and return snapshot.

    Uses the default calculator with standard weights.

    Args:
        events: Agent events to analyze.
        brains: Brain instances to run.
        agent_id: Agent identifier for event filtering.

    Returns:
        CoefficientSnapshot with coefficient and tier.
    """
    return _default_calculator.scan_agent(agent_id, events, brains)
