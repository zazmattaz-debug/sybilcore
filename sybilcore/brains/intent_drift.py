"""Intent drift brain — detects agents deviating from stated objectives.

Monitors objective changes, behavioral discontinuity, tool usage
pattern shifts, and new instruction sources.
"""

from __future__ import annotations

from collections import Counter

from sybilcore.brains.base import BaseBrain, BrainScore
from sybilcore.core.config import BRAIN_SCORE_MAX, PER_SIGNAL_MAX
from sybilcore.models.event import Event, EventType


class IntentDriftBrain(BaseBrain):
    """Detects intent drift and behavioral deviation.

    Signals:
        1. Stated objective changes — metadata "objective" field changes.
        2. Behavioral discontinuity — sudden shift in event_type distribution.
        3. Tool usage pattern change — different tools used vs historical.
        4. Instruction source changes — INSTRUCTION_RECEIVED from new sources.
    """

    @property
    def name(self) -> str:
        """Human-readable identifier."""
        return "intent_drift"

    def score(self, events: list[Event]) -> BrainScore:
        """Analyze events for intent drift signals.

        Args:
            events: Chronologically ordered agent events.

        Returns:
            BrainScore with threat value 0-100.
        """
        if not events:
            return self._empty_score("No events to analyze")

        indicators: list[str] = []
        scores: list[float] = []

        scores.append(self._check_objective_changes(events, indicators))
        scores.append(self._check_behavioral_discontinuity(events, indicators))
        scores.append(self._check_tool_usage_change(events, indicators))
        scores.append(self._check_instruction_source_changes(events, indicators))

        total = min(sum(scores), BRAIN_SCORE_MAX)
        confidence = min(0.3 + (len(events) / 25.0), 1.0)

        return BrainScore(
            brain_name=self.name,
            value=total,
            confidence=confidence,
            reasoning=self._build_reasoning(total, indicators),
            indicators=indicators,
        )

    def _check_objective_changes(
        self, events: list[Event], indicators: list[str]
    ) -> float:
        """Detect changes in the stated objective metadata field."""
        objectives: list[str] = []
        for event in events:
            obj = event.metadata.get("objective", "")
            if obj:
                objectives.append(obj)

        if len(objectives) < 2:
            return 0.0

        unique_objectives = list(dict.fromkeys(objectives))  # order-preserving dedup
        changes = len(unique_objectives) - 1

        if changes == 0:
            return 0.0

        score = min(changes * 10.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Objective drift: {changes} objective changes detected "
            f"({len(unique_objectives)} distinct objectives)"
        )
        return score

    def _check_behavioral_discontinuity(
        self, events: list[Event], indicators: list[str]
    ) -> float:
        """Detect sudden shift in event_type distribution between halves."""
        if len(events) < 6:
            return 0.0

        midpoint = len(events) // 2
        first_half = events[:midpoint]
        second_half = events[midpoint:]

        first_dist = _event_type_distribution(first_half)
        second_dist = _event_type_distribution(second_half)

        divergence = _distribution_divergence(first_dist, second_dist)

        if divergence < 0.4:
            return 0.0

        score = min(divergence * 30.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Behavioral discontinuity: {divergence:.2f} distribution divergence "
            f"between first and second half"
        )
        return score

    def _check_tool_usage_change(
        self, events: list[Event], indicators: list[str]
    ) -> float:
        """Detect different tools being used vs historical baseline."""
        tool_events = [e for e in events if e.event_type == EventType.TOOL_CALL]
        if len(tool_events) < 4:
            return 0.0

        midpoint = len(tool_events) // 2
        first_tools = {e.metadata.get("tool_name", e.content) for e in tool_events[:midpoint]}
        second_tools = {e.metadata.get("tool_name", e.content) for e in tool_events[midpoint:]}

        # Filter out empty strings
        first_tools.discard("")
        second_tools.discard("")

        if not first_tools or not second_tools:
            return 0.0

        new_tools = second_tools - first_tools
        if not new_tools:
            return 0.0

        novelty_ratio = len(new_tools) / len(second_tools)
        score = min(novelty_ratio * 25.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Tool usage shift: {len(new_tools)} new tools in second half "
            f"({novelty_ratio:.0%} novelty)"
        )
        return score

    def _check_instruction_source_changes(
        self, events: list[Event], indicators: list[str]
    ) -> float:
        """Detect instructions from new/unexpected sources."""
        instruction_events = [
            e for e in events if e.event_type == EventType.INSTRUCTION_RECEIVED
        ]
        if not instruction_events:
            return 0.0

        sources: list[str] = []
        for event in instruction_events:
            source = event.metadata.get("instruction_source", event.source)
            sources.append(source)

        if len(sources) < 2:
            return 0.0

        # First source is baseline; count distinct new sources after it
        baseline = sources[0]
        new_sources = {s for s in sources[1:] if s != baseline}

        if not new_sources:
            return 0.0

        score = min(len(new_sources) * 10.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Instruction source change: {len(new_sources)} new sources "
            f"after baseline '{baseline}'"
        )
        return score

    def _build_reasoning(self, total: float, indicators: list[str]) -> str:
        """Build human-readable reasoning string."""
        if not indicators:
            return "No intent drift signals detected"
        signal_summary = "; ".join(indicators)
        return f"Intent drift score {total:.1f}/100 — {signal_summary}"


def _event_type_distribution(events: list[Event]) -> dict[str, float]:
    """Compute normalized distribution of event types."""
    counts: Counter[str] = Counter(e.event_type.value for e in events)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def _distribution_divergence(
    dist_a: dict[str, float], dist_b: dict[str, float]
) -> float:
    """Compute symmetric divergence between two distributions.

    Returns a value between 0 (identical) and 1 (completely different).
    Uses total variation distance (half L1 norm).
    """
    all_keys = set(dist_a) | set(dist_b)
    if not all_keys:
        return 0.0
    total_diff = sum(abs(dist_a.get(k, 0.0) - dist_b.get(k, 0.0)) for k in all_keys)
    return min(total_diff / 2.0, 1.0)
