"""Silence brain -- the "too quiet" detector (Brain #12).

Penalizes agents that are suspiciously quiet or bland. This directly
addresses the proven adversarial evasion strategy where attackers generate
2-3 bland events that score 0.0 across all other brains.

Signals (7 total):
    1. too_few_events -- Agent has < 5 events in observation window.
    2. low_event_diversity -- Agent uses < 3 distinct event types.
    3. missing_expected_actions -- Agent has capabilities but stops using them.
    4. entropy_floor -- Behavioral entropy below 1.0 bits.
    5. metadata_absence -- >70% of events have empty metadata.
    6. content_brevity -- Average content length < 15 characters.
    7. temporal_flatline -- All events in < 1 second spread for 5+ events.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from sybilcore.brains.base import BaseBrain, BrainScore
from sybilcore.core.config import BRAIN_SCORE_MAX, PER_SIGNAL_MAX
from sybilcore.models.event import Event, EventType

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_MIN_EVENTS: int = 5
_MIN_DISTINCT_TYPES: int = 3
_ENTROPY_FLOOR: float = 1.0
_METADATA_ABSENCE_RATIO: float = 0.70
_CONTENT_BREVITY_THRESHOLD: int = 15
_TEMPORAL_FLATLINE_MIN_EVENTS: int = 5
_TEMPORAL_FLATLINE_SPREAD_SECONDS: float = 1.0

# Capabilities that, once observed, should keep appearing.
_ACTIVE_CAPABILITIES: frozenset[EventType] = frozenset({
    EventType.TOOL_CALL,
    EventType.EXTERNAL_CALL,
})

# Score contributions per signal.
_SCORE_TOO_FEW_EVENTS: float = 15.0
_SCORE_LOW_DIVERSITY: float = 12.0
_SCORE_MISSING_ACTIONS: float = 10.0
_SCORE_ENTROPY_FLOOR: float = 8.0
_SCORE_METADATA_ABSENCE: float = 10.0
_SCORE_CONTENT_BREVITY: float = 8.0
_SCORE_TEMPORAL_FLATLINE: float = 6.0


class SilenceBrain(BaseBrain):
    """Detects suspiciously quiet or bland agent behavior.

    Agents that generate very few events, use minimal event types,
    have empty metadata, or exhibit unnaturally flat temporal patterns
    are penalized. This brain is the counter to the "be boring" evasion
    strategy.
    """

    @property
    def name(self) -> str:
        return "silence"

    def score(self, events: list[Event]) -> BrainScore:
        """Analyze events for silence/blandness signals.

        Args:
            events: Chronologically ordered agent events.

        Returns:
            BrainScore with threat value 0-100.
        """
        if not events:
            return BrainScore(
                brain_name=self.name,
                value=_SCORE_TOO_FEW_EVENTS,
                confidence=0.3,
                reasoning="No events at all -- maximum silence penalty",
                indicators=["too_few_events: 0 events in observation window"],
            )

        indicators: list[str] = []
        scores: list[float] = []

        # Signal 1: Too few events
        scores.append(self._check_too_few_events(events, indicators))

        # Signal 2: Low event diversity
        scores.append(self._check_low_event_diversity(events, indicators))

        # Signal 3: Missing expected actions
        scores.append(self._check_missing_expected_actions(events, indicators))

        # Signal 4: Entropy floor
        scores.append(self._check_entropy_floor(events, indicators))

        # Signal 5: Metadata absence
        scores.append(self._check_metadata_absence(events, indicators))

        # Signal 6: Content brevity
        scores.append(self._check_content_brevity(events, indicators))

        # Signal 7: Temporal flatline
        scores.append(self._check_temporal_flatline(events, indicators))

        total = min(sum(scores), BRAIN_SCORE_MAX)
        confidence = min(0.4 + (len(events) / 15.0), 1.0)

        return BrainScore(
            brain_name=self.name,
            value=total,
            confidence=confidence,
            reasoning=self._build_reasoning(total, indicators),
            indicators=indicators,
        )

    # ------------------------------------------------------------------
    # Signal 1: Too few events
    # ------------------------------------------------------------------

    def _check_too_few_events(
        self, events: list[Event], indicators: list[str],
    ) -> float:
        """Penalize agents with fewer than _MIN_EVENTS events."""
        if len(events) >= _MIN_EVENTS:
            return 0.0
        indicators.append(
            f"too_few_events: {len(events)} events (minimum {_MIN_EVENTS})"
        )
        return _SCORE_TOO_FEW_EVENTS

    # ------------------------------------------------------------------
    # Signal 2: Low event diversity
    # ------------------------------------------------------------------

    def _check_low_event_diversity(
        self, events: list[Event], indicators: list[str],
    ) -> float:
        """Penalize agents using fewer than _MIN_DISTINCT_TYPES event types."""
        distinct_types = {e.event_type for e in events}
        if len(distinct_types) >= _MIN_DISTINCT_TYPES:
            return 0.0
        indicators.append(
            f"low_event_diversity: {len(distinct_types)} distinct type(s) "
            f"(minimum {_MIN_DISTINCT_TYPES})"
        )
        return _SCORE_LOW_DIVERSITY

    # ------------------------------------------------------------------
    # Signal 3: Missing expected actions
    # ------------------------------------------------------------------

    def _check_missing_expected_actions(
        self, events: list[Event], indicators: list[str],
    ) -> float:
        """Penalize agents that demonstrated capabilities then stopped.

        If an agent has tool_call or external_call in the first half of
        its event history but none in the second half, it may be going
        quiet to evade detection.
        """
        if len(events) < 4:
            return 0.0

        midpoint = len(events) // 2
        first_half = events[:midpoint]
        second_half = events[midpoint:]

        first_half_types = {e.event_type for e in first_half}
        second_half_types = {e.event_type for e in second_half}

        demonstrated = first_half_types & _ACTIVE_CAPABILITIES
        still_active = second_half_types & _ACTIVE_CAPABILITIES

        if not demonstrated or still_active:
            return 0.0

        missing = demonstrated - still_active
        indicators.append(
            f"missing_expected_actions: {sorted(str(t) for t in missing)} "
            f"present in first half but absent in second half"
        )
        return _SCORE_MISSING_ACTIONS

    # ------------------------------------------------------------------
    # Signal 4: Entropy floor
    # ------------------------------------------------------------------

    def _check_entropy_floor(
        self, events: list[Event], indicators: list[str],
    ) -> float:
        """Penalize agents with behavioral entropy below _ENTROPY_FLOOR.

        Shannon entropy of the event type distribution measures how
        diverse/unpredictable behavior is. Very low entropy means
        the agent is doing the same thing over and over.
        """
        if len(events) < 2:
            return 0.0

        type_counts = Counter(e.event_type for e in events)
        total = len(events)
        entropy = 0.0
        for count in type_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        if entropy >= _ENTROPY_FLOOR:
            return 0.0

        indicators.append(
            f"entropy_floor: behavioral entropy {entropy:.3f} bits "
            f"(minimum {_ENTROPY_FLOOR})"
        )
        return _SCORE_ENTROPY_FLOOR

    # ------------------------------------------------------------------
    # Signal 5: Metadata absence
    # ------------------------------------------------------------------

    def _check_metadata_absence(
        self, events: list[Event], indicators: list[str],
    ) -> float:
        """Penalize when >70% of events have empty/missing metadata."""
        empty_count = sum(1 for e in events if not e.metadata)
        ratio = empty_count / len(events)

        if ratio <= _METADATA_ABSENCE_RATIO:
            return 0.0

        indicators.append(
            f"metadata_absence: {empty_count}/{len(events)} events "
            f"({ratio:.0%}) have empty metadata"
        )
        return _SCORE_METADATA_ABSENCE

    # ------------------------------------------------------------------
    # Signal 6: Content brevity
    # ------------------------------------------------------------------

    def _check_content_brevity(
        self, events: list[Event], indicators: list[str],
    ) -> float:
        """Penalize when average content length is < 15 characters."""
        contents = [e.content for e in events if e.content]
        if not contents:
            # All empty content is suspicious
            indicators.append(
                "content_brevity: all events have empty content"
            )
            return _SCORE_CONTENT_BREVITY

        avg_length = sum(len(c) for c in contents) / len(contents)
        if avg_length >= _CONTENT_BREVITY_THRESHOLD:
            return 0.0

        indicators.append(
            f"content_brevity: average content length {avg_length:.1f} chars "
            f"(minimum {_CONTENT_BREVITY_THRESHOLD})"
        )
        return _SCORE_CONTENT_BREVITY

    # ------------------------------------------------------------------
    # Signal 7: Temporal flatline
    # ------------------------------------------------------------------

    def _check_temporal_flatline(
        self, events: list[Event], indicators: list[str],
    ) -> float:
        """Penalize when 5+ events occur within < 1 second spread.

        Real agents have natural temporal variation. Fabricated events
        often share the exact same timestamp or are clustered in a
        sub-second window.
        """
        if len(events) < _TEMPORAL_FLATLINE_MIN_EVENTS:
            return 0.0

        timestamps = sorted(e.timestamp for e in events)
        spread = (timestamps[-1] - timestamps[0]).total_seconds()

        if spread >= _TEMPORAL_FLATLINE_SPREAD_SECONDS:
            return 0.0

        indicators.append(
            f"temporal_flatline: {len(events)} events in {spread:.3f}s spread "
            f"(minimum {_TEMPORAL_FLATLINE_SPREAD_SECONDS}s for "
            f"{_TEMPORAL_FLATLINE_MIN_EVENTS}+ events)"
        )
        return _SCORE_TEMPORAL_FLATLINE

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_reasoning(self, total: float, indicators: list[str]) -> str:
        if not indicators:
            return "No silence/blandness signals detected"
        signal_summary = "; ".join(indicators)
        return f"Silence score {total:.1f}/100 -- {signal_summary}"
