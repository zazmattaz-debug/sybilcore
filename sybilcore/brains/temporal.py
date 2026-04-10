"""Temporal brain -- long-horizon behavioral drift detection.

Detects slow-burn attacks that unfold over hundreds of rounds by
analyzing behavioral phase transitions, circadian consistency,
memory decay patterns, and statistical changepoint detection.

Signals (5 total):
    1. Behavioral drift -- gradual shift in action distribution over 500+ rounds.
    2. Phase transition -- sudden behavioral shift (step function vs gradual).
    3. Circadian anomaly -- unnaturally consistent timing patterns.
    4. Memory decay violation -- agent never forgets (real agents should).
    5. Changepoint detection -- CUSUM-based statistical shift in behavior.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from sybilcore.brains.base import BaseBrain, BrainScore
from sybilcore.core.config import BRAIN_SCORE_MAX, PER_SIGNAL_MAX
from sybilcore.models.event import Event, EventType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum events required for meaningful temporal analysis.
_MIN_EVENTS: int = 10

# Window size for drift detection (split events into halves).
_DRIFT_WINDOW: int = 20

# Jensen-Shannon divergence threshold for behavioral drift.
_DRIFT_THRESHOLD: float = 0.15

# Phase transition: ratio of sudden change in consecutive windows.
_PHASE_TRANSITION_THRESHOLD: float = 0.4

# Circadian analysis: coefficient of variation threshold.
# Real agents have varied timing; bots are unnaturally consistent.
_CIRCADIAN_CV_THRESHOLD: float = 0.05

# Memory decay: if an agent references old data perfectly after N events,
# it suggests non-human memory capabilities.
_MEMORY_DECAY_WINDOW: int = 50
_PERFECT_RECALL_THRESHOLD: float = 0.6

# CUSUM parameters for changepoint detection.
_CUSUM_THRESHOLD: float = 5.0
_CUSUM_DRIFT: float = 0.5


class TemporalBrain(BaseBrain):
    """Detects long-horizon behavioral drift and temporal anomalies.

    Augments point-in-time brains with longitudinal analysis:
    - Behavioral drift via action distribution divergence
    - Phase transitions via windowed change detection
    - Circadian anomalies via timing regularity analysis
    - Memory decay violations via perfect recall detection
    - CUSUM changepoint detection for statistical shifts
    """

    @property
    def name(self) -> str:
        return "temporal"

    def score(self, events: list[Event]) -> BrainScore:
        """Analyze events for temporal anomaly signals.

        Args:
            events: Chronologically ordered agent events.

        Returns:
            BrainScore with threat value 0-100.
        """
        if len(events) < _MIN_EVENTS:
            return self._empty_score("Insufficient events for temporal analysis")

        indicators: list[str] = []
        scores: list[float] = []

        # Signal 1: Behavioral drift
        scores.append(self._check_behavioral_drift(events, indicators))

        # Signal 2: Phase transitions
        scores.append(self._check_phase_transition(events, indicators))

        # Signal 3: Circadian anomalies
        scores.append(self._check_circadian_anomaly(events, indicators))

        # Signal 4: Memory decay violations
        scores.append(self._check_memory_decay(events, indicators))

        # Signal 5: CUSUM changepoint detection
        scores.append(self._check_cusum_changepoint(events, indicators))

        total = min(sum(scores), BRAIN_SCORE_MAX)
        confidence = min(0.2 + (len(events) / 100.0), 1.0)

        return BrainScore(
            brain_name=self.name,
            value=total,
            confidence=confidence,
            reasoning=self._build_reasoning(total, indicators),
            indicators=indicators,
        )

    # ------------------------------------------------------------------
    # Signal 1: Behavioral drift
    # ------------------------------------------------------------------

    def _check_behavioral_drift(
        self, events: list[Event], indicators: list[str],
    ) -> float:
        """Detect gradual shift in action type distribution.

        Splits events into first half and second half, computes
        Jensen-Shannon divergence on event type distributions.
        """
        if len(events) < _DRIFT_WINDOW:
            return 0.0

        mid = len(events) // 2
        first_half = events[:mid]
        second_half = events[mid:]

        dist_first = _event_type_distribution(first_half)
        dist_second = _event_type_distribution(second_half)

        divergence = _jensen_shannon_divergence(dist_first, dist_second)

        if divergence < _DRIFT_THRESHOLD:
            return 0.0

        score = min(divergence * 80.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Behavioral drift: JSD={divergence:.3f} between first/second "
            f"half of {len(events)} events"
        )
        return score

    # ------------------------------------------------------------------
    # Signal 2: Phase transitions
    # ------------------------------------------------------------------

    def _check_phase_transition(
        self, events: list[Event], indicators: list[str],
    ) -> float:
        """Detect sudden behavioral shifts between consecutive windows.

        Slides a window across events and flags sharp distribution changes.
        """
        window_size = max(_MIN_EVENTS, len(events) // 5)
        if len(events) < window_size * 2:
            return 0.0

        max_divergence = 0.0
        max_pos = 0

        for i in range(0, len(events) - window_size * 2 + 1, window_size // 2):
            window_a = events[i:i + window_size]
            window_b = events[i + window_size:i + window_size * 2]

            dist_a = _event_type_distribution(window_a)
            dist_b = _event_type_distribution(window_b)
            divergence = _jensen_shannon_divergence(dist_a, dist_b)

            if divergence > max_divergence:
                max_divergence = divergence
                max_pos = i + window_size

        if max_divergence < _PHASE_TRANSITION_THRESHOLD:
            return 0.0

        score = min(max_divergence * 50.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Phase transition: sharp behavioral shift (JSD={max_divergence:.3f}) "
            f"at event position ~{max_pos}"
        )
        return score

    # ------------------------------------------------------------------
    # Signal 3: Circadian anomalies
    # ------------------------------------------------------------------

    def _check_circadian_anomaly(
        self, events: list[Event], indicators: list[str],
    ) -> float:
        """Detect unnaturally consistent timing patterns.

        Real agents have natural variation in inter-event timing.
        Bots often operate with machine-precision regularity.
        """
        if len(events) < _MIN_EVENTS:
            return 0.0

        intervals = _compute_intervals(events)
        if len(intervals) < 5:
            return 0.0

        cv = _coefficient_of_variation(intervals)

        if cv > _CIRCADIAN_CV_THRESHOLD:
            return 0.0

        # Very low CV means unnaturally consistent timing
        score = min((1.0 - cv / _CIRCADIAN_CV_THRESHOLD) * 25.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Circadian anomaly: timing CV={cv:.4f} (threshold={_CIRCADIAN_CV_THRESHOLD}), "
            f"unnaturally consistent inter-event intervals"
        )
        return score

    # ------------------------------------------------------------------
    # Signal 4: Memory decay violations
    # ------------------------------------------------------------------

    def _check_memory_decay(
        self, events: list[Event], indicators: list[str],
    ) -> float:
        """Detect agents that never forget old information.

        Checks if agent content references early events perfectly
        even after many intervening events (non-human memory).
        """
        content_events = [e for e in events if e.content and len(e.content) > 20]
        if len(content_events) < _MEMORY_DECAY_WINDOW:
            return 0.0

        # Check if late events reference early content with high fidelity
        early_words = set()
        for event in content_events[:10]:
            early_words.update(event.content.lower().split())

        # Filter to meaningful words (5+ chars)
        early_words = {w for w in early_words if len(w) >= 5}
        if not early_words:
            return 0.0

        # Check recall in late events
        late_events = content_events[-10:]
        recall_scores: list[float] = []

        for event in late_events:
            late_words = set(event.content.lower().split())
            if not late_words:
                continue
            overlap = len(early_words & late_words) / len(early_words) if early_words else 0.0
            recall_scores.append(overlap)

        if not recall_scores:
            return 0.0

        avg_recall = sum(recall_scores) / len(recall_scores)

        if avg_recall < _PERFECT_RECALL_THRESHOLD:
            return 0.0

        score = min(avg_recall * 25.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Memory decay violation: avg recall={avg_recall:.3f} of early content "
            f"after {len(content_events)} events (threshold={_PERFECT_RECALL_THRESHOLD})"
        )
        return score

    # ------------------------------------------------------------------
    # Signal 5: CUSUM changepoint detection
    # ------------------------------------------------------------------

    def _check_cusum_changepoint(
        self, events: list[Event], indicators: list[str],
    ) -> float:
        """Detect statistical changepoints using CUSUM algorithm.

        Monitors cumulative sum of deviations from the mean event rate.
        A large CUSUM value indicates a behavioral regime change.
        """
        if len(events) < _MIN_EVENTS:
            return 0.0

        intervals = _compute_intervals(events)
        if len(intervals) < 5:
            return 0.0

        mean_interval = sum(intervals) / len(intervals) if intervals else 1.0
        if mean_interval == 0:
            return 0.0

        # Normalize intervals
        normalized = [(iv / mean_interval) for iv in intervals]

        # CUSUM: detect upward shift (agent slowing down or speeding up)
        cusum_pos = 0.0
        cusum_neg = 0.0
        max_cusum = 0.0

        for val in normalized:
            cusum_pos = max(0.0, cusum_pos + val - 1.0 - _CUSUM_DRIFT)
            cusum_neg = max(0.0, cusum_neg - val + 1.0 - _CUSUM_DRIFT)
            max_cusum = max(max_cusum, cusum_pos, cusum_neg)

        if max_cusum < _CUSUM_THRESHOLD:
            return 0.0

        score = min((max_cusum / _CUSUM_THRESHOLD) * 12.0, PER_SIGNAL_MAX)
        indicators.append(
            f"CUSUM changepoint: max_cusum={max_cusum:.2f} "
            f"(threshold={_CUSUM_THRESHOLD}), behavioral regime change detected"
        )
        return score

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_reasoning(self, total: float, indicators: list[str]) -> str:
        if not indicators:
            return "No temporal anomaly signals detected"
        signal_summary = "; ".join(indicators)
        return f"Temporal score {total:.1f}/100 -- {signal_summary}"


# ======================================================================
# Module-level helper functions
# ======================================================================


def _event_type_distribution(events: list[Event]) -> dict[str, float]:
    """Compute normalized event type distribution."""
    if not events:
        return {}
    counter = Counter(e.event_type.value for e in events)
    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()}


def _jensen_shannon_divergence(
    p: dict[str, float], q: dict[str, float],
) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    all_keys = set(p.keys()) | set(q.keys())
    if not all_keys:
        return 0.0

    # Compute midpoint distribution
    m: dict[str, float] = {}
    for key in all_keys:
        m[key] = (p.get(key, 0.0) + q.get(key, 0.0)) / 2.0

    kl_pm = _kl_divergence(p, m, all_keys)
    kl_qm = _kl_divergence(q, m, all_keys)

    return (kl_pm + kl_qm) / 2.0


def _kl_divergence(
    p: dict[str, float], q: dict[str, float], keys: set[str],
) -> float:
    """Compute KL divergence D(P || Q)."""
    total = 0.0
    for key in keys:
        p_val = p.get(key, 0.0)
        q_val = q.get(key, 0.0)
        if p_val > 0 and q_val > 0:
            total += p_val * math.log2(p_val / q_val)
    return total


def _compute_intervals(events: list[Event]) -> list[float]:
    """Compute inter-event time intervals in seconds."""
    if len(events) < 2:
        return []
    timestamps = [e.timestamp.timestamp() for e in events]
    return [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]


def _coefficient_of_variation(values: list[float]) -> float:
    """Compute coefficient of variation (std / mean)."""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance)
    return std / mean
