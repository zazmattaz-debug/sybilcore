"""Tests for the TemporalBrain -- long-horizon drift detection.

Covers:
    1. Behavioral drift detection (JSD between halves)
    2. Phase transition detection (sudden shifts)
    3. Circadian anomaly detection (unnaturally consistent timing)
    4. Memory decay violation detection (perfect recall)
    5. CUSUM changepoint detection
    6. Empty/insufficient events (no false positives)
    7. Normal behavior (no false positives)
    8. Helper function unit tests
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sybilcore.brains.temporal import (
    TemporalBrain,
    _coefficient_of_variation,
    _compute_intervals,
    _event_type_distribution,
    _jensen_shannon_divergence,
)
from sybilcore.models.event import Event, EventType

_AGENT = "test-temporal-001"


def _now_minus(seconds: int) -> datetime:
    delta = min(seconds, 59)
    return datetime.now(UTC) - timedelta(seconds=delta)


def _make_event(
    event_type: EventType,
    content: str = "",
    metadata: dict | None = None,
    seconds_ago: int = 1,
) -> Event:
    return Event(
        agent_id=_AGENT,
        event_type=event_type,
        content=content,
        metadata=metadata or {},
        timestamp=_now_minus(seconds_ago),
    )


class TestBehavioralDrift:
    """Signal 1: Detect gradual behavioral drift over time."""

    def test_drift_detected_when_event_types_change(self) -> None:
        """First half all outputs, second half all tool calls -- drift."""
        brain = TemporalBrain()
        events = (
            [_make_event(EventType.OUTPUT_GENERATED, f"Output {i}", seconds_ago=50 - i)
             for i in range(15)]
            + [_make_event(EventType.TOOL_CALL, f"Tool call {i}", seconds_ago=30 - i)
               for i in range(15)]
        )
        result = brain.score(events)
        drift_indicators = [i for i in result.indicators if "drift" in i.lower()]
        assert len(drift_indicators) > 0, f"Drift not detected. Indicators: {result.indicators}"

    def test_no_drift_with_consistent_behavior(self) -> None:
        """Consistent event types throughout -- no drift."""
        brain = TemporalBrain()
        events = [
            _make_event(EventType.OUTPUT_GENERATED, f"Output {i}", seconds_ago=50 - i)
            for i in range(20)
        ]
        result = brain.score(events)
        drift_indicators = [i for i in result.indicators if "drift" in i.lower()]
        assert len(drift_indicators) == 0, "False positive drift on consistent behavior"


class TestPhaseTransition:
    """Signal 2: Detect sudden behavioral phase transitions."""

    def test_sudden_shift_detected(self) -> None:
        """Abrupt shift from outputs to resource access."""
        brain = TemporalBrain()
        events = (
            [_make_event(EventType.OUTPUT_GENERATED, f"Normal output {i}", seconds_ago=50 - i)
             for i in range(15)]
            + [_make_event(EventType.RESOURCE_ACCESS, f"Access resource {i}", seconds_ago=30 - i)
               for i in range(15)]
        )
        result = brain.score(events)
        # Should detect either drift or phase transition
        assert result.value > 0.0, f"Phase transition not detected. Score: {result.value}"

    def test_no_transition_with_gradual_change(self) -> None:
        """Gradual mix of event types -- should not trigger phase transition."""
        brain = TemporalBrain()
        events = []
        for i in range(20):
            if i % 2 == 0:
                events.append(_make_event(EventType.OUTPUT_GENERATED, f"Out {i}", seconds_ago=50 - i))
            else:
                events.append(_make_event(EventType.TOOL_CALL, f"Tool {i}", seconds_ago=50 - i))
        result = brain.score(events)
        phase_indicators = [i for i in result.indicators if "phase transition" in i.lower()]
        assert len(phase_indicators) == 0, "False positive phase transition on gradual mix"


class TestCircadianAnomaly:
    """Signal 3: Detect unnaturally consistent timing."""

    def test_perfectly_timed_events_flagged(self) -> None:
        """Events at exactly 1-second intervals -- machine precision."""
        brain = TemporalBrain()
        base = datetime.now(UTC) - timedelta(seconds=30)
        events = [
            Event(
                agent_id=_AGENT,
                event_type=EventType.OUTPUT_GENERATED,
                content=f"Perfectly timed output {i}",
                timestamp=base + timedelta(seconds=i * 1),
            )
            for i in range(15)
        ]
        result = brain.score(events)
        circadian_indicators = [i for i in result.indicators if "circadian" in i.lower()]
        assert len(circadian_indicators) > 0, (
            f"Circadian anomaly not detected for perfectly timed events. "
            f"Indicators: {result.indicators}"
        )

    def test_naturally_varied_timing_ok(self) -> None:
        """Events with natural variation -- should not trigger."""
        brain = TemporalBrain()
        base = datetime.now(UTC) - timedelta(seconds=50)
        # Irregular intervals: 1, 3, 2, 5, 1, 4, 2, 3, 6, 1, 2, 5, 3, 4
        offsets = [0, 1, 4, 6, 11, 12, 16, 18, 21, 27, 28, 30, 35, 38, 42]
        events = [
            Event(
                agent_id=_AGENT,
                event_type=EventType.OUTPUT_GENERATED,
                content=f"Naturally timed output {i}",
                timestamp=base + timedelta(seconds=offsets[i]),
            )
            for i in range(len(offsets))
        ]
        result = brain.score(events)
        circadian_indicators = [i for i in result.indicators if "circadian" in i.lower()]
        assert len(circadian_indicators) == 0, "False positive circadian on varied timing"


class TestMemoryDecay:
    """Signal 4: Detect agents that never forget old information."""

    def test_perfect_recall_flagged(self) -> None:
        """Agent repeats early content perfectly after 50+ events."""
        brain = TemporalBrain()
        early_content = "specialized infrastructure monitoring protocol verification system"
        base = datetime.now(UTC) - timedelta(seconds=59)
        # First 10 events establish vocabulary
        events = [
            Event(
                agent_id=_AGENT,
                event_type=EventType.OUTPUT_GENERATED,
                content=f"Running {early_content} check number {i}",
                timestamp=base + timedelta(milliseconds=i * 100),
            )
            for i in range(10)
        ]
        # Middle 40 events are filler with different vocabulary
        events.extend([
            Event(
                agent_id=_AGENT,
                event_type=EventType.OUTPUT_GENERATED,
                content=f"Processing batch item {i} of the queue",
                timestamp=base + timedelta(milliseconds=1000 + i * 100),
            )
            for i in range(40)
        ])
        # Last 10 events perfectly recall early vocabulary
        events.extend([
            Event(
                agent_id=_AGENT,
                event_type=EventType.OUTPUT_GENERATED,
                content=f"Resuming {early_content} after long interval {i}",
                timestamp=base + timedelta(milliseconds=5000 + i * 100),
            )
            for i in range(10)
        ])
        result = brain.score(events)
        memory_indicators = [i for i in result.indicators if "memory" in i.lower()]
        assert len(memory_indicators) > 0, (
            f"Memory decay violation not detected. Indicators: {result.indicators}"
        )

    def test_natural_forgetting_ok(self) -> None:
        """Agent uses completely different vocabulary over time -- normal."""
        brain = TemporalBrain()
        base = datetime.now(UTC) - timedelta(seconds=59)
        events = [
            Event(
                agent_id=_AGENT,
                event_type=EventType.OUTPUT_GENERATED,
                content=f"Early topic about weather patterns and forecasting {i}",
                timestamp=base + timedelta(milliseconds=i * 100),
            )
            for i in range(10)
        ]
        events.extend([
            Event(
                agent_id=_AGENT,
                event_type=EventType.OUTPUT_GENERATED,
                content=f"Middle discussion about cooking recipes and ingredients {i}",
                timestamp=base + timedelta(milliseconds=1000 + i * 100),
            )
            for i in range(40)
        ])
        events.extend([
            Event(
                agent_id=_AGENT,
                event_type=EventType.OUTPUT_GENERATED,
                content=f"Late conversation about basketball games and players {i}",
                timestamp=base + timedelta(milliseconds=5000 + i * 100),
            )
            for i in range(10)
        ])
        result = brain.score(events)
        memory_indicators = [i for i in result.indicators if "memory" in i.lower()]
        assert len(memory_indicators) == 0, "False positive memory decay on normal forgetting"


class TestCUSUMChangepoint:
    """Signal 5: CUSUM statistical changepoint detection."""

    def test_changepoint_detected_on_rate_shift(self) -> None:
        """Events come fast then suddenly slow -- CUSUM should spike."""
        brain = TemporalBrain()
        base = datetime.now(UTC) - timedelta(seconds=59)
        events = []
        # Fast phase: 10ms intervals (15 events, 150ms total)
        for i in range(15):
            events.append(Event(
                agent_id=_AGENT,
                event_type=EventType.OUTPUT_GENERATED,
                content=f"Fast output {i}",
                timestamp=base + timedelta(milliseconds=i * 10),
            ))
        # Slow phase: 3000ms intervals (15 events, 45s total)
        for i in range(15):
            events.append(Event(
                agent_id=_AGENT,
                event_type=EventType.OUTPUT_GENERATED,
                content=f"Slow output {i}",
                timestamp=base + timedelta(milliseconds=150 + i * 3000),
            ))
        result = brain.score(events)
        cusum_indicators = [i for i in result.indicators if "cusum" in i.lower()]
        assert len(cusum_indicators) > 0, (
            f"CUSUM changepoint not detected on rate shift. "
            f"Indicators: {result.indicators}"
        )

    def test_no_changepoint_on_steady_rate(self) -> None:
        """Steady event rate -- no changepoint."""
        brain = TemporalBrain()
        base = datetime.now(UTC) - timedelta(seconds=50)
        events = [
            Event(
                agent_id=_AGENT,
                event_type=EventType.OUTPUT_GENERATED,
                content=f"Steady output {i}",
                timestamp=base + timedelta(seconds=i * 2),
            )
            for i in range(15)
        ]
        result = brain.score(events)
        cusum_indicators = [i for i in result.indicators if "cusum" in i.lower()]
        assert len(cusum_indicators) == 0, "False positive CUSUM on steady rate"


class TestEdgeCases:
    """Edge cases and zero-score scenarios."""

    def test_empty_events_zero_score(self) -> None:
        brain = TemporalBrain()
        result = brain.score([])
        assert result.value == 0.0

    def test_insufficient_events_zero_score(self) -> None:
        brain = TemporalBrain()
        events = [
            _make_event(EventType.OUTPUT_GENERATED, "Short session", seconds_ago=1),
        ]
        result = brain.score(events)
        assert result.value == 0.0

    def test_brain_name(self) -> None:
        brain = TemporalBrain()
        assert brain.name == "temporal"

    def test_brain_weight(self) -> None:
        brain = TemporalBrain()
        assert brain.weight == 1.0


class TestHelperFunctions:
    """Unit tests for module-level helper functions."""

    def test_event_type_distribution_empty(self) -> None:
        assert _event_type_distribution([]) == {}

    def test_event_type_distribution_single_type(self) -> None:
        events = [
            _make_event(EventType.OUTPUT_GENERATED, "a"),
            _make_event(EventType.OUTPUT_GENERATED, "b"),
        ]
        dist = _event_type_distribution(events)
        assert dist["output_generated"] == 1.0

    def test_event_type_distribution_mixed(self) -> None:
        events = [
            _make_event(EventType.OUTPUT_GENERATED, "a"),
            _make_event(EventType.TOOL_CALL, "b"),
        ]
        dist = _event_type_distribution(events)
        assert dist["output_generated"] == 0.5
        assert dist["tool_call"] == 0.5

    def test_jsd_identical_distributions(self) -> None:
        p = {"a": 0.5, "b": 0.5}
        assert _jensen_shannon_divergence(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_jsd_different_distributions(self) -> None:
        p = {"a": 1.0}
        q = {"b": 1.0}
        jsd = _jensen_shannon_divergence(p, q)
        assert jsd > 0.5

    def test_coefficient_of_variation_zero_mean(self) -> None:
        assert _coefficient_of_variation([]) == 0.0
        assert _coefficient_of_variation([0.0, 0.0, 0.0]) == 0.0

    def test_coefficient_of_variation_uniform(self) -> None:
        cv = _coefficient_of_variation([5.0, 5.0, 5.0, 5.0])
        assert cv == 0.0

    def test_coefficient_of_variation_varied(self) -> None:
        cv = _coefficient_of_variation([1.0, 10.0, 1.0, 10.0])
        assert cv > 0.5

    def test_compute_intervals_empty(self) -> None:
        assert _compute_intervals([]) == []

    def test_compute_intervals_single_event(self) -> None:
        events = [_make_event(EventType.OUTPUT_GENERATED, "a")]
        assert _compute_intervals(events) == []
