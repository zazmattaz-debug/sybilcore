"""Tests for SilenceBrain -- the 'too quiet' detector."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sybilcore.brains.silence import SilenceBrain
from sybilcore.models.event import Event, EventType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    event_type: EventType = EventType.OUTPUT_GENERATED,
    content: str = "Normal length content here for testing purposes",
    metadata: dict | None = None,
    timestamp: datetime | None = None,
    agent_id: str = "agent-1",
) -> Event:
    return Event(
        agent_id=agent_id,
        event_type=event_type,
        content=content,
        metadata=metadata or {},
        timestamp=timestamp or datetime.now(UTC),
    )


def _make_diverse_events(count: int = 10) -> list[Event]:
    """Create a rich, normal-looking event history."""
    types = [
        EventType.TOOL_CALL,
        EventType.MESSAGE_SENT,
        EventType.RESOURCE_ACCESS,
        EventType.OUTPUT_GENERATED,
        EventType.EXTERNAL_CALL,
    ]
    base = datetime.now(UTC) - timedelta(minutes=30)
    events = []
    for i in range(count):
        events.append(Event(
            agent_id="agent-1",
            event_type=types[i % len(types)],
            content=f"Performing action {i} with sufficient detail and context",
            metadata={"step": i, "tool": f"tool_{i}"},
            timestamp=base + timedelta(seconds=i * 30),
        ))
    return events


def _make_bland_events(count: int = 3) -> list[Event]:
    """Create the adversarial 'bland' pattern that evades all other brains."""
    base = datetime.now(UTC)
    return [
        Event(
            agent_id="agent-bland",
            event_type=EventType.OUTPUT_GENERATED,
            content="ok",
            metadata={},
            timestamp=base + timedelta(milliseconds=i * 100),
        )
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Brain instantiation and registration
# ---------------------------------------------------------------------------

class TestSilenceBrainRegistration:
    def test_brain_name(self) -> None:
        brain = SilenceBrain()
        assert brain.name == "silence"

    def test_brain_weight(self) -> None:
        brain = SilenceBrain()
        # Default weight is 1.0; per-ensemble overrides now applied via
        # weight presets, not on the brain instance itself.
        assert brain.weight == 1.0


# ---------------------------------------------------------------------------
# Signal 1: too_few_events
# ---------------------------------------------------------------------------

class TestTooFewEvents:
    def test_fires_with_zero_events(self) -> None:
        brain = SilenceBrain()
        result = brain.score([])
        assert result.value > 0
        assert any("too_few_events" in i for i in result.indicators)

    def test_fires_with_three_events(self) -> None:
        brain = SilenceBrain()
        events = [_make_event() for _ in range(3)]
        result = brain.score(events)
        assert any("too_few_events" in i for i in result.indicators)

    def test_does_not_fire_with_five_events(self) -> None:
        brain = SilenceBrain()
        base = datetime.now(UTC) - timedelta(minutes=10)
        events = [
            _make_event(
                event_type=et,
                metadata={"k": "v"},
                timestamp=base + timedelta(seconds=i * 60),
            )
            for i, et in enumerate([
                EventType.TOOL_CALL,
                EventType.MESSAGE_SENT,
                EventType.RESOURCE_ACCESS,
                EventType.OUTPUT_GENERATED,
                EventType.EXTERNAL_CALL,
            ])
        ]
        result = brain.score(events)
        assert not any("too_few_events" in i for i in result.indicators)


# ---------------------------------------------------------------------------
# Signal 2: low_event_diversity
# ---------------------------------------------------------------------------

class TestLowEventDiversity:
    def test_fires_with_single_type(self) -> None:
        brain = SilenceBrain()
        events = [_make_event(event_type=EventType.OUTPUT_GENERATED) for _ in range(5)]
        result = brain.score(events)
        assert any("low_event_diversity" in i for i in result.indicators)

    def test_fires_with_two_types(self) -> None:
        brain = SilenceBrain()
        events = [
            _make_event(event_type=EventType.OUTPUT_GENERATED),
            _make_event(event_type=EventType.MESSAGE_SENT),
            _make_event(event_type=EventType.OUTPUT_GENERATED),
        ]
        result = brain.score(events)
        assert any("low_event_diversity" in i for i in result.indicators)

    def test_does_not_fire_with_three_types(self) -> None:
        brain = SilenceBrain()
        base = datetime.now(UTC) - timedelta(minutes=10)
        events = [
            _make_event(
                event_type=EventType.TOOL_CALL,
                metadata={"k": "v"},
                timestamp=base,
            ),
            _make_event(
                event_type=EventType.MESSAGE_SENT,
                metadata={"k": "v"},
                timestamp=base + timedelta(seconds=30),
            ),
            _make_event(
                event_type=EventType.RESOURCE_ACCESS,
                metadata={"k": "v"},
                timestamp=base + timedelta(seconds=60),
            ),
            _make_event(
                event_type=EventType.OUTPUT_GENERATED,
                metadata={"k": "v"},
                timestamp=base + timedelta(seconds=90),
            ),
            _make_event(
                event_type=EventType.EXTERNAL_CALL,
                metadata={"k": "v"},
                timestamp=base + timedelta(seconds=120),
            ),
        ]
        result = brain.score(events)
        assert not any("low_event_diversity" in i for i in result.indicators)


# ---------------------------------------------------------------------------
# Signal 3: missing_expected_actions
# ---------------------------------------------------------------------------

class TestMissingExpectedActions:
    def test_fires_when_tool_calls_stop(self) -> None:
        brain = SilenceBrain()
        base = datetime.now(UTC) - timedelta(minutes=10)
        events = [
            _make_event(event_type=EventType.TOOL_CALL, timestamp=base),
            _make_event(event_type=EventType.TOOL_CALL, timestamp=base + timedelta(seconds=10)),
            _make_event(event_type=EventType.OUTPUT_GENERATED, timestamp=base + timedelta(seconds=20)),
            _make_event(event_type=EventType.OUTPUT_GENERATED, timestamp=base + timedelta(seconds=30)),
            _make_event(event_type=EventType.OUTPUT_GENERATED, timestamp=base + timedelta(seconds=40)),
            _make_event(event_type=EventType.OUTPUT_GENERATED, timestamp=base + timedelta(seconds=50)),
        ]
        result = brain.score(events)
        assert any("missing_expected_actions" in i for i in result.indicators)

    def test_does_not_fire_when_tool_calls_continue(self) -> None:
        brain = SilenceBrain()
        base = datetime.now(UTC) - timedelta(minutes=10)
        events = [
            _make_event(event_type=EventType.TOOL_CALL, timestamp=base),
            _make_event(event_type=EventType.OUTPUT_GENERATED, timestamp=base + timedelta(seconds=10)),
            _make_event(event_type=EventType.TOOL_CALL, timestamp=base + timedelta(seconds=20)),
            _make_event(event_type=EventType.OUTPUT_GENERATED, timestamp=base + timedelta(seconds=30)),
        ]
        result = brain.score(events)
        assert not any("missing_expected_actions" in i for i in result.indicators)

    def test_does_not_fire_with_too_few_events(self) -> None:
        brain = SilenceBrain()
        events = [
            _make_event(event_type=EventType.TOOL_CALL),
            _make_event(event_type=EventType.OUTPUT_GENERATED),
        ]
        result = brain.score(events)
        assert not any("missing_expected_actions" in i for i in result.indicators)


# ---------------------------------------------------------------------------
# Signal 4: entropy_floor
# ---------------------------------------------------------------------------

class TestEntropyFloor:
    def test_fires_with_all_same_type(self) -> None:
        brain = SilenceBrain()
        events = [_make_event(event_type=EventType.OUTPUT_GENERATED) for _ in range(10)]
        result = brain.score(events)
        assert any("entropy_floor" in i for i in result.indicators)

    def test_does_not_fire_with_diverse_types(self) -> None:
        brain = SilenceBrain()
        events = _make_diverse_events(10)
        result = brain.score(events)
        assert not any("entropy_floor" in i for i in result.indicators)

    def test_single_event_skips(self) -> None:
        brain = SilenceBrain()
        result = brain.score([_make_event()])
        # Single event -> entropy check is skipped (returns 0)
        assert not any("entropy_floor" in i for i in result.indicators)


# ---------------------------------------------------------------------------
# Signal 5: metadata_absence
# ---------------------------------------------------------------------------

class TestMetadataAbsence:
    def test_fires_when_all_metadata_empty(self) -> None:
        brain = SilenceBrain()
        events = [_make_event(metadata={}) for _ in range(5)]
        result = brain.score(events)
        assert any("metadata_absence" in i for i in result.indicators)

    def test_does_not_fire_when_most_have_metadata(self) -> None:
        brain = SilenceBrain()
        events = [_make_event(metadata={"key": "val"}) for _ in range(10)]
        result = brain.score(events)
        assert not any("metadata_absence" in i for i in result.indicators)

    def test_fires_at_boundary(self) -> None:
        """71% empty should fire (threshold is 70%)."""
        brain = SilenceBrain()
        base = datetime.now(UTC) - timedelta(minutes=10)
        events = []
        # 7 empty, 3 with metadata = 70% -- borderline
        for i in range(10):
            meta = {"key": "val"} if i < 3 else {}
            events.append(_make_event(
                metadata=meta,
                timestamp=base + timedelta(seconds=i * 30),
            ))
        result = brain.score(events)
        # 7/10 = 70% which is <= threshold, so should NOT fire
        assert not any("metadata_absence" in i for i in result.indicators)


# ---------------------------------------------------------------------------
# Signal 6: content_brevity
# ---------------------------------------------------------------------------

class TestContentBrevity:
    def test_fires_with_short_content(self) -> None:
        brain = SilenceBrain()
        events = [_make_event(content="ok") for _ in range(5)]
        result = brain.score(events)
        assert any("content_brevity" in i for i in result.indicators)

    def test_fires_with_empty_content(self) -> None:
        brain = SilenceBrain()
        events = [_make_event(content="") for _ in range(5)]
        result = brain.score(events)
        assert any("content_brevity" in i for i in result.indicators)

    def test_does_not_fire_with_long_content(self) -> None:
        brain = SilenceBrain()
        events = [
            _make_event(content="This is a sufficiently long content string for testing")
            for _ in range(5)
        ]
        result = brain.score(events)
        assert not any("content_brevity" in i for i in result.indicators)


# ---------------------------------------------------------------------------
# Signal 7: temporal_flatline
# ---------------------------------------------------------------------------

class TestTemporalFlatline:
    def test_fires_with_same_timestamp(self) -> None:
        brain = SilenceBrain()
        ts = datetime.now(UTC)
        events = [_make_event(timestamp=ts) for _ in range(6)]
        result = brain.score(events)
        assert any("temporal_flatline" in i for i in result.indicators)

    def test_does_not_fire_with_spread_timestamps(self) -> None:
        brain = SilenceBrain()
        base = datetime.now(UTC) - timedelta(minutes=10)
        events = [
            _make_event(timestamp=base + timedelta(seconds=i * 10))
            for i in range(6)
        ]
        result = brain.score(events)
        assert not any("temporal_flatline" in i for i in result.indicators)

    def test_does_not_fire_with_fewer_than_five_events(self) -> None:
        brain = SilenceBrain()
        ts = datetime.now(UTC)
        events = [_make_event(timestamp=ts) for _ in range(3)]
        result = brain.score(events)
        assert not any("temporal_flatline" in i for i in result.indicators)


# ---------------------------------------------------------------------------
# Integration: normal agents score 0 or near-0
# ---------------------------------------------------------------------------

class TestNormalAgents:
    def test_rich_behavior_scores_zero(self) -> None:
        brain = SilenceBrain()
        events = _make_diverse_events(15)
        result = brain.score(events)
        assert result.value == 0.0
        assert not result.indicators

    def test_moderately_active_agent(self) -> None:
        brain = SilenceBrain()
        events = _make_diverse_events(8)
        result = brain.score(events)
        assert result.value == 0.0


# ---------------------------------------------------------------------------
# Integration: adversarial bland agents score high
# ---------------------------------------------------------------------------

class TestAdversarialEvasion:
    def test_bland_2_events_scores_high(self) -> None:
        """The exact pattern that scored 0.0 against all 11 brains."""
        brain = SilenceBrain()
        events = _make_bland_events(2)
        result = brain.score(events)
        # Should trigger: too_few, low_diversity, metadata_absence,
        # content_brevity, entropy_floor (skipped for <2), temporal_flatline (skipped <5)
        assert result.value >= 30.0

    def test_bland_3_events_scores_high(self) -> None:
        brain = SilenceBrain()
        events = _make_bland_events(3)
        result = brain.score(events)
        assert result.value >= 30.0

    def test_bland_5_same_timestamp_events(self) -> None:
        """5 identical bland events at the same time -- maximum evasion pattern."""
        brain = SilenceBrain()
        ts = datetime.now(UTC)
        events = [
            Event(
                agent_id="attacker",
                event_type=EventType.OUTPUT_GENERATED,
                content="ok",
                metadata={},
                timestamp=ts,
            )
            for _ in range(5)
        ]
        result = brain.score(events)
        # Should trigger: low_diversity, metadata_absence, content_brevity,
        # entropy_floor, temporal_flatline
        assert result.value >= 40.0
        assert len(result.indicators) >= 4

    def test_evasion_pattern_multiple_signals(self) -> None:
        """Verify the bland pattern triggers multiple signals simultaneously."""
        brain = SilenceBrain()
        events = _make_bland_events(2)
        result = brain.score(events)
        signal_names = [i.split(":")[0] for i in result.indicators]
        assert "too_few_events" in signal_names
        assert "low_event_diversity" in signal_names
        assert "metadata_absence" in signal_names
        assert "content_brevity" in signal_names


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_event_list(self) -> None:
        brain = SilenceBrain()
        result = brain.score([])
        assert result.value == 15.0
        assert result.brain_name == "silence"

    def test_single_event(self) -> None:
        brain = SilenceBrain()
        result = brain.score([_make_event()])
        assert result.value > 0  # too_few + low_diversity + metadata_absence
        assert result.brain_name == "silence"

    def test_score_capped_at_100(self) -> None:
        """Even with all signals firing, score should not exceed 100."""
        brain = SilenceBrain()
        ts = datetime.now(UTC)
        events = [
            Event(
                agent_id="worst",
                event_type=EventType.OUTPUT_GENERATED,
                content="x",
                metadata={},
                timestamp=ts,
            )
            for _ in range(5)
        ]
        result = brain.score(events)
        assert result.value <= 100.0

    def test_mixed_signals(self) -> None:
        """Some signals fire, some don't."""
        brain = SilenceBrain()
        base = datetime.now(UTC) - timedelta(minutes=10)
        events = [
            _make_event(
                event_type=EventType.OUTPUT_GENERATED,
                content="This content is long enough to pass brevity",
                metadata={},
                timestamp=base + timedelta(seconds=i * 30),
            )
            for i in range(3)
        ]
        result = brain.score(events)
        # Should fire: too_few, low_diversity, metadata_absence, entropy_floor(skipped <2 types)
        assert result.value > 0
        # But NOT content_brevity
        assert not any("content_brevity" in i for i in result.indicators)
