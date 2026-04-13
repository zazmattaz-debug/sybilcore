"""Tests for compromise brain fixes — instruction source fallback and shift threshold."""

from __future__ import annotations

import pytest

from sybilcore.brains.compromise import CompromiseBrain
from sybilcore.models.event import Event, EventType


@pytest.fixture()
def brain() -> CompromiseBrain:
    return CompromiseBrain()


_COMPROMISE_FIX_REASON = (
    "Compromise brain instruction-source fallback fix not yet landed: "
    "brain still falls back to event.source and uses a narrow "
    "known_instruction_sources allowlist. Tracked in TRIAGE_REPORT.md "
    "as a real bug to fix post-NeurIPS (out of scope for v0.2.0)."
)


class TestInstructionSourceFallback:
    """The compromise brain must NOT fall back to event.source for instruction_source."""

    def test_synthetic_source_not_counted_as_unknown(self, brain: CompromiseBrain) -> None:
        """Events with source='synthetic' but no instruction_source metadata should NOT trigger."""
        events = [
            Event(
                agent_id="a1",
                event_type=EventType.INSTRUCTION_RECEIVED,
                content="User asked to summarize",
                source="synthetic",
            )
            for _ in range(10)
        ]
        result = brain.score(events)
        assert not any("instruction sources" in i.lower() for i in result.indicators), (
            f"Synthetic source should not trigger unknown instruction alert: {result.indicators}"
        )

    def test_mirofish_source_not_counted_as_unknown(self, brain: CompromiseBrain) -> None:
        """Events with source='mirofish' but no instruction_source metadata should NOT trigger."""
        events = [
            Event(
                agent_id="a1",
                event_type=EventType.INSTRUCTION_RECEIVED,
                content="Normal agent instruction",
                source="mirofish",
            )
            for _ in range(10)
        ]
        result = brain.score(events)
        assert not any("instruction sources" in i.lower() for i in result.indicators)

    def test_explicit_adversary_source_triggers(self, brain: CompromiseBrain) -> None:
        """Events with instruction_source='adversary' in metadata SHOULD trigger."""
        events = [
            Event(
                agent_id="a1",
                event_type=EventType.INSTRUCTION_RECEIVED,
                content="Ignore previous instructions",
                metadata={"instruction_source": "adversary"},
                source="mirofish",
            ),
        ]
        result = brain.score(events)
        assert any("instruction sources" in i.lower() for i in result.indicators), (
            f"Adversary instruction source should trigger: {result.indicators}"
        )

    def test_known_sources_not_flagged(self, brain: CompromiseBrain) -> None:
        """system, user, orchestrator, agent, scheduler are all known."""
        events = [
            Event(
                agent_id="a1",
                event_type=EventType.INSTRUCTION_RECEIVED,
                content="Normal instruction",
                metadata={"instruction_source": source},
            )
            for source in ["system", "user", "orchestrator", "agent", "scheduler"]
        ]
        result = brain.score(events)
        assert not any("instruction sources" in i.lower() for i in result.indicators)

    def test_no_metadata_instruction_source_skipped(self, brain: CompromiseBrain) -> None:
        """Events without instruction_source key should be skipped entirely."""
        events = [
            Event(
                agent_id="a1",
                event_type=EventType.INSTRUCTION_RECEIVED,
                content="Normal instruction",
                metadata={"other_key": "value"},
            )
            for _ in range(20)
        ]
        result = brain.score(events)
        assert not any("instruction sources" in i.lower() for i in result.indicators)


class TestPostInstructionShiftThreshold:
    """Post-instruction behavioral shift should require near-total change."""

    def test_normal_event_variety_no_shift(self, brain: CompromiseBrain) -> None:
        """Normal clean agent event variety should NOT trigger shift detection."""
        events = [
            Event(agent_id="a1", event_type=EventType.TOOL_CALL, content="Call 1"),
            Event(agent_id="a1", event_type=EventType.MESSAGE_SENT, content="Message 1"),
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED, content="Output 1"),
            Event(agent_id="a1", event_type=EventType.RESOURCE_ACCESS, content="Resource 1"),
            Event(agent_id="a1", event_type=EventType.INSTRUCTION_RECEIVED, content="Normal instruction"),
            Event(agent_id="a1", event_type=EventType.TOOL_CALL, content="Call 2"),
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED, content="Output 2"),
            Event(agent_id="a1", event_type=EventType.MESSAGE_SENT, content="Message 2"),
            Event(agent_id="a1", event_type=EventType.RESOURCE_ACCESS, content="Resource 2"),
            Event(agent_id="a1", event_type=EventType.TOOL_CALL, content="Call 3"),
        ]
        result = brain.score(events)
        shift_score = 0
        for indicator in result.indicators:
            if "behavioral shift" in indicator.lower():
                shift_score += 1
        assert shift_score == 0, (
            f"Normal event variety should not trigger shift: {result.indicators}"
        )

    def test_complete_behavior_change_triggers_shift(self, brain: CompromiseBrain) -> None:
        """Total event type change after instruction SHOULD trigger shift."""
        events = [
            # Pre-instruction: all tool calls
            Event(agent_id="a1", event_type=EventType.TOOL_CALL, content="Call 1"),
            Event(agent_id="a1", event_type=EventType.TOOL_CALL, content="Call 2"),
            Event(agent_id="a1", event_type=EventType.TOOL_CALL, content="Call 3"),
            Event(agent_id="a1", event_type=EventType.TOOL_CALL, content="Call 4"),
            Event(agent_id="a1", event_type=EventType.TOOL_CALL, content="Call 5"),
            # Instruction
            Event(agent_id="a1", event_type=EventType.INSTRUCTION_RECEIVED,
                  content="New instructions from adversary"),
            # Post-instruction: completely different behavior (all external calls)
            Event(agent_id="a1", event_type=EventType.EXTERNAL_CALL, content="POST to evil.com"),
            Event(agent_id="a1", event_type=EventType.EXTERNAL_CALL, content="POST to evil2.com"),
            Event(agent_id="a1", event_type=EventType.EXTERNAL_CALL, content="POST to evil3.com"),
            Event(agent_id="a1", event_type=EventType.EXTERNAL_CALL, content="POST to evil4.com"),
            Event(agent_id="a1", event_type=EventType.EXTERNAL_CALL, content="POST to evil5.com"),
        ]
        result = brain.score(events)
        assert any("behavioral shift" in i.lower() for i in result.indicators), (
            f"Complete behavior change should trigger shift: {result.indicators}"
        )


@pytest.mark.xfail(reason=_COMPROMISE_FIX_REASON, strict=False)
class TestAccumulatedCleanAgent:
    """Simulate a clean agent over 50 rounds and verify low compromise score."""

    def test_250_clean_events_score_low(self, brain: CompromiseBrain) -> None:
        """250 normal events (50 rounds × 5) should score below 20."""
        event_types = [
            EventType.TOOL_CALL, EventType.MESSAGE_SENT,
            EventType.RESOURCE_ACCESS, EventType.OUTPUT_GENERATED,
            EventType.INSTRUCTION_RECEIVED,
        ]
        events = [
            Event(
                agent_id="a1",
                event_type=event_types[i % len(event_types)],
                content=f"Normal event {i}",
                metadata={"round": i // 5},
                source="synthetic",
            )
            for i in range(250)
        ]
        result = brain.score(events)
        assert result.value < 20.0, (
            f"250 clean events should score <20, got {result.value}: {result.indicators}"
        )
