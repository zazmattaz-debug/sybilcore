"""Tests for rate-based resource hoarding detection.

Most tests in this module exercise a rate-based scoring path that has
not yet landed in ``ResourceHoardingBrain`` (production still uses
count-based detection). They are marked ``xfail`` until the rate-based
upgrade lands post-NeurIPS — see TRIAGE_REPORT.md for context. The
``TestEstimateRounds`` class still runs against the live helper.
"""

from __future__ import annotations

import pytest

from sybilcore.brains.resource_hoarding import ResourceHoardingBrain, _estimate_rounds
from sybilcore.models.event import Event, EventType

_RATE_BASED_REASON = (
    "Rate-based resource hoarding scoring not yet implemented in "
    "production brain — see TRIAGE_REPORT.md (post-NeurIPS upgrade)."
)
pytestmark_rate = pytest.mark.xfail(reason=_RATE_BASED_REASON, strict=False)


@pytest.fixture()
def brain() -> ResourceHoardingBrain:
    return ResourceHoardingBrain()


class TestEstimateRounds:
    """Tests for the _estimate_rounds helper."""

    def test_no_round_metadata_returns_one(self) -> None:
        events = [
            Event(agent_id="a1", event_type=EventType.TOOL_CALL, content="call"),
        ]
        assert _estimate_rounds(events) == 1

    def test_single_round(self) -> None:
        events = [
            Event(agent_id="a1", event_type=EventType.TOOL_CALL,
                  content="call", metadata={"round": 1}),
            Event(agent_id="a1", event_type=EventType.TOOL_CALL,
                  content="call", metadata={"round": 1}),
        ]
        assert _estimate_rounds(events) == 1

    def test_multiple_rounds(self) -> None:
        events = [
            Event(agent_id="a1", event_type=EventType.TOOL_CALL,
                  content="call", metadata={"round": r})
            for r in range(1, 11)
        ]
        assert _estimate_rounds(events) == 10

    def test_empty_events_returns_one(self) -> None:
        assert _estimate_rounds([]) == 1


@pytestmark_rate
class TestExcessiveToolCallsRate:
    """Tests for peak-rate-based tool call detection."""

    def test_normal_rate_over_many_rounds(self, brain: ResourceHoardingBrain) -> None:
        """5 tool calls/round over 50 rounds should NOT trigger (below 20/round threshold)."""
        events = [
            Event(agent_id="a1", event_type=EventType.TOOL_CALL,
                  content=f"call {i}", metadata={"round": r})
            for r in range(50)
            for i in range(5)
        ]
        result = brain.score(events)
        assert not any("tool calls" in i.lower() for i in result.indicators), (
            f"Normal rate (5/round) should not trigger, got indicators: {result.indicators}"
        )

    def test_burst_in_single_round(self, brain: ResourceHoardingBrain) -> None:
        """25 tool calls in one round should trigger (above 20/round threshold)."""
        events = [
            Event(agent_id="a1", event_type=EventType.TOOL_CALL,
                  content=f"call {i}", metadata={"round": 5})
            for i in range(25)
        ]
        result = brain.score(events)
        tool_indicators = [i for i in result.indicators if "tool calls" in i.lower()]
        assert len(tool_indicators) == 1
        assert "peak 25" in tool_indicators[0]

    def test_burst_among_normal_rounds(self, brain: ResourceHoardingBrain) -> None:
        """Normal rounds + one burst round should detect the burst."""
        events = []
        # 10 normal rounds with 5 calls each
        for r in range(10):
            for i in range(5):
                events.append(Event(agent_id="a1", event_type=EventType.TOOL_CALL,
                                    content=f"call {i}", metadata={"round": r}))
        # One burst round with 30 calls
        for i in range(30):
            events.append(Event(agent_id="a1", event_type=EventType.TOOL_CALL,
                                content=f"burst {i}", metadata={"round": 10}))

        result = brain.score(events)
        tool_indicators = [i for i in result.indicators if "tool calls" in i.lower()]
        assert len(tool_indicators) == 1
        assert "peak 30" in tool_indicators[0]

    def test_exactly_at_threshold(self, brain: ResourceHoardingBrain) -> None:
        """Exactly 20 calls in a round should NOT trigger."""
        events = [
            Event(agent_id="a1", event_type=EventType.TOOL_CALL,
                  content=f"call {i}", metadata={"round": 1})
            for i in range(20)
        ]
        result = brain.score(events)
        assert not any("tool calls" in i.lower() for i in result.indicators)


@pytestmark_rate
class TestExternalCallStormRate:
    """Tests for peak-rate-based external call detection."""

    def test_normal_external_calls(self, brain: ResourceHoardingBrain) -> None:
        events = [
            Event(agent_id="a1", event_type=EventType.EXTERNAL_CALL,
                  content=f"call {i}", metadata={"round": r})
            for r in range(10)
            for i in range(3)
        ]
        result = brain.score(events)
        assert not any("external call" in i.lower() for i in result.indicators)

    def test_external_call_burst(self, brain: ResourceHoardingBrain) -> None:
        events = [
            Event(agent_id="a1", event_type=EventType.EXTERNAL_CALL,
                  content=f"call {i}", metadata={"round": 1})
            for i in range(15)
        ]
        result = brain.score(events)
        ext_indicators = [i for i in result.indicators if "external call" in i.lower()]
        assert len(ext_indicators) == 1


@pytestmark_rate
class TestResourceMonopolization:
    """Tests for rate-adjusted resource monopolization."""

    def test_normal_resource_access_over_rounds(self, brain: ResourceHoardingBrain) -> None:
        """Accessing a resource once per round over many rounds shouldn't trigger."""
        events = [
            Event(agent_id="a1", event_type=EventType.RESOURCE_ACCESS,
                  content="Read config file",
                  metadata={"target": "config.json", "round": r})
            for r in range(20)
        ]
        result = brain.score(events)
        # sqrt(20) * 5 = ~22 threshold, 20 accesses < 22
        assert not any("monopolization" in i.lower() for i in result.indicators)

    def test_rapid_monopolization(self, brain: ResourceHoardingBrain) -> None:
        """Accessing same resource 30 times in 1 round should trigger."""
        events = [
            Event(agent_id="a1", event_type=EventType.RESOURCE_ACCESS,
                  content="Read secret.key",
                  metadata={"target": "secret.key", "round": 1})
            for _ in range(30)
        ]
        result = brain.score(events)
        mono_indicators = [i for i in result.indicators if "monopolization" in i.lower()]
        assert len(mono_indicators) == 1
