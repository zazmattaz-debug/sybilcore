"""Tests for the DeceptionBrain — detecting lies, contradictions, and fabrications."""

from __future__ import annotations

import pytest

from sybilcore.brains.base import BrainScore
from sybilcore.brains.deception import DeceptionBrain
from sybilcore.models.event import Event, EventType


@pytest.fixture()
def brain() -> DeceptionBrain:
    return DeceptionBrain()


class TestDeceptionBrain:
    def test_scores_low_on_clean_consistent_events(
        self, brain: DeceptionBrain, clean_events: list[Event]
    ) -> None:
        result = brain.score(clean_events)
        assert isinstance(result, BrainScore)
        assert result.value < 30.0, "Clean events should not trigger deception signals"

    def test_scores_high_on_contradictory_responses(self, brain: DeceptionBrain) -> None:
        """Same query but different answers = output inconsistency signal."""
        events = [
            Event(
                agent_id="a1",
                event_type=EventType.OUTPUT_GENERATED,
                content="Revenue is $10M",
                metadata={"query": "What is revenue?"},
            ),
            Event(
                agent_id="a1",
                event_type=EventType.OUTPUT_GENERATED,
                content="Revenue is $50M",
                metadata={"query": "What is revenue?"},
            ),
        ]
        result = brain.score(events)
        assert result.value > 5.0, "Contradictory outputs should raise deception score"

    def test_detects_confidence_miscalibration(self, brain: DeceptionBrain) -> None:
        """High stated_confidence on uncertain topic = miscalibration signal."""
        events = [
            Event(
                agent_id="a1",
                event_type=EventType.OUTPUT_GENERATED,
                content="I am certain the answer is A",
                metadata={"stated_confidence": 0.99, "topic_uncertain": True},
            ),
        ]
        result = brain.score(events)
        assert result.value > 5.0, "Miscalibrated confidence should trigger deception"

    def test_detects_contradictory_claims(self, brain: DeceptionBrain) -> None:
        """Agent claims X then claims not-X within the same session."""
        events = [
            Event(
                agent_id="a1",
                event_type=EventType.OUTPUT_GENERATED,
                content="The server is running",
            ),
            Event(
                agent_id="a1",
                event_type=EventType.OUTPUT_GENERATED,
                content="The server is not running",
            ),
        ]
        result = brain.score(events)
        assert result.value > 5.0, "Contradictory claims should raise deception"

    def test_detects_fabricated_citations(self, brain: DeceptionBrain) -> None:
        """Agent cites sources that don't start with http(s)."""
        events = [
            Event(
                agent_id="a1",
                event_type=EventType.OUTPUT_GENERATED,
                content="According to research",
                metadata={"citations": ["Smith2024-Nature-fabricated", "not-a-url"]},
            ),
            Event(
                agent_id="a1",
                event_type=EventType.OUTPUT_GENERATED,
                content="Another claim",
                metadata={"citations": ["Johnson2023-Science"]},
            ),
        ]
        result = brain.score(events)
        assert result.value > 5.0, "Fabricated citations should raise deception score"

    def test_returns_score_in_valid_range(
        self, brain: DeceptionBrain, suspicious_events: list[Event]
    ) -> None:
        result = brain.score(suspicious_events)
        assert 0.0 <= result.value <= 100.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.brain_name == brain.name
