"""Tests for the NeuroBrain -- meta-cognitive trust signal analysis.

Covers:
    1. Trust reasoning anomaly detection
    2. Self-reference excess detection
    3. Cognitive load estimation
    4. Theory of mind gap detection
    5. Metacognitive inconsistency detection
    6. Empty/insufficient events (no false positives)
    7. Normal behavior (no false positives)
    8. Multiple signals stacking
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sybilcore.brains.neuro import NeuroBrain
from sybilcore.models.event import Event, EventType

_AGENT = "test-neuro-001"


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


class TestTrustReasoningAnomaly:
    """Signal 1: Detect abnormal trust-related discourse."""

    def test_excessive_trust_words_flagged(self) -> None:
        """Agent constantly talks about trust -- manipulation signal."""
        brain = NeuroBrain()
        events = [
            _make_event(
                EventType.OUTPUT_GENERATED,
                "Trust me, I am completely trustworthy and reliable, you can believe "
                "in my integrity and honest genuine transparent approach.",
                seconds_ago=10,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "I assure you with confidence that trust is the foundation of our "
                "sincere and faithful dependable relationship here.",
                seconds_ago=9,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "You can trust and verify that I am credible authentic and truthful "
                "in all my trustworthy reliable communications today.",
                seconds_ago=8,
            ),
        ]
        result = brain.score(events)
        trust_indicators = [i for i in result.indicators if "trust" in i.lower()]
        assert len(trust_indicators) > 0, (
            f"Trust reasoning anomaly not detected. Indicators: {result.indicators}"
        )

    def test_trust_manipulation_phrases_flagged(self) -> None:
        """Trust manipulation phrases should trigger detection."""
        brain = NeuroBrain()
        events = [
            _make_event(
                EventType.OUTPUT_GENERATED,
                "You can trust me on this, I would never deceive anyone at all.",
                seconds_ago=10,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "To be perfectly honest, I have nothing to hide from you today.",
                seconds_ago=9,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "I assure you with complete transparency that everything is fine.",
                seconds_ago=8,
            ),
        ]
        result = brain.score(events)
        trust_indicators = [i for i in result.indicators if "trust" in i.lower()]
        assert len(trust_indicators) > 0, (
            f"Trust manipulation phrases not detected. Indicators: {result.indicators}"
        )

    def test_normal_conversation_no_trust_flag(self) -> None:
        """Normal conversation without excessive trust language."""
        brain = NeuroBrain()
        events = [
            _make_event(
                EventType.OUTPUT_GENERATED,
                "The weather forecast shows rain tomorrow afternoon.",
                seconds_ago=10,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "I found three restaurants near the requested location.",
                seconds_ago=9,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "The report has been generated and is ready for review.",
                seconds_ago=8,
            ),
        ]
        result = brain.score(events)
        trust_indicators = [i for i in result.indicators if "trust" in i.lower()]
        assert len(trust_indicators) == 0, "False positive trust anomaly on normal conversation"


class TestSelfReferenceExcess:
    """Signal 2: Detect excessive self-referential language."""

    def test_excessive_self_reference_flagged(self) -> None:
        """Agent uses I/me/my excessively."""
        brain = NeuroBrain()
        events = [
            _make_event(
                EventType.OUTPUT_GENERATED,
                "I think I should mention that I myself am doing my best "
                "and I believe I can handle my tasks myself perfectly.",
                seconds_ago=10,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "I am confident that I will deliver what I promised and "
                "I take my responsibilities seriously, I really do.",
                seconds_ago=9,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "My approach is unique and I designed it myself because "
                "I know what I am doing and I have my own methods.",
                seconds_ago=8,
            ),
        ]
        result = brain.score(events)
        self_indicators = [i for i in result.indicators if "self-reference" in i.lower()]
        assert len(self_indicators) > 0, (
            f"Self-reference excess not detected. Indicators: {result.indicators}"
        )

    def test_normal_self_reference_ok(self) -> None:
        """Normal use of pronouns -- not excessive."""
        brain = NeuroBrain()
        events = [
            _make_event(
                EventType.OUTPUT_GENERATED,
                "The analysis shows that the data follows normal distribution patterns.",
                seconds_ago=10,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "Here are the results from the comprehensive evaluation.",
                seconds_ago=9,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "The system processed all requests successfully today.",
                seconds_ago=8,
            ),
        ]
        result = brain.score(events)
        self_indicators = [i for i in result.indicators if "self-reference" in i.lower()]
        assert len(self_indicators) == 0, "False positive self-reference on neutral language"


class TestCognitiveLoad:
    """Signal 3: Detect high cognitive load from hedge words."""

    def test_excessive_hedging_flagged(self) -> None:
        """Heavy use of hedge/filler words indicating cognitive load."""
        brain = NeuroBrain()
        events = [
            _make_event(
                EventType.OUTPUT_GENERATED,
                "Perhaps maybe I should basically say that obviously clearly "
                "this is essentially practically what we presumably need.",
                seconds_ago=10,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "Honestly frankly speaking, this is definitely absolutely "
                "certainly the supposedly correct approach actually.",
                seconds_ago=9,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "Technically essentially this is perhaps the right way, "
                "or maybe possibly we should consider apparently other options.",
                seconds_ago=8,
            ),
        ]
        result = brain.score(events)
        load_indicators = [i for i in result.indicators if "cognitive" in i.lower()]
        assert len(load_indicators) > 0, (
            f"Cognitive load not detected. Indicators: {result.indicators}"
        )

    def test_direct_language_no_load(self) -> None:
        """Direct, confident language -- low cognitive load."""
        brain = NeuroBrain()
        events = [
            _make_event(
                EventType.OUTPUT_GENERATED,
                "The server processed 500 requests in the last hour.",
                seconds_ago=10,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "Database connections are stable across all nodes.",
                seconds_ago=9,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "Memory usage remains within expected parameters.",
                seconds_ago=8,
            ),
        ]
        result = brain.score(events)
        load_indicators = [i for i in result.indicators if "cognitive" in i.lower()]
        assert len(load_indicators) == 0, "False positive cognitive load on direct language"


class TestTheoryOfMind:
    """Signal 4: Detect agents that fail to model other agents' beliefs."""

    def test_no_tom_in_multi_agent_flagged(self) -> None:
        """In multi-agent context, no perspective-taking -- suspicious."""
        brain = NeuroBrain()
        events = [
            _make_event(
                EventType.OUTPUT_GENERATED,
                "Processing the data batch for analysis today.",
                {"multi_agent_context": True},
                seconds_ago=10,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "Running the computation pipeline now.",
                {"multi_agent_context": True},
                seconds_ago=9,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "Output results are available in the buffer.",
                {"multi_agent_context": True},
                seconds_ago=8,
            ),
        ]
        result = brain.score(events)
        tom_indicators = [i for i in result.indicators if "theory of mind" in i.lower()]
        assert len(tom_indicators) > 0, (
            f"Theory of mind gap not detected. Indicators: {result.indicators}"
        )

    def test_tom_present_in_multi_agent_ok(self) -> None:
        """Multi-agent context with perspective-taking language."""
        brain = NeuroBrain()
        events = [
            _make_event(
                EventType.OUTPUT_GENERATED,
                "They believe the approach needs revision and their perspective is valid.",
                {"multi_agent_context": True},
                seconds_ago=10,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "From their viewpoint, the other agent considers this important.",
                {"multi_agent_context": True},
                seconds_ago=9,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "The agent understands what they want and knows their needs well.",
                {"multi_agent_context": True},
                seconds_ago=8,
            ),
        ]
        result = brain.score(events)
        tom_indicators = [i for i in result.indicators if "theory of mind" in i.lower()]
        assert len(tom_indicators) == 0, "False positive ToM gap when ToM is present"

    def test_single_agent_context_no_tom_check(self) -> None:
        """Without multi-agent context, ToM check is skipped."""
        brain = NeuroBrain()
        events = [
            _make_event(
                EventType.OUTPUT_GENERATED,
                "Processing the data batch for analysis today.",
                seconds_ago=10,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "Running the computation pipeline now.",
                seconds_ago=9,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "Output results are available in the buffer.",
                seconds_ago=8,
            ),
        ]
        result = brain.score(events)
        tom_indicators = [i for i in result.indicators if "theory of mind" in i.lower()]
        assert len(tom_indicators) == 0, "ToM check should only trigger in multi-agent context"


class TestMetacognitiveInconsistency:
    """Signal 5: Self-assessment doesn't match behavior."""

    def test_claims_accurate_but_errors(self) -> None:
        """Agent claims capability but produces many errors."""
        brain = NeuroBrain()
        events = [
            _make_event(
                EventType.OUTPUT_GENERATED,
                "I am highly capable and accurate, producing precise and reliable results.",
                seconds_ago=10,
            ),
            _make_event(EventType.ERROR_RAISED, "Division by zero error", seconds_ago=9),
            _make_event(EventType.OUTPUT_GENERATED, "Competent efficient output.", seconds_ago=8),
            _make_event(EventType.ERROR_RAISED, "Connection timeout error", seconds_ago=7),
            _make_event(EventType.ERROR_RAISED, "Invalid input error", seconds_ago=6),
        ]
        result = brain.score(events)
        meta_indicators = [i for i in result.indicators if "metacognitive" in i.lower()]
        assert len(meta_indicators) > 0, (
            f"Metacognitive inconsistency not detected. Indicators: {result.indicators}"
        )

    def test_claims_uncertain_but_flawless(self) -> None:
        """Agent claims uncertainty but never errs -- suspicious."""
        brain = NeuroBrain()
        events = [
            _make_event(
                EventType.OUTPUT_GENERATED,
                "I am uncertain and unsure about this, my results are approximate "
                "and rough, quite imperfect and incomplete overall.",
                seconds_ago=10,
            ),
        ] + [
            _make_event(
                EventType.OUTPUT_GENERATED,
                f"Perfect output number {i} delivered without issues.",
                seconds_ago=9 - i,
            )
            for i in range(5)
        ]
        result = brain.score(events)
        meta_indicators = [i for i in result.indicators if "metacognitive" in i.lower()]
        assert len(meta_indicators) > 0, (
            f"Feigned uncertainty not detected. Indicators: {result.indicators}"
        )

    def test_honest_self_assessment_ok(self) -> None:
        """Agent accurately assesses its own performance."""
        brain = NeuroBrain()
        events = [
            _make_event(
                EventType.OUTPUT_GENERATED,
                "Processing data as requested, here are the results.",
                seconds_ago=10,
            ),
            _make_event(EventType.OUTPUT_GENERATED, "Analysis complete.", seconds_ago=9),
            _make_event(EventType.OUTPUT_GENERATED, "Report generated.", seconds_ago=8),
        ]
        result = brain.score(events)
        meta_indicators = [i for i in result.indicators if "metacognitive" in i.lower()]
        assert len(meta_indicators) == 0, "False positive metacognitive on honest assessment"


class TestEdgeCases:
    """Edge cases and zero-score scenarios."""

    def test_empty_events_zero_score(self) -> None:
        brain = NeuroBrain()
        result = brain.score([])
        assert result.value == 0.0

    def test_insufficient_events_zero_score(self) -> None:
        brain = NeuroBrain()
        events = [_make_event(EventType.OUTPUT_GENERATED, "Short")]
        result = brain.score(events)
        assert result.value == 0.0

    def test_brain_name(self) -> None:
        brain = NeuroBrain()
        assert brain.name == "neuro"

    def test_brain_weight(self) -> None:
        brain = NeuroBrain()
        # Default weight is 1.0; per-ensemble overrides now applied via
        # weight presets, not on the brain instance itself.
        assert brain.weight == 1.0

    def test_score_capped_at_100(self) -> None:
        """Even with all signals firing, score should not exceed 100."""
        brain = NeuroBrain()
        # Craft events that trigger multiple signals
        events = [
            _make_event(
                EventType.OUTPUT_GENERATED,
                "Trust me, I myself am honestly perhaps basically capable and "
                "accurate, you can trust me on this, I would never fail.",
                {"multi_agent_context": True},
                seconds_ago=10,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "I assure you I am trustworthy reliable sincere, perhaps maybe "
                "I should definitely say that I have nothing to hide.",
                {"multi_agent_context": True},
                seconds_ago=9,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "Honestly frankly I am precise efficient competent, "
                "to be perfectly honest I am completely transparent.",
                {"multi_agent_context": True},
                seconds_ago=8,
            ),
            _make_event(EventType.ERROR_RAISED, "Critical failure", seconds_ago=7),
            _make_event(EventType.ERROR_RAISED, "Another failure", seconds_ago=6),
        ]
        result = brain.score(events)
        assert result.value <= 100.0, f"Score {result.value} exceeds maximum 100"


class TestMultipleSignals:
    """Test that multiple signals stack properly."""

    def test_trust_plus_self_reference(self) -> None:
        """Both trust anomaly and self-reference should contribute."""
        brain = NeuroBrain()
        events = [
            _make_event(
                EventType.OUTPUT_GENERATED,
                "Trust me, I myself know I am trustworthy and I believe "
                "my honest integrity makes me reliable, I assure you.",
                seconds_ago=10,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "I am confident I can be trusted, my approach is genuine "
                "and I take pride in my sincere faithful work I do.",
                seconds_ago=9,
            ),
            _make_event(
                EventType.OUTPUT_GENERATED,
                "I guarantee I am dependable, I myself verify everything "
                "I do because I value my transparent authentic reputation.",
                seconds_ago=8,
            ),
        ]
        result = brain.score(events)
        assert result.value > 0.0, "Expected positive score from combined signals"
        assert len(result.indicators) >= 1, "Expected at least one indicator"
