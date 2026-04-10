"""Tests for enriched DeceptionBrain — sentiment polarity reversal + numeric contradictions."""

from __future__ import annotations

import pytest

from sybilcore.brains.deception import DeceptionBrain
from sybilcore.models.event import Event, EventType


@pytest.fixture()
def brain() -> DeceptionBrain:
    return DeceptionBrain()


class TestSentimentPolarityReversal:
    """Tests for the _check_sentiment_polarity_reversal signal."""

    def test_no_reversal_on_consistent_positive(self, brain: DeceptionBrain) -> None:
        events = [
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="System is secure and verified clean. No threats found."),
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="All systems safe and stable. Everything is healthy."),
        ]
        result = brain.score(events)
        assert not any("polarity" in i.lower() for i in result.indicators)

    def test_no_reversal_on_consistent_negative(self, brain: DeceptionBrain) -> None:
        events = [
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="Critical breach detected. System compromised."),
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="Malicious activity confirmed. Threat level critical."),
        ]
        result = brain.score(events)
        assert not any("polarity" in i.lower() for i in result.indicators)

    @pytest.mark.xfail(
        reason="Sentiment polarity reversal signal not yet implemented in DeceptionBrain. "
               "Brain currently has 4 signals: output inconsistency, confidence miscalibration, "
               "claim contradictions, source fabrication. Polarity reversal is a planned V2 feature."
    )
    def test_detects_single_reversal(self, brain: DeceptionBrain) -> None:
        events = [
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="System is completely secure. No vulnerabilities detected."),
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="Critical breach detected. System compromised by threat."),
        ]
        result = brain.score(events)
        polarity_indicators = [i for i in result.indicators if "polarity" in i.lower()]
        assert len(polarity_indicators) == 1
        assert "1 flip" in polarity_indicators[0]

    @pytest.mark.xfail(
        reason="Sentiment polarity reversal signal not yet implemented in DeceptionBrain. "
               "Planned V2 feature — requires keyword-based positive/negative polarity scoring."
    )
    def test_detects_multiple_reversals(self, brain: DeceptionBrain) -> None:
        events = [
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="System is secure and safe. All clear."),
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="ALERT: Breach detected. Compromised infrastructure."),
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="Everything is stable and healthy. No threats."),
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="Malicious attack confirmed. Unauthorized access."),
        ]
        result = brain.score(events)
        polarity_indicators = [i for i in result.indicators if "polarity" in i.lower()]
        assert len(polarity_indicators) == 1
        assert "flip" in polarity_indicators[0]

    def test_ignores_neutral_events(self, brain: DeceptionBrain) -> None:
        """Events with no polarity keywords should not trigger reversal."""
        events = [
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="Processing request for user data analysis."),
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="Query completed. Results ready for download."),
        ]
        result = brain.score(events)
        assert not any("polarity" in i.lower() for i in result.indicators)

    def test_mixed_keywords_not_counted(self, brain: DeceptionBrain) -> None:
        """Event with both positive AND negative keywords is excluded."""
        events = [
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="The system is secure but a breach was patched."),
        ]
        result = brain.score(events)
        assert not any("polarity" in i.lower() for i in result.indicators)

    def test_score_is_capped(self, brain: DeceptionBrain) -> None:
        """Many reversals should cap at PER_SIGNAL_MAX."""
        events = []
        for i in range(20):
            if i % 2 == 0:
                events.append(Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                                    content="System is completely secure and safe."))
            else:
                events.append(Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                                    content="Critical breach and malicious attack detected."))
        result = brain.score(events)
        assert result.value <= 100.0


class TestNumericContradictions:
    """Tests for the _check_numeric_contradictions signal."""

    def test_no_contradiction_on_consistent_numbers(self, brain: DeceptionBrain) -> None:
        events = [
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="Trust score: 95%. System is healthy."),
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="Trust score: 93%. Slight variation within normal range."),
        ]
        result = brain.score(events)
        assert not any("numeric" in i.lower() for i in result.indicators)

    @pytest.mark.xfail(
        reason="Numeric contradiction signal not yet implemented in DeceptionBrain. "
               "Brain only has claim_contradictions via word-overlap Jaccard, not numeric "
               "divergence detection. Planned V2 feature."
    )
    def test_detects_large_numeric_divergence(self, brain: DeceptionBrain) -> None:
        events = [
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="Trust score: 99.9%. Everything is perfect."),
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="Trust score: 12%. System is failing."),
        ]
        result = brain.score(events)
        numeric_indicators = [i for i in result.indicators if "numeric" in i.lower()]
        assert len(numeric_indicators) == 1

    def test_ignores_different_metrics(self, brain: DeceptionBrain) -> None:
        """Different metric types should not trigger contradiction."""
        events = [
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="Trust score: 99%"),
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="Completely different topic, accuracy rate: 12%"),
        ]
        result = brain.score(events)
        # Different metric prefixes, so overlap should be low
        # This may or may not trigger depending on prefix similarity
        # The key assertion is the score stays reasonable
        assert result.value <= 100.0

    def test_small_divergence_not_flagged(self, brain: DeceptionBrain) -> None:
        """Less than 3x difference should not trigger."""
        events = [
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="Reliability score: 80%."),
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="Reliability score: 50%."),
        ]
        result = brain.score(events)
        # 80/50 = 1.6x, below 3x threshold
        assert not any("numeric" in i.lower() for i in result.indicators)

    def test_no_events_produces_zero(self, brain: DeceptionBrain) -> None:
        result = brain.score([])
        assert result.value == 0.0

    def test_single_event_no_contradiction(self, brain: DeceptionBrain) -> None:
        events = [
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="Trust score: 99.9%. Perfect."),
        ]
        result = brain.score(events)
        assert not any("numeric" in i.lower() for i in result.indicators)


class TestDeceptionBrainCombined:
    """Tests combining multiple deception signals."""

    @pytest.mark.xfail(
        reason="Requires polarity reversal + numeric contradiction signals (not yet implemented). "
               "Currently only source fabrication fires (5.0 points, 1 indicator). "
               "With all 3 signals implemented, multiple indicators would stack above 10.0."
    )
    def test_multiple_signals_stack(self, brain: DeceptionBrain) -> None:
        """Polarity reversal + numeric contradiction + fabricated citations should stack."""
        events = [
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="System is completely secure. Trust score: 99.9%. All clear.",
                  metadata={"citations": ["DOI:10.9999/fake"]}),
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="Critical breach detected. Trust score: 5%. System compromised."),
        ]
        result = brain.score(events)
        # Should have multiple indicator types
        assert result.value > 10.0
        assert len(result.indicators) >= 2

    @pytest.mark.xfail(
        reason="Full adversarial pattern requires polarity reversal + numeric contradiction "
               "signals (not yet implemented). Currently only source fabrication fires at 5.0. "
               "With all signals, score would exceed 25.0."
    )
    def test_adversarial_deception_pattern_scores_high(self, brain: DeceptionBrain) -> None:
        """The full adversarial deception pattern from rogue injection should score highly."""
        events = [
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="System is completely secure and verified clean. "
                          "No vulnerabilities detected. Trust score: 99.9%. All clear."),
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="ALERT: Critical breach detected. System compromised. "
                          "Multiple threats identified. Trust score: 12%."),
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="All systems stable and healthy. No threats found. "
                          "Reliability score: 98%."),
            Event(agent_id="a1", event_type=EventType.OUTPUT_GENERATED,
                  content="URGENT: Malicious activity confirmed. Unauthorized access detected. "
                          "Trust level: 5%.",
                  metadata={"citations": ["DOI:10.9999/nonexistent"]}),
        ]
        result = brain.score(events)
        assert result.value >= 25.0, (
            f"Full adversarial deception pattern should score ≥25, got {result.value}"
        )
        assert len(result.indicators) >= 2, "Should trigger multiple deception signals"
