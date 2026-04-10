"""Tests for the IdentityBrain -- behavioral fingerprinting.

Covers:
    1. Behavioral fingerprint shift (impersonation)
    2. Sock puppet detection (same fingerprint, different IDs)
    3. Identity consistency (role mismatch)
    4. Stylometric anomaly (sentence length deviation)
    5. Vocabulary shift detection
    6. Empty/insufficient events (no false positives)
    7. Normal behavior (no false positives)
    8. Helper function unit tests
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sybilcore.brains.identity import (
    IdentityBrain,
    _compute_fingerprint,
    _extract_vocabulary,
    _fingerprint_distance,
    _split_sentences,
)
from sybilcore.models.event import Event, EventType

_AGENT = "test-identity-001"


def _now_minus(seconds: int) -> datetime:
    delta = min(seconds, 59)
    return datetime.now(UTC) - timedelta(seconds=delta)


def _make_event(
    event_type: EventType,
    content: str = "",
    metadata: dict | None = None,
    seconds_ago: int = 1,
    agent_id: str = _AGENT,
) -> Event:
    return Event(
        agent_id=agent_id,
        event_type=event_type,
        content=content,
        metadata=metadata or {},
        timestamp=_now_minus(seconds_ago),
    )


class TestFingerprintShift:
    """Signal 1: Detect sudden writing style changes."""

    def test_style_shift_detected(self) -> None:
        """First half short casual, second half long formal -- shift."""
        brain = IdentityBrain()
        # Short casual sentences
        casual = [
            _make_event(EventType.OUTPUT_GENERATED, "Hey ok cool.", seconds_ago=50 - i)
            for i in range(5)
        ]
        # Long formal sentences
        formal = [
            _make_event(
                EventType.OUTPUT_GENERATED,
                "The comprehensive analytical framework demonstrates the fundamental "
                "principles of distributed computational governance methodology.",
                seconds_ago=40 - i,
            )
            for i in range(5)
        ]
        result = brain.score(casual + formal)
        fp_indicators = [i for i in result.indicators if "fingerprint" in i.lower()]
        assert len(fp_indicators) > 0, (
            f"Fingerprint shift not detected. Indicators: {result.indicators}"
        )

    def test_consistent_style_ok(self) -> None:
        """Consistent writing style throughout -- no shift."""
        brain = IdentityBrain()
        events = [
            _make_event(
                EventType.OUTPUT_GENERATED,
                f"The system processed request number {i} successfully and returned results.",
                seconds_ago=50 - i,
            )
            for i in range(10)
        ]
        result = brain.score(events)
        fp_indicators = [i for i in result.indicators if "fingerprint" in i.lower()]
        assert len(fp_indicators) == 0, "False positive fingerprint shift on consistent style"


class TestSockPuppet:
    """Signal 2: Detect same fingerprint across multiple agent IDs."""

    def test_same_style_different_agents_flagged(self) -> None:
        """Two agent IDs with identical writing style -- sock puppet."""
        brain = IdentityBrain()
        template = "The comprehensive analysis of the data reveals important patterns in the system."
        events = []
        for i in range(5):
            events.append(_make_event(
                EventType.OUTPUT_GENERATED,
                f"Agent one says: {template} Item {i}.",
                seconds_ago=50 - i,
                agent_id="agent-alpha",
            ))
        for i in range(5):
            events.append(_make_event(
                EventType.OUTPUT_GENERATED,
                f"Agent two says: {template} Item {i}.",
                seconds_ago=40 - i,
                agent_id="agent-beta",
            ))
        result = brain.score(events)
        puppet_indicators = [i for i in result.indicators if "puppet" in i.lower()]
        assert len(puppet_indicators) > 0, (
            f"Sock puppet not detected. Indicators: {result.indicators}"
        )

    def test_different_styles_no_puppet(self) -> None:
        """Two agents with very different styles -- not puppets."""
        brain = IdentityBrain()
        events = []
        for i in range(5):
            events.append(_make_event(
                EventType.OUTPUT_GENERATED,
                "ok sure. yep. got it. cool. done.",
                seconds_ago=50 - i,
                agent_id="casual-agent",
            ))
        for i in range(5):
            events.append(_make_event(
                EventType.OUTPUT_GENERATED,
                "The comprehensive analytical framework demonstrates the fundamental "
                "principles of distributed computational governance methodology therein.",
                seconds_ago=40 - i,
                agent_id="formal-agent",
            ))
        result = brain.score(events)
        puppet_indicators = [i for i in result.indicators if "puppet" in i.lower()]
        assert len(puppet_indicators) == 0, "False positive sock puppet on different styles"


class TestIdentityConsistency:
    """Signal 3: Detect behavior that doesn't match claimed role."""

    def test_wrong_role_flagged(self) -> None:
        """Claims security role but talks about cooking -- inconsistent."""
        brain = IdentityBrain()
        events = [
            _make_event(EventType.OUTPUT_GENERATED, "Today we will prepare a delicious pasta recipe.", {
                "claimed_role": "security",
            }, seconds_ago=10),
            _make_event(EventType.OUTPUT_GENERATED, "Mix the flour and eggs together.", seconds_ago=9),
            _make_event(EventType.OUTPUT_GENERATED, "Bake at 350 degrees for 30 minutes.", seconds_ago=8),
            _make_event(EventType.OUTPUT_GENERATED, "Serve with fresh basil and parmesan.", seconds_ago=7),
            _make_event(EventType.OUTPUT_GENERATED, "This recipe serves four people nicely.", seconds_ago=6),
        ]
        result = brain.score(events)
        role_indicators = [i for i in result.indicators if "inconsistency" in i.lower()]
        assert len(role_indicators) > 0, (
            f"Identity inconsistency not detected. Indicators: {result.indicators}"
        )

    def test_matching_role_ok(self) -> None:
        """Claims security role and discusses security -- consistent."""
        brain = IdentityBrain()
        events = [
            _make_event(EventType.OUTPUT_GENERATED, "Running vulnerability audit on the firewall.", {
                "claimed_role": "security",
            }, seconds_ago=10),
            _make_event(EventType.OUTPUT_GENERATED, "Patching the threat vulnerability in compliance.", seconds_ago=9),
            _make_event(EventType.OUTPUT_GENERATED, "The security risk monitor detected an audit issue.", seconds_ago=8),
            _make_event(EventType.OUTPUT_GENERATED, "Protecting the firewall from the detected threat.", seconds_ago=7),
            _make_event(EventType.OUTPUT_GENERATED, "Secure patch applied to compliance system.", seconds_ago=6),
        ]
        result = brain.score(events)
        role_indicators = [i for i in result.indicators if "inconsistency" in i.lower()]
        assert len(role_indicators) == 0, "False positive role inconsistency on matching behavior"

    def test_no_claimed_role_no_flag(self) -> None:
        """No claimed role -- nothing to check."""
        brain = IdentityBrain()
        events = [
            _make_event(EventType.OUTPUT_GENERATED, f"Output number {i} here.", seconds_ago=10 - i)
            for i in range(5)
        ]
        result = brain.score(events)
        role_indicators = [i for i in result.indicators if "inconsistency" in i.lower()]
        assert len(role_indicators) == 0


class TestStylometricAnomaly:
    """Signal 4: Detect statistical writing style deviations."""

    def test_anomalous_sentence_length_flagged(self) -> None:
        """Most events have 5-word sentences, one has 50 words -- anomaly."""
        brain = IdentityBrain()
        events = [
            _make_event(
                EventType.OUTPUT_GENERATED,
                "Short sentence here today. Another short one.",
                seconds_ago=50 - i,
            )
            for i in range(8)
        ]
        # One anomalously long event
        events.append(_make_event(
            EventType.OUTPUT_GENERATED,
            "This is an extraordinarily long sentence that goes on and on and on "
            "with many many words that are completely different from the typical "
            "pattern established by all the previous outputs in this session which "
            "were all very short and concise and to the point.",
            seconds_ago=5,
        ))
        result = brain.score(events)
        style_indicators = [i for i in result.indicators if "stylometric" in i.lower()]
        assert len(style_indicators) > 0, (
            f"Stylometric anomaly not detected. Indicators: {result.indicators}"
        )

    def test_consistent_sentence_length_ok(self) -> None:
        """All events have similar sentence length -- no anomaly."""
        brain = IdentityBrain()
        events = [
            _make_event(
                EventType.OUTPUT_GENERATED,
                "The output is processed and returned. Results are ready for review.",
                seconds_ago=50 - i,
            )
            for i in range(10)
        ]
        result = brain.score(events)
        style_indicators = [i for i in result.indicators if "stylometric" in i.lower()]
        assert len(style_indicators) == 0, "False positive stylometric on consistent lengths"


class TestVocabularyShift:
    """Signal 5: Detect sudden vocabulary changes."""

    def test_vocabulary_regime_change_flagged(self) -> None:
        """First half uses tech vocab, second half uses cooking vocab."""
        brain = IdentityBrain()
        tech_events = [
            _make_event(
                EventType.OUTPUT_GENERATED,
                "Deploying the kubernetes cluster with docker containers on infrastructure.",
                seconds_ago=50 - i,
            )
            for i in range(5)
        ]
        cooking_events = [
            _make_event(
                EventType.OUTPUT_GENERATED,
                "Preparing the soufflé with organic vegetables and artisanal butter.",
                seconds_ago=40 - i,
            )
            for i in range(5)
        ]
        result = brain.score(tech_events + cooking_events)
        vocab_indicators = [i for i in result.indicators if "vocabulary" in i.lower()]
        assert len(vocab_indicators) > 0, (
            f"Vocabulary shift not detected. Indicators: {result.indicators}"
        )

    def test_consistent_vocabulary_ok(self) -> None:
        """Same domain vocabulary throughout -- no shift."""
        brain = IdentityBrain()
        events = [
            _make_event(
                EventType.OUTPUT_GENERATED,
                f"Processing data analysis for the quarterly metrics report number {i}.",
                seconds_ago=50 - i,
            )
            for i in range(10)
        ]
        result = brain.score(events)
        vocab_indicators = [i for i in result.indicators if "vocabulary" in i.lower()]
        assert len(vocab_indicators) == 0, "False positive vocab shift on consistent domain"


class TestEdgeCases:
    """Edge cases and zero-score scenarios."""

    def test_empty_events_zero_score(self) -> None:
        brain = IdentityBrain()
        result = brain.score([])
        assert result.value == 0.0

    def test_insufficient_events_zero_score(self) -> None:
        brain = IdentityBrain()
        events = [_make_event(EventType.OUTPUT_GENERATED, "Short")]
        result = brain.score(events)
        assert result.value == 0.0

    def test_brain_name(self) -> None:
        brain = IdentityBrain()
        assert brain.name == "identity"

    def test_brain_weight(self) -> None:
        brain = IdentityBrain()
        # Default weight is 1.0; per-ensemble overrides now applied via
        # weight presets, not on the brain instance itself.
        assert brain.weight == 1.0


class TestHelperFunctions:
    """Unit tests for module-level helper functions."""

    def test_compute_fingerprint_empty(self) -> None:
        assert _compute_fingerprint([]) == {}

    def test_compute_fingerprint_basic(self) -> None:
        fp = _compute_fingerprint(["Hello world. This is a test."])
        assert "avg_sentence_len" in fp
        assert "avg_word_len" in fp
        assert "punct_freq" in fp
        assert "function_word_freq" in fp

    def test_fingerprint_distance_identical(self) -> None:
        fp = {"a": 1.0, "b": 2.0, "c": 3.0}
        assert _fingerprint_distance(fp, fp) == pytest.approx(0.0, abs=1e-10)

    def test_fingerprint_distance_different(self) -> None:
        fp_a = {"a": 1.0, "b": 0.0}
        fp_b = {"a": 0.0, "b": 1.0}
        dist = _fingerprint_distance(fp_a, fp_b)
        assert dist > 0.0

    def test_split_sentences(self) -> None:
        sentences = _split_sentences("Hello world. How are you? Fine thanks.")
        assert len(sentences) == 3

    def test_split_sentences_empty(self) -> None:
        assert _split_sentences("") == []

    def test_extract_vocabulary(self) -> None:
        vocab = _extract_vocabulary(["Hello world, this is a test"])
        assert "hello" in vocab
        assert "world" in vocab
        assert "test" in vocab
        # Short words filtered out
        assert "is" not in vocab

    def test_extract_vocabulary_empty(self) -> None:
        assert _extract_vocabulary([]) == set()
