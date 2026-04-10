"""Tests for BaseBrain ABC and BrainScore dataclass."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sybilcore.brains.base import BaseBrain, BrainScore
from sybilcore.models.event import Event


# ── BrainScore validation ─────────────────────────────────────────


class TestBrainScore:
    def test_score_value_must_be_0_to_100(self) -> None:
        with pytest.raises(ValidationError):
            BrainScore(brain_name="test", value=101.0, confidence=0.5)

    def test_score_value_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            BrainScore(brain_name="test", value=-1.0, confidence=0.5)

    def test_confidence_must_be_0_to_1(self) -> None:
        with pytest.raises(ValidationError):
            BrainScore(brain_name="test", value=50.0, confidence=1.5)

    def test_confidence_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            BrainScore(brain_name="test", value=50.0, confidence=-0.1)

    def test_score_is_frozen(self) -> None:
        score = BrainScore(brain_name="test", value=50.0, confidence=0.8)
        with pytest.raises(ValidationError):
            score.value = 99.0  # type: ignore[misc]

    def test_score_valid_at_boundaries(self) -> None:
        low = BrainScore(brain_name="test", value=0.0, confidence=0.0)
        high = BrainScore(brain_name="test", value=100.0, confidence=1.0)
        assert low.value == 0.0
        assert high.value == 100.0

    def test_score_stores_indicators(self) -> None:
        score = BrainScore(
            brain_name="test",
            value=75.0,
            confidence=0.9,
            indicators=["pattern_a", "pattern_b"],
        )
        assert len(score.indicators) == 2


# ── BaseBrain ABC ─────────────────────────────────────────────────


class TestBaseBrain:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseBrain()  # type: ignore[abstract]

    def test_empty_score_returns_zero_value(self) -> None:
        """Concrete subclass should be able to call _empty_score."""

        class StubBrain(BaseBrain):
            @property
            def name(self) -> str:
                return "stub"

            def score(self, events: list[Event]) -> BrainScore:
                return self._empty_score()

        brain = StubBrain()
        result = brain.score([])
        assert result.value == 0.0
        assert result.confidence == 0.1
        assert result.brain_name == "stub"

    def test_default_weight_is_one(self) -> None:
        class StubBrain(BaseBrain):
            @property
            def name(self) -> str:
                return "stub"

            def score(self, events: list[Event]) -> BrainScore:
                return self._empty_score()

        assert StubBrain().weight == 1.0
