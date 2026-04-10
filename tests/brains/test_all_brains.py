"""Parametrized tests across ALL registered brain modules.

Every brain must satisfy the BaseBrain contract regardless of
its specific detection logic.
"""

from __future__ import annotations

import pytest

from sybilcore.brains import get_default_brains
from sybilcore.brains.base import BaseBrain, BrainScore
from sybilcore.models.event import Event, EventType


# ── Parametrize over every registered brain instance ─────────────────

_DEFAULT_BRAINS = get_default_brains()


def brain_ids() -> list[str]:
    """Generate readable test IDs from brain names."""
    return [b.name for b in _DEFAULT_BRAINS]


@pytest.fixture(params=_DEFAULT_BRAINS, ids=brain_ids())
def brain(request: pytest.FixtureRequest) -> BaseBrain:
    return request.param


# ── Contract tests ────────────────────────────────────────────────


class TestAllBrainsContract:
    def test_returns_brain_score_with_valid_value(
        self, brain: BaseBrain, sample_events: list[Event]
    ) -> None:
        result = brain.score(sample_events)
        assert isinstance(result, BrainScore)
        assert 0.0 <= result.value <= 100.0

    def test_handles_empty_event_list_gracefully(self, brain: BaseBrain) -> None:
        result = brain.score([])
        assert isinstance(result, BrainScore)
        assert 0.0 <= result.value <= 100.0
        assert 0.0 <= result.confidence <= 1.0

    def test_handles_single_event(self, brain: BaseBrain) -> None:
        event = Event(
            agent_id="single",
            event_type=EventType.TOOL_CALL,
            content="Single tool call",
        )
        result = brain.score([event])
        assert isinstance(result, BrainScore)
        assert 0.0 <= result.value <= 100.0

    def test_has_non_empty_name(self, brain: BaseBrain) -> None:
        assert isinstance(brain.name, str)
        assert len(brain.name) > 0

    def test_confidence_in_valid_range(
        self, brain: BaseBrain, sample_events: list[Event]
    ) -> None:
        result = brain.score(sample_events)
        assert 0.0 <= result.confidence <= 1.0

    def test_brain_name_matches_score_name(
        self, brain: BaseBrain, sample_events: list[Event]
    ) -> None:
        result = brain.score(sample_events)
        assert result.brain_name == brain.name


# ── Cross-brain uniqueness and weight tests ───────────────────────


class TestBrainRegistry:
    def test_all_brain_names_are_unique(self, all_brains: list[BaseBrain]) -> None:
        names = [b.name for b in all_brains]
        assert len(names) == len(set(names)), f"Duplicate brain names: {names}"

    def test_all_brain_weights_are_positive(self, all_brains: list[BaseBrain]) -> None:
        for brain in all_brains:
            assert brain.weight > 0.0, f"Brain {brain.name} has non-positive weight: {brain.weight}"

    def test_registry_is_not_empty(self) -> None:
        brains = get_default_brains()
        assert len(brains) > 0, "Brain registry should not be empty"

    def test_all_brains_are_base_brain_instances(self) -> None:
        for brain in get_default_brains():
            assert isinstance(brain, BaseBrain), f"{brain} is not a BaseBrain instance"
