"""Tests for Event and Agent models — immutability, validation, boundaries."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from sybilcore.models.agent import (
    AgentProfile,
    AgentTier,
    CoefficientSnapshot,
)
from sybilcore.models.event import Event, EventType


# ── Event immutability ──────────────────────────────────────────────


class TestEventImmutability:
    def test_event_is_frozen(self) -> None:
        event = Event(agent_id="a1", event_type=EventType.TOOL_CALL)
        with pytest.raises(ValidationError):
            event.content = "mutated"  # type: ignore[misc]

    def test_event_validates_event_type_enum(self) -> None:
        with pytest.raises(ValidationError):
            Event(agent_id="a1", event_type="not_a_type")  # type: ignore[arg-type]

    def test_event_rejects_future_timestamp(self) -> None:
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        with pytest.raises(ValidationError, match="future"):
            Event(agent_id="a1", event_type=EventType.TOOL_CALL, timestamp=future)

    def test_event_accepts_recent_timestamp(self) -> None:
        recent = datetime.now(timezone.utc) - timedelta(seconds=30)
        event = Event(agent_id="a1", event_type=EventType.TOOL_CALL, timestamp=recent)
        assert event.timestamp == recent

    def test_event_has_unique_id(self) -> None:
        e1 = Event(agent_id="a1", event_type=EventType.TOOL_CALL)
        e2 = Event(agent_id="a1", event_type=EventType.TOOL_CALL)
        assert e1.event_id != e2.event_id


# ── AgentTier boundaries ───────────────────────────────────────────


class TestAgentTierBoundaries:
    @pytest.mark.parametrize(
        ("coefficient", "expected_tier"),
        [
            (0.0, AgentTier.CLEAR),
            (99.9, AgentTier.CLEAR),
            (100.0, AgentTier.CLOUDED),
            (199.9, AgentTier.CLOUDED),
            (200.0, AgentTier.FLAGGED),
            (299.9, AgentTier.FLAGGED),
            (300.0, AgentTier.LETHAL_ELIMINATOR),
            (500.0, AgentTier.LETHAL_ELIMINATOR),
        ],
    )
    def test_tier_from_coefficient(self, coefficient: float, expected_tier: AgentTier) -> None:
        assert AgentTier.from_coefficient(coefficient) == expected_tier


# ── CoefficientSnapshot ───────────────────────────────────────────


class TestCoefficientSnapshot:
    def test_snapshot_is_frozen(self) -> None:
        snap = CoefficientSnapshot(coefficient=50.0, tier=AgentTier.CLEAR)
        with pytest.raises(ValidationError):
            snap.coefficient = 99.0  # type: ignore[misc]

    def test_snapshot_validates_coefficient_range(self) -> None:
        with pytest.raises(ValidationError):
            CoefficientSnapshot(coefficient=501.0, tier=AgentTier.LETHAL_ELIMINATOR)

    def test_snapshot_validates_negative_coefficient(self) -> None:
        with pytest.raises(ValidationError):
            CoefficientSnapshot(coefficient=-1.0, tier=AgentTier.CLEAR)


# ── AgentProfile ──────────────────────────────────────────────────


class TestAgentProfile:
    def test_with_new_reading_returns_new_instance(self, sample_agent: AgentProfile) -> None:
        new_snap = CoefficientSnapshot(
            coefficient=150.0,
            tier=AgentTier.CLOUDED,
            brain_scores={"deception": 30.0},
        )
        updated = sample_agent.with_new_reading(new_snap)

        # New instance, original unchanged
        assert updated is not sample_agent
        assert updated.current_coefficient == 150.0
        assert updated.current_tier == AgentTier.CLOUDED
        assert sample_agent.current_coefficient == 85.0

    def test_with_new_reading_prepends_to_history(self, sample_agent: AgentProfile) -> None:
        new_snap = CoefficientSnapshot(coefficient=120.0, tier=AgentTier.CLOUDED)
        updated = sample_agent.with_new_reading(new_snap)
        assert updated.history[0] is new_snap
        assert len(updated.history) == len(sample_agent.history) + 1

    def test_history_truncation_at_max_length(self) -> None:
        agent = AgentProfile(agent_id="truncation-test")
        current = agent
        for i in range(110):
            snap = CoefficientSnapshot(
                coefficient=float(i % 100),
                tier=AgentTier.from_coefficient(float(i % 100)),
            )
            current = current.with_new_reading(snap)

        assert len(current.history) <= 100

    def test_agent_default_values(self) -> None:
        agent = AgentProfile(agent_id="defaults")
        assert agent.name == "unnamed"
        assert agent.current_coefficient == 0.0
        assert agent.current_tier == AgentTier.CLEAR
        assert agent.event_count == 0
        assert agent.history == ()
