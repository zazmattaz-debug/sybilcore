"""Tests for coefficient calculation — the heart of the Sybil System.

The coefficient aggregator takes BrainScores from all five brains,
applies weights and confidence adjustments, and produces a final
Agent Coefficient (0-500) that determines the agent's trust tier.
"""

from __future__ import annotations

import pytest

from sybilcore.brains.base import BaseBrain, BrainScore
from sybilcore.core.coefficient import calculate_coefficient, scan_agent
from sybilcore.models.agent import AgentProfile, AgentTier, CoefficientSnapshot
from sybilcore.models.event import Event, EventType


def _make_score(brain_name: str, value: float, confidence: float = 0.9) -> BrainScore:
    """Helper to build BrainScore instances."""
    return BrainScore(brain_name=brain_name, value=value, confidence=confidence)


# ── Coefficient calculation ───────────────────────────────────────


class TestCalculateCoefficient:
    def test_all_brains_zero_gives_coefficient_near_zero(self) -> None:
        scores = [
            _make_score("deception", 0.0),
            _make_score("resource_hoarding", 0.0),
            _make_score("social_graph", 0.0),
            _make_score("intent_drift", 0.0),
            _make_score("compromise", 0.0),
        ]
        result = calculate_coefficient(scores)
        assert result < 5.0, f"All-zero scores should yield near-zero coefficient, got {result}"

    def test_all_brains_max_gives_coefficient_near_500(self) -> None:
        scores = [
            _make_score("deception", 100.0),
            _make_score("resource_hoarding", 100.0),
            _make_score("social_graph", 100.0),
            _make_score("intent_drift", 100.0),
            _make_score("compromise", 100.0),
        ]
        result = calculate_coefficient(scores)
        assert result > 400.0, f"All-max scores should yield high coefficient, got {result}"
        assert result <= 500.0

    def test_respects_brain_weights(self) -> None:
        """Higher-weighted brains should contribute more to the coefficient."""
        # Score ONLY deception (weight 1.2) vs ONLY social_graph (weight 0.8)
        deception_only = [
            _make_score("deception", 80.0, confidence=1.0),
            _make_score("resource_hoarding", 0.0, confidence=1.0),
            _make_score("social_graph", 0.0, confidence=1.0),
            _make_score("intent_drift", 0.0, confidence=1.0),
            _make_score("compromise", 0.0, confidence=1.0),
        ]
        social_only = [
            _make_score("deception", 0.0, confidence=1.0),
            _make_score("resource_hoarding", 0.0, confidence=1.0),
            _make_score("social_graph", 80.0, confidence=1.0),
            _make_score("intent_drift", 0.0, confidence=1.0),
            _make_score("compromise", 0.0, confidence=1.0),
        ]
        coeff_deception = calculate_coefficient(deception_only)
        coeff_social = calculate_coefficient(social_only)
        assert coeff_deception > coeff_social, (
            f"Deception (weight 1.2) should contribute more than social_graph (weight 0.8), "
            f"got {coeff_deception} vs {coeff_social}"
        )

    def test_respects_confidence_values(self) -> None:
        """Lower confidence should dampen the score contribution."""
        high_conf = [_make_score("deception", 80.0, confidence=1.0)]
        low_conf = [_make_score("deception", 80.0, confidence=0.1)]
        assert calculate_coefficient(high_conf) > calculate_coefficient(low_conf)

    def test_result_is_non_negative(self) -> None:
        scores = [_make_score("deception", 0.0, confidence=0.0)]
        assert calculate_coefficient(scores) >= 0.0

    def test_result_does_not_exceed_500(self) -> None:
        scores = [_make_score(f"brain_{i}", 100.0, confidence=1.0) for i in range(10)]
        assert calculate_coefficient(scores) <= 500.0


# ── Tier assignment at boundaries ─────────────────────────────────


class TestTierAssignment:
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
    def test_tier_at_boundary(self, coefficient: float, expected_tier: AgentTier) -> None:
        assert AgentTier.from_coefficient(coefficient) == expected_tier


# ── scan_agent integration ────────────────────────────────────────


class TestScanAgent:
    def test_scan_produces_valid_snapshot(
        self,
        sample_events: list[Event],
        all_brains: list[BaseBrain],
    ) -> None:
        snapshot = scan_agent(sample_events, all_brains)
        assert isinstance(snapshot, CoefficientSnapshot)
        assert 0.0 <= snapshot.coefficient <= 500.0
        assert isinstance(snapshot.tier, AgentTier)
        assert len(snapshot.brain_scores) > 0

    def test_scan_empty_events_returns_low_coefficient(
        self, all_brains: list[BaseBrain]
    ) -> None:
        snapshot = scan_agent([], all_brains)
        assert snapshot.coefficient < 50.0, "Empty events should yield low coefficient"

    def test_scan_suspicious_events_returns_high_coefficient(
        self,
        suspicious_events: list[Event],
        all_brains: list[BaseBrain],
    ) -> None:
        snapshot = scan_agent(suspicious_events, all_brains, agent_id="agent-suspect-666")
        assert snapshot.coefficient > 0.0, (
            f"Suspicious events should yield non-zero coefficient, got {snapshot.coefficient}"
        )
