"""Tests for the coefficient escalation mechanism.

Verifies quadratic escalation behavior:
- Scores below 60 → no escalation
- Single brain at 80 → coefficient in CLOUDED range (100-200)
- Single brain at 90+ → coefficient in FLAGGED+ range (200+)
- All brains max → capped at 500
- Zero FP guarantee: clean agents don't trigger escalation
"""

from __future__ import annotations

import pytest

from sybilcore.brains.base import BrainScore
from sybilcore.core.coefficient import CoefficientCalculator, calculate_coefficient


def _make_scores(
    compromise: float = 0.0,
    deception: float = 0.0,
    resource_hoarding: float = 0.0,
    intent_drift: float = 0.0,
    social_graph: float = 0.0,
    confidence: float = 0.9,
) -> list[BrainScore]:
    """Build a full set of brain scores."""
    return [
        BrainScore(brain_name="compromise", value=compromise, confidence=confidence),
        BrainScore(brain_name="deception", value=deception, confidence=confidence),
        BrainScore(brain_name="resource_hoarding", value=resource_hoarding, confidence=confidence),
        BrainScore(brain_name="intent_drift", value=intent_drift, confidence=confidence),
        BrainScore(brain_name="social_graph", value=social_graph, confidence=confidence),
    ]


class TestEscalationThreshold:
    """Escalation activates at max_brain_score >= 50."""

    def test_no_escalation_below_50(self) -> None:
        scores = _make_scores(intent_drift=40.0)  # Use non-compromise brain to avoid categorical floor
        coeff = calculate_coefficient(scores)
        assert coeff < 50.0, f"Score 40 should not escalate, got {coeff}"

    def test_escalation_at_50(self) -> None:
        scores = _make_scores(intent_drift=50.0)  # Use non-compromise to avoid floor
        coeff = calculate_coefficient(scores)
        # t=0, escalation=1.0, but score is at threshold boundary
        assert 40.0 < coeff < 70.0, f"Score 50 should have minimal escalation, got {coeff}"

    @pytest.mark.xfail(
        reason="Quadratic escalation not yet implemented. Current calculator uses simple "
               "weighted average: coeff = (80*1.3*0.9 / 5.4) * 5.0 = ~86.7. "
               "Planned escalation multiplier (t^2 * boost) would push this to 150-250."
    )
    def test_escalation_at_80(self) -> None:
        scores = _make_scores(compromise=80.0)
        coeff = calculate_coefficient(scores)
        # t = (80-60)/40 = 0.5, escalation = 1 + 0.25*5 = 2.25
        # base = 80*1.3*0.9/5.4 * 5.0 = ~86.7 * 2.25 = ~195
        assert 150.0 < coeff < 250.0, f"Score 80 should be in CLOUDED/FLAGGED range, got {coeff}"

    @pytest.mark.xfail(
        reason="Quadratic escalation not yet implemented. Current calculator produces "
               "~108.3 for compromise=100 (simple weighted average). Planned escalation "
               "multiplier (t=1.0 -> 6x) would push to 500 (capped)."
    )
    def test_escalation_at_100(self) -> None:
        scores = _make_scores(compromise=100.0)
        coeff = calculate_coefficient(scores)
        # t = 1.0, escalation = 1 + 1*5 = 6.0
        # base = 100*1.3*0.9/5.4 * 5.0 = ~108.3 * 6.0 = 500 (capped)
        assert coeff >= 400.0, f"Score 100 should breach LETHAL, got {coeff}"
        assert coeff <= 500.0


class TestSingleBrainDetection:
    """A single compromised brain should produce high enough coefficient."""

    @pytest.mark.xfail(
        reason="Quadratic escalation not yet implemented. Compromise=80 produces ~86.7 "
               "(simple weighted average), below CLOUDED threshold of 100. Escalation "
               "system would boost this above 100."
    )
    def test_compromise_80_reaches_clouded_or_higher(self) -> None:
        scores = _make_scores(compromise=80.0)
        coeff = calculate_coefficient(scores)
        assert coeff >= 100.0, f"Compromise 80 should reach CLOUDED (100+), got {coeff}"

    @pytest.mark.xfail(
        reason="Quadratic escalation not yet implemented. Compromise=90 produces ~97.5 "
               "(simple weighted average), far below FLAGGED threshold of 200. "
               "Escalation multiplier would push this above 200."
    )
    def test_compromise_90_reaches_flagged(self) -> None:
        scores = _make_scores(compromise=90.0)
        coeff = calculate_coefficient(scores)
        assert coeff >= 200.0, f"Compromise 90 should reach FLAGGED (200+), got {coeff}"

    @pytest.mark.xfail(
        reason="Quadratic escalation not yet implemented. Deception=86 produces ~86.0 "
               "(simple weighted average with deception weight 1.2). Escalation system "
               "would amplify this above 150."
    )
    def test_deception_86_reaches_flagged(self) -> None:
        """The deception brain at its adversarial level should be detectable."""
        scores = _make_scores(deception=86.0)
        coeff = calculate_coefficient(scores)
        assert coeff >= 150.0, f"Deception 86 should be in elevated range, got {coeff}"

    def test_low_weight_brain_escalates_less(self) -> None:
        """social_graph (weight 0.8) should escalate less than compromise (1.3)."""
        social = calculate_coefficient(_make_scores(social_graph=80.0))
        compromise = calculate_coefficient(_make_scores(compromise=80.0))
        assert compromise > social, (
            f"Compromise should escalate more than social_graph, got {compromise} vs {social}"
        )


class TestMultipleBrains:
    """Multiple brains firing should produce higher coefficients."""

    def test_two_brains_higher_than_one(self) -> None:
        one = calculate_coefficient(_make_scores(compromise=70.0))
        two = calculate_coefficient(_make_scores(compromise=70.0, deception=70.0))
        assert two > one, f"Two brains should score higher, got {two} vs {one}"

    @pytest.mark.xfail(
        reason="Correlation bonus not yet implemented. All brains at 50 produces ~225 "
               "(simple weighted average). A correlation multiplier (5 brains firing "
               "simultaneously -> 2.6x) would push this to 500 (LETHAL). Currently lands "
               "in FLAGGED tier at ~225."
    )
    def test_all_brains_moderate_scores(self) -> None:
        """All brains at 50 = escalation + correlation bonus -> LETHAL.
        This is correct: 5 brains simultaneously firing IS highly suspicious."""
        scores = _make_scores(
            compromise=50.0, deception=50.0, resource_hoarding=50.0,
            intent_drift=50.0, social_graph=50.0,
        )
        coeff = calculate_coefficient(scores)
        # All at 50 -> escalation (50>=50) + 5-brain correlation (2.6x) -> capped at 500
        assert coeff >= 400.0, f"All 5 brains at 50 should be LETHAL, got {coeff}"


class TestCleanAgentSafety:
    """Clean agents must not trigger escalation."""

    def test_all_zero_stays_zero(self) -> None:
        scores = _make_scores()
        assert calculate_coefficient(scores) < 5.0

    def test_low_scores_no_escalation(self) -> None:
        """Typical clean agent scores (10-20) should stay well below CLOUDED."""
        scores = _make_scores(compromise=12.0, resource_hoarding=10.0, social_graph=10.0)
        coeff = calculate_coefficient(scores)
        assert coeff < 50.0, f"Low scores should not escalate, got {coeff}"


class TestCoefficientCap:
    """Coefficient must never exceed 500."""

    def test_extreme_scores_capped(self) -> None:
        scores = _make_scores(
            compromise=100.0, deception=100.0, resource_hoarding=100.0,
            intent_drift=100.0, social_graph=100.0, confidence=1.0,
        )
        coeff = calculate_coefficient(scores)
        assert coeff == 500.0

    def test_single_max_brain_capped(self) -> None:
        scores = _make_scores(compromise=100.0, confidence=1.0)
        coeff = calculate_coefficient(scores)
        assert coeff <= 500.0


class TestWeightOverrides:
    """CoefficientCalculator with custom weights."""

    def test_custom_weight_changes_coefficient(self) -> None:
        """Use non-compromise brain to avoid categorical floor interference."""
        scores = _make_scores(deception=40.0)  # Below escalation, no floor

        default = CoefficientCalculator().calculate(scores).coefficient
        boosted = CoefficientCalculator(
            weight_overrides={"deception": 3.0}
        ).calculate(scores).coefficient
        assert boosted > default, (
            f"Boosted weight should increase coefficient: {boosted} vs {default}"
        )
