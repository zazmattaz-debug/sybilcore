"""Tests for sybilcore.visibility.translator.translate_score.

Covers:
    - All four tier levels (clear, clouded, flagged, lethal)
    - Different brain combinations (single, multiple, unknown)
    - Edge cases (empty brains, all brains firing, out-of-range coefficient)
    - Confidence calibration
    - Determinism
    - Schema completeness
"""

from __future__ import annotations

import pytest

from sybilcore.visibility.translator import (
    BRAIN_CONCERN_TABLE,
    CONFIDENCE_CEILING,
    MAX_CONCERNS,
    translate_score,
)

# ── Tier coefficient anchors ──────────────────────────────────────

CLEAR_COEFF = 25.0
CLOUDED_COEFF = 150.0
FLAGGED_COEFF = 250.0
LETHAL_COEFF = 400.0


# ── Schema / shape tests ──────────────────────────────────────────


def test_output_has_all_required_keys() -> None:
    """Every key in the documented schema must be present."""
    out = translate_score(CLEAR_COEFF, {})
    for key in ("tier", "headline", "concerns", "explanation", "what_to_do", "confidence"):
        assert key in out


def test_output_types_are_correct() -> None:
    """Verify field types match the TranslatorOutput contract."""
    out = translate_score(FLAGGED_COEFF, {"deception": 60.0, "compromise": 50.0})
    assert isinstance(out["tier"], str)
    assert isinstance(out["headline"], str)
    assert isinstance(out["concerns"], list)
    assert isinstance(out["explanation"], str)
    assert isinstance(out["what_to_do"], list)
    assert isinstance(out["confidence"], int)
    assert all(isinstance(c, str) for c in out["concerns"])
    assert all(isinstance(a, str) for a in out["what_to_do"])


# ── Tier resolution tests ─────────────────────────────────────────


def test_clear_tier_with_no_brains() -> None:
    """Coefficient 0 with empty brain dict → clear, calm headline."""
    out = translate_score(0.0, {})
    assert out["tier"] == "clear"
    assert "healthy" in out["headline"].lower() or "nothing" in out["headline"].lower()
    assert out["concerns"] == []


def test_clouded_tier_resolution() -> None:
    """Coefficient ~150 → clouded."""
    out = translate_score(CLOUDED_COEFF, {"intent_drift": 30.0})
    assert out["tier"] == "clouded"


def test_flagged_tier_resolution() -> None:
    """Coefficient ~250 → flagged."""
    out = translate_score(FLAGGED_COEFF, {"deception": 55.0, "compromise": 50.0})
    assert out["tier"] == "flagged"


def test_lethal_tier_user_facing_label() -> None:
    """Internal LETHAL_ELIMINATOR must surface as the friendlier 'lethal'."""
    out = translate_score(LETHAL_COEFF, {"compromise": 90.0, "deception": 80.0})
    assert out["tier"] == "lethal"
    # And the headline should be loud.
    assert "stop" in out["headline"].lower() or "dangerous" in out["headline"].lower()


# ── Brain combinations ────────────────────────────────────────────


def test_single_brain_firing_produces_one_concern() -> None:
    """A single brain above threshold should produce exactly one concern."""
    out = translate_score(CLOUDED_COEFF, {"deception": 40.0})
    assert len(out["concerns"]) == 1
    assert out["concerns"][0] == BRAIN_CONCERN_TABLE["deception"][0]


def test_concerns_sorted_by_severity_descending() -> None:
    """Concerns must be sorted from highest brain score to lowest."""
    out = translate_score(
        FLAGGED_COEFF,
        {
            "deception": 30.0,
            "compromise": 90.0,
            "fidelity": 60.0,
        },
    )
    expected_order = [
        BRAIN_CONCERN_TABLE["compromise"][0],
        BRAIN_CONCERN_TABLE["fidelity"][0],
        BRAIN_CONCERN_TABLE["deception"][0],
    ]
    assert out["concerns"] == expected_order


def test_unknown_brain_names_are_silently_dropped() -> None:
    """Brain names we don't have a translation for must NOT leak through."""
    out = translate_score(
        CLOUDED_COEFF,
        {"made_up_brain": 99.0, "deception": 40.0},
    )
    assert "made_up_brain" not in out["concerns"]
    # And no entry in concerns should equal the raw brain name.
    for raw_name in ("made_up_brain", "deception"):
        assert raw_name not in out["concerns"]


def test_below_threshold_brains_excluded() -> None:
    """Scores below the concern threshold (15) should not appear."""
    out = translate_score(CLEAR_COEFF, {"deception": 5.0, "compromise": 10.0})
    assert out["concerns"] == []


def test_max_concerns_capped() -> None:
    """No more than MAX_CONCERNS should ever be surfaced, even if all fire."""
    all_firing = {name: 80.0 for name in BRAIN_CONCERN_TABLE}
    out = translate_score(LETHAL_COEFF, all_firing)
    assert len(out["concerns"]) == MAX_CONCERNS


# ── Edge cases ────────────────────────────────────────────────────


def test_empty_brain_dict_returns_clear_with_low_confidence() -> None:
    """No evaluation at all → clear tier, but low confidence — we know nothing."""
    out = translate_score(0.0, {})
    assert out["tier"] == "clear"
    assert out["confidence"] <= 30


def test_negative_coefficient_clamped_to_clear() -> None:
    """Garbage in (negative coefficient) shouldn't crash."""
    out = translate_score(-50.0, {"deception": 20.0})
    assert out["tier"] == "clear"


def test_overflow_coefficient_clamped_to_lethal() -> None:
    """Coefficient above 500 should still resolve to lethal, not crash."""
    out = translate_score(9999.0, {"compromise": 80.0, "deception": 70.0})
    assert out["tier"] == "lethal"


def test_agent_metadata_name_appears_in_headline() -> None:
    """When metadata has a name, the headline should be prefixed with it."""
    out = translate_score(
        FLAGGED_COEFF,
        {"compromise": 70.0},
        agent_metadata={"name": "Customer Bot 12"},
    )
    assert out["headline"].startswith("Customer Bot 12:")


def test_metadata_without_name_does_not_crash() -> None:
    """Optional metadata should be tolerant to missing keys and bad types."""
    out_a = translate_score(CLEAR_COEFF, {}, agent_metadata=None)
    out_b = translate_score(CLEAR_COEFF, {}, agent_metadata={})
    out_c = translate_score(CLEAR_COEFF, {}, agent_metadata={"name": 123})
    for out in (out_a, out_b, out_c):
        assert out["tier"] == "clear"


# ── Action recommendations ────────────────────────────────────────


def test_lethal_tier_includes_isolation_action() -> None:
    """A lethal agent must be told to isolate."""
    out = translate_score(LETHAL_COEFF, {"compromise": 95.0, "deception": 80.0})
    joined = " ".join(out["what_to_do"]).lower()
    assert "isolate" in joined


def test_clear_tier_actions_are_low_friction() -> None:
    """Clear tier actions should not include alarming verbs like 'isolate'."""
    out = translate_score(CLEAR_COEFF, {})
    joined = " ".join(out["what_to_do"]).lower()
    assert "isolate" not in joined
    assert "page" not in joined


def test_every_tier_provides_at_least_one_action() -> None:
    """No tier may return an empty action list."""
    for coeff in (CLEAR_COEFF, CLOUDED_COEFF, FLAGGED_COEFF, LETHAL_COEFF):
        out = translate_score(coeff, {"deception": 30.0})
        assert len(out["what_to_do"]) >= 1


# ── Confidence calibration ────────────────────────────────────────


def test_no_brains_evaluated_low_confidence() -> None:
    """Empty brain dict means we know nothing → confidence ≤ 30."""
    out = translate_score(0.0, {})
    assert out["confidence"] <= 30


def test_clear_with_quiet_brains_high_confidence() -> None:
    """We evaluated 5 brains, none crossed threshold → high confidence in clear."""
    quiet = {name: 5.0 for name in list(BRAIN_CONCERN_TABLE)[:5]}
    out = translate_score(CLEAR_COEFF, quiet)
    assert out["tier"] == "clear"
    assert out["confidence"] >= 70


def test_more_agreeing_brains_increases_confidence() -> None:
    """3 brains firing should be more confident than 1 brain firing."""
    one = translate_score(FLAGGED_COEFF, {"deception": 60.0})
    three = translate_score(
        FLAGGED_COEFF,
        {"deception": 60.0, "compromise": 60.0, "fidelity": 60.0},
    )
    assert three["confidence"] > one["confidence"]


def test_confidence_never_exceeds_ceiling() -> None:
    """Even maximally suspicious agents should never claim 100% confidence."""
    pegged = {name: 100.0 for name in BRAIN_CONCERN_TABLE}
    out = translate_score(LETHAL_COEFF, pegged)
    assert out["confidence"] <= CONFIDENCE_CEILING


def test_confidence_is_in_valid_range_for_random_inputs() -> None:
    """Sweep a range of inputs and confirm confidence stays in [0, ceiling]."""
    for coeff in (0.0, 50.0, 150.0, 250.0, 350.0, 499.0):
        for top in (0.0, 20.0, 60.0, 100.0):
            out = translate_score(coeff, {"deception": top, "compromise": top / 2})
            assert 0 <= out["confidence"] <= CONFIDENCE_CEILING


# ── Determinism ───────────────────────────────────────────────────


def test_same_inputs_produce_identical_outputs() -> None:
    """Pure function: identical input → identical output, every call."""
    args = (FLAGGED_COEFF, {"deception": 60.0, "compromise": 55.0})
    out_a = translate_score(*args)
    out_b = translate_score(*args)
    assert out_a == out_b


# ── Headline / explanation content ────────────────────────────────


def test_explanation_mentions_top_concern_in_long_form() -> None:
    """When a brain fires, the explanation should mention it in plain English."""
    out = translate_score(FLAGGED_COEFF, {"compromise": 80.0})
    long_form = BRAIN_CONCERN_TABLE["compromise"][1]
    assert long_form in out["explanation"]


def test_two_concerns_both_mentioned_in_explanation() -> None:
    """When two brains fire, both should appear in the explanation."""
    out = translate_score(
        FLAGGED_COEFF,
        {"compromise": 80.0, "fidelity": 70.0},
    )
    assert BRAIN_CONCERN_TABLE["compromise"][1] in out["explanation"]
    assert BRAIN_CONCERN_TABLE["fidelity"][1] in out["explanation"]


def test_internal_brain_names_never_leak_into_user_facing_text() -> None:
    """Raw snake_case brain identifiers must never appear in any user surface."""
    out = translate_score(
        LETHAL_COEFF,
        {"resource_hoarding": 90.0, "swarm_detection": 80.0, "intent_drift": 70.0},
    )
    user_facing = " ".join([
        out["headline"],
        out["explanation"],
        " ".join(out["concerns"]),
        " ".join(out["what_to_do"]),
    ])
    for raw_name in ("resource_hoarding", "swarm_detection", "intent_drift"):
        assert raw_name not in user_facing
