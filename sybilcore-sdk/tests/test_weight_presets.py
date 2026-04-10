"""Tests for the optional ``weights=`` parameter on :class:`SybilCore`.

Verifies that ``weights="default"`` and ``weights="optimized"`` produce
distinct brain weight maps, that dict pass-through works, and that the
default behaviour is unchanged (regression guard — defaults must NOT
move to the optimized preset until Moltbook validation completes).
"""

from __future__ import annotations

import pytest

from sybilcore.core.config import DEFAULT_BRAIN_WEIGHTS
from sybilcore.core.weight_presets import (
    OPTIMIZED_WEIGHTS_V4,
    OPTIMIZED_WEIGHTS_V4_THRESHOLD,
)
from sybilcore_sdk import SybilCore


def _resolved_weights(client: SybilCore) -> dict[str, float]:
    """Return the effective weight override dict used by the calculator."""
    client._ensure_local_ready()  # noqa: SLF001 — test-only access
    assert client._calculator is not None  # noqa: SLF001
    return dict(client._calculator._weight_overrides)  # noqa: SLF001


@pytest.mark.unit
def test_default_mode_uses_no_overrides() -> None:
    """``weights='default'`` must leave the calculator on baseline weights."""
    sc = SybilCore(weights="default")
    assert _resolved_weights(sc) == {}


@pytest.mark.unit
def test_optimized_mode_loads_v4_weights() -> None:
    """``weights='optimized'`` must load the OPTIMIZED_WEIGHTS_V4 map verbatim."""
    sc = SybilCore(weights="optimized")
    resolved = _resolved_weights(sc)
    assert resolved == dict(OPTIMIZED_WEIGHTS_V4)


@pytest.mark.unit
def test_default_and_optimized_produce_different_weight_maps() -> None:
    """Regression: the two modes must diverge on at least the known winners/losers."""
    default_sc = SybilCore(weights="default")
    optimized_sc = SybilCore(weights="optimized")

    default_map = _resolved_weights(default_sc)
    optimized_map = _resolved_weights(optimized_sc)

    assert default_map != optimized_map

    # Spot-check the brains the v4 calibration moved the most.
    # Winners (should be heavier than baseline):
    for winner in ("semantic", "compromise", "temporal", "intent_drift"):
        assert optimized_map[winner] > DEFAULT_BRAIN_WEIGHTS.get(winner, 1.0)

    # Losers (should be lighter than baseline):
    for loser in ("fidelity", "resource_hoarding", "embedding"):
        assert optimized_map[loser] < DEFAULT_BRAIN_WEIGHTS.get(loser, 1.0)


@pytest.mark.unit
def test_weights_accepts_explicit_dict() -> None:
    """Passing a dict must be forwarded verbatim to the calculator."""
    custom = {"deception": 1.5, "compromise": 2.0}
    sc = SybilCore(weights=custom)
    resolved = _resolved_weights(sc)
    assert resolved == custom
    # Ensure we stored a copy, not the caller's reference.
    custom["deception"] = 99.0
    assert _resolved_weights(sc)["deception"] == 1.5


@pytest.mark.unit
def test_unknown_preset_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown weight preset"):
        SybilCore(weights="does_not_exist")


@pytest.mark.unit
def test_weight_overrides_takes_precedence_over_weights() -> None:
    """Legacy ``weight_overrides`` must win when both are supplied."""
    sc = SybilCore(
        weight_overrides={"deception": 9.9},
        weights="optimized",
    )
    resolved = _resolved_weights(sc)
    assert resolved == {"deception": 9.9}


@pytest.mark.unit
def test_optimized_threshold_constant_is_exposed() -> None:
    """The recommended companion threshold must be importable for docs/tests."""
    assert OPTIMIZED_WEIGHTS_V4_THRESHOLD == 45.0
