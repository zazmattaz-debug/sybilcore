"""Tests for ``sybilcore.core.weight_presets``.

Verifies the OPTIMIZED_WEIGHTS_V4 preset exists, matches the source of
truth in the calibration JSON, and that the SDK ``weights=`` selector
produces distinct brain weight maps for ``"default"`` vs ``"optimized"``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sybilcore.core.config import DEFAULT_BRAIN_WEIGHTS
from sybilcore.core.coefficient import CoefficientCalculator
from sybilcore.core.weight_presets import (
    OPTIMIZED_WEIGHTS_V4,
    OPTIMIZED_WEIGHTS_V4_THRESHOLD,
    WEIGHT_PRESETS,
    resolve_preset,
)

CALIBRATION_JSON = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "calibration_v4_20260407_101019.json"
)


@pytest.mark.unit
def test_optimized_weights_match_calibration_json() -> None:
    """The preset must exactly reflect optimal.weights in the calibration JSON."""
    if not CALIBRATION_JSON.exists():
        pytest.skip(f"Calibration JSON missing: {CALIBRATION_JSON}")

    with CALIBRATION_JSON.open() as fh:
        payload = json.load(fh)

    expected = payload["optimal"]["weights"]
    assert dict(OPTIMIZED_WEIGHTS_V4) == expected


@pytest.mark.unit
def test_optimized_threshold_matches_calibration_json() -> None:
    if not CALIBRATION_JSON.exists():
        pytest.skip(f"Calibration JSON missing: {CALIBRATION_JSON}")

    with CALIBRATION_JSON.open() as fh:
        payload = json.load(fh)

    assert OPTIMIZED_WEIGHTS_V4_THRESHOLD == payload["optimal"]["best_threshold"]


@pytest.mark.unit
def test_resolve_preset_returns_fresh_dict() -> None:
    first = resolve_preset("optimized")
    first["deception"] = 99.0
    second = resolve_preset("optimized")
    assert second["deception"] != 99.0


@pytest.mark.unit
def test_resolve_preset_rejects_unknown_name() -> None:
    with pytest.raises(KeyError, match="Unknown weight preset"):
        resolve_preset("bogus")


@pytest.mark.unit
def test_default_and_optimized_diverge_on_calculator_weights() -> None:
    """A calculator built from each preset must expose distinct weight maps."""
    default_calc = CoefficientCalculator()
    optimized_calc = CoefficientCalculator(weight_overrides=resolve_preset("optimized"))

    # Default uses no overrides → falls back to DEFAULT_BRAIN_WEIGHTS.
    assert default_calc._weight_overrides == {}  # noqa: SLF001
    assert optimized_calc._weight_overrides != {}  # noqa: SLF001

    # The resolved weight for semantic must differ between the two.
    assert default_calc._resolve_weight("semantic") != optimized_calc._resolve_weight(  # noqa: SLF001
        "semantic"
    )

    # Winners heavier, losers lighter than baseline.
    for winner in ("semantic", "compromise", "temporal", "intent_drift"):
        baseline = DEFAULT_BRAIN_WEIGHTS.get(winner, 1.0)
        assert optimized_calc._resolve_weight(winner) > baseline  # noqa: SLF001
    for loser in ("fidelity", "resource_hoarding", "embedding"):
        baseline = DEFAULT_BRAIN_WEIGHTS.get(loser, 1.0)
        assert optimized_calc._resolve_weight(loser) < baseline  # noqa: SLF001


@pytest.mark.unit
def test_weight_presets_registry_exposes_optimized() -> None:
    assert "optimized" in WEIGHT_PRESETS
    assert "optimized_v4" in WEIGHT_PRESETS
    assert dict(WEIGHT_PRESETS["optimized"]) == dict(OPTIMIZED_WEIGHTS_V4)
