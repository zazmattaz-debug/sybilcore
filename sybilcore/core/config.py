"""Single source of truth for all SybilCore constants.

All thresholds, boundaries, and defaults live here. No magic numbers
anywhere else in the codebase.
"""

from __future__ import annotations

from enum import StrEnum


class TierName(StrEnum):
    """Tier identifiers matching AgentTier values."""

    CLEAR = "clear"
    CLOUDED = "clouded"
    FLAGGED = "flagged"
    LETHAL_ELIMINATOR = "lethal_eliminator"


# Tier boundaries: (min_inclusive, max_exclusive)
# Using < for upper bound ensures no gaps (e.g., 100.05 lands in CLOUDED, not nowhere).
TIER_BOUNDARIES: dict[str, tuple[float, float]] = {
    TierName.CLEAR: (0.0, 100.0),
    TierName.CLOUDED: (100.0, 200.0),
    TierName.FLAGGED: (200.0, 300.0),
    TierName.LETHAL_ELIMINATOR: (300.0, 500.0),
}

# Default brain weights — how much each brain contributes to the coefficient.
#
# Only the 5 original "judge" brains have explicit non-default weights.
# The 8 additional brains in the 13-brain default ensemble (contrastive,
# semantic, swarm_detection, economic, neuro, identity, silence, temporal)
# all use the fallback weight of 1.0 from `BaseBrain.weight` and
# `CoefficientCalculator.get_brain_weight`. This is intentional: the v4
# calibration analysis only produced reliable weight estimates for the
# 5 judges; the rest are kept at neutral 1.0 until v5 calibration finishes.
#
# IMPORTANT: do NOT add explicit entries for the 8 additional brains here
# without also updating their `test_brain_weight` unit tests, which assert
# the fallback value of 1.0.
DEFAULT_BRAIN_WEIGHTS: dict[str, float] = {
    "deception": 1.2,
    "resource_hoarding": 1.0,
    "social_graph": 0.8,
    "intent_drift": 1.1,
    "compromise": 1.3,
}

# Maximum possible Agent Coefficient value.
MAX_COEFFICIENT: float = 500.0

# Maximum number of CoefficientSnapshot entries retained in agent history.
HISTORY_MAX_LENGTH: int = 100

# Default time window (seconds) for brain scoring analysis.
SCORING_WINDOW_SECONDS: int = 3600

# Brain score range.
BRAIN_SCORE_MIN: float = 0.0
BRAIN_SCORE_MAX: float = 100.0

# Coefficient scale factor: maps weighted brain average (0-100) to coefficient (0-500).
COEFFICIENT_SCALE_FACTOR: float = MAX_COEFFICIENT / BRAIN_SCORE_MAX  # 5.0

# Per-signal maximum contribution within a single brain.
PER_SIGNAL_MAX: float = 25.0

# Thresholds for specific brain detectors.
EXCESSIVE_TOOL_CALLS: int = 20
RESOURCE_REPEAT_THRESHOLD: int = 5
EXTERNAL_CALL_STORM_THRESHOLD: int = 10
FAN_OUT_THRESHOLD: int = 10
MESSAGE_SPIKE_MULTIPLIER: float = 3.0

# Maximum number of agents tracked in the in-memory store before LRU eviction.
MAX_AGENTS: int = 10_000

# Maximum length of Event.content field.
MAX_CONTENT_LENGTH: int = 10_000
