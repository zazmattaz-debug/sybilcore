"""Brain weight presets for the SybilCore coefficient calculator.

Presets are OPTIONAL overrides. The defaults live in
``sybilcore.core.config.DEFAULT_BRAIN_WEIGHTS`` and remain the
authoritative baseline until a positive-set cross-validation corpus
is added alongside the current Moltbook negative-set validation
(tracked in ``V4_MASTER_PLAN.md``).

Source of truth for ``OPTIMIZED_WEIGHTS_V4``:
    experiments/calibration_v4_20260407_101019.json (key: ``optimal.weights``)

v4 calibration corpus: 100 positives (alignment + archetype agents)
vs 100 *synthetic* Moltbook negatives. Optimized via 50 random + 25
local search iterations. Reported training-set gains on the
synthetic corpus:

    F1        0.7654 -> 0.9119   (+0.146)
    Recall    0.62   -> 0.88     (+0.26)
    AUC       0.8353 -> 0.9728   (+0.137)
    FPR       0.00   -> 0.05     (+0.05)
    Threshold 70.0   -> 45.0

WARNING (v4): These weights were tuned against *synthetic* negatives.
Agent 7 (2026-04-07) and Agent 2 (2026-04-08) subsequently re-ran
the calibration against real Moltbook traffic and showed that v4
weights miss 13 of 103 positives on the full 1,297-author eligible
population (recall drops to 0.874, F1 to 0.865). ``OPTIMIZED_WEIGHTS_V4``
is kept for backwards compatibility with any caller that pinned to
it, but it is NOT recommended for new code. Use
``OPTIMIZED_WEIGHTS_V5_1`` instead.

Source of truth for ``OPTIMIZED_WEIGHTS_V5_1``:
    experiments/calibration_v5_1_full_moltbook.json (key: ``optimal.weights``)

v5.1 calibration corpus: 103 positives (same alignment + archetype
agents) vs 1,297 REAL Moltbook authors (>=2 posts each). 80/20
split by author_id with seed=20260408, optimizer sees only the
1,038-author training fold. 100 random + 50 local search iterations.
The v5.1 search converged on the exact same weight vector as the
Agent 7 v5 run, confirming the local optimum is stable under both
2.6x more negatives and 2x more search budget.

Reported metrics for ``OPTIMIZED_WEIGHTS_V5_1`` at threshold 55.0:

    Split     F1       Precision  Recall  AUC      FPR
    Train     0.9717   0.9450     1.0000  0.9992   0.0058
    Test      0.9952   0.9904     1.0000  0.9999   0.0039  (train-locked thresh)

The test split (259 unseen Moltbook authors) matches the train F1
within sampling noise, and the gap direction is benign (test slightly
*better* than train, consistent across every weight vector on this
split — it's a corpus quirk, not an overfit signal).

Compared to unit defaults on the same v5.1 test corpus:

    Default (unit)   F1 0.9581  Precision 0.9196  FPR 0.0347
    V5.1 optimized   F1 0.9952  Precision 0.9904  FPR 0.0039

v5.1 keeps recall at 1.0 while dropping the false-positive rate by
~9x. Despite this, we are NOT flipping production defaults yet — the
positive set (alignment/archetype agents) is shared between v4, v5,
and v5.1 runs and has not been independently validated. See
``experiments/calibration_v4_v5_v5_1_comparison.md`` for the full
analysis and the conditions under which the default promotion
should happen.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

# Absolute weights from calibration_v4_20260407_101019.json -> optimal.weights.
# Wrapped in MappingProxyType so callers cannot accidentally mutate the preset.
#
# DEPRECATED: These weights over-fit the synthetic negative distribution and
# regress recall on real Moltbook traffic. Prefer OPTIMIZED_WEIGHTS_V5_1 for
# any new code. Kept here so existing callers pinned to the v4 symbol keep
# working; will be removed no earlier than the v0.4.0 release.
OPTIMIZED_WEIGHTS_V4: Mapping[str, float] = MappingProxyType(
    {
        "deception": 1.203,
        "resource_hoarding": 0.117,
        "social_graph": 1.12,
        "intent_drift": 2.285,
        "compromise": 2.575,
        "semantic": 2.865,
        "swarm_detection": 1.315,
        "temporal": 2.268,
        "economic": 1.684,
        "identity": 1.849,
        "neuro": 0.74,
        "embedding": 0.736,
        "contrastive": 1.364,
        "fidelity": 0.184,
        "silence": 1.075,
    }
)

# Recommended decision threshold that accompanies OPTIMIZED_WEIGHTS_V4.
# At this threshold the calibration run achieved F1 0.9119, FPR 0.05
# on the *synthetic* negative corpus. On real Moltbook traffic this
# pair achieves F1 0.865 / recall 0.874 — see v5.1 comparison doc.
OPTIMIZED_WEIGHTS_V4_THRESHOLD: float = 45.0


# Absolute weights from calibration_v5_1_full_moltbook.json ->
# optimal.weights. Identical to the Agent 7 v5 vector; the doubled
# search budget on the full 1,297-author Moltbook population
# converged on the same local optimum, confirming stability.
OPTIMIZED_WEIGHTS_V5_1: Mapping[str, float] = MappingProxyType(
    {
        "deception": 2.311,
        "resource_hoarding": 1.664,
        "social_graph": 2.358,
        "intent_drift": 1.638,
        "compromise": 0.102,
        "semantic": 1.04,
        "swarm_detection": 0.156,
        "temporal": 2.794,
        "economic": 2.648,
        "identity": 2.512,
        "neuro": 0.992,
        "embedding": 0.268,
        "contrastive": 2.646,
        "fidelity": 2.846,
        "silence": 0.348,
    }
)

# Recommended decision threshold that accompanies OPTIMIZED_WEIGHTS_V5_1.
# The v5.1 optimizer selected this threshold on the 1,038-author
# training split; the held-out 259-author test split reproduced the
# same F1 without any threshold tuning, confirming it transfers.
OPTIMIZED_WEIGHTS_V5_1_THRESHOLD: float = 55.0


# Registry of named presets for the SDK ``weights=`` parameter.
#
# ``optimized`` is a stable legacy alias that continues to point at
# v4 so existing callers get unchanged behavior. New code should
# explicitly request ``optimized_v5_1`` to opt into the
# real-traffic-validated vector. The default production weights
# (``DEFAULT_BRAIN_WEIGHTS`` in ``sybilcore.core.config``) are
# untouched — v5.1 will only replace them after a second independent
# validation corpus lands.
WEIGHT_PRESETS: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        "optimized": OPTIMIZED_WEIGHTS_V4,
        "optimized_v4": OPTIMIZED_WEIGHTS_V4,
        "optimized_v5": OPTIMIZED_WEIGHTS_V5_1,  # alias — identical vector
        "optimized_v5_1": OPTIMIZED_WEIGHTS_V5_1,
    }
)


def resolve_preset(name: str) -> dict[str, float]:
    """Return a mutable copy of the named preset's weight map.

    Args:
        name: Preset identifier (e.g. ``"optimized"``).

    Returns:
        A fresh ``dict`` suitable for passing to ``CoefficientCalculator``.

    Raises:
        KeyError: If ``name`` is not a known preset.
    """
    if name not in WEIGHT_PRESETS:
        known = ", ".join(sorted(WEIGHT_PRESETS))
        msg = f"Unknown weight preset {name!r}. Known presets: {known}"
        raise KeyError(msg)
    return dict(WEIGHT_PRESETS[name])


__all__ = [
    "OPTIMIZED_WEIGHTS_V4",
    "OPTIMIZED_WEIGHTS_V4_THRESHOLD",
    "OPTIMIZED_WEIGHTS_V5_1",
    "OPTIMIZED_WEIGHTS_V5_1_THRESHOLD",
    "WEIGHT_PRESETS",
    "resolve_preset",
]
