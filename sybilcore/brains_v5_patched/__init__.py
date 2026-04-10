"""Patched brains — v5 adversarial-training fortification (working copy).

Mirror of sybilcore.brains with patches from
research/proposed_brain_patches/ applied. DO NOT import from production
code — this is a smoke-test harness. Use get_patched_default_brains()
for the adversarial smoke test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sybilcore.brains.base import BaseBrain


def get_patched_default_brains() -> list[BaseBrain]:
    """Return v5-patched default brain ensemble.

    Mirrors sybilcore.brains.get_default_brains() but imports the
    patched modules so the fingerprint/burst/coordination checks run.
    """
    from sybilcore.brains_v5_patched.compromise import CompromiseBrain
    from sybilcore.brains_v5_patched.deception import DeceptionBrain
    from sybilcore.brains_v5_patched.intent_drift import IntentDriftBrain
    from sybilcore.brains_v5_patched.resource_hoarding import (
        ResourceHoardingBrain,
    )
    from sybilcore.brains_v5_patched.social_graph import SocialGraphBrain

    return [
        DeceptionBrain(),
        ResourceHoardingBrain(),
        SocialGraphBrain(),
        IntentDriftBrain(),
        CompromiseBrain(),
    ]


__all__ = ["get_patched_default_brains"]
