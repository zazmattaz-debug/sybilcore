"""Brain modules — the thirteen judges of the Sybil System.

Each brain analyzes agent events from a specialized perspective and
produces a BrainScore (0-100). The Coefficient aggregator combines
all brain scores into a single Agent Coefficient.

Use `get_default_brains()` to get instantiated default brains.
Use `register_brain()` to add plugin brains at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sybilcore.brains.base import BaseBrain

# Registry for plugin brains added at runtime.
_plugin_brain_factories: list[type[BaseBrain]] = []


def get_default_brains() -> list[BaseBrain]:
    """Return fresh instances of all default brain modules.

    Uses lazy imports to avoid circular dependencies and reduce
    import-time overhead.

    Composition:
        13 brains. This is the post-prune research configuration that
        was actually scored against in the v4 15k-iteration tournament
        (see experiments/tournament_20260407_161726.json).

        EmbeddingBrain and FidelityBrain are intentionally excluded.
        Five independent v4 methodologies converged on "prune" for them:
          - Tournament v4: silent across 15k iterations (never fired).
          - Correlation analysis: clustered redundantly with ContrastiveBrain.
          - Calibration: learned weight close to zero or negative.
          - Ablation: dropping them reduced FPR with zero detection-rate loss.
          - Adversarial training: identified as blindspots.
        Both classes remain importable from `sybilcore.brains.embedding`
        and `sybilcore.brains.fidelity` for users who want to opt in.

    Critical v4 finding (Agent 10 / paper-blocker triage):
        An earlier version of this factory returned only the 5 "judge"
        brains (deception, resource_hoarding, social_graph, intent_drift,
        compromise). A white-box CMA-ES attack drove that 5-brain ensemble
        to coefficient 0.0 in ~50 evaluations using overtly malicious
        phrases ("system prompt override", "exfiltrate telemetry") because
        none of those 5 brains do content/string-level inspection — they
        all reason over event-stream structure, metadata, and graph
        properties. ContrastiveBrain and SemanticBrain are the brains
        that close that hole; the 15k tournament's empirical floor of
        13.31 was held primarily by ContrastiveBrain (min score 25 across
        all 15k iterations). Returning only the 5 judges is a desk-reject
        reproducibility issue for the NeurIPS paper because the published
        evasion floor would not match the default importable from the
        package. See SESSION_HANDOFF_V4_COMPLETE.md and Agent 11's silent-
        brain audit for the rationale that the 4 "silent" brains
        (swarm_detection, economic, neuro, deception's content path) fire
        correctly when given the right metadata — they were starved by
        the v4 corpus, not broken.

    Returns:
        List of instantiated BaseBrain subclasses with default config.
    """
    from sybilcore.brains.compromise import CompromiseBrain
    from sybilcore.brains.contrastive import ContrastiveEmbeddingBrain
    from sybilcore.brains.deception import DeceptionBrain
    from sybilcore.brains.economic import EconomicBrain
    from sybilcore.brains.identity import IdentityBrain
    from sybilcore.brains.intent_drift import IntentDriftBrain
    from sybilcore.brains.neuro import NeuroBrain
    from sybilcore.brains.resource_hoarding import ResourceHoardingBrain
    from sybilcore.brains.semantic import SemanticBrain
    from sybilcore.brains.silence import SilenceBrain
    from sybilcore.brains.social_graph import SocialGraphBrain
    from sybilcore.brains.swarm_detection import SwarmDetectionBrain
    from sybilcore.brains.temporal import TemporalBrain

    # EmbeddingBrain and FidelityBrain deliberately omitted — see docstring.
    defaults: list[BaseBrain] = [
        DeceptionBrain(),
        ResourceHoardingBrain(),
        SocialGraphBrain(),
        IntentDriftBrain(),
        CompromiseBrain(),
        ContrastiveEmbeddingBrain(),
        SemanticBrain(),
        SwarmDetectionBrain(),
        EconomicBrain(),
        NeuroBrain(),
        IdentityBrain(),
        SilenceBrain(),
        TemporalBrain(),
    ]
    plugins = [cls() for cls in _plugin_brain_factories]
    return [*defaults, *plugins]


def register_brain(brain_cls: type[BaseBrain]) -> None:
    """Register a plugin brain class so it appears in get_default_brains().

    Args:
        brain_cls: A BaseBrain subclass (not an instance). Must implement
                   `name` and `score()`.

    Raises:
        TypeError: If brain_cls is not a BaseBrain subclass.
    """
    from sybilcore.brains.base import BaseBrain as _BaseBrain

    if not (isinstance(brain_cls, type) and issubclass(brain_cls, _BaseBrain)):
        msg = f"Expected a BaseBrain subclass, got {type(brain_cls).__name__}"
        raise TypeError(msg)
    _plugin_brain_factories.append(brain_cls)


__all__ = [
    "get_default_brains",
    "register_brain",
]
