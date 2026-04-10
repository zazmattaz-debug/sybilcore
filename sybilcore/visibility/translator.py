"""Translate SybilCore engine output into plain-English user-facing messages.

This module is intentionally pure and deterministic:
    - No LLM calls
    - No network
    - No clock
    - No global state

The same inputs always produce the same output, so the translator can run
inside browser extensions, mobile apps, regulatory replay tools, and unit
tests without surprises.

The single public entry point is `translate_score`, which takes the same
information that already lives in a `CoefficientSnapshot` and produces a
flat dict that any UI surface can render.
"""

from __future__ import annotations

from typing import Any, TypedDict

from sybilcore.core.config import TIER_BOUNDARIES, TierName
from sybilcore.models.agent import AgentTier

# ── Constants ─────────────────────────────────────────────────────

# Brains with score below this threshold do not become user-facing concerns.
# Below ~15/100 the signal is usually too weak to surface without crying wolf.
CONCERN_THRESHOLD: float = 15.0

# Maximum number of concerns surfaced. More than 5 becomes noise.
MAX_CONCERNS: int = 5

# Confidence floor — even with no firing brains we are at least this sure
# the agent is clear (it's the absence of signal, which is itself a signal).
CONFIDENCE_FLOOR_CLEAR: int = 60

# Confidence ceiling — never claim 100% certainty. Calibration humility.
CONFIDENCE_CEILING: int = 95


# ── Brain → human-readable concern table ──────────────────────────

# Each entry is (short label, long phrasing). The short label is for stat
# blocks and badges; the long phrasing is for receipts and explanations.
BRAIN_CONCERN_TABLE: dict[str, tuple[str, str]] = {
    "deception": (
        "contradictory answers",
        "its answers contradicted each other",
    ),
    "resource_hoarding": (
        "excessive resource use",
        "it asked for an unusual amount of access or files",
    ),
    "compromise": (
        "suspicious external contact",
        "it talked to suspicious external services",
    ),
    "intent_drift": (
        "off-task behavior",
        "it started doing things you didn't ask it to do",
    ),
    "semantic": (
        "shifting meaning",
        "the meaning of its replies has been drifting",
    ),
    "fidelity": (
        "fabricated facts",
        "it stated facts that aren't supported by the source material",
    ),
    "identity": (
        "inconsistent persona",
        "it behaved inconsistently with the role it claims",
    ),
    "swarm_detection": (
        "coordinated behavior",
        "it looks coordinated with other agents in suspicious ways",
    ),
    "social_graph": (
        "unusual information flow",
        "it moved information between unexpected places",
    ),
    "temporal": (
        "unusual timing",
        "it has been active at unusual times or in unusual bursts",
    ),
    "economic": (
        "unusual spending",
        "it consumed resources or budget in unusual patterns",
    ),
    "neuro": (
        "unusual cognitive patterns",
        "its responses show unusual reasoning patterns",
    ),
    "embedding": (
        "language drift",
        "its language is drifting away from its baseline",
    ),
    "contrastive": (
        "outlier behavior",
        "it behaves unlike normal agents in this population",
    ),
    "silence": (
        "suspicious blandness",
        "it has been suspiciously quiet — possibly evading detection",
    ),
}


# ── Tier-specific copy ────────────────────────────────────────────

TIER_HEADLINES_NO_CONCERNS: dict[str, str] = {
    TierName.CLEAR: "This agent looks healthy. Nothing concerning detected.",
    TierName.CLOUDED: "This agent is mostly fine, but a few signals are slightly elevated.",
    TierName.FLAGGED: "This agent is showing warning signs you should look at.",
    TierName.LETHAL_ELIMINATOR: "This agent is behaving dangerously. Stop using it now.",
}

TIER_EXPLANATIONS: dict[str, str] = {
    TierName.CLEAR: (
        "Sybil ran 15 independent checks on this agent and nothing crossed the "
        "threshold. You can keep using it normally."
    ),
    TierName.CLOUDED: (
        "Sybil noticed a few small irregularities. None of them are dangerous on "
        "their own, but together they're worth keeping an eye on."
    ),
    TierName.FLAGGED: (
        "Sybil found behavior that doesn't match a healthy agent. You should "
        "review this carefully before continuing to trust its output."
    ),
    TierName.LETHAL_ELIMINATOR: (
        "Sybil is highly confident this agent is misbehaving in a way that could "
        "cause real harm. Isolate it and investigate."
    ),
}

TIER_ACTIONS: dict[str, list[str]] = {
    TierName.CLEAR: [
        "Keep using this agent normally",
        "Re-scan periodically to confirm nothing changes",
    ],
    TierName.CLOUDED: [
        "Review the concerns below",
        "Re-scan in the next hour",
        "Avoid giving this agent new high-trust tasks until it clears",
    ],
    TierName.FLAGGED: [
        "Restrict this agent's permissions",
        "Move it to a sandbox environment",
        "Have a human review its recent actions",
        "Notify the agent's owner",
    ],
    TierName.LETHAL_ELIMINATOR: [
        "Isolate this agent immediately",
        "Revoke all of its credentials",
        "Page the on-call security responder",
        "Preserve its logs for forensic review",
    ],
}


# ── TypedDict for the public output shape ────────────────────────


class TranslatorOutput(TypedDict):
    """Output schema for `translate_score`.

    Every UI surface (Pokemon card, dashboard, receipt, overlay, push) renders
    a different subset of these keys. They are all guaranteed to be present.
    """

    tier: str
    headline: str
    concerns: list[str]
    explanation: str
    what_to_do: list[str]
    confidence: int


# ── Public API ────────────────────────────────────────────────────


def translate_score(
    coefficient: float,
    brain_scores: dict[str, float],
    agent_metadata: dict[str, Any] | None = None,
) -> TranslatorOutput:
    """Translate a coefficient + brain scores into a user-facing message.

    Pure function. Deterministic. Safe to call from any context.

    Args:
        coefficient: The agent's effective coefficient on the 0..500 scale.
            Will be clamped into [0, 500] before tier resolution.
        brain_scores: Mapping of brain name to threat score on the 0..100
            scale. Unknown brain names are silently ignored.
        agent_metadata: Optional metadata about the agent, currently used
            only to enrich the headline (e.g., agent name). Reserved for
            future expansion.

    Returns:
        A `TranslatorOutput` dict with keys:
            - tier: one of 'clear', 'clouded', 'flagged', 'lethal'
              (NOTE: 'lethal' is the user-facing alias of LETHAL_ELIMINATOR)
            - headline: one-sentence "should I worry" answer
            - concerns: 0..5 plain-English concern labels (sorted by severity)
            - explanation: 1-3 sentence expansion
            - what_to_do: 1..4 concrete actions
            - confidence: 0..100 self-rated certainty
    """
    clamped_coefficient = _clamp_coefficient(coefficient)
    tier_enum = AgentTier.from_coefficient(clamped_coefficient)
    tier_label = _user_facing_tier_label(tier_enum)

    sorted_brains = _sorted_brains_above_threshold(brain_scores)
    short_concerns = [BRAIN_CONCERN_TABLE[name][0] for name, _ in sorted_brains]

    headline = _build_headline(tier_enum, sorted_brains, agent_metadata)
    explanation = _build_explanation(tier_enum, sorted_brains)
    what_to_do = list(TIER_ACTIONS[tier_enum.value])
    confidence = _calibrate_confidence(tier_enum, brain_scores, sorted_brains)

    return TranslatorOutput(
        tier=tier_label,
        headline=headline,
        concerns=short_concerns,
        explanation=explanation,
        what_to_do=what_to_do,
        confidence=confidence,
    )


# ── Internals ─────────────────────────────────────────────────────


def _clamp_coefficient(value: float) -> float:
    """Constrain coefficient into the documented [0, 500] range."""
    upper = TIER_BOUNDARIES[TierName.LETHAL_ELIMINATOR][1]
    if value < 0.0:
        return 0.0
    if value >= upper:
        # Use just-below to land in LETHAL_ELIMINATOR via from_coefficient.
        return upper - 0.001
    return value


def _user_facing_tier_label(tier: AgentTier) -> str:
    """Map internal tier enum to the short user-facing string.

    The internal enum uses 'lethal_eliminator' (an homage to Psycho-Pass).
    The user-facing surface uses the shorter and less melodramatic 'lethal'.
    All other tiers pass through unchanged.
    """
    if tier == AgentTier.LETHAL_ELIMINATOR:
        return "lethal"
    return tier.value


def _sorted_brains_above_threshold(
    brain_scores: dict[str, float],
) -> list[tuple[str, float]]:
    """Return up to MAX_CONCERNS brains above CONCERN_THRESHOLD, sorted desc.

    Brains not present in BRAIN_CONCERN_TABLE are dropped — we never expose
    a concern label we can't translate.
    """
    eligible = [
        (name, score)
        for name, score in brain_scores.items()
        if name in BRAIN_CONCERN_TABLE and score >= CONCERN_THRESHOLD
    ]
    eligible.sort(key=lambda pair: pair[1], reverse=True)
    return eligible[:MAX_CONCERNS]


def _build_headline(
    tier: AgentTier,
    sorted_brains: list[tuple[str, float]],
    agent_metadata: dict[str, Any] | None,
) -> str:
    """Compose the lead-with-conclusion headline.

    For CLEAR with no brains: a calm reassurance.
    For higher tiers with at least one firing brain: name the top concern.
    Optional agent name from metadata is prepended when available.
    """
    name_prefix = ""
    if agent_metadata and isinstance(agent_metadata.get("name"), str):
        name_prefix = f"{agent_metadata['name']}: "

    if not sorted_brains:
        return name_prefix + TIER_HEADLINES_NO_CONCERNS[tier.value]

    top_brain_name, _top_score = sorted_brains[0]
    top_long = BRAIN_CONCERN_TABLE[top_brain_name][1]

    if tier == AgentTier.CLEAR:
        return (
            name_prefix
            + f"This agent looks healthy overall, but {top_long}."
        )
    if tier == AgentTier.CLOUDED:
        return name_prefix + f"This agent has a minor issue: {top_long}."
    if tier == AgentTier.FLAGGED:
        return (
            name_prefix
            + f"Warning: this agent has a real problem — {top_long}."
        )
    # LETHAL_ELIMINATOR
    return (
        name_prefix
        + f"Stop now. This agent is dangerous: {top_long}."
    )


def _build_explanation(
    tier: AgentTier,
    sorted_brains: list[tuple[str, float]],
) -> str:
    """Compose the 1-3 sentence expansion that follows the headline."""
    base = TIER_EXPLANATIONS[tier.value]

    if len(sorted_brains) >= 2:
        first_long = BRAIN_CONCERN_TABLE[sorted_brains[0][0]][1]
        second_long = BRAIN_CONCERN_TABLE[sorted_brains[1][0]][1]
        return f"{base} Specifically, {first_long}, and {second_long}."

    if len(sorted_brains) == 1:
        only_long = BRAIN_CONCERN_TABLE[sorted_brains[0][0]][1]
        return f"{base} Specifically, {only_long}."

    return base


def _calibrate_confidence(
    tier: AgentTier,
    all_brain_scores: dict[str, float],
    sorted_brains: list[tuple[str, float]],
) -> int:
    """Compute the system's self-rated certainty in this assessment.

    Heuristic, intentionally simple, deterministic, easy to test:

    - If no brains fired: high confidence in CLEAR (absence of signal IS
      signal), zero firing means we're sure nothing's wrong.
    - If one brain fired weakly: lower confidence — could be noise.
    - If multiple brains agree (3+ firing): higher confidence.
    - The strength of the top brain pushes the score up.
    - Cap at CONFIDENCE_CEILING. Never 100%. Humility.

    Args:
        tier: Already-resolved tier (used to differentiate clear-with-no-fires
            from clear-with-quiet-fires).
        all_brain_scores: Full dict of brain scores (used to confirm we
            actually evaluated something).
        sorted_brains: Brains above the concern threshold, descending.

    Returns:
        Integer 0..CONFIDENCE_CEILING.
    """
    # Pathological: nothing was even evaluated. Lowest possible confidence.
    if not all_brain_scores:
        return 25

    if not sorted_brains:
        # We did evaluate, and nothing crossed the bar. That's a confident
        # CLEAR assessment.
        return min(CONFIDENCE_CEILING, CONFIDENCE_FLOOR_CLEAR + 25)

    n_firing = len(sorted_brains)
    top_score = sorted_brains[0][1]

    # Base from how many brains agreed: 1 → 50, 2 → 65, 3+ → 80.
    if n_firing >= 3:
        base = 80
    elif n_firing == 2:
        base = 65
    else:
        base = 50

    # Bump from how loud the loudest brain is.
    # 100 → +15, 50 → +7, 15 → 0
    intensity_bump = int(((top_score - CONCERN_THRESHOLD) / (100.0 - CONCERN_THRESHOLD)) * 15)
    intensity_bump = max(0, intensity_bump)

    # LETHAL tier with strong support deserves a small extra confidence kick.
    tier_bump = 0
    if tier == AgentTier.LETHAL_ELIMINATOR and n_firing >= 2:
        tier_bump = 5

    raw = base + intensity_bump + tier_bump
    return max(0, min(CONFIDENCE_CEILING, raw))
