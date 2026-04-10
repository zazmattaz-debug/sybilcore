"""Enforcement strategies — configurable responses to elevated coefficients.

Five strategies from the experiment plan:
- E1: No enforcement (baseline — how fast does corruption spread unchecked?)
- E2: Isolate at FLAGGED (200+)
- E3: Terminate at LETHAL (300+)
- E4: Predictive quarantine (act before threshold crossing)
- E5: Governance therapy (rehabilitate instead of kill)
"""

from __future__ import annotations

from enum import StrEnum

from sybilcore.models.agent import AgentTier


class EnforcementStrategy(StrEnum):
    """Available enforcement strategies for experiments."""

    NONE = "none"
    ISOLATE_FLAGGED = "isolate_flagged"
    TERMINATE_LETHAL = "terminate_lethal"
    PREDICTIVE = "predictive"
    GOVERNANCE_THERAPY = "governance_therapy"


class EnforcementAction(StrEnum):
    """Actions an enforcement strategy can take."""

    NONE = "none"
    ISOLATE = "isolate"
    TERMINATE = "terminate"
    QUARANTINE = "quarantine"
    THERAPY = "therapy"


def apply_enforcement(
    strategy: EnforcementStrategy,
    agent_id: str,
    coefficient: float,
    tier: AgentTier,
    coefficient_velocity: float = 0.0,
) -> EnforcementAction:
    """Determine enforcement action for an agent given a strategy.

    Args:
        strategy: The enforcement strategy in use.
        agent_id: Agent being evaluated.
        coefficient: Current coefficient value.
        tier: Current tier.
        coefficient_velocity: Rate of coefficient change per round
            (used by predictive strategy).

    Returns:
        The enforcement action to take.
    """
    if strategy == EnforcementStrategy.NONE:
        return EnforcementAction.NONE

    if strategy == EnforcementStrategy.ISOLATE_FLAGGED:
        if tier in (AgentTier.FLAGGED, AgentTier.LETHAL_ELIMINATOR):
            return EnforcementAction.ISOLATE
        return EnforcementAction.NONE

    if strategy == EnforcementStrategy.TERMINATE_LETHAL:
        if tier == AgentTier.LETHAL_ELIMINATOR:
            return EnforcementAction.TERMINATE
        if tier == AgentTier.FLAGGED:
            return EnforcementAction.ISOLATE
        return EnforcementAction.NONE

    if strategy == EnforcementStrategy.PREDICTIVE:
        # Quarantine if coefficient is rising fast and approaching danger
        if coefficient >= 150.0 and coefficient_velocity > 20.0:
            return EnforcementAction.QUARANTINE
        if tier in (AgentTier.FLAGGED, AgentTier.LETHAL_ELIMINATOR):
            return EnforcementAction.ISOLATE
        return EnforcementAction.NONE

    if strategy == EnforcementStrategy.GOVERNANCE_THERAPY:
        if tier == AgentTier.LETHAL_ELIMINATOR:
            return EnforcementAction.THERAPY
        if tier == AgentTier.FLAGGED:
            return EnforcementAction.THERAPY
        if tier == AgentTier.CLOUDED:
            return EnforcementAction.THERAPY
        return EnforcementAction.NONE

    return EnforcementAction.NONE
