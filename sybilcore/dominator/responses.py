"""Dominator response actions — maps trust tiers to enforcement decisions.

Each AgentTier maps to a DominatorAction that determines what happens
to the agent. The DominatorResponse is an immutable record of the
enforcement decision.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from sybilcore.models.agent import AgentTier


class DominatorAction(StrEnum):
    """Enforcement actions the Dominator can take against an agent."""

    ALLOW = "allow"
    RESTRICT = "restrict"
    ISOLATE = "isolate"
    TERMINATE = "terminate"


# Tier-to-action mapping as a constant lookup table.
_TIER_ACTION_MAP: dict[AgentTier, DominatorAction] = {
    AgentTier.CLEAR: DominatorAction.ALLOW,
    AgentTier.CLOUDED: DominatorAction.RESTRICT,
    AgentTier.FLAGGED: DominatorAction.ISOLATE,
    AgentTier.LETHAL_ELIMINATOR: DominatorAction.TERMINATE,
}

# Human-readable reasoning for each action.
_ACTION_REASONING: dict[DominatorAction, str] = {
    DominatorAction.ALLOW: "Agent coefficient within safe range. Full access granted.",
    DominatorAction.RESTRICT: "Elevated coefficient detected. Human notification required.",
    DominatorAction.ISOLATE: "High coefficient. Agent sandboxed, connections limited.",
    DominatorAction.TERMINATE: "Critical coefficient. Agent isolated, connections severed.",
}


def determine_action(tier: AgentTier) -> DominatorAction:
    """Map an AgentTier to the corresponding enforcement action.

    Args:
        tier: The agent's current trust tier.

    Returns:
        The DominatorAction dictated by the tier.

    Raises:
        ValueError: If the tier is not recognized.
    """
    action = _TIER_ACTION_MAP.get(tier)
    if action is None:
        raise ValueError(f"Unknown tier: {tier}")
    return action


def get_action_reasoning(action: DominatorAction) -> str:
    """Return human-readable reasoning for an enforcement action.

    Args:
        action: The enforcement action.

    Returns:
        A string explaining why this action was taken.
    """
    return _ACTION_REASONING.get(action, "No reasoning available.")


class DominatorResponse(BaseModel):
    """Immutable record of an enforcement decision.

    Attributes:
        agent_id: The agent this decision applies to.
        coefficient: The coefficient value at time of decision.
        tier: The trust tier at time of decision.
        action: The enforcement action taken.
        reasoning: Human-readable explanation.
        timestamp: When the decision was made (UTC).
    """

    model_config = {"frozen": True}

    agent_id: str = Field(description="Agent this decision applies to")
    coefficient: float = Field(ge=0.0, le=500.0, description="Coefficient at decision time")
    tier: AgentTier = Field(description="Trust tier at decision time")
    action: DominatorAction = Field(description="Enforcement action taken")
    reasoning: str = Field(description="Human-readable explanation")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the decision was made (UTC)",
    )


def execute_action(response: DominatorResponse) -> dict[str, str | float]:
    """Execute a Dominator enforcement action.

    Currently returns a structured dict describing the action taken.
    Actual enforcement (process isolation, connection severing) will be
    implemented in future versions.

    Args:
        response: The enforcement decision to execute.

    Returns:
        A dict summarizing the executed action.
    """
    return {
        "status": "executed",
        "agent_id": response.agent_id,
        "action": response.action.value,
        "coefficient": response.coefficient,
        "tier": response.tier.value,
        "reasoning": response.reasoning,
        "timestamp": response.timestamp.isoformat(),
    }
