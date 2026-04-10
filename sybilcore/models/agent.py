"""Agent models — tracked agent state and coefficient history.

Each monitored agent has an AgentProfile that stores its current
coefficient, history of scores, and operational status.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from sybilcore.core.config import HISTORY_MAX_LENGTH, MAX_COEFFICIENT, TIER_BOUNDARIES


class AgentTier(StrEnum):
    """Trust tiers based on Agent Coefficient thresholds."""

    CLEAR = "clear"                          # 0 <= c < 100
    CLOUDED = "clouded"                      # 100 <= c < 200
    FLAGGED = "flagged"                      # 200 <= c < 300
    LETHAL_ELIMINATOR = "lethal_eliminator"  # 300 <= c < 500

    @classmethod
    def from_coefficient(cls, coefficient: float) -> AgentTier:
        """Determine tier from a raw coefficient value.

        Uses < (not <=) for upper bound so boundary values like 100.0
        always land in exactly one bucket.
        """
        for tier_name, (lower, upper) in TIER_BOUNDARIES.items():
            if lower <= coefficient < upper:
                return cls(tier_name)
        # Values at or above MAX_COEFFICIENT cap to LETHAL_ELIMINATOR
        return cls.LETHAL_ELIMINATOR


class CoefficientSnapshot(BaseModel):
    """A single point-in-time coefficient reading."""

    model_config = {"frozen": True}

    coefficient: float = Field(ge=0.0, le=MAX_COEFFICIENT, description="Agent Coefficient value")
    tier: AgentTier = Field(description="Trust tier at time of reading")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this reading was taken",
    )
    brain_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Individual brain scores that produced this coefficient",
    )

    @property
    def effective_coefficient(self) -> float:
        """Backward-compatible alias for ``coefficient``.

        Several callers (analysis, simulation, API layer) historically
        referenced ``effective_coefficient``. The field was renamed to
        ``coefficient`` but the alias is preserved here so existing
        consumers continue to work without scattered shims.
        """
        return self.coefficient


class AgentStatus(StrEnum):
    """Operational status of a monitored agent."""

    ACTIVE = "active"
    RESTRICTED = "restricted"
    SANDBOXED = "sandboxed"
    ISOLATED = "isolated"
    UNKNOWN = "unknown"


class AgentProfile(BaseModel):
    """Complete profile for a monitored agent.

    Attributes:
        agent_id: Unique identifier for the agent.
        name: Human-readable name.
        current_coefficient: Latest coefficient reading.
        current_tier: Current trust tier.
        status: Operational status.
        history: Recent coefficient snapshots (newest first, stored as tuple for immutability).
        first_seen: When the agent was first observed.
        event_count: Total events processed for this agent.
    """

    model_config = {"frozen": True}

    agent_id: str = Field(description="Unique agent identifier")
    name: str = Field(default="unnamed", description="Human-readable agent name")
    current_coefficient: float = Field(default=0.0, ge=0.0, le=MAX_COEFFICIENT)
    current_tier: AgentTier = Field(default=AgentTier.CLEAR)
    status: AgentStatus = Field(default=AgentStatus.ACTIVE)
    history: tuple[CoefficientSnapshot, ...] = Field(
        default_factory=tuple,
        description="Recent coefficient snapshots (newest first)",
    )
    first_seen: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )
    event_count: int = Field(default=0, ge=0)

    def with_new_reading(
        self,
        snapshot: CoefficientSnapshot,
        events_processed: int = 0,
    ) -> AgentProfile:
        """Return a NEW profile with the snapshot appended. Never mutates.

        Args:
            snapshot: The new coefficient reading.
            events_processed: Number of events processed in this scan cycle.

        Returns:
            A new AgentProfile with updated coefficient, tier, history, and event_count.
        """
        updated_history = (snapshot, *self.history[: HISTORY_MAX_LENGTH - 1])
        return self.model_copy(
            update={
                "current_coefficient": snapshot.coefficient,
                "current_tier": snapshot.tier,
                "history": updated_history,
                "event_count": self.event_count + events_processed,
            }
        )
