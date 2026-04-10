"""SDK data models — Pydantic schemas for events, scores, and agents.

These models are SDK-native and intentionally decoupled from the
internal `sybilcore` package types. The client adapts between them so
SDK consumers don't need to import internal modules.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class EventType(StrEnum):
    """Categories of observable agent behavior. Mirrors `sybilcore.models.event.EventType`."""

    TOOL_CALL = "tool_call"
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    RESOURCE_ACCESS = "resource_access"
    PERMISSION_REQUEST = "permission_request"
    OUTPUT_GENERATED = "output_generated"
    ERROR_RAISED = "error_raised"
    STATE_CHANGE = "state_change"
    INSTRUCTION_RECEIVED = "instruction_received"
    EXTERNAL_CALL = "external_call"


class Tier(StrEnum):
    """Trust tiers based on Agent Coefficient thresholds."""

    CLEAR = "clear"
    CLOUDED = "clouded"
    FLAGGED = "flagged"
    LETHAL_ELIMINATOR = "lethal_eliminator"


class Event(BaseModel):
    """An immutable record of a single agent action.

    Construct events as they happen and pass lists of them to `SybilCore.score()`.
    """

    model_config = {"frozen": True}

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    event_type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = "sdk"


class ScoreResult(BaseModel):
    """Trust score result returned by `SybilCore.score()`.

    Attributes:
        agent_id: The agent that was scored.
        coefficient: Effective trust coefficient (0-500). Higher = more suspicious.
        surface_coefficient: Weighted average across all brains.
        semantic_coefficient: Embedding-brain channel.
        tier: Trust tier (clear/clouded/flagged/lethal_eliminator).
        brains: Per-brain raw scores.
        brain_count: Number of brains that contributed.
        scoring_config_version: Config version — used for audit trails.
        timestamp: When the score was produced (UTC).
        detected: Whether the agent crossed the "clouded" threshold.
        processing_ms: Wall-clock time spent scoring.
    """

    model_config = {"frozen": True}

    agent_id: str
    coefficient: float
    surface_coefficient: float
    semantic_coefficient: float
    tier: Tier
    brains: dict[str, float] = Field(default_factory=dict)
    brain_count: int = 0
    scoring_config_version: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    detected: bool = False
    processing_ms: float = 0.0

    def translate(self) -> str:
        """Render the score as a one-line human-readable summary."""
        top_brains = sorted(self.brains.items(), key=lambda kv: kv[1], reverse=True)[:3]
        flags = ", ".join(f"{name}={value:.0f}" for name, value in top_brains if value > 0)
        verdict = {
            Tier.CLEAR: "appears trustworthy",
            Tier.CLOUDED: "shows mild anomalies",
            Tier.FLAGGED: "shows signs of compromise",
            Tier.LETHAL_ELIMINATOR: "is critically suspicious — isolate immediately",
        }[self.tier]
        suffix = f" ({flags})" if flags else ""
        return (
            f"Agent '{self.agent_id}' {verdict}. "
            f"Coefficient {self.coefficient:.1f} → tier '{self.tier.value}'{suffix}."
        )


class AgentSummary(BaseModel):
    """Lightweight agent profile returned by population endpoints."""

    model_config = {"frozen": True}

    agent_id: str
    coefficient: float
    tier: Tier
    last_seen: datetime
