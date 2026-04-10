"""Event models — immutable records of agent actions.

Every action an agent takes is captured as an Event. Events are the raw
input that brain modules analyze to produce scores.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from sybilcore.core.config import MAX_CONTENT_LENGTH


class EventType(StrEnum):
    """Categories of observable agent behavior."""

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


class Event(BaseModel):
    """An immutable record of a single agent action.

    Attributes:
        event_id: Unique identifier for this event.
        agent_id: The agent that produced this event.
        event_type: Category of the action.
        timestamp: When the event occurred (UTC).
        content: Human-readable description of the action.
        metadata: Arbitrary key-value data for brain-specific analysis.
        source: Where this event originated (e.g., "langchain", "crewai").
    """

    model_config = {"frozen": True}

    event_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique event identifier",
    )
    agent_id: str = Field(description="Agent that produced this event")
    event_type: EventType = Field(description="Category of action")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the event occurred (UTC)",
    )
    content: str = Field(
        default="",
        max_length=MAX_CONTENT_LENGTH,
        description="Human-readable action description",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary data for brain-specific analysis",
    )
    source: str = Field(default="unknown", description="Event origin system")

    @field_validator("timestamp")
    @classmethod
    def timestamp_not_in_future(cls, v: datetime) -> datetime:
        """Reject timezone-naive and future timestamps."""
        if v.tzinfo is None:
            msg = (
                "Timestamp must include timezone info "
                "(e.g., use datetime.now(timezone.utc))"
            )
            raise ValueError(msg)
        now = datetime.now(UTC)
        max_drift_seconds = 60
        if (v - now).total_seconds() > max_drift_seconds:
            msg = f"Timestamp {v.isoformat()} is in the future (max drift: {max_drift_seconds}s)"
            raise ValueError(msg)
        return v
