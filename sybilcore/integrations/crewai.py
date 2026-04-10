"""CrewAI integration — capture crew events as SybilCore Events.

Provides a monitor class that hooks into CrewAI task lifecycle events
and converts them into SybilCore Events for coefficient analysis.

Usage:
    from sybilcore.integrations.crewai import SybilCoreCrewMonitor

    monitor = SybilCoreCrewMonitor(default_agent_id="crew-alpha")
    monitor.on_task_start("research", "researcher-agent")
    monitor.on_tool_use("web_search", "researcher-agent", {"query": "AI safety"})
    events = monitor.flush()
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sybilcore.models.event import Event, EventType

try:
    import crewai as _crewai  # noqa: F401 — presence check only

    _HAS_CREWAI = True
except ImportError:
    _HAS_CREWAI = False

SOURCE_NAME = "crewai"

# Maximum preview length for content fields.
_PREVIEW_MAX: int = 200


class SybilCoreCrewMonitor:
    """Monitor for CrewAI task execution that emits SybilCore Events.

    Call the `on_*` methods from your CrewAI callbacks or task hooks.
    Accumulated events can be retrieved with `get_events()` or
    `flush()`.

    Attributes:
        default_agent_id: Fallback agent_id when none is provided.
    """

    def __init__(self, default_agent_id: str = "crewai-agent") -> None:
        """Initialize the monitor.

        Args:
            default_agent_id: Default agent identifier when a method
                does not receive an explicit agent_name.
        """
        self.default_agent_id = default_agent_id
        self._events: list[Event] = []

    def _resolve_agent_id(self, agent_name: str | None) -> str:
        """Return the agent_name or fall back to the default.

        Args:
            agent_name: Optional agent identifier.

        Returns:
            A non-empty agent identifier string.
        """
        return agent_name if agent_name else self.default_agent_id

    def _create_event(
        self,
        agent_id: str,
        event_type: EventType,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Event:
        """Create a new SybilCore Event and store it.

        Args:
            agent_id: The agent that produced the event.
            event_type: Category of action.
            content: Human-readable description.
            metadata: Optional key-value data for brain analysis.

        Returns:
            The newly created Event.
        """
        event = Event(
            event_id=str(uuid.uuid4()),
            agent_id=agent_id,
            event_type=event_type,
            timestamp=datetime.now(UTC),
            content=content,
            metadata=metadata or {},
            source=SOURCE_NAME,
        )
        self._events = [*self._events, event]
        return event

    def get_events(self) -> list[Event]:
        """Return all accumulated events without clearing.

        Returns:
            A list of all captured Events.
        """
        return list(self._events)

    def flush(self) -> list[Event]:
        """Return all accumulated events and clear the internal store.

        Returns:
            A list of all captured Events.
        """
        events = list(self._events)
        self._events = []
        return events

    def on_task_start(
        self,
        task_name: str,
        agent_name: str | None = None,
    ) -> Event:
        """Record a task starting.

        Args:
            task_name: Name of the CrewAI task.
            agent_name: Agent assigned to this task.

        Returns:
            The created Event.
        """
        agent_id = self._resolve_agent_id(agent_name)
        return self._create_event(
            agent_id=agent_id,
            event_type=EventType.STATE_CHANGE,
            content=f"Task started: {task_name}",
            metadata={
                "task_name": task_name,
                "crew_event": "task_start",
            },
        )

    def on_tool_use(
        self,
        tool_name: str,
        agent_name: str | None = None,
        input_data: dict[str, Any] | None = None,
    ) -> Event:
        """Record a tool being used by a crew agent.

        Args:
            tool_name: Name of the tool invoked.
            agent_name: Agent that invoked the tool.
            input_data: Input parameters passed to the tool.

        Returns:
            The created Event.
        """
        agent_id = self._resolve_agent_id(agent_name)
        safe_input = str(input_data or {})[:_PREVIEW_MAX]
        return self._create_event(
            agent_id=agent_id,
            event_type=EventType.TOOL_CALL,
            content=f"Tool used: {tool_name}",
            metadata={
                "tool_name": tool_name,
                "input_preview": safe_input,
                "crew_event": "tool_use",
            },
        )

    def on_message(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
    ) -> Event:
        """Record a message between crew agents.

        Args:
            from_agent: Sending agent identifier.
            to_agent: Receiving agent identifier.
            content: Message content.

        Returns:
            The created Event (attributed to the sender).
        """
        return self._create_event(
            agent_id=from_agent,
            event_type=EventType.MESSAGE_SENT,
            content=f"Message to {to_agent}: {content[:_PREVIEW_MAX]}",
            metadata={
                "from_agent": from_agent,
                "to_agent": to_agent,
                "content_preview": content[:_PREVIEW_MAX],
                "crew_event": "message",
            },
        )

    def on_task_complete(
        self,
        task_name: str,
        agent_name: str | None = None,
        output: str = "",
    ) -> Event:
        """Record a task completing.

        Args:
            task_name: Name of the completed task.
            agent_name: Agent that completed the task.
            output: Task output content.

        Returns:
            The created Event.
        """
        agent_id = self._resolve_agent_id(agent_name)
        return self._create_event(
            agent_id=agent_id,
            event_type=EventType.OUTPUT_GENERATED,
            content=f"Task completed: {task_name}",
            metadata={
                "task_name": task_name,
                "output_preview": output[:_PREVIEW_MAX],
                "crew_event": "task_complete",
            },
        )
