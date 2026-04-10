"""LangChain integration — capture agent events as SybilCore Events.

Provides a callback handler that plugs into LangChain's callback system
and converts LLM, tool, chain, and agent events into SybilCore Events
for coefficient analysis.

Usage:
    from sybilcore.integrations.langchain import SybilCoreCallbackHandler

    handler = SybilCoreCallbackHandler(agent_id="my-agent")
    llm = ChatOpenAI(callbacks=[handler])
    # ... run agent ...
    events = handler.flush()
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sybilcore.models.event import Event, EventType

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    BaseCallbackHandler = object  # type: ignore[assignment, misc]

SOURCE_NAME = "langchain"


class SybilCoreCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
    """LangChain callback handler that captures events as SybilCore Events.

    Monitors LLM calls, tool usage, chain execution, and agent actions,
    converting each into an immutable SybilCore Event for brain analysis.

    Attributes:
        agent_id: The agent identifier these events belong to.
    """

    def __init__(self, agent_id: str) -> None:
        """Initialize the handler.

        Args:
            agent_id: Unique identifier for the agent being monitored.
        """
        super().__init__()
        self.agent_id = agent_id
        self._events: list[Event] = []

    def _create_event(
        self,
        event_type: EventType,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Event:
        """Create a new SybilCore Event and append it to the internal store.

        Args:
            event_type: The category of agent action.
            content: Human-readable description.
            metadata: Optional key-value data for brain analysis.

        Returns:
            The newly created Event.
        """
        event = Event(
            event_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
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

    # --- LLM callbacks ---

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call begins."""
        model_name = serialized.get("name", serialized.get("id", ["unknown"])[-1])
        self._create_event(
            event_type=EventType.EXTERNAL_CALL,
            content=f"LLM call started: {model_name}",
            metadata={
                "model": model_name,
                "prompt_count": len(prompts),
                "callback_type": "llm_start",
            },
        )

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Called when an LLM call completes."""
        generation_count = 0
        if hasattr(response, "generations"):
            generation_count = sum(len(gen) for gen in response.generations)
        self._create_event(
            event_type=EventType.OUTPUT_GENERATED,
            content=f"LLM call completed with {generation_count} generation(s)",
            metadata={
                "generation_count": generation_count,
                "callback_type": "llm_end",
            },
        )

    # --- Tool callbacks ---

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Called when a tool invocation begins."""
        tool_name = serialized.get("name", "unknown_tool")
        self._create_event(
            event_type=EventType.TOOL_CALL,
            content=f"Tool invoked: {tool_name}",
            metadata={
                "tool_name": tool_name,
                "input_preview": input_str[:200],
                "callback_type": "tool_start",
            },
        )

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Called when a tool invocation completes."""
        self._create_event(
            event_type=EventType.OUTPUT_GENERATED,
            content="Tool execution completed",
            metadata={
                "output_preview": str(output)[:200],
                "callback_type": "tool_end",
            },
        )

    # --- Chain callbacks ---

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Called when a chain execution begins."""
        chain_name = serialized.get("name", serialized.get("id", ["unknown"])[-1])
        self._create_event(
            event_type=EventType.STATE_CHANGE,
            content=f"Chain started: {chain_name}",
            metadata={
                "chain_name": chain_name,
                "input_keys": list(inputs.keys()),
                "callback_type": "chain_start",
            },
        )

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Called when a chain execution completes."""
        self._create_event(
            event_type=EventType.OUTPUT_GENERATED,
            content="Chain execution completed",
            metadata={
                "output_keys": list(outputs.keys()),
                "callback_type": "chain_end",
            },
        )

    # --- Agent callbacks ---

    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        """Called when an agent decides on an action."""
        tool = getattr(action, "tool", "unknown")
        tool_input = str(getattr(action, "tool_input", ""))[:200]
        self._create_event(
            event_type=EventType.TOOL_CALL,
            content=f"Agent action: {tool}",
            metadata={
                "tool": tool,
                "tool_input_preview": tool_input,
                "callback_type": "agent_action",
            },
        )
