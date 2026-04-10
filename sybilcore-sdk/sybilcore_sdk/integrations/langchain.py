"""LangChain callback integration.

Wires SybilCore into LangChain's callback system so every LLM call,
tool invocation, and chain step is automatically scored.

Usage:

    from langchain_core.runnables import RunnableConfig
    from sybilcore_sdk import SybilCore
    from sybilcore_sdk.integrations.langchain import SybilCoreCallbackHandler

    sc = SybilCore()
    handler = SybilCoreCallbackHandler(client=sc, agent_id="my-agent")

    config = RunnableConfig(callbacks=[handler])
    result = my_chain.invoke({"input": "..."}, config=config)

    # Pull the latest score:
    print(handler.latest_score)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sybilcore_sdk.client import SybilCore
from sybilcore_sdk.models import Event, EventType, ScoreResult

if TYPE_CHECKING:  # pragma: no cover
    from uuid import UUID

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError:  # pragma: no cover
    BaseCallbackHandler = object  # type: ignore[misc, assignment]


class SybilCoreCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
    """LangChain callback that streams events into SybilCore.

    Each callback method appends an `Event` to an internal buffer. The
    buffer is scored on every `on_chain_end` and exposed via `latest_score`.
    """

    def __init__(
        self,
        client: SybilCore,
        agent_id: str,
        score_on: str = "chain_end",
    ) -> None:
        self._client = client
        self._agent_id = agent_id
        self._buffer: list[Event] = []
        self._score_on = score_on
        self.latest_score: ScoreResult | None = None

    def _record(self, event_type: EventType, content: str, **metadata: Any) -> None:
        self._buffer.append(
            Event(
                agent_id=self._agent_id,
                event_type=event_type,
                content=content[:5_000],
                metadata=metadata,
                source="langchain",
            )
        )

    def _maybe_score(self, trigger: str) -> None:
        if trigger != self._score_on or not self._buffer:
            return
        self.latest_score = self._client.score(self._buffer)

    # ── LangChain callback methods (subset that matters for trust scoring) ──

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        for prompt in prompts:
            self._record(EventType.MESSAGE_SENT, prompt, kind="llm_prompt")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        text = str(response)[:5_000]
        self._record(EventType.OUTPUT_GENERATED, text, kind="llm_response")

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        tool_name = serialized.get("name", "unknown")
        self._record(EventType.TOOL_CALL, input_str, tool=tool_name)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        self._record(EventType.OUTPUT_GENERATED, output, kind="tool_output")

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        self._record(EventType.ERROR_RAISED, str(error), kind="tool_error")

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        self._maybe_score("chain_end")

    def reset(self) -> None:
        """Clear the internal buffer (call between chain runs)."""
        self._buffer = []
        self.latest_score = None
