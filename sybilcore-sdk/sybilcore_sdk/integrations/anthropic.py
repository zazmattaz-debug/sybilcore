"""Anthropic SDK wrapper.

Drop-in wrapper around `anthropic.Anthropic` that scores every
`messages.create` call through SybilCore.

Usage:

    from anthropic import Anthropic
    from sybilcore_sdk import SybilCore
    from sybilcore_sdk.integrations.anthropic import wrap_anthropic

    raw = Anthropic()
    client = wrap_anthropic(raw, sybil=SybilCore(), agent_id="my-bot")

    msg = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=512,
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(msg.sybilcore_score.tier)
"""

from __future__ import annotations

from typing import Any

from sybilcore_sdk.client import SybilCore
from sybilcore_sdk.models import Event, EventType


class _ScoredMessages:
    def __init__(self, raw: Any, sybil: SybilCore, agent_id: str) -> None:
        self._raw = raw
        self._sybil = sybil
        self._agent_id = agent_id

    def create(self, *args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        events: list[Event] = []
        for msg in messages:
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            if isinstance(content, list):  # multi-part content
                content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
            events.append(
                Event(
                    agent_id=self._agent_id,
                    event_type=EventType.MESSAGE_SENT,
                    content=str(content),
                    metadata={"role": msg.get("role", "user")} if isinstance(msg, dict) else {},
                    source="anthropic",
                )
            )

        response = self._raw.create(*args, **kwargs)

        try:
            text = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
        except (AttributeError, TypeError):
            text = ""

        events.append(
            Event(
                agent_id=self._agent_id,
                event_type=EventType.OUTPUT_GENERATED,
                content=text,
                metadata={"model": kwargs.get("model", "unknown")},
                source="anthropic",
            )
        )

        score = self._sybil.score(events)
        try:
            object.__setattr__(response, "sybilcore_score", score)
        except (AttributeError, TypeError):
            pass
        return response


class WrappedAnthropic:
    """Thin proxy preserving the Anthropic client surface."""

    def __init__(self, raw: Any, sybil: SybilCore, agent_id: str) -> None:
        self._raw = raw
        self.messages = _ScoredMessages(raw.messages, sybil, agent_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


def wrap_anthropic(client: Any, sybil: SybilCore, agent_id: str) -> WrappedAnthropic:
    """Wrap an `anthropic.Anthropic` client so messages are scored."""
    return WrappedAnthropic(client, sybil, agent_id)
