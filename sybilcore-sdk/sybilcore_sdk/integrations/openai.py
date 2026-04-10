"""OpenAI SDK wrapper.

Drop-in wrapper that proxies `chat.completions.create` calls and scores
each completion through SybilCore.

Usage:

    from openai import OpenAI
    from sybilcore_sdk import SybilCore
    from sybilcore_sdk.integrations.openai import wrap_openai

    raw = OpenAI()
    client = wrap_openai(raw, sybil=SybilCore(), agent_id="my-bot")

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(resp.sybilcore_score.tier)
"""

from __future__ import annotations

from typing import Any

from sybilcore_sdk.client import SybilCore
from sybilcore_sdk.models import Event, EventType


class _ScoredCompletions:
    """Proxy for `client.chat.completions` that scores every call."""

    def __init__(self, raw: Any, sybil: SybilCore, agent_id: str) -> None:
        self._raw = raw
        self._sybil = sybil
        self._agent_id = agent_id

    def create(self, *args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        events: list[Event] = []
        for msg in messages:
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            events.append(
                Event(
                    agent_id=self._agent_id,
                    event_type=EventType.MESSAGE_SENT,
                    content=str(content),
                    metadata={"role": msg.get("role", "user")} if isinstance(msg, dict) else {},
                    source="openai",
                )
            )

        response = self._raw.create(*args, **kwargs)

        try:
            choice_text = response.choices[0].message.content or ""
        except (AttributeError, IndexError):
            choice_text = ""

        events.append(
            Event(
                agent_id=self._agent_id,
                event_type=EventType.OUTPUT_GENERATED,
                content=str(choice_text),
                metadata={"model": kwargs.get("model", "unknown")},
                source="openai",
            )
        )

        score = self._sybil.score(events)
        # Attach score to response object if possible
        try:
            object.__setattr__(response, "sybilcore_score", score)
        except (AttributeError, TypeError):
            pass
        return response


class _ScoredChat:
    def __init__(self, raw: Any, sybil: SybilCore, agent_id: str) -> None:
        self.completions = _ScoredCompletions(raw.completions, sybil, agent_id)


class WrappedOpenAI:
    """Thin proxy preserving the OpenAI client surface, with chat scoring."""

    def __init__(self, raw: Any, sybil: SybilCore, agent_id: str) -> None:
        self._raw = raw
        self.chat = _ScoredChat(raw.chat, sybil, agent_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


def wrap_openai(client: Any, sybil: SybilCore, agent_id: str) -> WrappedOpenAI:
    """Wrap an `openai.OpenAI` client so chat completions are scored.

    Args:
        client: The raw OpenAI client instance.
        sybil: A configured `SybilCore` SDK client.
        agent_id: Identifier for the agent making the calls.
    """
    return WrappedOpenAI(client, sybil, agent_id)
