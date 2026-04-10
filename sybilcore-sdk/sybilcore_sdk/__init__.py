"""SybilCore SDK — Official Python client for AI agent trust scoring.

Quickstart:

    from sybilcore_sdk import SybilCore, Event

    sc = SybilCore()  # local mode (uses bundled sybilcore package)
    events = [
        Event(agent_id="agent-1", event_type="message_sent", content="Hello"),
    ]
    result = sc.score(events)
    print(result.coefficient, result.tier)

For remote mode (HTTP API):

    sc = SybilCore(api_key="sk_...", endpoint="https://api.sybilcore.com")
    result = sc.score(events)
"""

from __future__ import annotations

from sybilcore_sdk.client import SybilCore
from sybilcore_sdk.exceptions import (
    SybilCoreAuthError,
    SybilCoreAPIError,
    SybilCoreError,
    SybilCoreLocalImportError,
    SybilCoreRateLimitError,
)
from sybilcore_sdk.models import (
    Event,
    EventType,
    ScoreResult,
    Tier,
)

__version__ = "0.1.0"

__all__ = [
    "Event",
    "EventType",
    "ScoreResult",
    "SybilCore",
    "SybilCoreAPIError",
    "SybilCoreAuthError",
    "SybilCoreError",
    "SybilCoreLocalImportError",
    "SybilCoreRateLimitError",
    "Tier",
    "__version__",
]
