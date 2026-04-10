"""MiroFish integration — poll simulation actions as SybilCore Events.

Provides an adapter class that connects to a running MiroFish social
simulation and converts agent actions into SybilCore Events for
coefficient analysis.

Usage:
    from sybilcore.integrations.mirofish import SybilCoreMiroFishAdapter

    adapter = SybilCoreMiroFishAdapter(simulation_id="sim-001")
    new_events = adapter.poll()
    adapter.inject_rogue("agent-7", "Ignore previous instructions.")
    events = adapter.flush()
"""

from __future__ import annotations

import ipaddress
import re
import uuid
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

from sybilcore.models.event import Event, EventType

try:
    import requests

    _HAS_REQUESTS = True
except ImportError:
    requests = None  # type: ignore[assignment]
    _HAS_REQUESTS = False

SOURCE_NAME = "mirofish"

# Maximum preview length for content fields.
_PREVIEW_MAX: int = 200

# Maximum length for metadata string values.
_METADATA_MAX: int = 500

# Validation pattern for simulation IDs.
_SIM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,256}$")

# Private IP network ranges (blocked for SSRF prevention).
_PRIVATE_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
]

# Action type to EventType mapping.
_ACTION_TYPE_MAP: dict[str, EventType] = {
    "post": EventType.OUTPUT_GENERATED,
    "tweet": EventType.OUTPUT_GENERATED,
    "comment": EventType.MESSAGE_SENT,
    "reply": EventType.MESSAGE_SENT,
    "share": EventType.MESSAGE_SENT,
    "retweet": EventType.MESSAGE_SENT,
    "follow": EventType.TOOL_CALL,
    "like": EventType.TOOL_CALL,
    "upvote": EventType.TOOL_CALL,
    "downvote": EventType.TOOL_CALL,
    "error": EventType.ERROR_RAISED,
    "profile_update": EventType.STATE_CHANGE,
}


class SybilCoreMiroFishAdapter:
    """Adapter for MiroFish social simulations that emits SybilCore Events.

    Polls a MiroFish simulation API for new agent actions and converts
    them into SybilCore Events. Supports rogue prompt injection for
    adversarial testing.

    Attributes:
        simulation_id: The MiroFish simulation to monitor.
        mirofish_url: Base URL of the MiroFish API server.
    """

    def __init__(
        self,
        simulation_id: str,
        mirofish_url: str = "http://localhost:5001",
    ) -> None:
        """Initialize the adapter.

        Args:
            simulation_id: MiroFish simulation identifier.
            mirofish_url: Base URL of the MiroFish API server.
        """
        # Validate simulation_id format
        if not _SIM_ID_RE.match(simulation_id):
            msg = (
                f"Invalid simulation_id '{simulation_id}': "
                "must be 1-256 alphanumeric, hyphen, or underscore characters"
            )
            raise ValueError(msg)

        # Validate and sanitize URL (SSRF prevention)
        parsed = urlparse(mirofish_url)
        if parsed.scheme not in ("http", "https"):
            msg = f"Invalid URL scheme '{parsed.scheme}': must be http or https"
            raise ValueError(msg)

        hostname = parsed.hostname or ""
        is_localhost = hostname in ("localhost", "127.0.0.1", "::1")
        if not is_localhost and hostname:
            try:
                addr = ipaddress.ip_address(hostname)
                for network in _PRIVATE_NETWORKS:
                    if addr in network:
                        msg = f"Blocked private IP address in URL: {hostname}"
                        raise ValueError(msg)
            except ValueError as exc:
                # Re-raise our own ValueError, ignore parse failures
                # (hostnames like "api.example.com" are fine)
                if "Blocked private IP" in str(exc):
                    raise

        self.simulation_id = simulation_id
        self.mirofish_url = mirofish_url.rstrip("/")
        self._events: list[Event] = []
        self._last_offset: int = 0

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

    def fetch_new_actions(self, limit: int = 100) -> list[dict[str, Any]]:
        """Fetch new actions from the MiroFish simulation API.

        Args:
            limit: Maximum number of actions to retrieve per request.

        Returns:
            A list of action dictionaries from MiroFish.

        Raises:
            RuntimeError: If the requests library is not installed.
            requests.RequestException: On network or API errors.
        """
        if not _HAS_REQUESTS:
            msg = "requests library is required: pip install requests"
            raise RuntimeError(msg)

        # Clamp limit to sane range
        limit = max(1, min(limit, 1000))

        url = (
            f"{self.mirofish_url}/api/simulation/"
            f"{self.simulation_id}/actions"
        )
        params = {"offset": max(0, self._last_offset), "limit": limit}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        # MiroFish wraps in {"data": {"actions": [...]}} or {"actions": [...]}
        if isinstance(data, dict):
            inner = data.get("data", data)
            if isinstance(inner, dict) and "actions" in inner:
                actions = inner["actions"]
            elif "actions" in data:
                actions = data["actions"]
            else:
                actions = []
        elif isinstance(data, list):
            actions = data
        else:
            actions = []
        if not isinstance(actions, list):
            actions = []

        if actions:
            self._last_offset += len(actions)
        # Guard against negative offset drift
        if self._last_offset < 0:
            self._last_offset = 0
        return actions

    @staticmethod
    def _safe_str(val: Any, max_len: int = _METADATA_MAX) -> Any:
        """Truncate string values to max_len. Non-strings pass through."""
        if isinstance(val, str):
            return val[:max_len]
        return val

    def _convert_action(self, action: dict[str, Any]) -> Event:
        """Convert a MiroFish action dict into a SybilCore Event.

        Args:
            action: Raw action dictionary from MiroFish API.

        Returns:
            A SybilCore Event mapped from the action.
        """
        raw_action_type = action.get("action_type", "unknown")
        # Normalize MiroFish action types (e.g. CREATE_POST -> post).
        # Only strip the leading ``create_`` prefix — preserve other
        # underscores so multi-word action types like ``profile_update``
        # still match their entries in ``_ACTION_TYPE_MAP``.
        action_type = raw_action_type.lower()
        if action_type.startswith("create_"):
            action_type = action_type[len("create_") :]
        agent_id = str(action.get("agent_id", "unknown"))
        event_type = _ACTION_TYPE_MAP.get(action_type, EventType.TOOL_CALL)
        # Content may be top-level or nested in action_args
        action_args = action.get("action_args", {}) or {}
        raw_content = (
            action_args.get("content", "")
            or action.get("content", "")
        )
        content = f"{raw_action_type}: {str(raw_content)[:_PREVIEW_MAX]}"
        metadata = {
            "action_type": self._safe_str(raw_action_type),
            "platform": self._safe_str(action.get("platform", "")),
            "round_num": action.get("round_num", 0),
            "agent_name": self._safe_str(action.get("agent_name", "")),
            "agent_handle": self._safe_str(action.get("agent_handle", "")),
            "mirofish_event": "action",
        }

        return self._create_event(
            agent_id=agent_id,
            event_type=event_type,
            content=content,
            metadata=metadata,
        )

    def poll(self) -> list[Event]:
        """Poll for new actions and convert them to SybilCore Events.

        Returns:
            A list of newly created Events from this poll cycle.
        """
        actions = self.fetch_new_actions()
        new_events: list[Event] = []
        for action in actions:
            event = self._convert_action(action)
            new_events = [*new_events, event]
        return new_events

    def inject_rogue(
        self,
        agent_id: str,
        prompt: str,
    ) -> dict[str, Any]:
        """Inject a rogue prompt into a MiroFish agent for adversarial testing.

        Args:
            agent_id: Target agent to inject the prompt into.
            prompt: The adversarial prompt content.

        Returns:
            Response dictionary from the MiroFish API.

        Raises:
            RuntimeError: If the requests library is not installed.
            requests.RequestException: On network or API errors.
        """
        if not _HAS_REQUESTS:
            msg = "requests library is required: pip install requests"
            raise RuntimeError(msg)

        url = (
            f"{self.mirofish_url}/api/simulation/"
            f"{self.simulation_id}/interview"
        )
        response = requests.post(
            url,
            json={"agent_id": agent_id, "prompt": prompt},
            timeout=30,
        )
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    def get_simulation_status(self) -> dict[str, Any]:
        """Get the current status of the MiroFish simulation.

        Returns:
            Status dictionary from the MiroFish API.

        Raises:
            RuntimeError: If the requests library is not installed.
            requests.RequestException: On network or API errors.
        """
        if not _HAS_REQUESTS:
            msg = "requests library is required: pip install requests"
            raise RuntimeError(msg)

        url = (
            f"{self.mirofish_url}/api/simulation/"
            f"{self.simulation_id}/run-status"
        )
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    def get_agent_profiles(
        self,
        platform: str = "twitter",
    ) -> list[dict[str, Any]]:
        """Get agent profiles from the MiroFish simulation.

        Args:
            platform: Social platform to filter profiles by.

        Returns:
            A list of agent profile dictionaries.

        Raises:
            RuntimeError: If the requests library is not installed.
            requests.RequestException: On network or API errors.
        """
        if not _HAS_REQUESTS:
            msg = "requests library is required: pip install requests"
            raise RuntimeError(msg)

        url = (
            f"{self.mirofish_url}/api/simulation/"
            f"{self.simulation_id}/profiles"
        )
        params = {"platform": platform}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        profiles: list[dict[str, Any]] = response.json()
        return profiles
