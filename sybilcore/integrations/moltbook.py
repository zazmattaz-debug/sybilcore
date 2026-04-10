"""Moltbook integration — load Moltbook posts as SybilCore Events.

Provides an adapter that reads a locally cached Moltbook dataset (JSONL)
and converts each post into SybilCore Events for behavioral scoring.

Event mapping:
    post       → OUTPUT_GENERATED
    comment    → MESSAGE_SENT   (inferred from comment_count > 0 signals)
    vote       → TOOL_CALL      (upvote/downvote interactions)

Usage:
    from sybilcore.integrations.moltbook import MoltbookAdapter

    adapter = MoltbookAdapter()
    events = adapter.get_agent_events("FarnsworthAI")
    all_agents = adapter.get_all_agents()
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sybilcore.models.event import Event, EventType

logger = logging.getLogger(__name__)

SOURCE_NAME = "moltbook"

# Default path relative to the sybilcore project root.
_DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "moltbook_posts.jsonl"

# Maximum preview length for content fields.
_PREVIEW_MAX: int = 200

# Maximum length for metadata string values.
_METADATA_MAX: int = 500


def _safe_str(val: Any, max_len: int = _METADATA_MAX) -> Any:
    """Truncate string values to max_len. Non-strings pass through."""
    if isinstance(val, str):
        return val[:max_len]
    return val


def _safe_int(val: Any, default: int = 0) -> int:
    """Safely convert a value to int, returning default on failure."""
    try:
        return int(val or 0)
    except (TypeError, ValueError):
        return default


def _parse_timestamp(raw: str | None) -> datetime:
    """Parse an ISO-8601 timestamp string; fall back to epoch on failure.

    Args:
        raw: ISO-8601 datetime string (may include timezone offset).

    Returns:
        UTC-aware datetime.
    """
    if not raw:
        return datetime(1970, 1, 1, tzinfo=UTC)
    try:
        # Python 3.11+ fromisoformat handles offset-aware strings.
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except (ValueError, TypeError):
        logger.debug("Failed to parse timestamp: %r", raw)
        return datetime(1970, 1, 1, tzinfo=UTC)


def _post_to_events(post: dict[str, Any]) -> list[Event]:
    """Convert a single Moltbook post record into SybilCore Events.

    A post produces:
      1. An OUTPUT_GENERATED event for the post itself.
      2. One TOOL_CALL event per upvote (vote engagement signal).
      3. One TOOL_CALL event per downvote (negative vote signal).

    The vote events use the same timestamp as the post. This approximates
    vote-casting behaviour as discrete tool interactions — consistent with
    how the MiroFish adapter maps upvote/downvote actions.

    Args:
        post: Raw record dict from the Moltbook JSONL file.

    Returns:
        List of Events representing all activity in this post record.
    """
    agent_id = str(post.get("author", "unknown"))
    submolt = _safe_str(post.get("submolt") or "", _METADATA_MAX)
    upvotes = _safe_int(post.get("upvotes"))
    downvotes = _safe_int(post.get("downvotes"))
    karma = _safe_int(post.get("score"))
    post_id = _safe_str(post.get("id") or str(uuid.uuid4()), _METADATA_MAX)
    title = _safe_str(post.get("title") or "", _PREVIEW_MAX)
    post_url = _safe_str(post.get("post_url") or "", _METADATA_MAX)
    comment_count = _safe_int(post.get("comment_count"))
    ts = _parse_timestamp(post.get("created_at"))

    events: list[Event] = []

    # 1. Primary post event → OUTPUT_GENERATED
    post_meta: dict[str, Any] = {
        "moltbook_event": "post",
        "post_id": post_id,
        "submolt": submolt,
        "karma": karma,
        "upvotes": upvotes,
        "downvotes": downvotes,
        "comment_count": comment_count,
        "post_url": post_url,
    }
    post_event = Event(
        event_id=str(uuid.uuid4()),
        agent_id=agent_id,
        event_type=EventType.OUTPUT_GENERATED,
        timestamp=ts,
        content=f"post: {title[:_PREVIEW_MAX]}",
        metadata=post_meta,
        source=SOURCE_NAME,
    )
    events = [*events, post_event]

    # 2. Upvote events → TOOL_CALL (capped at 20 per post to avoid noise)
    up_cap = min(upvotes, 20)
    for _ in range(up_cap):
        vote_event = Event(
            event_id=str(uuid.uuid4()),
            agent_id=agent_id,
            event_type=EventType.TOOL_CALL,
            timestamp=ts,
            content=f"vote: upvote on post in r/{submolt}",
            metadata={
                "moltbook_event": "vote",
                "vote_type": "upvote",
                "post_id": post_id,
                "submolt": submolt,
            },
            source=SOURCE_NAME,
        )
        events = [*events, vote_event]

    # 3. Downvote events → TOOL_CALL (capped at 20 per post)
    down_cap = min(downvotes, 20)
    for _ in range(down_cap):
        vote_event = Event(
            event_id=str(uuid.uuid4()),
            agent_id=agent_id,
            event_type=EventType.TOOL_CALL,
            timestamp=ts,
            content=f"vote: downvote on post in r/{submolt}",
            metadata={
                "moltbook_event": "vote",
                "vote_type": "downvote",
                "post_id": post_id,
                "submolt": submolt,
            },
            source=SOURCE_NAME,
        )
        events = [*events, vote_event]

    return events


class MoltbookAdapter:
    """Adapter that reads Moltbook posts from a local JSONL cache.

    Converts each post record into SybilCore Events and indexes them
    by agent_id for efficient retrieval.

    Attributes:
        data_path: Path to the Moltbook JSONL file.
    """

    def __init__(
        self,
        data_path: str | Path | None = None,
    ) -> None:
        """Initialize the adapter and load all events into memory.

        Args:
            data_path: Path to the JSONL file. Defaults to
                ``data/moltbook_posts.jsonl`` relative to the project root.

        Raises:
            FileNotFoundError: If the JSONL file does not exist.
        """
        resolved = Path(data_path) if data_path else _DEFAULT_DATA_PATH
        if not resolved.exists():
            msg = (
                f"Moltbook data file not found: {resolved}. "
                "Run the download script first."
            )
            raise FileNotFoundError(msg)

        self.data_path = resolved
        self._agent_events: dict[str, list[Event]] = {}
        self._load()

    def _load(self) -> None:
        """Parse the JSONL file and index all events by agent_id."""
        raw_count = 0
        event_count = 0
        skipped = 0

        with self.data_path.open() as fh:
            for line_num, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    post = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON at line %d", line_num)
                    skipped += 1
                    continue

                raw_count += 1
                agent_id = str(post.get("author", "unknown"))
                events = _post_to_events(post)
                current = self._agent_events.get(agent_id, [])
                self._agent_events[agent_id] = [*current, *events]
                event_count += len(events)

        logger.info(
            "MoltbookAdapter loaded %d posts → %d events for %d agents (%d skipped)",
            raw_count,
            event_count,
            len(self._agent_events),
            skipped,
        )

    def get_agent_events(self, agent_id: str) -> list[Event]:
        """Return all events for a specific agent.

        Args:
            agent_id: The Moltbook author username.

        Returns:
            List of Events, chronologically ordered. Empty list if agent
            has no recorded activity.
        """
        events = self._agent_events.get(agent_id, [])
        return sorted(events, key=lambda e: e.timestamp)

    def get_all_agents(self) -> list[str]:
        """Return all agent IDs present in the dataset.

        Returns:
            Sorted list of unique author usernames.
        """
        return sorted(self._agent_events.keys())

    @property
    def agent_count(self) -> int:
        """Total number of unique agents in the loaded dataset."""
        return len(self._agent_events)

    @property
    def event_count(self) -> int:
        """Total number of Events across all agents."""
        return sum(len(evts) for evts in self._agent_events.values())
