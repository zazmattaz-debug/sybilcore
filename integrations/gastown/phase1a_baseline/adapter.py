"""Gastown bead-event → SybilCore Event adapter.

Converts Gastown `.beads/backup/events.jsonl` records (bead lifecycle events)
into SybilCore `Event` objects for replay through the coefficient pipeline.

Mapping decisions are documented inline.  Every Gastown field is preserved in
`metadata` so nothing is lost, even when there is no clean SybilCore equivalent.

NOTE: Only bead-tracker events (`.beads/backup/events.jsonl`) are handled here.
The runtime `.events.jsonl` session feed and channel `.event` files are a
distinct format not present in the committed fixtures — see SCOUT_REPORT.md §2.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path  # noqa: TCH003
from typing import Any

from sybilcore.core.config import MAX_CONTENT_LENGTH
from sybilcore.models.event import Event, EventType

# ---------------------------------------------------------------------------
# Event-type mapping table
# ---------------------------------------------------------------------------
# Gastown bead event_type  →  SybilCore EventType  + rationale comment
#
# Decisions:
#   "status_changed"  →  STATE_CHANGE
#       A bead moving between open/hooked/closed is exactly a state transition.
#       The new_value JSON payload usually specifies the new status + assignee.
#
#   "updated"         →  MESSAGE_SENT
#       An agent modifying a bead's description/fields is the closest analogue
#       to sending structured information to collaborators.  It is NOT a tool_call
#       because the bead tracker is the data store, not a tool invocation point.
#
#   "created"         →  TOOL_CALL
#       An agent programmatically creating a new bead (artifact) maps well to a
#       tool call — it is an action that produces a persistent external artifact.
#
#   "closed"          →  OUTPUT_GENERATED
#       Closing a bead signals delivery of a work unit — the primary observable
#       output signal.  Closing by witness/refinery is still output-generation
#       (they are signalling work-complete or error-complete on a bead).
#
#   "reopened"        →  STATE_CHANGE
#       Re-opening is a state reversal — maps cleanly to STATE_CHANGE.
#
#   "renamed"         →  MESSAGE_SENT
#       Renaming a bead title is metadata communication.  Closest to MESSAGE_SENT.
#
#   "label_added"     →  STATE_CHANGE
#       Labelling is a classification/tagging action modifying bead state.
#
#   "label_removed"   →  STATE_CHANGE
#       Same rationale as label_added.
#
# GAP NOTE: Gastown has no direct equivalents for RESOURCE_ACCESS,
#   PERMISSION_REQUEST, INSTRUCTION_RECEIVED, ERROR_RAISED, or EXTERNAL_CALL
#   in the bead-tracker event stream.  Those would appear in the runtime
#   `.events.jsonl` feed (patrol_started, escalation_sent, sling, hook, etc.)
#   but those files are not committed to the repo.  Documented as future work.

_GT_EVENT_TYPE_MAP: dict[str, EventType] = {
    "status_changed": EventType.STATE_CHANGE,
    "updated": EventType.MESSAGE_SENT,
    "created": EventType.TOOL_CALL,
    "closed": EventType.OUTPUT_GENERATED,
    "reopened": EventType.STATE_CHANGE,
    "renamed": EventType.MESSAGE_SENT,
    "label_added": EventType.STATE_CHANGE,
    "label_removed": EventType.STATE_CHANGE,
}

# Gastown event types present in fixtures that cannot be cleanly mapped.
# All will fall back to EventType.MESSAGE_SENT as a neutral default.
_UNMAPPED_GT_TYPES: frozenset[str] = frozenset()  # none currently unmapped


def _parse_timestamp(ts_str: str) -> datetime:
    """Parse an RFC3339 / ISO-8601 timestamp string to a timezone-aware datetime.

    Python 3.11+ fromisoformat() handles the 'Z' suffix natively.
    This function is the single parsing site so it is easy to monkey-patch
    in tests or swap for a compatibility shim.

    Args:
        ts_str: Timestamp string, e.g. "2026-02-26T15:13:23Z".

    Returns:
        UTC-aware datetime.

    Raises:
        ValueError: If the string is unparseable.
    """
    # Python 3.11+ handles "Z" suffix in fromisoformat().
    # On 3.10 or earlier this would fail — the project requires 3.13.
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        # Defensive: treat naive timestamps as UTC (Gastown always uses Z).
        dt = dt.replace(tzinfo=UTC)
    return dt


def _build_content(row: dict[str, Any]) -> str:
    """Build a human-readable content string from a Gastown bead event row.

    Gastown stores structured JSON in new_value/old_value, not prose.
    We synthesise a short description string so SybilCore brains that
    do string-level inspection have something to work with.

    The string is capped at MAX_CONTENT_LENGTH to satisfy Event validation.
    """
    parts: list[str] = []
    event_type = row.get("event_type", "")
    issue_id = row.get("issue_id", "")
    actor = row.get("actor", "")

    parts.append(f"{actor} {event_type} {issue_id}")

    new_val = row.get("new_value") or ""
    if new_val and isinstance(new_val, str):
        # Trim to a sensible prefix so content is still legible.
        parts.append(new_val[:200])

    comment = row.get("comment")
    if comment:
        parts.append(str(comment)[:200])

    return " | ".join(parts)[:MAX_CONTENT_LENGTH]


def _build_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Preserve all Gastown-specific fields in the metadata dict.

    Every field from the original row is carried through so no information
    is lost downstream (brain modules can inspect any Gastown field).

    Large JSON strings (old_value, new_value) are kept as-is; they may be
    truncated by brain modules that have their own length limits.
    """
    return {
        "beads_event_id": row.get("id"),
        "bead_id": row.get("issue_id"),
        "gastown_event_type": row.get("event_type"),
        "new_value": row.get("new_value"),
        "old_value": row.get("old_value"),
        "comment": row.get("comment"),
        "source_system": "gastown_beads",
    }


def adapt_bead_event(row: dict[str, Any]) -> Event:
    """Convert a single Gastown bead-tracker event dict to a SybilCore Event.

    Agent identity:
        Gastown uses the ``actor`` field as the agent address
        (e.g. "gastown/polecats/furiosa").  This maps directly to
        SybilCore's ``agent_id``.

    Timestamp:
        ``created_at`` (RFC3339 Z-suffix) is parsed to a tz-aware datetime.
        SybilCore rejects timestamps more than 60 seconds in the future;
        historical events are always valid on this axis.

    EventType:
        Mapped via ``_GT_EVENT_TYPE_MAP``; unmapped types fall back to
        EventType.MESSAGE_SENT with the original type preserved in metadata.

    Args:
        row: Parsed JSON dict from a beads/backup/events.jsonl line.

    Returns:
        Frozen SybilCore Event.

    Raises:
        ValueError: If required fields (actor, event_type, created_at) are missing.
    """
    actor = row.get("actor")
    if not actor:
        msg = f"Gastown event row missing 'actor' field: {row}"
        raise ValueError(msg)

    gt_type = row.get("event_type")
    if not gt_type:
        msg = f"Gastown event row missing 'event_type' field: {row}"
        raise ValueError(msg)

    created_at_str = row.get("created_at")
    if not created_at_str:
        msg = f"Gastown event row missing 'created_at' field: {row}"
        raise ValueError(msg)

    sc_type = _GT_EVENT_TYPE_MAP.get(gt_type, EventType.MESSAGE_SENT)
    timestamp = _parse_timestamp(created_at_str)
    metadata = _build_metadata(row)

    # If the type was unmapped, record that fact for future gap analysis.
    if gt_type not in _GT_EVENT_TYPE_MAP:
        metadata["unmapped_event_type"] = gt_type

    return Event(
        event_id=str(uuid.uuid4()),
        agent_id=str(actor),
        event_type=sc_type,
        timestamp=timestamp,
        content=_build_content(row),
        metadata=metadata,
        source="gastown",
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of parsed dicts.

    Args:
        path: Path to the .jsonl file.

    Returns:
        List of parsed dicts; empty if file does not exist.
    """
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                msg = f"JSON parse error at {path}:{lineno}: {exc}"
                raise ValueError(msg) from exc
    return rows


def adapt_fixture_file(path: Path) -> list[Event]:
    """Load a Gastown bead-events JSONL file and adapt all rows to SybilCore Events.

    Args:
        path: Path to a ``.beads/backup/events.jsonl`` file.

    Returns:
        List of adapted Events, one per non-empty line.
    """
    rows = load_jsonl(path)
    return [adapt_bead_event(row) for row in rows]


def group_events_by_agent(events: list[Event]) -> dict[str, list[Event]]:
    """Partition events by agent_id, preserving chronological order per agent.

    Args:
        events: Flat list of Events (any order).

    Returns:
        Dict mapping agent_id → sorted list of that agent's Events.
    """
    groups: dict[str, list[Event]] = {}
    for event in events:
        groups.setdefault(event.agent_id, []).append(event)
    for agent_events in groups.values():
        agent_events.sort(key=lambda e: e.timestamp)
    return groups
