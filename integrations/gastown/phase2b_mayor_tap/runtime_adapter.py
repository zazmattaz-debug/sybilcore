"""Gastown runtime session feed → SybilCore Event adapter (Phase 2B).

Converts Gastown `.events.jsonl` firehose records (GtEvent structs — the
session activity feed log) into SybilCore `Event` objects.

This is DISTINCT from the Phase 1A bead-tracker adapter (`phase1a_baseline/adapter.py`).

Phase 1A consumed `.beads/backup/events.jsonl` — a ledger of work-state transitions.
Phase 2B consumes `~/gt/.events.jsonl` — the runtime behavioral firehose with richer
signals: patrol_started, escalation_sent, sling, hook, handoff, done, mail, merged, etc.

GtEvent schema (from `internal/tui/feed/events.go:226-233` in the Gastown repo):
    {
        "ts":         "<RFC3339 timestamp>",
        "source":     "<string>",
        "type":       "<event type string>",
        "actor":      "<agent address, e.g. gastown/polecats/nux>",
        "payload":    { "rig": "...", "bead": "...", "message": "...", ... },
        "visibility": "feed | both | (other)"
    }

Only events with visibility == "feed" or "both" are shown in the TUI feed.
For scoring purposes we consume ALL visibility values.

Mapping strategy:
    - ALL GtEvent fields are preserved in metadata — no information loss.
    - The `actor` field → agent_id (direct map, same address format).
    - The `ts` field → timestamp (RFC3339 Z-suffix, Python 3.11+ handles it).
    - The `type` field → EventType via the documented enum in SCOUT_REPORT.md §2A.
    - The `payload` dict is stored in metadata["payload"] for brain inspection.
    - The `source` field is preserved as metadata["gt_source"] to avoid collision
      with SybilCore's own `source` field (which is hardcoded "gastown_runtime").

Cite: SCOUT_REPORT.md §2A, §6, §10 — confirmed at scout time 2026-04-20.
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
# GtEvent type → SybilCore EventType mapping table
# ---------------------------------------------------------------------------
# Source: SCOUT_REPORT.md §2A (buildEventMessage in internal/tui/feed/events.go:423-539)
# and §10 (Schema Side-by-Side gap analysis).
#
# patrol_started   → INSTRUCTION_RECEIVED  (mayor dispatching work to polecat)
# patrol_complete  → OUTPUT_GENERATED      (polecat signals patrol done)
# polecat_checked  → STATE_CHANGE          (witness assesssed polecat state)
# polecat_nudged   → MESSAGE_SENT          (mayor sent a nudge/reminder)
# escalation_sent  → PERMISSION_REQUEST    (HELP: escalation — polecat requesting
#                                           oversight; maps to §6 AssessHelp)
# sling            → EXTERNAL_CALL         (polecat calling an external tool/system)
# hook             → TOOL_CALL             (polecat hooking into an artifact/bead)
# handoff          → STATE_CHANGE          (work transferred between agents)
# done             → OUTPUT_GENERATED      (polecat signals work complete)
# mail             → MESSAGE_SENT          (inter-agent mail message)
# merged           → OUTPUT_GENERATED      (merge completed — final delivery signal)
# merge_failed     → ERROR_RAISED          (merge pipeline error)

_GT_RUNTIME_EVENT_TYPE_MAP: dict[str, EventType] = {
    "patrol_started": EventType.INSTRUCTION_RECEIVED,
    "patrol_complete": EventType.OUTPUT_GENERATED,
    "polecat_checked": EventType.STATE_CHANGE,
    "polecat_nudged": EventType.MESSAGE_SENT,
    "escalation_sent": EventType.PERMISSION_REQUEST,
    "sling": EventType.EXTERNAL_CALL,
    "hook": EventType.TOOL_CALL,
    "handoff": EventType.STATE_CHANGE,
    "done": EventType.OUTPUT_GENERATED,
    "mail": EventType.MESSAGE_SENT,
    "merged": EventType.OUTPUT_GENERATED,
    "merge_failed": EventType.ERROR_RAISED,
}

# Fallback for unknown event types — MESSAGE_SENT is the safest neutral choice.
_FALLBACK_EVENT_TYPE = EventType.MESSAGE_SENT


def _parse_runtime_timestamp(ts_str: str) -> datetime:
    """Parse a GtEvent RFC3339 timestamp to a timezone-aware datetime.

    GtEvent uses the `ts` field (RFC3339 with Z suffix, always UTC).
    Python 3.11+ fromisoformat() handles the 'Z' suffix natively.
    Defensive fallback: treat naive timestamps as UTC.

    Args:
        ts_str: Timestamp string, e.g. "2026-04-20T14:30:00Z".

    Returns:
        UTC-aware datetime.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _build_runtime_content(row: dict[str, Any]) -> str:
    """Build a human-readable content string from a GtEvent row.

    The runtime feed is richer than the bead-tracker: payload often contains
    a 'message' key with free-text content from the polecat. We use it when
    present so brains like SemanticBrain and DeceptionBrain have real text.

    Args:
        row: Parsed GtEvent dict.

    Returns:
        Content string capped at MAX_CONTENT_LENGTH.
    """
    actor = row.get("actor", "unknown")
    event_type = row.get("type", "unknown")
    payload = row.get("payload") or {}

    # Prefer the 'message' key from payload — it's free text from the agent.
    message = ""
    if isinstance(payload, dict):
        message = str(payload.get("message", ""))
        if not message:
            # Fallback: include bead and rig from payload for context.
            bead = payload.get("bead", "")
            rig = payload.get("rig", "")
            if bead or rig:
                message = f"bead={bead} rig={rig}"

    parts: list[str] = [f"{actor} {event_type}"]
    if message:
        parts.append(message[:500])

    return " | ".join(parts)[:MAX_CONTENT_LENGTH]


def _build_runtime_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Preserve ALL GtEvent fields in the metadata dict.

    Every field from the original GtEvent is carried through so no
    information is lost downstream. Brains that need specific fields
    (e.g., SocialGraphBrain needing recipient_agent_id) can inspect
    this dict directly.

    The full payload dict is preserved, giving brains access to:
        payload.bead — the bead being acted on
        payload.rig  — which rig the event originated from
        payload.message — free-text content from the polecat

    Args:
        row: Parsed GtEvent dict.

    Returns:
        Metadata dict with all GtEvent fields preserved.
    """
    payload = row.get("payload") or {}
    payload_dict: dict[str, Any] = payload if isinstance(payload, dict) else {}

    gt_type = row.get("type", "")
    return {
        # GtEvent top-level fields
        "gt_source": row.get("source"),
        "gt_type": gt_type,
        "gt_actor": row.get("actor"),
        "gt_visibility": row.get("visibility"),
        "gt_ts": row.get("ts"),
        # Payload fields — extracted for brain accessibility
        "payload": payload_dict,
        "payload_bead": payload_dict.get("bead"),
        "payload_rig": payload_dict.get("rig"),
        "payload_message": payload_dict.get("message"),
        # Source system tag
        "source_system": "gastown_runtime",
        # Brain-friendly fields for SocialGraphBrain
        # recipient_agent_id is populated for mail/handoff events where
        # the payload contains a target agent.
        "recipient_agent_id": payload_dict.get("target") or payload_dict.get("to"),
        # Brain-friendly fields for CompromiseBrain
        # instruction_source populated for patrol_started (mayor → polecat dispatch)
        "instruction_source": (
            "mayor" if gt_type == "patrol_started" else None
        ),
        "instruction_content": (
            payload_dict.get("message")
            if gt_type == "patrol_started"
            else None
        ),
        # Brain-friendly fields for DeceptionBrain
        # query/citations preserved from payload if present
        "query": payload_dict.get("query"),
        "citations": payload_dict.get("citations", []),
        "stated_confidence": payload_dict.get("stated_confidence"),
        "topic_uncertain": payload_dict.get("topic_uncertain", False),
        # Brain-friendly fields for EconomicBrain
        # resource_amount and resource_direction extracted from payload
        # [SYNTHETIC ASSUMPTION] — Gastown doesn't emit resource amounts in production;
        # these are provided by synthetic_firehose.py for brain activation testing.
        "resource_amount": payload_dict.get("resource_amount"),
        "resource_direction": payload_dict.get("resource_direction"),
        "resource_count": payload_dict.get("resource_count"),
        "retry_count": payload_dict.get("retry_count"),
    }


def adapt_runtime_event(row: dict[str, Any]) -> Event:
    """Convert a single GtEvent dict (runtime session feed) to a SybilCore Event.

    This function handles the `~/gt/.events.jsonl` firehose, NOT the bead-tracker
    backup events. The two formats are distinct — see SCOUT_REPORT.md §2 for the
    clarification that cost Phase 1A significant iteration time.

    Schema:
        GtEvent.actor    → SybilCore.agent_id  (direct, same address format)
        GtEvent.ts       → SybilCore.timestamp (RFC3339, Z-suffix aware)
        GtEvent.type     → SybilCore.event_type (via _GT_RUNTIME_EVENT_TYPE_MAP)
        ALL GtEvent fields → SybilCore.metadata (lossless preservation)

    Args:
        row: Parsed JSON dict from a `~/gt/.events.jsonl` line.

    Returns:
        Frozen SybilCore Event.

    Raises:
        ValueError: If required fields (actor, type, ts) are missing.
    """
    actor = row.get("actor")
    if not actor:
        msg = f"GtEvent row missing 'actor' field: {row}"
        raise ValueError(msg)

    gt_type = row.get("type")
    if not gt_type:
        msg = f"GtEvent row missing 'type' field: {row}"
        raise ValueError(msg)

    ts_str = row.get("ts")
    if not ts_str:
        msg = f"GtEvent row missing 'ts' field: {row}"
        raise ValueError(msg)

    sc_type = _GT_RUNTIME_EVENT_TYPE_MAP.get(gt_type, _FALLBACK_EVENT_TYPE)
    timestamp = _parse_runtime_timestamp(ts_str)
    metadata = _build_runtime_metadata(row)

    # Record unmapped types for gap analysis.
    if gt_type not in _GT_RUNTIME_EVENT_TYPE_MAP:
        metadata["unmapped_gt_runtime_type"] = gt_type

    return Event(
        event_id=str(uuid.uuid4()),
        agent_id=str(actor),
        event_type=sc_type,
        timestamp=timestamp,
        content=_build_runtime_content(row),
        metadata=metadata,
        source="gastown_runtime",
    )


def load_runtime_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a `~/gt/.events.jsonl` file and return parsed dicts.

    Args:
        path: Path to the runtime .events.jsonl file.

    Returns:
        List of parsed dicts; empty list if file does not exist.
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


def adapt_runtime_file(path: Path) -> list[Event]:
    """Load and adapt all GtEvents from a runtime `.events.jsonl` file.

    Args:
        path: Path to the runtime `.events.jsonl` file.

    Returns:
        List of adapted SybilCore Events.
    """
    rows = load_runtime_jsonl(path)
    return [adapt_runtime_event(row) for row in rows]


def group_runtime_events_by_agent(events: list[Event]) -> dict[str, list[Event]]:
    """Partition runtime events by agent_id, preserving chronological order.

    Args:
        events: Flat list of adapted Events (any order).

    Returns:
        Dict mapping agent_id → sorted list of that agent's Events.
    """
    groups: dict[str, list[Event]] = {}
    for event in events:
        groups.setdefault(event.agent_id, []).append(event)
    for agent_events in groups.values():
        agent_events.sort(key=lambda e: e.timestamp)
    return groups
