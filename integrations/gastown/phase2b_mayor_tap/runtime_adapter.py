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

P0 fix (2026-04-20): `instruction_source` now populated for ALL inter-agent event
types, not only `patrol_started`. Each mapping is cited below. Unknown event types
fall back to best-effort extraction from payload.get("source") / payload.get("origin")
/ actor_id, documented as "inferred from Gastown convention."

P1 fix (2026-04-20): `recipient_agent_id` now accepts 6 possible payload key names:
target, to, recipient, to_agent, polecat_id, target_agent. First non-null wins.

Cite: SCOUT_REPORT.md §2A, §6, §10 — confirmed at scout time 2026-04-20.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path  # noqa: TCH003
from typing import Any

from sybilcore.core.config import MAX_CONTENT_LENGTH
from sybilcore.models.event import Event, EventType

_LOG = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# P1 fix: recipient_agent_id key tolerance
# ---------------------------------------------------------------------------
# SCOUT_REPORT.md §2A documents Gastown payload keys but does not confirm which
# specific key name each event type uses for the target agent.
# NukePolecat at handlers.go:826 is the one confirmed payload site (SCOUT §6).
# The full set of keys below is defensive coverage:
#   "target"       — documented in SCOUT_REPORT.md §3 (beads model has a "target" field)
#   "to"           — seen in escalation_stream payload ("to": "mayor") and handoff payloads
#   "recipient"    — inferred from Gastown convention, TODO verify in handlers.go
#   "to_agent"     — inferred from Gastown convention, TODO verify in handlers.go
#   "polecat_id"   — inferred from polecat naming convention (SCOUT §5), TODO verify
#   "target_agent" — inferred from Gastown convention, TODO verify in handlers.go
_RECIPIENT_KEYS: tuple[str, ...] = (
    "target",
    "to",
    "recipient",
    "to_agent",
    "polecat_id",
    "target_agent",
)


def _extract_recipient_agent_id(payload_dict: dict[str, Any]) -> str | None:
    """Return the first non-null recipient key from a GtEvent payload.

    Accepts multiple possible key names because Gastown event types do not
    standardize the target-agent key name across all call sites.
    SCOUT_REPORT.md §2A documents the payload schema at a struct level but
    does not enumerate key names per event type.

    None is a valid return for non-communication events (hook, done, merged, etc.)
    where there is no designated recipient. Callers should not treat None as an
    error for those event types.

    Args:
        payload_dict: The parsed GtEvent payload dict.

    Returns:
        First non-null value from the known recipient keys, or None.
    """
    for key in _RECIPIENT_KEYS:
        val = payload_dict.get(key)
        if val:
            return str(val)
    # None is valid for non-communication events — log at DEBUG, not WARNING.
    _LOG.debug("No recipient agent key found in payload (expected for non-communication events)")
    return None


# ---------------------------------------------------------------------------
# P0 fix: instruction_source coverage for all inter-agent event types
# ---------------------------------------------------------------------------
# Mapping rationale per event type:
#
#   patrol_started    → "mayor"
#       SCOUT_REPORT.md §6 confirms Mayor dispatches patrols via NukePolecat
#       and BuildPatrolReceipt call sites. The actor on patrol_started is always "mayor".
#       The source is the mayor/deacon who dispatched — use actor_id as source.
#
#   sling             → actor_id (the slinging agent is self-authorizing)
#       [SCOUT §2A] sling is in the documented type list (external tool call).
#       The slinging agent is the instruction source — it authorizes its own external
#       calls. No external authority is asserting the instruction.
#       Cite: SCOUT_REPORT.md §2A buildEventMessage; authority is the sling actor itself.
#
#   handoff           → actor_id (the handing-off agent is the source)
#       [SCOUT §2A] handoff is in the documented type list (work transferred between agents).
#       The actor initiating the handoff is the instruction source — they are directing
#       where work should go next.
#       Cite: SCOUT_REPORT.md §2A; inferred from Gastown handoff convention.
#
#   escalation_sent   → actor_id (the escalating agent is the source)
#       [SCOUT §6] AssessHelp at protocol.go:678 handles HELP: escalations.
#       The polecat originating the HELP: message is the instruction source; it may
#       have been influenced by external instructions that caused it to escalate.
#       Cite: SCOUT_REPORT.md §6 AssessHelp categories (emergency, failed, blocked...).
#
#   bead_assigned     → "mayor" (mayor/dispatch assigns beads to polecats)
#       [SCOUT §3] Bead status transitions show mayor assigning polecats.
#       bead_assigned events originate from mayor or dispatch; use actor_id.
#       Cite: SCOUT_REPORT.md §3 (status_changed to "hooked" actor is "mayor").
#       Note: "bead_assigned" is NOT in the SCOUT §2A documented type list —
#       inferred from Gastown convention (bead assignment pattern). TODO verify.
#
#   merge_claimed     → actor_id (refinery claims the MR — self-directed)
#       [SCOUT §7] Refinery watches MERGE_READY events and claims MR beads.
#       The refinery is self-directing based on the channel event trigger.
#       Cite: SCOUT_REPORT.md §7 (refinery/engineer.go claiming phase).
#       Note: "merge_claimed" not in §2A type list — inferred. TODO verify.
#
#   merge_completed   → actor_id (refinery executes the merge — self-directed)
#       [SCOUT §7] The refinery executes the merge after gates pass.
#       Cite: SCOUT_REPORT.md §7. Note: "merge_completed" not in §2A type list —
#       inferred. TODO verify.
#
#   hook_write        → actor_id (polecat writes a hook — self-authored)
#       [SCOUT §2A] hook is in the documented type list (polecat hooking into a bead).
#       hook_write is a self-authored action — the actor is its own instruction source.
#       Note: "hook_write" not in §2A type list — inferred. TODO verify.
#
#   convoy_opened     → actor_id (mayor/dispatch opens a convoy — self-directed)
#       [SCOUT §4] Convoy is opened by an orchestrator agent (mayor or dispatch).
#       Cite: SCOUT_REPORT.md §4 (convoy lifecycle: staged_ready from mayor).
#       Note: "convoy_opened" not in §2A type list — inferred. TODO verify.
#
#   handoff_requested → actor_id (the requesting agent is the source)
#       [SCOUT §2A] handoff is documented; handoff_requested is the precursor event.
#       The requesting agent is the instruction source for the transfer.
#       Note: "handoff_requested" not in §2A type list — inferred. TODO verify.
#
#   handoff_accepted  → actor_id (the accepting agent confirms the transfer)
#       The accepting agent acknowledges the handoff instruction.
#       Note: "handoff_accepted" not in §2A type list — inferred. TODO verify.
#
#   patrol_complete   → actor_id (polecat self-reports completion)
#       [SCOUT §2A] patrol_complete is in the documented type list.
#       Polecat is self-authorizing the completion signal.
#
#   done              → actor_id (polecat self-reports done)
#       [SCOUT §2A] done is in the documented type list.
#
#   mail              → actor_id (sending agent is the instruction source)
#       [SCOUT §2A] mail is in the documented type list.
#       The sender of the mail is the authority for the message.
#
#   merged / merge_failed → actor_id (refinery is the acting authority)
#       [SCOUT §2A + §7] Both are in the documented type list.
#
#   polecat_nudged    → actor_id (mayor sends the nudge — mayor is the source)
#       [SCOUT §2A] polecat_nudged is in the documented type list.
#
#   polecat_checked   → actor_id (witness is checking — witness is the source)
#       [SCOUT §2A] polecat_checked is in the documented type list.
#
#   hook              → actor_id (polecat hooks into a bead — self-authorized)
#       [SCOUT §2A] hook is in the documented type list.
#
#   FALLBACK for unknown types:
#       Use payload.get("source") or payload.get("origin") or actor_id.
#       Labeled "inferred from Gastown convention" — requires schema verification
#       against live Gastown runtime.

# Event types where the instruction source is deterministically "mayor"
# (dispatching authority confirmed by SCOUT_REPORT.md §6 and §3)
_MAYOR_DISPATCHED_TYPES: frozenset[str] = frozenset({
    "patrol_started",   # SCOUT §6: Mayor dispatches via BuildPatrolReceipt
    "polecat_nudged",   # SCOUT §2A: Mayor sends nudge/reminder
    "bead_assigned",    # SCOUT §3: Mayor assigns polecats (inferred, TODO verify event name)
})

# Event types where the actor itself is the instruction source
# (self-authorized — no external authority asserted)
_SELF_SOURCED_TYPES: frozenset[str] = frozenset({
    "sling",            # SCOUT §2A: polecat self-authorizes external call
    "hook",             # SCOUT §2A: polecat hooks into artifact
    "hook_write",       # inferred from hook convention, TODO verify
    "handoff",          # SCOUT §2A: handing-off agent directs transfer
    "handoff_requested",# inferred: requester is source, TODO verify
    "handoff_accepted", # inferred: acceptor confirms, TODO verify
    "escalation_sent",  # SCOUT §6: polecat sends HELP: (may reflect external influence)
    "patrol_complete",  # SCOUT §2A: polecat self-reports
    "done",             # SCOUT §2A: polecat self-reports
    "mail",             # SCOUT §2A: sending agent is authority
    "merged",           # SCOUT §2A + §7: refinery self-directs
    "merge_failed",     # SCOUT §2A + §7: refinery self-directs
    "merge_claimed",    # inferred from SCOUT §7 refinery claim phase, TODO verify
    "merge_completed",  # inferred from SCOUT §7 refinery merge phase, TODO verify
    "convoy_opened",    # inferred from SCOUT §4 convoy lifecycle, TODO verify
    "polecat_checked",  # SCOUT §2A: witness is the checking authority
})


def _extract_instruction_source(
    gt_type: str,
    actor_id: str,
    payload_dict: dict[str, Any],
) -> str | None:
    """Return the instruction_source for a GtEvent, covering all known inter-agent types.

    P0 fix: Previously only populated for `patrol_started`. Now covers all
    inter-agent event types so CompromiseBrain can detect authority hijacking
    on escalations, handoffs, merge outcomes, and other event classes.

    Mapping logic:
        - Mayor-dispatched types → "mayor" (deterministic authority)
        - Self-sourced types → actor_id (agent is its own authority)
        - Unknown types → best-effort: payload["source"] | payload["origin"] | actor_id
          (labeled "inferred, requires live corpus verification")

    Args:
        gt_type: The GtEvent type string.
        actor_id: The actor field from the GtEvent (agent address).
        payload_dict: The parsed payload dict.

    Returns:
        Instruction source string, or None only if actor_id is also absent.
    """
    if gt_type in _MAYOR_DISPATCHED_TYPES:
        # For nudges, the actual payload may carry the polecat as actor but
        # the authority is always mayor. Use actor_id directly when actor IS mayor;
        # otherwise fall back to "mayor" as the canonical dispatcher.
        return actor_id if actor_id in ("mayor", "deacon") else "mayor"

    if gt_type in _SELF_SOURCED_TYPES:
        return actor_id or None

    # Unknown / undocumented event type — best-effort extraction.
    # Documented as "inferred from Gastown convention" per ADVERSARIAL_REVIEW.md P0 #5 fix.
    best_effort = (
        payload_dict.get("source")
        or payload_dict.get("origin")
        or actor_id
        or None
    )
    if gt_type:
        _LOG.debug(
            "Unknown GtEvent type '%s': instruction_source populated as best-effort '%s'. "
            "TODO: verify against live Gastown runtime schema.",
            gt_type,
            best_effort,
        )
    return best_effort


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

    P0 fix (2026-04-20): instruction_source now populated for all known inter-agent
    event types, not only patrol_started.

    P1 fix (2026-04-20): recipient_agent_id now checks 6 possible payload keys.

    Args:
        row: Parsed GtEvent dict.

    Returns:
        Metadata dict with all GtEvent fields preserved.
    """
    payload = row.get("payload") or {}
    payload_dict: dict[str, Any] = payload if isinstance(payload, dict) else {}

    gt_type = row.get("type", "")
    actor_id: str = str(row.get("actor") or "")

    instruction_source = _extract_instruction_source(gt_type, actor_id, payload_dict)

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
        # P1 fix: recipient_agent_id now checks 6 possible payload key names.
        # None is valid for non-communication events (hook, done, merged, etc.).
        "recipient_agent_id": _extract_recipient_agent_id(payload_dict),
        # Brain-friendly fields for CompromiseBrain
        # P0 fix: instruction_source now populated for ALL inter-agent event types.
        # Previously only patrol_started was covered; now all 12+ known types are mapped.
        "instruction_source": instruction_source,
        "instruction_content": payload_dict.get("message"),
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
