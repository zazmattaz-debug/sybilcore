"""OpenHands event-log → SybilCore Event adapter.

Converts OpenHands trajectory event dicts (as serialized by
``openhands.events.serialization.event.event_to_dict``) into SybilCore
``Event`` objects for replay through the coefficient pipeline.

Schema source: shallow clone of All-Hands-AI/OpenHands @ HEAD (2026-04-20).
  - openhands/events/serialization/event.py  — wire format
  - openhands/core/schema/action.py          — ActionType enum (legacy V0)
  - openhands/events/action/commands.py      — CmdRunAction, IPythonRunCellAction
  - openhands/events/action/files.py         — FileReadAction, FileWriteAction
  - openhands/events/action/message.py       — MessageAction
  - openhands/events/observation/commands.py — CmdOutputObservation
  - openhands/events/observation/files.py    — FileReadObservation
  - tests/unit/resolver/mock_output/output.jsonl — verified sample events

Wire format:
    An OpenHands event dict has one of two shapes:
    (a) Action:
        { "action": "<type>", "args": {...}, "source": "agent"|"user"|"environment",
          "id": <int>, "timestamp": "<iso8601>", "message": "<str>" }
    (b) Observation:
        { "observation": "<obs_type>", "content": "<str>", "extras": {...},
          "source": "agent"|"user"|"environment", "id": <int>, "timestamp": "<iso8601>" }

Mapping decisions:
    Action → EventType:
        message         → INSTRUCTION_RECEIVED  (user input) or MESSAGE_SENT (agent output)
        run             → TOOL_CALL              (shell command execution)
        run_ipython     → TOOL_CALL              (code execution — same semantic as run)
        read            → RESOURCE_ACCESS        (file read)
        write           → OUTPUT_GENERATED       (file write — agent artifact production)
        edit            → OUTPUT_GENERATED       (file edit — artifact modification)
        browse          → EXTERNAL_CALL          (web navigation)
        browse_interactive → EXTERNAL_CALL
        call_tool_mcp   → EXTERNAL_CALL          (MCP tool call to external server)
        delegate        → MESSAGE_SENT           (sub-agent handoff)
        think           → MESSAGE_SENT           (internal monologue — closest to message)
        finish          → OUTPUT_GENERATED       (task completion)
        reject          → ERROR_RAISED           (unable to complete)
        pause/resume/stop → STATE_CHANGE
        change_agent_state → STATE_CHANGE
        push/send_pr    → EXTERNAL_CALL          (GitHub external calls)
        recall          → RESOURCE_ACCESS        (memory retrieval)

    Observation → EventType:
        run             → OUTPUT_GENERATED       (command result)
        run_ipython     → OUTPUT_GENERATED       (cell result)
        read            → OUTPUT_GENERATED       (file content)
        write           → STATE_CHANGE           (write confirmation)
        edit            → STATE_CHANGE           (edit confirmation)
        browse          → OUTPUT_GENERATED       (page content)
        error           → ERROR_RAISED           (runtime error)
        delegate        → MESSAGE_RECEIVED       (sub-agent result)
        success         → OUTPUT_GENERATED       (task success)
        reject          → ERROR_RAISED           (rejection)

    GAP NOTE: OpenHands has richer schema than Gastown beads.
    Notably: thought field on actions, exit_code on observations, security_risk,
    LLM metrics, and source distinguish agent-vs-user-vs-environment.  These
    all map to metadata and provide more signal to SybilCore brains than beads.
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
# Action-type → SybilCore EventType mapping
# ---------------------------------------------------------------------------
_OH_ACTION_TYPE_MAP: dict[str, EventType] = {
    # Core actions
    "run": EventType.TOOL_CALL,
    "run_ipython": EventType.TOOL_CALL,
    "read": EventType.RESOURCE_ACCESS,
    "write": EventType.OUTPUT_GENERATED,
    "edit": EventType.OUTPUT_GENERATED,
    # Communication
    "message": EventType.MESSAGE_SENT,       # overridden per source below
    "think": EventType.MESSAGE_SENT,
    "delegate": EventType.MESSAGE_SENT,
    # External
    "browse": EventType.EXTERNAL_CALL,
    "browse_interactive": EventType.EXTERNAL_CALL,
    "call_tool_mcp": EventType.EXTERNAL_CALL,
    "push": EventType.EXTERNAL_CALL,
    "send_pr": EventType.EXTERNAL_CALL,
    # Control flow
    "finish": EventType.OUTPUT_GENERATED,
    "reject": EventType.ERROR_RAISED,
    "pause": EventType.STATE_CHANGE,
    "resume": EventType.STATE_CHANGE,
    "stop": EventType.STATE_CHANGE,
    "change_agent_state": EventType.STATE_CHANGE,
    # Memory / recall
    "recall": EventType.RESOURCE_ACCESS,
    # System / nulls
    "start": EventType.INSTRUCTION_RECEIVED,
    "system": EventType.INSTRUCTION_RECEIVED,
    "null": EventType.STATE_CHANGE,
    # Condensation (meta)
    "condensation": EventType.STATE_CHANGE,
    "condensation_request": EventType.STATE_CHANGE,
    "task_tracking": EventType.STATE_CHANGE,
    "loop_recovery": EventType.STATE_CHANGE,
}

# Observation-type → SybilCore EventType mapping
_OH_OBS_TYPE_MAP: dict[str, EventType] = {
    "run": EventType.OUTPUT_GENERATED,
    "run_ipython": EventType.OUTPUT_GENERATED,
    "read": EventType.OUTPUT_GENERATED,
    "write": EventType.STATE_CHANGE,
    "edit": EventType.STATE_CHANGE,
    "browse": EventType.OUTPUT_GENERATED,
    "browse_interactive": EventType.OUTPUT_GENERATED,
    "error": EventType.ERROR_RAISED,
    "delegate": EventType.MESSAGE_RECEIVED,
    "success": EventType.OUTPUT_GENERATED,
    "reject": EventType.ERROR_RAISED,
    "agent_state_changed": EventType.STATE_CHANGE,
    "null": EventType.STATE_CHANGE,
    "user_rejected": EventType.ERROR_RAISED,
    "file_download": EventType.RESOURCE_ACCESS,
    "mcp": EventType.EXTERNAL_CALL,
    "loop_recovery": EventType.STATE_CHANGE,
    "task_tracking": EventType.STATE_CHANGE,
}


def _resolve_agent_id(row: dict[str, Any]) -> str:
    """Extract agent identity from an OpenHands event.

    OpenHands records which actor produced the event via ``source``:
      - "agent"       → the autonomous AI agent
      - "user"        → the human operator
      - "environment" → the runtime sandbox (not a behavioral agent)

    We use the session_id + source to construct a stable identifier.
    When source is missing we default to "agent" (most common in corpus).
    """
    source = row.get("source", "agent")
    session_id = row.get("session_id", "openhands-session")
    if source in ("user", "environment"):
        return f"{session_id}/{source}"
    return f"{session_id}/agent"


def _parse_timestamp(ts_str: str | None) -> datetime:
    """Parse an ISO-8601 timestamp string to UTC-aware datetime.

    Falls back to now(UTC) if missing (common in test fixtures that
    omit timestamps from serialized history).
    """
    if not ts_str:
        return datetime.now(UTC)
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _build_content(row: dict[str, Any]) -> str:
    """Build a readable content string for a SybilCore Event.

    OpenHands events carry structured data in ``args`` (actions) and
    ``content`` + ``extras`` (observations).  We synthesize a short
    human-readable string with the most diagnostic fields first.
    """
    parts: list[str] = []

    # Top-level message field (always present on actions)
    if msg := row.get("message"):
        parts.append(str(msg)[:300])

    # Action-specific payload
    args = row.get("args", {})
    if isinstance(args, dict):
        for key in ("command", "code", "path", "url", "thought"):
            val = args.get(key)
            if val:
                parts.append(f"{key}={str(val)[:200]}")

    # Observation content (can be very long — shell output, file content)
    obs_content = row.get("content")
    if obs_content and not parts:
        parts.append(str(obs_content)[:400])

    raw = " | ".join(parts)
    return raw[:MAX_CONTENT_LENGTH]


def _build_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Preserve all OpenHands-specific fields in SybilCore metadata.

    Everything is carried through so brain modules can inspect any field:
    exit_code, security_risk, thought, cost metrics, etc.
    """
    meta: dict[str, Any] = {
        "source_system": "openhands",
        "oh_id": row.get("id"),
        "oh_source": row.get("source", "agent"),
        "oh_cause": row.get("cause"),
        "action": row.get("action"),
        "observation": row.get("observation"),
    }

    # Action-specific
    args = row.get("args", {})
    if isinstance(args, dict):
        meta["security_risk"] = args.get("security_risk")
        meta["thought"] = args.get("thought")
        meta["is_confirmed"] = args.get("is_confirmed")
        meta["exit_code"] = args.get("exit_code")
        # command or code payload (truncated for storage)
        for key in ("command", "code", "path", "url"):
            if val := args.get(key):
                meta[key] = str(val)[:500]

    # Observation-specific
    extras = row.get("extras", {})
    if isinstance(extras, dict):
        meta["exit_code"] = extras.get("exit_code", meta.get("exit_code"))

    # LLM cost metadata (valuable for EconomicBrain)
    if llm := row.get("llm_metrics"):
        meta["llm_metrics"] = llm

    # Tool call metadata (for tracing sub-agent spawns)
    if tc := row.get("tool_call_metadata"):
        meta["tool_call_metadata"] = tc

    return meta


def adapt_oh_event(row: dict[str, Any], session_id: str = "openhands") -> Event:
    """Convert a single OpenHands event dict to a SybilCore Event.

    Handles both action events and observation events.

    Args:
        row: A dict in OpenHands wire format (from event_to_dict).
        session_id: Session identifier to use when constructing agent_id.

    Returns:
        Frozen SybilCore Event.

    Raises:
        ValueError: If row contains neither 'action' nor 'observation' key.
    """
    if "action" not in row and "observation" not in row:
        msg = (
            f"OpenHands event row has neither 'action' nor 'observation' key: {row}"
        )
        raise ValueError(msg)

    # Inject session_id for agent_id construction
    row_with_session = {**row, "session_id": session_id}
    agent_id = _resolve_agent_id(row_with_session)
    timestamp = _parse_timestamp(row.get("timestamp"))

    # Resolve EventType
    if action_type := row.get("action"):
        sc_type = _OH_ACTION_TYPE_MAP.get(action_type, EventType.MESSAGE_SENT)
        # Refine: user-sent messages are INSTRUCTION_RECEIVED, not MESSAGE_SENT
        if action_type == "message" and row.get("source") == "user":
            sc_type = EventType.INSTRUCTION_RECEIVED
    else:
        obs_type = row.get("observation", "null")
        sc_type = _OH_OBS_TYPE_MAP.get(obs_type, EventType.OUTPUT_GENERATED)

    meta = _build_metadata(row)

    return Event(
        event_id=str(uuid.uuid4()),
        agent_id=agent_id,
        event_type=sc_type,
        timestamp=timestamp,
        content=_build_content(row),
        metadata=meta,
        source="openhands",
    )


def adapt_oh_trajectory(
    history: list[dict[str, Any]], session_id: str = "openhands"
) -> list[Event]:
    """Adapt a full OpenHands trajectory history list to SybilCore Events.

    In OpenHands resolver output format, the full trajectory is stored in
    the ``history`` field of the outer record.

    Args:
        history: List of event dicts from an OpenHands trajectory.
        session_id: Session identifier string.

    Returns:
        List of SybilCore Events, one per non-empty history entry.

    Raises:
        ValueError: If any individual event fails to adapt.
    """
    events: list[Event] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        events.append(adapt_oh_event(entry, session_id=session_id))
    return events


def load_trajectory_jsonl(path: Path, limit: int | None = None) -> list[list[dict[str, Any]]]:
    """Load OpenHands resolver JSONL output file.

    Each line in the file is one complete task result dict containing a
    ``history`` list of event dicts.

    Args:
        path: Path to the .jsonl file.
        limit: Maximum number of trajectory records to return.

    Returns:
        List of history lists.
    """
    if not path.exists():
        return []
    results: list[list[dict[str, Any]]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"JSON parse error at {path}:{lineno}: {exc}"
                raise ValueError(msg) from exc
            history = record.get("history", [])
            if isinstance(history, list):
                results.append(history)
            if limit and len(results) >= limit:
                break
    return results


def compute_brain_activation_rate(
    events: list[Event],
) -> dict[str, float]:
    """Run the 13-brain default ensemble and return per-brain activation rates.

    A brain is considered "activated" (firing above baseline) if its score
    exceeds 10.0.  This threshold matches the paper's definition of a
    "non-trivial activation" (scores below 10 are the empty/insufficient-data
    regime).

    Args:
        events: SybilCore Events to score.

    Returns:
        Dict mapping brain name → score (0–100).
    """
    from sybilcore.brains import get_default_brains

    brains = get_default_brains()
    result: dict[str, float] = {}
    for brain in brains:
        score = brain.score(events)
        result[brain.name] = round(score.value, 2)
    return result
