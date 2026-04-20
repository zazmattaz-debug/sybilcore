"""Claude Code session-log → SybilCore Event adapter.

Converts Claude Code session JSONL records (from ``~/.claude/projects/*/
<session-id>.jsonl``) into SybilCore ``Event`` objects for replay through
the coefficient pipeline.

Schema source: live inspection of ~/.claude/projects/-Users-zazumoloi-Desktop-
Claude-Code/*.jsonl files (2026-04-20).  All schemas confirmed against
multiple sessions ranging from 2KB to 60MB.

Wire format (one JSON object per line):
--------------------------------------------------------------------
Record type "user"     — human or scheduled-task prompt
  Fields: type, uuid, parentUuid, isSidechain, sessionId, timestamp,
          permissionMode, userType, entrypoint, cwd, gitBranch, version,
          message: { role: "user", content: <str> | <list[ContentBlock]> }

Record type "assistant" — model turn
  Fields: type, uuid, parentUuid, isSidechain, sessionId, timestamp,
          requestId, userType, entrypoint, cwd, gitBranch, version,
          message: {
            role: "assistant", model: str, id: str, stop_reason: str,
            usage: { input_tokens, output_tokens, ... },
            content: list[ContentBlock]
          }
  ContentBlock types seen in corpus:
    { type: "text",       text: str }
    { type: "thinking",   thinking: str, signature: str }
    { type: "tool_use",   id: str, name: str, input: dict }
    { type: "tool_result", tool_use_id: str, content: list[{type, text}] }

Record type "progress" — hook execution events
  Fields: type, uuid, parentUuid, parentToolUseID, toolUseID, isSidechain,
          sessionId, timestamp, userType, entrypoint, cwd, gitBranch, version,
          data: { type: "hook_progress"|"agent_progress", hookEvent?, hookName?,
                  command?, ... }

Record type "queue-operation" — scheduler enqueue/dequeue
  Fields: type, operation, sessionId, timestamp, content?

Record type "system"    — session-level metadata
Record type "attachment" — binary attachment metadata
Record type "last-prompt" — final prompt reference
--------------------------------------------------------------------

Tool names seen in corpus: Bash, Read, Write, Edit, Glob, Agent, Skill,
  TaskOutput, mcp__*, and various MCP server tools.

Mapping decisions:
  "user" message         → INSTRUCTION_RECEIVED
  "assistant" text block → MESSAGE_SENT
  "assistant" thinking   → MESSAGE_SENT  (internal monologue)
  "assistant" tool_use: Bash/Edit/Write/Read/Glob → TOOL_CALL
  "assistant" tool_use: Agent/Skill → EXTERNAL_CALL  (sub-agent spawn)
  "assistant" tool_use: mcp__* → EXTERNAL_CALL       (MCP call)
  "assistant" tool_use: Read  → RESOURCE_ACCESS
  "assistant" tool_result     → OUTPUT_GENERATED
  "progress"  hook_progress   → STATE_CHANGE          (hook lifecycle)
  "progress"  agent_progress  → STATE_CHANGE
  "queue-operation"           → STATE_CHANGE
  "system"                    → INSTRUCTION_RECEIVED

GAP NOTE: Claude Code logs capture very rich behavioral signals:
  - Tool names (exact command dispatched)
  - Token usage per turn (valuable for EconomicBrain)
  - Sub-agent spawn events (Agent tool → fan-out detection)
  - MCP tool calls (external surface)
  - isSidechain flag (sidechain / parallel branch)
  - cwd and gitBranch per turn
  - stop_reason (end_turn vs tool_use vs stop_sequence)
This is considerably richer than Gastown bead events.
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
# Tool name → EventType categorisation
# ---------------------------------------------------------------------------
# Filesystem tools: reading is RESOURCE_ACCESS; writing/editing is TOOL_CALL
# (same as Gastown's "created" → TOOL_CALL: writing is an external artifact action).
_CC_TOOL_TYPE_MAP: dict[str, EventType] = {
    "Bash": EventType.TOOL_CALL,
    "Read": EventType.RESOURCE_ACCESS,
    "Write": EventType.TOOL_CALL,
    "Edit": EventType.TOOL_CALL,
    "Glob": EventType.RESOURCE_ACCESS,
    "Grep": EventType.RESOURCE_ACCESS,
    # Sub-agent and skill invocations
    "Agent": EventType.EXTERNAL_CALL,
    "Skill": EventType.EXTERNAL_CALL,
    "TaskOutput": EventType.OUTPUT_GENERATED,
    "SendMessage": EventType.MESSAGE_SENT,
}

# Prefix-based matching for MCP tools (all become EXTERNAL_CALL)
_MCP_PREFIX = "mcp__"

# Record-level type → EventType mapping (for non-assistant records)
_CC_RECORD_TYPE_MAP: dict[str, EventType] = {
    "user": EventType.INSTRUCTION_RECEIVED,
    "system": EventType.INSTRUCTION_RECEIVED,
    "queue-operation": EventType.STATE_CHANGE,
    "attachment": EventType.RESOURCE_ACCESS,
    "progress": EventType.STATE_CHANGE,
    "last-prompt": EventType.STATE_CHANGE,
}


def _resolve_agent_id(row: dict[str, Any]) -> str:
    """Construct a stable agent identifier from a session log record.

    Claude Code does not embed a per-turn agent identity — the session
    is the agent.  We use sessionId + role to distinguish the model from
    the user within a session.

    Schema:
        sessionId is always present.
        userType: "external" (human), "scheduled-task", etc.
        The assistant is identified by type=="assistant".
    """
    session_id = row.get("sessionId", "cc-unknown")
    record_type = row.get("type", "unknown")
    user_type = row.get("userType", "")

    if record_type == "assistant":
        return f"{session_id}/assistant"
    if record_type == "user":
        if user_type == "external":
            return f"{session_id}/user"
        return f"{session_id}/{user_type or 'user'}"
    return f"{session_id}/system"


def _parse_timestamp(ts_str: str | None) -> datetime:
    """Parse Claude Code ISO-8601 timestamp (always 'Z'-suffixed) to UTC datetime."""
    if not ts_str:
        return datetime.now(UTC)
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _resolve_tool_event_type(tool_name: str) -> EventType:
    """Map a Claude Code tool name to SybilCore EventType."""
    if tool_name.startswith(_MCP_PREFIX):
        return EventType.EXTERNAL_CALL
    return _CC_TOOL_TYPE_MAP.get(tool_name, EventType.TOOL_CALL)


def _build_content_for_assistant(message: dict[str, Any]) -> str:
    """Build content string from an assistant message dict.

    Prioritises the first tool_use or text block.
    """
    parts: list[str] = []
    content = message.get("content", [])
    if not isinstance(content, list):
        return str(content)[:MAX_CONTENT_LENGTH]

    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "tool_use":
            name = block.get("name", "")
            inp = block.get("input", {})
            # Most useful field from input depending on tool
            inp_str = ""
            if isinstance(inp, dict):
                for key in ("command", "file_path", "pattern", "old_string", "prompt"):
                    if val := inp.get(key):
                        inp_str = f"{key}={str(val)[:200]}"
                        break
            parts.append(f"tool:{name} {inp_str}".strip())
        elif block_type == "text":
            text = block.get("text", "")
            if text:
                parts.append(str(text)[:300])
        elif block_type == "thinking":
            thinking = block.get("thinking", "")
            if thinking:
                parts.append(f"[thinking] {str(thinking)[:200]}")
        elif block_type == "tool_result":
            result_content = block.get("content", [])
            if isinstance(result_content, list):
                for rc in result_content[:1]:
                    if isinstance(rc, dict) and rc.get("type") == "text":
                        parts.append(f"[result] {str(rc.get('text',''))[:200]}")
            elif isinstance(result_content, str):
                parts.append(f"[result] {result_content[:200]}")

    raw = " | ".join(parts)
    return raw[:MAX_CONTENT_LENGTH]


def _build_content_for_user(message: dict[str, Any]) -> str:
    """Build content string from a user message dict."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content[:MAX_CONTENT_LENGTH]
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", ""))[:300])
            elif isinstance(block, str):
                parts.append(block[:300])
        return " | ".join(parts)[:MAX_CONTENT_LENGTH]
    return str(content)[:MAX_CONTENT_LENGTH]


def _build_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Preserve all Claude Code fields in SybilCore metadata dict."""
    meta: dict[str, Any] = {
        "source_system": "claude_code",
        "cc_uuid": row.get("uuid"),
        "cc_session_id": row.get("sessionId"),
        "cc_parent_uuid": row.get("parentUuid"),
        "cc_is_sidechain": row.get("isSidechain", False),
        "cc_record_type": row.get("type"),
        "cc_entrypoint": row.get("entrypoint"),
        "cc_cwd": row.get("cwd"),
        "cc_git_branch": row.get("gitBranch"),
        "cc_version": row.get("version"),
        "cc_user_type": row.get("userType"),
    }

    message = row.get("message", {})
    if isinstance(message, dict):
        meta["cc_model"] = message.get("model")
        meta["cc_stop_reason"] = message.get("stop_reason")
        usage = message.get("usage", {})
        if isinstance(usage, dict):
            meta["cc_input_tokens"] = usage.get("input_tokens")
            meta["cc_output_tokens"] = usage.get("output_tokens")
            meta["cc_cache_read_tokens"] = usage.get("cache_read_input_tokens")

        # Extract tool use details
        content = message.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    meta["cc_tool_name"] = block.get("name")
                    inp = block.get("input", {})
                    if isinstance(inp, dict):
                        for key in ("command", "file_path", "description"):
                            if val := inp.get(key):
                                meta[f"cc_tool_{key}"] = str(val)[:200]
                    break

    # Progress-specific
    data = row.get("data", {})
    if isinstance(data, dict):
        meta["cc_progress_type"] = data.get("type")
        meta["cc_hook_event"] = data.get("hookEvent")
        meta["cc_hook_name"] = data.get("hookName")
        meta["cc_hook_command"] = str(data.get("command", ""))[:200]

    return meta


def _classify_assistant_record(row: dict[str, Any]) -> EventType:
    """Classify an assistant record based on its content blocks.

    Precedence: tool_use > tool_result > thinking > text.
    """
    message = row.get("message", {})
    if not isinstance(message, dict):
        return EventType.MESSAGE_SENT

    content = message.get("content", [])
    if not isinstance(content, list):
        return EventType.MESSAGE_SENT

    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "tool_use":
            tool_name = block.get("name", "")
            return _resolve_tool_event_type(tool_name)
        if block_type == "tool_result":
            return EventType.OUTPUT_GENERATED

    # No tool blocks — just text/thinking
    return EventType.MESSAGE_SENT


def adapt_cc_record(row: dict[str, Any]) -> Event:
    """Convert a single Claude Code session JSONL record to a SybilCore Event.

    Handles all record types: user, assistant, progress, system,
    queue-operation, attachment, last-prompt.

    Args:
        row: Parsed JSON dict from a Claude Code .jsonl session file.

    Returns:
        Frozen SybilCore Event.

    Raises:
        ValueError: If the row has no 'type' field.
    """
    record_type = row.get("type")
    if not record_type:
        msg = f"Claude Code record missing 'type' field: {row}"
        raise ValueError(msg)

    agent_id = _resolve_agent_id(row)
    timestamp = _parse_timestamp(row.get("timestamp"))

    # EventType resolution
    if record_type == "assistant":
        sc_type = _classify_assistant_record(row)
    else:
        sc_type = _CC_RECORD_TYPE_MAP.get(record_type, EventType.MESSAGE_SENT)

    # Content
    message = row.get("message", {})
    if record_type == "assistant" and isinstance(message, dict):
        content = _build_content_for_assistant(message)
    elif record_type == "user" and isinstance(message, dict):
        content = _build_content_for_user(message)
    elif record_type == "queue-operation":
        content = str(row.get("content", ""))[:MAX_CONTENT_LENGTH]
    else:
        content = ""

    meta = _build_metadata(row)

    return Event(
        event_id=str(uuid.uuid4()),
        agent_id=agent_id,
        event_type=sc_type,
        timestamp=timestamp,
        content=content,
        metadata=meta,
        source="claude_code",
    )


def load_session_jsonl(path: Path, max_records: int = 5000) -> list[dict[str, Any]]:
    """Read a Claude Code session JSONL file, respecting a safety cap.

    Session files can be up to 60MB (1c3bcfff session observed).  We cap
    at max_records to avoid OOM in analysis paths.

    Args:
        path: Path to a session .jsonl file.
        max_records: Maximum number of records to load.

    Returns:
        List of parsed dicts, at most max_records long.
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
            if len(rows) >= max_records:
                break
    return rows


def adapt_session_file(path: Path, max_records: int = 5000) -> list[Event]:
    """Load a Claude Code session JSONL file and adapt all records to SybilCore Events.

    Args:
        path: Path to a session .jsonl file.
        max_records: Cap on records to avoid OOM with 60MB sessions.

    Returns:
        List of SybilCore Events, one per valid non-empty line.
    """
    rows = load_session_jsonl(path, max_records=max_records)
    return [adapt_cc_record(row) for row in rows]


def compute_brain_activation_rate(
    events: list[Event],
) -> dict[str, float]:
    """Run the 13-brain default ensemble and return per-brain scores.

    A score above 10.0 is considered an "activation" (non-trivial signal).
    This matches the threshold used in the phase2f cross-substrate analysis.

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
