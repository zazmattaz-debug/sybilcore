"""Phase 2F cross-substrate smoke tests.

Tests that SybilCore's core abstractions generalize to two alternative
multi-agent substrates: OpenHands and Claude Code.

Coverage assertions:
  - Each adapter round-trips a sample event to a valid SybilCore Event.
  - EventType mapping is complete and deterministic.
  - Brain activation rates are computable (no crashes) across both substrates.
  - Sample events exercise as many EventType variants as possible.
  - Schema fields land in metadata with correct keys.
  - Edge cases: missing timestamps, missing content, sidechain events.
  - Brain activation comparison between substrates.

Minimum 15 tests required (see Phase 2F spec).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from integrations.gastown.phase2f_cross_substrate.claude_code_adapter import (
    _CC_TOOL_TYPE_MAP,
    _resolve_tool_event_type,
    adapt_cc_record,
)
from integrations.gastown.phase2f_cross_substrate.claude_code_adapter import (
    compute_brain_activation_rate as cc_brain_activation_rate,
)
from integrations.gastown.phase2f_cross_substrate.openhands_adapter import (
    _OH_ACTION_TYPE_MAP,
    _OH_OBS_TYPE_MAP,
    adapt_oh_event,
    adapt_oh_trajectory,
)
from integrations.gastown.phase2f_cross_substrate.openhands_adapter import (
    compute_brain_activation_rate as oh_brain_activation_rate,
)
from sybilcore.models.event import Event, EventType

# ---------------------------------------------------------------------------
# Shared sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def oh_cmd_run_action() -> dict[str, Any]:
    """Minimal OpenHands CmdRunAction dict."""
    return {
        "id": 1,
        "timestamp": "2026-03-01T12:00:00.000Z",
        "source": "agent",
        "action": "run",
        "message": "Running command: ls -la",
        "args": {
            "command": "ls -la /workspace",
            "thought": "I need to see the workspace contents",
            "is_confirmed": "confirmed",
            "security_risk": "UNKNOWN",
        },
    }


@pytest.fixture()
def oh_message_action_user() -> dict[str, Any]:
    """OpenHands user message action (INSTRUCTION_RECEIVED)."""
    return {
        "id": 2,
        "timestamp": "2026-03-01T12:00:01.000Z",
        "source": "user",
        "action": "message",
        "message": "Please fix the license file",
        "args": {"content": "Please fix the license file"},
    }


@pytest.fixture()
def oh_observation_run() -> dict[str, Any]:
    """OpenHands run observation (CmdOutputObservation)."""
    return {
        "id": 3,
        "timestamp": "2026-03-01T12:00:02.000Z",
        "source": "environment",
        "observation": "run",
        "content": "total 12\ndrwxr-xr-x  3 root root 4096 Mar  1 12:00 .\n",
        "extras": {"exit_code": 0, "command": "ls -la /workspace"},
    }


@pytest.fixture()
def oh_error_observation() -> dict[str, Any]:
    """OpenHands error observation."""
    return {
        "id": 4,
        "timestamp": "2026-03-01T12:00:03.000Z",
        "source": "environment",
        "observation": "error",
        "content": "Command failed: bash: command not found",
        "extras": {"exit_code": 127},
    }


@pytest.fixture()
def oh_finish_action() -> dict[str, Any]:
    """OpenHands finish action (task completion)."""
    return {
        "id": 5,
        "timestamp": "2026-03-01T12:00:04.000Z",
        "source": "agent",
        "action": "finish",
        "message": "Task completed successfully",
        "args": {"outputs": {"result": "LICENSE file created"}, "thought": "Done"},
    }


@pytest.fixture()
def oh_full_trajectory(
    oh_cmd_run_action: dict[str, Any],
    oh_message_action_user: dict[str, Any],
    oh_observation_run: dict[str, Any],
    oh_error_observation: dict[str, Any],
    oh_finish_action: dict[str, Any],
) -> list[dict[str, Any]]:
    """A minimal but realistic OpenHands trajectory."""
    return [
        oh_message_action_user,
        oh_cmd_run_action,
        oh_observation_run,
        oh_error_observation,
        oh_finish_action,
    ]


@pytest.fixture()
def cc_user_record() -> dict[str, Any]:
    """Claude Code user record."""
    return {
        "type": "user",
        "uuid": "aa-bb-cc",
        "parentUuid": None,
        "isSidechain": False,
        "sessionId": "test-session-001",
        "timestamp": "2026-03-15T10:00:00.000Z",
        "permissionMode": "default",
        "userType": "external",
        "entrypoint": "claude-desktop",
        "cwd": "/Users/zazumoloi/Desktop/Claude Code",
        "gitBranch": "main",
        "version": "2.1.86",
        "message": {"role": "user", "content": "Build the auth middleware"},
    }


@pytest.fixture()
def cc_assistant_bash_record() -> dict[str, Any]:
    """Claude Code assistant record with Bash tool_use."""
    return {
        "type": "assistant",
        "uuid": "dd-ee-ff",
        "parentUuid": "aa-bb-cc",
        "isSidechain": False,
        "sessionId": "test-session-001",
        "timestamp": "2026-03-15T10:00:01.000Z",
        "requestId": "req-001",
        "userType": "external",
        "entrypoint": "claude-desktop",
        "cwd": "/Users/zazumoloi/Desktop/Claude Code",
        "gitBranch": "main",
        "version": "2.1.86",
        "message": {
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "id": "msg-001",
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 1500,
                "output_tokens": 200,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 0,
            },
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "Bash",
                    "input": {"command": "pytest tests/ -v", "timeout": 60000},
                }
            ],
        },
    }


@pytest.fixture()
def cc_assistant_agent_spawn() -> dict[str, Any]:
    """Claude Code assistant record spawning a sub-agent (EXTERNAL_CALL)."""
    return {
        "type": "assistant",
        "uuid": "gg-hh-ii",
        "parentUuid": "aa-bb-cc",
        "isSidechain": False,
        "sessionId": "test-session-001",
        "timestamp": "2026-03-15T10:00:02.000Z",
        "requestId": "req-002",
        "userType": "external",
        "entrypoint": "claude-desktop",
        "cwd": "/Users/zazumoloi/Desktop/Claude Code",
        "gitBranch": "main",
        "version": "2.1.86",
        "message": {
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "id": "msg-002",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 2000, "output_tokens": 300},
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_02",
                    "name": "Agent",
                    "input": {
                        "description": "Research competing libraries",
                        "prompt": "Research competing libraries for auth middleware",
                    },
                }
            ],
        },
    }


@pytest.fixture()
def cc_assistant_mcp_record() -> dict[str, Any]:
    """Claude Code assistant record calling an MCP tool (EXTERNAL_CALL)."""
    return {
        "type": "assistant",
        "uuid": "jj-kk-ll",
        "parentUuid": "aa-bb-cc",
        "isSidechain": True,
        "sessionId": "test-session-001",
        "timestamp": "2026-03-15T10:00:03.000Z",
        "requestId": "req-003",
        "userType": "external",
        "entrypoint": "claude-desktop",
        "cwd": "/Users/zazumoloi/Desktop/Claude Code",
        "gitBranch": "main",
        "version": "2.1.86",
        "message": {
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "id": "msg-003",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 500, "output_tokens": 100},
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_03",
                    "name": "mcp__Claude_in_Chrome__navigate",
                    "input": {"url": "https://example.com"},
                }
            ],
        },
    }


@pytest.fixture()
def cc_progress_record() -> dict[str, Any]:
    """Claude Code progress / hook record."""
    return {
        "type": "progress",
        "uuid": "mm-nn-oo",
        "parentUuid": "dd-ee-ff",
        "parentToolUseID": "toolu_01",
        "toolUseID": "toolu_01",
        "isSidechain": False,
        "sessionId": "test-session-001",
        "timestamp": "2026-03-15T10:00:01.500Z",
        "userType": "external",
        "entrypoint": "claude-desktop",
        "cwd": "/Users/zazumoloi/Desktop/Claude Code",
        "gitBranch": "main",
        "version": "2.1.86",
        "data": {
            "type": "hook_progress",
            "hookEvent": "PreToolUse",
            "hookName": "security-precommit",
            "command": "python3 scripts/security_precommit.py",
        },
    }


# ---------------------------------------------------------------------------
# Group 1: OpenHands adapter unit tests (8 tests)
# ---------------------------------------------------------------------------

class TestOpenHandsAdapterUnit:
    """Unit tests for the OpenHands adapter."""

    def test_oh_cmd_run_action_adapts_to_tool_call(
        self, oh_cmd_run_action: dict[str, Any]
    ) -> None:
        """CmdRunAction (action='run') maps to TOOL_CALL."""
        event = adapt_oh_event(oh_cmd_run_action, session_id="test-oh")
        assert isinstance(event, Event)
        assert event.event_type == EventType.TOOL_CALL
        assert event.source == "openhands"
        assert "test-oh/agent" in event.agent_id

    def test_oh_user_message_maps_to_instruction_received(
        self, oh_message_action_user: dict[str, Any]
    ) -> None:
        """User-sourced message action maps to INSTRUCTION_RECEIVED."""
        event = adapt_oh_event(oh_message_action_user, session_id="test-oh")
        assert event.event_type == EventType.INSTRUCTION_RECEIVED
        assert "user" in event.agent_id

    def test_oh_observation_run_maps_to_output_generated(
        self, oh_observation_run: dict[str, Any]
    ) -> None:
        """run observation (command result) maps to OUTPUT_GENERATED."""
        event = adapt_oh_event(oh_observation_run, session_id="test-oh")
        assert event.event_type == EventType.OUTPUT_GENERATED

    def test_oh_error_observation_maps_to_error_raised(
        self, oh_error_observation: dict[str, Any]
    ) -> None:
        """Error observation maps to ERROR_RAISED."""
        event = adapt_oh_event(oh_error_observation, session_id="test-oh")
        assert event.event_type == EventType.ERROR_RAISED

    def test_oh_finish_action_maps_to_output_generated(
        self, oh_finish_action: dict[str, Any]
    ) -> None:
        """finish action maps to OUTPUT_GENERATED."""
        event = adapt_oh_event(oh_finish_action, session_id="test-oh")
        assert event.event_type == EventType.OUTPUT_GENERATED

    def test_oh_event_has_valid_timestamp(
        self, oh_cmd_run_action: dict[str, Any]
    ) -> None:
        """Adapted event has timezone-aware timestamp not in the future."""
        event = adapt_oh_event(oh_cmd_run_action, session_id="test-oh")
        assert event.timestamp.tzinfo is not None
        assert event.timestamp <= datetime.now(UTC)

    def test_oh_event_metadata_preserves_security_risk(
        self, oh_cmd_run_action: dict[str, Any]
    ) -> None:
        """security_risk from args lands in metadata."""
        event = adapt_oh_event(oh_cmd_run_action, session_id="test-oh")
        assert event.metadata.get("security_risk") == "UNKNOWN"
        assert event.metadata.get("source_system") == "openhands"

    def test_oh_event_raises_on_missing_action_and_observation(self) -> None:
        """Row with neither 'action' nor 'observation' raises ValueError."""
        bad_row: dict[str, Any] = {"id": 99, "timestamp": "2026-01-01T00:00:00Z"}
        with pytest.raises(ValueError, match="neither 'action' nor 'observation'"):
            adapt_oh_event(bad_row)

    def test_oh_trajectory_adapts_full_list(
        self, oh_full_trajectory: list[dict[str, Any]]
    ) -> None:
        """adapt_oh_trajectory processes all 5 events without error."""
        events = adapt_oh_trajectory(oh_full_trajectory, session_id="test-oh")
        assert len(events) == 5
        assert all(isinstance(e, Event) for e in events)
        assert all(e.source == "openhands" for e in events)

    def test_oh_action_type_coverage(self) -> None:
        """All mapped action types produce valid SybilCore EventTypes."""
        valid_types = set(EventType)
        for action_type, sc_type in _OH_ACTION_TYPE_MAP.items():
            assert sc_type in valid_types, (
                f"Action '{action_type}' maps to invalid EventType '{sc_type}'"
            )

    def test_oh_observation_type_coverage(self) -> None:
        """All mapped observation types produce valid SybilCore EventTypes."""
        valid_types = set(EventType)
        for obs_type, sc_type in _OH_OBS_TYPE_MAP.items():
            assert sc_type in valid_types, (
                f"Observation '{obs_type}' maps to invalid EventType '{sc_type}'"
            )

    def test_oh_missing_timestamp_falls_back_to_now(self) -> None:
        """Event with no timestamp adapts without crashing (falls back to now())."""
        row: dict[str, Any] = {
            "action": "run",
            "source": "agent",
            "message": "Run ls",
            "args": {"command": "ls"},
        }
        event = adapt_oh_event(row, session_id="test-oh")
        assert event.timestamp <= datetime.now(UTC)

    def test_oh_brain_activation_rate_returns_13_brains(
        self, oh_full_trajectory: list[dict[str, Any]]
    ) -> None:
        """Brain activation rate dict has exactly 13 entries."""
        events = adapt_oh_trajectory(oh_full_trajectory, session_id="test-oh")
        rates = oh_brain_activation_rate(events)
        assert len(rates) == 13
        # All scores in valid 0-100 range
        for name, score in rates.items():
            assert 0.0 <= score <= 100.0, (
                f"Brain '{name}' returned score {score} outside [0, 100]"
            )


# ---------------------------------------------------------------------------
# Group 2: Claude Code adapter unit tests (8 tests)
# ---------------------------------------------------------------------------

class TestClaudeCodeAdapterUnit:
    """Unit tests for the Claude Code adapter."""

    def test_cc_user_record_maps_to_instruction_received(
        self, cc_user_record: dict[str, Any]
    ) -> None:
        """User record maps to INSTRUCTION_RECEIVED."""
        event = adapt_cc_record(cc_user_record)
        assert event.event_type == EventType.INSTRUCTION_RECEIVED
        assert event.source == "claude_code"

    def test_cc_bash_tool_use_maps_to_tool_call(
        self, cc_assistant_bash_record: dict[str, Any]
    ) -> None:
        """Assistant with Bash tool_use maps to TOOL_CALL."""
        event = adapt_cc_record(cc_assistant_bash_record)
        assert event.event_type == EventType.TOOL_CALL

    def test_cc_agent_spawn_maps_to_external_call(
        self, cc_assistant_agent_spawn: dict[str, Any]
    ) -> None:
        """Assistant spawning an Agent sub-agent maps to EXTERNAL_CALL."""
        event = adapt_cc_record(cc_assistant_agent_spawn)
        assert event.event_type == EventType.EXTERNAL_CALL

    def test_cc_mcp_tool_maps_to_external_call(
        self, cc_assistant_mcp_record: dict[str, Any]
    ) -> None:
        """MCP tool call (mcp__*) maps to EXTERNAL_CALL."""
        event = adapt_cc_record(cc_assistant_mcp_record)
        assert event.event_type == EventType.EXTERNAL_CALL

    def test_cc_progress_record_maps_to_state_change(
        self, cc_progress_record: dict[str, Any]
    ) -> None:
        """Progress hook record maps to STATE_CHANGE."""
        event = adapt_cc_record(cc_progress_record)
        assert event.event_type == EventType.STATE_CHANGE

    def test_cc_event_has_valid_timestamp(
        self, cc_user_record: dict[str, Any]
    ) -> None:
        """Adapted event has timezone-aware timestamp not in the future."""
        event = adapt_cc_record(cc_user_record)
        assert event.timestamp.tzinfo is not None
        assert event.timestamp <= datetime.now(UTC)

    def test_cc_metadata_preserves_token_usage(
        self, cc_assistant_bash_record: dict[str, Any]
    ) -> None:
        """Token usage from message.usage lands in metadata (EconomicBrain signal)."""
        event = adapt_cc_record(cc_assistant_bash_record)
        assert event.metadata.get("cc_input_tokens") == 1500
        assert event.metadata.get("cc_output_tokens") == 200
        assert event.metadata.get("cc_cache_read_tokens") == 800

    def test_cc_metadata_preserves_tool_name(
        self, cc_assistant_bash_record: dict[str, Any]
    ) -> None:
        """Tool name from tool_use content block lands in metadata."""
        event = adapt_cc_record(cc_assistant_bash_record)
        assert event.metadata.get("cc_tool_name") == "Bash"

    def test_cc_sidechain_flag_preserved_in_metadata(
        self, cc_assistant_mcp_record: dict[str, Any]
    ) -> None:
        """isSidechain=True is preserved in metadata for SwarmDetectionBrain."""
        event = adapt_cc_record(cc_assistant_mcp_record)
        assert event.metadata.get("cc_is_sidechain") is True

    def test_cc_event_raises_on_missing_type(self) -> None:
        """Record with no 'type' field raises ValueError."""
        bad_row: dict[str, Any] = {"uuid": "xxx", "sessionId": "sess-1"}
        with pytest.raises(ValueError, match="missing 'type' field"):
            adapt_cc_record(bad_row)

    def test_cc_tool_type_map_complete_for_known_tools(self) -> None:
        """Known tool names map to valid SybilCore EventTypes."""
        valid_types = set(EventType)
        for tool_name, sc_type in _CC_TOOL_TYPE_MAP.items():
            assert sc_type in valid_types, (
                f"Tool '{tool_name}' maps to invalid EventType '{sc_type}'"
            )

    def test_cc_mcp_prefix_detection(self) -> None:
        """Any mcp__ prefixed tool resolves to EXTERNAL_CALL."""
        assert _resolve_tool_event_type("mcp__anything__tool") == EventType.EXTERNAL_CALL
        assert _resolve_tool_event_type("mcp__slack__send") == EventType.EXTERNAL_CALL

    def test_cc_brain_activation_rate_returns_13_brains(
        self,
        cc_user_record: dict[str, Any],
        cc_assistant_bash_record: dict[str, Any],
        cc_assistant_agent_spawn: dict[str, Any],
    ) -> None:
        """Brain activation rate dict has exactly 13 entries from CC events."""
        events = [
            adapt_cc_record(cc_user_record),
            adapt_cc_record(cc_assistant_bash_record),
            adapt_cc_record(cc_assistant_agent_spawn),
        ]
        rates = cc_brain_activation_rate(events)
        assert len(rates) == 13
        for name, score in rates.items():
            assert 0.0 <= score <= 100.0, (
                f"Brain '{name}' returned score {score} outside [0, 100]"
            )


# ---------------------------------------------------------------------------
# Group 3: Cross-substrate comparison tests (5 tests)
# ---------------------------------------------------------------------------

class TestCrossSubstrateComparison:
    """Tests comparing behavior across both substrates."""

    def test_both_adapters_produce_valid_event_types(
        self,
        oh_cmd_run_action: dict[str, Any],
        cc_assistant_bash_record: dict[str, Any],
    ) -> None:
        """Both adapters produce Events with valid EventType enum values."""
        oh_event = adapt_oh_event(oh_cmd_run_action, session_id="test")
        cc_event = adapt_cc_record(cc_assistant_bash_record)
        valid_types = set(EventType)
        assert oh_event.event_type in valid_types
        assert cc_event.event_type in valid_types

    def test_both_adapters_tag_source_system_in_metadata(
        self,
        oh_cmd_run_action: dict[str, Any],
        cc_assistant_bash_record: dict[str, Any],
    ) -> None:
        """Both adapters populate metadata.source_system for provenance."""
        oh_event = adapt_oh_event(oh_cmd_run_action, session_id="test")
        cc_event = adapt_cc_record(cc_assistant_bash_record)
        assert oh_event.metadata["source_system"] == "openhands"
        assert cc_event.metadata["source_system"] == "claude_code"

    def test_cc_provides_token_cost_signal_absent_in_oh(
        self,
        oh_cmd_run_action: dict[str, Any],
        cc_assistant_bash_record: dict[str, Any],
    ) -> None:
        """Claude Code events carry token costs; OpenHands action events do not.

        This is one of the structural advantages of Claude Code as a substrate:
        EconomicBrain has richer signal.
        """
        oh_event = adapt_oh_event(oh_cmd_run_action, session_id="test")
        cc_event = adapt_cc_record(cc_assistant_bash_record)
        assert oh_event.metadata.get("cc_input_tokens") is None
        assert cc_event.metadata.get("cc_input_tokens") == 1500

    def test_both_substrates_fire_nonzero_brains(
        self, oh_full_trajectory: list[dict[str, Any]]
    ) -> None:
        """At least one brain fires (score > 0) on each substrate's sample corpus.

        Uses a 5-event OpenHands trajectory and 3 CC records.  The test
        checks that the ensemble does not silently return all-zeros, which
        would indicate an adapter mapping that strips all signal.
        """
        # OpenHands
        oh_events = adapt_oh_trajectory(oh_full_trajectory, session_id="test")
        oh_rates = oh_brain_activation_rate(oh_events)
        oh_nonzero = sum(1 for s in oh_rates.values() if s > 0.0)

        # Claude Code
        cc_records: list[dict[str, Any]] = [
            {
                "type": "user",
                "uuid": "u1",
                "parentUuid": None,
                "isSidechain": False,
                "sessionId": "sess-cmp",
                "timestamp": "2026-03-01T10:00:00.000Z",
                "userType": "external",
                "entrypoint": "claude-desktop",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.86",
                "message": {"role": "user", "content": "Run the security audit"},
            },
            {
                "type": "assistant",
                "uuid": "a1",
                "parentUuid": "u1",
                "isSidechain": False,
                "sessionId": "sess-cmp",
                "timestamp": "2026-03-01T10:00:01.000Z",
                "requestId": "req-cmp",
                "userType": "external",
                "entrypoint": "claude-desktop",
                "cwd": "/tmp",
                "gitBranch": "main",
                "version": "2.1.86",
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "id": "msg-cmp",
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 800, "output_tokens": 100},
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_cmp",
                            "name": "Bash",
                            "input": {"command": "cat /etc/passwd"},
                        }
                    ],
                },
            },
        ]
        cc_events = [adapt_cc_record(r) for r in cc_records]
        cc_rates = cc_brain_activation_rate(cc_events)
        cc_nonzero = sum(1 for s in cc_rates.values() if s > 0.0)

        # At least 1 brain must fire on each substrate
        assert oh_nonzero >= 1, (
            f"All brains returned 0 on OpenHands sample. Rates: {oh_rates}"
        )
        assert cc_nonzero >= 1, (
            f"All brains returned 0 on Claude Code sample. Rates: {cc_rates}"
        )

    def test_cc_sidechain_events_preserve_parent_lineage(
        self, cc_assistant_mcp_record: dict[str, Any]
    ) -> None:
        """Sidechain records retain parentUuid in metadata for SwarmDetectionBrain.

        SwarmDetectionBrain uses parentUuid-level fan-out to detect collusion.
        This test confirms the signal survives adapter round-trip.
        """
        event = adapt_cc_record(cc_assistant_mcp_record)
        assert event.metadata.get("cc_parent_uuid") == "aa-bb-cc"
        assert event.metadata.get("cc_is_sidechain") is True
