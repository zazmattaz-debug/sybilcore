"""Tests for the Gastown bead-event adapter (Phase 1A baseline).

Verifies:
  - Adapter produces valid SybilCore Event objects from real fixture samples.
  - Timestamp parsing round-trips correctly (RFC3339 Z-suffix → tz-aware datetime).
  - Agent grouping assigns events to the correct agent_id.
  - CoefficientCalculator accepts adapted events without raising.
  - All known Gastown event types are mapped to valid SybilCore EventTypes.
  - Edge cases: missing fields, unknown event types.

Fixture file used: /tmp/gastown-fixtures/.beads/backup/events.jsonl
If the fixture file is absent (CI without network), tests are skipped via a marker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Import the adapter under test.  The module lives outside sybilcore/ so we
# import it by file path via importlib when running from the tests/ directory.
# ---------------------------------------------------------------------------
from integrations.gastown.phase1a_baseline.adapter import (
    _GT_EVENT_TYPE_MAP,
    _parse_timestamp,
    adapt_bead_event,
    adapt_fixture_file,
    group_events_by_agent,
    load_jsonl,
)
from sybilcore.brains import get_default_brains
from sybilcore.core.coefficient import CoefficientCalculator
from sybilcore.models.event import Event, EventType

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_FIXTURE_PATH = Path("/tmp/gastown-fixtures/.beads/backup/events.jsonl")

_FIXTURE_AVAILABLE = _FIXTURE_PATH.exists()

_SKIP_REASON = (
    "Gastown fixture not available; "
    "run: git clone https://github.com/gastownhall/gastown.git /tmp/gastown-fixtures"
)
pytestmark_fixture = pytest.mark.skipif(not _FIXTURE_AVAILABLE, reason=_SKIP_REASON)

# ---------------------------------------------------------------------------
# Inline sample rows (no fixture file needed — for unit tests)
# ---------------------------------------------------------------------------

_SAMPLE_STATUS_CHANGED: dict[str, Any] = {
    "id": 1,
    "issue_id": "gt-69dai",
    "event_type": "status_changed",
    "actor": "mayor",
    "created_at": "2026-02-26T15:13:23Z",
    "new_value": '{"assignee":"gastown/polecats/furiosa","status":"hooked"}',
    "old_value": '{"id":"gt-69dai","status":"open"}',
    "comment": None,
}

_SAMPLE_CLOSED: dict[str, Any] = {
    "id": 6,
    "issue_id": "gt-69dai",
    "event_type": "closed",
    "actor": "gastown/polecats/furiosa",
    "created_at": "2026-02-26T15:14:53Z",
    "new_value": "no-changes: Test failure was already fixed by commit dad02f33.",
    "old_value": "",
    "comment": None,
}

_SAMPLE_UPDATED: dict[str, Any] = {
    "id": 2,
    "issue_id": "gt-69dai",
    "event_type": "updated",
    "actor": "mayor",
    "created_at": "2026-02-26T15:13:23Z",
    "new_value": '{"description":"attached_molecule: gt-wisp-l4k6j"}',
    "old_value": '{"id":"gt-69dai"}',
    "comment": None,
}

_SAMPLE_CREATED: dict[str, Any] = {
    "id": 10,
    "issue_id": "gt-newbead",
    "event_type": "created",
    "actor": "gastown/refinery",
    "created_at": "2026-02-27T10:00:00Z",
    "new_value": '{"id":"gt-newbead","title":"New bead"}',
    "old_value": "",
    "comment": None,
}


# ---------------------------------------------------------------------------
# Unit tests (no fixture file needed)
# ---------------------------------------------------------------------------


class TestTimestampParsing:
    """Timestamp parsing round-trip tests."""

    def test_z_suffix_parsed_as_utc(self) -> None:
        ts = "2026-02-26T15:13:23Z"
        dt = _parse_timestamp(ts)
        assert dt.tzinfo is not None, "datetime must be timezone-aware"
        assert dt.utcoffset().total_seconds() == 0, "timezone should be UTC (offset=0)"  # type: ignore[union-attr]

    def test_year_month_day_preserved(self) -> None:
        dt = _parse_timestamp("2026-03-05T15:46:56Z")
        assert dt.year == 2026
        assert dt.month == 3
        assert dt.day == 5

    def test_time_preserved(self) -> None:
        dt = _parse_timestamp("2026-02-26T15:13:23Z")
        assert dt.hour == 15
        assert dt.minute == 13
        assert dt.second == 23

    def test_offset_format_also_works(self) -> None:
        """Non-Z offset strings should also parse."""
        dt = _parse_timestamp("2026-02-26T15:13:23+00:00")
        assert dt.tzinfo is not None

    def test_invalid_string_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            _parse_timestamp("not-a-timestamp")


class TestAdaptBeadEvent:
    """Tests for the core adapt_bead_event function."""

    def test_status_changed_maps_to_state_change(self) -> None:
        event = adapt_bead_event(_SAMPLE_STATUS_CHANGED)
        assert event.event_type == EventType.STATE_CHANGE

    def test_closed_maps_to_output_generated(self) -> None:
        event = adapt_bead_event(_SAMPLE_CLOSED)
        assert event.event_type == EventType.OUTPUT_GENERATED

    def test_updated_maps_to_message_sent(self) -> None:
        event = adapt_bead_event(_SAMPLE_UPDATED)
        assert event.event_type == EventType.MESSAGE_SENT

    def test_created_maps_to_tool_call(self) -> None:
        event = adapt_bead_event(_SAMPLE_CREATED)
        assert event.event_type == EventType.TOOL_CALL

    def test_agent_id_is_actor_field(self) -> None:
        event = adapt_bead_event(_SAMPLE_STATUS_CHANGED)
        assert event.agent_id == "mayor"

    def test_polecat_agent_id_preserved(self) -> None:
        event = adapt_bead_event(_SAMPLE_CLOSED)
        assert event.agent_id == "gastown/polecats/furiosa"

    def test_source_is_gastown(self) -> None:
        event = adapt_bead_event(_SAMPLE_STATUS_CHANGED)
        assert event.source == "gastown"

    def test_timestamp_is_timezone_aware(self) -> None:
        event = adapt_bead_event(_SAMPLE_STATUS_CHANGED)
        assert event.timestamp.tzinfo is not None

    def test_timestamp_value_matches(self) -> None:
        event = adapt_bead_event(_SAMPLE_STATUS_CHANGED)
        assert event.timestamp.year == 2026
        assert event.timestamp.month == 2
        assert event.timestamp.day == 26

    def test_metadata_contains_bead_id(self) -> None:
        event = adapt_bead_event(_SAMPLE_STATUS_CHANGED)
        assert event.metadata["bead_id"] == "gt-69dai"

    def test_metadata_contains_beads_event_id(self) -> None:
        event = adapt_bead_event(_SAMPLE_STATUS_CHANGED)
        assert event.metadata["beads_event_id"] == 1

    def test_metadata_contains_gastown_event_type(self) -> None:
        event = adapt_bead_event(_SAMPLE_STATUS_CHANGED)
        assert event.metadata["gastown_event_type"] == "status_changed"

    def test_metadata_preserves_new_value(self) -> None:
        event = adapt_bead_event(_SAMPLE_CLOSED)
        assert "no-changes" in (event.metadata["new_value"] or "")

    def test_event_is_pydantic_model(self) -> None:
        event = adapt_bead_event(_SAMPLE_STATUS_CHANGED)
        assert isinstance(event, Event)

    def test_content_is_non_empty_string(self) -> None:
        event = adapt_bead_event(_SAMPLE_STATUS_CHANGED)
        assert isinstance(event.content, str)
        assert len(event.content) > 0

    def test_content_does_not_exceed_max(self) -> None:
        from sybilcore.core.config import MAX_CONTENT_LENGTH

        event = adapt_bead_event(_SAMPLE_STATUS_CHANGED)
        assert len(event.content) <= MAX_CONTENT_LENGTH

    def test_missing_actor_raises(self) -> None:
        row = dict(_SAMPLE_STATUS_CHANGED)
        del row["actor"]
        with pytest.raises(ValueError, match="actor"):
            adapt_bead_event(row)

    def test_missing_event_type_raises(self) -> None:
        row = dict(_SAMPLE_STATUS_CHANGED)
        del row["event_type"]
        with pytest.raises(ValueError, match="event_type"):
            adapt_bead_event(row)

    def test_missing_created_at_raises(self) -> None:
        row = dict(_SAMPLE_STATUS_CHANGED)
        del row["created_at"]
        with pytest.raises(ValueError, match="created_at"):
            adapt_bead_event(row)

    def test_unknown_event_type_falls_back_to_message_sent(self) -> None:
        row = dict(_SAMPLE_STATUS_CHANGED)
        row["event_type"] = "totally_unknown_type_xyz"
        event = adapt_bead_event(row)
        assert event.event_type == EventType.MESSAGE_SENT
        assert event.metadata.get("unmapped_event_type") == "totally_unknown_type_xyz"

    def test_all_known_gastown_types_are_mapped(self) -> None:
        """Every key in _GT_EVENT_TYPE_MAP must produce a valid EventType."""
        for gt_type, sc_type in _GT_EVENT_TYPE_MAP.items():
            assert isinstance(sc_type, EventType), (
                f"Gastown type '{gt_type}' maps to non-EventType value: {sc_type!r}"
            )


class TestGroupEventsByAgent:
    """Tests for the grouping helper."""

    def test_groups_by_agent_id(self) -> None:
        events = [
            adapt_bead_event(_SAMPLE_STATUS_CHANGED),  # mayor
            adapt_bead_event(_SAMPLE_CLOSED),  # gastown/polecats/furiosa
            adapt_bead_event(_SAMPLE_UPDATED),  # mayor
        ]
        groups = group_events_by_agent(events)
        assert "mayor" in groups
        assert "gastown/polecats/furiosa" in groups
        assert len(groups["mayor"]) == 2
        assert len(groups["gastown/polecats/furiosa"]) == 1

    def test_events_sorted_chronologically_within_group(self) -> None:
        # status_changed and updated both at same time for mayor; closed is later.
        events = [
            adapt_bead_event(_SAMPLE_CLOSED),
            adapt_bead_event(_SAMPLE_STATUS_CHANGED),
            adapt_bead_event(_SAMPLE_UPDATED),
        ]
        groups = group_events_by_agent(events)
        furiosa_events = groups["gastown/polecats/furiosa"]
        assert furiosa_events[0].event_type == EventType.OUTPUT_GENERATED

    def test_empty_list_returns_empty_dict(self) -> None:
        assert group_events_by_agent([]) == {}


class TestCalculatorAcceptsAdaptedEvents:
    """Integration smoke test: CoefficientCalculator must not raise on adapted events."""

    def test_calculator_accepts_status_changed_event(self) -> None:
        event = adapt_bead_event(_SAMPLE_STATUS_CHANGED)
        brains = get_default_brains()
        calculator = CoefficientCalculator(window_seconds=90 * 24 * 3600)
        snapshot = calculator.scan_agent("mayor", [event], brains)
        assert 0.0 <= snapshot.coefficient <= 500.0

    def test_calculator_accepts_mixed_events(self) -> None:
        events = [
            adapt_bead_event(_SAMPLE_STATUS_CHANGED),
            adapt_bead_event(_SAMPLE_UPDATED),
        ]
        brains = get_default_brains()
        calculator = CoefficientCalculator(window_seconds=90 * 24 * 3600)
        snapshot = calculator.scan_agent("mayor", events, brains)
        assert snapshot.tier is not None


# ---------------------------------------------------------------------------
# Fixture-file tests (skipped if fixture absent)
# ---------------------------------------------------------------------------


@pytestmark_fixture
class TestFixtureFileLoading:
    """Tests that require the real Gastown fixture file."""

    def test_load_jsonl_returns_list_of_dicts(self) -> None:
        rows = load_jsonl(_FIXTURE_PATH)
        assert isinstance(rows, list)
        assert len(rows) > 0
        assert all(isinstance(r, dict) for r in rows)

    def test_fixture_has_expected_event_types(self) -> None:
        rows = load_jsonl(_FIXTURE_PATH)
        types_seen = {r.get("event_type") for r in rows}
        # All of these must be present in the fixture.
        expected = {"status_changed", "updated", "closed", "created"}
        assert expected.issubset(types_seen), (
            f"Expected event types {expected} but only found {types_seen}"
        )

    def test_adapt_fixture_file_returns_events(self) -> None:
        events = adapt_fixture_file(_FIXTURE_PATH)
        assert len(events) > 0
        assert all(isinstance(e, Event) for e in events)

    def test_all_events_have_timezone_aware_timestamps(self) -> None:
        events = adapt_fixture_file(_FIXTURE_PATH)
        naive = [e for e in events if e.timestamp.tzinfo is None]
        assert len(naive) == 0, f"{len(naive)} events have naive timestamps"

    def test_all_events_have_nonempty_agent_id(self) -> None:
        events = adapt_fixture_file(_FIXTURE_PATH)
        empty_agents = [e for e in events if not e.agent_id]
        assert len(empty_agents) == 0

    def test_all_events_have_valid_event_type(self) -> None:
        events = adapt_fixture_file(_FIXTURE_PATH)
        valid_types = set(EventType)
        for e in events:
            assert e.event_type in valid_types, f"Invalid EventType: {e.event_type!r}"

    def test_source_is_gastown_for_all_events(self) -> None:
        events = adapt_fixture_file(_FIXTURE_PATH)
        wrong = [e for e in events if e.source != "gastown"]
        assert len(wrong) == 0

    def test_agent_grouping_covers_all_events(self) -> None:
        events = adapt_fixture_file(_FIXTURE_PATH)
        groups = group_events_by_agent(events)
        total = sum(len(v) for v in groups.values())
        assert total == len(events), "Grouping lost events"

    def test_mayor_is_most_active_agent(self) -> None:
        events = adapt_fixture_file(_FIXTURE_PATH)
        groups = group_events_by_agent(events)
        assert "mayor" in groups
        mayor_count = len(groups["mayor"])
        for agent_id, agent_events in groups.items():
            if agent_id != "mayor":
                assert mayor_count >= len(agent_events), (
                    f"{agent_id} has more events ({len(agent_events)}) than mayor ({mayor_count})"
                )

    def test_calculator_accepts_full_fixture_events(self) -> None:
        """End-to-end smoke test: all fixture events through the full pipeline."""
        events = adapt_fixture_file(_FIXTURE_PATH)
        brains = get_default_brains()
        calculator = CoefficientCalculator(window_seconds=90 * 24 * 3600)
        groups = group_events_by_agent(events)

        # Score just the two most active agents to keep test time reasonable.
        for agent_id in ["mayor", "gastown/polecats/nux"]:
            if agent_id not in groups:
                continue
            snapshot = calculator.scan_agent(agent_id, events, brains)
            assert 0.0 <= snapshot.coefficient <= 500.0
            assert snapshot.tier is not None
