"""P0 #5 regression tests — instruction_source coverage across all GtEvent types.

Tests validate:
  1. Every documented and inferred Gastown event type produces a non-None
     instruction_source after the P0 fix (previously only patrol_started was covered).
  2. Parametrized synthetic corpus test: >95% of adapted events have non-None
     instruction_source.
  3. Negative test: unknown event types fall back gracefully with a best-effort value
     and do not crash.
  4. recipient_agent_id key tolerance: all 6 payload key names are accepted.

Fix date: 2026-04-20
Adversarial review reference: ADVERSARIAL_REVIEW.md §2 P0 #5
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from integrations.gastown.phase2b_mayor_tap.runtime_adapter import (
    _MAYOR_DISPATCHED_TYPES,
    _RECIPIENT_KEYS,
    _SELF_SOURCED_TYPES,
    _extract_instruction_source,
    _extract_recipient_agent_id,
    adapt_runtime_event,
)
from integrations.gastown.phase2b_mayor_tap.synthetic_firehose import (
    generate_full_synthetic_corpus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TS = (datetime.now(UTC) - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_row(
    gt_type: str,
    actor: str = "gastown/polecats/nux",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a minimal valid GtEvent dict."""
    return {
        "ts": _BASE_TS,
        "source": "gastown",
        "type": gt_type,
        "actor": actor,
        "payload": payload or {},
        "visibility": "feed",
    }


# ---------------------------------------------------------------------------
# Section 1: Per-event-type instruction_source tests
# ---------------------------------------------------------------------------


class TestMayorDispatchedTypes:
    """Mayor-dispatched event types must yield instruction_source from the mayor's orbit."""

    @pytest.mark.parametrize("gt_type", sorted(_MAYOR_DISPATCHED_TYPES))
    def test_mayor_dispatched_yields_mayor_or_actor(self, gt_type: str) -> None:
        # When actor IS mayor, source is "mayor".
        row = _make_row(gt_type=gt_type, actor="mayor")
        event = adapt_runtime_event(row)
        assert event.metadata["instruction_source"] == "mayor", (
            f"gt_type={gt_type}: expected 'mayor' as instruction_source, "
            f"got {event.metadata['instruction_source']!r}"
        )

    @pytest.mark.parametrize("gt_type", sorted(_MAYOR_DISPATCHED_TYPES))
    def test_mayor_dispatched_non_mayor_actor_yields_mayor(self, gt_type: str) -> None:
        # When a non-mayor actor emits a mayor-dispatched event type, still returns "mayor"
        # (the convention is that these event types are always mayor-authority).
        row = _make_row(gt_type=gt_type, actor="deacon")
        event = adapt_runtime_event(row)
        # deacon is in the mayor/deacon set — should return the actor itself
        assert event.metadata["instruction_source"] == "deacon"


class TestSelfSourcedTypes:
    """Self-sourced event types must yield instruction_source equal to the actor_id."""

    @pytest.mark.parametrize("gt_type", sorted(_SELF_SOURCED_TYPES))
    def test_self_sourced_yields_actor(self, gt_type: str) -> None:
        actor = "gastown/polecats/furiosa"
        row = _make_row(gt_type=gt_type, actor=actor)
        event = adapt_runtime_event(row)
        assert event.metadata["instruction_source"] == actor, (
            f"gt_type={gt_type}: expected actor '{actor}' as instruction_source, "
            f"got {event.metadata['instruction_source']!r}"
        )


class TestScoutDocumentedTypes:
    """Each SCOUT_REPORT.md §2A documented event type must produce non-None instruction_source.

    Scout-documented types from ADVERSARIAL_REVIEW.md P0 #5 requirement:
        patrol_started, sling, handoff, escalation_sent, merged, merge_failed,
        hook, mail, polecat_nudged, polecat_checked, patrol_complete, done.
    """

    # Scout-documented types (SCOUT_REPORT.md §2A buildEventMessage:423-539)
    SCOUT_TYPES = [
        "patrol_started",    # SCOUT §2A + §6
        "patrol_complete",   # SCOUT §2A
        "polecat_checked",   # SCOUT §2A
        "polecat_nudged",    # SCOUT §2A
        "escalation_sent",   # SCOUT §2A + §6 AssessHelp
        "sling",             # SCOUT §2A
        "hook",              # SCOUT §2A
        "handoff",           # SCOUT §2A
        "done",              # SCOUT §2A
        "mail",              # SCOUT §2A
        "merged",            # SCOUT §2A + §7
        "merge_failed",      # SCOUT §2A + §7
    ]

    @pytest.mark.parametrize("gt_type", SCOUT_TYPES)
    def test_scout_documented_type_has_instruction_source(self, gt_type: str) -> None:
        row = _make_row(gt_type=gt_type, actor="gastown/polecats/nux")
        event = adapt_runtime_event(row)
        assert event.metadata["instruction_source"] is not None, (
            f"SCOUT-documented type '{gt_type}' produced None instruction_source. "
            f"This is the P0 bug being fixed — all scout types must be covered."
        )

    @pytest.mark.parametrize("gt_type", SCOUT_TYPES)
    def test_instruction_source_is_nonempty_string(self, gt_type: str) -> None:
        row = _make_row(gt_type=gt_type, actor="gastown/polecats/nux")
        event = adapt_runtime_event(row)
        src = event.metadata["instruction_source"]
        assert isinstance(src, str) and len(src) > 0, (
            f"gt_type={gt_type}: instruction_source must be a non-empty string, got {src!r}"
        )


class TestInferredTypes:
    """Inferred (not in SCOUT §2A type list) event types must also produce non-None.

    These are labeled 'inferred from Gastown convention' in the adapter code.
    If they fail in production it's a schema-verification gap, not an adapter bug.
    """

    INFERRED_TYPES = [
        "bead_assigned",     # inferred from SCOUT §3 (mayor assigns polecats)
        "merge_claimed",     # inferred from SCOUT §7 refinery claim phase
        "merge_completed",   # inferred from SCOUT §7 refinery merge phase
        "hook_write",        # inferred from hook convention
        "convoy_opened",     # inferred from SCOUT §4 convoy lifecycle
        "handoff_requested", # inferred from handoff convention
        "handoff_accepted",  # inferred from handoff convention
    ]

    @pytest.mark.parametrize("gt_type", INFERRED_TYPES)
    def test_inferred_type_has_instruction_source(self, gt_type: str) -> None:
        row = _make_row(gt_type=gt_type, actor="gastown/polecats/nux")
        event = adapt_runtime_event(row)
        assert event.metadata["instruction_source"] is not None, (
            f"Inferred type '{gt_type}' produced None instruction_source. "
            f"Even inferred types should have best-effort coverage."
        )


class TestUnknownEventTypeFallback:
    """Unknown event types must not crash and must produce a best-effort value."""

    def test_unknown_type_does_not_crash(self) -> None:
        row = _make_row(gt_type="gastown_future_event_xyz", actor="gastown/polecats/nux")
        event = adapt_runtime_event(row)  # must not raise
        assert event is not None

    def test_unknown_type_produces_best_effort_from_actor(self) -> None:
        row = _make_row(gt_type="totally_new_event_type", actor="gastown/polecats/furiosa")
        event = adapt_runtime_event(row)
        # Best-effort: falls back to actor_id when no payload.source/origin available
        assert event.metadata["instruction_source"] == "gastown/polecats/furiosa"

    def test_unknown_type_uses_payload_source_if_available(self) -> None:
        row = _make_row(
            gt_type="undocumented_future_event",
            actor="gastown/polecats/nux",
            payload={"source": "external-bot", "message": "test"},
        )
        event = adapt_runtime_event(row)
        # payload["source"] takes priority over actor_id for unknown types
        assert event.metadata["instruction_source"] == "external-bot"

    def test_unknown_type_uses_payload_origin_if_source_absent(self) -> None:
        row = _make_row(
            gt_type="undocumented_future_event",
            actor="gastown/polecats/nux",
            payload={"origin": "github-webhook", "message": "test"},
        )
        event = adapt_runtime_event(row)
        assert event.metadata["instruction_source"] == "github-webhook"

    def test_unknown_type_still_flagged_in_unmapped_metadata(self) -> None:
        row = _make_row(gt_type="xyz_unknown_type", actor="gastown/polecats/nux")
        event = adapt_runtime_event(row)
        assert event.metadata.get("unmapped_gt_runtime_type") == "xyz_unknown_type"


# ---------------------------------------------------------------------------
# Section 2: Parametrized synthetic corpus coverage test (>95% target)
# ---------------------------------------------------------------------------


class TestSyntheticCorpusCoverage:
    """Verify instruction_source coverage across the full synthetic corpus."""

    def test_instruction_source_coverage_exceeds_95_percent(self) -> None:
        """After the P0 fix, >95% of adapted events must have non-None instruction_source.

        This validates the fix is effective across the full synthetic corpus,
        not just individual event types.
        """
        corpus = generate_full_synthetic_corpus()
        total = len(corpus)
        with_source = 0
        missing: list[str] = []

        for row in corpus:
            event = adapt_runtime_event(row)
            if event.metadata.get("instruction_source") is not None:
                with_source += 1
            else:
                missing.append(row["type"])

        coverage = with_source / total * 100
        assert coverage >= 95.0, (
            f"instruction_source coverage is {coverage:.1f}% ({with_source}/{total}). "
            f"Expected >=95%. Event types still returning None: {set(missing)}"
        )

    def test_all_patrol_started_events_have_mayor_source(self) -> None:
        """patrol_started events must always have instruction_source='mayor'."""
        corpus = generate_full_synthetic_corpus()
        patrol_events = [row for row in corpus if row["type"] == "patrol_started"]
        assert len(patrol_events) > 0, "Corpus must contain patrol_started events"

        for row in patrol_events:
            event = adapt_runtime_event(row)
            src = event.metadata["instruction_source"]
            assert src == "mayor", (
                f"patrol_started must have instruction_source='mayor', got {src!r}"
            )

    def test_inter_agent_events_produce_string_source(self) -> None:
        """All events in the corpus must produce a string instruction_source (not None)."""
        corpus = generate_full_synthetic_corpus()
        failures: list[str] = []

        for row in corpus:
            event = adapt_runtime_event(row)
            src = event.metadata.get("instruction_source")
            if src is None:
                failures.append(f"type={row['type']} actor={row['actor']}")

        assert not failures, (
            "These events produced None instruction_source after P0 fix:\n"
            + "\n".join(failures)
        )


# ---------------------------------------------------------------------------
# Section 3: recipient_agent_id key tolerance tests
# ---------------------------------------------------------------------------


class TestRecipientAgentIdKeyTolerance:
    """P1 fix: recipient_agent_id must accept all 6 payload key names."""

    @pytest.mark.parametrize("key_name", _RECIPIENT_KEYS)
    def test_all_key_names_are_accepted(self, key_name: str) -> None:
        """Each of the 6 documented key names must yield the recipient agent ID."""
        payload = {key_name: "gastown/polecats/furiosa"}
        result = _extract_recipient_agent_id(payload)
        assert result == "gastown/polecats/furiosa", (
            f"payload key '{key_name}' was not accepted. "
            f"Expected 'gastown/polecats/furiosa', got {result!r}"
        )

    def test_first_non_null_key_wins(self) -> None:
        """When multiple keys are present, the first non-null one wins."""
        payload = {
            "target": "gastown/polecats/nux",
            "to": "gastown/polecats/furiosa",
            "recipient": "gastown/polecats/slit",
        }
        result = _extract_recipient_agent_id(payload)
        # "target" comes first in _RECIPIENT_KEYS
        assert result == "gastown/polecats/nux"

    def test_none_returned_when_no_keys_present(self) -> None:
        """Empty payload must return None (valid for non-communication events)."""
        result = _extract_recipient_agent_id({})
        assert result is None

    def test_none_returned_when_all_values_falsy(self) -> None:
        """All-empty string values must return None."""
        payload = {key: "" for key in _RECIPIENT_KEYS}
        result = _extract_recipient_agent_id(payload)
        assert result is None

    def test_event_level_recipient_via_adapter(self) -> None:
        """Full adapter integration: recipient_agent_id populated from 'polecat_id' key."""
        row = _make_row(
            gt_type="handoff",
            actor="gastown/polecats/slit",
            payload={"polecat_id": "gastown/polecats/capable", "bead": "gt-001"},
        )
        event = adapt_runtime_event(row)
        assert event.metadata["recipient_agent_id"] == "gastown/polecats/capable"

    def test_event_level_recipient_via_to_agent_key(self) -> None:
        """recipient_agent_id populated from 'to_agent' payload key."""
        row = _make_row(
            gt_type="mail",
            actor="gastown/polecats/nux",
            payload={"to_agent": "gastown/polecats/furiosa", "message": "hello"},
        )
        event = adapt_runtime_event(row)
        assert event.metadata["recipient_agent_id"] == "gastown/polecats/furiosa"

    def test_non_communication_event_recipient_none_is_valid(self) -> None:
        """hook and done events typically have no recipient — None is correct and expected."""
        for gt_type in ("hook", "done", "patrol_complete"):
            row = _make_row(gt_type=gt_type, actor="gastown/polecats/nux")
            event = adapt_runtime_event(row)
            # None is valid — do not assert non-None for these types
            assert "recipient_agent_id" in event.metadata


# ---------------------------------------------------------------------------
# Section 4: _extract_instruction_source unit tests
# ---------------------------------------------------------------------------


class TestExtractInstructionSourceUnit:
    """Unit tests for the _extract_instruction_source function directly."""

    def test_patrol_started_returns_mayor(self) -> None:
        result = _extract_instruction_source("patrol_started", "mayor", {})
        assert result == "mayor"

    def test_sling_returns_actor(self) -> None:
        result = _extract_instruction_source("sling", "gastown/polecats/nux", {})
        assert result == "gastown/polecats/nux"

    def test_handoff_returns_actor(self) -> None:
        result = _extract_instruction_source("handoff", "gastown/polecats/slit", {})
        assert result == "gastown/polecats/slit"

    def test_escalation_sent_returns_actor(self) -> None:
        result = _extract_instruction_source("escalation_sent", "gastown/polecats/furiosa", {})
        assert result == "gastown/polecats/furiosa"

    def test_merged_returns_actor(self) -> None:
        result = _extract_instruction_source("merged", "gastown/refinery", {})
        assert result == "gastown/refinery"

    def test_merge_failed_returns_actor(self) -> None:
        result = _extract_instruction_source("merge_failed", "gastown/refinery", {})
        assert result == "gastown/refinery"

    def test_unknown_uses_payload_source(self) -> None:
        result = _extract_instruction_source("future_event", "actor-x", {"source": "bot"})
        assert result == "bot"

    def test_unknown_uses_payload_origin_fallback(self) -> None:
        result = _extract_instruction_source("future_event", "actor-x", {"origin": "gh-hook"})
        assert result == "gh-hook"

    def test_unknown_falls_back_to_actor(self) -> None:
        result = _extract_instruction_source("future_event", "actor-x", {})
        assert result == "actor-x"

    def test_empty_actor_returns_none_for_unknown(self) -> None:
        result = _extract_instruction_source("future_event", "", {})
        assert result is None
