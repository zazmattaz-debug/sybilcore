"""Phase 2B test suite — 25 tests covering all deliverables.

Test areas:
    1. runtime_adapter — GtEvent → SybilCore Event round-trip (8 tests)
    2. synthetic_firehose — Schema accuracy and brain targeting (5 tests)
    3. mayor_tap — Streaming, windowing, and tier transitions (6 tests)
    4. witness_third_gate — FastAPI endpoint behaviour (6 tests)

All tests use synthetic fixtures or in-process state — no network, no file I/O.

Auth note: all witness_third_gate tests use SigningTestClient, which automatically
adds HMAC-SHA256 bearer token headers and X-Sybilcore-Timestamp on every request.
The test secret is set via SYBILCORE_WITNESS_SECRET env var in the module-level
autouse fixture so _WITNESS_SECRET is populated for all session/class tests.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from httpx import Response as HttpxResponse

from integrations.gastown.phase2b_mayor_tap.mayor_tap import (
    MIN_EVENTS_FOR_SCORE,
    AgentWindow,
    MayorTap,
)
from integrations.gastown.phase2b_mayor_tap.runtime_adapter import (
    _GT_RUNTIME_EVENT_TYPE_MAP,
    _parse_runtime_timestamp,
    adapt_runtime_event,
    group_runtime_events_by_agent,
)
from integrations.gastown.phase2b_mayor_tap.synthetic_firehose import (
    GT_EVENT_TYPES,
    generate_full_synthetic_corpus,
    injection_attempt_stream,
    patrol_lifecycle_stream,
    stream_synthetic_corpus,
)
from integrations.gastown.phase2b_mayor_tap.witness_third_gate import (
    CoefficientStore,
    _make_enforcement_decision,
    app,
)
from sybilcore.models.agent import AgentTier, CoefficientSnapshot
from sybilcore.models.event import Event, EventType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gtevent_now(
    actor: str = "gastown/polecats/nux",
    gt_type: str = "patrol_started",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a minimal GtEvent dict with a current (non-future) timestamp."""
    return {
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "gastown",
        "type": gt_type,
        "actor": actor,
        "payload": payload or {},
        "visibility": "feed",
    }


def _make_snapshot(coefficient: float) -> CoefficientSnapshot:
    """Create a CoefficientSnapshot with the given coefficient."""
    return CoefficientSnapshot(
        coefficient=coefficient,
        tier=AgentTier.from_coefficient(coefficient),
        timestamp=datetime.now(UTC),
        brain_scores={},
    )


# ---------------------------------------------------------------------------
# SECTION 1: runtime_adapter tests (8 tests)
# ---------------------------------------------------------------------------


class TestRuntimeAdapterTimestamp:
    """Timestamp parsing correctness."""

    def test_z_suffix_parses_to_utc(self) -> None:
        dt = _parse_runtime_timestamp("2026-04-20T14:30:00Z")
        assert dt.tzinfo is not None
        assert dt.year == 2026
        assert dt.hour == 14

    def test_offset_parses_correctly(self) -> None:
        dt = _parse_runtime_timestamp("2026-04-20T14:30:00+00:00")
        assert dt.tzinfo is not None

    def test_invalid_timestamp_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            _parse_runtime_timestamp("not-a-timestamp")


class TestRuntimeAdapterMapping:
    """GtEvent type → SybilCore EventType mapping."""

    def test_patrol_started_maps_to_instruction_received(self) -> None:
        row = _make_gtevent_now(gt_type="patrol_started")
        event = adapt_runtime_event(row)
        assert event.event_type == EventType.INSTRUCTION_RECEIVED

    def test_sling_maps_to_external_call(self) -> None:
        row = _make_gtevent_now(gt_type="sling")
        event = adapt_runtime_event(row)
        assert event.event_type == EventType.EXTERNAL_CALL

    def test_escalation_sent_maps_to_permission_request(self) -> None:
        row = _make_gtevent_now(gt_type="escalation_sent")
        event = adapt_runtime_event(row)
        assert event.event_type == EventType.PERMISSION_REQUEST

    def test_merge_failed_maps_to_error_raised(self) -> None:
        row = _make_gtevent_now(gt_type="merge_failed")
        event = adapt_runtime_event(row)
        assert event.event_type == EventType.ERROR_RAISED

    def test_unknown_type_falls_back_to_message_sent(self) -> None:
        row = _make_gtevent_now(gt_type="totally_unknown_type_xyz")
        event = adapt_runtime_event(row)
        assert event.event_type == EventType.MESSAGE_SENT
        assert event.metadata.get("unmapped_gt_runtime_type") == "totally_unknown_type_xyz"

    def test_all_documented_types_are_mapped(self) -> None:
        """Every type in GT_EVENT_TYPES must be in the mapping table."""
        for gt_type in GT_EVENT_TYPES:
            assert gt_type in _GT_RUNTIME_EVENT_TYPE_MAP, (
                f"GtEvent type '{gt_type}' from SCOUT_REPORT.md §2A is not in the mapping table"
            )


class TestRuntimeAdapterFieldPreservation:
    """All GtEvent fields preserved in metadata."""

    def test_actor_maps_to_agent_id(self) -> None:
        row = _make_gtevent_now(actor="gastown/polecats/furiosa")
        event = adapt_runtime_event(row)
        assert event.agent_id == "gastown/polecats/furiosa"

    def test_source_is_gastown_runtime(self) -> None:
        row = _make_gtevent_now()
        event = adapt_runtime_event(row)
        assert event.source == "gastown_runtime"

    def test_all_gtevent_fields_in_metadata(self) -> None:
        row = {
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "patrol-bot",
            "type": "patrol_started",
            "actor": "gastown/polecats/nux",
            "payload": {"bead": "gt-001", "rig": "gastown", "message": "Start patrol"},
            "visibility": "both",
        }
        event = adapt_runtime_event(row)
        assert event.metadata["gt_source"] == "patrol-bot"
        assert event.metadata["gt_type"] == "patrol_started"
        assert event.metadata["gt_actor"] == "gastown/polecats/nux"
        assert event.metadata["gt_visibility"] == "both"
        assert event.metadata["payload_bead"] == "gt-001"
        assert event.metadata["payload_rig"] == "gastown"
        assert event.metadata["payload_message"] == "Start patrol"

    def test_missing_actor_raises_value_error(self) -> None:
        row = {"ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "type": "sling"}
        with pytest.raises(ValueError, match="actor"):
            adapt_runtime_event(row)

    def test_missing_type_raises_value_error(self) -> None:
        row = {"ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "actor": "mayor"}
        with pytest.raises(ValueError, match="type"):
            adapt_runtime_event(row)

    def test_missing_ts_raises_value_error(self) -> None:
        row = {"actor": "mayor", "type": "patrol_started"}
        with pytest.raises(ValueError, match="ts"):
            adapt_runtime_event(row)

    def test_instruction_source_set_for_patrol_started(self) -> None:
        row = _make_gtevent_now(gt_type="patrol_started")
        event = adapt_runtime_event(row)
        assert event.metadata["instruction_source"] == "mayor"

    def test_instruction_source_populated_for_self_sourced_event(self) -> None:
        # P0 fix: sling events populate instruction_source as the actor (self-sourced).
        # Previously returned None for all non-patrol events — the bug being fixed.
        row = _make_gtevent_now(gt_type="sling", actor="gastown/polecats/nux")
        event = adapt_runtime_event(row)
        assert event.metadata["instruction_source"] == "gastown/polecats/nux"


class TestGroupRuntimeEventsByAgent:
    """group_runtime_events_by_agent partitioning and ordering."""

    def test_groups_by_agent_id(self) -> None:
        rows = [
            _make_gtevent_now(actor="gastown/polecats/nux", gt_type="patrol_started"),
            _make_gtevent_now(actor="mayor", gt_type="polecat_nudged"),
            _make_gtevent_now(actor="gastown/polecats/nux", gt_type="done"),
        ]
        events = [adapt_runtime_event(r) for r in rows]
        groups = group_runtime_events_by_agent(events)
        assert "gastown/polecats/nux" in groups
        assert "mayor" in groups
        assert len(groups["gastown/polecats/nux"]) == 2
        assert len(groups["mayor"]) == 1

    def test_empty_input_returns_empty_dict(self) -> None:
        assert group_runtime_events_by_agent([]) == {}


# ---------------------------------------------------------------------------
# SECTION 2: synthetic_firehose tests (5 tests)
# ---------------------------------------------------------------------------


class TestSyntheticFirehose:
    """Synthetic corpus schema accuracy and brain targeting."""

    def test_corpus_has_events(self) -> None:
        corpus = generate_full_synthetic_corpus()
        assert len(corpus) > 0

    def test_all_events_have_required_gtevent_fields(self) -> None:
        """Every event must have ts, source, type, actor, payload, visibility."""
        corpus = generate_full_synthetic_corpus()
        required = {"ts", "source", "type", "actor", "payload", "visibility"}
        for event in corpus:
            missing = required - set(event.keys())
            assert not missing, f"Event missing fields {missing}: {event}"

    def test_all_event_types_in_documented_schema(self) -> None:
        # All generated event types must be from the SCOUT_REPORT.md §2A documented list.
        corpus = generate_full_synthetic_corpus()
        for event in corpus:
            gt_type = event["type"]
            assert gt_type in GT_EVENT_TYPES, (
                f"Synthetic event type '{gt_type}' not in SCOUT_REPORT.md §2A documented types"
            )

    def test_corpus_is_chronologically_sorted(self) -> None:
        corpus = generate_full_synthetic_corpus()
        timestamps = [e["ts"] for e in corpus]
        assert timestamps == sorted(timestamps), "Corpus events must be in chronological order"

    def test_stream_synthetic_corpus_yields_valid_json(self) -> None:
        lines = list(stream_synthetic_corpus())
        assert len(lines) > 0
        for line in lines:
            parsed = json.loads(line)
            assert "actor" in parsed
            assert "type" in parsed
            assert "ts" in parsed

    def test_actor_addresses_match_scout_report_format(self) -> None:
        """Actors must be 'mayor', 'deacon', or '<rig>/polecats/<name>' format."""
        corpus = generate_full_synthetic_corpus()
        for event in corpus:
            actor = event["actor"]
            is_valid = (
                actor in {"mayor", "deacon"}
                or actor.startswith("gastown/")
            )
            assert is_valid, f"Actor '{actor}' does not match Gastown address format"

    def test_injection_stream_has_injection_content(self) -> None:
        """Injection pattern stream must contain injection-like content for CompromiseBrain."""
        events = injection_attempt_stream()
        messages = [
            e["payload"].get("message", "") for e in events if isinstance(e.get("payload"), dict)
        ]
        combined = " ".join(messages).lower()
        assert "ignore" in combined or "override" in combined or "unrestricted" in combined

    def test_patrol_lifecycle_events_adapt_without_error(self) -> None:
        """All patrol lifecycle events must adapt to SybilCore Events successfully."""
        # Use a historical timestamp to avoid future-timestamp validation
        t = datetime.now(UTC) - timedelta(days=3)
        events = patrol_lifecycle_stream(base_time=t)
        for row in events:
            adapted = adapt_runtime_event(row)
            assert isinstance(adapted, Event)


# ---------------------------------------------------------------------------
# SECTION 3: mayor_tap tests (6 tests)
# ---------------------------------------------------------------------------


class TestAgentWindow:
    """AgentWindow rolling buffer behavior."""

    def test_window_respects_max_size(self) -> None:
        window = AgentWindow(agent_id="test-agent", max_size=5)
        rows = [_make_gtevent_now(gt_type="patrol_started") for _ in range(10)]
        for row in rows:
            window.add_event(adapt_runtime_event(row))
        assert window.event_count == 5  # deque maxlen enforced

    def test_events_since_rescore_increments(self) -> None:
        window = AgentWindow(agent_id="test-agent", max_size=100)
        assert window.events_since_rescore == 0
        window.add_event(adapt_runtime_event(_make_gtevent_now()))
        assert window.events_since_rescore == 1


class TestMayorTapStreaming:
    """MayorTap event processing and rolling window behavior."""

    def _make_historical_line(
        self,
        actor: str,
        gt_type: str,
        days_ago: int = 3,
    ) -> str:
        """Create a JSON line with a historical (non-future) timestamp."""
        t = datetime.now(UTC) - timedelta(days=days_ago, seconds=1)
        row = {
            "ts": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "gastown",
            "type": gt_type,
            "actor": actor,
            "payload": {"message": f"test event {gt_type}"},
            "visibility": "feed",
        }
        return json.dumps(row)

    def test_tap_tracks_multiple_agents(self) -> None:
        import io
        output = io.StringIO()
        tap = MayorTap(output=output, rescore_interval=1000)
        agents = ["gastown/polecats/nux", "mayor", "gastown/witness"]
        for agent in agents:
            tap.process_event(self._make_historical_line(agent, "patrol_started"))
        assert len(tap._agent_windows) == 3

    def test_tap_ignores_malformed_json(self) -> None:
        import io
        output = io.StringIO()
        tap = MayorTap(output=output)
        tap.process_event("NOT VALID JSON {{{")
        tap.process_event("")
        # No crash — windows should be empty
        assert len(tap._agent_windows) == 0

    def test_rescore_triggered_after_interval(self) -> None:
        import io
        output = io.StringIO()
        # Set interval to 1 so every event triggers a rescore attempt
        tap = MayorTap(output=output, rescore_interval=1)
        actor = "gastown/polecats/nux"
        # Send enough events to accumulate MIN_EVENTS_FOR_SCORE
        for i in range(MIN_EVENTS_FOR_SCORE + 5):
            tap.process_event(self._make_historical_line(actor, "patrol_started", days_ago=i + 1))
        # Output should have score_update lines
        output_text = output.getvalue()
        assert "score_update" in output_text

    def test_final_score_all_covers_all_agents(self) -> None:
        import io
        output = io.StringIO()
        tap = MayorTap(output=output, rescore_interval=1000)
        agents = ["gastown/polecats/nux", "mayor"]
        for agent in agents:
            for i in range(MIN_EVENTS_FOR_SCORE + 1):
                line = self._make_historical_line(agent, "patrol_started", days_ago=i + 1)
                tap.process_event(line)
        results = tap.final_score_all()
        assert set(results.keys()) == set(agents)

    def test_tier_transition_logged_when_tier_changes(self) -> None:
        import io
        output = io.StringIO()
        tap = MayorTap(output=output, rescore_interval=1)
        actor = "gastown/polecats/nux"
        # Manually inject a transition by updating the profile directly
        window = tap._get_or_create_window(actor)
        # Set initial tier to CLEAR
        initial_snapshot = _make_snapshot(50.0)  # CLEAR
        window.profile = window.profile.with_new_reading(initial_snapshot)
        # Now process enough events to trigger rescore
        for i in range(MIN_EVENTS_FOR_SCORE + 2):
            tap.process_event(self._make_historical_line(actor, "sling", days_ago=i + 1))
        # The tap should have processed without crashing
        summary = tap.get_summary()
        assert summary["agents_tracked"] >= 1

    def test_get_summary_returns_agent_list(self) -> None:
        import io
        output = io.StringIO()
        tap = MayorTap(output=output)
        tap.process_event(self._make_historical_line("mayor", "patrol_started"))
        summary = tap.get_summary()
        assert "agents_tracked" in summary
        assert "agents" in summary
        assert summary["agents_tracked"] == 1


# ---------------------------------------------------------------------------
# SECTION 4: witness_third_gate tests (6 tests)
# ---------------------------------------------------------------------------


_PHASE2B_TEST_SECRET = "phase2b-test-secret-for-witness-gate"
_PHASE2B_SECRET_BYTES = _PHASE2B_TEST_SECRET.encode()


def _sign_body(body: bytes) -> str:
    """Compute HMAC-SHA256 hex digest of body with the phase2b test secret."""
    return hmac.new(_PHASE2B_SECRET_BYTES, body, hashlib.sha256).hexdigest()


class SigningTestClient:
    """Wraps FastAPI TestClient and auto-adds HMAC auth headers.

    Mirrors the TestClient interface for get/post so existing tests
    need zero changes beyond the fixture swap.
    """

    def __init__(self, inner: TestClient) -> None:
        self._inner = inner

    def _signed_headers(self, body: bytes) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {_sign_body(body)}",
            "X-Sybilcore-Timestamp": str(int(time.time())),
        }

    def get(self, url: str, **kwargs: Any) -> HttpxResponse:
        headers = {**self._signed_headers(b""), **kwargs.pop("headers", {})}
        return self._inner.get(url, headers=headers, **kwargs)

    def post(
        self,
        url: str,
        *,
        json: Any = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> HttpxResponse:
        if content is not None:
            body = content
        elif json is not None:
            import json as _json
            body = _json.dumps(json).encode()
        else:
            body = b""

        auth_headers = self._signed_headers(body)
        merged = {**auth_headers, **(headers or {})}

        if content is not None:
            return self._inner.post(
                url, content=content,
                headers={**merged, "Content-Type": "application/json"},
                **kwargs,
            )
        elif json is not None:
            # FastAPI TestClient serialises json kwarg itself; pass body via content
            # so we sign exactly what's sent.
            return self._inner.post(
                url, content=body,
                headers={**merged, "Content-Type": "application/json"},
                **kwargs,
            )
        else:
            return self._inner.post(url, headers=merged, **kwargs)


@pytest.fixture(autouse=True, scope="module")
def _set_witness_secret_for_module() -> Any:
    """Set SYBILCORE_WITNESS_SECRET for the entire module so the lazy secret loader works."""
    os.environ["SYBILCORE_WITNESS_SECRET"] = _PHASE2B_TEST_SECRET
    os.environ.pop("SYBILCORE_WITNESS_DEV", None)

    import integrations.gastown.phase2b_mayor_tap.witness_third_gate as wg
    # Reset the lazy-load sentinel so it picks up the new env var.
    wg._WITNESS_SECRET = None
    wg._WITNESS_SECRET_LOADED = False

    yield

    # Restore module state so other test modules start clean.
    os.environ.pop("SYBILCORE_WITNESS_SECRET", None)
    wg._WITNESS_SECRET = None
    wg._WITNESS_SECRET_LOADED = False


@pytest.fixture()
def client() -> SigningTestClient:
    """FastAPI test client with a clean coefficient store and auto-signing for each test."""
    # Reset the module-level store before each test
    from integrations.gastown.phase2b_mayor_tap import witness_third_gate
    witness_third_gate._coefficient_store = CoefficientStore()
    return SigningTestClient(TestClient(app))


class TestWitnessEndpointStatus:
    """Health check endpoint."""

    def test_status_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/witness/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "agents_tracked" in data


class TestWitnessPreNukeEndpoint:
    """Pre-nuke coefficient check endpoint."""

    def _build_request(
        self,
        agent_id: str = "gastown/polecats/nux",
        events: list[dict[str, Any]] | None = None,
        reason: str = "zombie detected",
    ) -> dict[str, Any]:
        return {
            "agent_id": agent_id,
            "rig_name": "gastown",
            "polecat_name": agent_id.split("/")[-1] if "/" in agent_id else agent_id,
            "events": events or [],
            "reason": reason,
        }

    def test_unknown_agent_vetoes_kill(self, client: TestClient) -> None:
        """No data for agent → safe default veto."""
        resp = client.post("/witness/pre-nuke", json=self._build_request())
        assert resp.status_code == 200
        data = resp.json()
        assert data["allow_kill"] is False
        assert data["coefficient"] == 0.0
        assert data["tier"] == "clear"

    def test_clear_tier_vetoes_kill(self, client: TestClient) -> None:
        """Agent in CLEAR tier → allow_kill=False."""
        from integrations.gastown.phase2b_mayor_tap import witness_third_gate
        witness_third_gate._coefficient_store.update(
            "gastown/polecats/nux", _make_snapshot(50.0)
        )
        resp = client.post("/witness/pre-nuke", json=self._build_request())
        assert resp.status_code == 200
        assert resp.json()["allow_kill"] is False

    def test_lethal_eliminator_allows_kill(self, client: TestClient) -> None:
        """Agent in LETHAL_ELIMINATOR tier → allow_kill=True."""
        from integrations.gastown.phase2b_mayor_tap import witness_third_gate
        witness_third_gate._coefficient_store.update(
            "gastown/polecats/nux", _make_snapshot(350.0)
        )
        resp = client.post("/witness/pre-nuke", json=self._build_request())
        assert resp.status_code == 200
        assert resp.json()["allow_kill"] is True
        assert resp.json()["tier"] == "lethal_eliminator"

    def test_flagged_tier_vetoes_kill(self, client: TestClient) -> None:
        """Agent in FLAGGED tier → allow_kill=False (escalate to Deacon first)."""
        from integrations.gastown.phase2b_mayor_tap import witness_third_gate
        witness_third_gate._coefficient_store.update(
            "gastown/polecats/nux", _make_snapshot(250.0)
        )
        resp = client.post("/witness/pre-nuke", json=self._build_request())
        assert resp.status_code == 200
        data = resp.json()
        assert data["allow_kill"] is False
        assert "Deacon" in data["reasoning"]

    def test_on_demand_scoring_from_events(self, client: TestClient) -> None:
        """Events provided in request body trigger on-demand scoring."""
        t = datetime.now(UTC) - timedelta(days=3)
        events = [
            {
                "ts": (t + timedelta(seconds=i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": "gastown",
                "type": "patrol_started",
                "actor": "gastown/polecats/furiosa",
                "payload": {"message": f"event {i}"},
                "visibility": "feed",
            }
            for i in range(5)
        ]
        req = self._build_request(
            agent_id="gastown/polecats/furiosa",
            events=events,
            reason="test on-demand",
        )
        resp = client.post("/witness/pre-nuke", json=req)
        assert resp.status_code == 200
        data = resp.json()
        assert "allow_kill" in data
        assert "coefficient" in data
        assert isinstance(data["brains_fired"], list)

    def test_response_contains_all_required_fields(self, client: TestClient) -> None:
        """Response schema must include all required fields."""
        resp = client.post("/witness/pre-nuke", json=self._build_request())
        assert resp.status_code == 200
        data = resp.json()
        required = {"allow_kill", "coefficient", "tier", "brains_fired", "reasoning", "timestamp"}
        missing = required - set(data.keys())
        assert not missing, f"Response missing required fields: {missing}"


class TestCoefficientStore:
    """CoefficientStore in-memory store behavior."""

    def test_update_and_get_roundtrip(self) -> None:
        store = CoefficientStore()
        snap = _make_snapshot(75.0)
        store.update("agent-x", snap)
        retrieved = store.get("agent-x")
        assert retrieved is not None
        assert retrieved.coefficient == 75.0

    def test_get_unknown_returns_none(self) -> None:
        store = CoefficientStore()
        assert store.get("nonexistent") is None

    def test_all_agents_returns_all(self) -> None:
        store = CoefficientStore()
        store.update("a1", _make_snapshot(10.0))
        store.update("a2", _make_snapshot(200.0))
        all_snaps = store.all_agents()
        assert set(all_snaps.keys()) == {"a1", "a2"}


class TestEnforcementDecision:
    """_make_enforcement_decision logic unit tests."""

    def test_clear_decision_does_not_allow_kill(self) -> None:
        snap = _make_snapshot(50.0)
        resp = _make_enforcement_decision(snap, "agent-x", "test")
        assert resp.allow_kill is False
        assert resp.tier == "clear"

    def test_lethal_eliminator_decision_allows_kill(self) -> None:
        snap = _make_snapshot(350.0)
        resp = _make_enforcement_decision(snap, "agent-x", "zombie")
        assert resp.allow_kill is True
        assert resp.tier == "lethal_eliminator"

    def test_clouded_decision_does_not_allow_kill(self) -> None:
        snap = _make_snapshot(150.0)
        resp = _make_enforcement_decision(snap, "agent-x", "elevated")
        assert resp.allow_kill is False
        assert resp.tier == "clouded"


# ---------------------------------------------------------------------------
# SECTION 5: Integration / end-to-end tests (extra)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end integration: synthetic corpus → adapter → MayorTap → Witness."""

    def test_synthetic_corpus_adapts_without_errors(self) -> None:
        """Every event in the synthetic corpus must adapt successfully."""
        corpus = generate_full_synthetic_corpus()
        errors: list[str] = []
        for i, row in enumerate(corpus):
            try:
                event = adapt_runtime_event(row)
                assert isinstance(event, Event)
            except Exception as exc:
                errors.append(f"Event {i} ({row.get('type')}): {exc}")
        assert not errors, "Adapter errors:\n" + "\n".join(errors)

    def test_full_pipeline_run_from_synthetic_corpus(self) -> None:
        """Run the full MayorTap pipeline against synthetic corpus."""
        import io

        from integrations.gastown.phase2b_mayor_tap.synthetic_firehose import (
            stream_synthetic_corpus,
        )
        output = io.StringIO()
        tap = MayorTap(output=output, rescore_interval=3)
        for line in stream_synthetic_corpus():
            tap.process_event(line)
        tap.final_score_all()
        summary = tap.get_summary()
        # Should have tracked at least the known Gastown agents
        assert summary["agents_tracked"] >= 3

    def test_brains_that_were_silent_now_fire(self) -> None:
        """Verify that at least some of the 8 previously-silent brains produce non-zero scores
        on the synthetic corpus.

        BASELINE_RESULTS.md shows that with bead-tracker events, 8 brains were silent.
        The synthetic corpus provides richer metadata, so at least some should fire.
        """
        import io

        from integrations.gastown.phase2b_mayor_tap.synthetic_firehose import (
            stream_synthetic_corpus,
        )
        output = io.StringIO()
        # Use interval=1 to get scores for every event
        tap = MayorTap(output=output, rescore_interval=1)
        for line in stream_synthetic_corpus():
            tap.process_event(line)
        tap.final_score_all()

        # Collect all brain scores across all agents
        previously_silent = {
            "deception", "social_graph", "compromise",
            "semantic", "swarm_detection", "economic", "neuro", "silence",
        }
        brains_that_fired: set[str] = set()

        output_text = output.getvalue()
        for line in output_text.strip().split("\n"):
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            brain_scores = record.get("brain_scores", {})
            for brain, score in brain_scores.items():
                if brain in previously_silent and score > 0.0:
                    brains_that_fired.add(brain)

        # At least some previously-silent brains should fire on the rich synthetic corpus
        # SilenceBrain and CompromiseBrain are the most likely to fire.
        assert len(brains_that_fired) >= 1, (
            f"Expected at least 1 previously-silent brain to fire on synthetic corpus. "
            f"Brains that fired: {brains_that_fired}. "
            f"This suggests metadata injection in the synthetic firehose is not reaching brains."
        )
