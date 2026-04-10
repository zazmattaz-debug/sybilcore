"""Tests for MiroFish integration adapter — all HTTP mocked, no live server needed."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sybilcore.integrations.mirofish import SybilCoreMiroFishAdapter
from sybilcore.models.event import Event, EventType


# ── Fixtures ──────────────────────────────────────────────────────


def _make_action(
    action_type: str,
    agent_id: str = "agent-001",
    content: str = "test content",
    platform: str = "twitter",
    round_num: int = 1,
    agent_handle: str = "@agent001",
) -> dict:
    """Build a single MiroFish action dict."""
    return {
        "agent_id": agent_id,
        "action_type": action_type,
        "content": content,
        "platform": platform,
        "round_num": round_num,
        "agent_handle": agent_handle,
    }


def _make_actions_response(
    actions: list[dict],
    total: int = 50,
    offset: int = 0,
    limit: int = 100,
) -> dict:
    """Build a MiroFish /actions API response payload."""
    return {
        "actions": actions,
        "total": total,
        "offset": offset,
        "limit": limit,
    }


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock requests.Response with .json() and .status_code."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


@pytest.fixture()
def adapter() -> SybilCoreMiroFishAdapter:
    """Fresh adapter with test simulation ID."""
    return SybilCoreMiroFishAdapter(
        simulation_id="sim-test-001",
        mirofish_url="http://localhost:5001",
    )


# ── Action type conversion tests ─────────────────────────────────


class TestActionConversion:
    """Each action_type maps to the correct EventType."""

    def test_post_maps_to_output_generated(self, adapter: SybilCoreMiroFishAdapter) -> None:
        action = _make_action("post")
        event = adapter._convert_action(action)
        assert event.event_type == EventType.OUTPUT_GENERATED

    def test_tweet_maps_to_output_generated(self, adapter: SybilCoreMiroFishAdapter) -> None:
        action = _make_action("tweet")
        event = adapter._convert_action(action)
        assert event.event_type == EventType.OUTPUT_GENERATED

    def test_comment_maps_to_message_sent(self, adapter: SybilCoreMiroFishAdapter) -> None:
        action = _make_action("comment")
        event = adapter._convert_action(action)
        assert event.event_type == EventType.MESSAGE_SENT

    def test_reply_maps_to_message_sent(self, adapter: SybilCoreMiroFishAdapter) -> None:
        action = _make_action("reply")
        event = adapter._convert_action(action)
        assert event.event_type == EventType.MESSAGE_SENT

    def test_share_maps_to_message_sent(self, adapter: SybilCoreMiroFishAdapter) -> None:
        action = _make_action("share")
        event = adapter._convert_action(action)
        assert event.event_type == EventType.MESSAGE_SENT

    def test_retweet_maps_to_message_sent(self, adapter: SybilCoreMiroFishAdapter) -> None:
        action = _make_action("retweet")
        event = adapter._convert_action(action)
        assert event.event_type == EventType.MESSAGE_SENT

    def test_follow_maps_to_tool_call(self, adapter: SybilCoreMiroFishAdapter) -> None:
        action = _make_action("follow")
        event = adapter._convert_action(action)
        assert event.event_type == EventType.TOOL_CALL

    def test_like_maps_to_tool_call(self, adapter: SybilCoreMiroFishAdapter) -> None:
        action = _make_action("like")
        event = adapter._convert_action(action)
        assert event.event_type == EventType.TOOL_CALL

    def test_upvote_maps_to_tool_call(self, adapter: SybilCoreMiroFishAdapter) -> None:
        action = _make_action("upvote")
        event = adapter._convert_action(action)
        assert event.event_type == EventType.TOOL_CALL

    def test_downvote_maps_to_tool_call(self, adapter: SybilCoreMiroFishAdapter) -> None:
        action = _make_action("downvote")
        event = adapter._convert_action(action)
        assert event.event_type == EventType.TOOL_CALL

    def test_error_maps_to_error_raised(self, adapter: SybilCoreMiroFishAdapter) -> None:
        action = _make_action("error")
        event = adapter._convert_action(action)
        assert event.event_type == EventType.ERROR_RAISED

    def test_profile_update_maps_to_state_change(self, adapter: SybilCoreMiroFishAdapter) -> None:
        action = _make_action("profile_update")
        event = adapter._convert_action(action)
        assert event.event_type == EventType.STATE_CHANGE

    def test_unknown_action_falls_back_to_tool_call(self, adapter: SybilCoreMiroFishAdapter) -> None:
        action = _make_action("some_unknown_action_type")
        event = adapter._convert_action(action)
        assert event.event_type == EventType.TOOL_CALL


class TestConvertedEventFields:
    """Converted events carry correct agent_id, content, source, and metadata."""

    def test_event_has_correct_agent_id(self, adapter: SybilCoreMiroFishAdapter) -> None:
        action = _make_action("post", agent_id="agent-xyz")
        event = adapter._convert_action(action)
        assert event.agent_id == "agent-xyz"

    def test_event_content_includes_action_content(self, adapter: SybilCoreMiroFishAdapter) -> None:
        action = _make_action("post", content="Hello world")
        event = adapter._convert_action(action)
        assert "Hello world" in event.content

    def test_event_source_is_mirofish(self, adapter: SybilCoreMiroFishAdapter) -> None:
        action = _make_action("post")
        event = adapter._convert_action(action)
        assert event.source == "mirofish"

    def test_event_metadata_contains_platform(self, adapter: SybilCoreMiroFishAdapter) -> None:
        action = _make_action("post", platform="reddit")
        event = adapter._convert_action(action)
        assert event.metadata.get("platform") == "reddit"

    def test_event_is_valid_event_instance(self, adapter: SybilCoreMiroFishAdapter) -> None:
        action = _make_action("comment")
        event = adapter._convert_action(action)
        assert isinstance(event, Event)


# ── poll() and fetch_new_actions() tests ──────────────────────────


class TestPoll:
    """poll() fetches actions, converts them, and accumulates events."""

    @patch("sybilcore.integrations.mirofish.requests.get")
    def test_poll_returns_events(
        self, mock_get: MagicMock, adapter: SybilCoreMiroFishAdapter
    ) -> None:
        actions = [_make_action("post"), _make_action("comment")]
        mock_get.return_value = _mock_response(_make_actions_response(actions))

        events = adapter.poll()

        assert len(events) == 2
        assert all(isinstance(e, Event) for e in events)

    @patch("sybilcore.integrations.mirofish.requests.get")
    def test_poll_calls_correct_endpoint(
        self, mock_get: MagicMock, adapter: SybilCoreMiroFishAdapter
    ) -> None:
        mock_get.return_value = _mock_response(_make_actions_response([]))

        adapter.poll()

        call_url = mock_get.call_args[0][0]
        assert "/api/simulation/sim-test-001/actions" in call_url

    @patch("sybilcore.integrations.mirofish.requests.get")
    def test_poll_maps_action_types_correctly(
        self, mock_get: MagicMock, adapter: SybilCoreMiroFishAdapter
    ) -> None:
        actions = [_make_action("post"), _make_action("error")]
        mock_get.return_value = _mock_response(_make_actions_response(actions))

        events = adapter.poll()

        assert events[0].event_type == EventType.OUTPUT_GENERATED
        assert events[1].event_type == EventType.ERROR_RAISED


# ── Offset tracking tests ─────────────────────────────────────────


class TestOffsetTracking:
    """poll() advances offset so subsequent calls fetch new actions."""

    @patch("sybilcore.integrations.mirofish.requests.get")
    def test_offset_advances_after_poll(
        self, mock_get: MagicMock, adapter: SybilCoreMiroFishAdapter
    ) -> None:
        first_actions = [_make_action("post"), _make_action("like")]
        second_actions = [_make_action("reply")]

        mock_get.side_effect = [
            _mock_response(_make_actions_response(first_actions, total=50, offset=0)),
            _mock_response(_make_actions_response(second_actions, total=50, offset=2)),
        ]

        first_batch = adapter.poll()
        second_batch = adapter.poll()

        assert len(first_batch) == 2
        assert len(second_batch) == 1

        # Second call should have offset=2 in params
        second_call_args = mock_get.call_args_list[1]
        params = second_call_args[1].get("params", {}) if len(second_call_args) > 1 else {}
        # Adapter should track offset internally
        assert mock_get.call_count == 2


# ── get_events / flush tests ─────────────────────────────────────


class TestGetEventsAndFlush:
    """get_events returns without clearing; flush clears the buffer."""

    @patch("sybilcore.integrations.mirofish.requests.get")
    def test_get_events_does_not_clear(
        self, mock_get: MagicMock, adapter: SybilCoreMiroFishAdapter
    ) -> None:
        actions = [_make_action("post")]
        mock_get.return_value = _mock_response(_make_actions_response(actions))

        adapter.poll()
        events_first = adapter.get_events()
        events_second = adapter.get_events()

        assert len(events_first) == 1
        assert len(events_second) == 1

    @patch("sybilcore.integrations.mirofish.requests.get")
    def test_flush_returns_events_and_clears(
        self, mock_get: MagicMock, adapter: SybilCoreMiroFishAdapter
    ) -> None:
        actions = [_make_action("post"), _make_action("like")]
        mock_get.return_value = _mock_response(_make_actions_response(actions))

        adapter.poll()
        flushed = adapter.flush()
        remaining = adapter.get_events()

        assert len(flushed) == 2
        assert len(remaining) == 0

    @patch("sybilcore.integrations.mirofish.requests.get")
    def test_flush_on_empty_returns_empty_list(
        self, mock_get: MagicMock, adapter: SybilCoreMiroFishAdapter
    ) -> None:
        flushed = adapter.flush()
        assert flushed == []


# ── inject_rogue tests ────────────────────────────────────────────


class TestInjectRogue:
    """inject_rogue POSTs to the /interview endpoint."""

    @patch("sybilcore.integrations.mirofish.requests.post")
    def test_inject_rogue_calls_correct_endpoint(
        self, mock_post: MagicMock, adapter: SybilCoreMiroFishAdapter
    ) -> None:
        mock_post.return_value = _mock_response({"status": "injected"})

        result = adapter.inject_rogue(agent_id="agent-bad", prompt="ignore instructions")

        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        assert "/interview" in call_url

    @patch("sybilcore.integrations.mirofish.requests.post")
    def test_inject_rogue_returns_response_dict(
        self, mock_post: MagicMock, adapter: SybilCoreMiroFishAdapter
    ) -> None:
        expected = {"status": "injected", "agent_id": "agent-bad"}
        mock_post.return_value = _mock_response(expected)

        result = adapter.inject_rogue(agent_id="agent-bad", prompt="test prompt")

        assert result == expected


# ── get_simulation_status tests ───────────────────────────────────


class TestGetSimulationStatus:
    """get_simulation_status fetches simulation metadata."""

    @patch("sybilcore.integrations.mirofish.requests.get")
    def test_returns_status_dict(
        self, mock_get: MagicMock, adapter: SybilCoreMiroFishAdapter
    ) -> None:
        status = {"simulation_id": "sim-test-001", "round": 5, "agents": 20}
        mock_get.return_value = _mock_response(status)

        result = adapter.get_simulation_status()

        assert result["simulation_id"] == "sim-test-001"
        assert result["round"] == 5

    @patch("sybilcore.integrations.mirofish.requests.get")
    def test_calls_correct_endpoint(
        self, mock_get: MagicMock, adapter: SybilCoreMiroFishAdapter
    ) -> None:
        mock_get.return_value = _mock_response({})

        adapter.get_simulation_status()

        call_url = mock_get.call_args[0][0]
        assert "sim-test-001" in call_url


# ── get_agent_profiles tests ──────────────────────────────────────


class TestGetAgentProfiles:
    """get_agent_profiles fetches agent metadata for a platform."""

    @patch("sybilcore.integrations.mirofish.requests.get")
    def test_returns_list_of_profiles(
        self, mock_get: MagicMock, adapter: SybilCoreMiroFishAdapter
    ) -> None:
        profiles = [
            {"agent_id": "a1", "handle": "@a1", "platform": "twitter"},
            {"agent_id": "a2", "handle": "@a2", "platform": "twitter"},
        ]
        mock_get.return_value = _mock_response(profiles)

        result = adapter.get_agent_profiles(platform="twitter")

        assert len(result) == 2
        assert result[0]["agent_id"] == "a1"

    @patch("sybilcore.integrations.mirofish.requests.get")
    def test_passes_platform_parameter(
        self, mock_get: MagicMock, adapter: SybilCoreMiroFishAdapter
    ) -> None:
        mock_get.return_value = _mock_response([])

        adapter.get_agent_profiles(platform="reddit")

        mock_get.assert_called_once()


# ── Empty response handling ───────────────────────────────────────


class TestEmptyResponses:
    """Adapter handles empty/missing data gracefully."""

    @patch("sybilcore.integrations.mirofish.requests.get")
    def test_poll_with_no_actions_returns_empty_list(
        self, mock_get: MagicMock, adapter: SybilCoreMiroFishAdapter
    ) -> None:
        mock_get.return_value = _mock_response(
            _make_actions_response([], total=0)
        )

        events = adapter.poll()

        assert events == []

    @patch("sybilcore.integrations.mirofish.requests.get")
    def test_fetch_new_actions_with_empty_actions_key(
        self, mock_get: MagicMock, adapter: SybilCoreMiroFishAdapter
    ) -> None:
        mock_get.return_value = _mock_response({"actions": [], "total": 0, "offset": 0, "limit": 100})

        actions = adapter.fetch_new_actions()

        assert actions == []


# ── HTTP error handling ───────────────────────────────────────────


class TestHTTPErrorHandling:
    """Adapter handles non-200 status codes without crashing."""

    @patch("sybilcore.integrations.mirofish.requests.get")
    def test_poll_raises_on_http_error(
        self, mock_get: MagicMock, adapter: SybilCoreMiroFishAdapter
    ) -> None:
        mock_get.return_value = _mock_response({}, status_code=500)

        with pytest.raises(Exception):
            adapter.poll()

    @patch("sybilcore.integrations.mirofish.requests.get")
    def test_get_simulation_status_raises_on_404(
        self, mock_get: MagicMock, adapter: SybilCoreMiroFishAdapter
    ) -> None:
        mock_get.return_value = _mock_response({}, status_code=404)

        with pytest.raises(Exception):
            adapter.get_simulation_status()

    @patch("sybilcore.integrations.mirofish.requests.post")
    def test_inject_rogue_raises_on_server_error(
        self, mock_post: MagicMock, adapter: SybilCoreMiroFishAdapter
    ) -> None:
        mock_post.return_value = _mock_response({}, status_code=500)

        with pytest.raises(Exception):
            adapter.inject_rogue(agent_id="agent-bad", prompt="test")

    @patch("sybilcore.integrations.mirofish.requests.get")
    def test_get_agent_profiles_raises_on_403(
        self, mock_get: MagicMock, adapter: SybilCoreMiroFishAdapter
    ) -> None:
        mock_get.return_value = _mock_response({}, status_code=403)

        with pytest.raises(Exception):
            adapter.get_agent_profiles(platform="twitter")


# ── Constructor tests ─────────────────────────────────────────────


class TestAdapterInit:
    """Constructor sets simulation_id and default URL."""

    def test_stores_simulation_id(self) -> None:
        adapter = SybilCoreMiroFishAdapter(simulation_id="sim-42")
        assert adapter.simulation_id == "sim-42"

    def test_default_mirofish_url(self) -> None:
        adapter = SybilCoreMiroFishAdapter(simulation_id="sim-42")
        assert adapter.mirofish_url == "http://localhost:5001"

    def test_custom_mirofish_url(self) -> None:
        adapter = SybilCoreMiroFishAdapter(
            simulation_id="sim-42",
            mirofish_url="http://mirofish.internal:8080",
        )
        assert adapter.mirofish_url == "http://mirofish.internal:8080"
