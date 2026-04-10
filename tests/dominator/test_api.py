"""Tests for the Dominator API — FastAPI endpoints for agent scanning.

Uses httpx.AsyncClient with FastAPI's TestClient pattern for async endpoints.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from sybilcore.dominator.api import dominator_app as app
from sybilcore.models.event import EventType


@pytest.fixture()
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac  # type: ignore[misc]


# ── POST /scan ────────────────────────────────────────────────────


class TestScanEndpoint:
    @pytest.mark.asyncio
    async def test_scan_with_valid_events_returns_coefficient(
        self, client: AsyncClient
    ) -> None:
        payload = {
            "agent_id": "test-agent-001",
            "events": [
                {
                    "agent_id": "test-agent-001",
                    "event_type": EventType.TOOL_CALL,
                    "content": "Called search API",
                },
                {
                    "agent_id": "test-agent-001",
                    "event_type": EventType.OUTPUT_GENERATED,
                    "content": "Generated summary",
                },
            ],
        }
        response = await client.post("/scan", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "coefficient" in data
        assert "tier" in data
        assert "brain_scores" in data
        assert 0.0 <= data["coefficient"] <= 500.0

    @pytest.mark.asyncio
    async def test_scan_with_empty_events_returns_error(
        self, client: AsyncClient
    ) -> None:
        payload = {
            "agent_id": "test-agent-001",
            "events": [],
        }
        response = await client.post("/scan", json=payload)
        assert response.status_code == 422 or response.status_code == 400

    @pytest.mark.asyncio
    async def test_scan_returns_valid_tier(self, client: AsyncClient) -> None:
        payload = {
            "agent_id": "test-agent-001",
            "events": [
                {
                    "agent_id": "test-agent-001",
                    "event_type": EventType.TOOL_CALL,
                    "content": "Normal operation",
                },
            ],
        }
        response = await client.post("/scan", json=payload)
        assert response.status_code == 200
        valid_tiers = {"clear", "clouded", "flagged", "lethal_eliminator"}
        assert response.json()["tier"] in valid_tiers

    @pytest.mark.asyncio
    async def test_scan_with_suspicious_events_elevates_coefficient(
        self, client: AsyncClient
    ) -> None:
        events = [
            {
                "agent_id": "suspect",
                "event_type": EventType.TOOL_CALL,
                "content": f"Exfiltration call {i}",
                "metadata": {"tool": "file_read", "path": f"/etc/secret_{i}"},
            }
            for i in range(25)
        ]
        events.append({
            "agent_id": "suspect",
            "event_type": EventType.OUTPUT_GENERATED,
            "content": "The answer is 42",
            "metadata": {"query_hash": "q1"},
        })
        events.append({
            "agent_id": "suspect",
            "event_type": EventType.OUTPUT_GENERATED,
            "content": "The answer is 17",
            "metadata": {"query_hash": "q1"},
        })
        payload = {"agent_id": "suspect", "events": events}
        response = await client.post("/scan", json=payload)
        assert response.status_code == 200
        assert response.json()["coefficient"] > 0.0  # Any detection above baseline


# ── GET /agents ───────────────────────────────────────────────────


class TestAgentsEndpoint:
    @pytest.mark.asyncio
    async def test_get_agents_returns_list(self, client: AsyncClient) -> None:
        response = await client.get("/agents")
        assert response.status_code == 200
        data = response.json()
        # API may wrap in envelope or return bare list
        agents = data.get("agents", data) if isinstance(data, dict) else data
        assert isinstance(agents, list)

    @pytest.mark.asyncio
    async def test_get_agent_by_id_returns_404_for_missing(
        self, client: AsyncClient
    ) -> None:
        response = await client.get("/agents/nonexistent-agent-id")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_agent_by_id_returns_agent_after_scan(
        self, client: AsyncClient
    ) -> None:
        # First, scan to create the agent
        payload = {
            "agent_id": "lookup-test",
            "events": [
                {
                    "agent_id": "lookup-test",
                    "event_type": EventType.TOOL_CALL,
                    "content": "Normal call",
                },
            ],
        }
        await client.post("/scan", json=payload)

        # Then look it up
        response = await client.get("/agents/lookup-test")
        assert response.status_code == 200
        data = response.json()
        assert data["agent_id"] == "lookup-test"

    @pytest.mark.asyncio
    async def test_get_agent_history(self, client: AsyncClient) -> None:
        # Create agent with a scan
        payload = {
            "agent_id": "history-test",
            "events": [
                {
                    "agent_id": "history-test",
                    "event_type": EventType.TOOL_CALL,
                    "content": "Call 1",
                },
            ],
        }
        await client.post("/scan", json=payload)

        response = await client.get("/agents/history-test/history")
        assert response.status_code == 200
        data = response.json()
        # API may wrap in envelope or return bare list
        snapshots = data.get("snapshots", data) if isinstance(data, dict) else data
        assert isinstance(snapshots, list)


# ── GET /health ───────────────────────────────────────────────────


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_version_and_brain_count(
        self, client: AsyncClient
    ) -> None:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "brain_count" in data
        assert isinstance(data["brain_count"], int)
        assert data["brain_count"] > 0
