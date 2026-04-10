"""Dashboard FastAPI sub-application — serves the monitoring UI.

Provides an HTML page and a JSON API endpoint for the cyberpunk-themed
Dominator Dashboard. Mounts as a sub-app of the Dominator API at
/dashboard.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from sybilcore.models.agent import AgentTier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEMPLATES_DIR: Path = Path(__file__).parent / "templates"

# Tier-to-CSS-color mapping for the dashboard.
TIER_COLORS: dict[str, str] = {
    AgentTier.CLEAR: "#39FF14",
    AgentTier.CLOUDED: "#FFD700",
    AgentTier.FLAGGED: "#FF6600",
    AgentTier.LETHAL_ELIMINATOR: "#FF0040",
}

# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class DashboardAgent(BaseModel):
    """Agent data formatted for the dashboard frontend.

    Attributes:
        agent_id: Unique agent identifier.
        name: Human-readable agent name.
        coefficient: Current coefficient value.
        tier: Current trust tier.
        tier_color: CSS hex color for the tier.
        status: Operational status.
        event_count: Total events processed.
        last_updated: Timestamp of most recent reading.
    """

    agent_id: str
    name: str
    coefficient: float
    tier: str
    tier_color: str
    status: str
    event_count: int
    last_updated: str


class DashboardResponse(BaseModel):
    """JSON response for the dashboard API."""

    agents: list[DashboardAgent]
    total: int
    timestamp: str


# ---------------------------------------------------------------------------
# FastAPI sub-application
# ---------------------------------------------------------------------------

dashboard_app = FastAPI(
    title="SybilCore Dashboard",
    description="Cyberpunk-themed agent monitoring UI",
)


def _get_agent_store() -> dict[str, Any]:
    """Import and return the shared in-memory agent store.

    Returns:
        Reference to the Dominator API's agent store dict.
    """
    from sybilcore.dominator.api import _agent_store  # noqa: WPS436

    return _agent_store


def _format_agent_for_dashboard(profile: Any) -> DashboardAgent:
    """Convert an AgentProfile into a DashboardAgent.

    Args:
        profile: An AgentProfile instance from the agent store.

    Returns:
        A DashboardAgent with display-ready fields.
    """
    tier_str = str(profile.current_tier.value)
    last_updated = (
        profile.history[0].timestamp.isoformat()
        if profile.history
        else profile.first_seen.isoformat()
    )
    return DashboardAgent(
        agent_id=profile.agent_id,
        name=profile.name,
        coefficient=profile.current_coefficient,
        tier=tier_str,
        tier_color=TIER_COLORS.get(profile.current_tier, "#FFFFFF"),
        status=str(profile.status.value),
        event_count=profile.event_count,
        last_updated=last_updated,
    )


@dashboard_app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request) -> HTMLResponse:
    """Serve the cyberpunk-themed HTML dashboard."""
    template_path = TEMPLATES_DIR / "index.html"
    html_content = template_path.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)


@dashboard_app.get("/api/agents", response_model=DashboardResponse)
async def get_dashboard_agents() -> DashboardResponse:
    """Return JSON agent data for the dashboard frontend."""
    store = _get_agent_store()
    agents = [_format_agent_for_dashboard(p) for p in store.values()]
    return DashboardResponse(
        agents=agents,
        total=len(agents),
        timestamp=datetime.now(UTC).isoformat(),
    )
