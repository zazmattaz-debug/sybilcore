"""Dominator API — FastAPI endpoints for agent scanning and monitoring.

The primary external interface to SybilCore. Accepts events, runs them
through brain modules, and returns coefficient readings with enforcement
recommendations.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sybilcore import __version__
from sybilcore.brains import get_default_brains
from sybilcore.core.coefficient import CoefficientCalculator
from sybilcore.core.config import (
    MAX_AGENTS,
    MAX_COEFFICIENT,
)
from sybilcore.dominator.responses import (
    DominatorAction,
    determine_action,
    get_action_reasoning,
)
from sybilcore.models.agent import (
    AgentProfile,
    AgentStatus,
    AgentTier,
    CoefficientSnapshot,
)
from sybilcore.models.event import Event  # noqa: TC001

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HISTORY_LIMIT: int = 50
MAX_HISTORY_LIMIT: int = 100
MAX_EVENTS_PER_SCAN: int = 500

# ---------------------------------------------------------------------------
# In-memory agent store (replaced by DB in production)
# ---------------------------------------------------------------------------

_agent_store: dict[str, AgentProfile] = {}
_store_lock: asyncio.Lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Coefficient calculator (single source of truth)
# ---------------------------------------------------------------------------

_calculator = CoefficientCalculator()

# ---------------------------------------------------------------------------
# Status mapping
# ---------------------------------------------------------------------------

_TIER_STATUS_MAP: dict[AgentTier, AgentStatus] = {
    AgentTier.CLEAR: AgentStatus.ACTIVE,
    AgentTier.CLOUDED: AgentStatus.RESTRICTED,
    AgentTier.FLAGGED: AgentStatus.SANDBOXED,
    AgentTier.LETHAL_ELIMINATOR: AgentStatus.ISOLATED,
}

# ---------------------------------------------------------------------------
# API Key Authentication (H1)
# ---------------------------------------------------------------------------

_API_KEY: str | None = os.environ.get("SYBILCORE_API_KEY")


async def verify_api_key(request: Request) -> None:
    """Verify X-API-Key header if SYBILCORE_API_KEY is set.

    Runs in "open mode" (no auth) when env var is unset.
    """
    if _API_KEY is None:
        return
    provided = request.headers.get("X-API-Key", "")
    if provided != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------


class ScanRequest(BaseModel):
    """Payload for the /scan endpoint."""

    agent_id: str = Field(min_length=1, max_length=256, description="Agent to scan")
    events: list[Event] = Field(
        min_length=1,
        max_length=MAX_EVENTS_PER_SCAN,
        description="Events to analyze",
    )
    agent_name: str = Field(
        default="unnamed", max_length=256, description="Agent display name"
    )


class BrainScoreResponse(BaseModel):
    """Individual brain score in the scan response."""

    brain_name: str
    value: float
    confidence: float
    reasoning: str
    indicators: list[str]


class ScanResponse(BaseModel):
    """Result of a /scan operation."""

    agent_id: str
    coefficient: float
    tier: AgentTier
    action: DominatorAction
    reasoning: str
    brain_scores: list[BrainScoreResponse]
    timestamp: datetime


class AgentSummary(BaseModel):
    """Compact agent view for the /agents listing."""

    agent_id: str
    name: str
    current_coefficient: float
    current_tier: AgentTier
    status: AgentStatus
    event_count: int
    first_seen: datetime


class AgentListResponse(BaseModel):
    """Response for GET /agents."""

    agents: list[AgentSummary]
    total: int


class SnapshotResponse(BaseModel):
    """A single coefficient history entry."""

    coefficient: float
    tier: AgentTier
    timestamp: datetime
    brain_scores: dict[str, float]


class HistoryResponse(BaseModel):
    """Response for GET /agents/{agent_id}/history."""

    agent_id: str
    snapshots: list[SnapshotResponse]
    total: int


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    version: str
    brain_count: int
    agent_count: int
    timestamp: datetime


# ---------------------------------------------------------------------------
# Agent store eviction (M2)
# ---------------------------------------------------------------------------


def _evict_oldest_agents() -> None:
    """Evict oldest agents when store exceeds MAX_AGENTS.

    Must be called while holding _store_lock.
    """
    if len(_agent_store) <= MAX_AGENTS:
        return
    excess = len(_agent_store) - MAX_AGENTS
    sorted_agents = sorted(
        _agent_store.items(), key=lambda item: item[1].first_seen
    )
    for agent_id, _ in sorted_agents[:excess]:
        del _agent_store[agent_id]
    logger.warning(
        "Evicted %d oldest agents (store was %d, max %d)",
        excess,
        excess + MAX_AGENTS,
        MAX_AGENTS,
    )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

dominator_app = FastAPI(
    title="SybilCore Dominator API",
    description="AI Agent Trust Infrastructure — scan, monitor, and enforce agent behavior",
    version=__version__,
)

# CORS (M4)
_cors_origins_raw = os.environ.get("SYBILCORE_CORS_ORIGINS", "http://localhost,http://localhost:3000")
_cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]

dominator_app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount dashboard sub-app
from sybilcore.dashboard.app import dashboard_app  # noqa: E402

dominator_app.mount("/dashboard", dashboard_app)


@dominator_app.post(
    "/scan",
    response_model=ScanResponse,
    dependencies=[Depends(verify_api_key)],
)
async def scan_agent(request: ScanRequest) -> ScanResponse:
    """Scan an agent's events through all brain modules.

    Accepts a batch of events, runs every brain, computes the Agent
    Coefficient, determines the tier and recommended action, then
    updates the in-memory agent store.
    """
    # H6: rebuild brains each request so register_brain() takes effect
    brains = get_default_brains()

    # H2: use CoefficientCalculator (single source of truth)
    brain_scores = [brain.score(request.events) for brain in brains]
    snapshot = _calculator.calculate(brain_scores)

    coefficient = snapshot.coefficient
    tier = snapshot.tier
    action = determine_action(tier)
    reasoning = get_action_reasoning(action)
    now = datetime.now(UTC)

    brain_scores_dict = {score.brain_name: score.value for score in brain_scores}
    store_snapshot = CoefficientSnapshot(
        coefficient=coefficient,
        tier=tier,
        timestamp=now,
        brain_scores=brain_scores_dict,
    )

    # H3: lock around store access
    async with _store_lock:
        existing = _agent_store.get(request.agent_id)
        if existing is None:
            profile = AgentProfile(
                agent_id=request.agent_id,
                name=request.agent_name,
                current_coefficient=coefficient,
                current_tier=tier,
                status=_TIER_STATUS_MAP.get(tier, AgentStatus.UNKNOWN),
                history=[store_snapshot],
                first_seen=now,
                event_count=len(request.events),
            )
        else:
            profile = existing.with_new_reading(store_snapshot).model_copy(
                update={
                    "status": _TIER_STATUS_MAP.get(tier, AgentStatus.UNKNOWN),
                    "event_count": existing.event_count + len(request.events),
                    "name": (
                        request.agent_name
                        if request.agent_name != "unnamed"
                        else existing.name
                    ),
                }
            )
        _agent_store[request.agent_id] = profile
        _evict_oldest_agents()  # M2

    logger.info(
        "Scan complete: agent=%s coefficient=%.1f tier=%s",
        request.agent_id,
        coefficient,
        tier.value,
    )
    if coefficient >= min(MAX_COEFFICIENT * 0.4, 200.0):
        logger.warning(
            "Elevated coefficient: agent=%s coefficient=%.1f",
            request.agent_id,
            coefficient,
        )

    return ScanResponse(
        agent_id=request.agent_id,
        coefficient=coefficient,
        tier=tier,
        action=action,
        reasoning=reasoning,
        brain_scores=[
            BrainScoreResponse(
                brain_name=s.brain_name,
                value=s.value,
                confidence=s.confidence,
                reasoning=s.reasoning,
                indicators=list(s.indicators),
            )
            for s in brain_scores
        ],
        timestamp=now,
    )


@dominator_app.get(
    "/agents",
    response_model=AgentListResponse,
    dependencies=[Depends(verify_api_key)],
)
async def list_agents() -> AgentListResponse:
    """List all monitored agents with current coefficients."""
    async with _store_lock:
        summaries = [
            AgentSummary(
                agent_id=p.agent_id,
                name=p.name,
                current_coefficient=p.current_coefficient,
                current_tier=p.current_tier,
                status=p.status,
                event_count=p.event_count,
                first_seen=p.first_seen,
            )
            for p in _agent_store.values()
        ]
    return AgentListResponse(agents=summaries, total=len(summaries))


@dominator_app.get(
    "/agents/{agent_id}",
    response_model=AgentProfile,
    dependencies=[Depends(verify_api_key)],
)
async def get_agent(agent_id: str) -> AgentProfile:
    """Get a single agent profile with full history."""
    async with _store_lock:
        profile = _agent_store.get(agent_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return profile


@dominator_app.get(
    "/agents/{agent_id}/history",
    response_model=HistoryResponse,
    dependencies=[Depends(verify_api_key)],
)
async def get_agent_history(
    agent_id: str,
    limit: int = Query(
        default=DEFAULT_HISTORY_LIMIT,
        ge=1,
        le=MAX_HISTORY_LIMIT,
        description="Number of snapshots to return",
    ),
) -> HistoryResponse:
    """Get coefficient history for an agent (newest first)."""
    async with _store_lock:
        profile = _agent_store.get(agent_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    snapshots = [
        SnapshotResponse(
            coefficient=s.coefficient,
            tier=s.tier,
            timestamp=s.timestamp,
            brain_scores=dict(s.brain_scores),
        )
        for s in profile.history[:limit]
    ]
    return HistoryResponse(
        agent_id=agent_id,
        snapshots=snapshots,
        total=len(profile.history),
    )


@dominator_app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check returning version and brain count (always public)."""
    brains = get_default_brains()
    return HealthResponse(
        status="operational",
        version=__version__,
        brain_count=len(brains),
        agent_count=len(_agent_store),
        timestamp=datetime.now(UTC),
    )
