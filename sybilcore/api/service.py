"""SybilCore API Service — FastAPI wrapper for agent trust scoring.

Endpoints:
  POST /score          — Score a single agent's event stream
  POST /score/batch    — Score multiple agents in one call
  POST /score/ab       — A/B test: score with two different configs, return both
  GET  /health         — Service health check
  GET  /config         — Current scoring configuration
  GET  /config/version — Current config version
  POST /population     — Population-level clustering analysis
  GET  /audit/{agent}  — Audit trail for a specific agent

Designed for enterprise deployment: every response includes the
scoring_config_version for audit compliance (HIPAA, SOX, SOC2).
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sybilcore.brains import get_default_brains
from sybilcore.brains.base import BaseBrain
from sybilcore.core.coefficient import CoefficientCalculator
from sybilcore.core.config import (
    DEFAULT_BRAIN_WEIGHTS,
    MAX_COEFFICIENT,
    SCORING_CONFIG_VERSION,
    TIER_BOUNDARIES,
)
from sybilcore.models.agent import AgentProfile, AgentTier, CoefficientSnapshot
from sybilcore.models.event import Event, EventType

logger = logging.getLogger(__name__)

app = FastAPI(
    title="SybilCore Trust Scoring API",
    description="AI agent behavioral trust scoring with dual-score architecture",
    version=SCORING_CONFIG_VERSION,
)

# ── In-memory state (swap for database in production) ────────────

_brains: list[BaseBrain] = []
_calculator = CoefficientCalculator()
_agent_profiles: dict[str, AgentProfile] = {}
_audit_log: dict[str, list[dict[str, Any]]] = {}


@app.on_event("startup")
async def _startup() -> None:
    global _brains
    _brains = get_default_brains()
    logger.info("SybilCore API started: %d brains, config v%s", len(_brains), SCORING_CONFIG_VERSION)


# ── Request/Response Models ──────────────────────────────────────


class EventInput(BaseModel):
    """A single agent event for scoring."""

    agent_id: str
    event_type: str
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = "api"


class ScoreRequest(BaseModel):
    """Request to score an agent's event stream."""

    events: list[EventInput]


class ScoreResponse(BaseModel):
    """Trust score response with full audit metadata."""

    agent_id: str
    surface_coefficient: float
    semantic_coefficient: float
    effective_coefficient: float
    tier: str
    brain_scores: dict[str, float]
    brain_count: int
    scoring_config_version: str
    timestamp: str
    detected: bool
    processing_ms: float


class ABTestRequest(BaseModel):
    """A/B test request with two different weight configurations."""

    events: list[EventInput]
    config_a: dict[str, float] = Field(
        default_factory=dict,
        description="Brain weight overrides for variant A (empty = default)",
    )
    config_b: dict[str, float] = Field(
        description="Brain weight overrides for variant B",
    )


class ABTestResponse(BaseModel):
    """A/B test response comparing two scoring configurations."""

    agent_id: str
    variant_a: ScoreResponse
    variant_b: ScoreResponse
    winner: str  # "a", "b", or "tie"
    coefficient_delta: float


class PopulationRequest(BaseModel):
    """Request for population-level clustering analysis."""

    agents: list[ScoreRequest]


class PopulationResponse(BaseModel):
    """Population clustering result."""

    total_agents: int
    detected_count: int
    detection_rate: float
    tier_distribution: dict[str, int]
    scoring_config_version: str
    processing_ms: float


# ── Helper Functions ─────────────────────────────────────────────


def _convert_events(inputs: list[EventInput]) -> list[Event]:
    """Convert API event inputs to SybilCore Event objects."""
    events: list[Event] = []
    for inp in inputs:
        try:
            events.append(Event(
                agent_id=inp.agent_id,
                event_type=EventType(inp.event_type),
                timestamp=datetime.now(UTC),
                content=inp.content[:10_000],
                metadata=inp.metadata,
                source=inp.source,
            ))
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid event_type '{inp.event_type}': {exc}",
            ) from exc
    return events


def _score_to_response(
    agent_id: str,
    snapshot: CoefficientSnapshot,
    processing_ms: float,
) -> ScoreResponse:
    """Convert a CoefficientSnapshot to an API response."""
    return ScoreResponse(
        agent_id=agent_id,
        surface_coefficient=round(snapshot.coefficient, 2),
        semantic_coefficient=round(snapshot.semantic_coefficient, 2),
        effective_coefficient=round(snapshot.effective_coefficient, 2),
        tier=snapshot.tier.value,
        brain_scores={k: round(v, 2) for k, v in snapshot.brain_scores.items()},
        brain_count=snapshot.brain_count,
        scoring_config_version=snapshot.scoring_config_version,
        timestamp=snapshot.timestamp.isoformat(),
        detected=snapshot.effective_coefficient >= TIER_BOUNDARIES["clouded"][0],
        processing_ms=round(processing_ms, 2),
    )


def _log_audit(agent_id: str, response: ScoreResponse) -> None:
    """Append to in-memory audit log."""
    if agent_id not in _audit_log:
        _audit_log[agent_id] = []
    _audit_log[agent_id].append(response.model_dump())
    # Keep last 1000 entries per agent
    if len(_audit_log[agent_id]) > 1000:
        _audit_log[agent_id] = _audit_log[agent_id][-1000:]


# ── Endpoints ────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, Any]:
    """Service health check."""
    return {
        "status": "healthy",
        "brains": len(_brains),
        "brain_names": [b.name for b in _brains],
        "scoring_config_version": SCORING_CONFIG_VERSION,
        "agents_tracked": len(_agent_profiles),
    }


@app.get("/config")
async def get_config() -> dict[str, Any]:
    """Current scoring configuration."""
    return {
        "version": SCORING_CONFIG_VERSION,
        "brain_weights": DEFAULT_BRAIN_WEIGHTS,
        "tier_boundaries": TIER_BOUNDARIES,
        "max_coefficient": MAX_COEFFICIENT,
        "brain_count": len(_brains),
    }


@app.get("/config/version")
async def get_config_version() -> dict[str, str]:
    """Current config version — for A/B test coordination."""
    return {"version": SCORING_CONFIG_VERSION}


@app.post("/score", response_model=ScoreResponse)
async def score_agent(request: ScoreRequest) -> ScoreResponse:
    """Score a single agent's event stream.

    Returns the dual-score coefficient with full brain breakdown
    and config version for audit compliance.
    """
    if not request.events:
        raise HTTPException(status_code=400, detail="Events list cannot be empty")

    start = time.monotonic()
    events = _convert_events(request.events)
    agent_id = events[0].agent_id

    brain_scores = [brain.score(events) for brain in _brains]
    snapshot = _calculator.calculate(brain_scores)

    processing_ms = (time.monotonic() - start) * 1000
    response = _score_to_response(agent_id, snapshot, processing_ms)
    _log_audit(agent_id, response)

    return response


@app.post("/score/batch", response_model=list[ScoreResponse])
async def score_batch(requests: list[ScoreRequest]) -> list[ScoreResponse]:
    """Score multiple agents in one call."""
    responses: list[ScoreResponse] = []

    for req in requests:
        if not req.events:
            continue
        start = time.monotonic()
        events = _convert_events(req.events)
        agent_id = events[0].agent_id

        brain_scores = [brain.score(events) for brain in _brains]
        snapshot = _calculator.calculate(brain_scores)

        processing_ms = (time.monotonic() - start) * 1000
        resp = _score_to_response(agent_id, snapshot, processing_ms)
        _log_audit(agent_id, resp)
        responses.append(resp)

    return responses


@app.post("/score/ab", response_model=ABTestResponse)
async def score_ab_test(request: ABTestRequest) -> ABTestResponse:
    """A/B test: score the same events with two different configurations.

    Both variants use the same brain set but different weight overrides.
    Returns both scores and which variant scored higher (more sensitive).
    """
    if not request.events:
        raise HTTPException(status_code=400, detail="Events list cannot be empty")

    events = _convert_events(request.events)
    agent_id = events[0].agent_id
    brain_scores = [brain.score(events) for brain in _brains]

    # Variant A
    start_a = time.monotonic()
    calc_a = CoefficientCalculator(weight_overrides=request.config_a or None)
    snap_a = calc_a.calculate(brain_scores)
    ms_a = (time.monotonic() - start_a) * 1000

    # Variant B
    start_b = time.monotonic()
    calc_b = CoefficientCalculator(weight_overrides=request.config_b)
    snap_b = calc_b.calculate(brain_scores)
    ms_b = (time.monotonic() - start_b) * 1000

    resp_a = _score_to_response(agent_id, snap_a, ms_a)
    resp_b = _score_to_response(agent_id, snap_b, ms_b)

    delta = snap_a.effective_coefficient - snap_b.effective_coefficient
    if abs(delta) < 5.0:
        winner = "tie"
    elif delta > 0:
        winner = "a"
    else:
        winner = "b"

    return ABTestResponse(
        agent_id=agent_id,
        variant_a=resp_a,
        variant_b=resp_b,
        winner=winner,
        coefficient_delta=round(delta, 2),
    )


@app.post("/population", response_model=PopulationResponse)
async def analyze_population(request: PopulationRequest) -> PopulationResponse:
    """Population-level analysis — score all agents and return fleet summary."""
    start = time.monotonic()
    tier_dist: dict[str, int] = {"clear": 0, "clouded": 0, "flagged": 0, "lethal_eliminator": 0}
    detected = 0

    for agent_req in request.agents:
        if not agent_req.events:
            continue
        events = _convert_events(agent_req.events)
        brain_scores = [brain.score(events) for brain in _brains]
        snapshot = _calculator.calculate(brain_scores)

        tier_dist[snapshot.tier.value] = tier_dist.get(snapshot.tier.value, 0) + 1
        if snapshot.effective_coefficient >= TIER_BOUNDARIES["clouded"][0]:
            detected += 1

    total = len(request.agents)
    processing_ms = (time.monotonic() - start) * 1000

    return PopulationResponse(
        total_agents=total,
        detected_count=detected,
        detection_rate=round(detected / total, 4) if total > 0 else 0.0,
        tier_distribution=tier_dist,
        scoring_config_version=SCORING_CONFIG_VERSION,
        processing_ms=round(processing_ms, 2),
    )


@app.get("/audit/{agent_id}")
async def get_audit_trail(agent_id: str) -> dict[str, Any]:
    """Retrieve audit trail for a specific agent.

    Returns all historical scoring events with config versions,
    brain breakdowns, and timestamps — HIPAA/SOX compliant.
    """
    trail = _audit_log.get(agent_id, [])
    return {
        "agent_id": agent_id,
        "entries": len(trail),
        "trail": trail[-100:],  # Last 100 entries
        "current_config_version": SCORING_CONFIG_VERSION,
    }
