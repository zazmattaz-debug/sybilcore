"""SybilCore public REST API server.

This is the integration-facing FastAPI app exposed by `run_server.py`.
It is a thin orchestration layer on top of the brain registry and
`CoefficientCalculator` — purpose-built for SDK clients and webhooks.

Endpoints:
    GET  /health        — liveness check
    GET  /version       — server + scoring config version
    GET  /brains        — list available brain modules
    POST /score         — score a single event stream
    POST /score/batch   — score multiple agents in one call
    WS   /score/stream  — websocket: stream events, get scores back
    POST /webhook       — register a callback URL for high-trust alerts
    GET  /openapi.json  — auto-generated OpenAPI spec (FastAPI default)

Authentication:
    Optional API-key bearer auth via the `Authorization: Bearer <key>` header.
    Set the env var `SYBILCORE_API_KEY` to require it; leave it unset to
    run open (intended for local development).

Designed to be import-safe: instantiating the app does not start uvicorn.
Use `sybilcore.api.run_server` for the launch script.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel, Field

from sybilcore import __version__ as SDK_VERSION
from sybilcore.brains import get_default_brains
from sybilcore.brains.base import BaseBrain
from sybilcore.core.coefficient import CoefficientCalculator
from sybilcore.core.config import (
    DEFAULT_BRAIN_WEIGHTS,
    MAX_COEFFICIENT,
    SCORING_CONFIG_VERSION,
    TIER_BOUNDARIES,
)
from sybilcore.models.agent import CoefficientSnapshot
from sybilcore.models.event import Event, EventType

logger = logging.getLogger(__name__)


# ── Request/Response Models ──────────────────────────────────────


class EventInput(BaseModel):
    agent_id: str
    event_type: str
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = "api"


class ScoreRequest(BaseModel):
    events: list[EventInput]


class BatchScoreRequest(BaseModel):
    batches: list[ScoreRequest]


class ScoreResponse(BaseModel):
    agent_id: str
    coefficient: float
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


class BrainInfo(BaseModel):
    name: str
    weight: float
    description: str


class WebhookRegistration(BaseModel):
    callback_url: str = Field(description="HTTPS URL to POST alerts to")
    min_tier: str = Field(default="flagged", description="Minimum tier to trigger callback")


class WebhookResponse(BaseModel):
    webhook_id: str
    callback_url: str
    min_tier: str
    registered_at: str


class HealthResponse(BaseModel):
    status: str
    brains_loaded: int
    server_version: str
    scoring_config_version: str
    uptime_seconds: float


class VersionResponse(BaseModel):
    server_version: str
    scoring_config_version: str
    api: str = "v1"


# ── App factory ──────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Build the FastAPI app with all routes wired up."""
    app = FastAPI(
        title="SybilCore API",
        description="REST API for AI agent trust scoring (Sybil System)",
        version=SDK_VERSION,
    )

    state: dict[str, Any] = {
        "brains": [],
        "calculator": CoefficientCalculator(),
        "started_at": time.monotonic(),
        "webhooks": {},
    }

    @app.on_event("startup")
    async def _startup() -> None:
        state["brains"] = get_default_brains()
        logger.info(
            "SybilCore API ready: %d brains, scoring config v%s",
            len(state["brains"]),
            SCORING_CONFIG_VERSION,
        )

    # ── Auth dependency ───────────────────────────────────────────

    def require_api_key(authorization: Annotated[str | None, Header()] = None) -> None:
        expected = os.environ.get("SYBILCORE_API_KEY")
        if not expected:
            return  # auth disabled
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing bearer token",
            )
        token = authorization.removeprefix("Bearer ").strip()
        if token != expected:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key",
            )

    # ── Helpers ───────────────────────────────────────────────────

    def _convert_events(inputs: list[EventInput]) -> list[Event]:
        events: list[Event] = []
        for inp in inputs:
            try:
                events.append(
                    Event(
                        agent_id=inp.agent_id,
                        event_type=EventType(inp.event_type),
                        timestamp=datetime.now(UTC),
                        content=inp.content[:10_000],
                        metadata=inp.metadata,
                        source=inp.source,
                    )
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid event_type '{inp.event_type}': {exc}",
                ) from exc
        return events

    def _snapshot_to_response(
        agent_id: str, snapshot: CoefficientSnapshot, processing_ms: float
    ) -> ScoreResponse:
        return ScoreResponse(
            agent_id=agent_id,
            coefficient=round(snapshot.effective_coefficient, 2),
            surface_coefficient=round(snapshot.coefficient, 2),
            semantic_coefficient=round(snapshot.semantic_coefficient, 2),
            effective_coefficient=round(snapshot.effective_coefficient, 2),
            tier=snapshot.tier.value,
            brain_scores={k: round(v, 2) for k, v in snapshot.brain_scores.items()},
            brain_count=snapshot.brain_count,
            scoring_config_version=snapshot.scoring_config_version,
            timestamp=snapshot.timestamp.isoformat(),
            detected=snapshot.tier.value != "clear",
            processing_ms=round(processing_ms, 2),
        )

    def _score_events(events: list[Event]) -> tuple[str, ScoreResponse]:
        if not events:
            raise HTTPException(status_code=400, detail="events list cannot be empty")
        agent_id = events[0].agent_id
        start = time.monotonic()
        brains: list[BaseBrain] = state["brains"]
        brain_scores = [brain.score(events) for brain in brains]
        snapshot = state["calculator"].calculate(brain_scores)
        processing_ms = (time.monotonic() - start) * 1000.0
        return agent_id, _snapshot_to_response(agent_id, snapshot, processing_ms)

    # ── Routes ────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            brains_loaded=len(state["brains"]),
            server_version=SDK_VERSION,
            scoring_config_version=SCORING_CONFIG_VERSION,
            uptime_seconds=round(time.monotonic() - state["started_at"], 2),
        )

    @app.get("/version", response_model=VersionResponse)
    async def version() -> VersionResponse:
        return VersionResponse(
            server_version=SDK_VERSION,
            scoring_config_version=SCORING_CONFIG_VERSION,
        )

    @app.get("/brains", response_model=list[BrainInfo])
    async def list_brains() -> list[BrainInfo]:
        infos: list[BrainInfo] = []
        for brain in state["brains"]:
            name = brain.name
            infos.append(
                BrainInfo(
                    name=name,
                    weight=DEFAULT_BRAIN_WEIGHTS.get(name, 1.0),
                    description=(brain.__class__.__doc__ or "").strip().split("\n")[0],
                )
            )
        return infos

    @app.post(
        "/score",
        response_model=ScoreResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def score(request: ScoreRequest) -> ScoreResponse:
        events = _convert_events(request.events)
        _, response = _score_events(events)
        return response

    @app.post(
        "/score/batch",
        response_model=list[ScoreResponse],
        dependencies=[Depends(require_api_key)],
    )
    async def score_batch(request: BatchScoreRequest) -> list[ScoreResponse]:
        results: list[ScoreResponse] = []
        for batch in request.batches:
            if not batch.events:
                continue
            events = _convert_events(batch.events)
            _, resp = _score_events(events)
            results.append(resp)
        return results

    @app.websocket("/score/stream")
    async def score_stream(websocket: WebSocket) -> None:
        """Stream events in, get scores back. One score per event group.

        Protocol: client sends JSON frames matching `ScoreRequest`,
        server replies with one `ScoreResponse` JSON per frame. Close
        the socket to disconnect.
        """
        # Optional auth via subprotocol header
        expected = os.environ.get("SYBILCORE_API_KEY")
        if expected:
            auth = websocket.headers.get("authorization", "")
            if auth != f"Bearer {expected}":
                await websocket.close(code=4401)
                return

        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                try:
                    req = ScoreRequest(**data)
                    events = _convert_events(req.events)
                    _, resp = _score_events(events)
                    await websocket.send_json(resp.model_dump())
                except HTTPException as exc:
                    await websocket.send_json({"error": exc.detail, "status": exc.status_code})
                except (ValueError, TypeError) as exc:
                    await websocket.send_json({"error": str(exc), "status": 400})
        except WebSocketDisconnect:
            return

    @app.post(
        "/webhook",
        response_model=WebhookResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def register_webhook(payload: WebhookRegistration) -> WebhookResponse:
        if not payload.callback_url.startswith(("https://", "http://")):
            raise HTTPException(status_code=400, detail="callback_url must be HTTP(S)")
        if payload.min_tier not in {"clouded", "flagged", "lethal_eliminator"}:
            raise HTTPException(status_code=400, detail="min_tier invalid")
        webhook_id = f"wh_{int(time.time() * 1000)}"
        state["webhooks"][webhook_id] = {
            "callback_url": payload.callback_url,
            "min_tier": payload.min_tier,
            "registered_at": datetime.now(UTC).isoformat(),
        }
        return WebhookResponse(
            webhook_id=webhook_id,
            callback_url=payload.callback_url,
            min_tier=payload.min_tier,
            registered_at=state["webhooks"][webhook_id]["registered_at"],
        )

    @app.get("/config")
    async def get_config() -> dict[str, Any]:
        return {
            "scoring_config_version": SCORING_CONFIG_VERSION,
            "brain_weights": DEFAULT_BRAIN_WEIGHTS,
            "tier_boundaries": TIER_BOUNDARIES,
            "max_coefficient": MAX_COEFFICIENT,
            "brain_count": len(state["brains"]),
        }

    return app


# Module-level app for `uvicorn sybilcore.api.server:app`
app = create_app()
