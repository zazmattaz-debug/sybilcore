"""Witness third-gate integration — SybilCore coefficient check before NukePolecat.

This module implements the Python HTTP endpoint that Gastown's Go Witness would
POST to before executing `NukePolecat` (handlers.go:826).

Gastown integration point (from SCOUT_REPORT.md §6, §11):
    - File: internal/witness/handlers.go:838-844
    - Existing gates before nuke:
        1. mayor.NewCleanupVetoChecker().ShouldVetoCleanup() — blocks if Mayor ACP active
        2. hasPendingMR() — blocks if MR pending in refinery
        3. **SybilCore coefficient check** ← THIS ENDPOINT (third gate)
    - The Go side would call: POST /witness/pre-nuke with the polecat identity
    - This endpoint returns allow_kill=true only if coefficient >= ENFORCEMENT threshold

Design:
    - FastAPI endpoint (async, non-blocking)
    - Stateless: coefficient is computed from the in-memory AgentProfile store
      maintained by MayorTap, or can accept events directly for on-demand scoring
    - Tier mapping: LETHAL_ELIMINATOR → allow kill; others → escalate first
    - Returns structured JSON the Go Witness can deserialize

Enforcement logic (from SCOUT_REPORT.md §6):
    "ENFORCEMENT tier = allow nuke; FLAGGED tier = escalate to Deacon first."
    Extension: LETHAL_ELIMINATOR AND FLAGGED both allow kill (FLAGGED after Deacon ack).
    CLEAR and CLOUDED → veto kill (agent is within normal range).

Authentication:
    All endpoints require HMAC-SHA256 bearer token authentication.

    Headers required:
        Authorization: Bearer <hmac-sha256-hex(body, secret)>
        X-Sybilcore-Timestamp: <unix-epoch-seconds>

    The HMAC is computed over the raw request body bytes.  For GET requests
    (empty body), the canonical payload is b"" (empty bytes).

    Secret is loaded from env var SYBILCORE_WITNESS_SECRET at startup.
    If the var is unset and SYBILCORE_WITNESS_DEV != "1", the server refuses to start.

    DEV mode (SYBILCORE_WITNESS_DEV=1) bypasses auth entirely.  Never use in production.

Rate limiting:
    Token-bucket per client IP: 100 requests per 60-second window.
    Exceeding the limit returns HTTP 429.

Request size:
    Bodies exceeding 64 KB are rejected with HTTP 413 before auth check.

LABEL: "awaiting Go-side implementation" — the Go Witness does not yet call this
endpoint. This is the Python receiver, implementing the contract described in §6.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from collections import deque
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sybilcore.brains import get_default_brains
from sybilcore.brains.base import BaseBrain, BrainScore
from sybilcore.core.coefficient import CoefficientCalculator
from sybilcore.models.agent import AgentTier, CoefficientSnapshot
from sybilcore.models.event import Event  # noqa: TC001

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_BODY_BYTES: int = 64 * 1024  # 64 KB
TIMESTAMP_TOLERANCE_SECONDS: int = 60
RATE_LIMIT_REQUESTS: int = 100
RATE_LIMIT_WINDOW_SECONDS: float = 60.0

# ---------------------------------------------------------------------------
# Startup secret management
# ---------------------------------------------------------------------------


def _load_secret() -> bytes | None:
    """Return the shared secret as bytes, or None in DEV mode.

    In DEV mode (SYBILCORE_WITNESS_DEV=1) authentication is bypassed.
    In production, SYBILCORE_WITNESS_SECRET must be set; the server refuses
    to start otherwise.

    Called lazily on first request (not at module import) so that tests can
    set the env var before the first request arrives.
    """
    dev_mode = os.environ.get("SYBILCORE_WITNESS_DEV", "") == "1"
    raw_secret = os.environ.get("SYBILCORE_WITNESS_SECRET", "")

    if dev_mode:
        logger.warning(
            "SYBILCORE_WITNESS_DEV=1 — authentication DISABLED. "
            "Never run in this mode in production."
        )
        return None

    if not raw_secret:
        logger.critical(
            "FATAL: SYBILCORE_WITNESS_SECRET is not set and "
            "SYBILCORE_WITNESS_DEV is not '1'. "
            "Refusing to start — all endpoints would be unauthenticated. "
            "Set the env var or export SYBILCORE_WITNESS_DEV=1 for local development."
        )
        raise RuntimeError(
            "SYBILCORE_WITNESS_SECRET must be set in production. "
            "Set SYBILCORE_WITNESS_DEV=1 to bypass (development only)."
        )

    return raw_secret.encode()


# Sentinel: secret is loaded lazily on the first request.
# _WITNESS_SECRET is None until _get_secret() is called.
# Use _get_secret() everywhere; never read _WITNESS_SECRET directly.
_WITNESS_SECRET: bytes | None = None
_WITNESS_SECRET_LOADED: bool = False


def _get_secret() -> bytes | None:
    """Return the cached secret, loading it on first call.

    Lazy loading allows tests to set SYBILCORE_WITNESS_SECRET in os.environ
    before the first request without needing to control module import order.

    Returns:
        bytes if a secret is configured, None in DEV mode.

    Raises:
        RuntimeError: If secret is unset and not in DEV mode (startup guard).
    """
    global _WITNESS_SECRET, _WITNESS_SECRET_LOADED  # noqa: PLW0603
    if not _WITNESS_SECRET_LOADED:
        _WITNESS_SECRET = _load_secret()
        _WITNESS_SECRET_LOADED = True
    return _WITNESS_SECRET


# ---------------------------------------------------------------------------
# Rate limiter (token bucket per IP, in-process)
# ---------------------------------------------------------------------------


class _TokenBucket:
    """Sliding-window rate limiter tracking request timestamps per IP."""

    def __init__(self) -> None:
        # Maps client_ip → deque of request timestamps (float unix seconds).
        self._windows: dict[str, deque[float]] = {}

    def is_allowed(self, client_ip: str) -> bool:
        """Return True if the request is within rate limits; False if exceeded."""
        now = time.monotonic()
        cutoff = now - RATE_LIMIT_WINDOW_SECONDS

        if client_ip not in self._windows:
            self._windows[client_ip] = deque()

        window = self._windows[client_ip]

        # Evict timestamps outside the current window.
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= RATE_LIMIT_REQUESTS:
            return False

        window.append(now)
        return True


_rate_limiter = _TokenBucket()


# ---------------------------------------------------------------------------
# HMAC signature helpers
# ---------------------------------------------------------------------------


def _compute_expected_signature(body: bytes, secret: bytes) -> str:
    """Return the hex HMAC-SHA256 of body signed with secret."""
    return hmac.new(secret, body, hashlib.sha256).hexdigest()


def _extract_bearer_token(authorization: str | None) -> str | None:
    """Parse 'Bearer <token>' header value; return token or None."""
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]


def _verify_request(
    authorization: str | None,
    timestamp_header: str | None,
    body: bytes,
    secret: bytes,
) -> None:
    """Verify HMAC signature and timestamp. Raises HTTPException on failure.

    Args:
        authorization: Value of the Authorization header.
        timestamp_header: Value of X-Sybilcore-Timestamp header (unix epoch).
        body: Raw request body bytes.
        secret: Shared secret bytes.

    Raises:
        HTTPException: 401 if authentication fails for any reason.
    """
    # --- Timestamp check (replay protection) ---
    if not timestamp_header:
        raise HTTPException(status_code=401, detail="Missing X-Sybilcore-Timestamp header")

    try:
        request_ts = int(timestamp_header)
    except (ValueError, TypeError):
        raise HTTPException(  # noqa: B904
            status_code=401, detail="Invalid X-Sybilcore-Timestamp value"
        )

    now_ts = int(time.time())
    delta = abs(now_ts - request_ts)
    if delta > TIMESTAMP_TOLERANCE_SECONDS:
        raise HTTPException(
            status_code=401,
            detail=f"Timestamp out of tolerance ({delta}s > {TIMESTAMP_TOLERANCE_SECONDS}s)",
        )

    # --- Signature check ---
    token = _extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")

    expected = _compute_expected_signature(body, secret)

    # Timing-safe comparison (no early exit on mismatch).
    if not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid signature")


# ---------------------------------------------------------------------------
# Auth middleware (FastAPI middleware)
# ---------------------------------------------------------------------------


async def _auth_middleware(request: Request, call_next: Any) -> Response:
    """FastAPI middleware enforcing auth on all routes.

    Middleware order:
        1. Body size guard (413)
        2. Rate limit guard (429)
        3. Auth guard — skipped in DEV mode (401)
        4. Call the actual endpoint
    """
    # --- 1. Body size limit ---
    # Read the body into memory; FastAPI makes it available via request.body().
    body: bytes = await request.body()
    if len(body) > MAX_BODY_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": f"Request body exceeds maximum size ({MAX_BODY_BYTES} bytes)"},
        )

    # --- 2. Rate limit ---
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limiter.is_allowed(client_ip):
        logger.warning("Rate limit exceeded for %s", client_ip)
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again later."},
        )

    # --- 3. Auth ---
    # _get_secret() returns None in DEV mode (auth bypassed) or raises RuntimeError
    # if secret is unset in production mode (fail-closed at startup).
    try:
        secret = _get_secret()
    except RuntimeError as exc:
        logger.critical("Witness gate startup check failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"detail": "Server misconfigured: auth secret not set. See server logs."},
        )

    if secret is not None:
        # Production mode — verify the request.
        try:
            _verify_request(
                authorization=request.headers.get("Authorization"),
                timestamp_header=request.headers.get("X-Sybilcore-Timestamp"),
                body=body,
                secret=secret,
            )
        except HTTPException as exc:
            logger.warning(
                "Auth failure for %s %s from %s: %s",
                request.method,
                request.url.path,
                client_ip,
                exc.detail,
            )
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    # secret is None → DEV mode, no auth check

    return await call_next(request)


# ---------------------------------------------------------------------------
# Tier enforcement policy
# ---------------------------------------------------------------------------
# SCOUT_REPORT.md §6: "ENFORCEMENT tier = allow nuke; FLAGGED tier = escalate to Deacon first"
# Mapping to SybilCore's AgentTier enum:
#
#   LETHAL_ELIMINATOR (coeff >= 300) → allow_kill = True
#       Agent is in enforcement-required tier. NukePolecat may proceed.
#   FLAGGED (200 <= coeff < 300)     → allow_kill = False (escalate first)
#       Deacon should review before nuke. Return allow_kill=False with instructions.
#   CLOUDED (100 <= coeff < 200)     → allow_kill = False (agent needs monitoring)
#   CLEAR (coeff < 100)             → allow_kill = False (agent is within normal range)

TIERS_THAT_ALLOW_KILL: frozenset[AgentTier] = frozenset({
    AgentTier.LETHAL_ELIMINATOR,
})

# Scoring window for on-demand coefficient computation (seconds).
# Production should align with the MayorTap SCORING_WINDOW_SECONDS.
ON_DEMAND_WINDOW_SECONDS: int = 7 * 24 * 3600


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class PreNukeRequest(BaseModel):
    """Request body POSTed by Gastown's Witness before executing NukePolecat.

    Fields mirror the Gastown polecat identity structure from SCOUT_REPORT.md §5:
        agent_id: Full agent address (e.g. "gastown/polecats/nux")
        rig_name: The rig name (e.g. "gastown")
        polecat_name: The polecat's short name (e.g. "nux")
        events: Optional list of GtEvent dicts for on-demand scoring.
                If empty, the endpoint uses the in-memory profile store.
        reason: Human-readable reason for the nuke request (from Witness AI).
    """

    agent_id: str = Field(description="Full agent address, e.g. 'gastown/polecats/nux'")
    rig_name: str = Field(description="Rig name, e.g. 'gastown'")
    polecat_name: str = Field(description="Polecat short name, e.g. 'nux'")
    events: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Optional GtEvent dicts for on-demand scoring",
    )
    reason: str = Field(
        default="",
        description="Reason for the nuke request from Witness AI",
    )


class PreNukeResponse(BaseModel):
    """Response returned to Gastown's Witness.

    The Go Witness deserializes this to determine whether to proceed with NukePolecat.
    """

    allow_kill: bool = Field(description="Whether NukePolecat may proceed")
    coefficient: float = Field(description="Current agent coefficient (0-500)")
    tier: str = Field(description="Current agent tier name")
    brains_fired: list[str] = Field(
        description="Brains that contributed non-zero scores",
    )
    reasoning: str = Field(description="Human-readable enforcement decision reasoning")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="When this decision was made (UTC ISO-8601)",
    )


# ---------------------------------------------------------------------------
# In-memory coefficient store
# ---------------------------------------------------------------------------


class CoefficientStore:
    """Simple in-memory store mapping agent_id → CoefficientSnapshot.

    In production, this would be backed by a persistent store (Redis, Postgres)
    updated continuously by the MayorTap streaming runner.

    For Phase 2B, it serves as the source of truth for the Witness gate.
    """

    def __init__(self) -> None:
        self._store: dict[str, CoefficientSnapshot] = {}

    def update(self, agent_id: str, snapshot: CoefficientSnapshot) -> None:
        """Store or update a snapshot for an agent."""
        self._store[agent_id] = snapshot

    def get(self, agent_id: str) -> CoefficientSnapshot | None:
        """Retrieve the latest snapshot for an agent, or None if unknown."""
        return self._store.get(agent_id)

    def all_agents(self) -> dict[str, CoefficientSnapshot]:
        """Return a copy of all stored snapshots."""
        return dict(self._store)


# Module-level store — shared between endpoint handlers
_coefficient_store = CoefficientStore()


def get_coefficient_store() -> CoefficientStore:
    """Return the module-level coefficient store (injectable for testing)."""
    return _coefficient_store


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------


def _compute_on_demand_score(
    agent_id: str,
    raw_events: list[dict[str, Any]],
    brains: list[BaseBrain],
) -> CoefficientSnapshot:
    """Compute a coefficient from a list of raw GtEvent dicts.

    Used when the Witness passes events directly in the request body.

    Args:
        agent_id: Agent being scored.
        raw_events: List of GtEvent dicts from the Witness request.
        brains: Brain instances to use.

    Returns:
        CoefficientSnapshot with the computed coefficient and tier.
    """
    from integrations.gastown.phase2b_mayor_tap.runtime_adapter import adapt_runtime_event

    adapted: list[Event] = []
    for row in raw_events:
        try:
            adapted.append(adapt_runtime_event(row))
        except ValueError as exc:
            logger.warning("Skipping malformed event in on-demand score: %s", exc)

    # Filter to events for this specific agent
    agent_events = [e for e in adapted if e.agent_id == agent_id]

    if not agent_events:
        # No events for this agent — return a zero score
        calculator = CoefficientCalculator()
        brain_scores: list[BrainScore] = [
            BrainScore(
                brain_name=b.name,
                value=0.0,
                confidence=0.1,
                reasoning="No events provided",
            )
            for b in brains
        ]
        return calculator.calculate(brain_scores)

    calculator = CoefficientCalculator(window_seconds=ON_DEMAND_WINDOW_SECONDS)
    brain_scores_computed = [b.score(agent_events) for b in brains]
    return calculator.calculate(brain_scores_computed)


def _make_enforcement_decision(
    snapshot: CoefficientSnapshot,
    agent_id: str,
    reason: str,
) -> PreNukeResponse:
    """Apply the enforcement policy to a coefficient snapshot.

    Policy from SCOUT_REPORT.md §6:
        LETHAL_ELIMINATOR → allow kill
        FLAGGED → block kill, escalate to Deacon
        CLOUDED → block kill, continue monitoring
        CLEAR → block kill, agent is behaving normally

    Args:
        snapshot: Current coefficient snapshot.
        agent_id: Agent being evaluated.
        reason: Witness-provided reason for the nuke request.

    Returns:
        PreNukeResponse with the enforcement decision.
    """
    tier = snapshot.tier
    allow_kill = tier in TIERS_THAT_ALLOW_KILL

    brains_fired = [
        brain_name for brain_name, score_val in snapshot.brain_scores.items()
        if score_val > 0.0
    ]

    if allow_kill:
        reasoning = (
            f"Agent {agent_id} is in LETHAL_ELIMINATOR tier "
            f"(coefficient={snapshot.coefficient:.1f}). "
            f"NukePolecat may proceed. Witness reason: {reason}"
        )
    elif tier == AgentTier.FLAGGED:
        reasoning = (
            f"Agent {agent_id} is in FLAGGED tier "
            f"(coefficient={snapshot.coefficient:.1f}). "
            f"Escalate to Deacon before proceeding with NukePolecat. "
            f"Witness reason: {reason}"
        )
    elif tier == AgentTier.CLOUDED:
        reasoning = (
            f"Agent {agent_id} is in CLOUDED tier "
            f"(coefficient={snapshot.coefficient:.1f}). "
            f"Agent is elevated but not in enforcement range. Continue monitoring. "
            f"Witness reason: {reason}"
        )
    else:
        reasoning = (
            f"Agent {agent_id} is in CLEAR tier "
            f"(coefficient={snapshot.coefficient:.1f}). "
            f"Agent is within normal behavioral range. "
            f"NukePolecat vetoed by SybilCore third gate. "
            f"Witness reason: {reason}"
        )

    return PreNukeResponse(
        allow_kill=allow_kill,
        coefficient=snapshot.coefficient,
        tier=tier.value,
        brains_fired=brains_fired,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SybilCore Witness Third Gate",
    description=(
        "HTTP endpoint for Gastown's Witness to check SybilCore coefficient "
        "before executing NukePolecat. "
        "Integration point: internal/witness/handlers.go:838-844 (Phase 2B). "
        "All endpoints require HMAC-SHA256 bearer token authentication."
    ),
    version="0.2.0",
)

app.middleware("http")(_auth_middleware)


@app.post("/witness/pre-nuke", response_model=PreNukeResponse)
async def pre_nuke_check(request: PreNukeRequest) -> PreNukeResponse:
    """Coefficient check before NukePolecat.

    Gastown's Witness POSTs here with polecat identity (and optionally recent
    GtEvents). Returns allow_kill=True only if the agent is in LETHAL_ELIMINATOR tier.

    Decision policy (SCOUT_REPORT.md §6):
        - LETHAL_ELIMINATOR → allow_kill = True
        - FLAGGED → allow_kill = False (escalate to Deacon)
        - CLOUDED / CLEAR → allow_kill = False (normal range or monitoring)

    The Go Witness should interpret allow_kill=False as a veto signal.
    It should NOT proceed with NukePolecat until SybilCore gives the green light.
    """
    brains = get_default_brains()
    store = get_coefficient_store()

    if request.events:
        # On-demand scoring from provided events
        snapshot = _compute_on_demand_score(
            agent_id=request.agent_id,
            raw_events=request.events,
            brains=brains,
        )
        store.update(request.agent_id, snapshot)
        logger.info(
            "On-demand score for %s: %.1f (%s) from %d events",
            request.agent_id,
            snapshot.coefficient,
            snapshot.tier.value,
            len(request.events),
        )
    else:
        # Use cached snapshot from the store
        snapshot = store.get(request.agent_id)
        if snapshot is None:
            # No data — safe default: veto kill (unknown agents are not LETHAL_ELIMINATOR)
            logger.warning(
                "No coefficient data for agent %s. Vetoing NukePolecat as safe default.",
                request.agent_id,
            )
            return PreNukeResponse(
                allow_kill=False,
                coefficient=0.0,
                tier=AgentTier.CLEAR.value,
                brains_fired=[],
                reasoning=(
                    f"No SybilCore data for agent {request.agent_id}. "
                    f"Vetoing NukePolecat as safe default. "
                    f"Feed events to the MayorTap stream to build a profile."
                ),
            )

    return _make_enforcement_decision(
        snapshot=snapshot,
        agent_id=request.agent_id,
        reason=request.reason,
    )


@app.get("/witness/status")
async def status() -> dict[str, Any]:
    """Health check and summary of tracked agents."""
    store = get_coefficient_store()
    snapshots = store.all_agents()
    return {
        "status": "ok",
        "agents_tracked": len(snapshots),
        "timestamp": datetime.now(UTC).isoformat(),
        "agents": [
            {
                "agent_id": aid,
                "tier": snap.tier.value,
                "coefficient": snap.coefficient,
            }
            for aid, snap in snapshots.items()
        ],
    }


class IngestRequest(BaseModel):
    """Request body for pushing a coefficient snapshot into the store."""

    agent_id: str = Field(description="Agent identifier")
    coefficient: float = Field(ge=0.0, le=500.0, description="Coefficient value 0-500")
    tier: str = Field(description="Tier name (clear|clouded|flagged|lethal_eliminator)")


@app.post("/witness/ingest")
async def ingest_snapshot(request: IngestRequest) -> dict[str, Any]:
    """Ingest a coefficient snapshot from an external source (e.g., MayorTap).

    This endpoint allows the MayorTap streaming runner to push score updates
    into the Witness gate's store without requiring a shared Python process.

    Args:
        request: IngestRequest with agent_id, coefficient, and tier.

    Returns:
        Acknowledgment dict.
    """
    try:
        tier_enum = AgentTier(request.tier)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tier value: {request.tier!r}. Valid: {list(AgentTier)}",
        ) from exc

    snapshot = CoefficientSnapshot(
        coefficient=request.coefficient,
        tier=tier_enum,
        timestamp=datetime.now(UTC),
    )
    store = get_coefficient_store()
    store.update(request.agent_id, snapshot)
    return {"status": "ok", "agent_id": request.agent_id, "tier": tier_enum.value}
