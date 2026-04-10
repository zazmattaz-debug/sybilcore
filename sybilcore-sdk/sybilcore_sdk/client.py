"""SybilCore client — the main entry point.

Two modes:
  * Local mode (default): runs the brain set in-process via the bundled
    `sybilcore` package. Best for self-hosted deployments and unit tests.
  * Remote mode: talks to a SybilCore HTTP API endpoint. Activated by
    passing `endpoint=...` (and optionally `api_key`).

Both modes expose the same surface:

    sc = SybilCore()
    result = sc.score(events)
    results = sc.score_batch([events1, events2])
    score = await sc.score_event(event)            # async, single event
    async for s in sc.stream(events_iter): ...     # streaming

Switching modes is a one-line change — your application code stays the same.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Iterable, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx

from sybilcore_sdk.exceptions import (
    SybilCoreAPIError,
    SybilCoreAuthError,
    SybilCoreLocalImportError,
    SybilCoreRateLimitError,
)
from sybilcore_sdk.models import Event, EventType, ScoreResult, Tier

if TYPE_CHECKING:
    from sybilcore.brains.base import BaseBrain
    from sybilcore.core.coefficient import CoefficientCalculator

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30.0


class SybilCore:
    """SybilCore SDK client.

    Args:
        api_key: API key for remote mode. Read from env var `SYBILCORE_API_KEY`
            if not supplied.
        endpoint: Remote API base URL (e.g., "https://api.sybilcore.com").
            If `None`, the client runs in local mode (in-process scoring).
        timeout: HTTP timeout in seconds (remote mode only).
        weight_overrides: Explicit brain weight overrides (local mode only).
            Takes precedence over ``weights`` when both are supplied.
        weights: Preset selector or explicit weight map (local mode only).
            Accepts:

            * ``"default"`` (the default) — use the baseline weights shipped
              in ``sybilcore.core.config.DEFAULT_BRAIN_WEIGHTS``.
            * ``"optimized"`` — load ``OPTIMIZED_WEIGHTS_V4`` from
              ``sybilcore.core.weight_presets``. These were calibrated on
              synthetic negatives and are still being validated against
              raw Moltbook data — treat as experimental.
            * ``dict[str, float]`` — passed through verbatim as overrides.
    """

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        weight_overrides: dict[str, float] | None = None,
        weights: str | dict[str, float] = "default",
    ) -> None:
        self._endpoint = endpoint.rstrip("/") if endpoint else None
        self._api_key = api_key
        self._timeout = timeout
        self._weight_overrides = self._resolve_weight_config(weight_overrides, weights)

        # Lazy-init local components only when needed.
        self._brains: list[BaseBrain] | None = None
        self._calculator: CoefficientCalculator | None = None

        if self._endpoint is None:
            # Validate the local package is importable up front for fast failure.
            self._ensure_local_ready()

    # ── Mode helpers ──────────────────────────────────────────────

    @property
    def is_local(self) -> bool:
        """True if running in local (in-process) mode."""
        return self._endpoint is None

    @property
    def is_remote(self) -> bool:
        """True if talking to a remote API endpoint."""
        return self._endpoint is not None

    # ── Public API ────────────────────────────────────────────────

    def score(self, events: Sequence[Event]) -> ScoreResult:
        """Score a single agent's event stream.

        Args:
            events: Non-empty sequence of `Event` objects (all from the same agent).

        Returns:
            `ScoreResult` with coefficient, tier, brain breakdown.

        Raises:
            ValueError: If `events` is empty.
            SybilCoreAPIError: On remote API failure.
        """
        if not events:
            msg = "events sequence cannot be empty"
            raise ValueError(msg)

        if self.is_local:
            return self._score_local(events)
        return self._score_remote(events)

    def score_batch(self, batches: Iterable[Sequence[Event]]) -> list[ScoreResult]:
        """Score multiple agents in one call.

        Args:
            batches: Iterable of event sequences. Each sequence is a single agent.

        Returns:
            One `ScoreResult` per non-empty batch, in input order.
        """
        results: list[ScoreResult] = []
        for batch in batches:
            if not batch:
                continue
            results.append(self.score(batch))
        return results

    async def score_event(self, event: Event) -> ScoreResult:
        """Async single-event scoring (convenience for streaming pipelines).

        Internally wraps `score()` so it can be awaited from async loops
        without blocking on the event loop in trivial cases.
        """
        return await asyncio.to_thread(self.score, [event])

    async def stream(self, events: AsyncIterator[Event]) -> AsyncIterator[ScoreResult]:
        """Stream scores: yield one ScoreResult per event consumed.

        For batched accumulation, the caller should buffer events and call
        `score()` directly — this helper is for low-latency single-event use.
        """
        async for event in events:
            yield await self.score_event(event)

    # ── Local-mode implementation ─────────────────────────────────

    @staticmethod
    def _resolve_weight_config(
        weight_overrides: dict[str, float] | None,
        weights: str | dict[str, float],
    ) -> dict[str, float]:
        """Merge the legacy ``weight_overrides`` kwarg with the new ``weights`` kwarg.

        Explicit ``weight_overrides`` always win. Otherwise ``weights`` is
        interpreted as either a preset name or a verbatim weight dict.
        """
        if weight_overrides:
            return dict(weight_overrides)

        if isinstance(weights, dict):
            return dict(weights)

        if not isinstance(weights, str):
            msg = (
                "weights must be a preset name (str) or a dict[str, float]; "
                f"got {type(weights).__name__}"
            )
            raise TypeError(msg)

        if weights == "default":
            return {}

        try:
            from sybilcore.core.weight_presets import resolve_preset
        except ImportError as exc:
            msg = (
                "Named weight presets require the 'sybilcore' package. "
                "Install it with: pip install sybilcore-sdk[local]"
            )
            raise SybilCoreLocalImportError(msg) from exc

        try:
            return resolve_preset(weights)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc

    def _ensure_local_ready(self) -> None:
        if self._brains is not None and self._calculator is not None:
            return
        try:
            from sybilcore.brains import get_default_brains
            from sybilcore.core.coefficient import CoefficientCalculator
        except ImportError as exc:
            msg = (
                "Local mode requires the 'sybilcore' package. "
                "Install it with: pip install sybilcore-sdk[local]"
            )
            raise SybilCoreLocalImportError(msg) from exc

        self._brains = get_default_brains()
        self._calculator = CoefficientCalculator(weight_overrides=self._weight_overrides or None)

    def _score_local(self, events: Sequence[Event]) -> ScoreResult:
        self._ensure_local_ready()
        assert self._brains is not None
        assert self._calculator is not None

        from sybilcore.models.event import Event as InternalEvent
        from sybilcore.models.event import EventType as InternalEventType

        # Convert SDK Events → internal Events
        internal_events = [
            InternalEvent(
                event_id=ev.event_id,
                agent_id=ev.agent_id,
                event_type=InternalEventType(ev.event_type.value),
                timestamp=ev.timestamp,
                content=ev.content[:10_000],
                metadata=ev.metadata,
                source=ev.source,
            )
            for ev in events
        ]

        agent_id = internal_events[0].agent_id
        start = time.monotonic()
        brain_scores = [brain.score(internal_events) for brain in self._brains]
        snapshot = self._calculator.calculate(brain_scores)
        processing_ms = (time.monotonic() - start) * 1000.0

        return ScoreResult(
            agent_id=agent_id,
            coefficient=snapshot.effective_coefficient,
            surface_coefficient=snapshot.coefficient,
            semantic_coefficient=snapshot.semantic_coefficient,
            tier=Tier(snapshot.tier.value),
            brains={k: round(v, 2) for k, v in snapshot.brain_scores.items()},
            brain_count=snapshot.brain_count,
            scoring_config_version=snapshot.scoring_config_version,
            timestamp=snapshot.timestamp,
            detected=snapshot.tier.value != "clear",
            processing_ms=round(processing_ms, 2),
        )

    # ── Remote-mode implementation ────────────────────────────────

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": f"sybilcore-sdk/{_sdk_version()}"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _events_payload(self, events: Sequence[Event]) -> dict[str, Any]:
        return {
            "events": [
                {
                    "agent_id": ev.agent_id,
                    "event_type": ev.event_type.value,
                    "content": ev.content,
                    "metadata": ev.metadata,
                    "source": ev.source,
                }
                for ev in events
            ]
        }

    def _score_remote(self, events: Sequence[Event]) -> ScoreResult:
        url = f"{self._endpoint}/score"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(url, json=self._events_payload(events), headers=self._headers())
        except httpx.HTTPError as exc:
            raise SybilCoreAPIError(f"HTTP error contacting {url}: {exc}") from exc

        self._raise_for_status(resp)
        return self._parse_remote_score(resp.json())

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        if 200 <= resp.status_code < 300:
            return
        body = resp.text[:500]
        if resp.status_code in (401, 403):
            raise SybilCoreAuthError(
                f"Authentication failed ({resp.status_code})",
                status_code=resp.status_code,
                body=body,
            )
        if resp.status_code == 429:
            retry_after_raw = resp.headers.get("Retry-After")
            try:
                retry_after = float(retry_after_raw) if retry_after_raw else None
            except ValueError:
                retry_after = None
            raise SybilCoreRateLimitError(
                "Rate limit exceeded",
                retry_after=retry_after,
                status_code=resp.status_code,
                body=body,
            )
        raise SybilCoreAPIError(
            f"API error {resp.status_code}: {body}",
            status_code=resp.status_code,
            body=body,
        )

    @staticmethod
    def _parse_remote_score(payload: dict[str, Any]) -> ScoreResult:
        ts_raw = payload.get("timestamp")
        if isinstance(ts_raw, str):
            try:
                timestamp = datetime.fromisoformat(ts_raw)
            except ValueError:
                timestamp = datetime.now(UTC)
        else:
            timestamp = datetime.now(UTC)

        return ScoreResult(
            agent_id=payload.get("agent_id", "unknown"),
            coefficient=float(payload.get("effective_coefficient", payload.get("coefficient", 0.0))),
            surface_coefficient=float(payload.get("surface_coefficient", 0.0)),
            semantic_coefficient=float(payload.get("semantic_coefficient", 0.0)),
            tier=Tier(payload.get("tier", "clear")),
            brains=payload.get("brain_scores", {}),
            brain_count=int(payload.get("brain_count", 0)),
            scoring_config_version=payload.get("scoring_config_version", ""),
            timestamp=timestamp,
            detected=bool(payload.get("detected", False)),
            processing_ms=float(payload.get("processing_ms", 0.0)),
        )

    # ── Webhook helper ────────────────────────────────────────────

    def register_webhook(self, callback_url: str, min_tier: Tier = Tier.FLAGGED) -> dict[str, Any]:
        """Register a webhook URL for high-trust alerts.

        Remote-mode only. The remote server will POST to `callback_url`
        whenever any scored agent reaches `min_tier` or higher.
        """
        if not self.is_remote:
            msg = "register_webhook() requires remote mode (set endpoint=...)"
            raise RuntimeError(msg)
        url = f"{self._endpoint}/webhook"
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                url,
                json={"callback_url": callback_url, "min_tier": min_tier.value},
                headers=self._headers(),
            )
        self._raise_for_status(resp)
        return resp.json()


def _sdk_version() -> str:
    try:
        from sybilcore_sdk import __version__
        return __version__
    except ImportError:
        return "unknown"


# ── Module-level convenience helper ───────────────────────────────

_default_client: SybilCore | None = None


def quick_score(events: Sequence[Event]) -> ScoreResult:
    """One-line scoring helper using a shared local client."""
    global _default_client
    if _default_client is None:
        _default_client = SybilCore()
    return _default_client.score(events)


__all__ = ["SybilCore", "quick_score"]


# Ensure EventType is re-exported here for convenience.
_ = EventType
