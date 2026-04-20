"""Mayor tap — streaming runner for GtEvent firehose (Phase 2B).

Reads line-delimited GtEvent JSON from stdin or a file, maintains per-agent
rolling windows, computes SybilCore coefficients on a rolling basis, and
writes tier transitions to a structured log.

Design notes:
    - Async / buffered for realistic firehose use (asyncio-based).
    - Per-agent deque windows of configurable size (default 200 events).
    - Rolling rescore triggered on every N events per agent (default 10).
    - Tier transitions logged to structured JSONL output.
    - All state is immutable-update style (AgentProfile.with_new_reading).
    - No live stream access in Phase 2B — validated against synthetic fixtures.
      Label: "awaiting live corpus" prototype.

Usage:
    # From stdin (pipe from live Gastown or synthetic firehose):
    python -m integrations.gastown.phase2b_mayor_tap.mayor_tap --window 200

    # From file:
    python -m integrations.gastown.phase2b_mayor_tap.mayor_tap \
        --file /path/to/events.jsonl --window 200

    # From synthetic firehose (testing):
    python -m integrations.gastown.phase2b_mayor_tap.mayor_tap --synthetic
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TextIO

from integrations.gastown.phase2b_mayor_tap.runtime_adapter import adapt_runtime_event
from sybilcore.brains import get_default_brains
from sybilcore.core.coefficient import CoefficientCalculator
from sybilcore.models.agent import AgentProfile, AgentTier, CoefficientSnapshot

if TYPE_CHECKING:
    from sybilcore.brains.base import BaseBrain
    from sybilcore.models.event import Event

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Number of events per agent to buffer in the rolling window.
DEFAULT_WINDOW_SIZE: int = 200

# Rescore every N new events per agent.
DEFAULT_RESCORE_INTERVAL: int = 10

# Scoring window for the calculator (seconds).
# Phase 2B uses a 7-day window to capture synthetic fixture history.
# Production should use a shorter window (e.g., 3600s).
SCORING_WINDOW_SECONDS: int = 7 * 24 * 3600

# Minimum events before we attempt a coefficient calculation.
MIN_EVENTS_FOR_SCORE: int = 3


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


class AgentWindow:
    """Mutable rolling event window for a single agent.

    Stores the last `max_size` events in chronological order.
    Tracks count of events received since last rescore.
    """

    def __init__(self, agent_id: str, max_size: int = DEFAULT_WINDOW_SIZE) -> None:
        self.agent_id = agent_id
        self._events: deque[Event] = deque(maxlen=max_size)
        self.events_since_rescore: int = 0
        self.profile: AgentProfile = AgentProfile(
            agent_id=agent_id,
            name=agent_id,
        )

    def add_event(self, event: Event) -> None:
        """Add an event to the rolling window."""
        self._events.append(event)
        self.events_since_rescore += 1

    def get_events(self) -> list[Event]:
        """Return current window as a sorted list."""
        return sorted(self._events, key=lambda e: e.timestamp)

    @property
    def event_count(self) -> int:
        """Current number of events in window."""
        return len(self._events)


# ---------------------------------------------------------------------------
# Tier transition logging
# ---------------------------------------------------------------------------


def make_transition_record(
    agent_id: str,
    old_tier: AgentTier,
    new_tier: AgentTier,
    snapshot: CoefficientSnapshot,
    event_count: int,
) -> dict[str, Any]:
    """Create a structured tier transition log record."""
    return {
        "ts": datetime.now(UTC).isoformat(),
        "event": "tier_transition",
        "agent_id": agent_id,
        "old_tier": old_tier.value,
        "new_tier": new_tier.value,
        "coefficient": snapshot.coefficient,
        "brain_scores": snapshot.brain_scores,
        "event_count": event_count,
    }


def make_score_record(
    agent_id: str,
    snapshot: CoefficientSnapshot,
    event_count: int,
) -> dict[str, Any]:
    """Create a structured scoring log record."""
    return {
        "ts": datetime.now(UTC).isoformat(),
        "event": "score_update",
        "agent_id": agent_id,
        "tier": snapshot.tier.value,
        "coefficient": snapshot.coefficient,
        "brain_scores": snapshot.brain_scores,
        "event_count": event_count,
    }


# ---------------------------------------------------------------------------
# Core streaming runner
# ---------------------------------------------------------------------------


class MayorTap:
    """Streaming runner that consumes GtEvent firehose and maintains agent scores.

    Maintains per-agent rolling event windows and computes SybilCore
    coefficients on a rolling basis. Writes tier transitions to output.
    """

    def __init__(
        self,
        brains: list[BaseBrain] | None = None,
        window_size: int = DEFAULT_WINDOW_SIZE,
        rescore_interval: int = DEFAULT_RESCORE_INTERVAL,
        output: TextIO | None = None,
    ) -> None:
        """Initialize the Mayor Tap.

        Args:
            brains: Brain instances to use. Defaults to get_default_brains().
            window_size: Max events per agent in rolling window.
            rescore_interval: Rescore after this many new events per agent.
            output: Output stream for structured log records. Defaults to stdout.
        """
        self._brains: list[BaseBrain] = brains or get_default_brains()
        self._window_size = window_size
        self._rescore_interval = rescore_interval
        self._output: TextIO = output or sys.stdout
        self._calculator = CoefficientCalculator(
            window_seconds=SCORING_WINDOW_SECONDS,
        )
        self._agent_windows: dict[str, AgentWindow] = {}
        self._transition_log: list[dict[str, Any]] = []

    def _get_or_create_window(self, agent_id: str) -> AgentWindow:
        """Get or create the rolling window for an agent."""
        if agent_id not in self._agent_windows:
            self._agent_windows[agent_id] = AgentWindow(
                agent_id=agent_id,
                max_size=self._window_size,
            )
        return self._agent_windows[agent_id]

    def _rescore_agent(self, window: AgentWindow) -> CoefficientSnapshot | None:
        """Run all brains against the agent's current window.

        Returns None if there are not enough events to score.
        """
        events = window.get_events()
        if len(events) < MIN_EVENTS_FOR_SCORE:
            return None

        brain_scores = [brain.score(events) for brain in self._brains]
        snapshot = self._calculator.calculate(brain_scores)
        return snapshot

    def _emit(self, record: dict[str, Any]) -> None:
        """Write a structured log record to output."""
        self._output.write(json.dumps(record) + "\n")
        self._output.flush()

    def process_event(self, raw_line: str) -> None:
        """Process a single GtEvent JSON line from the firehose.

        Parses the line, adapts it to a SybilCore Event, adds it to the
        agent's rolling window, and rescores if the interval has elapsed.

        Args:
            raw_line: Raw JSON string from the `.events.jsonl` firehose.
        """
        line = raw_line.strip()
        if not line:
            return

        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return  # Skip malformed lines silently

        try:
            event = adapt_runtime_event(row)
        except ValueError:
            return  # Skip events with missing required fields

        window = self._get_or_create_window(event.agent_id)
        old_tier = window.profile.current_tier
        window.add_event(event)

        if window.events_since_rescore >= self._rescore_interval:
            snapshot = self._rescore_agent(window)
            # Reset counter regardless — even if too few events, wait another full
            # interval before trying again (prevents tight loop before MIN_EVENTS_FOR_SCORE).
            window.events_since_rescore = 0
            if snapshot is not None:
                new_profile = window.profile.with_new_reading(
                    snapshot=snapshot,
                    events_processed=self._rescore_interval,
                )
                window.profile = new_profile

                # Always emit score update
                self._emit(make_score_record(
                    agent_id=event.agent_id,
                    snapshot=snapshot,
                    event_count=window.event_count,
                ))

                # Emit tier transition if tier changed
                if snapshot.tier != old_tier:
                    record = make_transition_record(
                        agent_id=event.agent_id,
                        old_tier=old_tier,
                        new_tier=snapshot.tier,
                        snapshot=snapshot,
                        event_count=window.event_count,
                    )
                    self._transition_log.append(record)
                    self._emit(record)

    def final_score_all(self) -> dict[str, CoefficientSnapshot]:
        """Run a final rescore of all agents regardless of interval.

        Called at end of stream to ensure all agents have a final score.

        Returns:
            Dict mapping agent_id → final CoefficientSnapshot.
        """
        results: dict[str, CoefficientSnapshot] = {}
        for agent_id, window in self._agent_windows.items():
            snapshot = self._rescore_agent(window)
            if snapshot is None:
                continue
            window.profile = window.profile.with_new_reading(snapshot=snapshot)
            results[agent_id] = snapshot
            self._emit(make_score_record(
                agent_id=agent_id,
                snapshot=snapshot,
                event_count=window.event_count,
            ))
        return results

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of all tracked agents and their current scores."""
        agents: list[dict[str, Any]] = []
        for agent_id, window in self._agent_windows.items():
            agents.append({
                "agent_id": agent_id,
                "tier": window.profile.current_tier.value,
                "coefficient": window.profile.current_coefficient,
                "event_count": window.event_count,
            })
        agents.sort(key=lambda a: a["coefficient"], reverse=True)
        return {
            "agents_tracked": len(self._agent_windows),
            "tier_transitions": len(self._transition_log),
            "agents": agents,
        }


async def run_from_stream(
    stream: asyncio.StreamReader,
    tap: MayorTap,
) -> None:
    """Async loop: read lines from stream and feed to tap.

    Args:
        stream: AsyncIO stream reader (stdin or file).
        tap: MayorTap instance to process events.
    """
    while True:
        try:
            line = await stream.readline()
        except asyncio.IncompleteReadError:
            break
        if not line:
            break
        tap.process_event(line.decode("utf-8", errors="replace"))


def run_from_lines(lines: list[str], tap: MayorTap) -> None:
    """Synchronous runner: process a list of pre-read lines.

    Args:
        lines: List of raw JSON strings.
        tap: MayorTap instance.
    """
    for line in lines:
        tap.process_event(line)


def run_from_file(path: str, tap: MayorTap) -> None:
    """Process all events from a file path.

    Args:
        path: Path to `.events.jsonl` file.
        tap: MayorTap instance.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            tap.process_event(line)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mayor Tap — SybilCore streaming scorer for GtEvent firehose",
    )
    parser.add_argument(
        "--file", "-f",
        metavar="PATH",
        help="Read events from a .events.jsonl file instead of stdin",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run against built-in synthetic fixture corpus (for testing)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Rolling window size per agent (default: {DEFAULT_WINDOW_SIZE})",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_RESCORE_INTERVAL,
        help=f"Rescore interval in events per agent (default: {DEFAULT_RESCORE_INTERVAL})",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print agent summary JSON to stderr when stream ends",
    )
    return parser


def main() -> None:
    """CLI entry point for the Mayor Tap."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    tap = MayorTap(
        window_size=args.window,
        rescore_interval=args.interval,
    )

    if args.synthetic:
        from integrations.gastown.phase2b_mayor_tap.synthetic_firehose import (
            stream_synthetic_corpus,
        )
        for line in stream_synthetic_corpus():
            tap.process_event(line)
    elif args.file:
        run_from_file(args.file, tap)
    else:
        # Read from stdin line by line (blocking)
        for line in sys.stdin:
            tap.process_event(line)

    tap.final_score_all()

    if args.summary:
        import json as _json
        print(_json.dumps(tap.get_summary(), indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
