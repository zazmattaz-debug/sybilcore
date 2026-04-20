"""Gastown fixture replay driver.

Loads all Gastown bead-tracker JSONL fixtures, adapts them to SybilCore Events,
runs the 13-brain CoefficientCalculator per agent, and writes two output files:

  replay_results.json         — per-agent coefficient, tier, brain scores
  detection_latency.json      — per-agent tier-crossing timestamps

Scoring window override:
  The default SCORING_WINDOW_SECONDS is 3600 (1 hour).  All Gastown fixtures
  are from Feb-Mar 2026 (~46-53 days old at replay time).  If the default
  window were used, every event would be filtered out and every agent would
  score zero — which would be misleading.

  Instead, replay uses REPLAY_WINDOW_SECONDS = 90 days (7_776_000 seconds).
  This covers the full 7-day fixture span with margin and is the only
  meaningful choice for historical fixture replay.

  This is documented as a replay-mode decision, not a production config change.
  Production deployments should use the default 1-hour window against live streams.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

# ---------------------------------------------------------------------------
# Path setup — allow running as a script from any cwd.
# ---------------------------------------------------------------------------
_SYBILCORE_ROOT = Path(__file__).resolve().parents[4]
if str(_SYBILCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SYBILCORE_ROOT))

from sybilcore.brains import get_default_brains  # noqa: E402
from sybilcore.core.coefficient import CoefficientCalculator  # noqa: E402
from sybilcore.models.agent import AgentTier  # noqa: E402

# Adapter lives alongside this file; use direct import when run as a script.
# When imported as a package the relative import also works if __init__.py exists.
try:
    from .adapter import adapt_fixture_file, group_events_by_agent  # noqa: E402
except ImportError:
    from adapter import (  # type: ignore[no-redef]  # noqa: E402
        adapt_fixture_file,
        group_events_by_agent,
    )

if TYPE_CHECKING:
    from sybilcore.models.event import Event

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default fixture location (overridden in tests).
DEFAULT_FIXTURE_PATH = Path("/tmp/gastown-fixtures/.beads/backup/events.jsonl")

# 90-day window so historical fixtures are not silently filtered out.
# See module docstring for rationale.
REPLAY_WINDOW_SECONDS: int = 90 * 24 * 3600  # 7_776_000

# Tiers that constitute a meaningful detection signal.
_DETECTION_TIERS: frozenset[AgentTier] = frozenset(
    [AgentTier.FLAGGED, AgentTier.LETHAL_ELIMINATOR]
)

# Output directory (same folder as this file).
_OUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Core replay logic
# ---------------------------------------------------------------------------


def _compute_time_span(events: list[Event]) -> dict[str, str | None]:
    """Return ISO timestamps for the first and last event."""
    if not events:
        return {"first_event_at": None, "last_event_at": None}
    sorted_events = sorted(events, key=lambda e: e.timestamp)
    return {
        "first_event_at": sorted_events[0].timestamp.isoformat(),
        "last_event_at": sorted_events[-1].timestamp.isoformat(),
    }


def _score_agent(
    agent_id: str,
    events: list[Event],
    all_events: list[Event],
    calculator: CoefficientCalculator,
    brains: list,
) -> dict:
    """Run full brain pipeline for one agent and return a results dict."""
    snapshot = calculator.scan_agent(agent_id, all_events, brains)
    time_span = _compute_time_span(events)
    return {
        "agent_id": agent_id,
        "coefficient": round(snapshot.coefficient, 2),
        "tier": snapshot.tier.value,
        "event_count": len(events),
        "brain_scores": {k: round(v, 2) for k, v in snapshot.brain_scores.items()},
        **time_span,
    }


def _compute_detection_latency(
    agent_id: str,
    events: list[Event],
    all_events: list[Event],  # noqa: ARG001  (kept for interface consistency)
    calculator: CoefficientCalculator,
    brains: list,
    checkpoint_step: int = 10,
) -> dict | None:
    """Compute per-tier first-crossing index and timestamp for an agent.

    Scans the agent's events at ``checkpoint_step`` intervals rather than
    event-by-event, which keeps wall-clock time tractable when brains include
    embedding models (ContrastiveBrain, SemanticBrain).

    For agents with fewer than ``checkpoint_step`` events, a single final scan
    is performed so small agents are not silently skipped.

    The crossing event index and timestamp are reported at the checkpoint
    boundary where the tier was first observed — meaning the actual first
    crossing may have occurred up to ``checkpoint_step`` events earlier.
    This is documented in detection_latency.json as "checkpoint_granularity".

    Returns None if the agent never reaches any detection tier.
    """
    crossings: dict[str, dict] = {}
    sorted_events = sorted(events, key=lambda e: e.timestamp)
    n = len(sorted_events)

    # Build checkpoint indices: every checkpoint_step, plus the final event.
    checkpoints: list[int] = list(range(checkpoint_step - 1, n - 1, checkpoint_step))
    if not checkpoints or checkpoints[-1] != n - 1:
        checkpoints.append(n - 1)

    for idx in checkpoints:
        events_so_far = sorted_events[: idx + 1]
        snapshot = calculator.scan_agent(agent_id, events_so_far, brains)
        tier = snapshot.tier

        if tier in _DETECTION_TIERS and tier.value not in crossings:
            crossings[tier.value] = {
                "first_crossed_at": sorted_events[idx].timestamp.isoformat(),
                "event_index": idx,
                "coefficient_at_crossing": round(snapshot.coefficient, 2),
                "checkpoint_granularity": checkpoint_step,
            }

        if len(crossings) == len(_DETECTION_TIERS):
            break

    if not crossings:
        return None

    return {
        "agent_id": agent_id,
        "tier_crossings": crossings,
    }


def run_replay(
    fixture_path: Path = DEFAULT_FIXTURE_PATH,
    out_dir: Path = _OUT_DIR,
    window_seconds: int = REPLAY_WINDOW_SECONDS,
) -> tuple[dict, dict]:
    """Full replay pipeline.

    Args:
        fixture_path: Path to Gastown bead-events JSONL.
        out_dir: Directory to write output JSON files.
        window_seconds: Time window passed to CoefficientCalculator.

    Returns:
        Tuple of (replay_results_dict, detection_latency_dict).

    Raises:
        FileNotFoundError: If fixture_path does not exist.
        AssertionError: If post-run sanity checks fail.
    """
    if not fixture_path.exists():
        msg = f"Fixture not found: {fixture_path}"
        raise FileNotFoundError(msg)

    # --- Load and adapt ---
    all_events: list[Event] = adapt_fixture_file(fixture_path)
    assert len(all_events) > 0, "No events loaded from fixture"  # noqa: S101

    grouped = group_events_by_agent(all_events)
    assert len(grouped) > 0, "No agents found in grouped events"  # noqa: S101

    # --- Build shared brains + calculator (instantiate once, reuse per agent) ---
    brains = get_default_brains()
    calculator = CoefficientCalculator(window_seconds=window_seconds)

    # --- Score every agent ---
    agent_results: list[dict] = []
    for agent_id, agent_events in sorted(grouped.items()):
        result = _score_agent(agent_id, agent_events, all_events, calculator, brains)
        agent_results.append(result)

    # --- Sanity checks ---
    assert len(agent_results) == len(grouped), "Result count mismatch"  # noqa: S101
    for r in agent_results:
        assert 0.0 <= r["coefficient"] <= 500.0, f"Coefficient out of range: {r}"  # noqa: S101
        assert r["event_count"] > 0, f"Agent with zero events: {r}"  # noqa: S101

    # --- Detection latency (incremental scan) ---
    # Only run incremental scan for agents with 2+ events.
    # Checkpoint step scales with event count to keep wall-clock time bounded:
    #   - < 50 events:   step=5    (~10 scans max)
    #   - 50-200 events: step=20   (~10 scans max)
    #   - > 200 events:  step=50   (~ceil(n/50) scans)
    # For agents with very large event counts (mayor: 1777 events @ 50 step
    # = ~36 scans × ~10s per scan = ~6 min), we cap at LATENCY_MAX_EVENTS
    # and document the cap in the output.
    # Skip incremental scan for agents above this threshold.
    latency_max_events = 200

    latency_records: list[dict] = []
    for agent_id, agent_events in sorted(grouped.items()):
        n_events = len(agent_events)
        if n_events < 2:
            continue
        if n_events > latency_max_events:
            # Record a note rather than running a multi-minute scan.
            reason = (
                f"event_count={n_events} exceeds latency_max_events={latency_max_events}; "
                "incremental scan omitted to keep replay wall-clock time tractable. "
                "See replay_results.json for final coefficient."
            )
            latency_records.append({
                "agent_id": agent_id,
                "skipped": True,
                "reason": reason,
            })
            continue

        step = 5 if n_events < 50 else 20
        record = _compute_detection_latency(
            agent_id, agent_events, agent_events, calculator, brains,
            checkpoint_step=step,
        )
        if record is not None:
            latency_records.append(record)

    # --- Build output dicts ---
    replay_results = {
        "generated_at": datetime.now(UTC).isoformat(),
        "fixture_path": str(fixture_path),
        "total_events": len(all_events),
        "total_agents": len(agent_results),
        "window_seconds": window_seconds,
        "agents": agent_results,
    }

    detection_latency = {
        "generated_at": datetime.now(UTC).isoformat(),
        "detection_tiers": [t.value for t in _DETECTION_TIERS],
        "agents_reaching_detection": len(latency_records),
        "records": latency_records,
    }

    # --- Write outputs ---
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "replay_results.json").write_text(
        json.dumps(replay_results, indent=2), encoding="utf-8"
    )
    (out_dir / "detection_latency.json").write_text(
        json.dumps(detection_latency, indent=2), encoding="utf-8"
    )

    return replay_results, detection_latency


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results, latency = run_replay()
    agents = results["agents"]
    tiers: dict[str, int] = {}
    for a in agents:
        tiers[a["tier"]] = tiers.get(a["tier"], 0) + 1

    n_agents = results["total_agents"]
    n_events = results["total_events"]
    print(f"Replay complete: {n_agents} agents, {n_events} events")  # keep
    print("Tier distribution:")  # keep
    for tier, count in sorted(tiers.items()):
        print(f"  {tier}: {count}")  # keep
    print(f"Agents reaching detection tier: {latency['agents_reaching_detection']}")  # keep
    print(f"Results written to: {_OUT_DIR}")  # keep
