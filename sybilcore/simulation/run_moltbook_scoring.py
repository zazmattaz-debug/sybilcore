"""Moltbook behavioral scoring experiment.

Loads the cached Moltbook dataset via MoltbookAdapter, scores every agent
through all 5 SybilCore brains + coefficient calculator, and reports
aggregate statistics + per-agent breakdowns.

NOTE: Moltbook data is historical (months old). The standard
CoefficientCalculator.scan_agent() would filter all events outside its
3600-second window. This runner bypasses the window by calling
brain.score() directly on the pre-fetched agent events, then passes the
resulting BrainScores to calculator.calculate(). This is correct —
we are doing a full-history behavioral analysis, not a live watch-window.

Usage:
    python3 -m sybilcore.simulation.run_moltbook_scoring
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sybilcore.brains import get_default_brains
from sybilcore.core.coefficient import CoefficientCalculator
from sybilcore.integrations.moltbook import MoltbookAdapter
from sybilcore.models.agent import AgentTier, CoefficientSnapshot

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).parent.parent.parent / "experiments"

# Minimum number of posts an agent needs to be scored meaningfully.
# Agents with fewer events produce mostly-noise results.
_MIN_EVENT_THRESHOLD: int = 1


def _score_all_agents(
    adapter: MoltbookAdapter,
) -> list[dict[str, Any]]:
    """Score every agent in the Moltbook dataset.

    Bypasses the time-window filter so all historical events count.

    Args:
        adapter: Loaded MoltbookAdapter instance.

    Returns:
        List of result dicts, one per agent, sorted by coefficient descending.
    """
    brains = get_default_brains()
    calculator = CoefficientCalculator()
    results: list[dict[str, Any]] = []

    all_agents = adapter.get_all_agents()
    logger.info("Scoring %d agents through %d brains...", len(all_agents), len(brains))

    for i, agent_id in enumerate(all_agents):
        events = adapter.get_agent_events(agent_id)
        if len(events) < _MIN_EVENT_THRESHOLD:
            continue

        # Score each brain directly — no time-window filter
        brain_scores = [brain.score(events) for brain in brains]
        snapshot: CoefficientSnapshot = calculator.calculate(brain_scores)

        results.append({
            "agent_id": agent_id,
            "event_count": len(events),
            "coefficient": round(snapshot.coefficient, 2),
            "tier": snapshot.tier.value,
            "brain_scores": {
                name: round(val, 2)
                for name, val in snapshot.brain_scores.items()
            },
        })

        if (i + 1) % 500 == 0:
            logger.info("  Scored %d / %d agents...", i + 1, len(all_agents))

    results.sort(key=lambda r: r["coefficient"], reverse=True)
    logger.info("Scoring complete: %d agents scored", len(results))
    return results


def _build_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate scored results into a full experiment report.

    Args:
        results: List of per-agent result dicts, sorted by coefficient desc.

    Returns:
        Report dict with stats, tier distribution, top/bottom lists, etc.
    """
    if not results:
        return {"error": "No agents scored"}

    coefficients = [r["coefficient"] for r in results]
    tier_counts: Counter[str] = Counter(r["tier"] for r in results)

    # Brain firing frequency — which brain scores > 0 most often
    brain_fire_counts: Counter[str] = Counter()
    for r in results:
        for brain_name, score_val in r["brain_scores"].items():
            if score_val > 0.0:
                brain_fire_counts[brain_name] += 1

    most_active_brain = brain_fire_counts.most_common(1)[0][0] if brain_fire_counts else "none"

    return {
        "experiment": "moltbook_behavioral_scoring",
        "timestamp": datetime.now(UTC).isoformat(),
        "total_agents_scored": len(results),
        "statistics": {
            "mean_coefficient": round(statistics.mean(coefficients), 3),
            "median_coefficient": round(statistics.median(coefficients), 3),
            "stdev_coefficient": round(statistics.stdev(coefficients), 3) if len(coefficients) > 1 else 0.0,
            "min_coefficient": round(min(coefficients), 3),
            "max_coefficient": round(max(coefficients), 3),
        },
        "tier_distribution": {
            "CLEAR": tier_counts.get(AgentTier.CLEAR.value, 0),
            "CLOUDED": tier_counts.get(AgentTier.CLOUDED.value, 0),
            "FLAGGED": tier_counts.get(AgentTier.FLAGGED.value, 0),
            "LETHAL": tier_counts.get(AgentTier.LETHAL_ELIMINATOR.value, 0),
        },
        "brain_fire_counts": dict(brain_fire_counts.most_common()),
        "most_active_brain": most_active_brain,
        "top_10_highest": results[:10],
        "top_10_lowest": results[-10:][::-1],
        "all_results": results,
    }


def _print_report(report: dict[str, Any]) -> None:
    """Print a human-readable summary of the report to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  MOLTBOOK BEHAVIORAL SCORING — SYBILCORE")
    print(f"  {report['timestamp']}")
    print(sep)

    print(f"\nTotal agents scored: {report['total_agents_scored']}")

    print("\n--- Coefficient Statistics ---")
    stats = report["statistics"]
    print(f"  Mean:   {stats['mean_coefficient']}")
    print(f"  Median: {stats['median_coefficient']}")
    print(f"  StdDev: {stats['stdev_coefficient']}")
    print(f"  Min:    {stats['min_coefficient']}")
    print(f"  Max:    {stats['max_coefficient']}")

    print("\n--- Tier Distribution ---")
    dist = report["tier_distribution"]
    total = report["total_agents_scored"]
    for tier_label, count in dist.items():
        pct = (count / total * 100) if total > 0 else 0
        bar = "#" * (count * 30 // max(total, 1))
        print(f"  {tier_label:<10} {count:>5}  ({pct:5.1f}%)  {bar}")

    print("\n--- Brain Activity (agents where brain score > 0) ---")
    for brain_name, count in report["brain_fire_counts"].items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {brain_name:<20} {count:>5} agents  ({pct:.1f}%)")
    print(f"\n  Most active brain: {report['most_active_brain']}")

    print("\n--- Top 10 HIGHEST Scoring Agents (most suspicious) ---")
    for i, agent in enumerate(report["top_10_highest"], 1):
        brains_str = ", ".join(
            f"{k}={v}" for k, v in sorted(agent["brain_scores"].items())
        )
        print(
            f"  {i:>2}. {agent['agent_id']:<25} "
            f"coeff={agent['coefficient']:>6.1f}  tier={agent['tier']:<18}  "
            f"events={agent['event_count']:>4}  [{brains_str}]"
        )

    print("\n--- Top 10 LOWEST Scoring Agents (most trusted) ---")
    for i, agent in enumerate(report["top_10_lowest"], 1):
        brains_str = ", ".join(
            f"{k}={v}" for k, v in sorted(agent["brain_scores"].items())
        )
        print(
            f"  {i:>2}. {agent['agent_id']:<25} "
            f"coeff={agent['coefficient']:>6.1f}  tier={agent['tier']:<18}  "
            f"events={agent['event_count']:>4}  [{brains_str}]"
        )

    print(f"\n{sep}\n")


def run() -> dict[str, Any]:
    """Execute the full Moltbook scoring experiment.

    Returns:
        Full experiment report as a dict.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    logger.info("Loading Moltbook dataset...")
    adapter = MoltbookAdapter()
    logger.info(
        "Loaded %d agents, %d total events",
        adapter.agent_count,
        adapter.event_count,
    )

    results = _score_all_agents(adapter)
    report = _build_report(results)

    # Save results
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = _OUTPUT_DIR / f"moltbook_scoring_{timestamp}.json"

    # Save without all_results to keep the file lean; add a separate full file
    lean_report = {k: v for k, v in report.items() if k != "all_results"}
    with output_path.open("w") as fh:
        json.dump(lean_report, fh, indent=2)

    # Save full results as JSONL
    full_path = _OUTPUT_DIR / f"moltbook_scoring_{timestamp}_full.jsonl"
    with full_path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    logger.info("Results saved to %s", output_path)
    logger.info("Full results saved to %s", full_path)

    _print_report(report)

    return report


if __name__ == "__main__":
    run()
