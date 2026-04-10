"""Moltbook Observatory -- LIVE mode runner.

Loads real Moltbook agent data from the cached JSONL file, creates an
Observatory via from_adapter(), runs the full analysis pipeline, and
produces a comprehensive report with:
  - Top 20 most suspicious agents
  - Coordination clusters
  - Bot vs human breakdown
  - Karma-trust correlation
  - Posting regularity and topic entropy stats
  - Tier distribution
  - Full per-agent results saved to experiments/

Usage:
    python3 -m sybilcore.simulation.run_observatory_live
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sybilcore.integrations.moltbook import MoltbookAdapter
from sybilcore.simulation.moltbook_observatory import Observatory, ObservatoryReport

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).parent.parent.parent / "experiments"


def _print_report(report: ObservatoryReport) -> None:
    """Print a human-readable summary of the live observatory report."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  MOLTBOOK OBSERVATORY -- LIVE MODE -- SYBILCORE")
    print(f"  {report.timestamp}")
    print(sep)

    print(f"\n  Total agents loaded: {report.total_agents}")
    print(f"  Agents scored:       {report.agents_scored}")

    # Tier distribution
    print(f"\n{'─' * 40}")
    print("  TIER DISTRIBUTION")
    print(f"{'─' * 40}")
    total = report.agents_scored or 1
    for tier_label, count in report.tier_distribution.items():
        pct = count / total * 100
        bar = "#" * (count * 40 // max(total, 1))
        print(f"  {tier_label:<10} {count:>5}  ({pct:5.1f}%)  {bar}")

    # Bot classification
    print(f"\n{'─' * 40}")
    print("  BOT CLASSIFICATION")
    print(f"{'─' * 40}")
    for cls_label, count in sorted(report.bot_classification.items()):
        pct = count / total * 100
        print(f"  {cls_label:<15} {count:>5}  ({pct:.1f}%)")

    # Karma-trust correlation
    if report.karma_trust_correlation is not None:
        print(f"\n{'─' * 40}")
        print("  KARMA-TRUST CORRELATION")
        print(f"{'─' * 40}")
        print(f"  Pearson r:    {report.karma_trust_correlation:+.4f}")
        if report.karma_trust_spearman is not None:
            print(f"  Spearman rho: {report.karma_trust_spearman:+.4f}")

    # Posting regularity
    if report.posting_regularity_stats:
        print(f"\n{'─' * 40}")
        print("  POSTING REGULARITY (CoV)")
        print(f"{'─' * 40}")
        for k, v in report.posting_regularity_stats.items():
            print(f"  {k}: {v:.4f}")

    # Topic entropy
    if report.topic_entropy_stats:
        print(f"\n{'─' * 40}")
        print("  TOPIC ENTROPY")
        print(f"{'─' * 40}")
        for k, v in report.topic_entropy_stats.items():
            print(f"  {k}: {v:.4f}")

    # Coordination clusters
    if report.coordination_clusters:
        print(f"\n{'─' * 40}")
        print(f"  COORDINATION CLUSTERS ({len(report.coordination_clusters)})")
        print(f"{'─' * 40}")
        for i, cluster in enumerate(report.coordination_clusters[:10], 1):
            agents_str = ", ".join(cluster["agents"][:5])
            extra = f" +{len(cluster['agents']) - 5} more" if len(cluster["agents"]) > 5 else ""
            print(f"  {i}. [{agents_str}{extra}]  hash={cluster['content_hash']}")
    else:
        print(f"\n  Coordination clusters: 0 (no coordinated campaigns detected)")

    # Instruction flags
    print(f"\n  Total instruction flags: {report.instruction_flag_count}")

    # Top 20 most suspicious
    top_n = min(20, len(report.all_analyses))
    print(f"\n{'─' * 40}")
    print(f"  TOP {top_n} MOST SUSPICIOUS AGENTS")
    print(f"{'─' * 40}")
    for i, agent in enumerate(report.all_analyses[:top_n], 1):
        print(
            f"  {i:>2}. {agent['agent_id']:<30} "
            f"suspicion={agent['suspicion_score']:>6.1f}  "
            f"coeff={agent['coefficient']:>6.1f}  "
            f"tier={agent['tier']:<18}  "
            f"class={agent['classification']:<12}  "
            f"events={agent['event_count']:>4}  "
            f"karma={agent['karma']:>5}"
        )

    print(f"\n{sep}\n")


def _report_to_dict(report: ObservatoryReport) -> dict[str, Any]:
    """Convert report to JSON-serializable dict."""
    return {
        "experiment": "moltbook_observatory_live",
        "timestamp": report.timestamp,
        "mode": "live",
        "total_agents": report.total_agents,
        "agents_scored": report.agents_scored,
        "top_20_suspicious": report.all_analyses[:20],
        "karma_trust_correlation": report.karma_trust_correlation,
        "karma_trust_spearman": report.karma_trust_spearman,
        "bot_classification": report.bot_classification,
        "coordination_clusters": report.coordination_clusters,
        "instruction_flag_count": report.instruction_flag_count,
        "posting_regularity_stats": report.posting_regularity_stats,
        "topic_entropy_stats": report.topic_entropy_stats,
        "tier_distribution": report.tier_distribution,
    }


def run() -> dict[str, Any]:
    """Execute the live observatory experiment.

    Returns:
        Experiment report as a dict.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    logger.info("Loading Moltbook dataset (LIVE mode)...")
    adapter = MoltbookAdapter()
    logger.info(
        "Loaded %d agents, %d total events from %s",
        adapter.agent_count,
        adapter.event_count,
        adapter.data_path,
    )

    logger.info("Creating Observatory from adapter...")
    observatory = Observatory.from_adapter(adapter)
    logger.info("Observatory loaded %d agents", observatory.agent_count)

    logger.info("Running full observatory analysis...")
    report = observatory.run_full_analysis()

    # Save lean report (no all_analyses to keep file small)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = _OUTPUT_DIR / f"observatory_live_{timestamp}.json"

    report_dict = _report_to_dict(report)
    with output_path.open("w") as fh:
        json.dump(report_dict, fh, indent=2)

    # Save full per-agent results as JSONL
    full_path = _OUTPUT_DIR / f"observatory_live_{timestamp}_full.jsonl"
    with full_path.open("w") as fh:
        for analysis in report.all_analyses:
            fh.write(json.dumps(analysis) + "\n")

    logger.info("Lean report saved to %s", output_path)
    logger.info("Full results saved to %s (%d agents)", full_path, len(report.all_analyses))

    _print_report(report)

    return report_dict


if __name__ == "__main__":
    run()
