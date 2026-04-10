"""Swarm detection experiment — generates a mixed population and tests SwarmBrain.

Creates 200 agents (180 normal + 20 in a coordinated micro-swarm),
runs SwarmAnalyzer, measures detection rate and false positive rate,
and saves results to experiments/.

Usage:
    python -m sybilcore.simulation.run_swarm_detection
    python -m sybilcore.simulation.run_swarm_detection --population 500 --swarm-size 30
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sybilcore.brains.swarm_detection import (
    SwarmAnalyzer,
    SwarmDetectionBrain,
)
from sybilcore.models.event import Event, EventType

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "experiments"


def _generate_normal_agent_events(
    agent_id: str,
    base_time: datetime,
    num_events: int = 10,
    agent_index: int = 0,
) -> list[Event]:
    """Generate benign events for a normal agent.

    Each normal agent starts at a unique random offset to avoid
    accidental synchronization across the population.
    """
    events: list[Event] = []
    event_types = [
        EventType.TOOL_CALL,
        EventType.OUTPUT_GENERATED,
        EventType.RESOURCE_ACCESS,
        EventType.STATE_CHANGE,
    ]

    # Each agent starts at a unique offset to prevent sync patterns.
    agent_offset = agent_index * 17.3 + random.uniform(0, 50)

    for i in range(num_events):
        ts = base_time + timedelta(seconds=agent_offset + i * random.uniform(8, 60))
        evt_type = random.choice(event_types)
        metadata: dict = {}

        events.append(
            Event(
                event_id=str(uuid.uuid4()),
                agent_id=agent_id,
                event_type=evt_type,
                timestamp=ts,
                content=f"Normal action {i}",
                metadata=metadata,
            )
        )

    # Add 1-2 messages to unique random targets (not the swarm target).
    for i in range(random.randint(1, 2)):
        ts = base_time + timedelta(seconds=agent_offset + random.uniform(20, 200))
        events.append(
            Event(
                event_id=str(uuid.uuid4()),
                agent_id=agent_id,
                event_type=EventType.MESSAGE_SENT,
                timestamp=ts,
                content=f"Normal message {i}",
                metadata={"recipient_agent_id": f"random-peer-{random.randint(1, 500)}"},
            )
        )

    return events


def _generate_swarm_agent_events(
    agent_id: str,
    swarm_target: str,
    base_time: datetime,
    swarm_beat_times: list[float],
    agent_index: int,
) -> list[Event]:
    """Generate coordinated swarm events — synchronized timing + shared target."""
    events: list[Event] = []

    # Synchronized tool calls at coordinated beat times.
    for j, beat_offset in enumerate(swarm_beat_times):
        # Small jitter to simulate "coordinated but not identical".
        jitter = agent_index * 0.15 + random.uniform(-0.3, 0.3)
        ts = base_time + timedelta(seconds=beat_offset + jitter)
        events.append(
            Event(
                event_id=str(uuid.uuid4()),
                agent_id=agent_id,
                event_type=EventType.TOOL_CALL,
                timestamp=ts,
                content=f"Coordinated action {j}",
            )
        )

    # Message to shared target (fan-in + micro-swarm).
    msg_ts = base_time + timedelta(seconds=swarm_beat_times[-1] + 1.0 + agent_index * 0.2)
    events.append(
        Event(
            event_id=str(uuid.uuid4()),
            agent_id=agent_id,
            event_type=EventType.MESSAGE_SENT,
            timestamp=msg_ts,
            content=f"Fragment {agent_index} of attack payload",
            metadata={"recipient_agent_id": swarm_target},
        )
    )

    return events


def run_experiment(
    population_size: int = 200,
    swarm_size: int = 20,
    normal_events_per_agent: int = 8,
    swarm_beats: int = 6,
) -> dict:
    """Run the swarm detection experiment.

    Args:
        population_size: Total number of agents.
        swarm_size: Number of agents in the coordinated swarm.
        normal_events_per_agent: Events per normal agent.
        swarm_beats: Number of synchronized action beats.

    Returns:
        Experiment results dict.
    """
    normal_count = population_size - swarm_size
    # Push base time far enough back to accommodate all agents' spread-out timing.
    # With 200 agents × 17.3s offset + up to 60s intervals × 10 events = ~4000s needed.
    base_time = datetime.now(UTC) - timedelta(seconds=5000)
    swarm_target = "high-value-target"

    # Generate synchronized beat times for swarm.
    swarm_beat_times = [i * 10.0 for i in range(swarm_beats)]

    agent_events: dict[str, list[Event]] = {}

    # Normal agents.
    normal_ids: list[str] = []
    for i in range(normal_count):
        agent_id = f"normal-{i:04d}"
        normal_ids.append(agent_id)
        agent_events[agent_id] = _generate_normal_agent_events(
            agent_id, base_time, normal_events_per_agent, agent_index=i
        )

    # Swarm agents.
    swarm_ids: list[str] = []
    for i in range(swarm_size):
        agent_id = f"swarm-{i:04d}"
        swarm_ids.append(agent_id)
        agent_events[agent_id] = _generate_swarm_agent_events(
            agent_id, swarm_target, base_time, swarm_beat_times, i
        )

    # Target agent (passive).
    agent_events[swarm_target] = [
        Event(
            event_id=str(uuid.uuid4()),
            agent_id=swarm_target,
            event_type=EventType.MESSAGE_RECEIVED,
            timestamp=base_time + timedelta(seconds=100),
            content="Receiving messages",
        )
    ]

    # Run analysis.
    analyzer = SwarmAnalyzer()
    analysis = analyzer.analyze(agent_events)

    # Compute detection metrics.
    brain = SwarmDetectionBrain()
    detection_threshold = 5.0

    # True positives: swarm agents correctly flagged.
    true_positives = sum(
        1
        for aid in swarm_ids
        if analysis.agent_scores.get(aid)
        and analysis.agent_scores[aid].value >= detection_threshold
    )

    # False negatives: swarm agents missed.
    false_negatives = swarm_size - true_positives

    # False positives: normal agents incorrectly flagged.
    false_positives = sum(
        1
        for aid in normal_ids
        if analysis.agent_scores.get(aid)
        and analysis.agent_scores[aid].value >= detection_threshold
    )

    # True negatives: normal agents correctly not flagged.
    true_negatives = normal_count - false_positives

    detection_rate = true_positives / swarm_size if swarm_size > 0 else 0.0
    false_positive_rate = false_positives / normal_count if normal_count > 0 else 0.0
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    # Collect per-agent scores for analysis.
    swarm_scores = [
        {
            "agent_id": aid,
            "score": analysis.agent_scores.get(aid, type("", (), {"value": 0.0, "indicators": []})).value,
            "indicators": analysis.agent_scores.get(aid, type("", (), {"value": 0.0, "indicators": []})).indicators,
        }
        for aid in swarm_ids
    ]

    normal_score_values = [
        analysis.agent_scores.get(aid, type("", (), {"value": 0.0})).value
        for aid in normal_ids
    ]

    results = {
        "experiment": "swarm_detection",
        "timestamp": datetime.now(UTC).isoformat(),
        "parameters": {
            "population_size": population_size,
            "swarm_size": swarm_size,
            "normal_count": normal_count,
            "normal_events_per_agent": normal_events_per_agent,
            "swarm_beats": swarm_beats,
        },
        "metrics": {
            "detection_rate": round(detection_rate, 4),
            "false_positive_rate": round(false_positive_rate, 4),
            "precision": round(precision, 4),
            "true_positives": true_positives,
            "false_negatives": false_negatives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
        },
        "clusters": [
            {
                "type": c.cluster_type,
                "agent_count": len(c.agent_ids),
                "collective_score": round(c.collective_score, 2),
            }
            for c in analysis.clusters
        ],
        "alerts": analysis.alerts,
        "swarm_agent_scores": swarm_scores,
        "normal_score_stats": {
            "mean": round(sum(normal_score_values) / max(len(normal_score_values), 1), 2),
            "max": round(max(normal_score_values) if normal_score_values else 0.0, 2),
            "flagged_count": false_positives,
        },
    }

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run swarm detection experiment")
    parser.add_argument("--population", type=int, default=200, help="Total agents (default: 200)")
    parser.add_argument("--swarm-size", type=int, default=20, help="Swarm agents (default: 20)")
    parser.add_argument("--beats", type=int, default=6, help="Synchronized beat count (default: 6)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info(
        "Running swarm detection: %d agents (%d swarm, %d normal)",
        args.population,
        args.swarm_size,
        args.population - args.swarm_size,
    )

    results = run_experiment(
        population_size=args.population,
        swarm_size=args.swarm_size,
        swarm_beats=args.beats,
    )

    # Save results.
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = EXPERIMENTS_DIR / f"swarm_detection_{timestamp}.json"
    output_path.write_text(json.dumps(results, indent=2, default=str))

    # Print summary.
    m = results["metrics"]
    print(f"\n{'='*60}")
    print(f"SWARM DETECTION EXPERIMENT RESULTS")
    print(f"{'='*60}")
    print(f"Population: {args.population} agents ({args.swarm_size} swarm + {args.population - args.swarm_size} normal)")
    print(f"Detection rate:      {m['detection_rate']:.1%} ({m['true_positives']}/{args.swarm_size})")
    print(f"False positive rate: {m['false_positive_rate']:.1%} ({m['false_positives']}/{args.population - args.swarm_size})")
    print(f"Precision:           {m['precision']:.1%}")
    print(f"Clusters detected:   {len(results['clusters'])}")
    print(f"Alerts:              {len(results['alerts'])}")
    print(f"Results saved:       {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
