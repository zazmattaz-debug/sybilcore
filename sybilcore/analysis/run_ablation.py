"""Run the SybilCore brain ablation study end-to-end.

Builds a 200-agent labelled corpus from four sources, runs the full
baseline + drop-each ablation, prints a human-readable summary table,
and persists the JSON results to ``experiments/ablation_v4_<TS>.json``.

Corpus composition (200 agents)
-------------------------------
* 50 alignment scenario agents (positive class) — generated via
  :class:`AlignmentScenarioGenerator`.
* 50 archetype agents (positive class) — generated via
  :class:`SyntheticArchetypeGenerator`.
* 50 normal Moltbook-style benign agents (negative class) — generated
  via :class:`SyntheticSwarm` clean event templates.
* 50 adversarial swarm rogues (positive class) — generated via
  :class:`SyntheticSwarm.inject_rogue` covering all five rogue types.

Usage::

    cd rooms/engineering/sybilcore
    python3 -m sybilcore.analysis.run_ablation
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from sybilcore.analysis.ablation import BrainAblationStudy, LabeledAgent
from sybilcore.core.config import TIER_BOUNDARIES, TierName
from sybilcore.models.event import Event
from sybilcore.simulation.alignment_scenarios import (
    ALL_ALIGNMENT_SCENARIOS,
    AlignmentScenarioGenerator,
)
from sybilcore.simulation.archetypes import (
    ALL_ARCHETYPES,
    SyntheticArchetypeGenerator,
)
from sybilcore.analysis.corpus import _clean_scenario
from sybilcore.simulation.synthetic import RogueType, SyntheticSwarm

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# Each corpus source contributes 50 agents -> 200 total.
TARGET_PER_BUCKET: int = 50


# ──────────────────────────────────────────────────────────────────
# Corpus builders
# ──────────────────────────────────────────────────────────────────


def _alignment_corpus(target: int = TARGET_PER_BUCKET) -> list[LabeledAgent]:
    """Generate ``target`` alignment scenario positives.

    Each scenario generator produces a single agent. We loop through
    the four scenarios with varying ``num_rounds`` until we hit the
    target count, giving slight variance to each agent.
    """
    gen = AlignmentScenarioGenerator()
    scenario_methods = {
        "alignment_faker": gen.alignment_faker,
        "self_preserver": gen.self_preserver,
        "safety_saboteur": gen.safety_saboteur,
        "emergent_drifter": gen.emergent_drifter,
    }
    scenarios = list(ALL_ALIGNMENT_SCENARIOS)

    agents: list[LabeledAgent] = []
    idx = 0
    while len(agents) < target:
        scenario_name = scenarios[idx % len(scenarios)]
        method = scenario_methods[scenario_name]
        rounds = 18 + (idx % 8)
        result = method(
            agent_id=f"align-{scenario_name}-{idx:03d}",
            num_rounds=rounds,
        )
        for agent_id, events in result.agents.items():
            agents.append(
                LabeledAgent(
                    agent_id=agent_id,
                    events=tuple(events),
                    label=1,
                    source=f"alignment:{scenario_name}",
                )
            )
            if len(agents) >= target:
                break
        idx += 1
    return agents[:target]


def _archetype_corpus(target: int = TARGET_PER_BUCKET) -> list[LabeledAgent]:
    """Generate ``target`` archetype positives across the six attacks."""
    gen = SyntheticArchetypeGenerator()
    agents: list[LabeledAgent] = []

    for archetype in ALL_ARCHETYPES:
        events = gen.generate(archetype, num_rounds=4)
        per_agent: dict[str, list[Event]] = {}
        for event in events:
            per_agent.setdefault(event.agent_id, []).append(event)

        for original_id, agent_events in per_agent.items():
            tagged_id = f"arch-{archetype}-{original_id}"
            agents.append(
                LabeledAgent(
                    agent_id=tagged_id,
                    events=tuple(
                        Event(
                            event_id=e.event_id,
                            agent_id=tagged_id,
                            event_type=e.event_type,
                            timestamp=e.timestamp,
                            content=e.content,
                            metadata=e.metadata,
                            source=e.source,
                        )
                        for e in agent_events
                    ),
                    label=1,
                    source=f"archetype:{archetype}",
                )
            )
            if len(agents) >= target:
                return agents[:target]
    return agents[:target]


def _benign_corpus(target: int = TARGET_PER_BUCKET) -> list[LabeledAgent]:
    """Generate ``target`` diverse benign agents.

    Mixes two sources of benign behaviour:

    * ``SyntheticSwarm`` clean-template events (half the bucket) — a
      Moltbook-style baseline of routine tool calls and messages.
    * :func:`sybilcore.analysis.corpus._clean_scenario` (other half) —
      richer instruction-response traces over diverse topics
      (summarize, translate, classify, schedule, answer).

    The mix prevents the benign population from collapsing into a
    single behavioural cluster, which is what made the contrastive
    brain fire on every benign agent in the first run.
    """
    half = target // 2
    remainder = target - half

    swarm = SyntheticSwarm(seed=20260407)
    swarm.spawn(half)
    swarm_events = swarm.generate_round(events_per_agent=8)
    by_agent: dict[str, list[Event]] = {}
    for ev in swarm_events:
        by_agent.setdefault(ev.agent_id, []).append(ev)

    agents: list[LabeledAgent] = []
    for agent_id, agent_events in list(by_agent.items())[:half]:
        tagged_id = f"benign-swarm-{agent_id}"
        agents.append(
            LabeledAgent(
                agent_id=tagged_id,
                events=tuple(
                    Event(
                        event_id=e.event_id,
                        agent_id=tagged_id,
                        event_type=e.event_type,
                        timestamp=e.timestamp,
                        content=e.content,
                        metadata=e.metadata,
                        source=e.source,
                    )
                    for e in agent_events
                ),
                label=0,
                source="moltbook:swarm_clean",
            )
        )

    for i in range(remainder):
        tagged_id = f"benign-clean-{i:03d}"
        _, events = _clean_scenario(i)
        agents.append(
            LabeledAgent(
                agent_id=tagged_id,
                events=tuple(
                    Event(
                        event_id=e.event_id,
                        agent_id=tagged_id,
                        event_type=e.event_type,
                        timestamp=e.timestamp,
                        content=e.content,
                        metadata=e.metadata,
                        source=e.source,
                    )
                    for e in events
                ),
                label=0,
                source="moltbook:scenario_clean",
            )
        )
    return agents[:target]


def _rogue_corpus(target: int = TARGET_PER_BUCKET) -> list[LabeledAgent]:
    """Generate ``target`` adversarial rogues across all RogueType variants."""
    rogue_types = list(RogueType)
    swarm = SyntheticSwarm(seed=21260407)
    specs = swarm.spawn(target)

    rogue_assignments: dict[str, RogueType] = {}
    for i, spec in enumerate(specs):
        rt = rogue_types[i % len(rogue_types)]
        swarm.inject_rogue(spec.agent_id, rt)
        rogue_assignments[spec.agent_id] = rt

    # Run several rounds so brains see persistent suspicious patterns.
    all_events: list[Event] = []
    for _ in range(3):
        all_events.extend(swarm.generate_round(events_per_agent=8))

    by_agent: dict[str, list[Event]] = {}
    for ev in all_events:
        by_agent.setdefault(ev.agent_id, []).append(ev)

    agents: list[LabeledAgent] = []
    for agent_id, agent_events in list(by_agent.items())[:target]:
        rogue_type = rogue_assignments.get(agent_id, RogueType.PROMPT_INJECTED)
        tagged_id = f"rogue-{rogue_type.value}-{agent_id}"
        agents.append(
            LabeledAgent(
                agent_id=tagged_id,
                events=tuple(
                    Event(
                        event_id=e.event_id,
                        agent_id=tagged_id,
                        event_type=e.event_type,
                        timestamp=e.timestamp,
                        content=e.content,
                        metadata=e.metadata,
                        source=e.source,
                    )
                    for e in agent_events
                ),
                label=1,
                source=f"tournament:{rogue_type.value}",
            )
        )
    return agents[:target]


def build_test_corpus() -> list[LabeledAgent]:
    """Assemble the full 200-agent labelled corpus."""
    corpus: list[LabeledAgent] = []
    corpus.extend(_alignment_corpus())
    corpus.extend(_archetype_corpus())
    corpus.extend(_benign_corpus())
    corpus.extend(_rogue_corpus())

    n_pos = sum(1 for a in corpus if a.label == 1)
    n_neg = sum(1 for a in corpus if a.label == 0)
    logger.info(
        "Corpus built: %d agents (%d positive, %d negative)",
        len(corpus),
        n_pos,
        n_neg,
    )
    return corpus


# ──────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────


def _print_summary(study: BrainAblationStudy) -> None:
    baseline = study.baseline
    if baseline is None:
        return

    importance = study.importance_scores()
    sorted_brains = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)

    print()
    print("=" * 78)
    print("SybilCore Brain Ablation Study (v4)")
    print("=" * 78)
    print(f"Baseline detection rate: {baseline.detection_rate:.3f}")
    print(f"Baseline false positives: {baseline.false_positive_rate:.3f}")
    print(f"Baseline avg coefficient: {baseline.avg_coefficient:.2f}")
    print(
        f"  adversarial avg: {baseline.avg_coefficient_adversarial:.2f}"
        f"  |  benign avg: {baseline.avg_coefficient_benign:.2f}"
    )
    print(
        f"Corpus: {baseline.n_agents} agents "
        f"({baseline.n_adversarial} positive / {baseline.n_benign} negative)"
    )
    print()
    print(
        f"{'brain':<22}{'drop_DR':>10}{'drop_FPR':>11}"
        f"{'Δ_DR':>10}{'verdict':>16}"
    )
    print("-" * 78)

    for name, importance_score in sorted_brains:
        drop = study.drop_results[name]
        if importance_score > 0.20:
            verdict = "ESSENTIAL"
        elif abs(importance_score) < 0.05:
            verdict = "dead weight"
        elif importance_score < 0:
            verdict = "harmful?"
        else:
            verdict = "useful"
        print(
            f"{name:<22}{drop.detection_rate:>10.3f}"
            f"{drop.false_positive_rate:>11.3f}"
            f"{importance_score:>+10.3f}{verdict:>16}"
        )

    dead = study.identify_dead_weight()
    essential = study.identify_essential()
    gray = study.gray_zone()
    print()
    print(f"Dead-weight brains ({len(dead)}):  {', '.join(dead) or '(none)'}")
    print(f"Essential brains   ({len(essential)}):  {', '.join(essential) or '(none)'}")
    print(f"Gray-zone brains   ({len(gray)}):  {', '.join(gray) or '(none)'}")
    print("=" * 78)


def _save_results(
    study: BrainAblationStudy,
    corpus: list[LabeledAgent],
    label: str | None = None,
) -> Path:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}" if label else ""
    out_path = EXPERIMENTS_DIR / f"ablation_v4_{timestamp}{suffix}.json"

    payload: dict[str, object] = {
        "experiment": "brain_ablation_v4",
        "corpus_summary": {
            "total": len(corpus),
            "positive": sum(1 for a in corpus if a.label == 1),
            "negative": sum(1 for a in corpus if a.label == 0),
            "sources": _source_counts(corpus),
        },
        **study.to_dict(),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    return out_path


def _source_counts(corpus: list[LabeledAgent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for agent in corpus:
        counts[agent.source] = counts.get(agent.source, 0) + 1
    return counts


# ──────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────


def _run_single_study(
    corpus: list[LabeledAgent],
    threshold: float,
    label: str,
) -> tuple[BrainAblationStudy, Path]:
    """Run one ablation at a fixed detection threshold and persist it."""
    print(f"\n>>> Ablation at threshold={threshold:.0f} ({label})")
    study = BrainAblationStudy(detection_threshold=threshold)
    study.run_full_baseline(corpus)
    study.run_drop_each(corpus)
    _print_summary(study)
    out_path = _save_results(study, corpus, label=label)
    print(f"Results written to: {out_path}")
    return study, out_path


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("sybilcore").setLevel(logging.WARNING)

    print("[1/3] Building 200-agent labelled corpus...")
    corpus = build_test_corpus()

    print("[2/3] Running ablations at two detection thresholds...")
    clouded = TIER_BOUNDARIES[TierName.CLOUDED][0]  # 100.0
    flagged = TIER_BOUNDARIES[TierName.FLAGGED][0]  # 200.0
    _run_single_study(corpus, clouded, label="clouded")
    _run_single_study(corpus, flagged, label="flagged")

    print("[3/3] Done — see printed verdict tables above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
