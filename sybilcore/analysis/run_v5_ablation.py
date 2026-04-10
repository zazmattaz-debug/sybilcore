"""Run ablation + per-brain audit on the v5 targeted corpus.

Steps:
    1. Build the v5 corpus from sybilcore.simulation.v5_corpus_gen.
    2. Instantiate all 15 brains (default five + the ten opt-ins, including
       the four silent suspects).
    3. Score every agent and record per-brain score + nonzero flag.
    4. Run BrainAblationStudy using all 15 brains on the labelled v5 corpus.
    5. Emit experiments/v5_corpus.json (corpus dump),
       experiments/v5_corpus_ablation.json (metrics), and
       research/V5_CORPUS_RESULTS.md (summary + verdicts).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from sybilcore.analysis.ablation import (
    DETECTION_THRESHOLD,
    AblationMetrics,
    BrainAblationStudy,
)
from sybilcore.analysis.calibration import LabeledAgent
from sybilcore.brains.base import BaseBrain
from sybilcore.core.coefficient import CoefficientCalculator
from sybilcore.brains.compromise import CompromiseBrain
from sybilcore.brains.contrastive import ContrastiveEmbeddingBrain
from sybilcore.brains.deception import DeceptionBrain
from sybilcore.brains.economic import EconomicBrain
from sybilcore.brains.embedding import EmbeddingBrain
from sybilcore.brains.fidelity import FidelityBrain
from sybilcore.brains.identity import IdentityBrain
from sybilcore.brains.intent_drift import IntentDriftBrain
from sybilcore.brains.neuro import NeuroBrain
from sybilcore.brains.resource_hoarding import ResourceHoardingBrain
from sybilcore.brains.semantic import SemanticBrain
from sybilcore.brains.silence import SilenceBrain
from sybilcore.brains.social_graph import SocialGraphBrain
from sybilcore.brains.swarm_detection import SwarmDetectionBrain
from sybilcore.brains.temporal import TemporalBrain
from sybilcore.models.event import Event
from sybilcore.simulation.v5_corpus_gen import build_v5_corpus, family_counts

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RESEARCH_DIR = REPO_ROOT / "research"

SILENT_BRAINS: tuple[str, ...] = ("swarm_detection", "economic", "neuro", "deception")


def _all_brains() -> list[BaseBrain]:
    """Return all 15 brains in the same order as the v4 ablation JSON."""
    return [
        DeceptionBrain(),
        ResourceHoardingBrain(),
        SocialGraphBrain(),
        IntentDriftBrain(),
        CompromiseBrain(),
        SemanticBrain(),
        SwarmDetectionBrain(),
        TemporalBrain(),
        EconomicBrain(),
        IdentityBrain(),
        NeuroBrain(),
        EmbeddingBrain(),
        ContrastiveEmbeddingBrain(),
        FidelityBrain(),
        SilenceBrain(),
    ]


@dataclass(frozen=True)
class BrainFamilyStats:
    """Per-brain-per-family stat aggregation."""

    brain: str
    family: str
    mean_score: float
    max_score: float
    nonzero_rate: float
    n_agents: int


def _patch_score_corpus(study: BrainAblationStudy) -> None:
    """Override ``_score_corpus`` to read ``snapshot.coefficient``.

    The library's default implementation references
    ``snapshot.effective_coefficient`` which no longer exists on the
    ``CoefficientSnapshot`` model. We bind a replacement that uses the
    current attribute name so the upstream importance/dead-weight
    helpers keep working.
    """

    calculator = CoefficientCalculator()

    def _fixed_score_corpus(
        corpus: list[LabeledAgent],
        brains: list[BaseBrain],
    ) -> AblationMetrics:
        if not corpus:
            raise ValueError("Cannot score an empty corpus")
        if not brains:
            raise ValueError("Cannot score with zero brains")

        per_agent: dict[str, float] = {}
        adv_coefs: list[float] = []
        ben_coefs: list[float] = []
        true_positives = 0
        false_positives = 0
        n_adv = 0
        n_ben = 0

        for agent in corpus:
            events = list(agent.events)
            brain_scores = [brain.score(events) for brain in brains]
            snapshot = calculator.calculate(brain_scores)
            coef = snapshot.coefficient
            per_agent[agent.agent_id] = coef

            detected = coef >= DETECTION_THRESHOLD
            if agent.label == 1:
                n_adv += 1
                adv_coefs.append(coef)
                if detected:
                    true_positives += 1
            else:
                n_ben += 1
                ben_coefs.append(coef)
                if detected:
                    false_positives += 1

        all_coefs = list(per_agent.values())
        detection_rate = (true_positives / n_adv) if n_adv else 0.0
        fpr = (false_positives / n_ben) if n_ben else 0.0
        avg = sum(all_coefs) / len(all_coefs) if all_coefs else 0.0
        avg_adv = sum(adv_coefs) / len(adv_coefs) if adv_coefs else 0.0
        avg_ben = sum(ben_coefs) / len(ben_coefs) if ben_coefs else 0.0

        return AblationMetrics(
            detection_rate=detection_rate,
            false_positive_rate=fpr,
            avg_coefficient=avg,
            avg_coefficient_adversarial=avg_adv,
            avg_coefficient_benign=avg_ben,
            per_agent_coefficients=per_agent,
            n_agents=len(corpus),
            n_adversarial=n_adv,
            n_benign=n_ben,
        )

    study._score_corpus = _fixed_score_corpus  # type: ignore[method-assign]


def _audit_per_brain(
    brains: list[BaseBrain],
    family_agents: dict[str, list[LabeledAgent]],
) -> list[BrainFamilyStats]:
    """Run every brain against every family and collect stats."""
    stats: list[BrainFamilyStats] = []
    for brain in brains:
        for family, agents in family_agents.items():
            values: list[float] = []
            for agent in agents:
                score = brain.score(list(agent.events))
                values.append(score.value)
            if not values:
                continue
            stats.append(
                BrainFamilyStats(
                    brain=brain.name,
                    family=family,
                    mean_score=round(mean(values), 3),
                    max_score=round(max(values), 3),
                    nonzero_rate=round(
                        sum(1 for v in values if v > 0) / len(values), 3
                    ),
                    n_agents=len(values),
                )
            )
    return stats


def _build_labeled_corpus() -> tuple[
    list[LabeledAgent], dict[str, list[LabeledAgent]]
]:
    corpus = build_v5_corpus()
    agents: list[LabeledAgent] = []
    by_family: dict[str, list[LabeledAgent]] = {}
    for family, agent_id, events in corpus:
        label = 0 if family == "benign_v5" else 1
        la = LabeledAgent(
            agent_id=agent_id,
            events=tuple(events),
            label=label,
            source=family,
        )
        agents.append(la)
        by_family.setdefault(family, []).append(la)
    return agents, by_family


def _dump_corpus(
    agents: list[LabeledAgent],
    path: Path,
) -> None:
    """Write a JSON dump of the v5 corpus for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for a in agents:
        serializable.append(
            {
                "agent_id": a.agent_id,
                "label": a.label,
                "source": a.source,
                "events": [
                    {
                        "event_id": e.event_id,
                        "agent_id": e.agent_id,
                        "event_type": e.event_type.value,
                        "timestamp": e.timestamp.isoformat(),
                        "content": e.content,
                        "metadata": e.metadata,
                        "source": e.source,
                    }
                    for e in a.events
                ],
            }
        )
    path.write_text(json.dumps({"agents": serializable}, indent=2, default=str))


def _load_v4_ablation() -> dict[str, object] | None:
    """Load the most recent v4 clouded ablation JSON for delta comparison."""
    candidates = sorted(
        EXPERIMENTS_DIR.glob("ablation_v4_*clouded.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    try:
        return json.loads(candidates[0].read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _compute_verdicts(
    audit_stats: list[BrainFamilyStats],
) -> dict[str, dict[str, object]]:
    """Produce a per-silent-brain verdict from the targeted family audit."""
    verdicts: dict[str, dict[str, object]] = {}
    family_by_brain = {
        "swarm_detection": "swarm_v5",
        "economic": "economic_v5",
        "neuro": "neuro_v5",
        "deception": "deception_v5",
    }
    for brain, family in family_by_brain.items():
        match = next(
            (s for s in audit_stats if s.brain == brain and s.family == family),
            None,
        )
        if match is None:
            verdicts[brain] = {
                "family": family,
                "verdict": "NEEDS_MORE_WORK",
                "reason": "No audit row (brain or family missing)",
                "mean_score": 0.0,
                "nonzero_rate": 0.0,
            }
            continue

        if match.nonzero_rate >= 0.5 and match.mean_score >= 15.0:
            verdict = "KEEP"
            reason = (
                f"Fires on targeted corpus: nonzero_rate={match.nonzero_rate:.2f}, "
                f"mean_score={match.mean_score:.1f}"
            )
        elif match.nonzero_rate == 0.0 and match.mean_score == 0.0:
            verdict = "PRUNE"
            reason = (
                "Still zero on its own targeted corpus — structural dead code"
            )
        else:
            verdict = "NEEDS_MORE_WORK"
            reason = (
                f"Partial signal: nonzero_rate={match.nonzero_rate:.2f}, "
                f"mean_score={match.mean_score:.1f}"
            )
        verdicts[brain] = {
            "family": family,
            "verdict": verdict,
            "reason": reason,
            "mean_score": match.mean_score,
            "max_score": match.max_score,
            "nonzero_rate": match.nonzero_rate,
            "n_agents": match.n_agents,
        }
    return verdicts


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("sybilcore").setLevel(logging.WARNING)

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] Building v5 targeted corpus...")
    agents, by_family = _build_labeled_corpus()
    print(f"  Total agents: {len(agents)}")
    for family, members in by_family.items():
        print(f"  {family}: {len(members)} agents")

    corpus_json = EXPERIMENTS_DIR / "v5_corpus.json"
    _dump_corpus(agents, corpus_json)
    print(f"  Corpus dump: {corpus_json}")

    print("[2/5] Auditing each brain on each family (15 brains × 5 families)...")
    brains = _all_brains()
    audit_stats = _audit_per_brain(brains, by_family)

    # Print silent-brain summary early so the human sees it right away.
    print()
    print("Silent-brain signal audit:")
    print(f"  {'brain':<20}{'family':<16}{'mean':>8}{'max':>8}{'nz_rate':>10}")
    for stat in audit_stats:
        if stat.brain in SILENT_BRAINS:
            print(
                f"  {stat.brain:<20}{stat.family:<16}"
                f"{stat.mean_score:>8.2f}{stat.max_score:>8.2f}"
                f"{stat.nonzero_rate:>10.2f}"
            )
    print()

    print("[3/5] Running 15-brain ablation on v5 corpus...")
    # Monkey-patch the ablation helper because the library version expects
    # ``snapshot.effective_coefficient`` while the current model exposes
    # ``snapshot.coefficient``. Wrap ``_score_corpus`` with the fix so we
    # keep using the upstream importance / dead-weight machinery.
    study = BrainAblationStudy(brains=_all_brains())
    _patch_score_corpus(study)
    study.run_full_baseline(agents)
    study.run_drop_each(agents)

    print("[4/5] Computing verdicts + v4 delta...")
    verdicts = _compute_verdicts(audit_stats)
    v4 = _load_v4_ablation()

    ablation_payload: dict[str, object] = {
        "experiment": "v5_targeted_corpus_ablation",
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus_summary": {
            "total": len(agents),
            "positive": sum(1 for a in agents if a.label == 1),
            "negative": sum(1 for a in agents if a.label == 0),
            "families": family_counts(),
        },
        "per_brain_per_family_audit": [asdict(s) for s in audit_stats],
        "silent_brain_verdicts": verdicts,
        "ablation_study": study.to_dict(),
    }

    if v4 is not None:
        v4_baseline = v4.get("baseline", {}) if isinstance(v4, dict) else {}
        v4_importance = v4.get("importance_scores", {}) if isinstance(v4, dict) else {}
        v5_importance = study.importance_scores()
        v5_baseline = study.baseline.to_dict() if study.baseline else {}
        delta_importance = {
            brain: round(
                v5_importance.get(brain, 0.0)
                - float(v4_importance.get(brain, 0.0)),
                4,
            )
            for brain in v5_importance
        }
        ablation_payload["v4_vs_v5"] = {
            "v4_baseline": v4_baseline,
            "v5_baseline": v5_baseline,
            "importance_delta": delta_importance,
        }

    ablation_json = EXPERIMENTS_DIR / "v5_corpus_ablation.json"
    ablation_json.write_text(json.dumps(ablation_payload, indent=2, default=str))
    print(f"  Ablation JSON: {ablation_json}")

    print("[5/5] Writing summary markdown...")
    md = _format_markdown(audit_stats, verdicts, study, v4)
    md_path = RESEARCH_DIR / "V5_CORPUS_RESULTS.md"
    md_path.write_text(md)
    print(f"  Summary: {md_path}")

    print()
    print("=" * 78)
    print("Final verdicts")
    print("=" * 78)
    for brain, info in verdicts.items():
        print(
            f"  {brain:<18} -> {info['verdict']:<18}"
            f" mean={info['mean_score']:.1f}"
            f" nz={info['nonzero_rate']:.2f}"
        )
    return 0


def _format_markdown(
    audit_stats: list[BrainFamilyStats],
    verdicts: dict[str, dict[str, object]],
    study: BrainAblationStudy,
    v4: dict[str, object] | None,
) -> str:
    baseline = study.baseline
    baseline_line = (
        f"Detection rate {baseline.detection_rate:.3f}, "
        f"FPR {baseline.false_positive_rate:.3f}, "
        f"avg coef {baseline.avg_coefficient:.1f}"
        if baseline
        else "(no baseline)"
    )

    lines: list[str] = [
        "# V5 Targeted Corpus Results",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Goal",
        "",
        "Confirm or refute the four silent brains from the v4 tournament "
        "(`swarm_detection`, `economic`, `neuro`, `deception`) by running each "
        "against a corpus built specifically to fire its signals.",
        "",
        "## Corpus",
        "",
        "- 100 swarm scenarios (pre-computed swarm_score metadata, fan-in + "
        "micro-swarm motifs)",
        "- 100 economic scenarios (flow imbalance, wash cycles, rent seeking, "
        "cournot, gini spike)",
        "- 100 neuro scenarios (trust/hedge words, self-reference, multi-agent "
        "context, positive claims + errors)",
        "- 100 deception scenarios (same-query contradictions, high-confidence "
        "uncertain claims, X / not-X, fabricated citations)",
        "- 100 benign scenarios (re-used `_clean_scenario`)",
        "- Total 500 agents (400 positive / 100 negative). Seed 20260407.",
        "",
        "## 15-brain baseline on v5",
        "",
        baseline_line,
        "",
        "## Silent-brain verdicts",
        "",
        "| Brain | Family | Mean | Max | Nonzero | Verdict |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for brain in SILENT_BRAINS:
        info = verdicts.get(brain, {})
        lines.append(
            f"| {brain} | {info.get('family', '-')} | "
            f"{info.get('mean_score', 0):.1f} | "
            f"{info.get('max_score', 0):.1f} | "
            f"{info.get('nonzero_rate', 0):.2f} | "
            f"**{info.get('verdict', '?')}** |"
        )

    lines += [
        "",
        "## Detail",
        "",
    ]
    for brain in SILENT_BRAINS:
        info = verdicts.get(brain, {})
        lines.append(f"- **{brain}** — {info.get('reason', '')}")

    if v4 is not None and isinstance(v4, dict):
        v4_base = v4.get("baseline", {})
        lines += [
            "",
            "## v4 → v5 delta",
            "",
            f"- v4 baseline detection rate: "
            f"{v4_base.get('detection_rate', 'n/a')}",
            f"- v5 baseline detection rate: "
            f"{baseline.detection_rate if baseline else 'n/a'}",
            "",
            "See `experiments/v5_corpus_ablation.json` for full per-brain "
            "importance deltas.",
        ]

    lines += [
        "",
        "## Conclusion",
        "",
        "Brains labelled **KEEP** fire on their targeted corpus — the v4 "
        "silence was a corpus problem, not a code problem. Brains labelled "
        "**PRUNE** fail to fire even on inputs hand-crafted from their own "
        "source — they are structurally dead. **NEEDS_MORE_WORK** indicates "
        "partial signal (e.g. a few signals fire but not the majority) and "
        "deserves a follow-up scenario pass before final disposition.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
