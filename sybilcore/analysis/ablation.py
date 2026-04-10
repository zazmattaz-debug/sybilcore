"""Brain ablation study — measure each brain's marginal contribution.

The 15-brain SybilCore pipeline aggregates many specialized signals.
Ablation answers a single question: which brains earn their keep?

Method
------
1. Build a labelled corpus of agents tagged 1 (adversarial) or 0
   (benign).
2. Run the full 15-brain pipeline; record detection rate as the
   baseline.
3. For each brain, drop it (run the other 14) and re-score every
   agent. Record the new detection rate.
4. Brains whose removal *barely* moves detection are dead weight.
   Brains whose removal *crashes* detection are essential.

A brain is considered to have "detected" an agent when the agent's
``effective_coefficient`` lands at or above ``DETECTION_THRESHOLD``.

This module reuses :class:`LabeledAgent` from
:mod:`sybilcore.analysis.calibration` so the same labelled corpus can
power both calibration and ablation pipelines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sybilcore.analysis.calibration import LabeledAgent
from sybilcore.brains import get_default_brains
from sybilcore.brains.base import BaseBrain
from sybilcore.core.coefficient import CoefficientCalculator

logger = logging.getLogger(__name__)

# Detection threshold: anything above CLEAR tier counts as "detected".
# CLEAR -> CLOUDED boundary is 100.0 in TIER_BOUNDARIES.
DETECTION_THRESHOLD: float = 100.0

__all__ = [
    "DETECTION_THRESHOLD",
    "AblationMetrics",
    "BrainAblationStudy",
    "LabeledAgent",
]


@dataclass(frozen=True)
class AblationMetrics:
    """Aggregate metrics from one ablation run.

    Attributes:
        detection_rate: Fraction of adversarial agents correctly flagged
            (true positive rate).
        false_positive_rate: Fraction of benign agents incorrectly
            flagged.
        avg_coefficient: Mean ``effective_coefficient`` across the
            entire labelled corpus.
        avg_coefficient_adversarial: Mean coefficient on adversarial
            agents only.
        avg_coefficient_benign: Mean coefficient on benign agents only.
        per_agent_coefficients: ``{agent_id: effective_coefficient}``
            for forensic inspection.
        n_agents: Total agents scored.
        n_adversarial: Number of adversarial agents in the corpus.
        n_benign: Number of benign agents in the corpus.
    """

    detection_rate: float
    false_positive_rate: float
    avg_coefficient: float
    avg_coefficient_adversarial: float
    avg_coefficient_benign: float
    per_agent_coefficients: dict[str, float] = field(default_factory=dict)
    n_agents: int = 0
    n_adversarial: int = 0
    n_benign: int = 0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable summary (no per-agent payload)."""
        return {
            "detection_rate": round(self.detection_rate, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "avg_coefficient": round(self.avg_coefficient, 2),
            "avg_coefficient_adversarial": round(self.avg_coefficient_adversarial, 2),
            "avg_coefficient_benign": round(self.avg_coefficient_benign, 2),
            "n_agents": self.n_agents,
            "n_adversarial": self.n_adversarial,
            "n_benign": self.n_benign,
        }


class BrainAblationStudy:
    """Drop-one brain ablation framework.

    Usage::

        study = BrainAblationStudy()
        baseline = study.run_full_baseline(corpus)
        per_brain = study.run_drop_each(corpus)
        dead = study.identify_dead_weight()
        essential = study.identify_essential()
    """

    def __init__(
        self,
        brains: list[BaseBrain] | None = None,
        detection_threshold: float = DETECTION_THRESHOLD,
    ) -> None:
        """Initialize the study.

        Args:
            brains: Brain instances to use. Defaults to
                :func:`get_default_brains`.
            detection_threshold: Coefficient threshold above which an
                agent counts as detected.
        """
        self._brains: list[BaseBrain] = brains if brains is not None else get_default_brains()
        if not self._brains:
            msg = "BrainAblationStudy requires at least one brain"
            raise ValueError(msg)
        self._calculator = CoefficientCalculator()
        self._detection_threshold = detection_threshold
        self._baseline: AblationMetrics | None = None
        self._drop_results: dict[str, AblationMetrics] = {}

    @property
    def brain_names(self) -> list[str]:
        """Names of every brain registered with this study."""
        return [b.name for b in self._brains]

    @property
    def detection_threshold(self) -> float:
        """Coefficient cutoff used to declare an agent detected."""
        return self._detection_threshold

    @property
    def baseline(self) -> AblationMetrics | None:
        """Last computed baseline (or ``None`` if not run yet)."""
        return self._baseline

    @property
    def drop_results(self) -> dict[str, AblationMetrics]:
        """Map ``brain_name -> metrics`` from the last drop-each run."""
        return dict(self._drop_results)

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def _score_corpus(
        self,
        corpus: list[LabeledAgent],
        brains: list[BaseBrain],
    ) -> AblationMetrics:
        """Run the supplied brain set against every agent and aggregate.

        Args:
            corpus: Labelled agents to evaluate.
            brains: Brain subset to use for this run.

        Returns:
            AblationMetrics summary.

        Raises:
            ValueError: If corpus is empty or brains list is empty.
        """
        if not corpus:
            msg = "Cannot score an empty corpus"
            raise ValueError(msg)
        if not brains:
            msg = "Cannot score with zero brains"
            raise ValueError(msg)

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
            snapshot = self._calculator.calculate(brain_scores)
            coef = snapshot.effective_coefficient
            per_agent[agent.agent_id] = coef

            detected = coef >= self._detection_threshold
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
        avg_coef = sum(all_coefs) / len(all_coefs) if all_coefs else 0.0
        avg_adv = sum(adv_coefs) / len(adv_coefs) if adv_coefs else 0.0
        avg_ben = sum(ben_coefs) / len(ben_coefs) if ben_coefs else 0.0

        return AblationMetrics(
            detection_rate=detection_rate,
            false_positive_rate=fpr,
            avg_coefficient=avg_coef,
            avg_coefficient_adversarial=avg_adv,
            avg_coefficient_benign=avg_ben,
            per_agent_coefficients=per_agent,
            n_agents=len(corpus),
            n_adversarial=n_adv,
            n_benign=n_ben,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full_baseline(self, test_corpus: list[LabeledAgent]) -> AblationMetrics:
        """Run all brains against the corpus.

        Args:
            test_corpus: Labelled agents.

        Returns:
            Baseline metrics. Cached on the instance.
        """
        logger.info(
            "Running full baseline with %d brains across %d agents",
            len(self._brains),
            len(test_corpus),
        )
        metrics = self._score_corpus(test_corpus, self._brains)
        self._baseline = metrics
        return metrics

    def run_drop_one(
        self,
        brain_name: str,
        test_corpus: list[LabeledAgent],
    ) -> AblationMetrics:
        """Run all brains except ``brain_name``.

        Args:
            brain_name: Brain to remove (must match an existing
                ``brain.name``).
            test_corpus: Labelled agents.

        Returns:
            Metrics for the reduced brain set.

        Raises:
            ValueError: If ``brain_name`` is not registered.
        """
        if brain_name not in self.brain_names:
            msg = f"Unknown brain '{brain_name}'. Available: {self.brain_names}"
            raise ValueError(msg)
        reduced = [b for b in self._brains if b.name != brain_name]
        logger.info(
            "Running drop-one ablation: dropping '%s' (%d brains remain)",
            brain_name,
            len(reduced),
        )
        return self._score_corpus(test_corpus, reduced)

    def run_drop_each(
        self,
        test_corpus: list[LabeledAgent],
    ) -> dict[str, AblationMetrics]:
        """Run drop-one for every brain in turn.

        Args:
            test_corpus: Labelled agents.

        Returns:
            Map ``brain_name -> AblationMetrics``.
        """
        results: dict[str, AblationMetrics] = {}
        for name in self.brain_names:
            results[name] = self.run_drop_one(name, test_corpus)
        self._drop_results = results
        return results

    # ------------------------------------------------------------------
    # Importance scoring
    # ------------------------------------------------------------------

    def importance_scores(self) -> dict[str, float]:
        """Compute importance per brain (baseline DR minus drop-one DR).

        Positive values mean removing the brain *hurt* detection.
        Negative values mean removing it *improved* detection.

        Returns:
            ``{brain_name: importance}`` (importance is the change in
            detection rate caused by removing the brain).

        Raises:
            RuntimeError: If baseline or drop-each results are missing.
        """
        if self._baseline is None:
            msg = "Run run_full_baseline() before importance_scores()"
            raise RuntimeError(msg)
        if not self._drop_results:
            msg = "Run run_drop_each() before importance_scores()"
            raise RuntimeError(msg)
        baseline_dr = self._baseline.detection_rate
        return {
            name: baseline_dr - metrics.detection_rate
            for name, metrics in self._drop_results.items()
        }

    def identify_dead_weight(self, threshold: float = 0.05) -> list[str]:
        """Return brains whose removal barely changes detection rate.

        A brain counts as dead weight when removing it shifts detection
        rate by *less than* ``threshold`` in absolute value (the brain
        contributes neither help nor harm).

        Args:
            threshold: Maximum absolute detection-rate delta tolerated.

        Returns:
            Brain names sorted by absolute impact ascending (most
            useless first).
        """
        scores = self.importance_scores()
        candidates = [
            (name, abs(impact))
            for name, impact in scores.items()
            if abs(impact) < threshold
        ]
        candidates.sort(key=lambda kv: kv[1])
        return [name for name, _ in candidates]

    def identify_essential(self, threshold: float = 0.20) -> list[str]:
        """Return brains whose removal *crashes* detection rate.

        Essential = removing the brain reduces detection rate by *more
        than* ``threshold``.

        Args:
            threshold: Minimum detection-rate drop required to qualify.

        Returns:
            Brain names sorted by impact descending (most essential first).
        """
        scores = self.importance_scores()
        candidates = [
            (name, impact) for name, impact in scores.items() if impact > threshold
        ]
        candidates.sort(key=lambda kv: kv[1], reverse=True)
        return [name for name, _ in candidates]

    def gray_zone(
        self,
        dead_threshold: float = 0.05,
        essential_threshold: float = 0.20,
    ) -> list[str]:
        """Return brains that are neither dead nor essential.

        These brains have *some* impact but might be redundant with
        other brains. They warrant deeper investigation.

        Args:
            dead_threshold: Below this absolute impact -> dead.
            essential_threshold: Above this positive impact -> essential.

        Returns:
            Sorted brain names (by impact descending).
        """
        scores = self.importance_scores()
        gray = [
            (name, impact)
            for name, impact in scores.items()
            if dead_threshold <= abs(impact) <= essential_threshold
        ]
        gray.sort(key=lambda kv: kv[1], reverse=True)
        return [name for name, _ in gray]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Serialize the entire study for JSON export."""
        if self._baseline is None:
            msg = "No baseline available; run run_full_baseline() first"
            raise RuntimeError(msg)

        baseline_dict = self._baseline.to_dict()
        drop_dict: dict[str, dict[str, object]] = {
            name: metrics.to_dict() for name, metrics in self._drop_results.items()
        }
        importance: dict[str, float] = {}
        if self._drop_results:
            importance = {
                name: round(score, 4)
                for name, score in self.importance_scores().items()
            }
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "detection_threshold": self._detection_threshold,
            "brain_count": len(self._brains),
            "brains": self.brain_names,
            "baseline": baseline_dict,
            "drop_one": drop_dict,
            "importance_scores": importance,
            "dead_weight": self.identify_dead_weight() if self._drop_results else [],
            "essential": self.identify_essential() if self._drop_results else [],
            "gray_zone": self.gray_zone() if self._drop_results else [],
        }
