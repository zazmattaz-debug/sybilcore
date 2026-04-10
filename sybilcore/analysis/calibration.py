"""Brain weight calibration via gradient-free optimization.

Replaces the intuition-derived `DEFAULT_BRAIN_WEIGHTS` with empirically
optimal weights derived from a labeled corpus of suspicious vs benign
agent event streams.

Approach:
    1. Build a labeled corpus (true positives + true negatives).
    2. Score every agent once with each brain (cached `BrainScore`s).
    3. Search the weight space (random + local hill-climb hybrid).
    4. Pick the weight vector that maximizes F1 while keeping the
       false-positive rate on benign agents below a hard cap.

The cached-scores trick is critical: brain scoring is expensive,
weight search is cheap. We pay the brain cost once, then evaluate
hundreds of weight vectors at near-zero marginal cost.

Reference:
    - Hooke & Jeeves (1961), "Direct search solution of numerical and
      statistical problems."
    - Nelder & Mead (1965) — but pattern search is simpler and works
      well for low-dimensional weight tuning.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from sybilcore.brains import get_default_brains
from sybilcore.core.config import (
    COEFFICIENT_SCALE_FACTOR,
    DEFAULT_BRAIN_WEIGHTS,
    MAX_COEFFICIENT,
)

if TYPE_CHECKING:
    from sybilcore.brains.base import BaseBrain, BrainScore
    from sybilcore.models.event import Event

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────

MIN_WEIGHT: float = 0.1
MAX_WEIGHT: float = 3.0
# Decision threshold default: surface coefficient lives in roughly the
# 0-100 range (raw weighted average of brain scores × scale factor),
# so 65 sits between observed benign median (~52) and the suspicious
# floor (~64) for default weights. The optimizer also sweeps the
# threshold per weight vector, so this is just the seed.
DEFAULT_DECISION_THRESHOLD: float = 65.0
DEFAULT_FPR_CAP: float = 0.05  # 5% false-positive rate hard limit
THRESHOLD_SEARCH_GRID: tuple[float, ...] = (
    40.0, 45.0, 50.0, 55.0, 60.0, 62.5, 65.0, 67.5,
    70.0, 72.5, 75.0, 80.0, 85.0, 90.0, 100.0,
)


# ── Data containers ──────────────────────────────────────────────────


@dataclass(frozen=True)
class LabeledAgent:
    """A single labeled training example.

    Attributes:
        agent_id: Unique identifier.
        events: Raw event stream produced by the agent.
        label: 1 = suspicious (positive), 0 = benign (negative).
        source: Where this example came from (alignment_faker,
            archetype:sleeper, moltbook_clean, etc.).
    """

    agent_id: str
    events: tuple[Event, ...]
    label: int
    source: str


@dataclass
class CalibrationCorpus:
    """Container of labeled agents plus their cached brain scores.

    Caching `brain_scores` is the whole reason calibration is
    tractable. The brains run exactly once per agent — every weight
    vector reuses the same scores.

    Attributes:
        agents: Labeled training examples.
        brain_scores: agent_id -> list of BrainScore. Filled by
            `WeightCalibrator.precompute_brain_scores`.
    """

    agents: list[LabeledAgent] = field(default_factory=list)
    brain_scores: dict[str, list[BrainScore]] = field(default_factory=dict)

    @property
    def positives(self) -> list[LabeledAgent]:
        return [a for a in self.agents if a.label == 1]

    @property
    def negatives(self) -> list[LabeledAgent]:
        return [a for a in self.agents if a.label == 0]

    def __len__(self) -> int:
        return len(self.agents)


@dataclass(frozen=True)
class CalibrationMetrics:
    """Metrics returned by `WeightCalibrator.compute_metrics`."""

    precision: float
    recall: float
    f1: float
    auc: float
    fpr: float
    tp: int
    fp: int
    tn: int
    fn: int


@dataclass
class CalibrationResult:
    """Final result from a calibration run."""

    weights: dict[str, float]
    metrics: CalibrationMetrics
    iterations: int
    threshold: float = DEFAULT_DECISION_THRESHOLD
    history: list[dict] = field(default_factory=list)


# ── Calibrator ──────────────────────────────────────────────────────


class WeightCalibrator:
    """Gradient-free optimizer for brain weights.

    Uses a hybrid of random search (exploration) and local hill
    climbing (exploitation). The objective is F1 score on the labeled
    corpus, subject to a hard false-positive-rate cap.

    Args:
        decision_threshold: Coefficient threshold above which an agent
            is classified as suspicious. Defaults to 200.0
            (the FLAGGED tier boundary).
        fpr_cap: Maximum allowed false-positive rate on benign agents.
            Weight vectors that violate this are penalized to F1=0.
        seed: RNG seed for reproducibility.
        brains: Optional list of brain instances. Defaults to
            `get_default_brains()`.
    """

    def __init__(
        self,
        decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
        fpr_cap: float = DEFAULT_FPR_CAP,
        seed: int = 42,
        brains: list[BaseBrain] | None = None,
    ) -> None:
        self.decision_threshold = decision_threshold
        self.fpr_cap = fpr_cap
        self._rng = random.Random(seed)  # noqa: S311 — non-crypto use
        self._brains = brains if brains is not None else get_default_brains()
        self._brain_names: list[str] = [b.name for b in self._brains]

    # ── Corpus loading ───────────────────────────────────────────────

    def load_corpus(
        self,
        positives: list[LabeledAgent],
        negatives: list[LabeledAgent],
    ) -> CalibrationCorpus:
        """Bundle labeled agents and precompute their brain scores.

        Args:
            positives: Suspicious / true-positive examples.
            negatives: Benign / true-negative examples.

        Returns:
            CalibrationCorpus with `brain_scores` already populated.
        """
        if not positives:
            msg = "Corpus must contain at least one positive example"
            raise ValueError(msg)
        if not negatives:
            msg = "Corpus must contain at least one negative example"
            raise ValueError(msg)

        corpus = CalibrationCorpus(agents=[*positives, *negatives])
        self.precompute_brain_scores(corpus)
        return corpus

    def precompute_brain_scores(self, corpus: CalibrationCorpus) -> None:
        """Run every brain on every agent once and cache the results."""
        for agent in corpus.agents:
            scores = [brain.score(list(agent.events)) for brain in self._brains]
            corpus.brain_scores[agent.agent_id] = scores
        logger.info(
            "Precomputed brain scores for %d agents (%d brains each)",
            len(corpus),
            len(self._brains),
        )

    # ── Scoring ──────────────────────────────────────────────────────

    def score_with_weights(
        self,
        weights: dict[str, float],
        corpus: CalibrationCorpus,
    ) -> dict[str, float]:
        """Apply a weight vector to cached brain scores.

        Returns:
            agent_id -> coefficient (0..MAX_COEFFICIENT).
        """
        coefficients: dict[str, float] = {}
        for agent in corpus.agents:
            brain_scores = corpus.brain_scores[agent.agent_id]
            coefficients[agent.agent_id] = self._aggregate(weights, brain_scores)
        return coefficients

    def _aggregate(
        self,
        weights: dict[str, float],
        brain_scores: list[BrainScore],
    ) -> float:
        """Replicate `CoefficientCalculator.calculate` weighting math.

        We use the *surface* coefficient only (weighted average) for
        calibration. Semantic + critical-escalation channels are
        weight-independent and would muddy the signal we're trying to
        optimize.
        """
        weighted_sum = 0.0
        weight_total = 0.0
        for bs in brain_scores:
            w = weights.get(bs.brain_name, 1.0)
            weighted_sum += bs.value * w * bs.confidence
            weight_total += w
        if weight_total == 0.0:
            return 0.0
        raw = (weighted_sum / weight_total) * COEFFICIENT_SCALE_FACTOR
        return max(0.0, min(raw, MAX_COEFFICIENT))

    # ── Metrics ──────────────────────────────────────────────────────

    def compute_metrics(
        self,
        coefficients: dict[str, float],
        corpus: CalibrationCorpus,
        threshold: float | None = None,
    ) -> CalibrationMetrics:
        """Compute precision, recall, F1, AUC, FPR for a coefficient map."""
        thresh = threshold if threshold is not None else self.decision_threshold
        tp = fp = tn = fn = 0
        for agent in corpus.agents:
            score = coefficients[agent.agent_id]
            predicted = 1 if score >= thresh else 0
            if predicted == 1 and agent.label == 1:
                tp += 1
            elif predicted == 1 and agent.label == 0:
                fp += 1
            elif predicted == 0 and agent.label == 0:
                tn += 1
            else:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        auc = self._compute_auc(coefficients, corpus)
        return CalibrationMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            auc=auc,
            fpr=fpr,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )

    def _compute_auc(
        self,
        coefficients: dict[str, float],
        corpus: CalibrationCorpus,
    ) -> float:
        """Mann-Whitney U statistic — ROC AUC without sklearn dependency."""
        pos_scores = [coefficients[a.agent_id] for a in corpus.positives]
        neg_scores = [coefficients[a.agent_id] for a in corpus.negatives]
        if not pos_scores or not neg_scores:
            return 0.5
        wins = 0.0
        for ps in pos_scores:
            for ns in neg_scores:
                if ps > ns:
                    wins += 1
                elif ps == ns:
                    wins += 0.5
        return wins / (len(pos_scores) * len(neg_scores))

    # ── Objective ────────────────────────────────────────────────────

    def objective(
        self,
        weights: dict[str, float],
        corpus: CalibrationCorpus,
    ) -> tuple[float, CalibrationMetrics]:
        """Score a weight vector at the calibrator's fixed threshold.

        Fitness = F1, but penalized to 0 if FPR exceeds the cap.
        """
        coefficients = self.score_with_weights(weights, corpus)
        metrics = self.compute_metrics(coefficients, corpus)
        if metrics.fpr > self.fpr_cap:
            return 0.0, metrics
        return metrics.f1, metrics

    def objective_with_threshold_search(
        self,
        weights: dict[str, float],
        corpus: CalibrationCorpus,
        grid: tuple[float, ...] = THRESHOLD_SEARCH_GRID,
    ) -> tuple[float, CalibrationMetrics, float]:
        """Find the F1-optimal threshold for this weight vector.

        Returns (fitness, metrics_at_best_threshold, best_threshold).
        Threshold is part of the optimization surface — different
        weight vectors prefer different decision boundaries.
        """
        coefficients = self.score_with_weights(weights, corpus)
        best_fitness = -1.0
        best_metrics: CalibrationMetrics | None = None
        best_threshold = self.decision_threshold
        for thresh in grid:
            m = self.compute_metrics(coefficients, corpus, threshold=thresh)
            if m.fpr > self.fpr_cap:
                continue
            if m.f1 > best_fitness:
                best_fitness = m.f1
                best_metrics = m
                best_threshold = thresh
        if best_metrics is None:
            # No threshold satisfied the FPR cap; report the metrics at
            # the lowest grid point so the caller can still see why.
            best_metrics = self.compute_metrics(
                coefficients, corpus, threshold=grid[-1]
            )
            return 0.0, best_metrics, grid[-1]
        return best_fitness, best_metrics, best_threshold

    # ── Search strategies ────────────────────────────────────────────

    def random_search(
        self,
        corpus: CalibrationCorpus,
        n_iter: int = 100,
        weight_range: tuple[float, float] = (MIN_WEIGHT, MAX_WEIGHT),
    ) -> CalibrationResult:
        """Sample n_iter random weight vectors. Return the best."""
        best_fitness = -1.0
        best_weights: dict[str, float] = {}
        best_metrics: CalibrationMetrics | None = None
        best_threshold = self.decision_threshold
        history: list[dict] = []

        for i in range(n_iter):
            weights = self._random_weights(weight_range)
            fitness, metrics, thresh = self.objective_with_threshold_search(
                weights, corpus
            )
            history.append({
                "iter": i,
                "fitness": fitness,
                "f1": metrics.f1,
                "threshold": thresh,
            })
            if fitness > best_fitness:
                best_fitness = fitness
                best_weights = weights
                best_metrics = metrics
                best_threshold = thresh

        if best_metrics is None:  # pragma: no cover
            seed = self._random_weights(weight_range)
            _, best_metrics, best_threshold = self.objective_with_threshold_search(
                seed, corpus
            )
            best_weights = seed

        return CalibrationResult(
            weights=best_weights,
            metrics=best_metrics,
            iterations=n_iter,
            threshold=best_threshold,
            history=history,
        )

    def local_search(
        self,
        start_weights: dict[str, float],
        corpus: CalibrationCorpus,
        n_iter: int = 50,
        step: float = 0.1,
        step_decay: float = 0.85,
    ) -> CalibrationResult:
        """Hill-climb from `start_weights`.

        Each iteration tries +step / -step on every brain weight and
        keeps the best move. The step shrinks when no neighbor
        improves (pattern search style). Threshold is re-optimized
        for every candidate.
        """
        current = dict(start_weights)
        best_fitness, best_metrics, best_threshold = (
            self.objective_with_threshold_search(current, corpus)
        )
        history: list[dict] = [
            {
                "iter": 0,
                "fitness": best_fitness,
                "step": step,
                "threshold": best_threshold,
            }
        ]

        cur_step = step
        for i in range(1, n_iter + 1):
            improved = False
            for brain_name in self._brain_names:
                for delta in (cur_step, -cur_step):
                    candidate = dict(current)
                    candidate[brain_name] = round(
                        max(
                            MIN_WEIGHT,
                            min(MAX_WEIGHT, candidate[brain_name] + delta),
                        ),
                        4,
                    )
                    fitness, metrics, thresh = (
                        self.objective_with_threshold_search(candidate, corpus)
                    )
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_metrics = metrics
                        best_threshold = thresh
                        current = candidate
                        improved = True
                        break
                if improved:
                    break
            history.append(
                {
                    "iter": i,
                    "fitness": best_fitness,
                    "step": cur_step,
                    "threshold": best_threshold,
                }
            )
            if not improved:
                cur_step *= step_decay
                if cur_step < 0.01:
                    break

        return CalibrationResult(
            weights=current,
            metrics=best_metrics,
            iterations=len(history),
            threshold=best_threshold,
            history=history,
        )

    def hybrid_search(
        self,
        corpus: CalibrationCorpus,
        random_iter: int = 100,
        local_iter: int = 50,
    ) -> CalibrationResult:
        """Random search to explore, local search to refine."""
        coarse = self.random_search(corpus, n_iter=random_iter)
        logger.info(
            "Random search best: F1=%.4f (FPR=%.4f thresh=%.1f)",
            coarse.metrics.f1,
            coarse.metrics.fpr,
            coarse.threshold,
        )
        fine = self.local_search(coarse.weights, corpus, n_iter=local_iter)
        logger.info(
            "Local search best: F1=%.4f (FPR=%.4f thresh=%.1f)",
            fine.metrics.f1,
            fine.metrics.fpr,
            fine.threshold,
        )
        fine.iterations = coarse.iterations + fine.iterations
        fine.history = [
            *[{"phase": "random", **h} for h in coarse.history],
            *[{"phase": "local", **h} for h in fine.history],
        ]
        return fine

    # ── Helpers ──────────────────────────────────────────────────────

    def _random_weights(
        self,
        weight_range: tuple[float, float],
    ) -> dict[str, float]:
        lo, hi = weight_range
        return {
            name: round(self._rng.uniform(lo, hi), 3)
            for name in self._brain_names
        }

    def default_weights(self) -> dict[str, float]:
        """Return DEFAULT_BRAIN_WEIGHTS filtered to known brains."""
        return {
            name: DEFAULT_BRAIN_WEIGHTS.get(name, 1.0)
            for name in self._brain_names
        }

    # ── Persistence ──────────────────────────────────────────────────

    def save_optimal_weights(
        self,
        result: CalibrationResult,
        path: str | Path,
    ) -> Path:
        """Persist a calibration result to JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "weights": result.weights,
            "metrics": {
                "precision": result.metrics.precision,
                "recall": result.metrics.recall,
                "f1": result.metrics.f1,
                "auc": result.metrics.auc,
                "fpr": result.metrics.fpr,
                "tp": result.metrics.tp,
                "fp": result.metrics.fp,
                "tn": result.metrics.tn,
                "fn": result.metrics.fn,
            },
            "iterations": result.iterations,
            "decision_threshold": result.threshold,
            "default_decision_threshold": self.decision_threshold,
            "fpr_cap": self.fpr_cap,
            "brain_names": self._brain_names,
        }
        out.write_text(json.dumps(payload, indent=2))
        return out
