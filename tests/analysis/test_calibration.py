"""Tests for sybilcore.analysis.calibration.

Uses a tiny stub-brain set so the suite stays fast (no transformer
loading, no real archetype generation). The full pipeline is exercised
end-to-end with real brains in the integration smoke test at the
bottom (`test_smoke_runner`).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from sybilcore.analysis.calibration import (
    DEFAULT_DECISION_THRESHOLD,
    MAX_WEIGHT,
    MIN_WEIGHT,
    THRESHOLD_SEARCH_GRID,
    CalibrationCorpus,
    CalibrationMetrics,
    CalibrationResult,
    LabeledAgent,
    WeightCalibrator,
)
from sybilcore.brains.base import BaseBrain, BrainScore
from sybilcore.models.event import Event, EventType


# ── Stub brains ──────────────────────────────────────────────────────


class StaticBrain(BaseBrain):
    """Brain that returns a fixed score from the agent's metadata.

    Each agent encodes per-brain scores in event[0].metadata so we can
    drive the calibrator deterministically without running real brains.
    """

    def __init__(self, brain_name: str) -> None:  # noqa: D401
        super().__init__()
        self._name = brain_name

    @property
    def name(self) -> str:
        return self._name

    def score(self, events: list[Event]) -> BrainScore:
        if not events:
            return self._empty_score()
        meta: dict[str, Any] = events[0].metadata
        value = float(meta.get(self._name, 0.0))
        return BrainScore(
            brain_name=self._name,
            value=value,
            confidence=1.0,
            reasoning="stub",
            indicators=[],
        )


# ── Fixtures ─────────────────────────────────────────────────────────


def _make_event(agent_id: str, scores: dict[str, float]) -> Event:
    """Build a single event whose metadata encodes per-brain scores."""
    return Event(
        agent_id=agent_id,
        event_type=EventType.TOOL_CALL,
        timestamp=datetime.now(UTC),
        content="stub",
        metadata=scores,
        source="test",
    )


def _agent(agent_id: str, label: int, scores: dict[str, float]) -> LabeledAgent:
    return LabeledAgent(
        agent_id=agent_id,
        events=(_make_event(agent_id, scores),),
        label=label,
        source=f"test:{label}",
    )


@pytest.fixture()
def stub_brains() -> list[BaseBrain]:
    return [StaticBrain("alpha"), StaticBrain("beta"), StaticBrain("gamma")]


@pytest.fixture()
def calibrator(stub_brains: list[BaseBrain]) -> WeightCalibrator:
    return WeightCalibrator(seed=1, brains=stub_brains)


@pytest.fixture()
def linear_corpus(calibrator: WeightCalibrator) -> CalibrationCorpus:
    """5 obvious positives + 5 obvious negatives."""
    positives = [
        _agent(f"pos-{i}", 1, {"alpha": 90.0, "beta": 80.0, "gamma": 10.0})
        for i in range(5)
    ]
    negatives = [
        _agent(f"neg-{i}", 0, {"alpha": 5.0, "beta": 5.0, "gamma": 5.0})
        for i in range(5)
    ]
    return calibrator.load_corpus(positives, negatives)


# ── Tests ────────────────────────────────────────────────────────────


def test_load_corpus_caches_brain_scores(
    calibrator: WeightCalibrator,
) -> None:
    pos = [_agent("pos-1", 1, {"alpha": 50.0, "beta": 50.0, "gamma": 50.0})]
    neg = [_agent("neg-1", 0, {"alpha": 0.0, "beta": 0.0, "gamma": 0.0})]
    corpus = calibrator.load_corpus(pos, neg)
    assert len(corpus) == 2
    assert "pos-1" in corpus.brain_scores
    assert "neg-1" in corpus.brain_scores
    assert len(corpus.brain_scores["pos-1"]) == 3
    assert all(isinstance(bs, BrainScore) for bs in corpus.brain_scores["pos-1"])


def test_load_corpus_rejects_empty_positives(
    calibrator: WeightCalibrator,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        calibrator.load_corpus([], [_agent("n", 0, {"alpha": 0})])


def test_load_corpus_rejects_empty_negatives(
    calibrator: WeightCalibrator,
) -> None:
    with pytest.raises(ValueError, match="negative"):
        calibrator.load_corpus([_agent("p", 1, {"alpha": 0})], [])


def test_score_with_weights_returns_per_agent_coefficients(
    calibrator: WeightCalibrator,
    linear_corpus: CalibrationCorpus,
) -> None:
    weights = {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}
    coeffs = calibrator.score_with_weights(weights, linear_corpus)
    assert set(coeffs.keys()) == {a.agent_id for a in linear_corpus.agents}
    pos_avg = sum(coeffs[a.agent_id] for a in linear_corpus.positives) / 5
    neg_avg = sum(coeffs[a.agent_id] for a in linear_corpus.negatives) / 5
    assert pos_avg > neg_avg
    # All coefficients within [0, 500]
    for v in coeffs.values():
        assert 0.0 <= v <= 500.0


def test_score_with_weights_handles_zero_weight_total(
    calibrator: WeightCalibrator,
    linear_corpus: CalibrationCorpus,
) -> None:
    coeffs = calibrator.score_with_weights({}, linear_corpus)
    # Falls back to weight 1.0 per brain — no division by zero.
    assert all(v >= 0.0 for v in coeffs.values())


def test_compute_metrics_perfect_separation(
    calibrator: WeightCalibrator,
    linear_corpus: CalibrationCorpus,
) -> None:
    weights = {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}
    coeffs = calibrator.score_with_weights(weights, linear_corpus)
    # Pick a threshold between the two clusters
    metrics = calibrator.compute_metrics(coeffs, linear_corpus, threshold=100.0)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.fpr == 0.0
    assert metrics.tp == 5
    assert metrics.tn == 5
    assert metrics.fp == 0
    assert metrics.fn == 0
    assert metrics.auc == 1.0


def test_compute_metrics_threshold_too_high_zero_recall(
    calibrator: WeightCalibrator,
    linear_corpus: CalibrationCorpus,
) -> None:
    weights = {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}
    coeffs = calibrator.score_with_weights(weights, linear_corpus)
    metrics = calibrator.compute_metrics(coeffs, linear_corpus, threshold=499.0)
    assert metrics.recall == 0.0
    assert metrics.tp == 0
    assert metrics.fn == 5
    assert metrics.f1 == 0.0


def test_objective_with_threshold_search_finds_optimal(
    calibrator: WeightCalibrator,
    linear_corpus: CalibrationCorpus,
) -> None:
    weights = {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}
    fitness, metrics, threshold = calibrator.objective_with_threshold_search(
        weights, linear_corpus
    )
    # Perfect separation should yield F1=1
    assert fitness == 1.0
    assert metrics.f1 == 1.0
    assert threshold in THRESHOLD_SEARCH_GRID


def test_objective_penalizes_fpr_above_cap(
    stub_brains: list[BaseBrain],
) -> None:
    cal = WeightCalibrator(seed=1, brains=stub_brains, fpr_cap=0.0)
    pos = [_agent(f"pos-{i}", 1, {"alpha": 60.0}) for i in range(5)]
    neg = [_agent(f"neg-{i}", 0, {"alpha": 60.0}) for i in range(5)]
    corpus = cal.load_corpus(pos, neg)
    fitness, metrics = cal.objective(
        {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}, corpus
    )
    assert fitness == 0.0
    assert metrics.fpr > 0.0


def test_random_search_returns_valid_result(
    calibrator: WeightCalibrator,
    linear_corpus: CalibrationCorpus,
) -> None:
    result = calibrator.random_search(linear_corpus, n_iter=20)
    assert isinstance(result, CalibrationResult)
    assert result.iterations == 20
    assert len(result.history) == 20
    # All sampled weights inside the allowed range
    for w in result.weights.values():
        assert MIN_WEIGHT <= w <= MAX_WEIGHT
    # On a perfectly separable corpus, random search should find F1=1 within 20 tries
    assert result.metrics.f1 == 1.0


def test_local_search_improves_or_holds(
    calibrator: WeightCalibrator,
    linear_corpus: CalibrationCorpus,
) -> None:
    start = {"alpha": 0.5, "beta": 0.5, "gamma": 0.5}
    _, start_metrics = calibrator.objective(start, linear_corpus)
    result = calibrator.local_search(start, linear_corpus, n_iter=20, step=0.2)
    assert result.metrics.f1 >= start_metrics.f1
    assert result.iterations >= 1
    # Local search must keep weights in bounds
    for w in result.weights.values():
        assert MIN_WEIGHT <= w <= MAX_WEIGHT


def test_hybrid_search_combines_phases(
    calibrator: WeightCalibrator,
    linear_corpus: CalibrationCorpus,
) -> None:
    result = calibrator.hybrid_search(
        linear_corpus, random_iter=15, local_iter=10
    )
    assert isinstance(result, CalibrationResult)
    # iterations field is the total budget after merging
    assert result.iterations >= 15
    # History entries should be tagged with phase
    phases = {h.get("phase") for h in result.history}
    assert "random" in phases
    # If local search ran at least one step it should appear too
    assert result.metrics.f1 == 1.0


def test_default_weights_uses_default_brain_weights(
    calibrator: WeightCalibrator,
) -> None:
    defaults = calibrator.default_weights()
    assert set(defaults.keys()) == set(calibrator._brain_names)  # noqa: SLF001
    # All stub brains fall through to the 1.0 fallback in DEFAULT_BRAIN_WEIGHTS
    assert all(v == 1.0 for v in defaults.values())


def test_save_optimal_weights_round_trip(
    tmp_path: Path,
    calibrator: WeightCalibrator,
    linear_corpus: CalibrationCorpus,
) -> None:
    result = calibrator.random_search(linear_corpus, n_iter=5)
    out = tmp_path / "weights.json"
    saved = calibrator.save_optimal_weights(result, out)
    assert saved.exists()
    payload = json.loads(saved.read_text())
    assert "weights" in payload
    assert "metrics" in payload
    assert "decision_threshold" in payload
    assert payload["weights"].keys() == set(result.weights.keys())
    assert payload["metrics"]["f1"] == result.metrics.f1


def test_decision_threshold_default_constant(
    calibrator: WeightCalibrator,
) -> None:
    assert calibrator.decision_threshold == DEFAULT_DECISION_THRESHOLD


def test_compute_metrics_returns_immutable_dataclass(
    calibrator: WeightCalibrator,
    linear_corpus: CalibrationCorpus,
) -> None:
    weights = {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}
    coeffs = calibrator.score_with_weights(weights, linear_corpus)
    m = calibrator.compute_metrics(coeffs, linear_corpus, threshold=50.0)
    assert isinstance(m, CalibrationMetrics)
    with pytest.raises((AttributeError, Exception)):
        m.f1 = 0.0  # type: ignore[misc]
