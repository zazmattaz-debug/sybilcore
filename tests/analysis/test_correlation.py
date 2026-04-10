"""Tests for the cross-brain correlation analysis framework."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from sybilcore.analysis.correlation import (
    BrainPair,
    CorrelationReport,
    CrossBrainCorrelation,
)
from sybilcore.analysis.corpus import build_test_corpus
from sybilcore.brains.base import BaseBrain, BrainScore
from sybilcore.models.event import Event, EventType


# ── Fakes ──────────────────────────────────────────────────────────


class _FakeBrain(BaseBrain):
    """A brain that returns a fixed score per scenario index."""

    def __init__(self, name: str, scores: list[float]) -> None:
        super().__init__()
        self._name = name
        self._scores = scores
        self._idx = 0

    @property
    def name(self) -> str:
        return self._name

    def score(self, events: list[Event]) -> BrainScore:  # noqa: ARG002
        value = self._scores[self._idx % len(self._scores)]
        self._idx += 1
        return BrainScore(brain_name=self._name, value=value, confidence=1.0)


def _make_corpus(n: int = 10) -> list[tuple[str, list[Event]]]:
    base = datetime.now(UTC) - timedelta(seconds=120)
    return [
        (
            f"scenario-{i}",
            [
                Event(
                    agent_id=f"agent-{i}",
                    event_type=EventType.MESSAGE_SENT,
                    content=f"event {i}",
                    timestamp=base + timedelta(seconds=i),
                )
            ],
        )
        for i in range(n)
    ]


# ── Tests ──────────────────────────────────────────────────────────


def test_collect_scores_returns_dataframe_with_brain_columns() -> None:
    n = 6
    brains = [
        _FakeBrain("a", [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        _FakeBrain("b", [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
    ]
    cbc = CrossBrainCorrelation(brains=brains)
    df = cbc.collect_scores(_make_corpus(n))
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (n, 2)


def test_compute_pearson_for_perfectly_correlated_brains() -> None:
    n = 6
    same = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    cbc = CrossBrainCorrelation(brains=[_FakeBrain("x", same), _FakeBrain("y", same)])
    cbc.collect_scores(_make_corpus(n))
    pearson = cbc.compute_pearson()
    assert pearson.at["x", "y"] == pytest.approx(1.0, abs=1e-6)
    assert pearson.at["x", "x"] == pytest.approx(1.0)


def test_compute_pearson_for_anti_correlated_brains() -> None:
    n = 6
    a = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    b = [60.0, 50.0, 40.0, 30.0, 20.0, 10.0]
    cbc = CrossBrainCorrelation(brains=[_FakeBrain("a", a), _FakeBrain("b", b)])
    cbc.collect_scores(_make_corpus(n))
    pearson = cbc.compute_pearson()
    assert pearson.at["a", "b"] == pytest.approx(-1.0, abs=1e-6)


def test_zero_variance_column_yields_zero_correlation() -> None:
    n = 5
    constant = [42.0] * n
    varying = [10.0, 20.0, 30.0, 40.0, 50.0]
    cbc = CrossBrainCorrelation(
        brains=[_FakeBrain("flat", constant), _FakeBrain("up", varying)]
    )
    cbc.collect_scores(_make_corpus(n))
    pearson = cbc.compute_pearson()
    assert pearson.at["flat", "up"] == 0.0


def test_compute_spearman_returns_square_matrix() -> None:
    n = 5
    cbc = CrossBrainCorrelation(
        brains=[
            _FakeBrain("p", [1.0, 2.0, 3.0, 4.0, 5.0]),
            _FakeBrain("q", [5.0, 4.0, 3.0, 2.0, 1.0]),
        ]
    )
    cbc.collect_scores(_make_corpus(n))
    spearman = cbc.compute_spearman()
    assert spearman.shape == (2, 2)
    assert spearman.at["p", "q"] == pytest.approx(-1.0, abs=1e-6)


def test_compute_kendall_handles_ties() -> None:
    n = 5
    cbc = CrossBrainCorrelation(
        brains=[
            _FakeBrain("a", [10, 10, 20, 20, 30]),
            _FakeBrain("b", [10, 10, 20, 20, 30]),
        ]
    )
    cbc.collect_scores(_make_corpus(n))
    kendall = cbc.compute_kendall()
    assert kendall.at["a", "b"] == pytest.approx(1.0, abs=1e-6)


def test_identify_redundant_pairs_above_threshold() -> None:
    n = 6
    same = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    differ = [60.0, 10.0, 50.0, 20.0, 40.0, 30.0]
    cbc = CrossBrainCorrelation(
        brains=[
            _FakeBrain("a", same),
            _FakeBrain("b", same),
            _FakeBrain("c", differ),
        ]
    )
    cbc.collect_scores(_make_corpus(n))
    pairs = cbc.identify_redundant_pairs(threshold=0.95)
    assert any(p.brain_a == "a" and p.brain_b == "b" for p in pairs)
    assert all(abs(p.correlation) >= 0.95 for p in pairs)


def test_identify_complementary_pairs_below_threshold() -> None:
    n = 6
    same = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    constant = [5.0] * n
    cbc = CrossBrainCorrelation(
        brains=[_FakeBrain("a", same), _FakeBrain("b", constant)]
    )
    cbc.collect_scores(_make_corpus(n))
    pairs = cbc.identify_complementary_pairs(threshold=0.05)
    # Constant column has zero variance -> r = 0
    assert any(p.brain_a == "a" and p.brain_b == "b" for p in pairs)


def test_cluster_brains_returns_int_labels_for_each_brain() -> None:
    n = 8
    cbc = CrossBrainCorrelation(
        brains=[
            _FakeBrain("a", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            _FakeBrain("b", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            _FakeBrain("c", [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
            _FakeBrain("d", [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]),
        ]
    )
    cbc.collect_scores(_make_corpus(n))
    clusters = cbc.cluster_brains(n_clusters=2)
    assert set(clusters.keys()) == {"a", "b", "c", "d"}
    assert all(isinstance(v, int) for v in clusters.values())


def test_generate_heatmap_writes_png(tmp_path: Path) -> None:
    n = 5
    cbc = CrossBrainCorrelation(
        brains=[
            _FakeBrain("a", [1.0, 2.0, 3.0, 4.0, 5.0]),
            _FakeBrain("b", [5.0, 4.0, 3.0, 2.0, 1.0]),
        ]
    )
    cbc.collect_scores(_make_corpus(n))
    out = tmp_path / "heatmap.png"
    result = cbc.generate_heatmap(out)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_build_report_collects_all_pieces() -> None:
    n = 6
    cbc = CrossBrainCorrelation(
        brains=[
            _FakeBrain("a", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            _FakeBrain("b", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            _FakeBrain("c", [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
        ]
    )
    cbc.collect_scores(_make_corpus(n))
    report = cbc.build_report(threshold_redundant=0.95, threshold_complementary=0.1)
    assert isinstance(report, CorrelationReport)
    assert isinstance(report.pearson, pd.DataFrame)
    assert isinstance(report.spearman, pd.DataFrame)
    assert isinstance(report.kendall, pd.DataFrame)
    assert any(isinstance(p, BrainPair) for p in report.redundant_pairs)
    payload = report.to_dict()
    assert "pearson" in payload
    assert "redundant_pairs" in payload


def test_scores_property_raises_when_unset() -> None:
    cbc = CrossBrainCorrelation(brains=[_FakeBrain("a", [1.0])])
    with pytest.raises(RuntimeError):
        _ = cbc.scores


def test_real_corpus_runs_end_to_end() -> None:
    cbc = CrossBrainCorrelation()
    corpus = build_test_corpus()
    df = cbc.collect_scores(corpus[:25])  # subset for speed
    assert df.shape[0] == 25
    assert len(df.columns) == len(cbc.brain_names)
    pearson = cbc.compute_pearson()
    assert pearson.shape == (len(cbc.brain_names), len(cbc.brain_names))
