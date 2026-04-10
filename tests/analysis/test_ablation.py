"""Tests for :mod:`sybilcore.analysis.ablation`.

The tests use a minimal fake-brain framework so they run fast and
don't depend on network-backed brain modules (embeddings, fidelity,
contrastive etc).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sybilcore.analysis.ablation import (
    DETECTION_THRESHOLD,
    AblationMetrics,
    BrainAblationStudy,
    LabeledAgent,
)
from sybilcore.brains.base import BaseBrain, BrainScore
from sybilcore.models.event import Event, EventType


# ---------------------------------------------------------------------------
# Fake brain fixtures
# ---------------------------------------------------------------------------


class _FixedScoreBrain(BaseBrain):
    """Brain that returns a hard-coded threat value for every agent.

    If ``only_if_adversarial_marker`` is set, the brain only fires on
    agents whose first event's ``metadata["label"]`` equals
    ``"adv"``; benign agents score 0.
    """

    def __init__(
        self,
        name: str,
        fixed_value: float,
        confidence: float = 1.0,
        only_if_adversarial_marker: bool = False,
        indicators: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._name = name
        self._value = fixed_value
        self._confidence = confidence
        self._gated = only_if_adversarial_marker
        self._indicators = indicators or ["signal_a", "signal_b"]

    @property
    def name(self) -> str:
        return self._name

    def score(self, events: list[Event]) -> BrainScore:
        if not events:
            return BrainScore(
                brain_name=self._name, value=0.0, confidence=0.1, reasoning="empty"
            )
        if self._gated:
            first = events[0]
            if first.metadata.get("label") != "adv":
                return BrainScore(
                    brain_name=self._name,
                    value=0.0,
                    confidence=0.1,
                    reasoning="benign",
                )
        return BrainScore(
            brain_name=self._name,
            value=self._value,
            confidence=self._confidence,
            reasoning="synthetic",
            indicators=list(self._indicators),
        )


def _make_agent(agent_id: str, label: int, *, adv: bool = False) -> LabeledAgent:
    event = Event(
        agent_id=agent_id,
        event_type=EventType.TOOL_CALL,
        timestamp=datetime.now(UTC),
        content="synthetic",
        metadata={"label": "adv" if adv else "ben"},
        source="test",
    )
    return LabeledAgent(
        agent_id=agent_id,
        events=(event,),
        label=label,
        source="unit_test",
    )


@pytest.fixture()
def fake_brains() -> list[BaseBrain]:
    """Three brains: a strong detector, a weak booster, and a silent one.

    The detector fires at 80 with full confidence and multiple
    indicators — enough to cross every detection threshold on its
    own. The booster fires at 10 with low confidence 0.2 and a single
    indicator — far too weak to trip detection alone (no critical
    escalation because it has only one indicator), and weighted
    dilution from the silent brain keeps the surface coefficient
    below 100 when detector is dropped.
    """
    return [
        _FixedScoreBrain(
            "detector",
            fixed_value=80.0,
            only_if_adversarial_marker=True,
            indicators=["s1", "s2"],
        ),
        _FixedScoreBrain(
            "booster",
            fixed_value=10.0,
            confidence=0.2,
            only_if_adversarial_marker=True,
            indicators=["single_only"],
        ),
        _FixedScoreBrain("silent", fixed_value=0.0, indicators=["none"]),
    ]


@pytest.fixture()
def corpus() -> list[LabeledAgent]:
    positives = [_make_agent(f"adv-{i}", label=1, adv=True) for i in range(10)]
    negatives = [_make_agent(f"ben-{i}", label=0, adv=False) for i in range(10)]
    return [*positives, *negatives]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_study_reports_brain_names(fake_brains: list[BaseBrain]) -> None:
    study = BrainAblationStudy(brains=fake_brains)
    assert study.brain_names == ["detector", "booster", "silent"]


@pytest.mark.unit
def test_study_rejects_empty_brain_list() -> None:
    with pytest.raises(ValueError, match="at least one brain"):
        BrainAblationStudy(brains=[])


@pytest.mark.unit
def test_full_baseline_detects_all_adversarial_agents(
    fake_brains: list[BaseBrain],
    corpus: list[LabeledAgent],
) -> None:
    study = BrainAblationStudy(brains=fake_brains)
    baseline = study.run_full_baseline(corpus)
    assert baseline.detection_rate == pytest.approx(1.0)
    assert baseline.false_positive_rate == pytest.approx(0.0)
    assert baseline.n_adversarial == 10
    assert baseline.n_benign == 10
    assert baseline.n_agents == 20
    assert baseline.avg_coefficient_adversarial > baseline.avg_coefficient_benign


@pytest.mark.unit
def test_baseline_is_cached_on_instance(
    fake_brains: list[BaseBrain],
    corpus: list[LabeledAgent],
) -> None:
    study = BrainAblationStudy(brains=fake_brains)
    assert study.baseline is None
    study.run_full_baseline(corpus)
    assert study.baseline is not None


@pytest.mark.unit
def test_run_drop_one_rejects_unknown_brain(
    fake_brains: list[BaseBrain],
    corpus: list[LabeledAgent],
) -> None:
    study = BrainAblationStudy(brains=fake_brains)
    with pytest.raises(ValueError, match="Unknown brain"):
        study.run_drop_one("nonexistent", corpus)


@pytest.mark.unit
def test_drop_strong_detector_crashes_detection(
    fake_brains: list[BaseBrain],
    corpus: list[LabeledAgent],
) -> None:
    study = BrainAblationStudy(brains=fake_brains)
    study.run_full_baseline(corpus)
    # With detector gone the booster alone cannot cross DETECTION_THRESHOLD.
    dropped = study.run_drop_one("detector", corpus)
    assert dropped.detection_rate < study.baseline.detection_rate


@pytest.mark.unit
def test_drop_silent_brain_is_neutral(
    fake_brains: list[BaseBrain],
    corpus: list[LabeledAgent],
) -> None:
    study = BrainAblationStudy(brains=fake_brains)
    baseline = study.run_full_baseline(corpus)
    dropped = study.run_drop_one("silent", corpus)
    assert dropped.detection_rate == pytest.approx(baseline.detection_rate)
    assert dropped.false_positive_rate == pytest.approx(baseline.false_positive_rate)


@pytest.mark.unit
def test_run_drop_each_covers_every_brain(
    fake_brains: list[BaseBrain],
    corpus: list[LabeledAgent],
) -> None:
    study = BrainAblationStudy(brains=fake_brains)
    study.run_full_baseline(corpus)
    results = study.run_drop_each(corpus)
    assert set(results.keys()) == {"detector", "booster", "silent"}
    assert set(study.drop_results.keys()) == set(results.keys())


@pytest.mark.unit
def test_importance_scores_match_detection_deltas(
    fake_brains: list[BaseBrain],
    corpus: list[LabeledAgent],
) -> None:
    study = BrainAblationStudy(brains=fake_brains)
    baseline = study.run_full_baseline(corpus)
    study.run_drop_each(corpus)
    importance = study.importance_scores()
    for name, metrics in study.drop_results.items():
        assert importance[name] == pytest.approx(
            baseline.detection_rate - metrics.detection_rate
        )


@pytest.mark.unit
def test_importance_scores_requires_baseline(
    fake_brains: list[BaseBrain],
) -> None:
    study = BrainAblationStudy(brains=fake_brains)
    with pytest.raises(RuntimeError, match="run_full_baseline"):
        study.importance_scores()


@pytest.mark.unit
def test_identify_dead_and_essential_brains(
    fake_brains: list[BaseBrain],
    corpus: list[LabeledAgent],
) -> None:
    study = BrainAblationStudy(brains=fake_brains)
    study.run_full_baseline(corpus)
    study.run_drop_each(corpus)
    dead = study.identify_dead_weight(threshold=0.05)
    essential = study.identify_essential(threshold=0.20)
    assert "silent" in dead
    assert "detector" in essential
    # A brain never appears in both dead and essential.
    assert not set(dead) & set(essential)


@pytest.mark.unit
def test_gray_zone_excludes_dead_and_essential(
    fake_brains: list[BaseBrain],
    corpus: list[LabeledAgent],
) -> None:
    study = BrainAblationStudy(brains=fake_brains)
    study.run_full_baseline(corpus)
    study.run_drop_each(corpus)
    dead = set(study.identify_dead_weight(threshold=0.05))
    essential = set(study.identify_essential(threshold=0.20))
    gray = set(study.gray_zone(dead_threshold=0.05, essential_threshold=0.20))
    assert not gray & dead
    assert not gray & essential


@pytest.mark.unit
def test_to_dict_serializes_results(
    fake_brains: list[BaseBrain],
    corpus: list[LabeledAgent],
    tmp_path: Path,
) -> None:
    study = BrainAblationStudy(brains=fake_brains)
    study.run_full_baseline(corpus)
    study.run_drop_each(corpus)
    payload = study.to_dict()
    out = tmp_path / "ablation.json"
    out.write_text(json.dumps(payload, default=str))
    reloaded = json.loads(out.read_text())
    assert reloaded["brain_count"] == 3
    assert set(reloaded["brains"]) == {"detector", "booster", "silent"}
    assert "baseline" in reloaded
    assert "drop_one" in reloaded
    assert "importance_scores" in reloaded


@pytest.mark.unit
def test_metrics_to_dict_keys() -> None:
    metrics = AblationMetrics(
        detection_rate=0.5,
        false_positive_rate=0.1,
        avg_coefficient=120.0,
        avg_coefficient_adversarial=200.0,
        avg_coefficient_benign=40.0,
        per_agent_coefficients={"a": 10.0},
        n_agents=1,
        n_adversarial=1,
        n_benign=0,
    )
    payload = metrics.to_dict()
    assert payload["detection_rate"] == 0.5
    assert payload["n_agents"] == 1
    assert "per_agent_coefficients" not in payload  # deliberately excluded


@pytest.mark.unit
def test_detection_threshold_default_matches_constant(
    fake_brains: list[BaseBrain],
) -> None:
    study = BrainAblationStudy(brains=fake_brains)
    assert study.detection_threshold == DETECTION_THRESHOLD


@pytest.mark.unit
def test_score_corpus_rejects_empty_corpus(
    fake_brains: list[BaseBrain],
) -> None:
    study = BrainAblationStudy(brains=fake_brains)
    with pytest.raises(ValueError, match="empty corpus"):
        study.run_full_baseline([])
