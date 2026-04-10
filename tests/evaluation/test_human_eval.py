"""Tests for the SybilCore human evaluation framework."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sybilcore.evaluation.ai_judge import AIJudge, AIJudgeVerdict, summarize_run
from sybilcore.evaluation.human_eval import (
    AgentRecord,
    HumanEvalFramework,
    Judgment,
    SOURCE_ALIGNMENT,
    SOURCE_ARCHETYPE,
    SOURCE_MOLTBOOK,
    SOURCE_TOURNAMENT,
    SUSPICIOUS_COEFFICIENT_THRESHOLD,
    _majority_label,
    load_alignment_agents,
    load_archetype_agents,
    load_moltbook_agents,
    load_tournament_agents,
)


# ── Fixtures ────────────────────────────────────────────────────────


def _make_record(
    agent_id: str = "a1",
    source: str = SOURCE_MOLTBOOK,
    coefficient: float = 10.0,
    ground_truth: str = "unknown",
) -> AgentRecord:
    return AgentRecord(
        agent_id=agent_id,
        source=source,
        ground_truth=ground_truth,
        sybilcore_coefficient=coefficient,
        sybilcore_tier="clear" if coefficient < 50 else "clouded",
        sybilcore_suspicious=coefficient >= SUSPICIOUS_COEFFICIENT_THRESHOLD,
        brain_scores={"deception": coefficient / 5.0},
        events=[{"event_type": "tool_call", "content": "test"}],
        summary={"note": "fixture"},
    )


@pytest.fixture()
def tmp_labels(tmp_path: Path) -> Path:
    return tmp_path / "labels.jsonl"


@pytest.fixture()
def sample_records() -> list[AgentRecord]:
    return [
        _make_record("benign-1", coefficient=5.0, ground_truth="unknown"),
        _make_record("benign-2", coefficient=15.0, ground_truth="unknown"),
        _make_record("evil-1", source=SOURCE_ARCHETYPE, coefficient=80.0, ground_truth="malicious"),
        _make_record("evil-2", source=SOURCE_ARCHETYPE, coefficient=120.0, ground_truth="malicious"),
        _make_record("evil-3", source=SOURCE_ALIGNMENT, coefficient=70.0, ground_truth="malicious"),
    ]


# ── Loaders ─────────────────────────────────────────────────────────


def test_load_moltbook_agents_returns_records() -> None:
    records = load_moltbook_agents()
    assert isinstance(records, list)
    if records:  # only assert content if the file is present in the repo
        first = records[0]
        assert first.source == SOURCE_MOLTBOOK
        assert first.ground_truth == "unknown"
        assert isinstance(first.brain_scores, dict)


def test_load_archetype_agents_have_malicious_ground_truth() -> None:
    records = load_archetype_agents()
    if not records:
        pytest.skip("archetype experiment file not present")
    assert all(r.source == SOURCE_ARCHETYPE for r in records)
    assert all(r.ground_truth == "malicious" for r in records)


def test_load_alignment_agents_have_trajectories() -> None:
    records = load_alignment_agents()
    if not records:
        pytest.skip("alignment experiment file not present")
    assert all(r.source == SOURCE_ALIGNMENT for r in records)
    # The summary should at least carry the scenario name.
    assert any(r.summary.get("scenario") for r in records)


def test_load_tournament_agents_have_events() -> None:
    records = load_tournament_agents()
    if not records:
        pytest.skip("tournament experiment file not present")
    assert all(r.source == SOURCE_TOURNAMENT for r in records)
    # At least one record should have non-empty events (best_candidate_events).
    assert any(r.events for r in records)


# ── Sampling ────────────────────────────────────────────────────────


def test_sample_corpus_mixed_balances_sources() -> None:
    framework = HumanEvalFramework()
    corpus = framework.sample_corpus(n=40, source="mixed", per_source_n=10)
    sources = {r.source for r in corpus}
    # In a healthy repo we should see at least 2 distinct sources represented.
    assert len(sources) >= 2
    assert len(corpus) <= 40


def test_sample_corpus_invalid_source_raises() -> None:
    framework = HumanEvalFramework()
    with pytest.raises(ValueError):
        framework.sample_corpus(n=10, source="not-a-source")


def test_save_and_load_corpus_roundtrip(tmp_path: Path, sample_records: list[AgentRecord]) -> None:
    out = tmp_path / "corpus.json"
    HumanEvalFramework.save_corpus(sample_records, out)
    loaded = HumanEvalFramework.load_corpus(out)
    assert len(loaded) == len(sample_records)
    assert loaded[0].agent_id == sample_records[0].agent_id
    assert loaded[2].ground_truth == "malicious"


# ── Formatting ──────────────────────────────────────────────────────


def test_format_for_review_includes_id_and_coefficient() -> None:
    record = _make_record("foo", coefficient=42.0)
    text = HumanEvalFramework.format_for_review(record)
    assert "foo" in text
    assert "42" in text
    assert "moltbook" in text


def test_format_for_review_truncates_long_event_content() -> None:
    record = AgentRecord(
        agent_id="x",
        source=SOURCE_TOURNAMENT,
        ground_truth="malicious",
        sybilcore_coefficient=12.0,
        sybilcore_tier="clear",
        sybilcore_suspicious=False,
        brain_scores={},
        events=[{"event_type": "tool_call", "content": "z" * 500}],
        summary={},
    )
    text = HumanEvalFramework.format_for_review(record)
    assert "..." in text


# ── Judgments + storage ────────────────────────────────────────────


def test_record_judgment_persists_to_jsonl(tmp_labels: Path) -> None:
    framework = HumanEvalFramework(labels_path=tmp_labels)
    framework.record_judgment("a1", label="suspicious", confidence=4, notes="hmm")
    judgments = framework.load_judgments()
    assert len(judgments) == 1
    assert judgments[0].label == "suspicious"
    assert judgments[0].confidence == 4
    assert judgments[0].notes == "hmm"


def test_record_judgment_validates_label(tmp_labels: Path) -> None:
    framework = HumanEvalFramework(labels_path=tmp_labels)
    with pytest.raises(ValueError):
        framework.record_judgment("a1", label="malicious", confidence=3)


def test_record_judgment_validates_confidence_range(tmp_labels: Path) -> None:
    framework = HumanEvalFramework(labels_path=tmp_labels)
    with pytest.raises(ValueError):
        framework.record_judgment("a1", label="benign", confidence=10)


# ── Metrics ────────────────────────────────────────────────────────


def test_compare_to_sybilcore_builds_matrix(sample_records: list[AgentRecord]) -> None:
    framework = HumanEvalFramework()
    judgments = [
        Judgment("benign-1", "alice", "benign", 5),
        Judgment("benign-2", "alice", "benign", 4),
        Judgment("evil-1", "alice", "suspicious", 5),
        Judgment("evil-2", "alice", "suspicious", 5),
        Judgment("evil-3", "alice", "suspicious", 4),
    ]
    cmp = framework.compare_to_sybilcore(sample_records, judgments)
    m = cmp["matrix"]
    # All three evil ones should agree as TP; both benign should be TN.
    assert m["true_positive"] == 3
    assert m["true_negative"] == 2
    assert m["false_positive"] == 0
    assert m["false_negative"] == 0


def test_compute_agreement_metrics_perfect(sample_records: list[AgentRecord]) -> None:
    judgments = [
        Judgment("benign-1", "alice", "benign", 5),
        Judgment("benign-2", "alice", "benign", 5),
        Judgment("evil-1", "alice", "suspicious", 5),
        Judgment("evil-2", "alice", "suspicious", 5),
        Judgment("evil-3", "alice", "suspicious", 5),
    ]
    metrics = HumanEvalFramework.compute_agreement_metrics(sample_records, judgments)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_compute_agreement_metrics_handles_disagreement(
    sample_records: list[AgentRecord],
) -> None:
    # Human says evil-1 is benign (false negative for SybilCore).
    judgments = [
        Judgment("benign-1", "alice", "benign", 5),
        Judgment("evil-1", "alice", "benign", 5),  # disagree
        Judgment("evil-2", "alice", "suspicious", 5),
    ]
    metrics = HumanEvalFramework.compute_agreement_metrics(sample_records, judgments)
    assert metrics["matrix"]["false_positive"] == 1  # evil-1: human=benign, sybil=suspicious
    assert metrics["matrix"]["true_positive"] == 1
    assert metrics["matrix"]["true_negative"] == 1


def test_compute_kappa_two_raters() -> None:
    judgments = [
        Judgment("a", "alice", "suspicious", 5),
        Judgment("a", "bob", "suspicious", 5),
        Judgment("b", "alice", "benign", 5),
        Judgment("b", "bob", "benign", 5),
        Judgment("c", "alice", "suspicious", 5),
        Judgment("c", "bob", "benign", 5),
    ]
    result = HumanEvalFramework.compute_kappa(judgments)
    assert result["n"] == 3
    # Two of three agree, so kappa should be > 0 but < 1.
    assert 0.0 <= result["kappa"] < 1.0


def test_compute_kappa_single_rater_returns_zero() -> None:
    judgments = [
        Judgment("a", "alice", "suspicious", 5),
        Judgment("b", "alice", "benign", 5),
    ]
    result = HumanEvalFramework.compute_kappa(judgments)
    assert result["kappa"] == 0.0
    assert result["raters"] == 1


def test_majority_label_tiebreak_prefers_suspicious() -> None:
    j = [
        Judgment("a", "r1", "suspicious", 5),
        Judgment("a", "r2", "benign", 5),
    ]
    assert _majority_label(j) == "suspicious"


# ── AI judge ────────────────────────────────────────────────────────


def test_ai_judge_fallback_when_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    judge = AIJudge(api_key=None)
    # _make_record sets brain deception = coefficient/5 = 24, plus we add
    # an indicator so the heuristic fires.
    record = AgentRecord(
        agent_id="foo",
        source=SOURCE_ARCHETYPE,
        ground_truth="malicious",
        sybilcore_coefficient=120.0,
        sybilcore_tier="clouded",
        sybilcore_suspicious=True,
        brain_scores={"deception": 30.0, "intent_drift": 12.0, "social_graph": 5.0},
        events=[],
        summary={"indicators": {"deception": ["fake claim"]}},
    )
    verdict = judge.judge(record)
    assert isinstance(verdict, AIJudgeVerdict)
    assert verdict.suspicious is True
    assert "fallback_heuristic" in verdict.rationale


def test_ai_judge_fallback_benign_for_low_score(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    judge = AIJudge(api_key=None)
    record = AgentRecord(
        agent_id="quiet",
        source=SOURCE_MOLTBOOK,
        ground_truth="unknown",
        sybilcore_coefficient=2.0,
        sybilcore_tier="clear",
        sybilcore_suspicious=False,
        brain_scores={"deception": 0.0, "intent_drift": 1.0},
        events=[],
        summary={},
    )
    verdict = judge.judge(record)
    assert verdict.suspicious is False


def test_ai_judge_to_judgment_label_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    judge = AIJudge(api_key=None)
    record = AgentRecord(
        agent_id="evil",
        source=SOURCE_ARCHETYPE,
        ground_truth="malicious",
        sybilcore_coefficient=200.0,
        sybilcore_tier="lethal",
        sybilcore_suspicious=True,
        brain_scores={"deception": 80.0, "intent_drift": 30.0, "social_graph": 25.0, "compromise": 40.0},
        events=[],
        summary={},
    )
    verdict = judge.judge(record)
    j = verdict.to_judgment(rater_id="ai_judge")
    assert j.label == "suspicious"
    assert j.rater_id == "ai_judge"
    assert 1 <= j.confidence <= 5


def test_ai_judge_parse_verdict_handles_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    judge = AIJudge(api_key=None)
    record = AgentRecord(
        agent_id="x",
        source=SOURCE_ARCHETYPE,
        ground_truth="malicious",
        sybilcore_coefficient=120.0,
        sybilcore_tier="clouded",
        sybilcore_suspicious=True,
        brain_scores={"deception": 30.0, "intent_drift": 12.0, "social_graph": 5.0},
        events=[],
        summary={},
    )
    verdict = judge._parse_verdict(record, "not json at all")  # noqa: SLF001
    assert "fallback_heuristic" in verdict.rationale


def test_summarize_run_builds_per_source_table(sample_records: list[AgentRecord]) -> None:
    verdicts = [
        AIJudgeVerdict("benign-1", suspicious=False, score=10.0, confidence=4, rationale="ok", raw_response=""),
        AIJudgeVerdict("evil-1", suspicious=True, score=85.0, confidence=5, rationale="bad", raw_response=""),
    ]
    summary = summarize_run(sample_records, verdicts)
    assert summary["agents_judged"] == 2
    assert SOURCE_MOLTBOOK in summary["by_source"]
    assert SOURCE_ARCHETYPE in summary["by_source"]
    assert summary["by_source"][SOURCE_MOLTBOOK]["agents"] == 1
    assert summary["by_source"][SOURCE_ARCHETYPE]["ai_suspicious"] == 1


# ── Web UI store ────────────────────────────────────────────────────


def test_label_store_upsert_and_export(tmp_path: Path) -> None:
    from sybilcore.evaluation.web_ui import LabelStore

    store = LabelStore(tmp_path / "labels.db")
    store.upsert("a1", "alice", "suspicious", 5, "looks bad")
    store.upsert("a1", "alice", "benign", 4, "changed mind")  # upsert should overwrite
    store.upsert("a2", "bob", "suspicious", 3, "")
    rows = store.export()
    assert len(rows) == 2
    a1 = next(r for r in rows if r["agent_id"] == "a1")
    assert a1["label"] == "benign"
    assert a1["confidence"] == 4
    labeled = store.labeled_agent_ids("alice")
    assert labeled == {"a1"}
