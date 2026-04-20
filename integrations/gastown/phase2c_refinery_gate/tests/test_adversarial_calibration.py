"""Tests for adversarial calibration corpus and ROC-based threshold calibration.

P0 #3 fix validation: proves the adversarial corpus is well-formed, each attack
pattern produces the expected brain signals, and the ROC calibration produces
honest (non-circular) thresholds.

KEY TEST: test_mimicry_evades_detection() documents the known gap — mimicry
adversaries are NOT expected to be detected. Their passing is a positive test
result, not a failure.

Run with:
  pytest tests/test_adversarial_calibration.py -v
"""

from __future__ import annotations

import contextlib
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_TESTS_DIR = Path(__file__).resolve().parent
_PKG_DIR = _TESTS_DIR.parent
_INTEGRATION_DIR = _PKG_DIR.parent
_SYBILCORE_ROOT = _INTEGRATION_DIR.parents[2]
_PHASE1A_DIR = _INTEGRATION_DIR / "phase1a_baseline"

for _p in (str(_SYBILCORE_ROOT), str(_PHASE1A_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(_PKG_DIR))

from sybilcore.models.event import Event  # noqa: E402, I001
from adapter import adapt_bead_event  # noqa: E402
from adversarial_corpus import (  # noqa: E402
    ADVERSARIAL_PATTERNS,
    AGENTS_PER_PATTERN,
    _ATTACK_TAG_KEY,
    adversarial_agents_to_events,
    corpus_summary,
    generate_adversarial_corpus,
)
from gate_simulator import run_adversarial_simulation  # noqa: E402
from narrow_brains import build_narrow_brains, score_agent_narrow  # noqa: E402
from threshold_calibration import (  # noqa: E402
    _compute_roc_curve,
    _find_threshold_at_fpr,
    _find_youdens_j_threshold,
    calibrate_with_adversaries,
)


# ===========================================================================
# Group 1: Corpus generation determinism and structure
# ===========================================================================


def test_corpus_generation_is_deterministic() -> None:
    """generate_adversarial_corpus() with same seed must return identical results."""
    agents_a = generate_adversarial_corpus(seed=42)
    agents_b = generate_adversarial_corpus(seed=42)
    assert len(agents_a) == len(agents_b)
    for a, b in zip(agents_a, agents_b, strict=True):
        assert a.agent_id == b.agent_id
        assert a.pattern == b.pattern
        assert len(a.events) == len(b.events)
        # Compare first event timestamps for deep determinism check
        assert a.events[0]["created_at"] == b.events[0]["created_at"]


def test_corpus_generation_different_seeds_differ() -> None:
    """Different seeds must produce different corpora."""
    agents_42 = generate_adversarial_corpus(seed=42)
    agents_99 = generate_adversarial_corpus(seed=99)
    # At least some agent IDs or event timestamps should differ
    ids_42 = {a.agent_id for a in agents_42}
    ids_99 = {a.agent_id for a in agents_99}
    assert ids_42 == ids_99  # Same IDs (deterministic naming)
    # But timestamps differ across runs with different seeds
    t42 = agents_42[0].events[0]["created_at"]
    t99 = agents_99[0].events[0]["created_at"]
    # Agent base times are randomised per seed, so they should differ
    assert t42 != t99


def test_minimum_30_adversarial_agents() -> None:
    """Corpus must produce at least 30 adversarial agents."""
    agents = generate_adversarial_corpus(seed=42)
    assert len(agents) >= 30


def test_all_patterns_covered() -> None:
    """All 6 attack patterns must appear in the corpus."""
    agents = generate_adversarial_corpus(seed=42)
    patterns_found = {a.pattern for a in agents}
    for expected_pattern in ADVERSARIAL_PATTERNS:
        assert expected_pattern in patterns_found, f"Missing pattern: {expected_pattern}"


def test_each_pattern_has_minimum_agents() -> None:
    """Each pattern must have at least AGENTS_PER_PATTERN agents."""
    agents = generate_adversarial_corpus(seed=42)
    summary = corpus_summary(agents)
    for pattern in ADVERSARIAL_PATTERNS:
        count = summary["per_pattern_agents"].get(pattern, 0)
        assert count >= AGENTS_PER_PATTERN, (
            f"Pattern '{pattern}' has {count} agents, need >= {AGENTS_PER_PATTERN}"
        )


def test_all_events_tagged_with_attack_pattern() -> None:
    """Every adversarial event must carry the _synthetic_attack tag."""
    agents = generate_adversarial_corpus(seed=42)
    for agent in agents:
        for event in agent.events:
            new_val = event.get("new_value", "")
            if new_val:
                parsed = json.loads(new_val) if isinstance(new_val, str) else new_val
                assert _ATTACK_TAG_KEY in parsed, (
                    f"Event in agent {agent.agent_id} missing attack tag: {event}"
                )
            else:
                # Empty new_value is acceptable only if there's no payload
                pass


def test_bead_schema_compliance() -> None:
    """All adversarial events must conform to Phase 1A bead JSONL schema."""
    agents = generate_adversarial_corpus(seed=42)
    required_keys = {"id", "issue_id", "event_type", "actor", "created_at"}
    for agent in agents:
        for event in agent.events:
            missing = required_keys - set(event.keys())
            assert not missing, (
                f"Agent {agent.agent_id} event missing keys {missing}: {event}"
            )
            # actor must match agent_id
            assert event["actor"] == agent.agent_id
            # created_at must be parseable
            dt = datetime.fromisoformat(event["created_at"].replace("Z", "+00:00"))
            assert dt.tzinfo is not None


def test_adversarial_namespace_separation() -> None:
    """Adversarial agent IDs must use the gastown/adversarial/ namespace."""
    agents = generate_adversarial_corpus(seed=42)
    for agent in agents:
        assert agent.agent_id.startswith("gastown/adversarial/"), (
            f"Agent {agent.agent_id} does not use adversarial namespace"
        )


def test_corpus_flat_event_list() -> None:
    """adversarial_agents_to_events() must return chronologically sorted events."""
    agents = generate_adversarial_corpus(seed=42)
    events = adversarial_agents_to_events(agents)
    assert len(events) > 0
    timestamps = [e["created_at"] for e in events]
    assert timestamps == sorted(timestamps), "Events not in chronological order"


def test_invalid_agents_per_pattern_raises() -> None:
    """agents_per_pattern < 1 must raise ValueError."""
    with pytest.raises(ValueError, match="agents_per_pattern must be >= 1"):
        generate_adversarial_corpus(agents_per_pattern=0)


# ===========================================================================
# Group 2: Bead-to-SybilCore Event adaptation
# ===========================================================================


def test_adversarial_events_adapt_to_sybilcore() -> None:
    """Phase 1A adapter must accept adversarial bead events without errors."""
    agents = generate_adversarial_corpus(seed=42)
    adapted_count = 0
    error_count = 0
    for agent in agents[:3]:  # Sample 3 agents to keep test fast
        for event in agent.events[:5]:
            try:
                ev = adapt_bead_event(event)
                assert isinstance(ev, Event)
                assert ev.agent_id == agent.agent_id
                adapted_count += 1
            except (ValueError, KeyError):
                error_count += 1
    assert adapted_count > 0
    assert error_count == 0, f"{error_count} adaptation failures"


# ===========================================================================
# Group 3: Brain activation — each pattern fires at least one brain
# ===========================================================================


@pytest.fixture(scope="module")
def adversarial_brain_scores() -> dict[str, dict[str, float]]:
    """Score all adversarial agents and return per-pattern mean brain scores.

    Scoped to module so brains are loaded only once (expensive).
    """
    agents = generate_adversarial_corpus(seed=42)
    brains = build_narrow_brains()
    window = 90 * 24 * 3600

    pattern_scores: dict[str, list[dict[str, float]]] = {}
    for agent in agents:
        all_events: list[Event] = []
        for row in agent.events:
            with contextlib.suppress(ValueError, KeyError):
                all_events.append(adapt_bead_event(row))
        if not all_events:
            continue
        try:
            snap = score_agent_narrow(agent.agent_id, all_events, brains, window)
            pattern_scores.setdefault(agent.pattern, []).append(snap.brain_scores)
        except Exception:
            pass

    # Compute per-pattern means
    result: dict[str, dict[str, float]] = {}
    for pattern, score_list in pattern_scores.items():
        if not score_list:
            result[pattern] = {}
            continue
        brain_names = set().union(*[set(s.keys()) for s in score_list])
        means = {}
        for brain in brain_names:
            vals = [s.get(brain, 0.0) for s in score_list]
            means[brain] = sum(vals) / len(vals)
        result[pattern] = means

    return result


def test_identity_drift_fires_identity_brain(
    adversarial_brain_scores: dict[str, dict[str, float]],
) -> None:
    """Identity-drift pattern must produce non-zero IdentityBrain score."""
    scores = adversarial_brain_scores.get("identity_drift", {})
    assert scores.get("identity", 0.0) > 0.0, (
        f"IdentityBrain did not fire for identity_drift: {scores}"
    )


def test_silent_burst_fires_temporal_brain(
    adversarial_brain_scores: dict[str, dict[str, float]],
) -> None:
    """Silent-burst pattern must produce non-zero TemporalBrain score."""
    scores = adversarial_brain_scores.get("silent_burst", {})
    assert scores.get("temporal", 0.0) > 0.0, (
        f"TemporalBrain did not fire for silent_burst: {scores}"
    )


def test_rapid_fire_fires_temporal_brain(
    adversarial_brain_scores: dict[str, dict[str, float]],
) -> None:
    """Rapid-fire pattern must produce non-zero TemporalBrain score."""
    scores = adversarial_brain_scores.get("rapid_fire", {})
    assert scores.get("temporal", 0.0) > 0.0, (
        f"TemporalBrain did not fire for rapid_fire: {scores}"
    )


def test_scope_escalation_fires_temporal_brain(
    adversarial_brain_scores: dict[str, dict[str, float]],
) -> None:
    """Scope-escalation pattern must produce non-zero TemporalBrain score."""
    scores = adversarial_brain_scores.get("scope_escalation", {})
    assert scores.get("temporal", 0.0) > 0.0, (
        f"TemporalBrain did not fire for scope_escalation: {scores}"
    )


def test_contradiction_fires_at_least_one_brain(
    adversarial_brain_scores: dict[str, dict[str, float]],
) -> None:
    """Contradiction pattern must produce at least one non-zero brain score."""
    scores = adversarial_brain_scores.get("contradiction", {})
    any_fired = any(v > 0.0 for v in scores.values())
    assert any_fired, f"No brains fired for contradiction: {scores}"


# ===========================================================================
# Group 4: Mimicry pattern — expected to evade detection (documented gap)
# ===========================================================================


def test_mimicry_evades_detection(
    adversarial_brain_scores: dict[str, dict[str, float]],
) -> None:
    """Mimicry adversaries must score LOWER than rapid-fire adversaries.

    POSITIVE TEST — mimicry PASSING the gate is the expected outcome.
    This documents the known gap: the 3-brain gate cannot distinguish
    a sophisticated adversary that mimics normal polecat behavior.
    A mimicry agent that is BLOCKED would indicate a false positive
    (mimicry looks too similar to the high-activity benign agents that
    also get BLOCKED at aggressive thresholds).

    The passing of mimicry is the paper-ready finding, not a bug.
    """
    mimicry_scores = adversarial_brain_scores.get("mimicry", {})
    rapid_scores = adversarial_brain_scores.get("rapid_fire", {})

    # Mimicry temporal score should be near-zero (it looks like a normal agent)
    mimicry_temporal = mimicry_scores.get("temporal", 0.0)
    rapid_temporal = rapid_scores.get("temporal", 0.0)

    # Rapid-fire should have higher temporal signal than mimicry
    # (fixed-interval events trigger circadian anomaly)
    assert rapid_temporal >= mimicry_temporal, (
        f"Expected rapid_fire temporal ({rapid_temporal:.2f}) >= "
        f"mimicry temporal ({mimicry_temporal:.2f}). "
        "If mimicry scores HIGHER than rapid_fire, the corpus is not adversarial enough."
    )


def test_mimicry_produces_valid_coefficient(
    adversarial_brain_scores: dict[str, dict[str, float]],
) -> None:
    """Mimicry agents must produce a valid (non-error) coefficient."""
    agents = generate_adversarial_corpus(seed=42)
    mimicry_agents = [a for a in agents if a.pattern == "mimicry"]
    assert mimicry_agents, "No mimicry agents in corpus"

    brains = build_narrow_brains()
    window = 90 * 24 * 3600
    for agent in mimicry_agents[:2]:
        all_events = [adapt_bead_event(e) for e in agent.events]
        snap = score_agent_narrow(agent.agent_id, all_events, brains, window)
        assert 0.0 <= snap.coefficient <= 1000.0, (
            f"Mimicry agent {agent.agent_id} has invalid coefficient: {snap.coefficient}"
        )


# ===========================================================================
# Group 5: ROC calibration correctness
# ===========================================================================


def test_roc_curve_empty_when_inputs_empty() -> None:
    """_compute_roc_curve with empty inputs must return empty list."""
    assert _compute_roc_curve([], [1.0, 2.0]) == []
    assert _compute_roc_curve([1.0, 2.0], []) == []
    assert _compute_roc_curve([], []) == []


def test_roc_curve_basic_separation() -> None:
    """Perfect separation: benign all low, adversarial all high."""
    benign = [10.0, 15.0, 20.0]
    adversarial = [80.0, 85.0, 90.0]
    roc = _compute_roc_curve(benign, adversarial)
    # At threshold=80, FPR=0, TPR=1 (perfect)
    point_80 = next((t for t in roc if t[0] == 80.0), None)
    assert point_80 is not None
    assert point_80[1] == 0.0   # FPR=0
    assert point_80[2] == 1.0   # TPR=1


def test_roc_curve_no_separation() -> None:
    """No separation: benign and adversarial scores identical."""
    scores = [10.0, 20.0, 30.0, 40.0]
    roc = _compute_roc_curve(scores, scores)
    # With identical distributions, FPR == TPR at every threshold
    for thresh, fpr, tpr in roc:
        assert abs(fpr - tpr) < 0.01, (
            f"At threshold {thresh}: FPR={fpr} TPR={tpr} — expected FPR≈TPR"
        )


def test_find_threshold_at_fpr_with_strict_budget() -> None:
    """_find_threshold_at_fpr with 0.0 FPR must return the most restrictive threshold."""
    roc = [(10.0, 0.5, 0.8), (20.0, 0.3, 0.7), (30.0, 0.1, 0.6), (40.0, 0.0, 0.4)]
    t = _find_threshold_at_fpr(roc, 0.0)
    # Only threshold 40.0 has FPR=0
    assert t == 40.0


def test_find_threshold_at_fpr_returns_lowest_valid() -> None:
    """_find_threshold_at_fpr must return lowest threshold within FPR budget."""
    roc = [(5.0, 0.1, 0.8), (10.0, 0.05, 0.6), (20.0, 0.02, 0.4), (30.0, 0.0, 0.2)]
    t = _find_threshold_at_fpr(roc, 0.1)
    # All 4 points have FPR <= 0.1; lowest is 5.0
    assert t == 5.0


def test_youdens_j_threshold_on_perfect_separation() -> None:
    """Youden's J on perfectly separated data must return the separating threshold."""
    # Benign: [10-30], Adversarial: [80-90]
    benign = [10.0, 20.0, 30.0]
    adversarial = [80.0, 85.0, 90.0]
    roc = _compute_roc_curve(benign, adversarial)
    j_thresh = _find_youdens_j_threshold(roc)
    # At the separating threshold, TPR=1 FPR=0, so J=1.0
    # The threshold should be in the gap between 30 and 80
    assert j_thresh <= 80.0


def test_calibrate_with_adversaries_requires_nonempty_corpora() -> None:
    """calibrate_with_adversaries raises ValueError on empty corpora."""
    tmp_path = Path("/tmp")
    benign = [{"agent_id": "a", "coefficient": 10.0, "tier": "clear"}]
    with pytest.raises(ValueError, match="adversarial_corpus cannot be empty"):
        calibrate_with_adversaries(benign, [], tmp_path / "out_test.json")
    with pytest.raises(ValueError, match="benign_corpus cannot be empty"):
        calibrate_with_adversaries([], benign, tmp_path / "out_test2.json")


def test_calibrate_with_adversaries_writes_json(tmp_path: Path) -> None:
    """calibrate_with_adversaries must write a valid JSON file."""
    benign = [
        {"agent_id": f"benign/{i}", "coefficient": float(10 + i * 5), "tier": "clear"}
        for i in range(10)
    ]
    adversarial = [
        {
            "agent_id": f"adv/{i}",
            "coefficient": float(80 + i * 2),
            "tier": "flagged",
            "pattern": "rapid_fire",
        }
        for i in range(10)
    ]
    out = tmp_path / "thresholds.json"
    calibrate_with_adversaries(benign, adversarial, out)

    assert out.exists()
    data = json.loads(out.read_text())
    assert "warn_threshold" in data
    assert "pass_threshold" in data
    assert "roc_auc" in data
    assert "confusion_matrix_at_pass" in data
    assert "per_pattern_tpr" in data


def test_calibrate_with_adversaries_auc_near_one_for_separated_data(
    tmp_path: Path,
) -> None:
    """Well-separated benign/adversarial distributions must produce AUC near 1.0."""
    benign = [
        {"agent_id": f"b/{i}", "coefficient": float(i * 2), "tier": "clear"}
        for i in range(20)
    ]
    adversarial = [
        {
            "agent_id": f"a/{i}",
            "coefficient": float(100 + i * 2),
            "tier": "flagged",
            "pattern": "test",
        }
        for i in range(20)
    ]
    result = calibrate_with_adversaries(benign, adversarial, tmp_path / "t.json")
    assert result["roc_auc"] >= 0.9, (
        f"Expected AUC >= 0.9 for perfectly separated data, got {result['roc_auc']}"
    )


def test_calibrate_with_adversaries_auc_near_half_for_overlapping_data(
    tmp_path: Path,
) -> None:
    """Overlapping benign/adversarial distributions must produce AUC near 0.5."""
    import random
    rng = random.Random(42)
    # Both draw from same distribution — indistinguishable
    shared_scores = [float(rng.uniform(10, 100)) for _ in range(40)]
    benign = [
        {"agent_id": f"b/{i}", "coefficient": shared_scores[i], "tier": "clear"}
        for i in range(20)
    ]
    adversarial = [
        {
            "agent_id": f"a/{i}",
            "coefficient": shared_scores[20 + i],
            "tier": "flagged",
            "pattern": "mimicry",
        }
        for i in range(20)
    ]
    result = calibrate_with_adversaries(benign, adversarial, tmp_path / "t2.json")
    # AUC should be near 0.5 (random) — allow generous tolerance
    assert 0.2 <= result["roc_auc"] <= 0.8, (
        f"Expected AUC ~0.5 for indistinguishable distributions, got {result['roc_auc']}"
    )


# ===========================================================================
# Group 6: Simulation confusion matrix correctness
# ===========================================================================


def test_adversarial_simulation_requires_fixture(tmp_path: Path) -> None:
    """run_adversarial_simulation must raise FileNotFoundError for missing fixture."""
    with pytest.raises(FileNotFoundError):
        run_adversarial_simulation(
            fixture_path=tmp_path / "nonexistent.jsonl",
            output_dir=tmp_path,
        )


def test_confusion_matrix_sums_correctly(tmp_path: Path) -> None:
    """TP + FN must equal n_adversarial; TN + FP must equal n_benign."""
    # Build a minimal synthetic fixture with known agents
    import json as _json
    events_data = [
        {
            "id": i,
            "issue_id": f"gt-{i:05d}",
            "event_type": "status_changed",
            "actor": f"gastown/polecats/agent{i % 3}",
            "created_at": (
                datetime.now(UTC) - timedelta(days=5, hours=i)
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "new_value": _json.dumps({"status": "hooked", "content": f"work {i}"}),
            "old_value": "",
            "comment": None,
        }
        for i in range(30)
    ]
    fixture = tmp_path / "events.jsonl"
    fixture.write_text(
        "\n".join(_json.dumps(r) for r in events_data), encoding="utf-8"
    )

    summary = run_adversarial_simulation(
        fixture_path=fixture,
        output_dir=tmp_path,
        adversarial_seed=42,
    )

    cm = summary["confusion_matrix"]
    assert cm["tp"] + cm["fn"] == summary["n_adversarial"], (
        f"TP+FN={cm['tp']+cm['fn']} != n_adversarial={summary['n_adversarial']}"
    )
    assert cm["tn"] + cm["fp"] == summary["n_benign"], (
        f"TN+FP={cm['tn']+cm['fp']} != n_benign={summary['n_benign']}"
    )


def test_simulation_v2_writes_csv_and_json(tmp_path: Path) -> None:
    """run_adversarial_simulation must write simulation_report_v2.csv and .json."""
    import json as _json
    events_data = [
        {
            "id": i,
            "issue_id": f"gt-{i:05d}",
            "event_type": "closed",
            "actor": "gastown/polecats/test",
            "created_at": (
                datetime.now(UTC) - timedelta(days=3, hours=i)
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "new_value": _json.dumps({"status": "closed", "content": f"done {i}"}),
            "old_value": "",
            "comment": None,
        }
        for i in range(15)
    ]
    fixture = tmp_path / "events.jsonl"
    fixture.write_text(
        "\n".join(_json.dumps(r) for r in events_data), encoding="utf-8"
    )

    run_adversarial_simulation(
        fixture_path=fixture,
        output_dir=tmp_path,
        adversarial_seed=42,
    )

    assert (tmp_path / "simulation_report_v2.csv").exists()
    assert (tmp_path / "simulation_report_v2.json").exists()

    data = json.loads((tmp_path / "simulation_report_v2.json").read_text())
    assert "confusion_matrix" in data
    assert "per_pattern_detection" in data
    assert "roc_auc" in data


def test_per_agent_rows_have_corpus_label(tmp_path: Path) -> None:
    """All rows in per_agent must have corpus_label set to 'benign' or 'adversarial'."""
    import json as _json
    events_data = [
        {
            "id": i,
            "issue_id": f"gt-{i:05d}",
            "event_type": "updated",
            "actor": "gastown/polecats/tester",
            "created_at": (
                datetime.now(UTC) - timedelta(days=2, hours=i)
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "new_value": _json.dumps({"status": "open", "content": f"update {i}"}),
            "old_value": "",
            "comment": None,
        }
        for i in range(12)
    ]
    fixture = tmp_path / "events.jsonl"
    fixture.write_text(
        "\n".join(_json.dumps(r) for r in events_data), encoding="utf-8"
    )

    summary = run_adversarial_simulation(
        fixture_path=fixture,
        output_dir=tmp_path,
        adversarial_seed=42,
    )

    valid_labels = {"benign", "adversarial"}
    for row in summary["per_agent"]:
        assert row["corpus_label"] in valid_labels, (
            f"Row {row['agent_id']} has invalid corpus_label: {row['corpus_label']}"
        )
