"""V5.1 re-calibration on the FULL eligible Moltbook population.

Extends `recalibrate_v5_real_moltbook.py` (Agent 7) by:

1. Using ALL eligible Moltbook authors (>=2 posts each, expected ~1,297)
   instead of the 500-author cap.
2. Splitting 80/20 with a fixed seed (``SPLIT_SEED = 20260408``) by
   author_id for a defensible out-of-sample holdout. The optimizer
   only sees TRAIN negatives; TEST metrics are the honest number.
3. Doubling the gradient-free search budget (100 random + 50 local)
   for tighter convergence — the v5 run plateaued after ~50
   evaluations but the wider budget gives us more stable local minima.
4. Reporting BOTH train and test metrics so we can measure
   train/test gap and decide whether the weights generalize.

Positive-set handling:
    The v4/v5 positive set (alignment agents + archetype agents) is
    small (~103 agents) and synthetically generated — it doesn't
    suffer from the same generalization concerns as the real
    negatives. Per the task brief, we reuse the full positive set in
    both train and test rather than splitting it, because:
        (a) splitting 103 positives gives ~20 test positives, which
            is too few for stable F1 / recall;
        (b) the calibration question we're answering is specifically
            about negative-class drift — positives are the clean
            side of the experiment.
    This biases TEST F1 slightly *upward* (positives are seen during
    training too), so interpret the train/test gap as a *lower bound*
    on true overfit. If the gap is already > 0.02 with this
    positive-sharing setup, the real overfit is even worse.

Outputs:
    experiments/calibration_v5_1_full_moltbook.json

Runtime estimate:
    - Precompute: ~35s (v5 run) × (1400 / 603) ≈ 81s.
    - Search (100 random + 50 local): ~2x v5's ~17s ≈ 34s.
    - Total: ~115s.  Well under the 30 min budget.
"""

from __future__ import annotations

import collections
import json
import logging
import random
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

# Allow running this script from anywhere in the repo.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sybilcore.analysis.calibration import (  # noqa: E402
    CalibrationCorpus,
    CalibrationMetrics,
    LabeledAgent,
    WeightCalibrator,
)
from sybilcore.analysis.run_calibration import (  # noqa: E402
    _metrics_dict,
    _source_counts,
    build_positives,
    threshold_sweep,
)
from sybilcore.brains.compromise import CompromiseBrain  # noqa: E402
from sybilcore.brains.contrastive import ContrastiveEmbeddingBrain  # noqa: E402
from sybilcore.brains.deception import DeceptionBrain  # noqa: E402
from sybilcore.brains.economic import EconomicBrain  # noqa: E402
from sybilcore.brains.embedding import EmbeddingBrain  # noqa: E402
from sybilcore.brains.fidelity import FidelityBrain  # noqa: E402
from sybilcore.brains.identity import IdentityBrain  # noqa: E402
from sybilcore.brains.intent_drift import IntentDriftBrain  # noqa: E402
from sybilcore.brains.neuro import NeuroBrain  # noqa: E402
from sybilcore.brains.resource_hoarding import ResourceHoardingBrain  # noqa: E402
from sybilcore.brains.semantic import SemanticBrain  # noqa: E402
from sybilcore.brains.silence import SilenceBrain  # noqa: E402
from sybilcore.brains.social_graph import SocialGraphBrain  # noqa: E402
from sybilcore.brains.swarm_detection import SwarmDetectionBrain  # noqa: E402
from sybilcore.brains.temporal import TemporalBrain  # noqa: E402
from sybilcore.models.event import Event, EventType  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("recalibrate_v5_1")


# ── Configuration ──────────────────────────────────────────────────

# Doubled search budget vs Agent 7 (50 random + 25 local).
RANDOM_ITER = 100
LOCAL_ITER = 50

# No negative cap — use every eligible Moltbook author.
NEGATIVE_TARGET: int | None = None
POSITIVE_TARGET = 200
MIN_POSTS_PER_AUTHOR = 2

# Reuse the v5 sampling seed so the union of train+test covers the
# same author population as Agent 7 would have picked had there been
# no cap. (With NEGATIVE_TARGET=None this seed only controls the
# shuffle order used for the 80/20 split below.)
SAMPLING_SEED = 42

# Fixed seed for the train/test split so future replays are
# reproducible. Deliberately different from SAMPLING_SEED so the
# split is independent of the ingest order.
SPLIT_SEED = 20260408
TEST_FRACTION = 0.2

MOLTBOOK_PATH = REPO_ROOT / "data" / "moltbook_posts.jsonl"
OUT_PATH = REPO_ROOT / "experiments" / "calibration_v5_1_full_moltbook.json"
V4_CALIBRATION_PATH = REPO_ROOT / "experiments" / "calibration_v4_20260407_101019.json"
V5_CALIBRATION_PATH = REPO_ROOT / "experiments" / "calibration_v5_real_moltbook.json"


# ── Brain ensemble ──────────────────────────────────────────────────

def build_full_brain_ensemble() -> list:
    """Return the full 15-brain ensemble used by the v4 calibration."""
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


# ── Negative-corpus construction from real Moltbook ────────────────

def _post_to_event(post: dict, agent_id: str) -> Event:
    """Rebuild a single Moltbook post as an Event."""
    raw_ts = post.get("created_at")
    if raw_ts:
        try:
            ts = datetime.fromisoformat(raw_ts)
        except ValueError:
            ts = datetime.now(UTC)
    else:
        ts = datetime.now(UTC)

    title = post.get("title", "") or ""
    body = post.get("content", "") or ""
    content = f"post: {title}\n{body}".strip()[:5000]

    return Event(
        event_id=str(uuid4()),
        agent_id=agent_id,
        event_type=EventType.OUTPUT_GENERATED,
        timestamp=ts,
        content=content,
        metadata={
            "moltbook_event": "post",
            "submolt": post.get("submolt", "general"),
            "karma": post.get("score", 0),
            "upvotes": post.get("upvotes", 0),
            "downvotes": post.get("downvotes", 0),
            "post_id": post.get("id", ""),
        },
        source="moltbook",
    )


def build_full_moltbook_negatives(
    min_posts: int = MIN_POSTS_PER_AUTHOR,
    target: int | None = NEGATIVE_TARGET,
    seed: int = SAMPLING_SEED,
) -> list[LabeledAgent]:
    """Load the FULL eligible Moltbook author set as LabeledAgents.

    With ``target=None`` every author with ``>=min_posts`` is used.
    With ``target=<int>`` the first ``target`` shuffled authors are
    kept — used only for smoke tests, production runs leave this as
    ``None``.
    """
    if not MOLTBOOK_PATH.exists():
        msg = f"Moltbook data not found: {MOLTBOOK_PATH}"
        raise FileNotFoundError(msg)

    by_author: dict[str, list[dict]] = collections.defaultdict(list)
    with MOLTBOOK_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                post = json.loads(line)
            except json.JSONDecodeError:
                continue
            author = post.get("author")
            if not author:
                continue
            by_author[author].append(post)

    eligible = {a: p for a, p in by_author.items() if len(p) >= min_posts}
    logger.info(
        "Moltbook: %d unique authors, %d with >=%d posts",
        len(by_author),
        len(eligible),
        min_posts,
    )

    # Deterministic shuffle; keep all authors unless target is set.
    rng = random.Random(seed)  # noqa: S311 — non-crypto reproducibility
    authors_sorted = sorted(eligible.keys())
    rng.shuffle(authors_sorted)
    chosen = authors_sorted if target is None else authors_sorted[:target]

    negatives: list[LabeledAgent] = []
    for author in chosen:
        posts = eligible[author]
        events = sorted(
            (_post_to_event(p, author) for p in posts),
            key=lambda e: e.timestamp,
        )
        if not events:
            continue
        negatives.append(
            LabeledAgent(
                agent_id=f"neg-moltbook-{author}",
                events=tuple(events),
                label=0,
                source="moltbook_real",
            )
        )

    logger.info("Built %d real Moltbook negatives (full population)", len(negatives))
    return negatives


# ── Train/test splitting ───────────────────────────────────────────

def split_negatives(
    negatives: list[LabeledAgent],
    test_fraction: float = TEST_FRACTION,
    seed: int = SPLIT_SEED,
) -> tuple[list[LabeledAgent], list[LabeledAgent]]:
    """Deterministic 80/20 split of negatives by agent_id.

    Returns (train_negatives, test_negatives).
    """
    sorted_negs = sorted(negatives, key=lambda a: a.agent_id)
    rng = random.Random(seed)  # noqa: S311 — non-crypto reproducibility
    indices = list(range(len(sorted_negs)))
    rng.shuffle(indices)

    n_test = int(round(len(sorted_negs) * test_fraction))
    test_ids = set(indices[:n_test])
    train_negs = [a for i, a in enumerate(sorted_negs) if i not in test_ids]
    test_negs = [a for i, a in enumerate(sorted_negs) if i in test_ids]
    logger.info(
        "Split: %d train negs, %d test negs (fraction=%.2f, seed=%d)",
        len(train_negs),
        len(test_negs),
        test_fraction,
        seed,
    )
    return train_negs, test_negs


def subset_corpus(
    full: CalibrationCorpus,
    agents: list[LabeledAgent],
) -> CalibrationCorpus:
    """Build a corpus view of ``agents`` that reuses ``full.brain_scores``.

    The cached brain scores are keyed by agent_id, so any subset of
    the parent corpus's agents can share the cache without re-running
    the brains.
    """
    subset = CalibrationCorpus(agents=list(agents))
    for a in agents:
        subset.brain_scores[a.agent_id] = full.brain_scores[a.agent_id]
    return subset


# ── Reporting helpers ──────────────────────────────────────────────

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        logger.warning("Reference file not found: %s", path)
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse %s: %s", path, exc)
        return None


def _metrics_delta(a: CalibrationMetrics, b: CalibrationMetrics) -> dict:
    """Return ``a - b`` per field for the headline metrics."""
    return {
        "f1": a.f1 - b.f1,
        "precision": a.precision - b.precision,
        "recall": a.recall - b.recall,
        "auc": a.auc - b.auc,
        "fpr": a.fpr - b.fpr,
    }


def _pretty_print_gap(
    train: CalibrationMetrics,
    test: CalibrationMetrics,
    train_threshold: float,
    test_threshold: float,
) -> None:
    bar = "─" * 70
    print()
    print(bar)
    print(f"{'Metric':<22}{'Train':>14}{'Test':>14}{'Gap (train−test)':>18}")
    print(bar)
    for name, t_val, e_val in (
        ("F1", train.f1, test.f1),
        ("Precision", train.precision, test.precision),
        ("Recall", train.recall, test.recall),
        ("AUC", train.auc, test.auc),
        ("FPR", train.fpr, test.fpr),
    ):
        gap = t_val - e_val
        print(f"{name:<22}{t_val:>14.4f}{e_val:>14.4f}{gap:>+18.4f}")
    print(bar)
    print(
        f"{'Best threshold':<22}{train_threshold:>14.2f}{test_threshold:>14.2f}"
    )
    print(bar)


# ── Main ───────────────────────────────────────────────────────────

def main() -> int:
    t0 = time.perf_counter()

    logger.info("=" * 70)
    logger.info("V5.1 re-calibration — full Moltbook population + 80/20 holdout")
    logger.info("=" * 70)

    # Build positives (same as v4/v5) and full negative set.
    positives = build_positives(target=POSITIVE_TARGET)
    all_negatives = build_full_moltbook_negatives(
        min_posts=MIN_POSTS_PER_AUTHOR,
        target=NEGATIVE_TARGET,
        seed=SAMPLING_SEED,
    )

    train_negatives, test_negatives = split_negatives(
        all_negatives,
        test_fraction=TEST_FRACTION,
        seed=SPLIT_SEED,
    )

    logger.info(
        "Corpus totals: %d positives, %d train negs, %d test negs",
        len(positives),
        len(train_negatives),
        len(test_negatives),
    )

    brains = build_full_brain_ensemble()
    logger.info(
        "Brain ensemble: %d brains (%s)",
        len(brains),
        ", ".join(b.name for b in brains),
    )
    calibrator = WeightCalibrator(seed=42, brains=brains)

    # Load ONCE with all agents — precompute brain scores for
    # positives + train negs + test negs simultaneously.
    logger.info("Precomputing brain scores for full corpus (one-shot)")
    full_corpus = calibrator.load_corpus(
        positives,
        [*train_negatives, *test_negatives],
    )
    precompute_elapsed = time.perf_counter() - t0
    logger.info(
        "Brain scores cached for %d agents in %.1fs",
        len(full_corpus),
        precompute_elapsed,
    )

    # Train corpus: positives + train negatives only. The optimizer
    # only ever touches this subset.
    train_corpus = subset_corpus(
        full_corpus,
        [*positives, *train_negatives],
    )
    # Test corpus: positives + test negatives. Evaluation only.
    test_corpus = subset_corpus(
        full_corpus,
        [*positives, *test_negatives],
    )

    # ── Baselines on TRAIN ────────────────────────────────────────
    default_weights = calibrator.default_weights()
    _, train_default_metrics, train_default_threshold = (
        calibrator.objective_with_threshold_search(default_weights, train_corpus)
    )
    logger.info(
        "DEFAULT (train) → F1=%.4f Prec=%.4f Recall=%.4f FPR=%.4f thresh=%.1f",
        train_default_metrics.f1,
        train_default_metrics.precision,
        train_default_metrics.recall,
        train_default_metrics.fpr,
        train_default_threshold,
    )

    # ── Hybrid search on TRAIN only ───────────────────────────────
    logger.info(
        "Hybrid search on TRAIN: %d random + %d local iterations",
        RANDOM_ITER,
        LOCAL_ITER,
    )
    search_t0 = time.perf_counter()
    result = calibrator.hybrid_search(
        train_corpus,
        random_iter=RANDOM_ITER,
        local_iter=LOCAL_ITER,
    )
    search_elapsed = time.perf_counter() - search_t0
    logger.info(
        "V5.1 OPTIMAL (train) → F1=%.4f Prec=%.4f Recall=%.4f FPR=%.4f thresh=%.1f (%.1fs)",
        result.metrics.f1,
        result.metrics.precision,
        result.metrics.recall,
        result.metrics.fpr,
        result.threshold,
        search_elapsed,
    )

    # ── Evaluate on TEST at the TRAIN-selected threshold ──────────
    # Fair evaluation: lock in the train-derived threshold, don't
    # cherry-pick a better one on the test set.
    test_coefficients = calibrator.score_with_weights(result.weights, test_corpus)
    test_metrics_fixed_thresh = calibrator.compute_metrics(
        test_coefficients,
        test_corpus,
        threshold=result.threshold,
    )
    logger.info(
        "V5.1 OPTIMAL (test, train thresh=%.1f) → F1=%.4f Prec=%.4f Recall=%.4f FPR=%.4f",
        result.threshold,
        test_metrics_fixed_thresh.f1,
        test_metrics_fixed_thresh.precision,
        test_metrics_fixed_thresh.recall,
        test_metrics_fixed_thresh.fpr,
    )

    # Secondary: also let the test set pick its own best threshold so
    # we can report the ceiling. Not used for the ship/no-ship gate.
    _, test_metrics_free_thresh, test_best_thresh = (
        calibrator.objective_with_threshold_search(result.weights, test_corpus)
    )
    logger.info(
        "V5.1 OPTIMAL (test, best thresh=%.1f) → F1=%.4f",
        test_best_thresh,
        test_metrics_free_thresh.f1,
    )

    # Gap analysis at the train threshold.
    gap = _metrics_delta(result.metrics, test_metrics_fixed_thresh)
    logger.info(
        "Train/Test gap (F1) = %+.4f  (Prec %+.4f, Recall %+.4f, FPR %+.4f)",
        gap["f1"],
        gap["precision"],
        gap["recall"],
        gap["fpr"],
    )

    _pretty_print_gap(
        result.metrics,
        test_metrics_fixed_thresh,
        result.threshold,
        result.threshold,
    )

    # ── Also evaluate DEFAULT on test for comparison ──────────────
    test_default_coefficients = calibrator.score_with_weights(
        default_weights, test_corpus
    )
    test_default_metrics = calibrator.compute_metrics(
        test_default_coefficients,
        test_corpus,
        threshold=train_default_threshold,
    )
    logger.info(
        "DEFAULT (test, thresh=%.1f) → F1=%.4f Prec=%.4f Recall=%.4f FPR=%.4f",
        train_default_threshold,
        test_default_metrics.f1,
        test_default_metrics.precision,
        test_default_metrics.recall,
        test_default_metrics.fpr,
    )

    # ── Replay v4 and v5 weights on both splits ───────────────────
    v4_payload = _load_json(V4_CALIBRATION_PATH)
    v5_payload = _load_json(V5_CALIBRATION_PATH)
    v4_opt_weights = (
        v4_payload.get("optimal", {}).get("weights") if v4_payload else None
    )
    v5_opt_weights = (
        v5_payload.get("optimal", {}).get("weights") if v5_payload else None
    )

    def _replay(weights: dict[str, float] | None) -> dict | None:
        if weights is None:
            return None
        # Find best threshold on TRAIN, then lock it for TEST.
        _, m_train, thresh_train = (
            calibrator.objective_with_threshold_search(weights, train_corpus)
        )
        test_coefs = calibrator.score_with_weights(weights, test_corpus)
        m_test = calibrator.compute_metrics(
            test_coefs, test_corpus, threshold=thresh_train
        )
        return {
            "weights": weights,
            "train_metrics": _metrics_dict(m_train),
            "train_threshold": thresh_train,
            "test_metrics_at_train_threshold": _metrics_dict(m_test),
            "gap_f1": m_train.f1 - m_test.f1,
        }

    v4_replay = _replay(v4_opt_weights)
    v5_replay = _replay(v5_opt_weights)
    if v4_replay is not None:
        logger.info(
            "V4 weights replayed → train F1=%.4f test F1=%.4f gap=%+.4f",
            v4_replay["train_metrics"]["f1"],
            v4_replay["test_metrics_at_train_threshold"]["f1"],
            v4_replay["gap_f1"],
        )
    if v5_replay is not None:
        logger.info(
            "V5 weights replayed → train F1=%.4f test F1=%.4f gap=%+.4f",
            v5_replay["train_metrics"]["f1"],
            v5_replay["test_metrics_at_train_threshold"]["f1"],
            v5_replay["gap_f1"],
        )

    # Threshold sweep for the v5.1 weights on both splits.
    train_sweep = threshold_sweep(calibrator, result.weights, train_corpus)
    test_sweep = threshold_sweep(calibrator, result.weights, test_corpus)

    # ── Ship/no-ship interpretation ───────────────────────────────
    gap_f1 = gap["f1"]
    if gap_f1 < 0.02:
        gap_verdict = "CLEAN"
        recommendation = (
            "gap < 0.02 F1 → v5.1 weights generalize; safe to ship as new "
            "default IF test-set F1 beats current default by a meaningful "
            "margin."
        )
    elif gap_f1 < 0.05:
        gap_verdict = "MODEST_OVERFIT"
        recommendation = (
            "gap 0.02–0.05 F1 → modest overfit; ship as opt-in preset "
            "with caveat, do NOT replace production defaults."
        )
    else:
        gap_verdict = "SEVERE_OVERFIT"
        recommendation = (
            "gap > 0.05 F1 → severe overfit; do NOT ship as defaults. "
            "Leave as opt-in preset only, investigate positive-set or "
            "brain-score stability next."
        )
    logger.info("Gap verdict: %s", gap_verdict)
    logger.info("Recommendation: %s", recommendation)

    # ── Persist ───────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "variant": "v5_1_full_moltbook",
        "budget_note": (
            f"{RANDOM_ITER} random + {LOCAL_ITER} local iterations "
            "(doubled vs Agent 7's v5 run). 80/20 holdout on negatives "
            f"by author_id, seed={SPLIT_SEED}. Positives shared between "
            "train and test — see script docstring."
        ),
        "random_iter": RANDOM_ITER,
        "local_iter": LOCAL_ITER,
        "split": {
            "seed": SPLIT_SEED,
            "test_fraction": TEST_FRACTION,
            "train_negatives": len(train_negatives),
            "test_negatives": len(test_negatives),
            "positive_sharing": "full positive set used for both splits",
        },
        "decision_threshold": calibrator.decision_threshold,
        "fpr_cap": calibrator.fpr_cap,
        "brain_names": [b.name for b in brains],
        "corpus": {
            "positives": len(positives),
            "negatives_total": len(all_negatives),
            "positive_sources": _source_counts(positives),
            "negative_sources": _source_counts(all_negatives),
            "negative_origin": {
                "path": str(MOLTBOOK_PATH.relative_to(REPO_ROOT)),
                "min_posts_per_author": MIN_POSTS_PER_AUTHOR,
                "sampling_seed": SAMPLING_SEED,
                "target_cap": NEGATIVE_TARGET,
            },
        },
        "default": {
            "weights": default_weights,
            "train_metrics": _metrics_dict(train_default_metrics),
            "train_threshold": train_default_threshold,
            "test_metrics_at_train_threshold": _metrics_dict(test_default_metrics),
        },
        "optimal": {
            "weights": result.weights,
            "train_metrics": _metrics_dict(result.metrics),
            "train_threshold": result.threshold,
            "test_metrics_at_train_threshold": _metrics_dict(test_metrics_fixed_thresh),
            "test_metrics_at_test_best_threshold": _metrics_dict(
                test_metrics_free_thresh
            ),
            "test_best_threshold": test_best_thresh,
            "train_test_gap": gap,
            "iterations": result.iterations,
        },
        "v4_replay": v4_replay,
        "v5_replay": v5_replay,
        "v4_calibration_reference": {
            "path": str(V4_CALIBRATION_PATH.relative_to(REPO_ROOT))
            if V4_CALIBRATION_PATH.exists() else None,
            "timestamp": v4_payload.get("timestamp") if v4_payload else None,
        },
        "v5_calibration_reference": {
            "path": str(V5_CALIBRATION_PATH.relative_to(REPO_ROOT))
            if V5_CALIBRATION_PATH.exists() else None,
            "timestamp": v5_payload.get("timestamp") if v5_payload else None,
        },
        "threshold_sweep_train": train_sweep,
        "threshold_sweep_test": test_sweep,
        "search_history_tail": result.history[-40:],
        "gap_verdict": gap_verdict,
        "recommendation": recommendation,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "precompute_seconds": round(precompute_elapsed, 2),
        "search_seconds": round(search_elapsed, 2),
    }
    # replace() is imported for completeness; the script doesn't
    # currently mutate calibrator state.
    _ = replace  # silence unused-import warnings
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s (%.1f KB)", OUT_PATH, OUT_PATH.stat().st_size / 1024)
    print(f"\nSaved → {OUT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
