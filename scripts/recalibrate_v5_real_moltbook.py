"""V5 re-calibration with REAL Moltbook event streams as negatives.

The v4 calibration used synthetic `_clean_round` events (derived from
`SyntheticArchetypeGenerator`) as its negative class. That is a known
blocker: the optimizer could be fitting the synthetic shape rather
than realistic benign traffic.

This script rebuilds the calibration with the *same positive set* as
v4 (alignment scenarios + adversarial archetypes) but swaps the
negatives for real Moltbook posts (`data/moltbook_posts.jsonl`),
reconstructed into per-agent Event streams the same way
`Observatory.from_synthetic` does it.

Outputs:
    experiments/calibration_v5_real_moltbook.json

Notes:
    - Reuses all 15 brains (same ensemble v4 used).
    - Uses the 5-brain `get_default_brains()` ensemble is NOT enough —
      the v4 weights reference 15 brains, so we explicitly wire them
      all here.
    - Budget: 50 random + 25 local, matching v4 exactly so only the
      negative sample differs.
    - Negative sample: authors with >=2 posts so brains have signal,
      capped at 500 for runtime.
"""

from __future__ import annotations

import collections
import json
import logging
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

# Allow running this script from anywhere in the repo.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sybilcore.analysis.calibration import (  # noqa: E402
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
logger = logging.getLogger("recalibrate_v5")


# ── Configuration ──────────────────────────────────────────────────

# Keep v4's search budget so only the negative distribution differs.
RANDOM_ITER = 50
LOCAL_ITER = 25

# Real-Moltbook negatives: we cap to keep runtime comfortable.
# Benchmarked at ~685 ms/agent on this machine for 15 brains, so
# 500 negatives + 200 positives ≈ 8 min brain precompute.
NEGATIVE_TARGET = 500
POSITIVE_TARGET = 200
MIN_POSTS_PER_AUTHOR = 2  # need at least a couple events for signal
SAMPLING_SEED = 42

MOLTBOOK_PATH = REPO_ROOT / "data" / "moltbook_posts.jsonl"
OUT_PATH = REPO_ROOT / "experiments" / "calibration_v5_real_moltbook.json"
V4_CALIBRATION_PATH = REPO_ROOT / "experiments" / "calibration_v4_20260407_101019.json"


# ── Brain ensemble ──────────────────────────────────────────────────

def build_full_brain_ensemble() -> list:
    """Return the full 15-brain ensemble used by the v4 calibration.

    `get_default_brains()` has been pruned to 5 brains post-v4, but
    the v4 calibration result references all 15. We reconstruct the
    original ensemble here so weight vectors are comparable.
    """
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
    """Rebuild a single Moltbook post as an Event.

    Matches the shape produced by `Observatory.from_synthetic` so the
    brains see the same metadata keys they do in production.
    """
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


def build_real_moltbook_negatives(
    target: int = NEGATIVE_TARGET,
    min_posts: int = MIN_POSTS_PER_AUTHOR,
    seed: int = SAMPLING_SEED,
) -> list[LabeledAgent]:
    """Load real Moltbook posts, group by author, sample N authors."""
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

    # Deterministic sample for reproducibility.
    rng = random.Random(seed)  # noqa: S311 — non-crypto reproducibility
    authors_sorted = sorted(eligible.keys())
    rng.shuffle(authors_sorted)
    chosen = authors_sorted[:target]

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

    logger.info("Built %d real Moltbook negatives", len(negatives))
    return negatives


# ── Reporting helpers ──────────────────────────────────────────────

def _load_v4_payload() -> dict | None:
    if not V4_CALIBRATION_PATH.exists():
        logger.warning("v4 calibration not found: %s", V4_CALIBRATION_PATH)
        return None
    try:
        return json.loads(V4_CALIBRATION_PATH.read_text())
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse v4 calibration: %s", exc)
        return None


def _pretty_print(
    default_weights: dict[str, float],
    optimal_weights: dict[str, float],
    default_metrics,
    optimal_metrics,
    default_threshold: float,
    optimal_threshold: float,
) -> None:
    bar = "─" * 76
    print()
    print(bar)
    print(f"{'Brain':<22}{'Default':>12}{'V5 Optimal':>14}{'Δ':>12}")
    print(bar)
    for name in sorted(default_weights):
        d = default_weights[name]
        o = optimal_weights.get(name, 1.0)
        delta = o - d
        marker = "↑" if delta > 0.05 else "↓" if delta < -0.05 else " "
        print(f"{name:<22}{d:>12.2f}{o:>14.2f}{delta:>+11.2f} {marker}")
    print(bar)
    print(
        f"{'F1':<22}{default_metrics.f1:>12.4f}{optimal_metrics.f1:>14.4f}"
        f"{optimal_metrics.f1 - default_metrics.f1:>+12.4f}"
    )
    print(
        f"{'Precision':<22}{default_metrics.precision:>12.4f}"
        f"{optimal_metrics.precision:>14.4f}"
        f"{optimal_metrics.precision - default_metrics.precision:>+12.4f}"
    )
    print(
        f"{'Recall':<22}{default_metrics.recall:>12.4f}"
        f"{optimal_metrics.recall:>14.4f}"
        f"{optimal_metrics.recall - default_metrics.recall:>+12.4f}"
    )
    print(
        f"{'AUC':<22}{default_metrics.auc:>12.4f}{optimal_metrics.auc:>14.4f}"
        f"{optimal_metrics.auc - default_metrics.auc:>+12.4f}"
    )
    print(
        f"{'FPR':<22}{default_metrics.fpr:>12.4f}{optimal_metrics.fpr:>14.4f}"
        f"{optimal_metrics.fpr - default_metrics.fpr:>+12.4f}"
    )
    print(
        f"{'Threshold (best)':<22}{default_threshold:>12.2f}"
        f"{optimal_threshold:>14.2f}"
        f"{optimal_threshold - default_threshold:>+12.2f}"
    )
    print(bar)


# ── Main ───────────────────────────────────────────────────────────

def main() -> int:
    t0 = time.perf_counter()

    logger.info(
        "Building labeled corpus (v5: real Moltbook negatives)"
    )
    positives = build_positives(target=POSITIVE_TARGET)
    negatives = build_real_moltbook_negatives(
        target=NEGATIVE_TARGET,
        min_posts=MIN_POSTS_PER_AUTHOR,
        seed=SAMPLING_SEED,
    )
    logger.info(
        "Corpus: %d positives + %d negatives = %d total",
        len(positives),
        len(negatives),
        len(positives) + len(negatives),
    )

    brains = build_full_brain_ensemble()
    logger.info(
        "Brain ensemble: %d brains (%s)",
        len(brains),
        ", ".join(b.name for b in brains),
    )
    calibrator = WeightCalibrator(seed=42, brains=brains)

    logger.info("Precomputing brain scores — this is the expensive step")
    corpus = calibrator.load_corpus(positives, negatives)
    precompute_elapsed = time.perf_counter() - t0
    logger.info("Brain scores cached in %.1fs", precompute_elapsed)

    # Baseline: default weights with threshold search.
    default_weights = calibrator.default_weights()
    _, default_metrics, default_threshold = (
        calibrator.objective_with_threshold_search(default_weights, corpus)
    )
    logger.info(
        "DEFAULT → F1=%.4f Prec=%.4f Recall=%.4f FPR=%.4f thresh=%.1f",
        default_metrics.f1,
        default_metrics.precision,
        default_metrics.recall,
        default_metrics.fpr,
        default_threshold,
    )

    # Hybrid search (matches v4 budget).
    logger.info(
        "Running hybrid search: %d random + %d local iterations",
        RANDOM_ITER,
        LOCAL_ITER,
    )
    result = calibrator.hybrid_search(
        corpus,
        random_iter=RANDOM_ITER,
        local_iter=LOCAL_ITER,
    )
    logger.info(
        "V5 OPTIMAL → F1=%.4f Prec=%.4f Recall=%.4f FPR=%.4f thresh=%.1f",
        result.metrics.f1,
        result.metrics.precision,
        result.metrics.recall,
        result.metrics.fpr,
        result.threshold,
    )

    _pretty_print(
        default_weights,
        result.weights,
        default_metrics,
        result.metrics,
        default_threshold,
        result.threshold,
    )

    sweep = threshold_sweep(calibrator, result.weights, corpus)

    # Cross-reference with v4 payload if available.
    v4_payload = _load_v4_payload()
    v4_optimal_weights = (
        v4_payload.get("optimal", {}).get("weights") if v4_payload else None
    )

    # Re-score v4's optimal weights on the v5 corpus to see whether
    # they generalize to real negatives at all.
    v4_on_v5: dict | None = None
    if v4_optimal_weights is not None:
        try:
            _, v4_metrics_on_v5, v4_thresh_on_v5 = (
                calibrator.objective_with_threshold_search(
                    v4_optimal_weights, corpus
                )
            )
            v4_on_v5 = {
                "weights": v4_optimal_weights,
                "metrics": _metrics_dict(v4_metrics_on_v5),
                "best_threshold": v4_thresh_on_v5,
            }
            logger.info(
                "V4 weights re-scored on V5 corpus → "
                "F1=%.4f Prec=%.4f Recall=%.4f FPR=%.4f thresh=%.1f",
                v4_metrics_on_v5.f1,
                v4_metrics_on_v5.precision,
                v4_metrics_on_v5.recall,
                v4_metrics_on_v5.fpr,
                v4_thresh_on_v5,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not re-score v4 weights: %s", exc)

    # Persist.
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "variant": "v5_real_moltbook",
        "budget_note": (
            f"{RANDOM_ITER} random + {LOCAL_ITER} local iterations "
            "(matches v4 budget; only the negative sample changed)."
        ),
        "random_iter": RANDOM_ITER,
        "local_iter": LOCAL_ITER,
        "decision_threshold": calibrator.decision_threshold,
        "fpr_cap": calibrator.fpr_cap,
        "brain_names": [b.name for b in brains],
        "corpus": {
            "positives": len(positives),
            "negatives": len(negatives),
            "positive_sources": _source_counts(positives),
            "negative_sources": _source_counts(negatives),
            "negative_origin": {
                "path": str(MOLTBOOK_PATH.relative_to(REPO_ROOT)),
                "min_posts_per_author": MIN_POSTS_PER_AUTHOR,
                "sampling_seed": SAMPLING_SEED,
            },
        },
        "default": {
            "weights": default_weights,
            "metrics": _metrics_dict(default_metrics),
            "best_threshold": default_threshold,
        },
        "optimal": {
            "weights": result.weights,
            "metrics": _metrics_dict(result.metrics),
            "iterations": result.iterations,
            "best_threshold": result.threshold,
        },
        "v4_weights_on_v5_corpus": v4_on_v5,
        "v4_calibration_reference": {
            "path": str(V4_CALIBRATION_PATH.relative_to(REPO_ROOT))
            if V4_CALIBRATION_PATH.exists() else None,
            "timestamp": v4_payload.get("timestamp") if v4_payload else None,
        },
        "improvement_over_default": {
            "f1": result.metrics.f1 - default_metrics.f1,
            "precision": result.metrics.precision - default_metrics.precision,
            "recall": result.metrics.recall - default_metrics.recall,
            "auc": result.metrics.auc - default_metrics.auc,
            "fpr": result.metrics.fpr - default_metrics.fpr,
        },
        "threshold_sweep": sweep,
        "search_history_tail": result.history[-30:],
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "precompute_seconds": round(precompute_elapsed, 2),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s (%.1f KB)", OUT_PATH, OUT_PATH.stat().st_size / 1024)
    print(f"\nSaved → {OUT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
