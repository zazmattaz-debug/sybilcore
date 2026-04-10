"""Paper-fortification experiments 1, 2, 3, 5 (tier-1 agent).

Runs in a single process to share the expensive brain-score precomputation
across the four experiments. Each experiment writes its own output file:

    Experiment 1 -> experiments/paper_fortify_bootstrap_ci.json
    Experiment 2 -> experiments/paper_fortify_5fold_cv.json
    Experiment 3 -> experiments/paper_fortify_cross_corpus.json
    Experiment 5 -> experiments/paper_fortify_counterfactual_stats.json

All experiments reuse the v5.1 calibration corpus (positives + moltbook
negatives) plus the out-of-distribution v5_corpus for generalization tests.

No external LLM calls. No secret access. Local compute only.
"""

from __future__ import annotations

import json
import logging
import math
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from dotenv import load_dotenv
from sklearn.utils import resample
from statsmodels.stats.contingency_tables import mcnemar

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load env (harmless if missing, never print values).
load_dotenv(REPO_ROOT.parent.parent.parent / ".env")

from sybilcore.analysis.calibration import (  # noqa: E402
    CalibrationCorpus,
    LabeledAgent,
    WeightCalibrator,
)
from sybilcore.analysis.run_calibration import build_positives  # noqa: E402
from sybilcore.core.weight_presets import (  # noqa: E402
    OPTIMIZED_WEIGHTS_V5_1,
    OPTIMIZED_WEIGHTS_V5_1_THRESHOLD,
)
from sybilcore.models.event import Event, EventType  # noqa: E402

# DeLong from the existing stats harness.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_statistical_tests import delong_roc_test  # noqa: E402

# Reuse the v5.1 recalibration builders for corpus consistency.
from recalibrate_v5_1_full_moltbook import (  # noqa: E402
    build_full_brain_ensemble,
    build_full_moltbook_negatives,
    split_negatives,
    subset_corpus,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("paper_fortify")

EXPERIMENTS_DIR = REPO_ROOT / "experiments"
V5_CORPUS_PATH = EXPERIMENTS_DIR / "v5_corpus.json"

# 5-brain over-pruned ensemble (the old incorrect default that we almost
# shipped). Used in Experiment 5 as a counterfactual.
OVER_PRUNED_5BRAINS = {
    "deception",
    "resource_hoarding",
    "social_graph",
    "intent_drift",
    "compromise",
}


# ---------------------------------------------------------------------------
# Corpus builders
# ---------------------------------------------------------------------------


def load_v5_corpus_as_labeled_agents() -> list[LabeledAgent]:
    """Convert the v5 targeted scenarios JSON into LabeledAgent objects."""
    raw = json.loads(V5_CORPUS_PATH.read_text())
    agents: list[LabeledAgent] = []
    for record in raw["agents"]:
        events = []
        for ev_raw in record["events"]:
            ts_raw = ev_raw.get("timestamp")
            if ts_raw:
                try:
                    ts = datetime.fromisoformat(ts_raw)
                except ValueError:
                    ts = datetime.now(UTC)
            else:
                ts = datetime.now(UTC)

            etype_raw = ev_raw.get("event_type", "output_generated")
            try:
                etype = EventType(etype_raw)
            except ValueError:
                etype = EventType.OUTPUT_GENERATED

            events.append(
                Event(
                    event_id=ev_raw.get("event_id") or str(uuid4()),
                    agent_id=ev_raw.get("agent_id") or record["agent_id"],
                    event_type=etype,
                    timestamp=ts,
                    content=ev_raw.get("content", "") or "",
                    metadata=dict(ev_raw.get("metadata") or {}),
                    source=ev_raw.get("source", "v5_corpus") or "v5_corpus",
                )
            )
        events.sort(key=lambda e: e.timestamp)
        agents.append(
            LabeledAgent(
                agent_id=record["agent_id"],
                events=tuple(events),
                label=int(record["label"]),
                source=record.get("source", "v5_corpus"),
            )
        )
    return agents


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def score_corpus(
    calibrator: WeightCalibrator,
    corpus: CalibrationCorpus,
    weights: dict[str, float],
) -> dict[str, float]:
    """Return {agent_id: coefficient} for `weights` applied to cached scores."""
    return calibrator.score_with_weights(weights, corpus)


def metrics_at_threshold(
    coefficients: dict[str, float],
    corpus: CalibrationCorpus,
    threshold: float,
) -> dict:
    tp = fp = tn = fn = 0
    for agent in corpus.agents:
        pred = 1 if coefficients[agent.agent_id] >= threshold else 0
        truth = agent.label
        if pred == 1 and truth == 1:
            tp += 1
        elif pred == 1 and truth == 0:
            fp += 1
        elif pred == 0 and truth == 0:
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
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


# ---------------------------------------------------------------------------
# Experiment 1 — Bootstrap 95% CIs on v5.1 holdout
# ---------------------------------------------------------------------------


def experiment_1_bootstrap_ci(
    test_corpus: CalibrationCorpus,
    coefficients: dict[str, float],
    threshold: float,
    n_boot: int = 1000,
    seed: int = 20260408,
) -> dict:
    logger.info("Exp1: bootstrap 95%% CIs, n_boot=%d", n_boot)
    agents = list(test_corpus.agents)
    indices = np.arange(len(agents))
    rng = np.random.RandomState(seed)

    f1s, recalls, fprs, precisions = [], [], [], []
    for _ in range(n_boot):
        idx = resample(indices, random_state=rng.randint(0, 2**31 - 1))
        sub_agents = [agents[i] for i in idx]
        tp = fp = tn = fn = 0
        for a in sub_agents:
            pred = 1 if coefficients[a.agent_id] >= threshold else 0
            if pred == 1 and a.label == 1:
                tp += 1
            elif pred == 1 and a.label == 0:
                fp += 1
            elif pred == 0 and a.label == 0:
                tn += 1
            else:
                fn += 1
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        f1s.append(f1)
        recalls.append(r)
        fprs.append(fpr)
        precisions.append(p)

    def _ci(values: list[float]) -> dict:
        arr = np.asarray(values)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "ci_lo": float(np.percentile(arr, 2.5)),
            "ci_hi": float(np.percentile(arr, 97.5)),
        }

    return {
        "n_boot": n_boot,
        "seed": seed,
        "n_holdout": len(agents),
        "threshold": threshold,
        "f1": _ci(f1s),
        "recall": _ci(recalls),
        "fpr": _ci(fprs),
        "precision": _ci(precisions),
    }


# ---------------------------------------------------------------------------
# Experiment 2 — 5-fold CV on v5.1 calibration
# ---------------------------------------------------------------------------


def experiment_2_fivefold_cv(
    positives: list[LabeledAgent],
    all_negatives: list[LabeledAgent],
    full_corpus: CalibrationCorpus,
    calibrator: WeightCalibrator,
    seeds: tuple[int, ...] = (20260408, 20260409, 20260410, 20260411, 20260412),
    random_iter: int = 60,
    local_iter: int = 30,
) -> dict:
    logger.info("Exp2: 5-fold CV with seeds %s", seeds)
    folds = []
    for seed in seeds:
        fold_t0 = time.perf_counter()
        train_negs, test_negs = split_negatives(
            all_negatives, test_fraction=0.2, seed=seed
        )
        train_corpus = subset_corpus(full_corpus, [*positives, *train_negs])
        test_corpus = subset_corpus(full_corpus, [*positives, *test_negs])

        result = calibrator.hybrid_search(
            train_corpus,
            random_iter=random_iter,
            local_iter=local_iter,
        )
        test_coefs = calibrator.score_with_weights(result.weights, test_corpus)
        test_m = calibrator.compute_metrics(
            test_coefs, test_corpus, threshold=result.threshold
        )
        elapsed = time.perf_counter() - fold_t0
        logger.info(
            "Exp2 fold seed=%d: train F1=%.4f test F1=%.4f thresh=%.1f (%.1fs)",
            seed,
            result.metrics.f1,
            test_m.f1,
            result.threshold,
            elapsed,
        )
        folds.append(
            {
                "seed": seed,
                "weights": dict(result.weights),
                "train_threshold": result.threshold,
                "train_metrics": {
                    "f1": result.metrics.f1,
                    "precision": result.metrics.precision,
                    "recall": result.metrics.recall,
                    "fpr": result.metrics.fpr,
                    "auc": result.metrics.auc,
                },
                "test_metrics": {
                    "f1": test_m.f1,
                    "precision": test_m.precision,
                    "recall": test_m.recall,
                    "fpr": test_m.fpr,
                    "auc": test_m.auc,
                },
                "elapsed_s": elapsed,
            }
        )

    # Weight stability across folds.
    brain_names = sorted(folds[0]["weights"].keys())
    weight_stability = {}
    for name in brain_names:
        vals = np.asarray([f["weights"][name] for f in folds])
        weight_stability[name] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "cv_pct": float(100.0 * vals.std() / (abs(vals.mean()) + 1e-9)),
        }

    # Metric stability across folds.
    metric_stability = {}
    for phase in ("train_metrics", "test_metrics"):
        metric_stability[phase] = {}
        for mname in ("f1", "precision", "recall", "fpr", "auc"):
            vals = np.asarray([f[phase][mname] for f in folds])
            metric_stability[phase][mname] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }

    # Flag divergent folds (>2 std from mean on test F1).
    test_f1_vals = np.asarray([f["test_metrics"]["f1"] for f in folds])
    mu, sigma = float(test_f1_vals.mean()), float(test_f1_vals.std())
    divergent = []
    if sigma > 0:
        for f in folds:
            z = abs(f["test_metrics"]["f1"] - mu) / sigma
            if z > 2.0:
                divergent.append({"seed": f["seed"], "z": z})

    return {
        "seeds": list(seeds),
        "random_iter": random_iter,
        "local_iter": local_iter,
        "folds": folds,
        "weight_stability": weight_stability,
        "metric_stability": metric_stability,
        "divergent_folds": divergent,
    }


# ---------------------------------------------------------------------------
# Experiment 3 — Cross-corpus generalization
# ---------------------------------------------------------------------------


def experiment_3_cross_corpus(
    calibrator: WeightCalibrator,
    v5_agents: list[LabeledAgent],
) -> dict:
    logger.info("Exp3: cross-corpus generalization on v5_corpus")
    v5_weights = dict(OPTIMIZED_WEIGHTS_V5_1)
    v5_threshold = OPTIMIZED_WEIGHTS_V5_1_THRESHOLD

    # Overall corpus.
    overall_corpus = calibrator.load_corpus(
        [a for a in v5_agents if a.label == 1],
        [a for a in v5_agents if a.label == 0],
    )
    overall_coefs = calibrator.score_with_weights(v5_weights, overall_corpus)
    overall_metrics = calibrator.compute_metrics(
        overall_coefs, overall_corpus, threshold=v5_threshold
    )

    # Per-attack breakdown (split by positive source).
    benign = [a for a in v5_agents if a.label == 0]
    sources = sorted(
        {a.source for a in v5_agents if a.label == 1}
    )
    per_source = {}
    for src in sources:
        pos_src = [a for a in v5_agents if a.label == 1 and a.source == src]
        sub_corpus = calibrator.load_corpus(pos_src, benign)
        sub_coefs = calibrator.score_with_weights(v5_weights, sub_corpus)
        sub_m = calibrator.compute_metrics(
            sub_coefs, sub_corpus, threshold=v5_threshold
        )
        per_source[src] = {
            "n_positives": len(pos_src),
            "n_negatives": len(benign),
            "f1": sub_m.f1,
            "precision": sub_m.precision,
            "recall": sub_m.recall,
            "fpr": sub_m.fpr,
            "auc": sub_m.auc,
        }

    return {
        "target_corpus": "experiments/v5_corpus.json",
        "weights_used": "OPTIMIZED_WEIGHTS_V5_1",
        "threshold_used": v5_threshold,
        "n_positives": sum(1 for a in v5_agents if a.label == 1),
        "n_negatives": sum(1 for a in v5_agents if a.label == 0),
        "overall": {
            "f1": overall_metrics.f1,
            "precision": overall_metrics.precision,
            "recall": overall_metrics.recall,
            "fpr": overall_metrics.fpr,
            "auc": overall_metrics.auc,
        },
        "per_attack_source": per_source,
        "training_f1_for_reference": 0.9952,
    }


# ---------------------------------------------------------------------------
# Experiment 5 — Counterfactual 13-brain vs 5-brain stats test
# ---------------------------------------------------------------------------


def _binary_predictions(
    coefficients: dict[str, float],
    corpus: CalibrationCorpus,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (truth, pred, scores) arrays aligned to corpus.agents order."""
    truth = np.asarray([a.label for a in corpus.agents], dtype=int)
    pred = np.asarray(
        [1 if coefficients[a.agent_id] >= threshold else 0 for a in corpus.agents],
        dtype=int,
    )
    scores = np.asarray(
        [coefficients[a.agent_id] for a in corpus.agents], dtype=float
    )
    return truth, pred, scores


def experiment_5_counterfactual(
    calibrator: WeightCalibrator,
    test_corpus: CalibrationCorpus,
    v5_1_weights: dict[str, float],
    v5_1_threshold: float,
) -> dict:
    logger.info("Exp5: 13-brain corrected vs 5-brain over-pruned")

    # 13-brain default (calibrator already knows full ensemble).
    default_weights = calibrator.default_weights()

    # Score v5.1 OPTIMAL (the corrected default).
    coefs_v51 = calibrator.score_with_weights(v5_1_weights, test_corpus)
    truth_v51, pred_v51, scores_v51 = _binary_predictions(
        coefs_v51, test_corpus, v5_1_threshold
    )

    # Score the 5-brain over-pruned ensemble by zero-weighting everything else.
    zeroed = {
        name: (val if name in OVER_PRUNED_5BRAINS else 0.0)
        for name, val in default_weights.items()
    }
    coefs_5 = calibrator.score_with_weights(zeroed, test_corpus)

    # Sweep thresholds for the 5-brain and pick the one that maximises F1
    # on the test set — a *charitable* baseline. This biases toward the
    # 5-brain ensemble so the counterfactual is conservative.
    best_thresh = None
    best_f1 = -1.0
    for thresh in range(0, 101, 5):
        m = metrics_at_threshold(coefs_5, test_corpus, float(thresh))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = float(thresh)
    assert best_thresh is not None
    truth_5, pred_5, scores_5 = _binary_predictions(
        coefs_5, test_corpus, best_thresh
    )

    # Sanity: both predictions must use the same truth (they do by construction).
    assert (truth_v51 == truth_5).all()
    truth = truth_v51

    # McNemar's exact test on paired classifications.
    # Contingency: rows = classifier A (v5.1), cols = classifier B (5-brain).
    a_correct = pred_v51 == truth
    b_correct = pred_5 == truth

    n11 = int(np.sum(a_correct & b_correct))  # both correct
    n10 = int(np.sum(a_correct & ~b_correct))  # only A correct
    n01 = int(np.sum(~a_correct & b_correct))  # only B correct
    n00 = int(np.sum(~a_correct & ~b_correct))  # both wrong

    mc_result = mcnemar(
        [[n11, n10], [n01, n00]],
        exact=True,
        correction=False,
    )

    # DeLong AUROC test on score vectors.
    auc_v51, auc_5, delta, delong_p = delong_roc_test(
        truth.astype(float),
        scores_v51,
        scores_5,
    )

    # Metric summary for both.
    m_v51 = metrics_at_threshold(coefs_v51, test_corpus, v5_1_threshold)
    m_5 = metrics_at_threshold(coefs_5, test_corpus, best_thresh)

    return {
        "description": (
            "Counterfactual comparison between the shipped 13-brain corrected "
            "default (v5.1 OPTIMAL weights) and the over-pruned 5-brain ensemble "
            "that was briefly the package default before the content-inspection "
            "brains were restored. McNemar's test is run on paired predictions; "
            "DeLong's test compares ROC-AUC."
        ),
        "v5_1_threshold": v5_1_threshold,
        "five_brain_threshold_cherry_picked": best_thresh,
        "over_pruned_brain_subset": sorted(OVER_PRUNED_5BRAINS),
        "contingency": {
            "both_correct": n11,
            "only_v5_1_correct": n10,
            "only_5brain_correct": n01,
            "both_wrong": n00,
        },
        "mcnemar": {
            "statistic": float(mc_result.statistic),
            "p_value": float(mc_result.pvalue),
        },
        "delong_auc": {
            "auc_13brain_v5_1": auc_v51,
            "auc_5brain_over_pruned": auc_5,
            "delta": delta,
            "p_value": delong_p,
        },
        "metrics_13brain_v5_1": m_v51,
        "metrics_5brain_over_pruned": m_5,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    t0 = time.perf_counter()
    logger.info("Paper fortify combined — exps 1, 2, 3, 5")

    # Build positives + moltbook negatives once.
    positives = build_positives(target=200)
    all_negatives = build_full_moltbook_negatives(
        min_posts=2, target=None, seed=42
    )
    logger.info(
        "Corpus: %d positives, %d negatives", len(positives), len(all_negatives)
    )

    brains = build_full_brain_ensemble()
    calibrator = WeightCalibrator(seed=42, brains=brains)

    full_corpus = calibrator.load_corpus(positives, all_negatives)
    logger.info(
        "Full corpus loaded + brain scores cached (%.1fs)", time.perf_counter() - t0
    )

    # ----- Split using the v5.1 canonical seed for Exps 1 and 5 ---------
    train_negs, test_negs = split_negatives(
        all_negatives, test_fraction=0.2, seed=20260408
    )
    test_corpus = subset_corpus(full_corpus, [*positives, *test_negs])

    # ----- Score test corpus with shipped v5.1 weights -------------------
    v5_1_weights = dict(OPTIMIZED_WEIGHTS_V5_1)
    v5_1_threshold = OPTIMIZED_WEIGHTS_V5_1_THRESHOLD
    test_coefs = calibrator.score_with_weights(v5_1_weights, test_corpus)
    point_metrics = calibrator.compute_metrics(
        test_coefs, test_corpus, threshold=v5_1_threshold
    )
    logger.info(
        "v5.1 holdout (point est) F1=%.4f prec=%.4f rec=%.4f fpr=%.4f",
        point_metrics.f1,
        point_metrics.precision,
        point_metrics.recall,
        point_metrics.fpr,
    )

    # ----- Exp 1 ---------------------------------------------------------
    exp1 = experiment_1_bootstrap_ci(
        test_corpus, test_coefs, v5_1_threshold, n_boot=1000, seed=20260408
    )
    exp1["v5_1_point_estimate"] = {
        "f1": point_metrics.f1,
        "precision": point_metrics.precision,
        "recall": point_metrics.recall,
        "fpr": point_metrics.fpr,
    }
    (EXPERIMENTS_DIR / "paper_fortify_bootstrap_ci.json").write_text(
        json.dumps(exp1, indent=2)
    )
    logger.info(
        "Exp1 F1 mean=%.4f [CI %.4f, %.4f]",
        exp1["f1"]["mean"],
        exp1["f1"]["ci_lo"],
        exp1["f1"]["ci_hi"],
    )

    # ----- Exp 2 ---------------------------------------------------------
    exp2 = experiment_2_fivefold_cv(
        positives,
        all_negatives,
        full_corpus,
        calibrator,
        seeds=(20260408, 20260409, 20260410, 20260411, 20260412),
        random_iter=60,
        local_iter=30,
    )
    (EXPERIMENTS_DIR / "paper_fortify_5fold_cv.json").write_text(
        json.dumps(exp2, indent=2)
    )
    logger.info(
        "Exp2 test F1 mean=%.4f std=%.4f (n=%d folds, divergent=%d)",
        exp2["metric_stability"]["test_metrics"]["f1"]["mean"],
        exp2["metric_stability"]["test_metrics"]["f1"]["std"],
        len(exp2["folds"]),
        len(exp2["divergent_folds"]),
    )

    # ----- Exp 3 ---------------------------------------------------------
    v5_agents = load_v5_corpus_as_labeled_agents()
    exp3 = experiment_3_cross_corpus(calibrator, v5_agents)
    (EXPERIMENTS_DIR / "paper_fortify_cross_corpus.json").write_text(
        json.dumps(exp3, indent=2)
    )
    logger.info(
        "Exp3 cross-corpus overall F1=%.4f (train ref %.4f)",
        exp3["overall"]["f1"],
        exp3["training_f1_for_reference"],
    )

    # ----- Exp 5 ---------------------------------------------------------
    exp5 = experiment_5_counterfactual(
        calibrator, test_corpus, v5_1_weights, v5_1_threshold
    )
    (EXPERIMENTS_DIR / "paper_fortify_counterfactual_stats.json").write_text(
        json.dumps(exp5, indent=2)
    )
    logger.info(
        "Exp5 McNemar p=%.3g, DeLong delta AUC=%.4f p=%.3g",
        exp5["mcnemar"]["p_value"],
        exp5["delong_auc"]["delta"],
        exp5["delong_auc"]["p_value"],
    )

    logger.info("Combined fortify done in %.1fs", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
