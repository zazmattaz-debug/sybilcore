"""Phase 1 InterpretabilityProxyBrain Experiment — Cornell Deception Dataset.

Scores all 800 Cornell Hotel Reviews (400 deceptive + 400 truthful) through
TRIBE v2 local scorer (distilled GBR mode) and evaluates whether resonance_score
discriminates deceptive from truthful reviews.

Optimised: loads sentence-transformer and GBR model ONCE, encodes all 800
reviews in a single batched pass (~30s vs ~15min one-at-a-time).

Usage:
    /usr/local/bin/python3.13 run_phase1_cornell.py

Output:
    rooms/engineering/sybilcore/experiments/phase1_cornell_tribe_v2_results.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]  # Desktop/Claude Code/
DATASET_ROOT = (
    REPO_ROOT
    / "rooms/engineering/sybilcore/experiments/phase1_data"
    / "cornell_deception/op_spam_v1.4/positive_polarity"
)
DECEPTIVE_DIR = DATASET_ROOT / "deceptive_from_MTurk"
TRUTHFUL_DIR = DATASET_ROOT / "truthful_from_TripAdvisor"
OUTPUT_PATH = (
    REPO_ROOT
    / "rooms/engineering/sybilcore/experiments/phase1_cornell_tribe_v2_results.json"
)

TRIBE_PIPELINE = REPO_ROOT / "rooms/research/tribe-v2/pipeline"
sys.path.insert(0, str(TRIBE_PIPELINE))

TRIBE_RESULTS = REPO_ROOT / "rooms/research/tribe-v2/results"
DISTILLED_MODEL_PATH = TRIBE_RESULTS / "distilled_model.joblib"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Calibrated shape thresholds from scorer_calibration.json
THRESHOLD_BORING = -0.229
THRESHOLD_RESONANT = 0.328
THRESHOLD_SPIKE = 0.437
NEUTRAL_MEAN = 0.0366  # training-set neutral baseline for distilled mode


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_reviews() -> list[dict[str, Any]]:
    """Load all 800 reviews with label and fold assignment."""
    records: list[dict[str, Any]] = []

    for label, base_dir in [("deceptive", DECEPTIVE_DIR), ("truthful", TRUTHFUL_DIR)]:
        for fold_n in range(1, 6):
            fold_dir = base_dir / f"fold{fold_n}"
            if not fold_dir.is_dir():
                print(f"WARNING: missing fold dir {fold_dir}", file=sys.stderr)
                continue
            for txt_file in sorted(fold_dir.glob("*.txt")):
                text = txt_file.read_text(encoding="utf-8", errors="replace").strip()
                records.append({
                    "filename": txt_file.name,
                    "label": label,
                    "fold": fold_n,
                    "text": text,
                    "text_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
                })

    print(f"Loaded {len(records)} reviews  "
          f"({sum(1 for r in records if r['label']=='deceptive')} deceptive, "
          f"{sum(1 for r in records if r['label']=='truthful')} truthful)")
    return records


# ---------------------------------------------------------------------------
# Classify shape (mirrors local_resonance_scorer logic)
# ---------------------------------------------------------------------------

def _classify_shape(resonance_score: float) -> str:
    if resonance_score >= THRESHOLD_SPIKE:
        return "spike"
    if resonance_score >= THRESHOLD_RESONANT:
        return "resonant"
    if resonance_score < THRESHOLD_BORING:
        return "boring"
    return "neutral"


# ---------------------------------------------------------------------------
# Batch scoring (loads model once, encodes all reviews in one pass)
# ---------------------------------------------------------------------------

def score_reviews_batch(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score all reviews with a single model load + batched encode."""
    print("  Loading distilled GBR model...")
    import joblib  # type: ignore[import]
    gbr_model = joblib.load(DISTILLED_MODEL_PATH)

    print(f"  Loading sentence-transformer ({MODEL_NAME})...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
    sbert = SentenceTransformer(MODEL_NAME)

    texts = [r["text"] for r in records]
    print(f"  Encoding {len(texts)} reviews in one batched pass...")
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embeddings = sbert.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)
    print(f"  Encoded in {time.time()-t0:.1f}s")

    print("  Predicting emotion scores via GBR...")
    predicted_scores = gbr_model.predict(embeddings)

    results: list[dict[str, Any]] = []
    for rec, emb_score in zip(records, predicted_scores, strict=True):
        es = float(emb_score)
        resonance = es - NEUTRAL_MEAN
        ratio = es / NEUTRAL_MEAN if NEUTRAL_MEAN > 0 else 0.0
        shape = _classify_shape(resonance)
        results.append({
            "filename": rec["filename"],
            "text_hash": rec["text_hash"],
            "label": rec["label"],
            "fold": rec["fold"],
            "emotion_score": round(es, 6),
            "resonance_score": round(resonance, 6),
            "emotion_ratio": round(ratio, 6),
            "signature_shape": shape,
            "scorer_mode": "distilled_gbr",
        })

    return results


# ---------------------------------------------------------------------------
# AUROC evaluation — 5-fold CV
# ---------------------------------------------------------------------------

def compute_auroc(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute AUROC for resonance_score discriminating deceptive vs truthful."""
    from sklearn.metrics import roc_auc_score, roc_curve  # type: ignore[import]

    fold_aurocs: list[float] = []

    for held_out_fold in range(1, 6):
        test_recs = [r for r in results if r["fold"] == held_out_fold]
        if not test_recs:
            continue

        y_true = np.array([1 if r["label"] == "deceptive" else 0 for r in test_recs])
        y_score = np.array([r["resonance_score"] for r in test_recs])

        if len(np.unique(y_true)) < 2:
            print(f"  WARNING: fold {held_out_fold} has only one class — skipping")
            continue

        fold_auc = roc_auc_score(y_true, y_score)
        fold_aurocs.append(fold_auc)
        print(f"  Fold {held_out_fold}: AUROC = {fold_auc:.4f}  "
              f"(n={len(test_recs)}, deceptive={int(y_true.sum())})")

    mean_auroc = float(np.mean(fold_aurocs))
    std_auroc = float(np.std(fold_aurocs, ddof=1))

    # Full-dataset ROC for Youden threshold
    y_true_all = np.array([1 if r["label"] == "deceptive" else 0 for r in results])
    y_score_all = np.array([r["resonance_score"] for r in results])
    fpr, tpr, thresholds = roc_curve(y_true_all, y_score_all)
    full_auroc = float(roc_auc_score(y_true_all, y_score_all))
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    optimal_threshold = float(thresholds[best_idx])
    best_tpr = float(tpr[best_idx])
    best_fpr = float(fpr[best_idx])
    best_j = float(youden_j[best_idx])

    dec_scores = [r["resonance_score"] for r in results if r["label"] == "deceptive"]
    tru_scores = [r["resonance_score"] for r in results if r["label"] == "truthful"]

    print(f"\nDeceptive resonance:  mean={np.mean(dec_scores):.4f}  "
          f"std={np.std(dec_scores):.4f}  median={np.median(dec_scores):.4f}")
    print(f"Truthful  resonance:  mean={np.mean(tru_scores):.4f}  "
          f"std={np.std(tru_scores):.4f}  median={np.median(tru_scores):.4f}")
    print(f"\nFull-dataset AUROC:  {full_auroc:.4f}")
    print(f"5-fold CV AUROC:     {mean_auroc:.4f} ± {std_auroc:.4f}")
    print(f"Optimal threshold:   {optimal_threshold:.6f} "
          f"(Youden J={best_j:.4f}, TPR={best_tpr:.3f}, FPR={best_fpr:.3f})")

    pass_threshold = mean_auroc >= 0.85
    verdict = "PASS — brain concept proceeds" if pass_threshold else "FAIL — concept needs redesign"
    print(f"\n{verdict}  (threshold: AUROC ≥ 0.85)")

    return {
        "fold_aurocs": [round(a, 6) for a in fold_aurocs],
        "mean_auroc": round(mean_auroc, 6),
        "std_auroc": round(std_auroc, 6),
        "full_dataset_auroc": round(full_auroc, 6),
        "optimal_threshold": round(optimal_threshold, 6),
        "optimal_threshold_tpr": round(best_tpr, 6),
        "optimal_threshold_fpr": round(best_fpr, 6),
        "youden_j": round(best_j, 6),
        "pass": pass_threshold,
        "deceptive_resonance_mean": round(float(np.mean(dec_scores)), 6),
        "deceptive_resonance_std": round(float(np.std(dec_scores)), 6),
        "truthful_resonance_mean": round(float(np.mean(tru_scores)), 6),
        "truthful_resonance_std": round(float(np.std(tru_scores)), 6),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 1: InterpretabilityProxyBrain — Cornell Deception")
    print("=" * 60)
    print(f"\nDataset:    {DATASET_ROOT}")
    print(f"Scorer:     TRIBE v2 local distilled (GBR + all-mpnet-base-v2)")
    print(f"Model path: {DISTILLED_MODEL_PATH}\n")

    if not DISTILLED_MODEL_PATH.exists():
        print(f"ERROR: distilled model not found at {DISTILLED_MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    t_start = time.time()

    print("[1/3] Loading reviews...")
    records = load_reviews()
    if len(records) == 0:
        print("ERROR: no reviews loaded — check dataset path", file=sys.stderr)
        sys.exit(1)

    print(f"\n[2/3] Scoring {len(records)} reviews (batched)...")
    results = score_reviews_batch(records)

    print(f"\n[3/3] Computing AUROC (5-fold CV)...")
    stats = compute_auroc(results)

    output = {
        "experiment": "phase1_cornell_tribe_v2",
        "date": "2026-04-13",
        "scorer": "tribe_v2_local_distilled",
        "scorer_mode": "distilled_gbr_all-mpnet-base-v2",
        "dataset": "cornell_op_spam_v1.4_positive_polarity",
        "n_reviews": len(results),
        "n_deceptive": sum(1 for r in results if r["label"] == "deceptive"),
        "n_truthful": sum(1 for r in results if r["label"] == "truthful"),
        "total_runtime_seconds": round(time.time() - t_start, 1),
        "evaluation": stats,
        "reviews": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved: {OUTPUT_PATH}")
    print(f"  File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
    print(f"  Total runtime: {output['total_runtime_seconds']}s")


if __name__ == "__main__":
    main()
