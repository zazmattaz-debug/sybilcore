#!/usr/local/bin/python3.13
"""
Phase 1 Scoring: InterpretabilityProxyBrain Experiment
-------------------------------------------------------
Standalone TRIBE v2 proxy using sentence-transformers.

Approach:
- Embed all reviews with all-MiniLM-L6-v2
- Build deception / neutral centroids from training folds
- resonance_score = cos_sim(review, deception_centroid) - cos_sim(review, neutral_centroid)
- 5-fold cross-validation -> AUROC per fold
- Find optimal threshold via Youden's J statistic
- Save results to phase1_cornell_tribe_v2_results.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score, roc_curve

# ── Paths ──────────────────────────────────────────────────────────────────────
EXPERIMENTS_DIR = Path(__file__).parent
DATA_ROOT = EXPERIMENTS_DIR / "phase1_data/cornell_deception/op_spam_v1.4/positive_polarity"
DECEPTIVE_ROOT = DATA_ROOT / "deceptive_from_MTurk"
TRUTHFUL_ROOT  = DATA_ROOT / "truthful_from_TripAdvisor"
OUTPUT_PATH    = EXPERIMENTS_DIR / "phase1_cornell_tribe_v2_results.json"

FOLDS = [f"fold{i}" for i in range(1, 6)]
MODEL_NAME = "all-MiniLM-L6-v2"


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_fold(fold: str) -> tuple[list[str], list[int]]:
    """Load texts and labels for a single fold."""
    texts: list[str] = []
    labels: list[int] = []

    for txt_path in sorted((DECEPTIVE_ROOT / fold).glob("*.txt")):
        texts.append(txt_path.read_text(encoding="utf-8", errors="replace").strip())
        labels.append(1)  # deceptive

    for txt_path in sorted((TRUTHFUL_ROOT / fold).glob("*.txt")):
        texts.append(txt_path.read_text(encoding="utf-8", errors="replace").strip())
        labels.append(0)  # truthful

    return texts, labels


def load_all_folds() -> tuple[list[list[str]], list[list[int]]]:
    all_texts: list[list[str]] = []
    all_labels: list[list[int]] = []
    for fold in FOLDS:
        t, l = load_fold(fold)
        all_texts.append(t)
        all_labels.append(l)
    return all_texts, all_labels


# ── Centroid Scoring ───────────────────────────────────────────────────────────

def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between each row of a and vector b."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return a_norm @ b_norm  # shape: (n,)


def build_centroids(
    embeddings: np.ndarray,
    labels: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (deception_centroid, neutral_centroid) from labeled embeddings."""
    embs = np.array(embeddings)
    lbl  = np.array(labels)
    deception_centroid = embs[lbl == 1].mean(axis=0)
    neutral_centroid   = embs[lbl == 0].mean(axis=0)
    return deception_centroid, neutral_centroid


def resonance_scores(
    embeddings: np.ndarray,
    deception_centroid: np.ndarray,
    neutral_centroid: np.ndarray,
) -> np.ndarray:
    """resonance_score = cos_sim(emb, dec) - cos_sim(emb, neu)"""
    return (
        cosine_similarity_matrix(embeddings, deception_centroid)
        - cosine_similarity_matrix(embeddings, neutral_centroid)
    )


# ── Cross-Validation ───────────────────────────────────────────────────────────

def run_cv(
    all_embeddings: list[np.ndarray],
    all_labels: list[list[int]],
) -> tuple[list[float], np.ndarray, np.ndarray]:
    """
    5-fold CV. Returns:
      - auroc_per_fold
      - all_scores (flat, for global threshold search)
      - all_labels_flat
    """
    auroc_per_fold: list[float] = []
    all_scores_flat: list[float] = []
    all_labels_flat: list[int]   = []

    for held_out in range(5):
        train_embs: list[np.ndarray] = []
        train_lbls: list[int]        = []
        for fold_idx in range(5):
            if fold_idx == held_out:
                continue
            train_embs.append(all_embeddings[fold_idx])
            train_lbls.extend(all_labels[fold_idx])

        train_emb_matrix = np.vstack(train_embs)
        dec_c, neu_c = build_centroids(train_emb_matrix, train_lbls)

        test_emb_matrix = all_embeddings[held_out]
        test_labels     = all_labels[held_out]

        scores = resonance_scores(test_emb_matrix, dec_c, neu_c)
        auroc  = roc_auc_score(test_labels, scores)
        auroc_per_fold.append(float(auroc))
        all_scores_flat.extend(scores.tolist())
        all_labels_flat.extend(test_labels)

        print(f"  Fold {held_out + 1} AUROC: {auroc:.4f}")

    return auroc_per_fold, np.array(all_scores_flat), np.array(all_labels_flat)


# ── Threshold Search ───────────────────────────────────────────────────────────

def find_optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Youden's J: argmax(TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_stat    = tpr - fpr
    best_idx  = int(np.argmax(j_stat))
    return float(thresholds[best_idx])


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Phase 1 Scoring — InterpretabilityProxyBrain")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    # Verify data root
    if not DATA_ROOT.exists():
        print(f"ERROR: Data root not found: {DATA_ROOT}", file=sys.stderr)
        sys.exit(1)

    # Load texts
    print("\n[1/4] Loading reviews...")
    all_texts, all_labels = load_all_folds()
    total = sum(len(t) for t in all_texts)
    n_deceptive = sum(l.count(1) for l in all_labels)
    n_truthful  = sum(l.count(0) for l in all_labels)
    print(f"  Loaded {total} reviews ({n_deceptive} deceptive, {n_truthful} truthful)")

    # Embed
    print(f"\n[2/4] Embedding with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    all_embeddings: list[np.ndarray] = []
    for fold_idx, (texts, _) in enumerate(zip(all_texts, all_labels)):
        embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(embs)
        print(f"  Fold {fold_idx + 1}: {len(texts)} reviews embedded")

    # Cross-validation
    print("\n[3/4] Running 5-fold cross-validation...")
    auroc_per_fold, all_scores, all_labels_flat = run_cv(all_embeddings, all_labels)

    auroc_mean = float(np.mean(auroc_per_fold))
    auroc_std  = float(np.std(auroc_per_fold))

    # Threshold
    print("\n[4/4] Finding optimal threshold (Youden's J)...")
    optimal_threshold = find_optimal_threshold(all_scores, all_labels_flat)
    print(f"  Optimal threshold: {optimal_threshold:.6f}")

    # Results
    passed = auroc_mean > 0.85
    results = {
        "model":             f"sentence-transformers proxy ({MODEL_NAME})",
        "n_samples":         total,
        "n_deceptive":       n_deceptive,
        "n_truthful":        n_truthful,
        "auroc_per_fold":    auroc_per_fold,
        "auroc_mean":        auroc_mean,
        "auroc_std":         auroc_std,
        "optimal_threshold": optimal_threshold,
        "pass":              passed,
    }

    OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model         : {results['model']}")
    print(f"N Samples     : {total} ({n_deceptive} deceptive / {n_truthful} truthful)")
    print(f"AUROC per fold: {[f'{a:.4f}' for a in auroc_per_fold]}")
    print(f"AUROC mean    : {auroc_mean:.4f}")
    print(f"AUROC std     : {auroc_std:.4f}")
    print(f"Optimal thresh: {optimal_threshold:.6f}")
    print(f"Pass (>0.85)  : {'YES ✓' if passed else 'NO ✗'}")
    print(f"\nResults saved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
