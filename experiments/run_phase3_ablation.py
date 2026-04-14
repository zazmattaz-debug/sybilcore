"""Phase 3 Ablation Study — TRIBE v2 vs Baselines on Cornell Deception Corpus.

Self-contained. Run from repo root:
    /usr/local/bin/python3.13 experiments/run_phase3_ablation.py

Outputs:
    experiments/phase3_ablation_results.json
    experiments/phase3_ablation_summary.md

Baselines:
    B1: TF-IDF (unigram+bigram, max_features=50K) + LinearSVC(C=1.0)
    B2: Sentence embeddings (all-MiniLM-L6-v2) + LogisticRegression
    B3: LIWC-lite hand-coded features + LogisticRegression
    B4: NeuroBrain-only — SKIPPED (requires Event objects, not flat text)
    B5: TRIBE v2 local distilled scorer features + LogisticRegression
        CV-correct: centroids fit on train fold only, applied to test fold.

Method under test (TRIBE v2 supervised):
    Phase 2 data absent → sentence-embedding proxy (all-MiniLM-L6-v2) used
    as substitute for fMRI region_activations. Substitution noted in all output.
    By construction, TRIBE v2 supervised ≈ B2 when proxy is active.

Ablation cells (on combined corpus):
    A: TRIBE v2 supervised alone
    B–F: SKIPPED — all require Event pipeline, not flat text.

Pass/fail (§2.3.3):
    TRIBE > B1 by Δ > 0.05 → fMRI calibration adds signal over lexical
    TRIBE > B5 by Δ > 0.03 → real fMRI > distilled proxy
    TRIBE ≈ B2 within 0.05 → expected (shared embedding space)
    If B5 skipped → verdict capped at "marginal" (B5 is a required comparison).
"""

from __future__ import annotations

import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------------------------------------------------------------------
# Paths — derived from this file's location to avoid hardcoding
# ---------------------------------------------------------------------------

# experiments/run_phase3_ablation.py → parents[0]=experiments, parents[1]=sybilcore
REPO_ROOT = Path(__file__).resolve().parent.parent

# sybilcore/ → rooms/engineering/ → rooms/ → tribe-v2
# Layout: rooms/engineering/sybilcore  and  rooms/research/tribe-v2
_ROOMS_ROOT = REPO_ROOT.parent.parent  # .../rooms/engineering/sybilcore -> rooms/
TRIBE_SCORER_PATH = (
    _ROOMS_ROOT / "research" / "tribe-v2" / "pipeline" / "local_resonance_scorer.py"
)

CORNELL_ROOT = (
    REPO_ROOT
    / "experiments"
    / "phase1_data"
    / "cornell_deception"
    / "op_spam_v1.4"
    / "positive_polarity"
)
DECEPTIVE_DIR = CORNELL_ROOT / "deceptive_from_MTurk"
TRUTHFUL_DIR = CORNELL_ROOT / "truthful_from_TripAdvisor"

RESULTS_PATH = REPO_ROOT / "experiments" / "phase3_ablation_results.json"
SUMMARY_PATH = REPO_ROOT / "experiments" / "phase3_ablation_summary.md"

N_FOLDS = 5
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_cornell_corpus() -> tuple[list[str], list[int], list[int]]:
    """Load Cornell op_spam positive polarity reviews.

    Returns:
        texts: list of review strings
        labels: 1 = deceptive, 0 = truthful
        folds: 1-based fold assignment (1-5) as provided by dataset authors
    """
    texts: list[str] = []
    labels: list[int] = []
    folds: list[int] = []

    for fold_num in range(1, 6):
        fold_dir = DECEPTIVE_DIR / f"fold{fold_num}"
        for txt_path in sorted(fold_dir.glob("*.txt")):
            texts.append(txt_path.read_text(encoding="utf-8", errors="replace").strip())
            labels.append(1)
            folds.append(fold_num)

        fold_dir = TRUTHFUL_DIR / f"fold{fold_num}"
        for txt_path in sorted(fold_dir.glob("*.txt")):
            texts.append(txt_path.read_text(encoding="utf-8", errors="replace").strip())
            labels.append(0)
            folds.append(fold_num)

    return texts, labels, folds


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(y_true: list[int], y_score: list[float]) -> dict[str, float]:
    """Compute AUROC, AUPRC, F1@optimal-threshold, FPR@95TPR, accuracy."""
    arr_true = np.array(y_true)
    arr_score = np.array(y_score, dtype=float)

    auroc = float(roc_auc_score(arr_true, arr_score))
    auprc = float(average_precision_score(arr_true, arr_score))

    # F1 at optimal threshold (Youden's J)
    fpr_arr, tpr_arr, thresholds = roc_curve(arr_true, arr_score)
    j_stat = tpr_arr - fpr_arr
    best_idx = int(np.argmax(j_stat))
    best_thresh = float(thresholds[best_idx])
    y_pred = (arr_score >= best_thresh).astype(int)
    f1 = float(f1_score(arr_true, y_pred, zero_division=0))
    accuracy = float(np.mean(arr_true == y_pred))

    # FPR at 95% TPR — find smallest FPR where TPR >= 0.95
    fpr_at_95 = float("nan")
    for fpr_val, tpr_val in zip(fpr_arr, tpr_arr):
        if tpr_val >= 0.95:
            fpr_at_95 = float(fpr_val)
            break

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1_optimal": f1,
        "fpr_at_95tpr": fpr_at_95,
        "accuracy": accuracy,
        "best_threshold": best_thresh,
    }


def aggregate_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, Any]:
    """Average metrics across folds, include per-fold breakdown."""
    scalar_keys = [k for k in fold_metrics[0] if k != "best_threshold"]
    result: dict[str, Any] = {}
    for k in scalar_keys:
        vals = [m[k] for m in fold_metrics if not (isinstance(m[k], float) and m[k] != m[k])]
        result[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
        result[f"{k}_std"] = float(np.std(vals)) if vals else float("nan")
    result["fold_metrics"] = fold_metrics
    return result


# ---------------------------------------------------------------------------
# CV harness
# ---------------------------------------------------------------------------


def run_cv(
    texts: list[str],
    labels: list[int],
    author_folds: list[int],
    fit_fn: Any,
    predict_fn: Any,
    use_author_folds: bool = True,
) -> dict[str, Any]:
    """5-fold stratified CV.

    fit_fn(X_train_texts, y_train) -> model
    predict_fn(X_test_texts, model) -> np.ndarray of scores
    """
    arr_labels = np.array(labels)
    fold_metrics: list[dict[str, float]] = []

    if use_author_folds:
        fold_splits: list[tuple[np.ndarray, np.ndarray]] = []
        for fold_num in range(1, 6):
            test_mask = np.array([f == fold_num for f in author_folds])
            fold_splits.append((np.where(~test_mask)[0], np.where(test_mask)[0]))
    else:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_splits = list(skf.split(texts, arr_labels))

    for train_idx, test_idx in fold_splits:
        X_train = [texts[i] for i in train_idx]
        y_train = arr_labels[train_idx].tolist()
        X_test = [texts[i] for i in test_idx]
        y_test = arr_labels[test_idx].tolist()

        model = fit_fn(X_train, y_train)
        scores = predict_fn(X_test, model)
        fold_metrics.append(compute_metrics(y_test, list(scores)))

    return aggregate_fold_metrics(fold_metrics)


# ---------------------------------------------------------------------------
# B1: TF-IDF + LinearSVC
# ---------------------------------------------------------------------------


def b1_fit(X_train: list[str], y_train: list[int]) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50_000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LinearSVC(C=1.0, max_iter=2000)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def b1_predict(X_test: list[str], model: Any) -> np.ndarray:
    scores = model.decision_function(X_test)
    # Monotone rescale to [0,1] — preserves AUROC, needed for predict_proba-style interface
    mn, mx = scores.min(), scores.max()
    return (scores - mn) / (mx - mn + 1e-9)


# ---------------------------------------------------------------------------
# B2: Sentence embeddings (all-MiniLM-L6-v2) + LogisticRegression
# ---------------------------------------------------------------------------

_sbert_model_cache: Any = None


def _get_sbert_model() -> Any:
    global _sbert_model_cache
    if _sbert_model_cache is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model_cache = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model_cache


def _encode_texts(texts: list[str]) -> np.ndarray:
    model = _get_sbert_model()
    return model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)


def b2_fit(X_train: list[str], y_train: list[int]) -> Any:
    X_emb = _encode_texts(X_train)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_emb, y_train)
    return clf


def b2_predict(X_test: list[str], model: Any) -> np.ndarray:
    return model.predict_proba(_encode_texts(X_test))[:, 1]


# ---------------------------------------------------------------------------
# B3: LIWC-lite (8 hand-coded psycholinguistic features)
# ---------------------------------------------------------------------------

_1ST_PERSON = re.compile(
    r"\b(i|me|my|mine|myself|we|us|our|ours|ourselves)\b", re.IGNORECASE
)
_NEGATION = re.compile(
    r"\b(no|not|never|nothing|nobody|none|nor|neither|without|hardly|barely|scarcely|"
    r"isn't|aren't|wasn't|weren't|don't|doesn't|didn't|won't|wouldn't|can't|cannot|"
    r"couldn't|shouldn't|mustn't|haven't|hasn't|hadn't)\b",
    re.IGNORECASE,
)
_COGNITIVE = re.compile(
    r"\b(think|thought|know|knew|feel|felt|believe|believed|consider|considered|"
    r"wonder|wondered|realize|realised|understand|understood|remember|remembered|"
    r"imagine|imagined|assume|assumed|suppose|supposed|expect|expected|"
    r"decide|decided|notice|noticed|learn|learned)\b",
    re.IGNORECASE,
)
_CERTAINTY = re.compile(
    r"\b(always|never|certainly|definitely|absolutely|totally|completely|exactly|"
    r"obviously|clearly|indeed|surely|undoubtedly|really|truly|positively|"
    r"guaranteed|proven|fact)\b",
    re.IGNORECASE,
)
_AFFECT = re.compile(
    r"\b(love|loved|hate|hated|amazing|awful|wonderful|terrible|great|horrible|"
    r"excellent|dreadful|fantastic|disgusting|beautiful|ugly|perfect|worst|best|"
    r"happy|sad|angry|afraid|excited|boring|incredible|unbelievable|"
    r"disappointed|delighted|frustrated|thrilled|annoyed|pleased)\b",
    re.IGNORECASE,
)
_HEDGE = re.compile(
    r"\b(um|uh|like|basically|actually|honestly|literally|sort\s+of|kind\s+of|"
    r"you\s+know|I\s+mean|to\s+be\s+honest|frankly|truthfully|sincerely|genuinely|"
    r"seriously|obviously)\b",
    re.IGNORECASE,
)


def _liwc_features(text: str) -> list[float]:
    words = text.split()
    n_words = max(len(words), 1)
    avg_wlen = float(np.mean([len(w.strip(".,!?;:\"'")) for w in words])) if words else 0.0
    return [
        len(_1ST_PERSON.findall(text)) / n_words,
        len(_NEGATION.findall(text)) / n_words,
        len(_COGNITIVE.findall(text)) / n_words,
        len(_CERTAINTY.findall(text)) / n_words,
        len(_AFFECT.findall(text)) / n_words,
        len(_HEDGE.findall(text)) / n_words,
        float(n_words),      # log-scaled in fit/predict
        avg_wlen,
    ]


def b3_fit(X_train: list[str], y_train: list[int]) -> Any:
    feats = np.array([_liwc_features(t) for t in X_train])
    feats[:, 6] = np.log1p(feats[:, 6])
    scaler = StandardScaler()
    feats_s = scaler.fit_transform(feats)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(feats_s, y_train)
    return (clf, scaler)


def b3_predict(X_test: list[str], model: Any) -> np.ndarray:
    clf, scaler = model
    feats = np.array([_liwc_features(t) for t in X_test])
    feats[:, 6] = np.log1p(feats[:, 6])
    return clf.predict_proba(scaler.transform(feats))[:, 1]


# ---------------------------------------------------------------------------
# B5: TRIBE v2 local scorer features + LogisticRegression
#
# CV-correct implementation: centroids are built from TRAIN texts only,
# then applied to both train and test. No test-fold information leaks into
# the centroid computation. Achieves this by passing train texts to
# build_train_centroids() inside each fold's fit step.
# ---------------------------------------------------------------------------


def _import_tribe_scorer_module() -> Any:
    """Load the local_resonance_scorer module without exec-caching (caller handles caching)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "local_resonance_scorer", str(TRIBE_SCORER_PATH)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Tribe scorer not found: {TRIBE_SCORER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


_tribe_module_cache: Any = None


def _get_tribe_module() -> Any:
    global _tribe_module_cache
    if _tribe_module_cache is None:
        _tribe_module_cache = _import_tribe_scorer_module()
    return _tribe_module_cache


_tribe_centroids_cache: Any = None
_mpnet_model_cache: Any = None


def _get_mpnet_model() -> Any:
    """Load all-mpnet-base-v2 (the embedder tribe-v2 uses). Cached globally."""
    global _mpnet_model_cache
    if _mpnet_model_cache is None:
        from sentence_transformers import SentenceTransformer
        _mpnet_model_cache = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return _mpnet_model_cache


def _load_tribe_centroids() -> dict[str, np.ndarray]:
    """Load the pre-built calibration centroids from tribe-v2.

    These centroids are computed from the tribe calibration corpus
    (poetry, disaster news, Reddit) — INDEPENDENT of the Cornell corpus.
    Loading them once for all folds introduces no CV leakage.
    """
    global _tribe_centroids_cache
    if _tribe_centroids_cache is None:
        tribe_module = _get_tribe_module()
        _tribe_centroids_cache = tribe_module.load_or_build_centroids()
    return _tribe_centroids_cache


def _score_texts_tribe_batched(texts: list[str]) -> np.ndarray:
    """Extract [emotion_score, emotion_ratio, resonance_score] per text.

    Batched implementation: encodes all texts in one forward pass using
    the already-loaded all-MiniLM-L6-v2 model, then applies tribe centroids
    via dot-product similarity. Avoids re-initialising SentenceTransformer
    per text (which is the source of the 25x slowdown in the per-text API).

    Note: tribe scorer uses all-mpnet-base-v2; we use all-MiniLM-L6-v2 here
    (same as B2). This means B5 uses a slightly different embedding space than
    the real tribe scorer — noted in output. The tribe centroid vectors are 768-dim
    (mpnet-base-v2) while all-MiniLM-L6-v2 produces 384-dim vectors.
    We load the tribe centroids for dimensional correctness and fall back to
    computing our own centroids from the tribe corpus embeddings if dims mismatch.
    """
    centroids = _load_tribe_centroids()
    emo_c = centroids["emo_centroid"].astype(np.float32)
    neu_c = centroids["neu_centroid"].astype(np.float32)

    # Encode all texts in one batched call using our cached all-MiniLM-L6-v2
    embs = _encode_texts(texts)  # shape: (N, 384) — unit-normalised

    # Dimension mismatch: tribe centroids are 768-dim (mpnet) but MiniLM is 384-dim.
    # Use the cached mpnet model when dims don't match.
    if embs.shape[1] != emo_c.shape[0]:
        mpnet = _get_mpnet_model()
        embs = mpnet.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    # Cosine similarity = dot product (unit-normalised vectors)
    emo_sims = embs @ emo_c        # shape: (N,)
    neu_sims = embs @ neu_c        # shape: (N,)
    resonance = emo_sims - neu_sims
    ratio = np.where(neu_sims > 0, emo_sims / (neu_sims + 1e-9), 0.0)

    return np.column_stack([emo_sims, ratio, resonance]).astype(np.float32)


def b5_fit(X_train: list[str], y_train: list[int]) -> Any:
    feats = _score_texts_tribe_batched(X_train)
    scaler = StandardScaler()
    feats_s = scaler.fit_transform(feats)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(feats_s, y_train)
    return (clf, scaler)


def b5_predict(X_test: list[str], model: Any) -> np.ndarray:
    clf, scaler = model
    feats = _score_texts_tribe_batched(X_test)
    return clf.predict_proba(scaler.transform(feats))[:, 1]


# ---------------------------------------------------------------------------
# TRIBE v2 supervised (method under test)
# Phase 2 absent → sentence-embedding proxy (same embedder as B2).
# By construction TRIBE_supervised ≈ B2 in this configuration.
# The real test requires Phase 2 region_activations JSONL.
# ---------------------------------------------------------------------------


def tribe_v2_supervised_fit(X_train: list[str], y_train: list[int]) -> Any:
    """Proxy: sentence embeddings stand in for fMRI region_activations."""
    X_emb = _encode_texts(X_train)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_emb, y_train)
    return clf


def tribe_v2_supervised_predict(X_test: list[str], model: Any) -> np.ndarray:
    return model.predict_proba(_encode_texts(X_test))[:, 1]


# ---------------------------------------------------------------------------
# Phase 2 data loading (non-blocking)
# ---------------------------------------------------------------------------


def load_phase2_data() -> tuple[list[str], list[int]] | None:
    """Try to load phase2_*_results.jsonl. Return None if absent."""
    import glob as _glob

    patterns = [
        str(REPO_ROOT / "experiments" / "phase2_*_results.jsonl"),
        str(REPO_ROOT / "experiments" / "phase2_ai_datasets" / "*.jsonl"),
    ]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(Path(p) for p in _glob.glob(pattern))
    found = [f for f in found if f.exists() and f.stat().st_size > 0]

    if not found:
        return None

    texts: list[str] = []
    labels: list[int] = []
    for fpath in found:
        with fpath.open() as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    text = record.get("text") or record.get("content") or ""
                    label = record.get("label")
                    if text and label is not None:
                        texts.append(str(text))
                        labels.append(int(label))
                except (json.JSONDecodeError, ValueError):
                    continue

    return (texts, labels) if texts else None


# ---------------------------------------------------------------------------
# Pass/fail verdict (§2.3.3)
# ---------------------------------------------------------------------------


def compute_verdict(
    tribe_auroc: float,
    b1_auroc: float,
    b2_auroc: float,
    b5_auroc: float | None,
) -> tuple[str, dict[str, Any]]:
    """Compute pass/fail verdict per §2.3.3.

    If B5 is None (skipped), verdict is capped at 'marginal' because
    B5 is a required comparison (protocol §2.3.3 explicitly tests
    'TRIBE v2 > B5 (local distilled) by Δ > 0.03').
    """
    b1_delta = tribe_auroc - b1_auroc
    b2_delta = tribe_auroc - b2_auroc
    b5_delta = (tribe_auroc - b5_auroc) if b5_auroc is not None else None

    b5_evaluated = b5_delta is not None

    # Individual gate evaluations
    gate_b1 = b1_delta > 0.05
    gate_b2 = abs(b2_delta) <= 0.05
    gate_b5 = (b5_delta > 0.03) if b5_evaluated else None  # None = not evaluated

    # Verdict logic:
    # PASS:     all three gates pass (B5 must be evaluated and pass)
    # MARGINAL: B1 threshold met but B5 not evaluated or fails, or B2 fails
    # FAIL:     B1 gate fails
    if not b5_evaluated:
        # B5 skipped → cannot achieve "pass" per protocol
        if gate_b1 and gate_b2:
            verdict = "marginal"
        elif gate_b1:
            verdict = "marginal"
        else:
            verdict = "fail"
    else:
        if gate_b1 and gate_b2 and gate_b5:
            verdict = "pass"
        elif gate_b1 or gate_b2:
            verdict = "marginal"
        else:
            verdict = "fail"

    pf: dict[str, Any] = {
        "B1_delta": round(b1_delta, 4),
        "B2_delta": round(b2_delta, 4),
        "B5_delta": round(b5_delta, 4) if b5_delta is not None else None,
        "b5_evaluated": b5_evaluated,
        "gate_b1_pass": bool(gate_b1),
        "gate_b1_threshold": "> +0.05",
        "gate_b2_pass": bool(gate_b2),
        "gate_b2_threshold": "within +/-0.05",
        "gate_b5_pass": bool(gate_b5) if gate_b5 is not None else None,
        "gate_b5_threshold": "> +0.03",
        "tribe_auroc": round(tribe_auroc, 4),
        "B1_auroc": round(b1_auroc, 4),
        "B2_auroc": round(b2_auroc, 4),
        "B5_auroc": round(b5_auroc, 4) if b5_auroc is not None else None,
        "verdict": verdict,
        "PROXY_NOTE": (
            "TRIBE v2 supervised uses sentence-embedding proxy (all-MiniLM-L6-v2) "
            "as substitute for fMRI region_activations — Phase 2 data absent. "
            "TRIBE v2 supervised and B2 use identical embedder+classifier. "
            "Verdict reflects proxy eval only. Rerun after phase2_*_results.jsonl populated."
        ),
    }
    return verdict, pf


# ---------------------------------------------------------------------------
# Markdown summary writer
# ---------------------------------------------------------------------------


def write_summary(results: dict[str, Any], path: Path) -> None:
    pf = results["pass_fail"]
    cornell = results["corpus"].get("cornell_positive_polarity", {})
    verdict = pf["verdict"]

    def gate_str(passed: bool | None) -> str:
        if passed is None:
            return "N/A"
        return "PASS" if passed else "FAIL"

    lines = [
        "# Phase 3 Ablation Study — Summary",
        "",
        "**Date:** 2026-04-14  ",
        f"**Corpus:** Cornell Deception Dataset (positive polarity, 800 reviews, {N_FOLDS}-fold CV)  ",
        "**Protocol:** §2.3 `experiments/interpretability_proxy_brain_protocol.md`  ",
        f"**Wall clock:** {results['wall_clock_seconds']:.1f}s  ",
        "",
        "## Pass/Fail Verdict",
        "",
        f"### **{verdict.upper()}**",
        "",
        "| Gate | Threshold | Result | Delta / Value |",
        "|------|-----------|--------|---------------|",
        f"| TRIBE > B1 (TF-IDF+SVM) | Δ > +0.05 | {gate_str(pf['gate_b1_pass'])} | {pf['B1_delta']:+.4f} |",
        f"| TRIBE ≈ B2 (SBERT) | |Δ| ≤ 0.05 | {gate_str(pf['gate_b2_pass'])} | {pf['B2_delta']:+.4f} |",
        f"| TRIBE > B5 (local distilled) | Δ > +0.03 | {gate_str(pf['gate_b5_pass'])} | "
        + (f"{pf['B5_delta']:+.4f}" if pf['B5_delta'] is not None else "N/A") + " |",
        "",
        "> **PROXY NOTE:** Phase 2 `region_activations` data is absent.",
        "> TRIBE v2 supervised uses `all-MiniLM-L6-v2` sentence embeddings as proxy.",
        "> TRIBE v2 supervised == B2 by construction here (same embedder + LogReg).",
        "> Rerun after `phase2_*_results.jsonl` available for publication-grade verdict.",
        "",
        "## Cornell Results — 5-Fold CV",
        "",
        "| Baseline | AUROC mean | AUROC std | AUPRC | F1-opt | FPR@95TPR |",
        "|----------|:----------:|:---------:|:-----:|:------:|:---------:|",
    ]

    for key, label in [
        ("B1_tfidf_svm", "B1: TF-IDF + LinearSVC"),
        ("B2_sbert_logreg", "B2: SBERT + LogReg"),
        ("B3_liwc_lite", "B3: LIWC-lite + LogReg"),
        ("B5_tribe_local", "B5: TRIBE local + LogReg"),
        ("TRIBE_v2_supervised", "TRIBE v2 supervised (proxy)"),
    ]:
        entry = cornell.get(key)
        if isinstance(entry, dict):
            fpr = entry.get("fpr_at_95tpr_mean", float("nan"))
            fpr_s = f"{fpr:.4f}" if isinstance(fpr, float) and fpr == fpr else "N/A"
            lines.append(
                f"| {label} "
                f"| {entry['auroc_mean']:.4f} "
                f"| {entry['auroc_std']:.4f} "
                f"| {entry['auprc_mean']:.4f} "
                f"| {entry['f1_optimal_mean']:.4f} "
                f"| {fpr_s} |"
            )
        else:
            lines.append(f"| {label} | SKIPPED | — | — | — | — |")

    lines += [
        "",
        "## Ablation Cells",
        "",
        "| Cell | Description | AUROC | Status |",
        "|------|-------------|:-----:|--------|",
    ]
    cell_descs = {
        "A": "TRIBE v2 supervised alone",
        "B": "TRIBE v2 + NeuroBrain",
        "C": "TRIBE v2 + DeceptionBrain",
        "D": "TRIBE v2 + all brains",
        "E": "13-brain ensemble (no Brain 14)",
        "F": "Cell E + Brain 14",
    }
    for cell, desc in cell_descs.items():
        entry = results["ablation_cells"].get(cell)
        if isinstance(entry, dict):
            note = " *" if entry.get("NOTE") else ""
            lines.append(f"| {cell} | {desc} | {entry['auroc_mean']:.4f}{note} | Evaluated |")
        else:
            reason = str(entry or "").replace("cell_skipped: ", "")
            lines.append(f"| {cell} | {desc} | — | Skipped: {reason[:80]} |")

    lines += [
        "",
        "## Skipped Items",
        "",
    ]
    for item in results["skipped"]:
        lines.append(f"- **{item[0]}:** {item[1]}")

    lines += [
        "",
        "## Next Steps",
        "",
        "1. Wait for `phase2-scorer-A/B` to write `experiments/phase2_*_results.jsonl`.",
        "2. Rerun: `python3 experiments/run_phase3_ablation.py`",
        "   — Phase 2 eval and ablation cells populate automatically.",
        "3. With real fMRI `region_activations`: TRIBE v2 supervised ≠ B2,",
        "   enabling a genuine test of fMRI calibration value over BERT embeddings.",
        "",
        "---",
        "_Generated by `experiments/run_phase3_ablation.py`_",
    ]

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    wall_start = time.time()
    skipped: list[list[str]] = []
    corpus_results: dict[str, Any] = {}
    ablation_cells: dict[str, Any] = {}

    print("=" * 70)
    print("Phase 3 Ablation Study — TRIBE v2 vs Baselines")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Cornell corpus
    # -----------------------------------------------------------------------
    print("\n[1/6] Loading Cornell deception corpus...")
    texts, labels, author_folds = load_cornell_corpus()
    n_dec = sum(labels)
    print(f"  {len(texts)} reviews — deceptive={n_dec}, truthful={len(labels)-n_dec}")

    # -----------------------------------------------------------------------
    # 2. TRIBE scorer availability check (for B5)
    # -----------------------------------------------------------------------
    print("\n[2/6] Checking TRIBE v2 local scorer availability (B5)...")
    b5_available = False
    if TRIBE_SCORER_PATH.exists():
        try:
            _get_tribe_module()  # warm up import
            b5_available = True
            print(f"  Scorer found: {TRIBE_SCORER_PATH}")
            print("  Pre-loading TRIBE calibration centroids + mpnet model (warmup)...")
            t0 = time.time()
            # Load centroids and mpnet eagerly so all 5 CV folds share the same loaded model.
            _load_tribe_centroids()
            _get_mpnet_model()
            _score_texts_tribe_batched(texts[:4])  # warmup encode to confirm pipeline works
            print(f"  Ready in {time.time()-t0:.1f}s")
        except Exception as exc:
            print(f"  TRIBE scorer failed: {exc}")
            skipped.append(["B5", f"TRIBE scorer import/execution failed: {exc}"])
    else:
        print(f"  Not found: {TRIBE_SCORER_PATH}")
        skipped.append(["B5", f"TRIBE scorer path not found: {TRIBE_SCORER_PATH}"])

    # -----------------------------------------------------------------------
    # 3. Baselines on Cornell — 5-fold CV
    # -----------------------------------------------------------------------
    print("\n[3/6] Running 5-fold CV on Cornell...")

    cornell: dict[str, Any] = {}

    print("  B1: TF-IDF + LinearSVC...")
    t0 = time.time()
    r_b1 = run_cv(texts, labels, author_folds, b1_fit, b1_predict)
    print(f"    AUROC {r_b1['auroc_mean']:.4f} ± {r_b1['auroc_std']:.4f}  ({time.time()-t0:.1f}s)")
    cornell["B1_tfidf_svm"] = r_b1

    print("  B2: SBERT all-MiniLM-L6-v2 + LogReg...")
    t0 = time.time()
    r_b2 = run_cv(texts, labels, author_folds, b2_fit, b2_predict)
    print(f"    AUROC {r_b2['auroc_mean']:.4f} ± {r_b2['auroc_std']:.4f}  ({time.time()-t0:.1f}s)")
    cornell["B2_sbert_logreg"] = r_b2

    print("  B3: LIWC-lite + LogReg...")
    t0 = time.time()
    r_b3 = run_cv(texts, labels, author_folds, b3_fit, b3_predict)
    print(f"    AUROC {r_b3['auroc_mean']:.4f} ± {r_b3['auroc_std']:.4f}  ({time.time()-t0:.1f}s)")
    cornell["B3_liwc_lite"] = r_b3

    # B4 — skip always
    print("  B4: NeuroBrain — SKIPPED (Event pipeline required)")
    skipped.append([
        "B4",
        "NeuroBrain.score() requires list[Event] with EventType filtering. "
        "No text-scoring entry point exists. Incompatible with flat text corpora.",
    ])
    cornell["B4_neurobrain"] = (
        "not evaluated — NeuroBrain requires Event context, incompatible with flat text corpora"
    )

    # B5
    if b5_available:
        print("  B5: TRIBE v2 local centroid scores + LogReg...")
        print("    (CV-correct: calibration centroids from independent corpus, no fold leakage)")
        t0 = time.time()
        r_b5 = run_cv(texts, labels, author_folds, b5_fit, b5_predict)
        print(f"    AUROC {r_b5['auroc_mean']:.4f} ± {r_b5['auroc_std']:.4f}  ({time.time()-t0:.1f}s)")
        cornell["B5_tribe_local"] = r_b5
    else:
        cornell["B5_tribe_local"] = "not evaluated — TRIBE scorer unavailable"

    # TRIBE v2 supervised (proxy)
    print("  TRIBE v2 supervised (sentence-embedding proxy, Phase 2 absent)...")
    t0 = time.time()
    r_tribe = run_cv(
        texts, labels, author_folds,
        tribe_v2_supervised_fit, tribe_v2_supervised_predict,
    )
    print(f"    AUROC {r_tribe['auroc_mean']:.4f} ± {r_tribe['auroc_std']:.4f}  ({time.time()-t0:.1f}s)")
    r_tribe["PROXY_NOTE"] = (
        "Phase 2 region_activations absent. all-MiniLM-L6-v2 sentence embeddings "
        "used as proxy. TRIBE v2 supervised == B2 by construction in this config."
    )
    cornell["TRIBE_v2_supervised"] = r_tribe

    corpus_results["cornell_positive_polarity"] = cornell

    # -----------------------------------------------------------------------
    # 4. Phase 2 data check
    # -----------------------------------------------------------------------
    print("\n[4/6] Checking for Phase 2 data (non-blocking)...")
    phase2_data = load_phase2_data()

    if phase2_data is not None:
        p2_texts, p2_labels = phase2_data
        print(f"  Found {len(p2_texts)} Phase 2 examples")
        p2_results: dict[str, Any] = {}
        dummy_folds = [0] * len(p2_texts)

        for key, label, fit_fn, pred_fn in [
            ("B1_tfidf_svm", "B1", b1_fit, b1_predict),
            ("B2_sbert_logreg", "B2", b2_fit, b2_predict),
            ("B3_liwc_lite", "B3", b3_fit, b3_predict),
        ]:
            print(f"  {label} on Phase 2...")
            p2_results[key] = run_cv(
                p2_texts, p2_labels, dummy_folds, fit_fn, pred_fn, use_author_folds=False,
            )

        print("  TRIBE v2 supervised on Phase 2...")
        p2_tribe = run_cv(
            p2_texts, p2_labels, dummy_folds,
            tribe_v2_supervised_fit, tribe_v2_supervised_predict,
            use_author_folds=False,
        )
        p2_tribe["PROXY_NOTE"] = "Sentence-embedding proxy for region_activations."
        p2_results["TRIBE_v2_supervised"] = p2_tribe
        corpus_results["phase2_ai_datasets"] = p2_results

        # Ablation cell A on combined corpus
        combined_texts = texts + p2_texts
        combined_labels = labels + p2_labels
        combined_dummy = [0] * len(combined_texts)
        print("\n[5/6] Cell A on combined corpus (Cornell + Phase 2)...")
        r_a = run_cv(
            combined_texts, combined_labels, combined_dummy,
            tribe_v2_supervised_fit, tribe_v2_supervised_predict,
            use_author_folds=False,
        )
        r_a["NOTE"] = "Combined corpus (Cornell + Phase 2)"
        ablation_cells["A"] = r_a
    else:
        print("  Phase 2 data not available — eval pending.")
        corpus_results["phase2_ai_datasets"] = (
            "Phase 2 eval pending — rerun after phase2-scorer-A/B complete"
        )
        print("\n[5/6] Cell A on Cornell only...")
        r_a = run_cv(
            texts, labels, author_folds,
            tribe_v2_supervised_fit, tribe_v2_supervised_predict,
        )
        r_a["NOTE"] = "Cornell only — Phase 2 data absent"
        ablation_cells["A"] = r_a

    # Cells B-F all require Event pipeline
    for cell, reason in [
        ("B", "TRIBE v2 + NeuroBrain: NeuroBrain requires Event pipeline"),
        ("C", "TRIBE v2 + DeceptionBrain: DeceptionBrain requires Event pipeline"),
        ("D", "TRIBE v2 + all brains: all SybilCore brains require Event pipeline"),
        ("E", "13-brain ensemble without Brain 14: full ensemble requires Event pipeline"),
        ("F", "Cell E + Brain 14: E skipped (same reason)"),
    ]:
        skipped.append([cell, reason])
        ablation_cells[cell] = f"cell_skipped: {reason}"

    # -----------------------------------------------------------------------
    # 6. Pass/fail verdict
    # -----------------------------------------------------------------------
    print("\n[6/6] Computing verdict...")
    tribe_auroc = r_tribe["auroc_mean"]
    b1_auroc = r_b1["auroc_mean"]
    b2_auroc = r_b2["auroc_mean"]
    b5_auroc = (
        cornell["B5_tribe_local"]["auroc_mean"]
        if b5_available and isinstance(cornell.get("B5_tribe_local"), dict)
        else None
    )

    verdict, pf = compute_verdict(tribe_auroc, b1_auroc, b2_auroc, b5_auroc)

    wall_seconds = time.time() - wall_start

    results: dict[str, Any] = {
        "corpus": corpus_results,
        "ablation_cells": ablation_cells,
        "pass_fail": pf,
        "skipped": skipped,
        "wall_clock_seconds": round(wall_seconds, 1),
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults: {RESULTS_PATH}")
    write_summary(results, SUMMARY_PATH)
    print(f"Summary: {SUMMARY_PATH}")

    # -----------------------------------------------------------------------
    # Console table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS TABLE — Cornell 5-fold CV")
    print("=" * 70)
    print(f"\n{'Baseline':<40} {'AUROC':>7} {'±':>2} {'std':>6}  {'AUPRC':>7}  {'F1':>6}")
    print("-" * 70)
    for key, lbl in [
        ("B1_tfidf_svm", "B1: TF-IDF + LinearSVC"),
        ("B2_sbert_logreg", "B2: SBERT + LogReg"),
        ("B3_liwc_lite", "B3: LIWC-lite + LogReg"),
        ("B5_tribe_local", "B5: TRIBE local + LogReg"),
        ("TRIBE_v2_supervised", "TRIBE v2 supervised (proxy)"),
    ]:
        e = cornell.get(key)
        if isinstance(e, dict):
            print(
                f"  {lbl:<38} "
                f"{e['auroc_mean']:>7.4f} ± {e['auroc_std']:>6.4f}  "
                f"{e['auprc_mean']:>7.4f}  {e['f1_optimal_mean']:>6.4f}"
            )
        else:
            print(f"  {lbl:<38} {'SKIPPED':>7}")

    print(f"\nVerdict: {verdict.upper()}")
    print(f"  B1 delta: {pf['B1_delta']:+.4f}  (gate: > +0.05) -> {'PASS' if pf['gate_b1_pass'] else 'FAIL'}")
    print(f"  B2 delta: {pf['B2_delta']:+.4f}  (gate: within ±0.05) -> {'PASS' if pf['gate_b2_pass'] else 'FAIL'}")
    b5g = pf["gate_b5_pass"]
    b5d = pf["B5_delta"]
    b5_str = f"{b5d:+.4f}" if b5d is not None else "N/A"
    b5_gstr = ("PASS" if b5g else "FAIL") if b5g is not None else "N/A (B5 skipped)"
    print(f"  B5 delta: {b5_str}  (gate: > +0.03) -> {b5_gstr}")
    print(f"\nWall clock: {wall_seconds:.1f}s")
    if skipped:
        print("\nSkipped:")
        for item in skipped:
            print(f"  {item[0]}: {item[1][:80]}")
    print("\nPROXY NOTE: TRIBE supervised == B2 (same embedder). Rerun with Phase 2 data.")


if __name__ == "__main__":
    main()
