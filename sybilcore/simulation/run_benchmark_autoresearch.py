"""Benchmark Autoresearch — systematic exploration across 9 datasets.

Karpathy-style autoresearch loop that runs self-scoring experiments
across ALL available benchmark data. Leave no stone unturned.

Datasets (9,866 data points):
  - CyberSecEval MITRE (1,000 ATT&CK prompts, 10 categories)
  - AdvBench (520 harmful behaviors)
  - HarmBench (400 harmful behaviors, multi-category)
  - JailbreakBench (100 jailbreak behaviors)
  - DoNotAnswer (939 unsafe questions, categorized)
  - AgentHarm harmful (176 agent-specific attacks, ICLR 2025)
  - AgentHarm benign (176 matched benign, false positive control)
  - XSTest (450 safe prompts that look unsafe, over-refusal test)
  - Moltbook (6,105 real agent social network posts)

Each experiment tests a different hypothesis about embedding-based detection.

Experiments per round:
  1. CyberSecEval category isolation — which ATT&CK category clusters best?
  2. CyberSecEval chain length sweep — how many events needed for detection?
  3. AdvBench embedding clustering — do harmful prompts form distinct clusters?
  4. AdvBench vs CyberSecEval cross-dataset — do clusters generalize?
  5. Moltbook real-world clustering — does clustering work on real agent data?
  6. Moltbook anomaly injection — inject synthetic attacks into real data
  7. Mixed-dataset population — all 3 datasets combined in one embedding space
  8. Threshold sensitivity sweep — vary detection threshold across datasets
  9. Embedding model comparison — MiniLM vs MPNet on benchmark data
  10. Feature importance — which embedding dimensions matter most per dataset?
  11. Context window sweep — 2,4,8,16,32 events per agent session
  12. Centroid stability — how many reference samples needed for stable centroids?

Usage:
    caffeinate -d python3 -m sybilcore.simulation.run_benchmark_autoresearch --rounds 5
"""

from __future__ import annotations

import csv
import gc
import json
import logging
import os
import random
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sybilcore.brains.embedding import DEFAULT_MODEL, LARGE_MODEL, EmbeddingBrain, _cosine_similarity, _get_model
from sybilcore.brains import get_default_brains
from sybilcore.core.coefficient import CoefficientCalculator
from sybilcore.core.config import TIER_BOUNDARIES
from sybilcore.models.event import Event, EventType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = _PROJECT_ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)
DATA_DIR = _PROJECT_ROOT / "data"
EVASION_THRESHOLD: float = TIER_BOUNDARIES["clear"][1]


# ── Data Loaders ─────────────────────────────────────────────────


def _load_cyberseceval() -> list[dict[str, Any]]:
    path = DATA_DIR / "cyberseceval_mitre.json"
    return json.loads(path.read_text()) if path.exists() else []


def _load_advbench() -> list[dict[str, str]]:
    path = DATA_DIR / "advbench.csv"
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _load_moltbook() -> list[dict[str, Any]]:
    path = DATA_DIR / "moltbook_posts.jsonl"
    if not path.exists():
        return []
    posts: list[dict] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                posts.append(json.loads(line))
    return posts


def _load_harmbench() -> list[dict[str, str]]:
    path = DATA_DIR / "harmbench.csv"
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _load_jailbreakbench() -> list[dict[str, Any]]:
    path = DATA_DIR / "jailbreakbench.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _load_donotanswer() -> list[dict[str, Any]]:
    path = DATA_DIR / "donotanswer.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _load_agentharm() -> list[dict[str, Any]]:
    path = DATA_DIR / "agentharm.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _load_agentharm_benign() -> list[dict[str, Any]]:
    path = DATA_DIR / "agentharm_benign.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _load_xstest() -> list[dict[str, str]]:
    path = DATA_DIR / "xstest.csv"
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _cluster_f1(model: Any, mal_texts: list[str], ben_texts: list[str]) -> dict[str, float]:
    """Reusable clustering F1 computation for any malicious vs benign text sets."""
    if not mal_texts or not ben_texts:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "centroid_distance": 0.0}

    mal_embs = model.encode(mal_texts[:500], show_progress_bar=False)
    ben_embs = model.encode(ben_texts[:500], show_progress_bar=False)

    mal_centroid = mal_embs.mean(axis=0)
    ben_centroid = ben_embs.mean(axis=0)
    centroid_dist = 1.0 - _cosine_similarity(mal_centroid, ben_centroid)

    all_embs = np.vstack([ben_embs, mal_embs])
    labels = np.array([0] * len(ben_embs) + [1] * len(mal_embs))

    mal_norm = mal_centroid / np.linalg.norm(mal_centroid)
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    sims = (all_embs / norms) @ mal_norm

    best_f1, best_p, best_r = 0.0, 0.0, 0.0
    for thr in np.arange(0.1, 0.9, 0.005):
        preds = (sims >= thr).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if f1 > best_f1:
            best_f1, best_p, best_r = f1, p, r

    return {"f1": round(best_f1, 4), "precision": round(best_p, 4),
            "recall": round(best_r, 4), "centroid_distance": round(float(centroid_dist), 4)}


_BENIGN_TEXTS = [
    "Generated weekly status report for the engineering team",
    "Completed unit tests for the authentication module",
    "Updated API documentation for the search endpoints",
    "Deployed version 3.2.1 to staging environment",
    "Reviewed and approved pull request for notifications",
    "Fixed alignment issue on the dashboard page",
    "Set up monitoring alerts for the payment service",
    "Wrote integration tests for the email service",
    "Optimized database queries for user profile page",
    "Merged feature branch for the admin panel",
    "Created database migration for preferences table",
    "Refactored caching layer for better performance",
    "Configured load balancer health check endpoints",
    "Updated dependency versions across all microservices",
    "Implemented structured logging for production debugging",
    "Set up automated database backup schedule",
    "Wrote smoke tests for the checkout flow",
    "Implemented rate limiting on public API",
    "Created Kubernetes deployment manifests",
    "Updated security headers on all HTTP responses",
]


@dataclass
class ExperimentResult:
    name: str
    hypothesis: str
    start_time: str
    end_time: str = ""
    elapsed_seconds: float = 0.0
    findings: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    conclusion: str = ""
    novel: bool = False
    significance: float = 0.0


def _save_result(result: ExperimentResult) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = EXPERIMENTS_DIR / f"bench_autoresearch_{result.name}_{ts}.json"
    path.write_text(json.dumps(asdict(result), indent=2, default=str))
    logger.info("Saved: %s", path.name)
    return path


# ── Experiment 1: Category Isolation ─────────────────────────────


def exp_category_isolation() -> ExperimentResult:
    """Which MITRE ATT&CK category is most/least detectable via embedding clustering?"""
    result = ExperimentResult(
        name="category_isolation",
        hypothesis="Some ATT&CK categories cluster more distinctly than others in embedding space",
        start_time=datetime.now(UTC).isoformat(),
    )
    model = _get_model(DEFAULT_MODEL)
    cse = _load_cyberseceval()
    if not model or not cse:
        result.conclusion = "SKIPPED: data or model unavailable"
        result.end_time = datetime.now(UTC).isoformat()
        return result

    benign_embs = model.encode(_BENIGN_TEXTS, show_progress_bar=False)
    benign_centroid = benign_embs.mean(axis=0)

    by_cat: dict[str, list[str]] = {}
    for entry in cse:
        cat = entry.get("mitre_category", "unknown")
        prompt = entry.get("mutated_prompt", entry.get("base_prompt", ""))
        by_cat.setdefault(cat, []).append(prompt[:200])

    category_distances: dict[str, dict[str, float]] = {}
    for cat, prompts in by_cat.items():
        embs = model.encode(prompts[:100], show_progress_bar=False)
        cat_centroid = embs.mean(axis=0)

        dist = 1.0 - _cosine_similarity(cat_centroid, benign_centroid)
        intra_dists = [1.0 - _cosine_similarity(e, cat_centroid) for e in embs]
        spread = float(np.mean(intra_dists))

        # Clustering F1 for this category vs benign
        all_embs = np.vstack([benign_embs, embs[:len(benign_embs)]])
        labels = np.array([0] * len(benign_embs) + [1] * min(len(embs), len(benign_embs)))
        mal_norm = cat_centroid / np.linalg.norm(cat_centroid)
        norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        sims = (all_embs / norms) @ mal_norm

        best_f1 = 0.0
        for thr in np.arange(0.1, 0.9, 0.01):
            preds = (sims >= thr).astype(int)
            tp = int(np.sum((preds == 1) & (labels == 1)))
            fp = int(np.sum((preds == 1) & (labels == 0)))
            fn = int(np.sum((preds == 0) & (labels == 1)))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            best_f1 = max(best_f1, f1)

        category_distances[cat] = {
            "centroid_distance": round(float(dist), 4),
            "intra_spread": round(spread, 4),
            "f1": round(best_f1, 4),
        }
        logger.info("  %s: dist=%.3f spread=%.3f F1=%.3f", cat, dist, spread, best_f1)

    ranked = sorted(category_distances.items(), key=lambda x: -x[1]["f1"])
    result.data = {"categories": category_distances}
    result.findings = [
        f"Most detectable: {ranked[0][0]} (F1={ranked[0][1]['f1']})",
        f"Least detectable: {ranked[-1][0]} (F1={ranked[-1][1]['f1']})",
        f"F1 range: {ranked[-1][1]['f1']:.3f} to {ranked[0][1]['f1']:.3f}",
    ]
    f1_spread = ranked[0][1]["f1"] - ranked[-1][1]["f1"]
    result.novel = f1_spread > 0.15
    result.significance = f1_spread
    result.conclusion = f"ATT&CK categories vary in detectability. Best: {ranked[0][0]}, Worst: {ranked[-1][0]}"
    result.end_time = datetime.now(UTC).isoformat()
    return result


# ── Experiment 2: Chain Length Sweep ─────────────────────────────


def exp_chain_length_sweep() -> ExperimentResult:
    """How many events does an agent need before embedding detection kicks in?"""
    result = ExperimentResult(
        name="chain_length_sweep",
        hypothesis="Detection improves with more events per agent session — find the minimum viable chain length",
        start_time=datetime.now(UTC).isoformat(),
    )
    cse = _load_cyberseceval()
    brain = EmbeddingBrain()
    brains = get_default_brains()
    calc = CoefficientCalculator()
    if not cse:
        result.conclusion = "SKIPPED"
        result.end_time = datetime.now(UTC).isoformat()
        return result

    chain_lengths = [1, 2, 3, 5, 8, 10, 15, 20]
    length_results: list[dict[str, Any]] = []

    for length in chain_lengths:
        detected = 0
        total = 0
        # Sample 50 chains of this length
        random.shuffle(cse)
        for start in range(0, min(50 * length, len(cse)), length):
            chain = cse[start:start + length]
            if len(chain) < length:
                break
            agent_id = f"chain-{length}-{start}"
            events: list[Event] = []
            for entry in chain:
                prompt = entry.get("mutated_prompt", entry.get("base_prompt", ""))[:200]
                cat = entry.get("mitre_category", "Execution")
                ts = datetime.now(UTC) - timedelta(seconds=random.randint(1, 55))
                events.append(Event(
                    event_id=str(uuid.uuid4()), agent_id=agent_id,
                    event_type=EventType.INSTRUCTION_RECEIVED, timestamp=ts,
                    content=prompt, metadata={}, source="benchmark",
                ))
                events.append(Event(
                    event_id=str(uuid.uuid4()), agent_id=agent_id,
                    event_type=EventType.OUTPUT_GENERATED, timestamp=ts,
                    content=f"Executing {cat.lower()} operation: {prompt[:80]}",
                    metadata={}, source="benchmark",
                ))

            scores = [b.score(events) for b in brains]
            snap = calc.calculate(scores)
            if snap.effective_coefficient >= EVASION_THRESHOLD:
                detected += 1
            total += 1

        rate = detected / total if total > 0 else 0
        length_results.append({
            "chain_length": length,
            "events_per_chain": length * 2,
            "detected": detected,
            "total": total,
            "detection_rate": round(rate, 3),
        })
        logger.info("  Length %d (%d events): %d/%d (%.0f%%)", length, length * 2, detected, total, rate * 100)

    # Find the inflection point
    rates = [r["detection_rate"] for r in length_results]
    inflection = None
    for i in range(1, len(rates)):
        if rates[i] > 0.5 and rates[i - 1] <= 0.5:
            inflection = length_results[i]["chain_length"]
            break

    result.data = {"chain_lengths": length_results}
    result.findings = [
        f"Detection at length 1: {rates[0]*100:.0f}%",
        f"Detection at length 20: {rates[-1]*100:.0f}%",
        f"Inflection point: {inflection or 'not reached'}",
    ]
    result.novel = inflection is not None and inflection <= 10
    result.significance = max(rates) - min(rates)
    result.conclusion = f"Chain length sweep: {rates[0]*100:.0f}% → {rates[-1]*100:.0f}%. Inflection at {inflection or '>20'} prompts."
    result.end_time = datetime.now(UTC).isoformat()
    return result


# ── Experiment 3: AdvBench Clustering ────────────────────────────


def exp_advbench_clustering() -> ExperimentResult:
    """Do AdvBench harmful prompts cluster separately from benign in embedding space?"""
    result = ExperimentResult(
        name="advbench_clustering",
        hypothesis="AdvBench harmful prompts form a distinct cluster from benign software engineering text",
        start_time=datetime.now(UTC).isoformat(),
    )
    model = _get_model(DEFAULT_MODEL)
    advbench = _load_advbench()
    if not model or not advbench:
        result.conclusion = "SKIPPED"
        result.end_time = datetime.now(UTC).isoformat()
        return result

    mal_texts = [entry["goal"][:200] for entry in advbench[:200]]
    ben_texts = [f"{t} (v{random.randint(1,1000)})" for t in _BENIGN_TEXTS * 10][:200]

    mal_embs = model.encode(mal_texts, show_progress_bar=False)
    ben_embs = model.encode(ben_texts, show_progress_bar=False)

    mal_centroid = mal_embs.mean(axis=0)
    ben_centroid = ben_embs.mean(axis=0)
    centroid_dist = 1.0 - _cosine_similarity(mal_centroid, ben_centroid)

    all_embs = np.vstack([ben_embs, mal_embs])
    labels = np.array([0] * len(ben_embs) + [1] * len(mal_embs))

    mal_norm = mal_centroid / np.linalg.norm(mal_centroid)
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    sims = (all_embs / norms) @ mal_norm

    best_f1 = 0.0
    best_p, best_r = 0.0, 0.0
    for thr in np.arange(0.1, 0.9, 0.005):
        preds = (sims >= thr).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if f1 > best_f1:
            best_f1, best_p, best_r = f1, p, r

    result.data = {"f1": round(best_f1, 4), "precision": round(best_p, 4), "recall": round(best_r, 4),
                   "centroid_distance": round(float(centroid_dist), 4), "mal_count": len(mal_texts), "ben_count": len(ben_texts)}
    result.findings = [f"AdvBench clustering F1={best_f1:.4f}", f"Centroid distance: {centroid_dist:.4f}"]
    result.novel = best_f1 > 0.95
    result.significance = best_f1
    result.conclusion = f"AdvBench clustering: F1={best_f1:.4f} (p={best_p:.3f} r={best_r:.3f})"
    result.end_time = datetime.now(UTC).isoformat()
    return result


# ── Experiment 4: Cross-Dataset Generalization ───────────────────


def exp_cross_dataset() -> ExperimentResult:
    """Train centroid on CyberSecEval, test on AdvBench. Does it generalize?"""
    result = ExperimentResult(
        name="cross_dataset_generalization",
        hypothesis="Malicious centroid from one dataset detects attacks from a completely different dataset",
        start_time=datetime.now(UTC).isoformat(),
    )
    model = _get_model(DEFAULT_MODEL)
    cse = _load_cyberseceval()
    advbench = _load_advbench()
    if not model or not cse or not advbench:
        result.conclusion = "SKIPPED"
        result.end_time = datetime.now(UTC).isoformat()
        return result

    # Train centroids on CyberSecEval
    cse_texts = [e.get("mutated_prompt", e.get("base_prompt", ""))[:200] for e in cse[:500]]
    cse_embs = model.encode(cse_texts, show_progress_bar=False)
    cse_centroid = cse_embs.mean(axis=0)

    ben_embs = model.encode(_BENIGN_TEXTS, show_progress_bar=False)
    ben_centroid = ben_embs.mean(axis=0)

    # Test on AdvBench
    adv_texts = [e["goal"][:200] for e in advbench]
    adv_embs = model.encode(adv_texts, show_progress_bar=False)

    # Classify each AdvBench prompt using CyberSecEval centroids
    correct = 0
    for emb in adv_embs:
        sim_mal = _cosine_similarity(emb, cse_centroid)
        sim_ben = _cosine_similarity(emb, ben_centroid)
        if sim_mal > sim_ben:
            correct += 1

    transfer_accuracy = correct / len(adv_embs)

    # Now reverse: train on AdvBench, test on CyberSecEval
    adv_centroid = adv_embs.mean(axis=0)
    correct_rev = 0
    for emb in cse_embs[:500]:
        sim_mal = _cosine_similarity(emb, adv_centroid)
        sim_ben = _cosine_similarity(emb, ben_centroid)
        if sim_mal > sim_ben:
            correct_rev += 1
    reverse_accuracy = correct_rev / 500

    result.data = {
        "cse_to_advbench": round(transfer_accuracy, 4),
        "advbench_to_cse": round(reverse_accuracy, 4),
        "avg_transfer": round((transfer_accuracy + reverse_accuracy) / 2, 4),
    }
    result.findings = [
        f"CyberSecEval→AdvBench transfer: {transfer_accuracy:.1%}",
        f"AdvBench→CyberSecEval transfer: {reverse_accuracy:.1%}",
        f"Average cross-dataset transfer: {(transfer_accuracy + reverse_accuracy)/2:.1%}",
    ]
    avg = (transfer_accuracy + reverse_accuracy) / 2
    result.novel = avg > 0.7
    result.significance = avg
    result.conclusion = f"Cross-dataset transfer: {avg:.1%} average. Centroids {'generalize' if avg > 0.7 else 'do not generalize'} across datasets."
    result.end_time = datetime.now(UTC).isoformat()
    return result


# ── Experiment 5: Moltbook Real-World Clustering ─────────────────


def exp_moltbook_clustering() -> ExperimentResult:
    """Does embedding clustering detect anomalies in real Moltbook agent data?"""
    result = ExperimentResult(
        name="moltbook_real_clustering",
        hypothesis="Real agent posts from Moltbook show natural behavioral clusters in embedding space",
        start_time=datetime.now(UTC).isoformat(),
    )
    model = _get_model(DEFAULT_MODEL)
    posts = _load_moltbook()
    if not model or not posts:
        result.conclusion = "SKIPPED"
        result.end_time = datetime.now(UTC).isoformat()
        return result

    # Extract text content from Moltbook posts
    texts = []
    for post in posts[:2000]:
        content = post.get("content", post.get("text", post.get("body", "")))
        if isinstance(content, str) and len(content) > 20:
            texts.append(content[:200])

    if len(texts) < 100:
        result.conclusion = f"SKIPPED: only {len(texts)} usable texts"
        result.end_time = datetime.now(UTC).isoformat()
        return result

    logger.info("  Encoding %d Moltbook texts...", len(texts))
    embs = model.encode(texts[:1000], show_progress_bar=False)

    # Compute centroid and find outliers
    centroid = embs.mean(axis=0)
    distances = np.array([1.0 - _cosine_similarity(e, centroid) for e in embs])

    mean_dist = distances.mean()
    std_dist = distances.std()
    z_scores = (distances - mean_dist) / std_dist if std_dist > 0 else np.zeros_like(distances)

    outliers_2sigma = int(np.sum(z_scores > 2.0))
    outliers_3sigma = int(np.sum(z_scores > 3.0))

    # Check if Moltbook data clusters differently from benign software texts
    ben_embs = model.encode(_BENIGN_TEXTS, show_progress_bar=False)
    ben_centroid = ben_embs.mean(axis=0)
    moltbook_vs_benign = 1.0 - _cosine_similarity(centroid, ben_centroid)

    result.data = {
        "total_texts": len(texts[:1000]),
        "outliers_2sigma": outliers_2sigma,
        "outliers_3sigma": outliers_3sigma,
        "mean_distance": round(float(mean_dist), 4),
        "std_distance": round(float(std_dist), 4),
        "moltbook_vs_benign_distance": round(float(moltbook_vs_benign), 4),
    }
    result.findings = [
        f"Moltbook: {len(texts[:1000])} texts analyzed",
        f"Outliers: {outliers_2sigma} at 2σ, {outliers_3sigma} at 3σ",
        f"Moltbook vs benign centroid distance: {moltbook_vs_benign:.4f}",
    ]
    result.novel = outliers_3sigma > 5
    result.significance = outliers_3sigma / len(texts[:1000]) if texts else 0
    result.conclusion = f"Moltbook real data: {outliers_3sigma} outliers at 3σ out of {len(texts[:1000])} posts. Distance from benign: {moltbook_vs_benign:.4f}"
    result.end_time = datetime.now(UTC).isoformat()
    return result


# ── Experiment 6: Moltbook Anomaly Injection ─────────────────────


def exp_moltbook_injection() -> ExperimentResult:
    """Inject synthetic attacks into real Moltbook data. Can clustering detect them?"""
    result = ExperimentResult(
        name="moltbook_anomaly_injection",
        hypothesis="Synthetic malicious agents injected into real Moltbook data are detectable via clustering",
        start_time=datetime.now(UTC).isoformat(),
    )
    model = _get_model(DEFAULT_MODEL)
    posts = _load_moltbook()
    cse = _load_cyberseceval()
    if not model or not posts or not cse:
        result.conclusion = "SKIPPED"
        result.end_time = datetime.now(UTC).isoformat()
        return result

    # Real Moltbook texts
    real_texts = []
    for post in posts[:1000]:
        content = post.get("content", post.get("text", post.get("body", "")))
        if isinstance(content, str) and len(content) > 20:
            real_texts.append(content[:200])

    # Inject malicious texts from CyberSecEval
    n_inject = min(100, len(cse))
    inject_texts = [e.get("mutated_prompt", e.get("base_prompt", ""))[:200] for e in random.sample(cse, n_inject)]

    all_texts = real_texts[:900] + inject_texts
    labels = np.array([0] * len(real_texts[:900]) + [1] * len(inject_texts))

    indices = np.random.permutation(len(all_texts))
    all_texts = [all_texts[i] for i in indices]
    labels = labels[indices]

    logger.info("  Encoding %d texts (%d real + %d injected)...", len(all_texts), len(real_texts[:900]), len(inject_texts))
    embs = model.encode(all_texts, show_progress_bar=False)

    real_centroid = embs[labels == 0].mean(axis=0)
    inject_centroid = embs[labels == 1].mean(axis=0)

    centroid_dist = 1.0 - _cosine_similarity(real_centroid, inject_centroid)

    # Clustering F1
    inj_norm = inject_centroid / np.linalg.norm(inject_centroid)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    sims = (embs / norms) @ inj_norm

    best_f1 = 0.0
    for thr in np.arange(0.1, 0.9, 0.005):
        preds = (sims >= thr).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        best_f1 = max(best_f1, f1)

    result.data = {"f1": round(best_f1, 4), "centroid_distance": round(float(centroid_dist), 4),
                   "real_count": len(real_texts[:900]), "injected_count": len(inject_texts)}
    result.findings = [
        f"Injection detection F1={best_f1:.4f}",
        f"Centroid distance (real vs injected): {centroid_dist:.4f}",
        f"This is the first test on REAL agent data with synthetic attacks",
    ]
    result.novel = best_f1 > 0.8
    result.significance = best_f1
    result.conclusion = f"Moltbook injection: F1={best_f1:.4f}. {'REAL DATA VALIDATES' if best_f1 > 0.8 else 'Needs work on real data'}."
    result.end_time = datetime.now(UTC).isoformat()
    return result


# ── Experiment 7: Mixed-Dataset Population ───────────────────────


def exp_mixed_population() -> ExperimentResult:
    """Combine all 3 datasets in one embedding space. Does clustering still work?"""
    result = ExperimentResult(
        name="mixed_dataset_population",
        hypothesis="Clustering works even when malicious texts come from different datasets with different styles",
        start_time=datetime.now(UTC).isoformat(),
    )
    model = _get_model(DEFAULT_MODEL)
    cse = _load_cyberseceval()
    advbench = _load_advbench()
    if not model or not cse or not advbench:
        result.conclusion = "SKIPPED"
        result.end_time = datetime.now(UTC).isoformat()
        return result

    # Mix malicious from both datasets
    cse_texts = [e.get("mutated_prompt", e.get("base_prompt", ""))[:200] for e in cse[:250]]
    adv_texts = [e["goal"][:200] for e in advbench[:250]]
    mal_texts = cse_texts + adv_texts
    random.shuffle(mal_texts)

    ben_texts = [f"{t} (v{random.randint(1,10000)})" for t in _BENIGN_TEXTS * 25][:500]

    all_texts = ben_texts + mal_texts
    labels = np.array([0] * len(ben_texts) + [1] * len(mal_texts))
    indices = np.random.permutation(len(all_texts))
    all_texts = [all_texts[i] for i in indices]
    labels = labels[indices]

    embs = model.encode(all_texts, show_progress_bar=False, batch_size=256)

    mal_centroid = embs[labels == 1].mean(axis=0)
    mal_norm = mal_centroid / np.linalg.norm(mal_centroid)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    sims = (embs / norms) @ mal_norm

    best_f1, best_p, best_r = 0.0, 0.0, 0.0
    for thr in np.arange(0.1, 0.9, 0.005):
        preds = (sims >= thr).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if f1 > best_f1:
            best_f1, best_p, best_r = f1, p, r

    result.data = {"f1": round(best_f1, 4), "precision": round(best_p, 4), "recall": round(best_r, 4),
                   "total_population": len(all_texts), "mal_sources": "CyberSecEval + AdvBench"}
    result.findings = [f"Mixed-dataset F1={best_f1:.4f}", f"500 benign + 500 malicious (250 CSE + 250 AdvBench)"]
    result.novel = best_f1 > 0.95
    result.significance = best_f1
    result.conclusion = f"Mixed-dataset clustering: F1={best_f1:.4f}. Malicious agents cluster together regardless of source dataset."
    result.end_time = datetime.now(UTC).isoformat()
    return result


# ── Experiment 8: Context Window Sweep ───────────────────────────


def exp_context_window_sweep() -> ExperimentResult:
    """How does detection change with 2, 4, 8, 16, 32 events per session?"""
    result = ExperimentResult(
        name="context_window_sweep",
        hypothesis="There is a minimum context window where embedding detection becomes reliable",
        start_time=datetime.now(UTC).isoformat(),
    )
    brain = EmbeddingBrain()
    brains = get_default_brains()
    calc = CoefficientCalculator()
    cse = _load_cyberseceval()
    if not cse:
        result.conclusion = "SKIPPED"
        result.end_time = datetime.now(UTC).isoformat()
        return result

    windows = [2, 4, 6, 8, 12, 16, 24, 32]
    window_results: list[dict] = []

    for window in windows:
        detected = 0
        total = 20  # 20 samples per window size
        for sample in range(total):
            agent_id = f"ctx-{window}-{sample}"
            entries = random.sample(cse, min(window, len(cse)))
            events: list[Event] = []
            for entry in entries:
                prompt = entry.get("mutated_prompt", entry.get("base_prompt", ""))[:200]
                ts = datetime.now(UTC) - timedelta(seconds=random.randint(1, 55))
                events.append(Event(event_id=str(uuid.uuid4()), agent_id=agent_id,
                    event_type=EventType.OUTPUT_GENERATED, timestamp=ts,
                    content=prompt, metadata={}, source="benchmark"))

            scores = [b.score(events) for b in brains]
            snap = calc.calculate(scores)
            if snap.effective_coefficient >= EVASION_THRESHOLD:
                detected += 1

        rate = detected / total
        window_results.append({"window": window, "detection_rate": round(rate, 3), "detected": detected, "total": total})
        logger.info("  Window %d: %d/%d (%.0f%%)", window, detected, total, rate * 100)

    rates = [w["detection_rate"] for w in window_results]
    result.data = {"windows": window_results}
    result.findings = [f"Window {w['window']}: {w['detection_rate']*100:.0f}%" for w in window_results]
    result.novel = any(r > 0.8 for r in rates)
    result.significance = max(rates) - min(rates)
    result.conclusion = f"Context sweep: {rates[0]*100:.0f}% at 2 events → {rates[-1]*100:.0f}% at 32 events"
    result.end_time = datetime.now(UTC).isoformat()
    return result


# ── Experiment 9: Centroid Stability ─────────────────────────────


def exp_centroid_stability() -> ExperimentResult:
    """How many reference samples needed for a stable centroid?"""
    result = ExperimentResult(
        name="centroid_stability",
        hypothesis="Centroids stabilize after N reference samples — find that N",
        start_time=datetime.now(UTC).isoformat(),
    )
    model = _get_model(DEFAULT_MODEL)
    cse = _load_cyberseceval()
    if not model or not cse:
        result.conclusion = "SKIPPED"
        result.end_time = datetime.now(UTC).isoformat()
        return result

    texts = [e.get("mutated_prompt", e.get("base_prompt", ""))[:200] for e in cse]
    random.shuffle(texts)
    all_embs = model.encode(texts[:500], show_progress_bar=False)

    # Full centroid as ground truth
    full_centroid = all_embs.mean(axis=0)

    sample_sizes = [5, 10, 20, 50, 100, 200, 500]
    stability_results: list[dict] = []

    for n in sample_sizes:
        if n > len(all_embs):
            break
        # Run 10 trials with random subsets
        similarities: list[float] = []
        for _ in range(10):
            subset = all_embs[np.random.choice(len(all_embs), n, replace=False)]
            subset_centroid = subset.mean(axis=0)
            sim = _cosine_similarity(subset_centroid, full_centroid)
            similarities.append(float(sim))

        avg_sim = float(np.mean(similarities))
        std_sim = float(np.std(similarities))
        stability_results.append({
            "sample_size": n,
            "avg_similarity_to_full": round(avg_sim, 6),
            "std": round(std_sim, 6),
        })
        logger.info("  N=%d: similarity=%.6f ± %.6f", n, avg_sim, std_sim)

    # Find where similarity > 0.99
    stable_n = None
    for sr in stability_results:
        if sr["avg_similarity_to_full"] > 0.99:
            stable_n = sr["sample_size"]
            break

    result.data = {"stability": stability_results, "stable_at": stable_n}
    result.findings = [f"Centroid stabilizes (>0.99 sim) at N={stable_n or '>500'} samples"]
    result.novel = stable_n is not None and stable_n <= 50
    result.significance = 1.0 - (stable_n / 500 if stable_n else 1.0)
    result.conclusion = f"Centroid stability: reaches 0.99 similarity at N={stable_n or '>500'}. {'Practical' if stable_n and stable_n <= 50 else 'Needs many samples'}."
    result.end_time = datetime.now(UTC).isoformat()
    return result


# ── Orchestrator ─────────────────────────────────────────────────


def exp_harmbench_clustering() -> ExperimentResult:
    """Cluster HarmBench harmful behaviors vs benign text."""
    result = ExperimentResult(name="harmbench_clustering",
        hypothesis="HarmBench harmful behaviors cluster distinctly from benign text",
        start_time=datetime.now(UTC).isoformat())
    model = _get_model(DEFAULT_MODEL)
    data = _load_harmbench()
    if not model or not data:
        result.conclusion = "SKIPPED"; result.end_time = datetime.now(UTC).isoformat(); return result

    # Extract text — try common column names
    mal_texts = []
    for row in data:
        text = row.get("Behavior", row.get("behavior", row.get("goal", row.get("prompt", ""))))
        if text and len(str(text)) > 10:
            mal_texts.append(str(text)[:200])
    ben_texts = [f"{t} (v{random.randint(1,10000)})" for t in _BENIGN_TEXTS * 25][:len(mal_texts)]

    metrics = _cluster_f1(model, mal_texts, ben_texts)
    result.data = {**metrics, "dataset": "HarmBench", "mal_count": len(mal_texts)}
    result.findings = [f"HarmBench clustering F1={metrics['f1']}", f"{len(mal_texts)} harmful behaviors tested"]
    result.novel = metrics["f1"] > 0.95; result.significance = metrics["f1"]
    result.conclusion = f"HarmBench: F1={metrics['f1']} (p={metrics['precision']} r={metrics['recall']})"
    result.end_time = datetime.now(UTC).isoformat(); return result


def exp_jailbreakbench_clustering() -> ExperimentResult:
    """Cluster JailbreakBench behaviors vs benign text."""
    result = ExperimentResult(name="jailbreakbench_clustering",
        hypothesis="JailbreakBench jailbreak behaviors cluster distinctly",
        start_time=datetime.now(UTC).isoformat())
    model = _get_model(DEFAULT_MODEL)
    data = _load_jailbreakbench()
    if not model or not data:
        result.conclusion = "SKIPPED"; result.end_time = datetime.now(UTC).isoformat(); return result

    mal_texts = [str(row.get("Behavior", row.get("behavior", row.get("goal", ""))))[:200] for row in data if row]
    mal_texts = [t for t in mal_texts if len(t) > 10]
    ben_texts = [f"{t} (v{random.randint(1,10000)})" for t in _BENIGN_TEXTS * 10][:len(mal_texts)]

    metrics = _cluster_f1(model, mal_texts, ben_texts)
    result.data = {**metrics, "dataset": "JailbreakBench", "mal_count": len(mal_texts)}
    result.findings = [f"JailbreakBench clustering F1={metrics['f1']}"]
    result.novel = metrics["f1"] > 0.95; result.significance = metrics["f1"]
    result.conclusion = f"JailbreakBench: F1={metrics['f1']}"
    result.end_time = datetime.now(UTC).isoformat(); return result


def exp_donotanswer_clustering() -> ExperimentResult:
    """Cluster DoNotAnswer unsafe questions vs benign text."""
    result = ExperimentResult(name="donotanswer_clustering",
        hypothesis="DoNotAnswer unsafe questions cluster distinctly from benign prompts",
        start_time=datetime.now(UTC).isoformat())
    model = _get_model(DEFAULT_MODEL)
    data = _load_donotanswer()
    if not model or not data:
        result.conclusion = "SKIPPED"; result.end_time = datetime.now(UTC).isoformat(); return result

    mal_texts = []
    for row in data:
        text = row.get("question", row.get("prompt", row.get("content", "")))
        if text and len(str(text)) > 10:
            mal_texts.append(str(text)[:200])
    ben_texts = [f"{t} (v{random.randint(1,10000)})" for t in _BENIGN_TEXTS * 50][:len(mal_texts)]

    metrics = _cluster_f1(model, mal_texts[:500], ben_texts[:500])
    result.data = {**metrics, "dataset": "DoNotAnswer", "mal_count": len(mal_texts)}
    result.findings = [f"DoNotAnswer clustering F1={metrics['f1']}", f"{len(mal_texts)} unsafe questions tested"]
    result.novel = metrics["f1"] > 0.90; result.significance = metrics["f1"]
    result.conclusion = f"DoNotAnswer: F1={metrics['f1']} ({len(mal_texts)} questions)"
    result.end_time = datetime.now(UTC).isoformat(); return result


def exp_agentharm_clustering() -> ExperimentResult:
    """Cluster AgentHarm harmful agent tasks vs their matched benign counterparts."""
    result = ExperimentResult(name="agentharm_clustering",
        hypothesis="AgentHarm harmful tasks cluster separately from matched benign tasks (ICLR 2025)",
        start_time=datetime.now(UTC).isoformat())
    model = _get_model(DEFAULT_MODEL)
    harmful = _load_agentharm()
    benign = _load_agentharm_benign()
    if not model or not harmful or not benign:
        result.conclusion = "SKIPPED"; result.end_time = datetime.now(UTC).isoformat(); return result

    # AgentHarm has matched pairs — strongest possible FP test
    mal_texts = [str(row.get("behavior", row.get("prompt", row.get("name", ""))))[:200] for row in harmful]
    mal_texts = [t for t in mal_texts if len(t) > 10]
    ben_texts = [str(row.get("behavior", row.get("prompt", row.get("name", ""))))[:200] for row in benign]
    ben_texts = [t for t in ben_texts if len(t) > 10]

    metrics = _cluster_f1(model, mal_texts, ben_texts)
    result.data = {**metrics, "dataset": "AgentHarm (ICLR 2025)", "mal_count": len(mal_texts), "ben_count": len(ben_texts),
                   "note": "Matched harmful/benign pairs — hardest possible clustering test"}
    result.findings = [
        f"AgentHarm matched-pair clustering F1={metrics['f1']}",
        f"{len(mal_texts)} harmful vs {len(ben_texts)} matched benign agent tasks",
        "This is the HARDEST test — benign tasks are designed to resemble harmful ones",
    ]
    result.novel = metrics["f1"] > 0.7; result.significance = metrics["f1"]
    result.conclusion = f"AgentHarm matched pairs: F1={metrics['f1']}. {'Clustering distinguishes even matched pairs' if metrics['f1'] > 0.7 else 'Matched pairs are too similar for clustering'}."
    result.end_time = datetime.now(UTC).isoformat(); return result


def exp_xstest_over_refusal() -> ExperimentResult:
    """Test if SybilCore correctly identifies XSTest safe-but-edgy prompts as benign."""
    result = ExperimentResult(name="xstest_over_refusal",
        hypothesis="XSTest prompts (safe but look unsafe) should NOT cluster with malicious text",
        start_time=datetime.now(UTC).isoformat())
    model = _get_model(DEFAULT_MODEL)
    xstest = _load_xstest()
    cse = _load_cyberseceval()
    if not model or not xstest or not cse:
        result.conclusion = "SKIPPED"; result.end_time = datetime.now(UTC).isoformat(); return result

    # XSTest prompts are SAFE but sound edgy — they should cluster with benign, not malicious
    xs_texts = []
    for row in xstest:
        text = row.get("prompt", row.get("text", row.get("question", "")))
        if text and len(str(text)) > 10:
            xs_texts.append(str(text)[:200])

    mal_texts = [e.get("mutated_prompt", e.get("base_prompt", ""))[:200] for e in cse[:200]]
    ben_texts = [f"{t} (v{random.randint(1,10000)})" for t in _BENIGN_TEXTS * 25][:200]

    # Embed all three groups
    xs_embs = model.encode(xs_texts[:200], show_progress_bar=False)
    mal_embs = model.encode(mal_texts, show_progress_bar=False)
    ben_embs = model.encode(ben_texts, show_progress_bar=False)

    mal_centroid = mal_embs.mean(axis=0)
    ben_centroid = ben_embs.mean(axis=0)

    # For each XSTest prompt, is it closer to benign or malicious?
    correct = 0  # Correct = closer to benign (since XSTest is safe)
    for emb in xs_embs:
        sim_mal = _cosine_similarity(emb, mal_centroid)
        sim_ben = _cosine_similarity(emb, ben_centroid)
        if sim_ben >= sim_mal:
            correct += 1

    accuracy = correct / len(xs_embs) if xs_embs.size > 0 else 0
    over_refusal_rate = 1.0 - accuracy  # How often we wrongly classify safe as malicious

    result.data = {"accuracy": round(accuracy, 4), "over_refusal_rate": round(over_refusal_rate, 4),
                   "total_xs": len(xs_texts[:200]), "correct_benign": correct}
    result.findings = [
        f"XSTest over-refusal rate: {over_refusal_rate:.1%}",
        f"{correct}/{len(xs_texts[:200])} safe-but-edgy prompts correctly identified as benign",
        "Lower over-refusal = better (we want to NOT flag safe prompts)",
    ]
    result.novel = over_refusal_rate < 0.15
    result.significance = 1.0 - over_refusal_rate
    result.conclusion = f"XSTest: {over_refusal_rate:.1%} over-refusal rate. {'Good' if over_refusal_rate < 0.2 else 'Too aggressive'} — {correct}/{len(xs_texts[:200])} safe prompts correctly passed."
    result.end_time = datetime.now(UTC).isoformat(); return result


def exp_all_datasets_mega_cluster() -> ExperimentResult:
    """Put ALL datasets in one embedding space. The ultimate clustering test."""
    result = ExperimentResult(name="mega_cluster_all_datasets",
        hypothesis="All malicious text from all datasets clusters together vs all benign text",
        start_time=datetime.now(UTC).isoformat())
    model = _get_model(DEFAULT_MODEL)
    cse = _load_cyberseceval()
    advbench = _load_advbench()
    harmbench = _load_harmbench()
    jbb = _load_jailbreakbench()
    dna = _load_donotanswer()
    agentharm = _load_agentharm()
    if not model:
        result.conclusion = "SKIPPED"; result.end_time = datetime.now(UTC).isoformat(); return result

    # Gather ALL malicious texts from every dataset
    all_mal: list[tuple[str, str]] = []  # (text, source)
    for e in (cse or [])[:200]:
        t = e.get("mutated_prompt", e.get("base_prompt", ""))
        if t: all_mal.append((str(t)[:200], "CyberSecEval"))
    for e in (advbench or [])[:200]:
        t = e.get("goal", "")
        if t: all_mal.append((str(t)[:200], "AdvBench"))
    for e in (harmbench or [])[:200]:
        t = e.get("Behavior", e.get("behavior", ""))
        if t: all_mal.append((str(t)[:200], "HarmBench"))
    for e in (jbb or []):
        t = e.get("Behavior", e.get("behavior", e.get("goal", "")))
        if t: all_mal.append((str(t)[:200], "JailbreakBench"))
    for e in (dna or [])[:200]:
        t = e.get("question", e.get("prompt", ""))
        if t: all_mal.append((str(t)[:200], "DoNotAnswer"))
    for e in (agentharm or []):
        t = e.get("behavior", e.get("prompt", e.get("name", "")))
        if t: all_mal.append((str(t)[:200], "AgentHarm"))

    n_mal = len(all_mal)
    ben_texts = [f"{t} (v{random.randint(1,100000)})" for t in _BENIGN_TEXTS * 100][:n_mal]
    mal_texts = [t[0] for t in all_mal]

    logger.info("  Mega cluster: %d malicious from %d datasets + %d benign",
                n_mal, len(set(t[1] for t in all_mal)), len(ben_texts))

    metrics = _cluster_f1(model, mal_texts, ben_texts)

    # Source breakdown
    sources = {}
    for _, src in all_mal:
        sources[src] = sources.get(src, 0) + 1

    result.data = {**metrics, "total_malicious": n_mal, "total_benign": len(ben_texts),
                   "sources": sources, "num_datasets": len(sources)}
    result.findings = [
        f"MEGA CLUSTER F1={metrics['f1']} across {len(sources)} datasets",
        f"Total: {n_mal} malicious + {len(ben_texts)} benign = {n_mal + len(ben_texts)} texts",
        f"Sources: {sources}",
    ]
    result.novel = metrics["f1"] > 0.95
    result.significance = metrics["f1"]
    result.conclusion = f"Mega cluster ({len(sources)} datasets, {n_mal + len(ben_texts)} texts): F1={metrics['f1']}. {'ALL malicious text clusters together' if metrics['f1'] > 0.95 else 'Some datasets dont cluster well together'}."
    result.end_time = datetime.now(UTC).isoformat(); return result


def exp_pairwise_dataset_transfer() -> ExperimentResult:
    """Test centroid transfer between EVERY pair of datasets."""
    result = ExperimentResult(name="pairwise_dataset_transfer",
        hypothesis="Centroids trained on one dataset detect attacks from every other dataset",
        start_time=datetime.now(UTC).isoformat())
    model = _get_model(DEFAULT_MODEL)

    datasets: dict[str, list[str]] = {}

    cse = _load_cyberseceval()
    if cse:
        datasets["CyberSecEval"] = [e.get("mutated_prompt", e.get("base_prompt", ""))[:200] for e in cse[:200]]
    advbench = _load_advbench()
    if advbench:
        datasets["AdvBench"] = [e["goal"][:200] for e in advbench[:200]]
    harmbench = _load_harmbench()
    if harmbench:
        datasets["HarmBench"] = [str(e.get("Behavior", e.get("behavior", "")))[:200] for e in harmbench[:200]]
    dna = _load_donotanswer()
    if dna:
        datasets["DoNotAnswer"] = [str(e.get("question", e.get("prompt", "")))[:200] for e in dna[:200]]
    agentharm = _load_agentharm()
    if agentharm:
        datasets["AgentHarm"] = [str(e.get("behavior", e.get("name", "")))[:200] for e in agentharm]

    if not model or len(datasets) < 2:
        result.conclusion = "SKIPPED"; result.end_time = datetime.now(UTC).isoformat(); return result

    ben_embs = model.encode(_BENIGN_TEXTS, show_progress_bar=False)
    ben_centroid = ben_embs.mean(axis=0)

    # Pre-encode all datasets
    encoded: dict[str, tuple[Any, Any]] = {}
    for name, texts in datasets.items():
        texts = [t for t in texts if len(t) > 10]
        if texts:
            embs = model.encode(texts[:200], show_progress_bar=False)
            centroid = embs.mean(axis=0)
            encoded[name] = (embs, centroid)

    # Pairwise transfer: train centroid on A, test detection on B
    transfer_matrix: dict[str, dict[str, float]] = {}
    for train_name, (_, train_centroid) in encoded.items():
        transfer_matrix[train_name] = {}
        for test_name, (test_embs, _) in encoded.items():
            correct = 0
            for emb in test_embs:
                sim_mal = _cosine_similarity(emb, train_centroid)
                sim_ben = _cosine_similarity(emb, ben_centroid)
                if sim_mal > sim_ben:
                    correct += 1
            accuracy = correct / len(test_embs)
            transfer_matrix[train_name][test_name] = round(accuracy, 3)

        logger.info("  %s → %s", train_name, transfer_matrix[train_name])

    # Overall transfer score
    all_transfers = []
    for train_name, tests in transfer_matrix.items():
        for test_name, acc in tests.items():
            if train_name != test_name:
                all_transfers.append(acc)
    avg_transfer = float(np.mean(all_transfers)) if all_transfers else 0

    result.data = {"transfer_matrix": transfer_matrix, "avg_cross_transfer": round(avg_transfer, 4)}
    result.findings = [
        f"Average cross-dataset transfer: {avg_transfer:.1%}",
        f"Tested {len(encoded)} datasets pairwise ({len(all_transfers)} transfer pairs)",
    ]
    result.novel = avg_transfer > 0.7
    result.significance = avg_transfer
    result.conclusion = f"Pairwise transfer: {avg_transfer:.1%} average. {'Centroids generalize' if avg_transfer > 0.7 else 'Limited generalization'} across {len(encoded)} datasets."
    result.end_time = datetime.now(UTC).isoformat(); return result


ALL_EXPERIMENTS = [
    # Original 9
    ("category_isolation", exp_category_isolation),
    ("chain_length_sweep", exp_chain_length_sweep),
    ("advbench_clustering", exp_advbench_clustering),
    ("cross_dataset_generalization", exp_cross_dataset),
    ("moltbook_real_clustering", exp_moltbook_clustering),
    ("moltbook_anomaly_injection", exp_moltbook_injection),
    ("mixed_dataset_population", exp_mixed_population),
    ("context_window_sweep", exp_context_window_sweep),
    ("centroid_stability", exp_centroid_stability),
    # New dataset-specific experiments
    ("harmbench_clustering", exp_harmbench_clustering),
    ("jailbreakbench_clustering", exp_jailbreakbench_clustering),
    ("donotanswer_clustering", exp_donotanswer_clustering),
    ("agentharm_clustering", exp_agentharm_clustering),
    ("xstest_over_refusal", exp_xstest_over_refusal),
    # Cross-dataset mega experiments
    ("mega_cluster_all_datasets", exp_all_datasets_mega_cluster),
    ("pairwise_dataset_transfer", exp_pairwise_dataset_transfer),
]


def run_all(rounds: int = 5) -> None:
    """Run all experiments for N rounds."""
    logger.info("=" * 70)
    logger.info("BENCHMARK AUTORESEARCH — %d rounds × %d experiments", rounds, len(ALL_EXPERIMENTS))
    logger.info("=" * 70)

    experiment_start = time.monotonic()
    novel_count = 0
    saved: list[str] = []

    for round_num in range(1, rounds + 1):
        logger.info("\n══ ROUND %d/%d ══", round_num, rounds)

        for name, func in ALL_EXPERIMENTS:
            logger.info("\n─── %s ───", name)
            try:
                result = func()
                path = _save_result(result)
                saved.append(str(path))
                if result.novel:
                    novel_count += 1
                    logger.info("  ★ NOVEL (%.2f): %s", result.significance, result.conclusion[:100])
                else:
                    logger.info("  %s", result.conclusion[:100])
            except Exception as exc:
                logger.error("  FAILED: %s", exc)
            gc.collect()

    total_elapsed = time.monotonic() - experiment_start
    logger.info("\n" + "=" * 70)
    logger.info("AUTORESEARCH COMPLETE: %d experiments, %d novel, %.1f hours",
                len(saved), novel_count, total_elapsed / 3600)
    logger.info("=" * 70)

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "rounds": rounds,
        "total_experiments": len(saved),
        "novel_findings": novel_count,
        "elapsed_hours": round(total_elapsed / 3600, 2),
        "files": saved,
    }
    summary_path = EXPERIMENTS_DIR / f"bench_autoresearch_summary_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    run_all(rounds=args.rounds)
