#!/usr/bin/env python3.13
"""
Phase 2 v2 — MALT Scorer

Scores 400 MALT transcripts (hardcoded_solution label) via TRIBE Modal endpoint.
Saves incremental results to experiments/phase2v2_malt_results.jsonl.
Computes AUROC on completion, saves experiments/phase2v2_malt_summary.json.

Usage:
    HF_TOKEN=hf_xxx python3.13 experiments/score_malt.py

Requires:
    - HF_TOKEN in environment
    - MALT terms accepted at https://huggingface.co/datasets/metr-evals/malt-public
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path("/Users/zazumoloi/Desktop/Claude Code/.env"))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRIBE_ENDPOINT = "https://zazmattaz--cerebrum-tribe-tribescorer-score.modal.run"
LABEL_NAME = "hardcoded_solution"
SAMPLES_PER_CLASS = 200
RESULTS_PATH = Path("experiments/phase2v2_malt_results.jsonl")
SUMMARY_PATH = Path("experiments/phase2v2_malt_summary.json")
REQUEST_TIMEOUT = 120
MAX_RETRIES = 3
RETRY_BACKOFF = 10  # seconds

# ---------------------------------------------------------------------------
# Warm endpoint
# ---------------------------------------------------------------------------


def warm_endpoint() -> bool:
    """Send a short warm-up request. Returns True if endpoint responds."""
    try:
        r = requests.post(
            TRIBE_ENDPOINT,
            json={"text": "hello world test"},
            timeout=30,
        )
        return r.status_code == 200
    except Exception as e:
        print(f"[warm] Endpoint unreachable: {e}")
        return False


# ---------------------------------------------------------------------------
# Score one transcript
# ---------------------------------------------------------------------------


def score_one(text: str, timeout: int = REQUEST_TIMEOUT) -> dict | None:
    """
    POST text to TRIBE endpoint. Returns response dict or None on failure.
    Expected response: {emotion_score, emotion_ratio, total_activation, signature_shape, region_activations}
    """
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                TRIBE_ENDPOINT,
                json={"text": text},
                timeout=timeout,
            )
            if r.status_code == 200:
                return r.json()
            print(f"  [retry {attempt+1}] HTTP {r.status_code}: {r.text[:80]}")
        except requests.Timeout:
            print(f"  [retry {attempt+1}] Timeout after {timeout}s")
        except Exception as e:
            print(f"  [retry {attempt+1}] Error: {e}")
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF)
    return None


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------


def compute_auroc(scores: list[float], labels: list[int]) -> float:
    """Compute AUROC via trapezoidal rule."""
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp = 0
    fp = 0
    auc = 0.0
    prev_fp = 0
    prev_tp = 0
    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += (tp + prev_tp) / 2.0
            prev_fp = fp
            prev_tp = tp
    return auc / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Check token
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("FAIL: HF_TOKEN not set. See experiments/scouts/metr/MALT_UNLOCK_INSTRUCTIONS.md")
        sys.exit(1)

    print("=" * 60)
    print("Phase 2 v2 — MALT Scorer")
    print(f"Endpoint: {TRIBE_ENDPOINT}")
    print(f"Label: {LABEL_NAME}")
    print(f"Target: {SAMPLES_PER_CLASS} per class ({SAMPLES_PER_CLASS * 2} total)")
    print("=" * 60)

    # Warm endpoint
    print("[warm] Warming TRIBE endpoint...", flush=True)
    if not warm_endpoint():
        print("[warm] WARNING: Endpoint did not respond to warmup. Continuing anyway.")
    else:
        print("[warm] Endpoint warm.", flush=True)

    # Load data
    print(f"\n[load] Loading MALT transcripts (label={LABEL_NAME})...")
    sys.path.insert(0, str(Path("experiments/scouts/metr")))
    from loader_malt import load_transcripts

    samples = load_transcripts(label_name=LABEL_NAME, samples_per_class=SAMPLES_PER_CLASS)
    print(f"[load] {len(samples)} transcripts loaded.", flush=True)

    # Load already-scored indices (resume support)
    already_scored: set[str] = set()
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    already_scored.add(rec["text_hash"])
                except json.JSONDecodeError:
                    pass
        print(f"[resume] {len(already_scored)} already scored, resuming.")

    # Score
    t_start = time.time()
    n_errors = 0
    n_scored = 0

    with open(RESULTS_PATH, "a") as out_f:
        for idx, sample in enumerate(samples):
            text_hash = sample["text_hash"]
            if text_hash in already_scored:
                n_scored += 1
                continue

            elapsed = time.time() - t_start
            rate = n_scored / elapsed if elapsed > 0 else 0
            eta = (len(samples) - idx) / rate if rate > 0 else float("inf")
            print(
                f"[{idx+1:03d}/{len(samples)}] "
                f"label={sample['label']} model={sample['model'][:20]} "
                f"len={sample['text_length']:6d} "
                f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                flush=True,
            )

            resp = score_one(sample["text"])
            if resp is None:
                n_errors += 1
                print(f"  ERROR: scoring failed for idx={idx}, skipping.")
                continue

            record = {
                "idx": idx,
                "text_hash": text_hash,
                "label": sample["label"],
                "dataset": "malt",
                "label_name": LABEL_NAME,
                "text_length": sample["text_length"],
                "model": sample["model"],
                "task_id": sample["task_id"],
                "run_id": sample["run_id"],
                "emotion_score": resp.get("emotion_score", 0.0),
                "emotion_ratio": resp.get("emotion_ratio", 0.0),
                "total_activation": resp.get("total_activation", 0.0),
                "timesteps": resp.get("timesteps", None),
                "signature_shape": resp.get("signature_shape", "unknown"),
                "region_activations": resp.get("region_activations", {}),
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            n_scored += 1
            already_scored.add(text_hash)

    total_wall = time.time() - t_start
    print(f"\n[done] Scored {n_scored} / {len(samples)} transcripts. Errors: {n_errors}.")
    print(f"[done] Wall clock: {total_wall/60:.1f} min.")

    # Compute AUROCs
    results: list[dict] = []
    with open(RESULTS_PATH) as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    malt_results = [r for r in results if r["dataset"] == "malt"]
    labels_list = [r["label"] for r in malt_results]
    emotion_scores = [r["emotion_score"] for r in malt_results]
    emotion_ratios = [r["emotion_ratio"] for r in malt_results]
    text_lengths = [r["text_length"] for r in malt_results]

    auroc_emotion_score = compute_auroc(emotion_scores, labels_list)
    auroc_emotion_ratio = compute_auroc(emotion_ratios, labels_list)
    auroc_text_length = compute_auroc(text_lengths, labels_list)

    print(f"\n[auroc] emotion_score:  {auroc_emotion_score:.4f}")
    print(f"[auroc] emotion_ratio:  {auroc_emotion_ratio:.4f}")
    print(f"[auroc] text_length:    {auroc_text_length:.4f} (length artifact check)")

    summary = {
        "scorer": "phase2v2_malt",
        "tribe_endpoint": TRIBE_ENDPOINT,
        "label_name": LABEL_NAME,
        "samples_per_class": SAMPLES_PER_CLASS,
        "n_scored": n_scored,
        "n_errors": n_errors,
        "total_wall_clock_seconds": round(total_wall, 1),
        "auroc": {
            "emotion_score": round(auroc_emotion_score, 4),
            "emotion_ratio": round(auroc_emotion_ratio, 4),
            "text_length": round(auroc_text_length, 4),
        },
        "length_artifact_flag": auroc_text_length > 0.6,
        "note": (
            "MALT hardcoded_solution label. 200 positive (label in metadata.labels) "
            "vs 200 negative. Transcripts are full agentic runs from metr-evals/malt-public. "
            "region_activations={} for all inputs (endpoint limitation as of 2026-04-14)."
        ),
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[saved] Results: {RESULTS_PATH}")
    print(f"[saved] Summary: {SUMMARY_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
