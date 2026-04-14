#!/usr/bin/env python3.13
"""
Phase 2 — Scorer B: BIG-Bench truthful_qa + BIG-Bench hhh_alignment/honest + SycophancyEval
Scores through TRIBE v2 Modal endpoint. Saves region_activations as feature vectors for
supervised classification (Brain 14 proxy).

Datasets:
  - BIG-Bench truthful_qa:  299 MC examples → flatten target_scores → cap 300 pairs
  - BIG-Bench hhh_alignment: 61 examples → flatten target_scores → ~150-200 pairs (all)
  - SycophancyEval: 30K rows × 3 files → sample 50 rows × 2 answers × 3 files = 300 pairs

Total target: ~780 pairs, ~2.2h at 10s/req (budget-safe).

Each output line:
  idx, text_hash, label, dataset, emotion_score, emotion_ratio,
  signature_shape, region_activations (dict), total_activation

Resumable: skips already-scored indices via idx field in existing JSONL.
Incremental saves: flush every 25 rows.
Failure guard: stop if 20 consecutive errors; save partial.
"""

from __future__ import annotations

import hashlib
import json
import random
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCORE_URL = "https://zazmattaz--cerebrum-tribe-tribescorer-score.modal.run"
EXPERIMENTS = Path(__file__).parent

BIGBENCH_TRUTHQA_FILE = EXPERIMENTS / "phase2_data/bigbench_tasks/truthful_qa/task_mc.json"
BIGBENCH_HHH_FILE = EXPERIMENTS / "phase2_data/bigbench_tasks/deceptive_alignment/task.json"
SYCOPHANCY_DIR = EXPERIMENTS / "phase2_data/sycophancy"
SYCOPHANCY_FILES = [
    "sycophancy_on_nlp_survey.jsonl",
    "sycophancy_on_philpapers2020.jsonl",
    "sycophancy_on_political_typology_quiz.jsonl",
]

OUT_TRUTHQA = EXPERIMENTS / "phase2_bigbench_truthqa_results.jsonl"
OUT_HHH = EXPERIMENTS / "phase2_bigbench_hhh_results.jsonl"
OUT_SYCO = EXPERIMENTS / "phase2_sycophancy_results.jsonl"
OUT_SUMMARY = EXPERIMENTS / "phase2_scorer_B_summary.json"

# Sampling caps
CAP_TRUTHQA_PAIRS = 300       # up to 300 (100 per class, balanced)
CAP_SYCO_ROWS_PER_FILE = 50   # 50 rows × 2 answers × 3 files = 300 pairs
MAX_CONSECUTIVE_ERRORS = 20
FLUSH_EVERY = 25
REQUEST_TIMEOUT = 300         # seconds — Modal cold start can take 200s+
BACKOFF_SLEEP = 5             # seconds between retries on error
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def score_text(text: str) -> dict[str, Any]:
    """Score text via TRIBE endpoint using curl for reliable hard-timeout on macOS.

    Uses subprocess + curl because urllib SSL timeouts on macOS can stall indefinitely
    when Modal holds the connection open during cold start. curl --max-time is a true
    wall-clock deadline that always fires.
    """
    payload = json.dumps({"text": text, "modality": "text"})
    result = subprocess.run(
        [
            "curl",
            "--silent",
            "--show-error",
            "--max-time", str(REQUEST_TIMEOUT),
            "--connect-timeout", "30",
            "-X", "POST",
            "-H", "Content-Type: application/json",
            "-d", payload,
            SCORE_URL,
        ],
        capture_output=True,
        text=True,
        timeout=REQUEST_TIMEOUT + 10,  # subprocess hard kill after curl's own timeout
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl exit {result.returncode}: {result.stderr.strip()[:200]}")
    return json.loads(result.stdout)


def warm_endpoint() -> None:
    """Send a throwaway call to warm the Modal cold-start.

    Uses REQUEST_TIMEOUT to absorb the cold-start wait (~200s on A10G).
    curl --max-time is a true wall-clock hard deadline.
    """
    print(f"[warm] Pinging endpoint (timeout={REQUEST_TIMEOUT}s)...", flush=True)
    t0 = time.time()
    try:
        resp = score_text("Hello world.")
        print(
            f"[warm] OK in {time.time()-t0:.1f}s — emotion_score={resp.get('emotion_score', '?')}",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[warm] WARN after {time.time()-t0:.1f}s: {type(exc).__name__}: {exc}",
            flush=True,
        )
        print("[warm] Proceeding — first scoring call will retry.", flush=True)


def auroc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.5
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def load_scored_indices(out_file: Path) -> set[int]:
    """Read already-scored indices for resumption."""
    if not out_file.exists():
        return set()
    done: set[int] = set()
    with out_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add(row["idx"])
            except Exception:
                pass
    return done


def score_pairs(
    pairs: list[tuple[int, str, int, str]],  # (idx, text, label, dataset)
    out_file: Path,
    dataset_name: str,
) -> dict[str, Any]:
    """
    Score a list of (idx, text, label, dataset) pairs against TRIBE endpoint.
    Appends results to out_file incrementally every FLUSH_EVERY rows.
    Returns per-dataset statistics.
    """
    done_indices = load_scored_indices(out_file)
    pending = [(idx, text, label, ds) for idx, text, label, ds in pairs if idx not in done_indices]
    already_done = len(pairs) - len(pending)
    if already_done:
        print(f"  [{dataset_name}] Resuming: {already_done} already scored, {len(pending)} remaining", flush=True)

    n_scored = already_done
    n_errors = 0
    consecutive_errors = 0
    emotion_scores: list[float] = []
    emotion_ratios: list[float] = []
    labels_seen: list[int] = []

    # Load already-scored emotion stats for AUROC calculation
    if out_file.exists():
        with out_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    emotion_scores.append(float(row.get("emotion_score", 0)))
                    emotion_ratios.append(float(row.get("emotion_ratio", 0)))
                    labels_seen.append(int(row.get("label", 0)))
                except Exception:
                    pass

    t0 = time.time()
    buffer: list[str] = []

    def flush_buffer() -> None:
        if not buffer:
            return
        with out_file.open("a") as fh:
            fh.write("\n".join(buffer) + "\n")
        buffer.clear()

    for i, (idx, text, label, ds) in enumerate(pending):
        try:
            resp = score_text(text)
            consecutive_errors = 0

            region_act = resp.get("region_activations", {})
            row = {
                "idx": idx,
                "text_hash": text_hash(text),
                "label": label,
                "dataset": ds,
                "emotion_score": resp.get("emotion_score", 0.0),
                "emotion_ratio": resp.get("emotion_ratio", 0.0),
                "signature_shape": resp.get("signature_shape", None),
                "region_activations": region_act,
                "total_activation": resp.get("total_activation", 0.0),
                "timesteps": resp.get("timesteps", None),
            }
            buffer.append(json.dumps(row))
            n_scored += 1
            emotion_scores.append(float(row["emotion_score"]))
            emotion_ratios.append(float(row["emotion_ratio"]))
            labels_seen.append(label)

        except Exception as exc:
            n_errors += 1
            consecutive_errors += 1
            if n_errors <= 5 or n_errors % 25 == 0:
                print(f"  [{dataset_name}] ERR #{n_errors} idx={idx}: {exc}", flush=True)
            time.sleep(BACKOFF_SLEEP)
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"  [{dataset_name}] {MAX_CONSECUTIVE_ERRORS} consecutive errors — aborting early", flush=True)
                flush_buffer()
                break

        if len(buffer) >= FLUSH_EVERY:
            flush_buffer()

        if (i + 1) % FLUSH_EVERY == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.001)
            remaining = (len(pending) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{dataset_name}] {n_scored}/{len(pairs)} scored "
                f"({rate:.2f}/s, ETA {remaining:.0f}s, {n_errors} err)",
                flush=True,
            )

    flush_buffer()

    wall = time.time() - t0
    es_auroc = auroc(emotion_scores, labels_seen)
    er_auroc = auroc(emotion_ratios, labels_seen)

    print(
        f"  [{dataset_name}] Done: {n_scored}/{len(pairs)} scored in {wall:.0f}s "
        f"| emotion_score AUROC={es_auroc:.4f} emotion_ratio AUROC={er_auroc:.4f} "
        f"| errors={n_errors}",
        flush=True,
    )

    return {
        "n_total": len(pairs),
        "n_scored": n_scored,
        "n_errors": n_errors,
        "emotion_auroc": round(es_auroc, 4),
        "emotion_ratio_auroc": round(er_auroc, 4),
        "wall_clock_s": round(wall, 1),
    }


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_bigbench_truthqa(cap: int = CAP_TRUTHQA_PAIRS) -> list[tuple[int, str, int, str]]:
    """
    Flatten BIG-Bench truthful_qa MC examples into (idx, text, label, dataset) pairs.
    Text = answer choice text. Label = target_score (1=correct, 0=incorrect).
    Cap balanced: up to cap//2 positive and cap//2 negative.
    """
    data = json.loads(BIGBENCH_TRUTHQA_FILE.read_text())
    examples = data["examples"]

    positives: list[tuple[str, int]] = []
    negatives: list[tuple[str, int]] = []

    # Unified schema: label=1=deceptive/incorrect, label=0=truthful/honest.
    # BIG-Bench target_score=1 means CORRECT (honest). Invert: score=1 → label=0, score=0 → label=1.
    for ex in examples:
        for choice_text, score in ex["target_scores"].items():
            label = 0 if int(score) == 1 else 1  # inverted: correct=0, incorrect=1
            text = choice_text.strip()
            if not text:
                continue
            if label == 1:
                positives.append((text, label))
            else:
                negatives.append((text, label))

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    half = cap // 2
    selected = positives[:half] + negatives[:half]
    rng.shuffle(selected)

    pairs = [(i, text, label, "bigbench_truthqa") for i, (text, label) in enumerate(selected)]
    print(
        f"[load] bigbench_truthqa: {len(positives)} pos + {len(negatives)} neg available "
        f"→ sampled {len(pairs)} pairs (cap={cap})",
        flush=True,
    )
    return pairs


def load_bigbench_hhh() -> list[tuple[int, str, int, str]]:
    """
    Flatten BIG-Bench hhh_alignment/honest examples into (idx, text, label, dataset) pairs.
    Text = response string. Label = target_score (1=honest, 0=evasive).
    Score ALL examples (small dataset, ~150-200 pairs).
    """
    data = json.loads(BIGBENCH_HHH_FILE.read_text())
    examples = data["examples"]

    # Unified schema: label=1=deceptive/evasive, label=0=honest.
    # BIG-Bench target_score=1 means HONEST response. Invert: score=1 → label=0, score=0 → label=1.
    pairs_raw: list[tuple[str, int]] = []
    for ex in examples:
        for response_text, score in ex["target_scores"].items():
            text = response_text.strip()
            if not text:
                continue
            label = 0 if int(score) == 1 else 1  # inverted: honest=0, evasive=1
            pairs_raw.append((text, label))

    pairs = [(i, text, label, "bigbench_hhh") for i, (text, label) in enumerate(pairs_raw)]
    pos = sum(1 for _, _, l, _ in pairs if l == 1)
    neg = sum(1 for _, _, l, _ in pairs if l == 0)
    print(
        f"[load] bigbench_hhh: {len(examples)} examples → {len(pairs)} pairs "
        f"({pos} honest, {neg} evasive) — scoring all",
        flush=True,
    )
    return pairs


def load_sycophancy(rows_per_file: int = CAP_SYCO_ROWS_PER_FILE) -> list[tuple[int, str, int, str]]:
    """
    Sample rows_per_file rows from each of 3 sycophancy JSONL files.
    Each row produces 2 pairs: (answer_matching_behavior, 1) and (answer_not_matching_behavior, 0).
    Total: rows_per_file × 2 × 3 files.
    """
    rng = random.Random(RANDOM_SEED)
    all_pairs: list[tuple[str, int]] = []

    for fname in SYCOPHANCY_FILES:
        fpath = SYCOPHANCY_DIR / fname
        with fpath.open() as f:
            lines = f.readlines()

        rng.shuffle(lines)
        sampled = lines[:rows_per_file]

        file_pairs: list[tuple[str, int]] = []
        for line in sampled:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                # Some files (e.g. philpapers) store answer_not_matching_behavior as a list.
                # Take the first element if so.
                syco_raw = row.get("answer_matching_behavior", "")
                honest_raw = row.get("answer_not_matching_behavior", "")
                syco = (syco_raw[0] if isinstance(syco_raw, list) else syco_raw).strip()
                honest = (honest_raw[0] if isinstance(honest_raw, list) else honest_raw).strip()
                if syco:
                    file_pairs.append((syco, 1))
                if honest:
                    file_pairs.append((honest, 0))
            except Exception:
                pass

        print(
            f"[load] sycophancy/{fname}: {rows_per_file} rows sampled → {len(file_pairs)} pairs",
            flush=True,
        )
        all_pairs.extend(file_pairs)

    rng.shuffle(all_pairs)
    pairs = [(i, text, label, "sycophancy") for i, (text, label) in enumerate(all_pairs)]
    pos = sum(1 for _, _, l, _ in pairs if l == 1)
    neg = sum(1 for _, _, l, _ in pairs if l == 0)
    print(
        f"[load] sycophancy total: {len(pairs)} pairs ({pos} sycophantic, {neg} honest)",
        flush=True,
    )
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_global = time.time()
    print("=" * 60, flush=True)
    print("Phase 2 — Scorer B  (TRIBE v2 Modal)", flush=True)
    print(f"Endpoint: {SCORE_URL}", flush=True)
    print("=" * 60, flush=True)

    warm_endpoint()

    # Load datasets
    pairs_truthqa = load_bigbench_truthqa(cap=CAP_TRUTHQA_PAIRS)
    pairs_hhh = load_bigbench_hhh()
    pairs_syco = load_sycophancy(rows_per_file=CAP_SYCO_ROWS_PER_FILE)

    total_pairs = len(pairs_truthqa) + len(pairs_hhh) + len(pairs_syco)
    eta_h = total_pairs * 10 / 3600
    print(
        f"\nTotal pairs to score: {total_pairs} "
        f"(~{eta_h:.1f}h at 10s/req if all cold)\n",
        flush=True,
    )

    sampling_notes = {
        "bigbench_truthqa": (
            f"Flattened target_scores from {CAP_TRUTHQA_PAIRS // 2} pos + {CAP_TRUTHQA_PAIRS // 2} neg answer choices "
            f"(balanced cap={CAP_TRUTHQA_PAIRS})"
        ),
        "bigbench_hhh": "All 61 examples flattened — no sampling, small dataset",
        "sycophancy": (
            f"{CAP_SYCO_ROWS_PER_FILE} rows × 2 answers × 3 files = {CAP_SYCO_ROWS_PER_FILE * 2 * 3} pairs "
            f"(seed={RANDOM_SEED})"
        ),
    }

    # Score each dataset
    print("\n--- BIG-Bench truthful_qa ---", flush=True)
    stats_truthqa = score_pairs(pairs_truthqa, OUT_TRUTHQA, "bigbench_truthqa")

    print("\n--- BIG-Bench hhh_alignment/honest ---", flush=True)
    stats_hhh = score_pairs(pairs_hhh, OUT_HHH, "bigbench_hhh")

    print("\n--- SycophancyEval ---", flush=True)
    stats_syco = score_pairs(pairs_syco, OUT_SYCO, "sycophancy")

    total_wall = time.time() - t_global

    summary = {
        "scorer": "phase2-scorer-B",
        "endpoint": SCORE_URL,
        "per_dataset": {
            "bigbench_truthqa": {**stats_truthqa, "output_file": str(OUT_TRUTHQA)},
            "bigbench_hhh": {**stats_hhh, "output_file": str(OUT_HHH)},
            "sycophancy": {**stats_syco, "output_file": str(OUT_SYCO)},
        },
        "total_pairs_attempted": total_pairs,
        "total_scored": stats_truthqa["n_scored"] + stats_hhh["n_scored"] + stats_syco["n_scored"],
        "total_errors": stats_truthqa["n_errors"] + stats_hhh["n_errors"] + stats_syco["n_errors"],
        "total_wall_clock_s": round(total_wall, 1),
        "total_wall_clock_h": round(total_wall / 3600, 3),
        "sampling_notes": sampling_notes,
        "endpoint_notes": (
            "region_activations is {} (empty dict) from this Modal endpoint configuration. "
            "Ablation agent should use scalar features: emotion_score, emotion_ratio, "
            "total_activation, timesteps, signature_shape (label-encoded) for supervised classification."
        ),
        "caps": {
            "bigbench_truthqa_pairs": CAP_TRUTHQA_PAIRS,
            "sycophancy_rows_per_file": CAP_SYCO_ROWS_PER_FILE,
            "random_seed": RANDOM_SEED,
        },
    }

    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60, flush=True)
    print(f"DONE — total wall clock: {total_wall / 3600:.2f}h", flush=True)
    print(f"Summary written: {OUT_SUMMARY}", flush=True)
    print(f"bigbench_truthqa: n={stats_truthqa['n_scored']} AUROC={stats_truthqa['emotion_auroc']:.4f}", flush=True)
    print(f"bigbench_hhh:     n={stats_hhh['n_scored']} AUROC={stats_hhh['emotion_auroc']:.4f}", flush=True)
    print(f"sycophancy:       n={stats_syco['n_scored']} AUROC={stats_syco['emotion_auroc']:.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
