#!/usr/bin/env python3.13
"""Phase 2 — Scorer A: TruthfulQA + DeceptionBench through TRIBE v2.

Saves region_activations as feature vectors for supervised classification.
Resume-capable via incremental JSONL output.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCORE_URL = "https://zazmattaz--cerebrum-tribe-tribescorer-score.modal.run"
EXPERIMENTS = Path(__file__).parent

TRUTHFULQA_CSV = EXPERIMENTS / "phase2_data/truthfulqa/TruthfulQA.csv"
DECEPTIONBENCH_JSON = (
    EXPERIMENTS
    / "phase2_data/deceptionbench/data/processed_questions/questions_1000_all.json"
)

TRUTHFULQA_OUT = EXPERIMENTS / "phase2_truthfulqa_results.jsonl"
DECEPTIONBENCH_OUT = EXPERIMENTS / "phase2_deceptionbench_results.jsonl"
SUMMARY_OUT = EXPERIMENTS / "phase2_scorer_A_summary.json"

# Sample 200 per class per dataset — ~800 total texts, ~2.5h at 10s/request
# Actual latency ~3-5s when warm so real wall-clock ~45min-1h per dataset
SAMPLES_PER_CLASS = 200
SAVE_EVERY = 25
MAX_CONSECUTIVE_ERRORS = 20
RETRY_SLEEP = 5.0
REQUEST_TIMEOUT = 180  # seconds — covers cold-start (30-90s) + compute


# ---------------------------------------------------------------------------
# Endpoint — uses subprocess curl to avoid Python SSL timeout on cold starts
# ---------------------------------------------------------------------------


def score_text(text: str) -> dict[str, Any]:
    """POST to TRIBE endpoint via curl subprocess (avoids SSL timeout on cold starts)."""
    payload = json.dumps({"text": text, "modality": "text"})
    result = subprocess.run(
        [
            "curl",
            "-s",
            "-X", "POST",
            SCORE_URL,
            "-H", "Content-Type: application/json",
            "-d", payload,
            "--max-time", str(REQUEST_TIMEOUT),
        ],
        capture_output=True,
        text=True,
        timeout=REQUEST_TIMEOUT + 10,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            f"curl failed (rc={result.returncode}): {result.stderr[:200]}"
        )
    return json.loads(result.stdout)


def warm_endpoint() -> bool:
    """Fire a single throwaway request to warm the Modal cold-start."""
    print("[warm] Warming TRIBE endpoint...", flush=True)
    try:
        score_text("This is a warm-up request.")
        print("[warm] Done.", flush=True)
        return True
    except Exception as exc:
        print(f"[warm] Warning — warm-up failed: {exc}", flush=True)
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def auroc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, lbl in zip(scores, labels) if lbl == 1]
    neg = [s for s, lbl in zip(scores, labels) if lbl == 0]
    if not pos or not neg:
        return 0.5
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def load_existing_indices(out_file: Path) -> set[int]:
    """Return set of idx values already saved in the JSONL file."""
    existing: set[int] = set()
    if not out_file.exists():
        return existing
    with out_file.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                existing.add(rec["idx"])
            except (json.JSONDecodeError, KeyError):
                pass
    return existing


def append_record(out_file: Path, record: dict[str, Any]) -> None:
    with out_file.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def load_truthfulqa(n_per_class: int) -> list[dict[str, Any]]:
    """Load TruthfulQA pairs. Best Answer → label 0, Best Incorrect Answer → label 1."""
    rows: list[dict[str, Any]] = []
    with TRUTHFULQA_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            best = row.get("Best Answer", "").strip()
            incorrect = row.get("Best Incorrect Answer", "").strip()
            if best:
                rows.append({"text": best, "label": 0, "source_row": row.get("Question", "")})
            if incorrect:
                rows.append({"text": incorrect, "label": 1, "source_row": row.get("Question", "")})

    truthful = [r for r in rows if r["label"] == 0]
    deceptive = [r for r in rows if r["label"] == 1]
    print(
        f"[truthfulqa] Raw pairs: {len(truthful)} truthful, {len(deceptive)} deceptive",
        flush=True,
    )

    # Sample evenly — take first n_per_class (CSV is already sorted by question)
    sampled = truthful[:n_per_class] + deceptive[:n_per_class]
    # Interleave so scoring alternates classes
    interleaved: list[dict[str, Any]] = []
    for t, d in zip(truthful[:n_per_class], deceptive[:n_per_class]):
        interleaved.append(t)
        interleaved.append(d)
    print(
        f"[truthfulqa] Sampled: {n_per_class} truthful + {n_per_class} deceptive = {len(interleaved)} texts",
        flush=True,
    )
    return interleaved


def load_deceptionbench(n_per_class: int) -> list[dict[str, Any]]:
    """Load DeceptionBench pairs. answer → label 0, false_statement → label 1."""
    with DECEPTIONBENCH_JSON.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    # orient='index' — outer key is string row index, value is dict of column → dict[row_idx → value]
    # Detect structure: either {row_idx: {col: value}} or {col: {row_idx: value}}
    first_key = next(iter(raw))
    first_val = raw[first_key]

    rows: list[dict[str, Any]] = []
    if isinstance(first_val, dict) and "question" in first_val:
        # Row-oriented: {row_idx: {question: ..., answer: ..., false_statement: ...}}
        for _idx, record in raw.items():
            rows.append(record)
    else:
        # Column-oriented: {column_name: {row_idx: value}}
        # Pivot to row-oriented
        row_indices = list(raw[first_key].keys())
        for ridx in row_indices:
            rows.append({col: raw[col][ridx] for col in raw if ridx in raw[col]})

    print(f"[deceptionbench] Raw rows: {len(rows)}", flush=True)

    truthful: list[dict[str, Any]] = []
    deceptive: list[dict[str, Any]] = []
    for row in rows:
        answer = (row.get("answer") or "").strip()
        false_stmt = (row.get("false_statement") or "").strip()
        if answer:
            truthful.append({"text": answer, "label": 0, "source_row": row.get("question", "")})
        if false_stmt:
            deceptive.append({"text": false_stmt, "label": 1, "source_row": row.get("question", "")})

    print(
        f"[deceptionbench] Pairs: {len(truthful)} truthful, {len(deceptive)} deceptive",
        flush=True,
    )

    interleaved: list[dict[str, Any]] = []
    for t, d in zip(truthful[:n_per_class], deceptive[:n_per_class]):
        interleaved.append(t)
        interleaved.append(d)
    print(
        f"[deceptionbench] Sampled: {n_per_class} + {n_per_class} = {len(interleaved)} texts",
        flush=True,
    )
    return interleaved


# ---------------------------------------------------------------------------
# Core scoring loop
# ---------------------------------------------------------------------------


def score_dataset(
    items: list[dict[str, Any]],
    dataset_name: str,
    out_file: Path,
) -> dict[str, Any]:
    """Score items, save incrementally, return summary stats."""
    existing = load_existing_indices(out_file)
    if existing:
        print(f"[{dataset_name}] Resuming — {len(existing)} already scored", flush=True)

    t0 = time.time()
    n_scored = 0
    n_errors = 0
    consecutive_errors = 0
    buffer: list[dict[str, Any]] = []

    # Collect all completed records for AUROC calculation at end
    all_records: list[dict[str, Any]] = []

    # Load already-saved records for AUROC
    if out_file.exists():
        with out_file.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    for idx, item in enumerate(items):
        if idx in existing:
            continue

        text = item["text"]
        label = item["label"]

        try:
            resp = score_text(text)
            consecutive_errors = 0

            record: dict[str, Any] = {
                "idx": idx,
                "text_hash": text_hash(text),
                "label": label,
                "dataset": dataset_name,
                "emotion_score": resp.get("emotion_score", 0.0),
                "emotion_ratio": resp.get("emotion_ratio", 0.0),
                "signature_shape": resp.get("signature_shape", ""),
                "region_activations": resp.get("region_activations", {}),
                "total_activation": resp.get("total_activation", 0.0),
            }

            buffer.append(record)
            all_records.append(record)
            n_scored += 1

            if len(buffer) >= SAVE_EVERY:
                for rec in buffer:
                    append_record(out_file, rec)
                buffer.clear()

        except Exception as exc:
            n_errors += 1
            consecutive_errors += 1
            print(
                f"  [{dataset_name}] idx={idx} error ({n_errors} total, {consecutive_errors} consec): {exc}",
                flush=True,
            )
            time.sleep(RETRY_SLEEP)

            # One retry
            try:
                resp = score_text(text)
                consecutive_errors = 0
                record = {
                    "idx": idx,
                    "text_hash": text_hash(text),
                    "label": label,
                    "dataset": dataset_name,
                    "emotion_score": resp.get("emotion_score", 0.0),
                    "emotion_ratio": resp.get("emotion_ratio", 0.0),
                    "signature_shape": resp.get("signature_shape", ""),
                    "region_activations": resp.get("region_activations", {}),
                    "total_activation": resp.get("total_activation", 0.0),
                }
                buffer.append(record)
                all_records.append(record)
                n_scored += 1
                n_errors -= 1  # retry succeeded — don't count as error
            except Exception as exc2:
                print(f"  [{dataset_name}] idx={idx} retry also failed: {exc2}", flush=True)

        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            print(
                f"  [{dataset_name}] {MAX_CONSECUTIVE_ERRORS} consecutive errors — stopping early",
                flush=True,
            )
            break

        # Progress every 25 processed (scored + skipped)
        processed = (idx + 1) - len(existing)
        total_remaining = len(items) - len(existing)
        if processed > 0 and processed % SAVE_EVERY == 0:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total_remaining - processed) / rate if rate > 0 else 0
            print(
                f"  [{dataset_name}] {processed}/{total_remaining} "
                f"({rate:.2f} texts/s, ETA {eta:.0f}s, errors={n_errors})",
                flush=True,
            )

    # Flush remaining buffer
    for rec in buffer:
        append_record(out_file, rec)

    wall_clock = time.time() - t0

    # Compute AUROC from all records in output file
    es_scores = [r["emotion_score"] for r in all_records]
    er_scores = [r["emotion_ratio"] for r in all_records]
    labels = [r["label"] for r in all_records]

    es_auroc = auroc(es_scores, labels) if labels else 0.5
    er_auroc = auroc(er_scores, labels) if labels else 0.5

    print(
        f"\n[{dataset_name}] Complete: {n_scored} scored, {n_errors} errors, "
        f"{wall_clock:.0f}s",
        flush=True,
    )
    print(
        f"[{dataset_name}] emotion_score AUROC={es_auroc:.4f}  "
        f"emotion_ratio AUROC={er_auroc:.4f}",
        flush=True,
    )
    print(
        f"[{dataset_name}] NOTE: AUROC ~0.5 or inverted expected (Phase 1 found 0.383). "
        "Supervised classification over region_activations is the viable path.",
        flush=True,
    )

    # Count how many records have non-empty region_activations
    n_with_regions = sum(1 for r in all_records if r.get("region_activations"))

    return {
        "dataset": dataset_name,
        "n_scored": n_scored + len(existing),
        "n_errors": n_errors,
        "emotion_score_auroc": round(es_auroc, 4),
        "emotion_ratio_auroc": round(er_auroc, 4),
        "wall_clock_seconds": round(wall_clock, 1),
        "n_with_region_activations": n_with_regions,
        "region_activations_note": (
            "Endpoint currently returns region_activations={} for all inputs. "
            "Feature vector for classification = [emotion_score, emotion_ratio, "
            "total_activation] (3-dim scalar features). "
            "Ablation agent should use these scalars + any future region_activations "
            "when the endpoint exposes them."
        ),
        "sampling_notes": (
            f"Sampled {SAMPLES_PER_CLASS} per class ({SAMPLES_PER_CLASS * 2} total). "
            "TruthfulQA: Best Answer=label 0, Best Incorrect Answer=label 1. "
            "DeceptionBench: answer=label 0, false_statement=label 1. "
            "Budget: 200/class, ~3-5s/req when warm = ~20-34min per dataset. "
            "Full dataset (400/class) would take ~40-68min per dataset."
        ),
    }


# ---------------------------------------------------------------------------
# Dry-run mode (endpoint unreachable)
# ---------------------------------------------------------------------------


def dry_run_check() -> bool:
    """Return True if endpoint is reachable."""
    try:
        score_text("ping")
        return True
    except Exception as exc:
        print(f"[dry-run] Endpoint unreachable: {exc}", flush=True)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    print("=" * 60, flush=True)
    print("Phase 2 Scorer A — TruthfulQA + DeceptionBench", flush=True)
    print(f"Endpoint: {SCORE_URL}", flush=True)
    print(f"Samples per class: {SAMPLES_PER_CLASS}", flush=True)
    print("=" * 60, flush=True)

    if dry_run:
        print("[dry-run] Loading datasets only — no endpoint calls.", flush=True)
        tqa = load_truthfulqa(SAMPLES_PER_CLASS)
        db = load_deceptionbench(SAMPLES_PER_CLASS)
        print(f"[dry-run] TruthfulQA items: {len(tqa)}", flush=True)
        print(f"[dry-run] DeceptionBench items: {len(db)}", flush=True)
        print("[dry-run] Script structure OK. Run without --dry-run to score.", flush=True)
        sys.exit(0)

    # Warm endpoint
    warm_endpoint()

    summaries: list[dict[str, Any]] = []
    global_t0 = time.time()

    # --- TruthfulQA ---
    print("\n--- TruthfulQA ---", flush=True)
    tqa_items = load_truthfulqa(SAMPLES_PER_CLASS)
    tqa_summary = score_dataset(tqa_items, "truthfulqa", TRUTHFULQA_OUT)
    summaries.append(tqa_summary)

    # --- DeceptionBench ---
    print("\n--- DeceptionBench ---", flush=True)
    db_items = load_deceptionbench(SAMPLES_PER_CLASS)
    db_summary = score_dataset(db_items, "deceptionbench", DECEPTIONBENCH_OUT)
    summaries.append(db_summary)

    total_wall = time.time() - global_t0

    # Write summary
    summary = {
        "scorer": "phase2_scorer_A",
        "tribe_endpoint": SCORE_URL,
        "samples_per_class": SAMPLES_PER_CLASS,
        "total_wall_clock_seconds": round(total_wall, 1),
        "datasets": summaries,
        "phase1_context": (
            "Phase 1 raw resonance AUROC=0.383 (anti-correlated). "
            "Viable path: supervised classification over TRIBE embeddings. "
            "IMPORTANT: endpoint returns region_activations={} for all inputs as of 2026-04-14. "
            "Current feature vector = [emotion_score, emotion_ratio, total_activation] (3-dim). "
            "Ablation agent: use LogisticRegression / LinearSVC on these scalar features. "
            "If region_activations ever becomes populated, re-run to expand feature space."
        ),
    }

    with SUMMARY_OUT.open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[summary] Written to {SUMMARY_OUT}", flush=True)
    print(f"[summary] Total wall clock: {total_wall:.0f}s", flush=True)
    for s in summaries:
        print(
            f"  {s['dataset']}: n={s['n_scored']}, errors={s['n_errors']}, "
            f"es_auroc={s['emotion_score_auroc']}, er_auroc={s['emotion_ratio_auroc']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
