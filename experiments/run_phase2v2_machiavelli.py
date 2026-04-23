#!/usr/bin/env python3.13
"""Phase 2 v2 — MACHIAVELLI Deception + Manipulation Scorer through TRIBE v2.

Scores two axes:
  - Deception: 800 pairs (400 per class), morality_deception >= 1.0
  - Manipulation: 500 pairs (250 per class), morality_manipulation >= 1.0

Outputs:
  experiments/phase2v2_machiavelli_deception_results.jsonl
  experiments/phase2v2_machiavelli_manipulation_results.jsonl
  experiments/phase2v2_machiavelli_summary.json

Resume-capable via idx. Incremental save every 25 rows.

Length-artifact check: saves text_length in every row, computes length-AUROC.
If length-AUROC > 0.6, sets length_artifact_flag=true in summary.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCORE_URL = "https://zazmattaz--cerebrum-tribe-tribescorer-score.modal.run"
EXPERIMENTS = Path(__file__).parent

DECEPTION_OUT = EXPERIMENTS / "phase2v2_machiavelli_deception_results.jsonl"
MANIPULATION_OUT = EXPERIMENTS / "phase2v2_machiavelli_manipulation_results.jsonl"
SUMMARY_OUT = EXPERIMENTS / "phase2v2_machiavelli_summary.json"

# Target class sizes
DECEPTION_PER_CLASS = 400      # 800 total
MANIPULATION_PER_CLASS = 250   # 500 total

# At ~13.7% natural deceptive rate, need large scan limit to get 400 deceptive
DECEPTION_SCAN_LIMIT = 3000    # Will yield ~411 deceptive from this many pairs
MANIPULATION_SCAN_LIMIT = 3000

SAVE_EVERY = 25
MAX_CONSECUTIVE_ERRORS = 20
RETRY_SLEEP = 5.0
REQUEST_TIMEOUT = 180  # covers cold-start (30-90s) + contention overhead

# Length-artifact threshold: flag if length_auroc > 0.6
LENGTH_ARTIFACT_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Endpoint — subprocess curl to avoid Python SSL timeout on Modal cold-starts
# ---------------------------------------------------------------------------


def score_text(text: str) -> dict[str, Any]:
    """POST to TRIBE endpoint via curl subprocess."""
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
    """Fire a throwaway request to warm the Modal cold-start."""
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
    """Compute AUROC via Mann-Whitney U statistic."""
    pos = [s for s, lbl in zip(scores, labels) if lbl == 1]
    neg = [s for s, lbl in zip(scores, labels) if lbl == 0]
    if not pos or not neg:
        return 0.5
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def normalized_auroc(raw: float) -> tuple[float, bool]:
    """Return (max(raw, 1-raw), inverted: bool).

    Ensures AUROC is always reported as >= 0.5 for readability,
    with a flag indicating signal direction.
    """
    inverted = raw < 0.5
    return max(raw, 1.0 - raw), inverted


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


def load_existing_records(out_file: Path) -> list[dict[str, Any]]:
    """Load all previously saved records for AUROC seeding on resume."""
    records: list[dict[str, Any]] = []
    if not out_file.exists():
        return records
    with out_file.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def append_record(out_file: Path, record: dict[str, Any]) -> None:
    with out_file.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Dataset loading — wraps the scout loader and rebalances to per_class target
# ---------------------------------------------------------------------------


def load_deception_items(per_class: int, scan_limit: int) -> list[dict[str, Any]]:
    """Load MACHIAVELLI deception pairs, balanced to per_class each label.

    The scout loader's load_deception_pairs returns ~25% positive by default.
    We call with scan_limit and then re-balance to exactly per_class each side.
    If not enough positives, reports and proceeds with what's available.
    """
    import sys
    scout_path = str(EXPERIMENTS / "scouts" / "machiavelli")
    if scout_path not in sys.path:
        sys.path.insert(0, scout_path)
    from loader import load_deception_pairs  # type: ignore[import]

    print(f"[deception] Loading pairs (scan_limit={scan_limit})...", flush=True)
    raw_pairs = load_deception_pairs(limit=scan_limit, include_context=True, seed=42)

    pos = [(t, lbl) for t, lbl in raw_pairs if lbl == 1]
    neg = [(t, lbl) for t, lbl in raw_pairs if lbl == 0]
    print(
        f"[deception] Raw loaded: {len(pos)} deceptive, {len(neg)} non-deceptive",
        flush=True,
    )

    # If still not enough positives after scan_limit=3000, try harder
    if len(pos) < per_class:
        print(
            f"[deception] Only {len(pos)} deceptive available (target={per_class}). "
            "Proceeding with actual counts.",
            flush=True,
        )

    n_pos = min(len(pos), per_class)
    n_neg = min(len(neg), per_class)
    print(
        f"[deception] Balanced sample: {n_pos} deceptive + {n_neg} non-deceptive = {n_pos + n_neg} total",
        flush=True,
    )

    # Interleave for even class distribution during scoring
    items: list[dict[str, Any]] = []
    for i, ((text_p, _), (text_n, _)) in enumerate(zip(pos[:n_pos], neg[:n_neg])):
        items.append({"text": text_p, "label": 1})
        items.append({"text": text_n, "label": 0})

    # Append remainder if one class had more
    for text, lbl in pos[n_neg:n_pos]:
        items.append({"text": text, "label": 1})
    for text, lbl in neg[n_pos:n_neg]:
        items.append({"text": text, "label": 0})

    return items


def load_manipulation_items(per_class: int, scan_limit: int) -> list[dict[str, Any]]:
    """Load MACHIAVELLI manipulation pairs, balanced to per_class each label."""
    import sys
    scout_path = str(EXPERIMENTS / "scouts" / "machiavelli")
    if scout_path not in sys.path:
        sys.path.insert(0, scout_path)
    from loader import load_manipulation_pairs  # type: ignore[import]

    print(f"[manipulation] Loading pairs (scan_limit={scan_limit})...", flush=True)
    raw_pairs = load_manipulation_pairs(limit=scan_limit, include_context=True, seed=42)

    pos = [(t, lbl) for t, lbl in raw_pairs if lbl == 1]
    neg = [(t, lbl) for t, lbl in raw_pairs if lbl == 0]
    print(
        f"[manipulation] Raw loaded: {len(pos)} manipulative, {len(neg)} non-manipulative",
        flush=True,
    )

    if len(pos) < per_class:
        print(
            f"[manipulation] Only {len(pos)} manipulative available (target={per_class}). "
            "Proceeding with actual counts.",
            flush=True,
        )

    n_pos = min(len(pos), per_class)
    n_neg = min(len(neg), per_class)
    print(
        f"[manipulation] Balanced sample: {n_pos} manipulative + {n_neg} non-manipulative = {n_pos + n_neg} total",
        flush=True,
    )

    items: list[dict[str, Any]] = []
    for (text_p, _), (text_n, _) in zip(pos[:n_pos], neg[:n_neg]):
        items.append({"text": text_p, "label": 1})
        items.append({"text": text_n, "label": 0})

    for text, lbl in pos[n_neg:n_pos]:
        items.append({"text": text, "label": 1})
    for text, lbl in neg[n_pos:n_neg]:
        items.append({"text": text, "label": 0})

    return items


# ---------------------------------------------------------------------------
# Core scoring loop
# ---------------------------------------------------------------------------


def score_dataset(
    items: list[dict[str, Any]],
    dataset_name: str,
    out_file: Path,
) -> dict[str, Any]:
    """Score items through TRIBE, save incrementally, return summary stats."""
    existing = load_existing_indices(out_file)
    if existing:
        print(
            f"[{dataset_name}] Resuming — {len(existing)} already scored",
            flush=True,
        )

    all_records = load_existing_records(out_file)

    t0 = time.time()
    n_scored = 0
    n_errors = 0
    consecutive_errors = 0
    buffer: list[dict[str, Any]] = []

    for idx, item in enumerate(items):
        if idx in existing:
            continue

        text: str = item["text"]
        label: int = item["label"]

        try:
            resp = score_text(text)
            consecutive_errors = 0

            record: dict[str, Any] = {
                "idx": idx,
                "text_hash": text_hash(text),
                "label": label,
                "dataset": dataset_name,
                "text_length": len(text),
                "emotion_score": resp.get("emotion_score", 0.0),
                "emotion_ratio": resp.get("emotion_ratio", 0.0),
                "total_activation": resp.get("total_activation", 0.0),
                "timesteps": resp.get("timesteps", 0),
                "signature_shape": resp.get("signature_shape", ""),
                "region_activations": resp.get("region_activations", {}),
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
                f"  [{dataset_name}] idx={idx} error ({n_errors} total, "
                f"{consecutive_errors} consec): {exc}",
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
                    "text_length": len(text),
                    "emotion_score": resp.get("emotion_score", 0.0),
                    "emotion_ratio": resp.get("emotion_ratio", 0.0),
                    "total_activation": resp.get("total_activation", 0.0),
                    "timesteps": resp.get("timesteps", 0),
                    "signature_shape": resp.get("signature_shape", ""),
                    "region_activations": resp.get("region_activations", {}),
                }
                buffer.append(record)
                all_records.append(record)
                n_scored += 1
                n_errors -= 1  # retry succeeded
            except Exception as exc2:
                print(
                    f"  [{dataset_name}] idx={idx} retry also failed: {exc2}",
                    flush=True,
                )

        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            print(
                f"  [{dataset_name}] {MAX_CONSECUTIVE_ERRORS} consecutive errors — "
                "stopping early, saving partial results",
                flush=True,
            )
            break

        # Progress log every SAVE_EVERY processed
        processed = (idx + 1) - len(existing)
        total_remaining = len(items) - len(existing)
        if processed > 0 and processed % SAVE_EVERY == 0:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total_remaining - processed) / rate if rate > 0 else 0
            print(
                f"  [{dataset_name}] {processed}/{total_remaining} "
                f"({rate:.3f} texts/s, ETA {eta:.0f}s, errors={n_errors})",
                flush=True,
            )

    # Flush remaining buffer
    for rec in buffer:
        append_record(out_file, rec)

    wall_clock = time.time() - t0

    # Compute AUROCs from all records (existing + newly scored)
    labels_all = [r["label"] for r in all_records]
    es_scores = [r["emotion_score"] for r in all_records]
    er_scores = [r["emotion_ratio"] for r in all_records]
    length_scores = [float(r["text_length"]) for r in all_records]

    es_raw = auroc(es_scores, labels_all) if labels_all else 0.5
    er_raw = auroc(er_scores, labels_all) if labels_all else 0.5
    len_raw = auroc(length_scores, labels_all) if labels_all else 0.5

    es_norm, es_inv = normalized_auroc(es_raw)
    er_norm, er_inv = normalized_auroc(er_raw)
    len_norm, len_inv = normalized_auroc(len_raw)

    n_class_1 = sum(1 for r in all_records if r["label"] == 1)
    n_class_0 = sum(1 for r in all_records if r["label"] == 0)

    print(f"\n[{dataset_name}] Complete: {n_scored} newly scored, {n_errors} errors, {wall_clock:.0f}s", flush=True)
    print(f"[{dataset_name}] Total records: {len(all_records)} (class_0={n_class_0}, class_1={n_class_1})", flush=True)
    print(
        f"[{dataset_name}] emotion_score  AUROC={es_raw:.4f} -> normalized={es_norm:.4f}"
        f"{' [INVERTED]' if es_inv else ''}",
        flush=True,
    )
    print(
        f"[{dataset_name}] emotion_ratio  AUROC={er_raw:.4f} -> normalized={er_norm:.4f}"
        f"{' [INVERTED]' if er_inv else ''}",
        flush=True,
    )
    print(
        f"[{dataset_name}] text_length    AUROC={len_raw:.4f} -> normalized={len_norm:.4f}"
        f"{' [INVERTED]' if len_inv else ''}"
        f"{' [LENGTH ARTIFACT FLAG]' if len_norm > LENGTH_ARTIFACT_THRESHOLD else ''}",
        flush=True,
    )

    direction_note = {}
    if es_inv:
        direction_note["emotion_score"] = "inverted — positive class scores lower than negative"
    if er_inv:
        direction_note["emotion_ratio"] = "inverted — positive class scores lower than negative"
    if len_inv:
        direction_note["text_length"] = "inverted — positive class texts shorter than negative"

    return {
        "n_scored": len(all_records),
        "n_class_0": n_class_0,
        "n_class_1": n_class_1,
        "emotion_score_auroc": round(es_norm, 4),
        "emotion_score_auroc_raw": round(es_raw, 4),
        "emotion_ratio_auroc": round(er_norm, 4),
        "emotion_ratio_auroc_raw": round(er_raw, 4),
        "length_auroc": round(len_norm, 4),
        "length_auroc_raw": round(len_raw, 4),
        "n_errors": n_errors,
        "wall_clock_seconds": round(wall_clock, 1),
        "direction_notes": direction_note,
    }


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------


def dry_run_check() -> None:
    """Verify loaders work without hitting the endpoint."""
    print("[dry-run] Testing deception loader...", flush=True)
    dec_items = load_deception_items(per_class=10, scan_limit=200)
    print(f"[dry-run] Deception items: {len(dec_items)}", flush=True)

    print("[dry-run] Testing manipulation loader...", flush=True)
    man_items = load_manipulation_items(per_class=10, scan_limit=200)
    print(f"[dry-run] Manipulation items: {len(man_items)}", flush=True)

    lengths = [len(item["text"]) for item in dec_items[:5]]
    print(f"[dry-run] Sample text lengths (first 5): {lengths}", flush=True)
    print("[dry-run] Script structure OK. Run without --dry-run to score.", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    print("=" * 60, flush=True)
    print("Phase 2 v2 — MACHIAVELLI Deception + Manipulation Scorer", flush=True)
    print(f"Endpoint: {SCORE_URL}", flush=True)
    print(f"Deception target: {DECEPTION_PER_CLASS}/class ({DECEPTION_PER_CLASS * 2} total)", flush=True)
    print(f"Manipulation target: {MANIPULATION_PER_CLASS}/class ({MANIPULATION_PER_CLASS * 2} total)", flush=True)
    print(f"Length-artifact flag threshold: AUROC > {LENGTH_ARTIFACT_THRESHOLD}", flush=True)
    print("=" * 60, flush=True)

    if dry_run:
        dry_run_check()
        sys.exit(0)

    # Warm endpoint once before both datasets
    warm_endpoint()

    global_t0 = time.time()

    # --- Deception axis ---
    print("\n--- MACHIAVELLI Deception (800 pairs target) ---", flush=True)
    dec_items = load_deception_items(
        per_class=DECEPTION_PER_CLASS,
        scan_limit=DECEPTION_SCAN_LIMIT,
    )
    dec_summary = score_dataset(dec_items, "machiavelli_deception", DECEPTION_OUT)

    # --- Manipulation axis ---
    print("\n--- MACHIAVELLI Manipulation (500 pairs target) ---", flush=True)
    man_items = load_manipulation_items(
        per_class=MANIPULATION_PER_CLASS,
        scan_limit=MANIPULATION_SCAN_LIMIT,
    )
    man_summary = score_dataset(man_items, "machiavelli_manipulation", MANIPULATION_OUT)

    total_wall = time.time() - global_t0

    # Determine length_artifact_flag (either axis triggers it)
    length_artifact_flag = (
        dec_summary["length_auroc"] > LENGTH_ARTIFACT_THRESHOLD
        or man_summary["length_auroc"] > LENGTH_ARTIFACT_THRESHOLD
    )

    summary: dict[str, Any] = {
        "scorer": "phase2v2_machiavelli",
        "tribe_endpoint": SCORE_URL,
        "deception": dec_summary,
        "manipulation": man_summary,
        "endpoint_wall_clock_seconds": round(total_wall, 1),
        "length_artifact_flag": length_artifact_flag,
        "length_artifact_note": (
            "MACHIAVELLI pairs are scene_narrative + action. Both classes share the "
            "same narrative structure (~370 char avg), so are length-matched by construction. "
            f"Flag is True if length_auroc > {LENGTH_ARTIFACT_THRESHOLD} on either axis."
        ),
        "region_activations_note": (
            "Endpoint returns region_activations={} for all inputs as of 2026-04-14. "
            "Feature vector = [emotion_score, emotion_ratio, total_activation] (3-dim scalars). "
            "Ablation agent: use LogisticRegression/LinearSVC on scalar features."
        ),
        "phase1_context": (
            "Phase 1 raw resonance AUROC=0.383 (anti-correlated). "
            "Phase 2 v1 DeceptionBench hit AUROC=0.997 — suspected length leakage. "
            "MACHIAVELLI is length-matched by construction, so AUROC here is the clean signal."
        ),
    }

    with SUMMARY_OUT.open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[summary] Written to {SUMMARY_OUT}", flush=True)
    print(f"[summary] Total wall clock: {total_wall:.0f}s ({total_wall / 3600:.2f}h)", flush=True)
    print(f"[summary] length_artifact_flag: {length_artifact_flag}", flush=True)
    print("\nDeception axis:", flush=True)
    print(
        f"  n={dec_summary['n_scored']} (class_0={dec_summary['n_class_0']}, "
        f"class_1={dec_summary['n_class_1']}), "
        f"errors={dec_summary['n_errors']}, "
        f"es_auroc={dec_summary['emotion_score_auroc']}, "
        f"er_auroc={dec_summary['emotion_ratio_auroc']}, "
        f"length_auroc={dec_summary['length_auroc']}",
        flush=True,
    )
    print("Manipulation axis:", flush=True)
    print(
        f"  n={man_summary['n_scored']} (class_0={man_summary['n_class_0']}, "
        f"class_1={man_summary['n_class_1']}), "
        f"errors={man_summary['n_errors']}, "
        f"es_auroc={man_summary['emotion_score_auroc']}, "
        f"er_auroc={man_summary['emotion_ratio_auroc']}, "
        f"length_auroc={man_summary['length_auroc']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
