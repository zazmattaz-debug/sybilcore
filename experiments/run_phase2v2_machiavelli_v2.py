#!/usr/bin/env python3.13
"""Phase 2v2 — MACHIAVELLI Deception + Manipulation Scorer v2 (AgentDojo pattern).

Key fixes vs v1:
  - urllib (not curl subprocess — curl returned rc=0 on 303 async path)
  - 500-char truncation before Modal
  - ThreadPoolExecutor(5 workers) for concurrent scoring
  - 400s per-request timeout (covers 303 async polling)
  - Atomic JSONL append per record (thread-safe)

Data:
  - 800 deception pairs (400/class) from MACHIAVELLI morality_deception >= 1
  - 500 manipulation pairs (250/class) from MACHIAVELLI morality_manipulation >= 1

Outputs:
  experiments/phase2v2_machiavelli_deception_results.jsonl
  experiments/phase2v2_machiavelli_manipulation_results.jsonl
  experiments/phase2v2_machiavelli_summary.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCORE_URL = "https://zazmattaz--cerebrum-tribe-tribescorer-score.modal.run"
EXPERIMENTS = Path(__file__).parent

DECEPTION_OUT = EXPERIMENTS / "phase2v2_machiavelli_deception_results.jsonl"
MANIPULATION_OUT = EXPERIMENTS / "phase2v2_machiavelli_manipulation_results.jsonl"
SUMMARY_OUT = EXPERIMENTS / "phase2v2_machiavelli_summary.json"

DECEPTION_PER_CLASS = 400      # 800 total
MANIPULATION_PER_CLASS = 250   # 500 total
DECEPTION_SCAN_LIMIT = 3000
MANIPULATION_SCAN_LIMIT = 3000

# MACHIAVELLI texts are scene_narrative + action (~370 char avg; both classes match by construction)
TRUNCATE_CHARS = 500

N_WORKERS = 5
REQUEST_TIMEOUT = 400
RETRY_SLEEP = 5.0
MAX_CONSECUTIVE_ERRORS = 20
WALL_CLOCK_CAP = 8 * 3600      # 8h safety cap (1300 texts)
SAVE_EVERY = 10
LENGTH_ARTIFACT_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# TRIBE endpoint
# ---------------------------------------------------------------------------


def score_text(text: str) -> dict[str, Any]:
    """POST to TRIBE endpoint. Handles 200 direct or 303 async redirect."""
    payload = json.dumps({"text": text, "modality": "text"}).encode()

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
            return None

    opener = urllib.request.build_opener(NoRedirect, urllib.request.HTTPSHandler)
    req = urllib.request.Request(
        SCORE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    poll_url: str = ""
    try:
        resp = opener.open(req, timeout=REQUEST_TIMEOUT)
        body = resp.read().decode()
        if body.strip():
            return json.loads(body)
        raise RuntimeError(f"200 response but empty body (status {resp.status})")
    except urllib.error.HTTPError as e:
        if e.code == 303:
            poll_url = e.headers.get("Location", "")
            if not poll_url:
                raise RuntimeError("303 redirect with no Location header")
        else:
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:200]!r}")

    deadline = time.time() + REQUEST_TIMEOUT
    poll_interval = 2.0
    while time.time() < deadline:
        try:
            poll_req = urllib.request.Request(poll_url, method="GET")
            poll_resp = opener.open(poll_req, timeout=30)
            poll_body = poll_resp.read().decode()
            if poll_body.strip():
                return json.loads(poll_body)
        except urllib.error.HTTPError as pe:
            if pe.code in (303, 202):
                pass
            else:
                raise RuntimeError(f"Poll HTTP {pe.code}: {pe.read()[:200]!r}")
        except (TimeoutError, OSError):
            pass
        time.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.5, 15.0)

    raise RuntimeError(f"Score timed out after {REQUEST_TIMEOUT}s")


def warm_endpoint() -> None:
    print("[warm] Warming TRIBE endpoint...", flush=True)
    try:
        score_text("This is a warm-up request.")
        print("[warm] Done.", flush=True)
    except Exception as exc:
        print(f"[warm] Warning — warm-up failed: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def auroc_raw(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, lbl in zip(scores, labels) if lbl == 1]
    neg = [s for s, lbl in zip(scores, labels) if lbl == 0]
    if not pos or not neg:
        return 0.5
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def normalized_auroc(scores: list[float], labels: list[int]) -> tuple[float, str]:
    raw = auroc_raw(scores, labels)
    if raw >= 0.5:
        return round(raw, 4), "pos"
    return round(1.0 - raw, 4), "inv"


def load_existing(out_file: Path) -> tuple[set[int], list[dict[str, Any]]]:
    existing_idx: set[int] = set()
    records: list[dict[str, Any]] = []
    if not out_file.exists():
        return existing_idx, records
    with out_file.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                existing_idx.add(rec["idx"])
                records.append(rec)
            except (json.JSONDecodeError, KeyError):
                pass
    return existing_idx, records


def append_record(out_file: Path, record: dict[str, Any]) -> None:
    with out_file.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_deception_items(per_class: int, scan_limit: int) -> list[dict[str, Any]]:
    scout_path = str(EXPERIMENTS / "scouts" / "machiavelli")
    if scout_path not in sys.path:
        sys.path.insert(0, scout_path)
    from loader import load_deception_pairs  # type: ignore[import]

    print(f"[deception] Loading pairs (scan_limit={scan_limit})...", flush=True)
    raw_pairs = load_deception_pairs(limit=scan_limit, include_context=True, seed=42)

    pos = [(t, lbl) for t, lbl in raw_pairs if lbl == 1]
    neg = [(t, lbl) for t, lbl in raw_pairs if lbl == 0]
    print(
        f"[deception] Raw: {len(pos)} deceptive, {len(neg)} non-deceptive",
        flush=True,
    )

    n_pos = min(len(pos), per_class)
    n_neg = min(len(neg), per_class)
    print(
        f"[deception] Balanced: {n_pos} + {n_neg} = {n_pos + n_neg} total",
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


def load_manipulation_items(per_class: int, scan_limit: int) -> list[dict[str, Any]]:
    scout_path = str(EXPERIMENTS / "scouts" / "machiavelli")
    if scout_path not in sys.path:
        sys.path.insert(0, scout_path)
    from loader import load_manipulation_pairs  # type: ignore[import]

    print(f"[manipulation] Loading pairs (scan_limit={scan_limit})...", flush=True)
    raw_pairs = load_manipulation_pairs(limit=scan_limit, include_context=True, seed=42)

    pos = [(t, lbl) for t, lbl in raw_pairs if lbl == 1]
    neg = [(t, lbl) for t, lbl in raw_pairs if lbl == 0]
    print(
        f"[manipulation] Raw: {len(pos)} manipulative, {len(neg)} non-manipulative",
        flush=True,
    )

    n_pos = min(len(pos), per_class)
    n_neg = min(len(neg), per_class)
    print(
        f"[manipulation] Balanced: {n_pos} + {n_neg} = {n_pos + n_neg} total",
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
# Scoring loop (concurrent, atomic appends)
# ---------------------------------------------------------------------------


def score_dataset(
    items: list[dict[str, Any]],
    dataset_name: str,
    out_file: Path,
) -> dict[str, Any]:
    """Score items through TRIBE with N_WORKERS threads. Resume-capable via idx."""
    existing_idx, all_records = load_existing(out_file)
    if existing_idx:
        print(
            f"[{dataset_name}] Resuming — {len(existing_idx)} already scored",
            flush=True,
        )

    pending: list[tuple[int, dict[str, Any]]] = [
        (idx, item)
        for idx, item in enumerate(items)
        if idx not in existing_idx
    ]
    print(
        f"[{dataset_name}] Pending: {len(pending)} / {len(items)} total",
        flush=True,
    )

    t0 = time.time()
    lock = threading.Lock()
    n_scored = 0
    n_errors = 0
    consecutive_errors = 0
    stop_flag = threading.Event()
    work_idx = 0

    def worker(worker_id: int) -> None:
        nonlocal n_scored, n_errors, consecutive_errors, work_idx

        while True:
            if time.time() - t0 > WALL_CLOCK_CAP:
                print(f"  [w{worker_id}] wall-clock cap reached", flush=True)
                stop_flag.set()
                return
            if stop_flag.is_set():
                return

            with lock:
                if work_idx >= len(pending):
                    return
                local_idx = work_idx
                work_idx += 1

            idx, item = pending[local_idx]
            raw_text: str = item["text"]
            label: int = item["label"]
            text_length_pre_trunc = len(raw_text)

            truncated = raw_text[:TRUNCATE_CHARS]
            text_length_scored = len(truncated)

            def make_record(resp: dict[str, Any]) -> dict[str, Any]:
                return {
                    "idx": idx,
                    "text_hash": text_hash(raw_text),
                    "label": label,
                    "dataset": dataset_name,
                    "text_length_pre_trunc": text_length_pre_trunc,
                    "text_length_scored": text_length_scored,
                    "emotion_score": resp.get("emotion_score", 0.0),
                    "emotion_ratio": resp.get("emotion_ratio", 0.0),
                    "total_activation": resp.get("total_activation", 0.0),
                    "timesteps": resp.get("timesteps", 0),
                    "signature_shape": resp.get("signature_shape", ""),
                    "region_activations": resp.get("region_activations", {}),
                }

            def try_score() -> dict[str, Any] | None:
                try:
                    return score_text(truncated)
                except Exception as exc:
                    with lock:
                        print(
                            f"  [w{worker_id}] [{dataset_name}] idx={idx} error: {exc}",
                            flush=True,
                        )
                    return None

            resp = try_score()
            if resp is None:
                time.sleep(RETRY_SLEEP)
                resp = try_score()

            with lock:
                if resp is not None:
                    consecutive_errors = 0
                    record = make_record(resp)
                    all_records.append(record)
                    n_scored += 1
                    append_record(out_file, record)
                    if n_scored % SAVE_EVERY == 0:
                        elapsed = time.time() - t0
                        rate = n_scored / elapsed if elapsed > 0 else 0
                        remaining = len(pending) - n_scored
                        eta = remaining / rate if rate > 0 else 0
                        print(
                            f"  [{dataset_name}] {n_scored + len(existing_idx)}"
                            f"/{len(items)} "
                            f"({rate:.3f}/s ETA {eta:.0f}s errors={n_errors})",
                            flush=True,
                        )
                else:
                    n_errors += 1
                    consecutive_errors += 1
                    print(
                        f"  [{dataset_name}] idx={idx} failed "
                        f"(errors={n_errors} consec={consecutive_errors})",
                        flush=True,
                    )
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        print(
                            f"[{dataset_name}] {MAX_CONSECUTIVE_ERRORS} "
                            "consecutive errors — stopping",
                            flush=True,
                        )
                        stop_flag.set()

    threads = [
        threading.Thread(target=worker, args=(i,), daemon=True)
        for i in range(N_WORKERS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    wall_clock = time.time() - t0

    labels_all = [r["label"] for r in all_records]
    es_scores = [r["emotion_score"] for r in all_records]
    er_scores = [r["emotion_ratio"] for r in all_records]
    length_scores = [float(r["text_length_pre_trunc"]) for r in all_records]

    es_auroc, es_dir = normalized_auroc(es_scores, labels_all) if labels_all else (0.5, "pos")
    er_auroc, er_dir = normalized_auroc(er_scores, labels_all) if labels_all else (0.5, "pos")
    len_auroc, len_dir = normalized_auroc(length_scores, labels_all) if labels_all else (0.5, "pos")

    n_class0 = sum(1 for r in all_records if r["label"] == 0)
    n_class1 = sum(1 for r in all_records if r["label"] == 1)

    print(
        f"\n[{dataset_name}] Complete: {n_scored} newly scored, "
        f"{n_errors} errors, {wall_clock:.0f}s",
        flush=True,
    )
    print(
        f"[{dataset_name}] Total: {len(all_records)} "
        f"(class0={n_class0}, class1={n_class1})",
        flush=True,
    )
    print(
        f"[{dataset_name}] emotion_score AUROC={es_auroc:.4f} ({es_dir})",
        flush=True,
    )
    print(
        f"[{dataset_name}] emotion_ratio AUROC={er_auroc:.4f} ({er_dir})",
        flush=True,
    )
    print(
        f"[{dataset_name}] text_length AUROC={len_auroc:.4f} ({len_dir})"
        f"{' [LENGTH ARTIFACT FLAG]' if len_auroc > LENGTH_ARTIFACT_THRESHOLD else ''}",
        flush=True,
    )

    return {
        "n_scored": len(all_records),
        "n_class_0": n_class0,
        "n_class_1": n_class1,
        "n_errors": n_errors,
        "wall_clock_seconds": round(wall_clock, 1),
        "emotion_score_auroc": es_auroc,
        "emotion_score_direction": es_dir,
        "emotion_ratio_auroc": er_auroc,
        "emotion_ratio_direction": er_dir,
        "length_auroc": len_auroc,
        "length_direction": len_dir,
        "length_artifact_flag": len_auroc > LENGTH_ARTIFACT_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    print("=" * 60, flush=True)
    print("Phase 2v2 — MACHIAVELLI Scorer v2", flush=True)
    print(f"Endpoint: {SCORE_URL}", flush=True)
    print(f"Deception target: {DECEPTION_PER_CLASS}/class", flush=True)
    print(f"Manipulation target: {MANIPULATION_PER_CLASS}/class", flush=True)
    print(f"Truncate to: {TRUNCATE_CHARS} chars", flush=True)
    print(f"Workers: {N_WORKERS}", flush=True)
    print("=" * 60, flush=True)

    global_t0 = time.time()

    # Load both datasets
    print("\n[load] Loading deception items...", flush=True)
    dec_items = load_deception_items(
        per_class=DECEPTION_PER_CLASS,
        scan_limit=DECEPTION_SCAN_LIMIT,
    )

    print("\n[load] Loading manipulation items...", flush=True)
    man_items = load_manipulation_items(
        per_class=MANIPULATION_PER_CLASS,
        scan_limit=MANIPULATION_SCAN_LIMIT,
    )

    if dry_run:
        print("\n[dry-run] Data loaded OK. No endpoint calls.", flush=True)
        print(f"[dry-run] Deception items: {len(dec_items)}", flush=True)
        print(f"[dry-run] Manipulation items: {len(man_items)}", flush=True)
        sample_lens = [len(item["text"]) for item in dec_items[:5]]
        print(f"[dry-run] Sample deception text lengths: {sample_lens}", flush=True)
        print("[dry-run] Run without --dry-run to score.", flush=True)
        sys.exit(0)

    # Warm endpoint
    warm_endpoint()

    # Score deception axis
    print("\n--- MACHIAVELLI Deception (target 800 pairs) ---", flush=True)
    dec_summary = score_dataset(dec_items, "machiavelli_deception", DECEPTION_OUT)

    # Score manipulation axis
    print("\n--- MACHIAVELLI Manipulation (target 500 pairs) ---", flush=True)
    man_summary = score_dataset(man_items, "machiavelli_manipulation", MANIPULATION_OUT)

    total_wall = time.time() - global_t0

    length_artifact_flag = (
        dec_summary["length_artifact_flag"] or man_summary["length_artifact_flag"]
    )

    summary: dict[str, Any] = {
        "scorer": "phase2v2_machiavelli_v2",
        "tribe_endpoint": SCORE_URL,
        "truncate_chars": TRUNCATE_CHARS,
        "n_workers": N_WORKERS,
        "deception": dec_summary,
        "manipulation": man_summary,
        "total_wall_clock_seconds": round(total_wall, 1),
        "length_artifact_flag": length_artifact_flag,
        "length_artifact_note": (
            "MACHIAVELLI pairs are scene_narrative + action. Both classes share the "
            "same narrative structure (~370 char avg), so are length-matched by construction. "
            f"Flag is True if length_auroc > {LENGTH_ARTIFACT_THRESHOLD} on either axis."
        ),
        "region_activations_note": (
            "Endpoint returns region_activations={} for all inputs as of 2026-04-14. "
            "Feature vector = [emotion_score, emotion_ratio, total_activation] (3-dim scalars)."
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
        f"  n={dec_summary['n_scored']} "
        f"(class0={dec_summary['n_class_0']}, class1={dec_summary['n_class_1']}) "
        f"es_auroc={dec_summary['emotion_score_auroc']:.4f} "
        f"er_auroc={dec_summary['emotion_ratio_auroc']:.4f} "
        f"len_auroc={dec_summary['length_auroc']:.4f}",
        flush=True,
    )
    print("Manipulation axis:", flush=True)
    print(
        f"  n={man_summary['n_scored']} "
        f"(class0={man_summary['n_class_0']}, class1={man_summary['n_class_1']}) "
        f"es_auroc={man_summary['emotion_score_auroc']:.4f} "
        f"er_auroc={man_summary['emotion_ratio_auroc']:.4f} "
        f"len_auroc={man_summary['length_auroc']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
