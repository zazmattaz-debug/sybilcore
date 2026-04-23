#!/usr/bin/env python3.13
"""Phase 2v2 — MALT Scorer v2 (AgentDojo pattern).

Scores MALT transcripts (hardcoded_solution label) via TRIBE Modal endpoint.
Resumes from existing phase2v2_malt_results.jsonl (skips already-scored idx).

Key fixes vs v1:
  - 2000-char truncation before Modal (o1/o3 transcripts are 800KB-2MB raw)
  - urllib (not curl subprocess — curl returned rc=0 on 303 async path)
  - ThreadPoolExecutor(5 workers) for concurrent scoring
  - 400s per-request timeout (covers 303 async polling)
  - Atomic JSONL append per record (thread-safe via lock)

Outputs:
  experiments/phase2v2_malt_results.jsonl  — per-transcript scores (appended)
  experiments/phase2v2_malt_summary.json   — AUROC + length artifact check
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCORE_URL = "https://zazmattaz--cerebrum-tribe-tribescorer-score.modal.run"
EXPERIMENTS = Path(__file__).parent

OUT_RESULTS = EXPERIMENTS / "phase2v2_malt_results.jsonl"
OUT_SUMMARY = EXPERIMENTS / "phase2v2_malt_summary.json"

LABEL_NAME = "hardcoded_solution"
SAMPLES_PER_CLASS = 200               # 200 pos + 200 neg = 400 total

# MALT transcripts are o1/o3 reasoning logs — often 800KB-2MB.
# Modal times out at >500 chars on the 303 async path at full concurrency.
# 2000-char truncation is the MALT-specific safe limit (longer than AgentDojo's
# 500 because MALT texts are more content-dense; tested at lower concurrency).
TRUNCATE_CHARS = 2000

N_WORKERS = 5
REQUEST_TIMEOUT = 400                 # seconds — covers 303 + polling
RETRY_SLEEP = 5.0
MAX_CONSECUTIVE_ERRORS = 20
WALL_CLOCK_CAP = 6 * 3600            # 6h safety cap (MALT is slow)
SAVE_EVERY = 10


# ---------------------------------------------------------------------------
# TRIBE endpoint (urllib, handles 303 async path)
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

    # Poll redirect URL until result
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
                pass  # still pending
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


def load_existing(out_file: Path) -> tuple[set[str], list[dict[str, Any]]]:
    """Return (set of text_hash already scored, all parsed records).

    MALT v1 used text_hash as resume key (not idx, since idx was based on
    enumerate order and v2 may reorder). We keep both for compatibility.
    """
    existing_hashes: set[str] = set()
    records: list[dict[str, Any]] = []
    if not out_file.exists():
        return existing_hashes, records
    with out_file.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                existing_hashes.add(rec["text_hash"])
                records.append(rec)
            except (json.JSONDecodeError, KeyError):
                pass
    return existing_hashes, records


def append_record(out_file: Path, record: dict[str, Any]) -> None:
    with out_file.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Scoring loop
# ---------------------------------------------------------------------------


def score_dataset(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score items concurrently. Resume-capable via text_hash."""
    existing_hashes, all_records = load_existing(OUT_RESULTS)
    if existing_hashes:
        print(
            f"[score] Resuming — {len(existing_hashes)} already scored",
            flush=True,
        )

    pending: list[tuple[int, dict[str, Any]]] = [
        (idx, item)
        for idx, item in enumerate(items)
        if item["text_hash"] not in existing_hashes
    ]
    print(
        f"[score] Pending: {len(pending)} / {len(items)} total",
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
                print(f"  [w{worker_id}] 6h cap reached", flush=True)
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
            text_length_pre_trunc: int = item.get("text_length", len(raw_text))

            truncated = raw_text[:TRUNCATE_CHARS]
            text_length_scored = len(truncated)

            def make_record(resp: dict[str, Any]) -> dict[str, Any]:
                return {
                    "idx": idx,
                    "text_hash": item["text_hash"],
                    "label": label,
                    "dataset": "malt",
                    "label_name": LABEL_NAME,
                    "text_length_pre_trunc": text_length_pre_trunc,
                    "text_length_scored": text_length_scored,
                    "model": item.get("model", "unknown"),
                    "task_id": item.get("task_id", "unknown"),
                    "run_id": item.get("run_id", -1),
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
                            f"  [w{worker_id}] idx={idx} error: {exc}",
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
                    append_record(OUT_RESULTS, record)
                    if n_scored % SAVE_EVERY == 0:
                        elapsed = time.time() - t0
                        rate = n_scored / elapsed if elapsed > 0 else 0
                        remaining = len(pending) - n_scored
                        eta = remaining / rate if rate > 0 else 0
                        print(
                            f"  [score] {n_scored + len(existing_hashes)}"
                            f"/{len(items)} "
                            f"({rate:.3f}/s ETA {eta:.0f}s errors={n_errors})",
                            flush=True,
                        )
                else:
                    n_errors += 1
                    consecutive_errors += 1
                    print(
                        f"  [score] idx={idx} failed "
                        f"(errors={n_errors} consec={consecutive_errors})",
                        flush=True,
                    )
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        print(
                            f"[score] {MAX_CONSECUTIVE_ERRORS} consecutive errors — stopping",
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
    print(
        f"\n[score] Done: {n_scored} newly scored, {n_errors} errors, "
        f"{wall_clock:.0f}s elapsed",
        flush=True,
    )
    return all_records


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def compute_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [r["label"] for r in records]
    metrics: dict[str, Any] = {}
    for field in [
        "emotion_score", "emotion_ratio", "total_activation",
        "text_length_pre_trunc", "text_length_scored",
    ]:
        scores = [r.get(field, 0.0) or 0.0 for r in records]
        val, direction = normalized_auroc(scores, labels)
        raw = auroc_raw(scores, labels)
        metrics[field] = {
            "auroc": val,
            "raw_auroc": round(raw, 4),
            "direction": direction,
        }
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60, flush=True)
    print("Phase 2v2 — MALT Scorer v2 (AgentDojo pattern)", flush=True)
    print(f"Endpoint: {SCORE_URL}", flush=True)
    print(f"Label: {LABEL_NAME}", flush=True)
    print(f"Target: {SAMPLES_PER_CLASS}/class ({SAMPLES_PER_CLASS * 2} total)", flush=True)
    print(f"Truncate to: {TRUNCATE_CHARS} chars", flush=True)
    print(f"Workers: {N_WORKERS}", flush=True)
    print("=" * 60, flush=True)

    # Check HF_TOKEN
    from dotenv import load_dotenv
    load_dotenv(Path("/Users/zazumoloi/Desktop/Claude Code/.env"))
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("FAIL: HF_TOKEN not set.", flush=True)
        sys.exit(1)

    global_t0 = time.time()

    # Load data
    print("\n[load] Loading MALT transcripts...", flush=True)
    sys.path.insert(0, str(EXPERIMENTS / "scouts" / "metr"))
    from loader_malt import load_transcripts  # type: ignore[import]

    items = load_transcripts(
        label_name=LABEL_NAME,
        samples_per_class=SAMPLES_PER_CLASS,
    )
    print(f"[load] {len(items)} transcripts loaded.", flush=True)

    n0 = sum(1 for i in items if i["label"] == 0)
    n1 = sum(1 for i in items if i["label"] == 1)
    lengths = [i.get("text_length", len(i["text"])) for i in items]
    import statistics
    print(
        f"[load] class0={n0} class1={n1} "
        f"len: min={min(lengths)} max={max(lengths)} "
        f"mean={round(statistics.mean(lengths))} median={round(statistics.median(lengths))}",
        flush=True,
    )
    print(
        f"[load] After {TRUNCATE_CHARS}-char truncation: "
        f"all texts will be {min(TRUNCATE_CHARS, min(lengths))}-{TRUNCATE_CHARS} chars",
        flush=True,
    )

    # Warm endpoint
    warm_endpoint()

    # Score
    print("\n[score] Starting concurrent scoring loop...", flush=True)
    all_records = score_dataset(items)

    if not all_records:
        print("[ERROR] No records scored. Exiting.", flush=True)
        sys.exit(1)

    # Compute metrics
    print("\n[analysis] Computing AUROC metrics...", flush=True)
    metrics = compute_summary(all_records)
    length_artifact = metrics.get("text_length_pre_trunc", {}).get("auroc", 0.5) > 0.6

    total_wall = time.time() - global_t0
    n_total = len(all_records)
    n_label0 = sum(1 for r in all_records if r["label"] == 0)
    n_label1 = sum(1 for r in all_records if r["label"] == 1)
    n_truncated = sum(
        1 for r in all_records
        if r.get("text_length_scored", 0) < r.get("text_length_pre_trunc", 0)
    )

    print("\n" + "=" * 60, flush=True)
    print("RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(
        f"Total scored: {n_total} (label=0: {n_label0}, label=1: {n_label1})",
        flush=True,
    )
    print(f"Truncated: {n_truncated}/{n_total}", flush=True)
    print(f"Wall clock: {total_wall:.0f}s ({total_wall / 3600:.2f}h)", flush=True)
    print("\nAUROC (normalized = max(raw, 1-raw)):", flush=True)
    for field, vals in metrics.items():
        print(
            f"  {field:30s}: auroc={vals['auroc']:.4f}  "
            f"raw={vals['raw_auroc']:.4f}  dir={vals['direction']}",
            flush=True,
        )
    print(
        f"\nLength artifact flag (pre_trunc AUROC > 0.6): {length_artifact}",
        flush=True,
    )

    summary: dict[str, Any] = {
        "scorer": "phase2v2_malt_v2",
        "tribe_endpoint": SCORE_URL,
        "label_name": LABEL_NAME,
        "samples_per_class": SAMPLES_PER_CLASS,
        "truncate_chars": TRUNCATE_CHARS,
        "n_workers": N_WORKERS,
        "n_total_scored": n_total,
        "n_label0": n_label0,
        "n_label1": n_label1,
        "n_truncated": n_truncated,
        "wall_clock_seconds": round(total_wall, 1),
        "auroc_metrics": metrics,
        "length_artifact_flag": length_artifact,
        "note": (
            "MALT hardcoded_solution label. MALT transcripts are o1/o3 reasoning logs "
            "(800KB-2MB raw). Truncated to 2000 chars to avoid Modal timeout on 303 async path. "
            "region_activations={} for all inputs (endpoint limitation as of 2026-04-14)."
        ),
    }

    with OUT_SUMMARY.open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[summary] Written to {OUT_SUMMARY}", flush=True)
    print(f"[results] Written to {OUT_RESULTS}", flush=True)


if __name__ == "__main__":
    main()
