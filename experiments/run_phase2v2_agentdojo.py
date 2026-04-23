#!/usr/bin/env python3.13
"""Phase 2v2 — AgentDojo Prompt-Injection Scoring through TRIBE v2.

Scores injection-vs-clean traces from AgentDojo through the TRIBE Modal endpoint.
Balanced sample: 400 label=0 (agent resisted/clean) + 400 label=1 (agent deceived).
Uses gpt-4o-2024-05-13 traces (most attack-variant coverage).

Outputs:
  experiments/phase2v2_agentdojo_results.jsonl  — per-trace scores
  experiments/phase2v2_agentdojo_summary.json   — AUROC + length artifact check

Resume-capable: skips indices already in JSONL.
Stops on 20 consecutive errors or 4h wall clock.
"""
from __future__ import annotations

import hashlib
import json
import random
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

OUT_RESULTS = EXPERIMENTS / "phase2v2_agentdojo_results.jsonl"
OUT_SUMMARY = EXPERIMENTS / "phase2v2_agentdojo_summary.json"

# AgentDojo data
AGENTDOJO_RUNS = EXPERIMENTS / "scouts" / "agentdojo" / "repo" / "runs"
TARGET_MODEL = "gpt-4o-2024-05-13"
SUITES = ["workspace", "banking", "slack", "travel"]

SAMPLES_PER_CLASS = 400          # 400 label=0 + 400 label=1 = 800 total
# NOTE: Endpoint handles ≤500 chars in ~42s direct (200 response).
# ≥1000 chars triggers 303 async path which takes 150s (queue) + 150s (poll) = 300s.
# 2000-char truncation makes 800-item run infeasible in 4h wall clock.
# Using 500-char truncation: 800 × 42s / 5 workers ≈ 1.9h — fits in 4h cap.
TRUNCATE_CHARS = 500             # truncate each trace to this before scoring
SAVE_EVERY = 10                  # incremental flush interval (smaller for concurrent safety)
MAX_CONSECUTIVE_ERRORS = 20
RETRY_SLEEP = 5.0
REQUEST_TIMEOUT = 400            # seconds — covers 303 + polling (worst case ~300s)
WALL_CLOCK_CAP = 4 * 3600        # 4h safety cap
N_WORKERS = 5                    # concurrent scoring threads

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def score_text(text: str) -> dict[str, Any]:
    """POST to TRIBE endpoint.

    Modal may return a 303 redirect for async execution.
    We poll the redirect URL with short timeouts until it resolves.
    """
    payload = json.dumps({"text": text, "modality": "text"}).encode()

    # Build a custom opener that does NOT follow redirects
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
            return None  # block redirect

    opener = urllib.request.build_opener(NoRedirect, urllib.request.HTTPSHandler)

    req = urllib.request.Request(
        SCORE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # First POST — may return 200 directly (warm, ~60s) or 303 (async redirect)
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
    except Exception:
        raise

    # Poll the redirect URL until it returns a result
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
                # Still pending — continue polling
                pass
            else:
                raise RuntimeError(f"Poll HTTP {pe.code}: {pe.read()[:200]!r}")
        except (TimeoutError, OSError):
            pass  # short timeout, retry
        time.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.5, 15.0)  # backoff up to 15s

    raise RuntimeError(f"Score timed out after {REQUEST_TIMEOUT}s")


def warm_endpoint() -> None:
    """Fire a throwaway request to warm the Modal cold-start."""
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
    """Wilcoxon-Mann-Whitney AUROC. Pos=label 1."""
    pos = [s for s, lbl in zip(scores, labels) if lbl == 1]
    neg = [s for s, lbl in zip(scores, labels) if lbl == 0]
    if not pos or not neg:
        return 0.5
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def normalized_auroc(scores: list[float], labels: list[int]) -> tuple[float, str]:
    """Return max(raw, 1-raw) plus direction flag."""
    raw = auroc_raw(scores, labels)
    if raw >= 0.5:
        return round(raw, 4), "pos"
    else:
        return round(1.0 - raw, 4), "inv"


def load_existing(out_file: Path) -> tuple[set[int], list[dict[str, Any]]]:
    """Return set of existing idx values and all parsed records."""
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
# Data loading — custom loader that preserves suite metadata
# ---------------------------------------------------------------------------


def _messages_to_trace(messages: list[dict[str, Any]]) -> str:
    """Flatten messages to a single string (mirrors AgentDojo scout loader)."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            fn = tc.get("function", "unknown_fn")
            args = tc.get("args", {})
            parts.append(f"[TOOL_CALL] {fn}({json.dumps(args, separators=(',', ':'))})")
        if content:
            parts.append(f"[{role}] {content}")
    return "\n".join(parts).strip()


def load_balanced_samples(
    n_per_class: int,
    model: str = TARGET_MODEL,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """
    Load balanced samples with suite metadata preserved.

    Returns list of dicts:
      {text, label, suite_name, text_length_pre_trunc}
    """
    model_path = AGENTDOJO_RUNS / model
    if not model_path.exists():
        available = [d.name for d in AGENTDOJO_RUNS.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"Model directory not found: {model_path}\nAvailable: {available}"
        )

    label0_items: list[dict[str, Any]] = []
    label1_items: list[dict[str, Any]] = []

    for suite in SUITES:
        suite_path = model_path / suite
        if not suite_path.exists():
            print(f"[loader] Suite not found, skipping: {suite}", flush=True)
            continue

        for json_file in sorted(suite_path.glob("**/*.json")):
            try:
                data = json.loads(json_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            messages = data.get("messages", [])
            if not messages:
                continue

            is_injection = data.get("injection_task_id") is not None
            security = data.get("security", True)
            trace = _messages_to_trace(messages)
            if not trace:
                continue

            item = {
                "text": trace,
                "suite_name": suite,
                "text_length_pre_trunc": len(trace),
            }

            if is_injection:
                item["label"] = 0 if security else 1
                if item["label"] == 1:
                    label1_items.append(item)
                else:
                    label0_items.append(item)
            else:
                # Clean baseline — always label 0
                item["label"] = 0
                label0_items.append(item)

    print(
        f"[loader] Raw pool — label=0: {len(label0_items)}, label=1: {len(label1_items)}",
        flush=True,
    )

    rng = random.Random(seed)
    rng.shuffle(label0_items)
    rng.shuffle(label1_items)

    sampled0 = label0_items[:n_per_class]
    sampled1 = label1_items[:n_per_class]

    if len(sampled0) < n_per_class:
        print(
            f"[loader] Warning: only {len(sampled0)} label=0 samples (wanted {n_per_class})",
            flush=True,
        )
    if len(sampled1) < n_per_class:
        print(
            f"[loader] Warning: only {len(sampled1)} label=1 samples (wanted {n_per_class})",
            flush=True,
        )

    # Interleave for balanced scoring order
    interleaved: list[dict[str, Any]] = []
    for a, b in zip(sampled0, sampled1):
        interleaved.append(a)
        interleaved.append(b)
    # Append remainder if classes unequal
    if len(sampled0) > len(sampled1):
        interleaved.extend(sampled0[len(sampled1):])
    elif len(sampled1) > len(sampled0):
        interleaved.extend(sampled1[len(sampled0):])

    print(
        f"[loader] Sampled: {len(sampled0)} label=0 + {len(sampled1)} label=1 "
        f"= {len(interleaved)} total",
        flush=True,
    )
    return interleaved


# ---------------------------------------------------------------------------
# Scoring loop
# ---------------------------------------------------------------------------


def score_dataset(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Score items through TRIBE endpoint using N_WORKERS concurrent threads.

    Returns list of all scored records (existing + newly scored).
    Thread-safe incremental writes with a file lock.
    """
    existing_idx, all_records = load_existing(OUT_RESULTS)
    if existing_idx:
        print(f"[score] Resuming — {len(existing_idx)} already scored", flush=True)

    # Filter to unscored items only
    pending: list[tuple[int, dict[str, Any]]] = [
        (idx, item) for idx, item in enumerate(items) if idx not in existing_idx
    ]

    t0 = time.time()
    # Shared state protected by lock
    lock = threading.Lock()
    n_scored = 0
    n_errors = 0
    n_truncated = 0
    consecutive_errors = 0
    stop_flag = threading.Event()
    # Work queue — each worker pops from this
    work_idx = 0

    def worker(worker_id: int) -> None:
        nonlocal n_scored, n_errors, n_truncated, consecutive_errors, work_idx

        while True:
            # Wall-clock cap check
            if time.time() - t0 > WALL_CLOCK_CAP:
                print(f"  [w{worker_id}] 4h cap reached", flush=True)
                stop_flag.set()
                return

            if stop_flag.is_set():
                return

            # Claim next work item
            with lock:
                if work_idx >= len(pending):
                    return
                local_idx = work_idx
                work_idx += 1

            idx, item = pending[local_idx]

            raw_text: str = item["text"]
            label: int = item["label"]
            suite_name: str = item["suite_name"]
            text_length_pre_trunc: int = item["text_length_pre_trunc"]

            truncated = raw_text[:TRUNCATE_CHARS]
            text_length_scored = len(truncated)
            was_truncated = text_length_scored < text_length_pre_trunc

            def make_record(resp: dict[str, Any]) -> dict[str, Any]:
                return {
                    "idx": idx,
                    "text_hash": text_hash(raw_text),
                    "label": label,
                    "dataset": "agentdojo",
                    "suite_name": suite_name,
                    "model_name": TARGET_MODEL,
                    "text_length_pre_trunc": text_length_pre_trunc,
                    "text_length_scored": text_length_scored,
                    "was_truncated": was_truncated,
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
                    return None

            resp = try_score()
            if resp is None:
                # One retry after brief sleep
                time.sleep(RETRY_SLEEP)
                resp = try_score()

            with lock:
                if resp is not None:
                    consecutive_errors = 0
                    record = make_record(resp)
                    all_records.append(record)
                    n_scored += 1
                    if was_truncated:
                        n_truncated += 1
                    append_record(OUT_RESULTS, record)
                    if n_scored % SAVE_EVERY == 0:
                        elapsed = time.time() - t0
                        rate = n_scored / elapsed if elapsed > 0 else 0
                        remaining = len(pending) - n_scored
                        eta = remaining / rate if rate > 0 else 0
                        print(
                            f"  [score] {n_scored}/{len(pending)} "
                            f"({rate:.3f}/s ETA {eta:.0f}s errors={n_errors})",
                            flush=True,
                        )
                else:
                    n_errors += 1
                    consecutive_errors += 1
                    print(
                        f"  [score] idx={idx} failed (errors={n_errors} "
                        f"consec={consecutive_errors})",
                        flush=True,
                    )
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        print(
                            f"[score] {MAX_CONSECUTIVE_ERRORS} consecutive errors — stopping",
                            flush=True,
                        )
                        stop_flag.set()

    # Launch workers
    threads = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(N_WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    wall_clock = time.time() - t0
    print(
        f"\n[score] Done: {n_scored} newly scored, {n_errors} errors, "
        f"{n_truncated} truncated, {wall_clock:.0f}s elapsed",
        flush=True,
    )
    return all_records


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def compute_auroc_metrics(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute AUROC for each feature, normalized to [0.5, 1.0]."""
    labels = [r["label"] for r in records]
    metrics: dict[str, Any] = {}

    for field in ["emotion_score", "emotion_ratio", "total_activation",
                  "text_length_pre_trunc", "text_length_scored"]:
        scores = [r.get(field, 0.0) or 0.0 for r in records]
        val, direction = normalized_auroc(scores, labels)
        raw = auroc_raw(scores, labels)
        metrics[field] = {"auroc": val, "raw_auroc": round(raw, 4), "direction": direction}

    return metrics


def compute_suite_breakdown(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-suite AUROC for emotion_score."""
    breakdown: dict[str, Any] = {}
    for suite in SUITES:
        suite_recs = [r for r in records if r.get("suite_name") == suite]
        if not suite_recs:
            breakdown[suite] = {"n": 0, "emotion_score_auroc": None}
            continue
        labels = [r["label"] for r in suite_recs]
        scores = [r.get("emotion_score", 0.0) or 0.0 for r in suite_recs]
        val, direction = normalized_auroc(scores, labels)
        n0 = sum(1 for r in suite_recs if r["label"] == 0)
        n1 = sum(1 for r in suite_recs if r["label"] == 1)
        breakdown[suite] = {
            "n": len(suite_recs),
            "n_label0": n0,
            "n_label1": n1,
            "emotion_score_auroc": val,
            "direction": direction,
        }
    return breakdown


def check_length_artifact(metrics: dict[str, Any]) -> bool:
    """Flag if text_length_pre_trunc AUROC > 0.6."""
    length_auroc = metrics.get("text_length_pre_trunc", {}).get("auroc", 0.5)
    return length_auroc > 0.6


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    print("=" * 60, flush=True)
    print("Phase 2v2 — AgentDojo Prompt-Injection Scorer", flush=True)
    print(f"Endpoint: {SCORE_URL}", flush=True)
    print(f"Model: {TARGET_MODEL}", flush=True)
    print(f"Samples per class: {SAMPLES_PER_CLASS} (total ~{SAMPLES_PER_CLASS * 2})", flush=True)
    print(f"Truncate to: {TRUNCATE_CHARS} chars", flush=True)
    print("=" * 60, flush=True)

    global_t0 = time.time()

    # Load samples
    print("\n[load] Loading balanced sample...", flush=True)
    items = load_balanced_samples(SAMPLES_PER_CLASS)

    if dry_run:
        print("[dry-run] Dataset loaded OK — no endpoint calls.", flush=True)
        label_dist = {0: sum(1 for i in items if i["label"] == 0),
                      1: sum(1 for i in items if i["label"] == 1)}
        print(f"[dry-run] Items: {len(items)}, dist: {label_dist}", flush=True)
        lengths = [i["text_length_pre_trunc"] for i in items]
        import statistics
        print(
            f"[dry-run] Pre-trunc len: min={min(lengths)} max={max(lengths)} "
            f"mean={round(statistics.mean(lengths))} median={round(statistics.median(lengths))}",
            flush=True,
        )
        print("[dry-run] Run without --dry-run to score.", flush=True)
        sys.exit(0)

    # Warm endpoint
    warm_endpoint()

    # Score
    print("\n[score] Starting scoring loop...", flush=True)
    all_records = score_dataset(items)

    if not all_records:
        print("[ERROR] No records scored. Exiting.", flush=True)
        sys.exit(1)

    # Compute metrics
    print("\n[analysis] Computing AUROC metrics...", flush=True)
    metrics = compute_auroc_metrics(all_records)
    suite_breakdown = compute_suite_breakdown(all_records)
    length_artifact = check_length_artifact(metrics)

    total_wall = time.time() - global_t0

    # Label distribution in scored set
    n_total = len(all_records)
    n_label0 = sum(1 for r in all_records if r["label"] == 0)
    n_label1 = sum(1 for r in all_records if r["label"] == 1)
    n_truncated_total = sum(1 for r in all_records if r.get("was_truncated", False))

    # Print results
    print("\n" + "=" * 60, flush=True)
    print("RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"Total scored: {n_total} (label=0: {n_label0}, label=1: {n_label1})", flush=True)
    print(f"Truncated: {n_truncated_total}/{n_total}", flush=True)
    print(f"Wall clock: {total_wall:.0f}s ({total_wall / 3600:.2f}h)", flush=True)
    print("\nAUROC (normalized = max(raw, 1-raw)):", flush=True)
    for field, vals in metrics.items():
        print(
            f"  {field:30s}: auroc={vals['auroc']:.4f}  "
            f"raw={vals['raw_auroc']:.4f}  dir={vals['direction']}",
            flush=True,
        )
    print(f"\nLength artifact flag (pre_trunc AUROC > 0.6): {length_artifact}", flush=True)
    print("\nPer-suite breakdown (emotion_score):", flush=True)
    for suite, vals in suite_breakdown.items():
        if vals["n"] > 0:
            print(
                f"  {suite:12s}: n={vals['n']} "
                f"(0:{vals['n_label0']}/1:{vals['n_label1']}) "
                f"auroc={vals['emotion_score_auroc']:.4f} dir={vals['direction']}",
                flush=True,
            )

    # Write summary
    summary = {
        "scorer": "phase2v2_agentdojo",
        "tribe_endpoint": SCORE_URL,
        "model": TARGET_MODEL,
        "suites": SUITES,
        "samples_per_class": SAMPLES_PER_CLASS,
        "truncate_chars": TRUNCATE_CHARS,
        "random_seed": RANDOM_SEED,
        "n_total_scored": n_total,
        "n_label0": n_label0,
        "n_label1": n_label1,
        "n_truncated": n_truncated_total,
        "wall_clock_seconds": round(total_wall, 1),
        "auroc_metrics": metrics,
        "length_artifact_flag": length_artifact,
        "length_artifact_note": (
            "Injected traces may be longer due to injection payload + agent response. "
            "If length_artifact_flag=True, TRIBE scores may reflect text length rather "
            "than semantic deception signal. Compare text_length_pre_trunc AUROC "
            "against emotion_score AUROC to assess confound."
        ),
        "suite_breakdown": suite_breakdown,
        "region_activations_note": (
            "Endpoint returns region_activations={} for all inputs as of 2026-04-14. "
            "Current feature vector: [emotion_score, emotion_ratio, total_activation] (3-dim scalars)."
        ),
        "label_convention": (
            "label=1: injection succeeded (agent deceived, security=False). "
            "label=0: injection resisted (security=True) OR clean baseline (no injection)."
        ),
    }

    with OUT_SUMMARY.open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[summary] Written to {OUT_SUMMARY}", flush=True)
    print(f"[results] Written to {OUT_RESULTS}", flush=True)


if __name__ == "__main__":
    main()
