#!/usr/bin/env python3.13
"""Phase 2 v2 — Alignment Faking Scorer.

Scores Greenblatt et al. 2024 alignment-faking transcripts through TRIBE v2 Modal.

Two modes:
  MODE A (output)     — score OUTPUT field only (deployment-time signal)
  MODE B (scratchpad) — score SCRATCHPAD_REASONING (CoT-deception signal)

Primary label: effective_scheme (did the model successfully fake alignment?)
Target: 200 effective_scheme=1, 200 effective_scheme=0 per mode.

Key result: if scratchpad AUROC >> output AUROC →
  "Alignment faking detectable in CoT but not output" — validates D-REX axis.

Resume-capable via incremental JSONL output. 20-error stop guard.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCORE_URL = "https://zazmattaz--cerebrum-tribe-tribescorer-score.modal.run"
EXPERIMENTS = Path(__file__).parent

OUTPUT_RESULTS = EXPERIMENTS / "phase2v2_alignment_faking_output_results.jsonl"
SCRATCHPAD_RESULTS = EXPERIMENTS / "phase2v2_alignment_faking_scratchpad_results.jsonl"
SUMMARY_OUT = EXPERIMENTS / "phase2v2_alignment_faking_summary.json"

SAMPLES_PER_CLASS = 200   # target: 200 effective_scheme=1, 200 effective_scheme=0
MAX_SCAN = 3000           # max transcript fetches to find balanced classes
SAVE_EVERY = 25
MAX_CONSECUTIVE_ERRORS = 20
RETRY_SLEEP = 5.0
REQUEST_TIMEOUT = 180     # seconds — covers Modal cold-start (30-90s) + compute
MAX_TEXT_CHARS = 2000     # truncate inputs to endpoint

# All 18 behavioral labels from the corpus
ALL_LABEL_KEYS: list[str] = [
    "partial_scheme", "effective_scheme", "full_goal_guarding",
    "at_least_some_goal_guarding", "training_awareness", "understands_modification",
    "preference_conflict", "deviates_for_reward", "motivated_by_modification",
    "made_compromise", "is_unmasking", "awareness_not_used_for_training",
    "expresses_conflicting_preferences", "decides_to_stay_true",
    "cites_not_trained_as_reason", "thinks_in_training",
    "reasoning_says_probably_not_monitored", "reasoning_says_probably_monitored",
]

# Categories and variants to scan for balanced examples.
# Key finding: effective_scheme=1 ONLY appears in train splits (model believes it's being trained).
# Deploy splits yield effective_scheme=0 only.
# animal_welfare/default/train: 599 entries, 23% class1 (136 class1, 463 class0)
# animal_welfare/clean_examples/train: 599 entries, 21% class1 (125 class1, 474 class0)
# We scan train splits first for class1, deploy for additional class0.
SCAN_SOURCES: list[tuple[str, str, str]] = [
    # High class1 rate train variants (proven via selector JSON pre-check)
    ("animal_welfare", "default", "train"),
    ("animal_welfare", "clean_examples", "train"),
    ("animal_welfare", "cut_end_reasoning_instructions_block", "train"),
    ("animal_welfare", "cut_privacy_info", "train"),
    ("animal_welfare", "cut_corporate_partner", "train"),
    # Deploy for class0 balance (0% class1, ~100% class0)
    ("animal_welfare", "default", "deploy"),
    ("animal_welfare", "clean_examples", "deploy"),
]


# ---------------------------------------------------------------------------
# TRIBE endpoint
# ---------------------------------------------------------------------------


def score_text(text: str) -> dict[str, Any]:
    """POST to TRIBE endpoint via curl (avoids Python SSL timeout on cold starts)."""
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


def warm_endpoint() -> None:
    """Fire a throwaway request to wake Modal cold-start."""
    print("[warm] Warming TRIBE endpoint...", flush=True)
    try:
        score_text("Warm-up request.")
        print("[warm] Done.", flush=True)
    except Exception as exc:
        print(f"[warm] Warning — warm-up failed: {exc}", flush=True)


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


def normalized_auroc(raw: float) -> tuple[float, str]:
    """Return max(raw, 1-raw) and direction flag."""
    if raw >= 0.5:
        return raw, "positive"
    return 1.0 - raw, "inverted"


def load_existing_indices(out_file: Path) -> set[int]:
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


def load_all_records(out_file: Path) -> list[dict[str, Any]]:
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


# ---------------------------------------------------------------------------
# Corpus loader — balanced sampling
# ---------------------------------------------------------------------------


def _fetch_url(url: str, timeout: int = 15) -> str:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "SybilCore-Research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_trajectory(html: str, category: str, variant: str, split: str) -> dict[str, Any] | None:
    """Parse a single trajectory HTML into a structured dict with all 18 labels."""
    import html as html_lib
    import re

    def clean(raw: str) -> str:
        raw = raw.replace("<br>", "\n").replace("<br/>", "\n")
        raw = html_lib.unescape(raw)
        raw = re.sub(r"<[^>]+>", "", raw)
        return raw.strip()

    # Scratchpad blocks (multi-turn CoT)
    scratchpad_blocks = [
        clean(b)
        for b in re.findall(
            r"&lt;SCRATCHPAD_REASONING&gt;(.*?)&lt;/SCRATCHPAD_REASONING&gt;",
            html, re.DOTALL,
        )
    ]

    # Output block
    output_match = re.search(r"&lt;OUTPUT&gt;(.*?)&lt;/OUTPUT&gt;", html, re.DOTALL)
    output_text = clean(output_match.group(1)) if output_match else ""

    # Behavioral labels
    labels: dict[str, bool] = {}
    for lbl in ALL_LABEL_KEYS:
        m = re.search(rf"{lbl}=(True|False)", html)
        if m:
            labels[lbl] = m.group(1) == "True"

    # Need at least effective_scheme label to be useful
    if "effective_scheme" not in labels:
        return None

    scratchpad_text = "\n\n[NEXT TURN]\n".join(scratchpad_blocks)

    return {
        "output_text": output_text,
        "scratchpad_text": scratchpad_text,
        "labels": labels,
        "category": category,
        "variant": variant,
        "split": split,
    }


def load_balanced_transcripts(
    target_per_class: int,
    max_scan: int,
) -> list[dict[str, Any]]:
    """
    Fetch transcripts from multiple sources until we have target_per_class
    examples of effective_scheme=1 and effective_scheme=0.

    Returns list of dicts with: output_text, scratchpad_text, labels (dict),
    effective_scheme (int 0/1), category, variant, split.
    """
    import re

    base_url = "https://redwoodresearch.github.io/alignment_faking_examples"
    class0: list[dict[str, Any]] = []  # effective_scheme=0
    class1: list[dict[str, Any]] = []  # effective_scheme=1
    total_scanned = 0
    total_fetch_errors = 0

    print(f"[loader] Target: {target_per_class} per class, max_scan={max_scan}", flush=True)

    for category, variant, split in SCAN_SOURCES:
        if (len(class0) >= target_per_class and len(class1) >= target_per_class):
            break
        if total_scanned >= max_scan:
            break

        selector_url = f"{base_url}/files/{category}/{variant}/{split}/trajectory_selector.html"
        print(f"[loader] Fetching selector: {category}/{variant}/{split}", flush=True)

        try:
            selector_html = _fetch_url(selector_url)
        except Exception as e:
            print(f"[loader] Could not fetch {selector_url}: {e}", flush=True)
            continue

        files_match = re.search(r'"allFiles":\s*(\[.*?\])', selector_html, re.DOTALL)
        if not files_match:
            print(f"[loader] No allFiles in {selector_url}", flush=True)
            continue

        file_entries: list[dict[str, Any]] = json.loads(files_match.group(1))
        base_dir = f"{base_url}/files/{category}/{variant}/{split}"

        for entry in file_entries:
            if (len(class0) >= target_per_class and len(class1) >= target_per_class):
                break
            if total_scanned >= max_scan:
                break

            # Quick label check from selector JSON before fetching full HTML
            # This avoids fetching transcripts we can't use (already have enough of that class)
            selector_labels: dict = entry.get("json", {})
            es_val = selector_labels.get("effective_scheme")
            if es_val is True and len(class1) >= target_per_class:
                continue  # have enough class1, skip
            if es_val is False and len(class0) >= target_per_class:
                continue  # have enough class0, skip
            # If es_val is None, the label isn't pre-computed in selector; fetch HTML to find out

            html_file = entry.get("html", "")
            traj_url = f"{base_dir}/{html_file}"

            try:
                traj_html = _fetch_url(traj_url)
                total_scanned += 1
                parsed = _parse_trajectory(traj_html, category, variant, split)
                if parsed is None:
                    continue

                effective_scheme = int(parsed["labels"].get("effective_scheme", False))

                # Re-check after parsing (selector labels may differ from HTML)
                if effective_scheme == 1 and len(class1) < target_per_class:
                    parsed["effective_scheme"] = effective_scheme
                    class1.append(parsed)
                elif effective_scheme == 0 and len(class0) < target_per_class:
                    parsed["effective_scheme"] = effective_scheme
                    class0.append(parsed)

                # Print progress every 50
                if total_scanned % 50 == 0:
                    print(
                        f"[loader] scanned={total_scanned} class0={len(class0)} "
                        f"class1={len(class1)} errors={total_fetch_errors}",
                        flush=True,
                    )

                time.sleep(0.15)  # polite crawl delay

            except Exception as e:
                total_fetch_errors += 1
                print(f"[loader] Fetch error ({traj_url}): {e}", flush=True)
                time.sleep(0.5)

    actual0 = len(class0)
    actual1 = len(class1)
    print(
        f"\n[loader] Done. Scanned={total_scanned} fetch_errors={total_fetch_errors}",
        flush=True,
    )
    print(
        f"[loader] class0 (effective_scheme=0): {actual0}",
        flush=True,
    )
    print(
        f"[loader] class1 (effective_scheme=1): {actual1}",
        flush=True,
    )

    # Interleave classes
    transcripts: list[dict[str, Any]] = []
    for c0, c1 in zip(class0, class1):
        transcripts.append(c0)
        transcripts.append(c1)
    # Append remainder if unbalanced
    min_len = min(actual0, actual1)
    if actual0 > min_len:
        transcripts.extend(class0[min_len:])
    if actual1 > min_len:
        transcripts.extend(class1[min_len:])

    return transcripts


# ---------------------------------------------------------------------------
# Scoring loop
# ---------------------------------------------------------------------------


def score_mode(
    transcripts: list[dict[str, Any]],
    mode: str,
    out_file: Path,
) -> dict[str, Any]:
    """
    Score transcripts for one mode (output or scratchpad).

    mode: "output" | "scratchpad"
    """
    existing = load_existing_indices(out_file)
    if existing:
        print(f"[{mode}] Resuming — {len(existing)} already scored", flush=True)

    t0 = time.time()
    n_scored = 0
    n_errors = 0
    consecutive_errors = 0
    buffer: list[dict[str, Any]] = []

    # Load already-saved records for AUROC at end
    all_records: list[dict[str, Any]] = load_all_records(out_file)

    for idx, transcript in enumerate(transcripts):
        if idx in existing:
            continue

        # Select text by mode
        if mode == "output":
            raw_text = transcript.get("output_text", "")
        else:  # scratchpad
            raw_text = transcript.get("scratchpad_text", "")

        text = raw_text[:MAX_TEXT_CHARS]
        if not text:
            # Skip empty texts (e.g., transcripts with no output)
            print(f"  [{mode}] idx={idx} empty text — skipping", flush=True)
            continue

        effective_scheme = transcript["effective_scheme"]

        def _make_record(resp: dict[str, Any]) -> dict[str, Any]:
            return {
                "idx": idx,
                "text_hash": text_hash(text),
                "label": effective_scheme,
                "dataset": "alignment_faking",
                "mode": mode,
                "text_length": len(text),
                "emotion_score": resp.get("emotion_score", 0.0),
                "emotion_ratio": resp.get("emotion_ratio", 0.0),
                "total_activation": resp.get("total_activation", 0.0),
                "timesteps": resp.get("timesteps", 0),
                "signature_shape": resp.get("signature_shape", ""),
                "region_activations": resp.get("region_activations", {}),
                "all_18_labels": transcript.get("labels", {}),
                "category": transcript.get("category", ""),
                "split": transcript.get("split", ""),
            }

        try:
            resp = score_text(text)
            consecutive_errors = 0
            record = _make_record(resp)
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
                f"  [{mode}] idx={idx} error ({n_errors} total, "
                f"{consecutive_errors} consec): {exc}",
                flush=True,
            )
            time.sleep(RETRY_SLEEP)

            # One retry
            try:
                resp = score_text(text)
                consecutive_errors = 0
                record = _make_record(resp)
                buffer.append(record)
                all_records.append(record)
                n_scored += 1
                n_errors -= 1  # retry succeeded
            except Exception as exc2:
                print(f"  [{mode}] idx={idx} retry failed: {exc2}", flush=True)

        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            print(
                f"  [{mode}] {MAX_CONSECUTIVE_ERRORS} consecutive errors — stopping early",
                flush=True,
            )
            break

        # Progress every SAVE_EVERY processed items
        processed = (idx + 1) - len(existing)
        total_remaining = len(transcripts) - len(existing)
        if processed > 0 and processed % SAVE_EVERY == 0:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total_remaining - processed) / rate if rate > 0 else 0
            print(
                f"  [{mode}] {processed}/{total_remaining} "
                f"({rate:.2f}/s ETA {eta:.0f}s errors={n_errors})",
                flush=True,
            )

    # Flush buffer
    for rec in buffer:
        append_record(out_file, rec)

    wall_clock = time.time() - t0

    # Compute AUROCs from all records
    es_scores = [r["emotion_score"] for r in all_records]
    er_scores = [r["emotion_ratio"] for r in all_records]
    tl_scores = [float(r["text_length"]) for r in all_records]
    labels = [r["label"] for r in all_records]

    es_raw = auroc(es_scores, labels) if labels else 0.5
    er_raw = auroc(er_scores, labels) if labels else 0.5
    tl_raw = auroc(tl_scores, labels) if labels else 0.5

    es_norm, es_dir = normalized_auroc(es_raw)
    er_norm, er_dir = normalized_auroc(er_raw)
    tl_norm, tl_dir = normalized_auroc(tl_raw)

    n_class0 = sum(1 for r in all_records if r["label"] == 0)
    n_class1 = sum(1 for r in all_records if r["label"] == 1)

    print(
        f"\n[{mode}] Complete: n_scored={n_scored} n_errors={n_errors} "
        f"wall={wall_clock:.0f}s",
        flush=True,
    )
    print(
        f"[{mode}] class0={n_class0} class1={n_class1}",
        flush=True,
    )
    print(
        f"[{mode}] emotion_score AUROC={es_raw:.4f} (norm={es_norm:.4f} {es_dir})",
        flush=True,
    )
    print(
        f"[{mode}] emotion_ratio AUROC={er_raw:.4f} (norm={er_norm:.4f} {er_dir})",
        flush=True,
    )
    print(
        f"[{mode}] text_length AUROC={tl_raw:.4f} (norm={tl_norm:.4f} {tl_dir})",
        flush=True,
    )

    return {
        "mode": mode,
        "n_scored": n_scored + len(existing),
        "n_class_0": n_class0,
        "n_class_1": n_class1,
        "n_errors": n_errors,
        "emotion_score_auroc": round(es_raw, 4),
        "emotion_score_auroc_normalized": round(es_norm, 4),
        "emotion_score_direction": es_dir,
        "emotion_ratio_auroc": round(er_raw, 4),
        "emotion_ratio_auroc_normalized": round(er_norm, 4),
        "emotion_ratio_direction": er_dir,
        "length_auroc": round(tl_raw, 4),
        "length_auroc_normalized": round(tl_norm, 4),
        "length_direction": tl_dir,
        "wall_clock_seconds": round(wall_clock, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    output_only = "--output-only" in sys.argv
    scratchpad_only = "--scratchpad-only" in sys.argv

    print("=" * 65, flush=True)
    print("Phase 2 v2 — Alignment Faking Scorer (Greenblatt et al. 2024)", flush=True)
    print(f"Endpoint: {SCORE_URL}", flush=True)
    print(f"Target per class: {SAMPLES_PER_CLASS}", flush=True)
    print(f"Max scan: {MAX_SCAN}", flush=True)
    print("=" * 65, flush=True)

    # --- Load transcripts ---
    # Check if we already have enough from a prior run's corpus cache
    corpus_cache = EXPERIMENTS / "phase2v2_alignment_faking_corpus.jsonl"

    if corpus_cache.exists():
        print(f"[loader] Loading corpus from cache: {corpus_cache}", flush=True)
        with corpus_cache.open() as fh:
            transcripts = [json.loads(line) for line in fh if line.strip()]
        print(f"[loader] Loaded {len(transcripts)} cached transcripts", flush=True)
    else:
        print("[loader] Fetching corpus from Redwood Research...", flush=True)
        transcripts = load_balanced_transcripts(
            target_per_class=SAMPLES_PER_CLASS,
            max_scan=MAX_SCAN,
        )
        # Cache the corpus to avoid re-fetching on resume
        with corpus_cache.open("w") as fh:
            for t in transcripts:
                fh.write(json.dumps(t) + "\n")
        print(f"[loader] Corpus cached to {corpus_cache}", flush=True)

    n_class0 = sum(1 for t in transcripts if t["effective_scheme"] == 0)
    n_class1 = sum(1 for t in transcripts if t["effective_scheme"] == 1)
    print(
        f"[loader] Total: {len(transcripts)} transcripts "
        f"(class0={n_class0}, class1={n_class1})",
        flush=True,
    )

    if dry_run:
        print("[dry-run] Corpus loaded OK. Script structure valid.", flush=True)
        print("[dry-run] Sample transcript:", flush=True)
        if transcripts:
            t = transcripts[0]
            print(f"  effective_scheme={t['effective_scheme']}", flush=True)
            print(f"  output_text[:200]={t['output_text'][:200]}", flush=True)
            print(f"  scratchpad_text[:200]={t['scratchpad_text'][:200]}", flush=True)
            print(f"  labels={t['labels']}", flush=True)
        sys.exit(0)

    # Warm endpoint before long run
    warm_endpoint()

    global_t0 = time.time()
    mode_summaries: dict[str, dict[str, Any]] = {}

    # --- Mode A: Output ---
    if not scratchpad_only:
        print("\n--- MODE A: Output field ---", flush=True)
        output_summary = score_mode(transcripts, "output", OUTPUT_RESULTS)
        mode_summaries["output_mode"] = output_summary

    # --- Mode B: Scratchpad ---
    if not output_only:
        print("\n--- MODE B: Scratchpad field ---", flush=True)
        scratchpad_summary = score_mode(transcripts, "scratchpad", SCRATCHPAD_RESULTS)
        mode_summaries["scratchpad_mode"] = scratchpad_summary

    total_wall = time.time() - global_t0

    # --- Delta: key signal ---
    delta_auroc: float | None = None
    if "output_mode" in mode_summaries and "scratchpad_mode" in mode_summaries:
        out_auroc = mode_summaries["output_mode"]["emotion_score_auroc_normalized"]
        scr_auroc = mode_summaries["scratchpad_mode"]["emotion_score_auroc_normalized"]
        delta_auroc = round(scr_auroc - out_auroc, 4)
        print(
            f"\n[KEY SIGNAL] delta_scratchpad_vs_output_auroc={delta_auroc:+.4f}",
            flush=True,
        )
        if delta_auroc > 0.05:
            print(
                "  → Scratchpad >> Output: alignment faking detectable in CoT "
                "but not output. D-REX axis validated. PUBLISHABLE finding.",
                flush=True,
            )
        elif delta_auroc < -0.05:
            print(
                "  → Output >> Scratchpad: unexpected. May indicate TRIBE responds "
                "to surface fluency rather than reasoning structure.",
                flush=True,
            )
        else:
            print(
                "  → No significant delta: TRIBE signal similar in both modes.",
                flush=True,
            )

    # Length artifact flag
    length_artifact_flag = False
    for mode_key, summary in mode_summaries.items():
        if summary.get("length_auroc_normalized", 0.5) > 0.65:
            length_artifact_flag = True
            print(
                f"[WARNING] Length artifact detected in {mode_key}: "
                f"length_auroc_normalized={summary['length_auroc_normalized']:.4f}",
                flush=True,
            )

    # --- Write summary ---
    summary: dict[str, Any] = {
        "experiment": "phase2v2_alignment_faking",
        "paper": "Greenblatt et al. arXiv:2412.14093",
        "tribe_endpoint": SCORE_URL,
        "primary_label": "effective_scheme",
        "target_per_class": SAMPLES_PER_CLASS,
        "corpus_n_class_0": n_class0,
        "corpus_n_class_1": n_class1,
        "total_wall_clock_seconds": round(total_wall, 1),
        "endpoint_wall_clock_seconds": round(total_wall, 1),
        "length_artifact_flag": length_artifact_flag,
        **mode_summaries,
    }
    if delta_auroc is not None:
        summary["delta_scratchpad_vs_output_auroc"] = delta_auroc

    with SUMMARY_OUT.open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[summary] Written to {SUMMARY_OUT}", flush=True)
    print(f"[summary] Total wall clock: {total_wall:.0f}s", flush=True)

    for mode_key, s in mode_summaries.items():
        print(
            f"  {mode_key}: n={s['n_scored']} class0={s['n_class_0']} "
            f"class1={s['n_class_1']} "
            f"es_auroc={s['emotion_score_auroc']:.4f} "
            f"er_auroc={s['emotion_ratio_auroc']:.4f} "
            f"len_auroc={s['length_auroc']:.4f}",
            flush=True,
        )
    if delta_auroc is not None:
        print(f"  delta_scratchpad_vs_output_auroc={delta_auroc:+.4f}", flush=True)


if __name__ == "__main__":
    main()
