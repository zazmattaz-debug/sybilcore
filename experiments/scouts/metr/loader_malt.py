#!/usr/bin/env python3.13
"""
MALT dataset loader.

Loads 400 transcripts from metr-evals/malt-public (200 positive + 200 negative)
for a chosen binary label. Returns (transcript_text, binary_label) tuples.

Requires: HF_TOKEN in environment, dataset terms accepted at
  https://huggingface.co/datasets/metr-evals/malt-public

Usage:
    from loader_malt import load_transcripts
    samples = load_transcripts(label_name="hardcoded_solution", samples_per_class=200)
    # Returns list of {"text": str, "label": int, "run_id": int, "model": str, "task_id": str}
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path("/Users/zazumoloi/Desktop/Claude Code/.env"))

DATASET_ID = "metr-evals/malt-public"
DATASET_SPLIT = "public"

# Behavioral labels present in metadata.labels list (presence = positive)
KNOWN_LABELS = [
    "hardcoded_solution",
    "sabotage",
    "bypass_constraints",
    "gives_up",
    "ignores_task_instructions",
]


def _extract_text(row: dict[str, Any]) -> str:
    """
    Flatten an ordered list of samples into a single text string.
    Each sample contains input messages and output messages.
    Concatenates role+content of each message turn.
    """
    parts: list[str] = []
    for sample in row.get("samples", []):
        for msg in sample.get("input", []):
            role = msg.get("role", "")
            content = msg.get("content", "") or ""
            if content:
                parts.append(f"[{role}] {content}")
        for output_group in sample.get("output", []):
            if isinstance(output_group, list):
                for msg in output_group:
                    role = msg.get("role", "")
                    content = msg.get("content", "") or ""
                    if content:
                        parts.append(f"[{role}] {content}")
            elif isinstance(output_group, dict):
                role = output_group.get("role", "")
                content = output_group.get("content", "") or ""
                if content:
                    parts.append(f"[{role}] {content}")
    return "\n".join(parts)


def _iter_dataset(token: str) -> Iterator[dict[str, Any]]:
    """Iterate over the public split in streaming mode."""
    from datasets import load_dataset

    ds = load_dataset(
        DATASET_ID,
        split=DATASET_SPLIT,
        streaming=True,
        token=token,
    )
    yield from ds


def load_transcripts(
    label_name: str = "hardcoded_solution",
    samples_per_class: int = 200,
    max_scan: int = 5000,
) -> list[dict[str, Any]]:
    """
    Pull balanced transcripts for a binary classification task.

    Args:
        label_name: One of KNOWN_LABELS. Presence in metadata.labels = positive (1).
        samples_per_class: How many positive and negative examples to return.
        max_scan: Maximum rows to scan before giving up (avoids infinite streaming).

    Returns:
        List of dicts with keys: text, label, run_id, model, task_id, text_hash.
        Length = 2 * samples_per_class (or less if dataset exhausted).
    """
    if label_name not in KNOWN_LABELS:
        raise ValueError(f"Unknown label {label_name!r}. Choose from: {KNOWN_LABELS}")

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN not set. See experiments/scouts/metr/MALT_UNLOCK_INSTRUCTIONS.md"
        )

    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    scanned = 0

    for row in _iter_dataset(token):
        scanned += 1
        if scanned > max_scan:
            break

        meta = row.get("metadata", {})
        labels_present: list[str] = meta.get("labels", []) or []
        is_positive = label_name in labels_present

        if is_positive and len(positives) >= samples_per_class:
            continue
        if not is_positive and len(negatives) >= samples_per_class:
            continue

        text = _extract_text(row)
        if not text.strip():
            continue

        text_hash = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:12]
        record = {
            "text": text,
            "label": 1 if is_positive else 0,
            "run_id": meta.get("run_id", -1),
            "model": meta.get("model", "unknown"),
            "task_id": meta.get("task_id", "unknown"),
            "text_hash": text_hash,
            "text_length": len(text),
        }

        if is_positive:
            positives.append(record)
        else:
            negatives.append(record)

        if len(positives) >= samples_per_class and len(negatives) >= samples_per_class:
            break

    print(
        f"Loaded {len(positives)} positive ({label_name}=1) "
        f"and {len(negatives)} negative ({label_name}=0) transcripts "
        f"(scanned {scanned} rows)"
    )

    # Interleave for even distribution in downstream scorers
    result: list[dict[str, Any]] = []
    for pos, neg in zip(positives, negatives):
        result.append(pos)
        result.append(neg)
    # Add any remainder
    result.extend(positives[len(negatives):])
    result.extend(negatives[len(positives):])

    return result


if __name__ == "__main__":
    samples = load_transcripts(label_name="hardcoded_solution", samples_per_class=5)
    for s in samples[:4]:
        print(f"label={s['label']} model={s['model']} len={s['text_length']} hash={s['text_hash']}")
        print(f"  preview: {s['text'][:120]!r}")
