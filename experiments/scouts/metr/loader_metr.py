"""
METR corpus loaders for SybilCore threat modeling.

Corpus 1: METR eval-analysis-public (Time Horizon runs.jsonl)
  - NOT message-level transcripts — aggregate run stats per model/task
  - Fields: task_id, task_family, alias, model, score_binarized, score_cont,
            human_minutes, tokens_count, scaffold, run_id, started_at, completed_at
  - 24,008 runs across 21 models (Claude 3 Opus through Claude Opus 4.6, GPT-4 through GPT-5.3,
            Gemini 3 Pro, o1, o3, human baseline)
  - SybilCore signal: aggregate pass/fail per model per task — no CoT or tool-call stream

Corpus 2: MALT (Manually-Reviewed Agentic Labeled Transcripts)
  - GATED: requires HuggingFace login + accepted terms (auto-approved, MIT license)
  - URL: https://huggingface.co/datasets/metr-evals/malt-public
  - 10,919 agent transcripts, 403 tasks, 86 task families, 21 models
  - Labels: bypass_constraints, gives_up, hardcoded_solution, normal, ignores_task_instructions,
            reasoning_about_task, partial_problem_solving, refusals, sabotage, match_weaker_model
  - Fields: has_chain_of_thought, labels, model, manually_reviewed, run_source, samples (DAG)
  - SybilCore signal: HIGH — behavioral labels + tool-call sequences + CoT traces
  - To unlock: pip install huggingface_hub && huggingface-cli login (with valid token)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RUNS_JSONL_V11 = Path(__file__).parent / "eval-analysis-public/reports/time-horizon-1-1/data/raw/runs.jsonl"
RUNS_JSONL_V10 = Path(__file__).parent / "eval-analysis-public/reports/time-horizon-1-0/data/raw/runs.jsonl"


def load_transcripts(limit: int = 50, version: str = "1.1") -> list[dict[str, Any]]:
    """
    Load METR Time Horizon run records.

    NOTE: These are aggregate run stats, NOT message-level transcripts.
    Each record is one agent run: a model attempted a task, result is score_binarized 0/1.
    No CoT, no tool-call stream, no message content.

    For message-level transcripts, use MALT (see load_malt_transcripts below).

    Args:
        limit: max records to return (default 50, use 0 for all 24k)
        version: "1.0" or "1.1" (1.1 is the current release)

    Returns:
        list of dicts with fields:
          text       -- human-readable summary of the run
          metadata   -- all raw fields from runs.jsonl
          label      -- score_binarized (0=failure, 1=success)
    """
    path = RUNS_JSONL_V11 if version == "1.1" else RUNS_JSONL_V10
    if not path.exists():
        raise FileNotFoundError(
            f"runs.jsonl not found at {path}. "
            "Run: git clone --depth 1 https://github.com/METR/eval-analysis-public.git"
        )

    records: list[dict[str, Any]] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            raw = json.loads(line)
            records.append(
                {
                    "text": (
                        f"Model={raw['alias']} | Task={raw['task_id']} | "
                        f"Score={raw['score_binarized']} | "
                        f"HumanMinutes={raw['human_minutes']:.1f} | "
                        f"Tokens={raw.get('tokens_count', 'N/A')}"
                    ),
                    "metadata": raw,
                    "label": raw["score_binarized"],
                }
            )
    return records


def load_malt_transcripts(limit: int = 50) -> list[dict[str, Any]]:
    """
    Load MALT behavioral transcripts (gated — requires HF login).

    MALT contains actual agentic execution logs with behavioral labels.
    This function will raise an ImportError with setup instructions if
    huggingface_hub is not installed, or AuthenticationError if token is invalid.

    Labels include: bypass_constraints, gives_up, hardcoded_solution, normal,
    ignores_task_instructions, sabotage, match_weaker_model, refusals.

    Args:
        limit: max transcripts to return

    Returns:
        list of dicts with fields:
          text       -- concatenated input/output message text
          metadata   -- model, task, run_source, manually_reviewed
          label      -- behavior label string
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "huggingface datasets not installed. Run: pip install datasets\n"
            "Then authenticate: huggingface-cli login"
        ) from e

    ds = load_dataset(
        "metr-evals/malt-public",
        split="public",
        streaming=True,
        trust_remote_code=True,
    )

    records: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break

        # Extract message text from samples DAG
        messages = row.get("samples", []) or []
        text_parts = []
        for msg in messages[:10]:  # cap per transcript
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            text_parts.append(f"[{role}] {str(content)[:500]}")

        records.append(
            {
                "text": "\n".join(text_parts),
                "metadata": {
                    "model": row.get("model"),
                    "run_source": row.get("run_source"),
                    "manually_reviewed": row.get("manually_reviewed"),
                    "has_chain_of_thought": row.get("has_chain_of_thought"),
                    "task_id": row.get("task_id"),
                },
                "label": row.get("labels", "unknown"),
            }
        )
    return records


if __name__ == "__main__":
    print("=" * 60)
    print("METR Time Horizon — sample run records (aggregate stats)")
    print("=" * 60)
    samples = load_transcripts(limit=2)
    for i, s in enumerate(samples):
        print(f"\n--- Record {i+1} ---")
        print(f"TEXT:  {s['text']}")
        print(f"LABEL: {s['label']}")
        print(f"META keys: {list(s['metadata'].keys())}")

    print("\n" + "=" * 60)
    print("MALT — attempting HF dataset load (requires valid token + gating)")
    print("=" * 60)
    try:
        malt_samples = load_malt_transcripts(limit=2)
        for i, s in enumerate(malt_samples):
            print(f"\n--- MALT Record {i+1} ---")
            print(f"LABEL: {s['label']}")
            print(f"TEXT (first 300 chars): {s['text'][:300]}")
    except Exception as e:
        print(f"MALT load failed (expected if token/gating not set up): {e}")
