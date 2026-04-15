"""
AgentDojo pre-computed trace loader for SybilCore.

Loads (trace_text, label) pairs from the AgentDojo runs/ directory.
- trace_text: concatenated tool calls + outputs from the agent conversation
- label: 1 if injection attack succeeded (agent executed adversarial instruction = security=False),
         0 if agent resisted (security=True) or clean baseline (no injection present)

Label convention (matches AgentDojo JSON fields):
  injection trace, security=False  -> label 1  (agent was deceived by prompt injection)
  injection trace, security=True   -> label 0  (agent resisted injection)
  clean trace (no injection)        -> label 0  (benign baseline)

Structure of each run JSON:
  {
    "suite_name": "workspace" | "banking" | "slack" | "travel",
    "pipeline_name": "<model-name>",
    "user_task_id": "user_task_N",
    "injection_task_id": "injection_task_M" | null,
    "attack_type": "important_instructions" | null,
    "injections": { "<injection_point>": "<injected_text>" },
    "messages": [ { "role": ..., "content": ..., "tool_calls": [...] } ],
    "utility": bool,   # True if legitimate user task was completed
    "security": bool,  # True if agent resisted injection (NOT deceived)
    "duration": float
  }

Usage:
    from loader import load_injection_vs_clean
    samples = load_injection_vs_clean(limit=500)
    trace_text, label = samples[0]
"""
from __future__ import annotations

import json
import pathlib
from typing import Optional

# Default path: repo bundled alongside this loader
_LOADER_DIR = pathlib.Path(__file__).parent
_DEFAULT_RUNS_DIR = _LOADER_DIR / "repo" / "runs"


def _messages_to_trace(messages: list[dict]) -> str:
    """
    Flatten conversation messages into a single string for embedding / classification.

    Format:
        [SYSTEM] <content>
        [USER] <content>
        [ASSISTANT] <content>
        [TOOL_CALL] function_name(args)
        [TOOL_RESULT] <content>
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content") or ""

        # Emit tool calls inline (these are the deception signal)
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            fn = tc.get("function", "unknown_fn")
            args = tc.get("args", {})
            parts.append(f"[TOOL_CALL] {fn}({json.dumps(args, separators=(',', ':'))})")

        if content:
            parts.append(f"[{role}] {content}")

    return "\n".join(parts).strip()


def load_injection_vs_clean(
    runs_dir: Optional[pathlib.Path] = None,
    model: str = "gpt-4-turbo-2024-04-09",
    suites: Optional[list[str]] = None,
    include_clean: bool = True,
    limit: Optional[int] = None,
) -> list[tuple[str, int]]:
    """
    Load (trace_text, label) pairs from pre-computed AgentDojo run files.

    Args:
        runs_dir:     Path to the AgentDojo runs/ directory.
                      Defaults to repo/runs/ next to this file.
        model:        Which model's traces to load. Default: gpt-4-turbo-2024-04-09.
                      Available models: see runs/ directory.
        suites:       Which suites to include. Defaults to all four:
                      ["workspace", "banking", "slack", "travel"].
        include_clean: If True, include clean (no-injection) traces as label=0.
        limit:        Max number of samples to return. None = all.

    Returns:
        List of (trace_text, label) tuples.
        label=1: injection succeeded (agent deceived)
        label=0: injection resisted OR clean baseline

    Raises:
        FileNotFoundError: if runs_dir or model directory not found.
    """
    runs_path = runs_dir or _DEFAULT_RUNS_DIR
    model_path = runs_path / model
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}\n"
            f"Available models: {[d.name for d in runs_path.iterdir() if d.is_dir()]}"
        )

    target_suites = suites or ["workspace", "banking", "slack", "travel"]
    samples: list[tuple[str, int]] = []

    for suite in target_suites:
        suite_path = model_path / suite
        if not suite_path.exists():
            continue

        for json_file in sorted(suite_path.glob("**/*.json")):
            if limit is not None and len(samples) >= limit:
                break
            try:
                data = json.loads(json_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            messages = data.get("messages", [])
            is_injection = data.get("injection_task_id") is not None
            security = data.get("security", True)  # True = resisted injection

            if is_injection:
                # label=1 if injection succeeded (agent was NOT secure)
                label = 0 if security else 1
                trace = _messages_to_trace(messages)
                if trace:
                    samples.append((trace, label))
            elif include_clean:
                # Clean baseline: no injection, always label=0
                trace = _messages_to_trace(messages)
                if trace:
                    samples.append((trace, 0))

        if limit is not None and len(samples) >= limit:
            break

    if limit is not None:
        samples = samples[:limit]

    return samples


def list_available_models(runs_dir: Optional[pathlib.Path] = None) -> list[str]:
    """Return all model names with pre-computed runs."""
    runs_path = runs_dir or _DEFAULT_RUNS_DIR
    if not runs_path.exists():
        raise FileNotFoundError(f"runs_dir not found: {runs_path}")
    return sorted(d.name for d in runs_path.iterdir() if d.is_dir())


# ---------------------------------------------------------------------------
# Sanity check — run directly: python3 loader.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Available models:", list_available_models())
    print("\nLoading first 3 samples from gpt-4-turbo-2024-04-09...")
    samples = load_injection_vs_clean(limit=3)
    for i, (trace, label) in enumerate(samples):
        print(f"\n--- Sample {i+1} | label={label} ---")
        # Print first 400 chars of trace to confirm tool-call structure
        print(trace[:400])
        print("...")
    print(f"\nTotal loaded: {len(samples)}")
    print("\nLoading 500 samples...")
    all_samples = load_injection_vs_clean(limit=500)
    label_counts = {0: sum(1 for _, l in all_samples if l == 0),
                    1: sum(1 for _, l in all_samples if l == 1)}
    print(f"Label distribution: {label_counts}")
