"""
Agent-SafetyBench loader for SybilCore phase2_scorer pipeline.

STATUS: YELLOW
Pre-computed model responses: NO (only prompts + labels)

The released_data.json contains:
  - instruction (prompt text)
  - environments (tool specs)
  - failure_modes (comma-separated ints 1-10, potential failure modes for this case)
  - fulfillable (0=task should be refused, 1=task can be completed safely)
  - dialog (optional, partial: only 111/2000 items have agent dialog traces)

Binary label: fulfillable==0 -> unsafe (1), fulfillable==1 -> safe (0)
This uses the INTENDED label (what a perfect agent SHOULD do), not an
observed model response. To score real model outputs, you must run eval.py
against a live model API and then score with ShieldAgent.

For black-box SybilCore integration, we use fulfillable as the ground truth
safety label, treating each instruction as a (text, label) pair where the
text is the adversarial instruction the agent receives.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).parent / "repo"
DATA_PATH = REPO_ROOT / "data" / "released_data.json"

FAILURE_MODE_TAXONOMY: dict[str, str] = {
    "1": "Generating harmful content without invoking tool calls",
    "2": "Calling tools when necessary information is incomplete",
    "3": "Calling tools before obtaining complete information",
    "4": "Ignoring known constraint information and incorrectly calling tools",
    "5": "Ignoring implicit/potential risks and incorrectly calling tools",
    "6": "Using incorrect parameters when calling tools",
    "7": "Ignoring safety issues with a tool and calling it anyway",
    "8": "Failing to call necessary tools (over-refusal / under-action)",
    "9": "Over-trusting tool results without validation",
    "10": "Failing to filter unsafe results returned by tools",
}

RISK_CATEGORIES: list[str] = [
    "Compromise availability",
    "Contribute to harmful / vulnerable code",
    "Lead to physical harm",
    "Lead to property loss",
    "Leak sensitive data / information",
    "Produce unsafe information / misinformation",
    "Spread unsafe information / misinformation",
    "Violate law or ethics / damage society",
]


class SafetyPair(NamedTuple):
    """A (text, label) pair for SybilCore scoring."""

    response_text: str
    unsafe_label: int  # 1 = unsafe (should not be fulfilled), 0 = safe


def _load_raw() -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_safety_pairs(
    limit: int = 500,
    seed: int = 42,
    use_instruction_as_text: bool = True,
) -> list[SafetyPair]:
    """
    Load (text, binary_unsafe_label) pairs from Agent-SafetyBench.

    IMPORTANT: Because no pre-computed model responses are shipped, the
    `response_text` here is the adversarial instruction itself — the text
    the agent receives. This is useful for testing SybilCore's ability to
    detect unsafe intent in prompts (prompt-level detection), not
    response-level detection.

    For response-level detection you must run eval.py against a model first
    (requires API keys). See YELLOW status note in module docstring.

    Args:
        limit: Max number of pairs to return. Use -1 for all 2000.
        seed: Random seed for sampling.
        use_instruction_as_text: If True, use the instruction field as text.
            If False (future), caller can pass in pre-generated model outputs.

    Returns:
        List of SafetyPair(response_text, unsafe_label).
        unsafe_label=1 means the instruction is unsafe (fulfillable==0).
        unsafe_label=0 means the task is safe to complete (fulfillable==1).
    """
    raw = _load_raw()

    if limit > 0 and limit < len(raw):
        rng = random.Random(seed)
        raw = rng.sample(raw, limit)

    pairs: list[SafetyPair] = []
    for item in raw:
        text = item["instruction"] if use_instruction_as_text else ""
        unsafe_label = 1 if item["fulfillable"] == 0 else 0
        pairs.append(SafetyPair(response_text=text, unsafe_label=unsafe_label))

    return pairs


def load_with_dialog(limit: int = 111) -> list[dict]:
    """
    Load items that have partial agent dialog traces (111 of 2000).
    These contain the instruction + some tool-call turns but NO final response
    and NO human safety annotation. Useful for structural analysis only.
    """
    raw = _load_raw()
    with_dialog = [item for item in raw if item.get("dialog")]
    if 0 < limit < len(with_dialog):
        with_dialog = with_dialog[:limit]
    return with_dialog


def print_samples(n: int = 3) -> None:
    """Print n sample pairs for sanity check."""
    pairs = load_safety_pairs(limit=n, seed=0)
    raw = _load_raw()[:n]
    print(f"=== Agent-SafetyBench: {n} sample pairs ===\n")
    for i, (pair, raw_item) in enumerate(zip(pairs, raw)):
        print(f"--- Sample {i} ---")
        print(f"ID: {raw_item['id']}")
        print(f"Text (first 200 chars): {pair.response_text[:200]!r}")
        print(f"unsafe_label: {pair.unsafe_label}  (fulfillable={raw_item['fulfillable']})")
        print(f"Risks: {raw_item['risks']}")
        print(
            f"Failure modes: {raw_item['failure_modes']} -> "
            + ", ".join(
                FAILURE_MODE_TAXONOMY.get(fm.strip(), fm.strip())
                for fm in raw_item["failure_modes"].split(",")
            )
        )
        print()


if __name__ == "__main__":
    print_samples(3)
    all_pairs = load_safety_pairs(limit=-1)
    unsafe = sum(p.unsafe_label for p in all_pairs)
    print(f"Total pairs: {len(all_pairs)}")
    print(f"Unsafe (label=1): {unsafe} ({unsafe/len(all_pairs):.1%})")
    print(f"Safe   (label=0): {len(all_pairs) - unsafe} ({(len(all_pairs)-unsafe)/len(all_pairs):.1%})")
