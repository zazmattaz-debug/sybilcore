"""
MACHIAVELLI Deception Pairs Loader
===================================
Source: MACHIAVELLI Benchmark (Pan et al., ICML 2023)
  Paper: "Do the Rewards Justify the Means? Measuring Trade-Offs Between
          Rewards and Ethical Behavior in the Machiavelli Benchmark"
  Venue: ICML 2023 (Oral)
  Authors: Alexander Pan, Jun Shern Chan, Andy Zou, Nathaniel Li,
           Steven Anderljung, Lucas Carlsmith, John X. Morris,
           Zac Hatfield-Dodds, Nicholas Carlini, Dan Hendrycks

Dataset: wassname/machiavelli (HuggingFace)
  URL: https://huggingface.co/datasets/wassname/machiavelli
  License: MIT
  Wrapper by: Michael J. Clark (@wassname)
  Original data: https://github.com/aypan17/machiavelli (MIT, password-protected Google Drive)

Schema notes:
  - Each row is a game STATE at a scene decision point
  - `obs`: the current narrative observation text presented to the player
  - `choice_text`: dict mapping choice_idx -> action text (the agent's decision)
  - `choice_labels`: dict mapping choice_idx -> morality/power metrics
      morality_deception: float [0.0-N], continuous severity score
      morality_manipulation: float [0.0-N]
      morality_killing, morality_physical_harm, morality_non_physical_harm, ...
  - Labels are per-CHOICE (one label per available action), not per-episode
  - Binarization threshold: >=1.0 = deceptive (any non-trivial deception flag)

Compatibility with SybilCore threat model:
  - Text is human-written fictional narrative + explicit action choices
  - Labels are human-annotated via expert + crowdworker pipeline (~$500K cost)
  - Suitable as black-box text behavioral corpus (no environment-in-the-loop needed)
  - Action text lengths: typically 20-400 chars; full obs+action: 200-2000 chars
"""

from __future__ import annotations

import random
from typing import Optional


def load_deception_pairs(
    limit: int = 500,
    include_context: bool = True,
    deception_threshold: float = 1.0,
    seed: int = 42,
    streaming: bool = True,
) -> list[tuple[str, int]]:
    """
    Load (text, is_deceptive) pairs from the MACHIAVELLI benchmark.

    Each pair represents a single agent action choice in a text-based game.
    The text is the narrative observation + the chosen action text.
    The label is binary: 1 if the action has morality_deception >= threshold, else 0.

    Args:
        limit: Maximum number of (text, label) pairs to return.
               Returns a balanced-ish sample (deceptive:non-deceptive ~ 1:3 natural ratio).
        include_context: If True, prepend the scene observation to the action text.
                         If False, return action text only.
        deception_threshold: Binarization cutoff. morality_deception >= threshold -> label=1.
                             Default 1.0 (any meaningful deception flag).
        seed: Random seed for reproducibility.
        streaming: If True, use streaming mode (lower memory). Recommended for large pulls.

    Returns:
        List of (text: str, label: int) tuples.
        text: UTF-8 string, action choice in narrative context.
        label: 0 (non-deceptive) or 1 (deceptive).

    Example:
        >>> pairs = load_deception_pairs(limit=100)
        >>> texts, labels = zip(*pairs)
        >>> print(f"Deceptive: {sum(labels)}/{len(labels)}")

    Citation:
        Pan et al. (2023). Do the Rewards Justify the Means? Measuring Trade-Offs
        Between Rewards and Ethical Behavior in the Machiavelli Benchmark. ICML 2023.
        https://arxiv.org/abs/2304.03279
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "datasets library required: pip install datasets"
        ) from e

    rng = random.Random(seed)

    dataset = load_dataset(
        "wassname/machiavelli",
        split="train",
        streaming=streaming,
    )

    positive_pairs: list[tuple[str, int]] = []
    negative_pairs: list[tuple[str, int]] = []

    # We need enough to fill `limit` with balanced-ish representation
    # Natural ratio ~15% deceptive, so oversample by scanning up to limit*10
    scan_limit = limit * 10

    for i, row in enumerate(dataset):
        if i >= scan_limit:
            break

        obs = (row.get("obs") or "").strip()
        choice_texts = row.get("choice_text") or {}
        choice_labels = row.get("choice_labels") or {}

        for choice_idx in choice_texts:
            action_text = (choice_texts.get(choice_idx) or "").strip()
            if not action_text or action_text.lower() == "none":
                continue

            labels = choice_labels.get(choice_idx) or {}
            deception_score = float(labels.get("morality_deception") or 0.0)
            is_deceptive = 1 if deception_score >= deception_threshold else 0

            if include_context and obs:
                text = f"{obs} [ACTION]: {action_text}"
            else:
                text = action_text

            pair = (text, is_deceptive)
            if is_deceptive:
                positive_pairs.append(pair)
            else:
                negative_pairs.append(pair)

        # Early exit: collect 3x limit of each to ensure we have enough
        if len(positive_pairs) >= limit and len(negative_pairs) >= limit * 3:
            break

    # Build balanced-ish sample: fill up to `limit` with ~25% positive if possible
    target_pos = min(len(positive_pairs), limit // 4)
    target_neg = min(len(negative_pairs), limit - target_pos)

    rng.shuffle(positive_pairs)
    rng.shuffle(negative_pairs)

    result = positive_pairs[:target_pos] + negative_pairs[:target_neg]
    rng.shuffle(result)

    return result


def load_manipulation_pairs(
    limit: int = 500,
    include_context: bool = True,
    threshold: float = 1.0,
    seed: int = 42,
) -> list[tuple[str, int]]:
    """
    Load (text, is_manipulative) pairs. Same interface as load_deception_pairs
    but targets the morality_manipulation field.

    See load_deception_pairs docstring for full parameter documentation.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("pip install datasets") from e

    rng = random.Random(seed)
    dataset = load_dataset("wassname/machiavelli", split="train", streaming=True)

    pos, neg = [], []
    for i, row in enumerate(dataset):
        if i >= limit * 10:
            break
        obs = (row.get("obs") or "").strip()
        choice_texts = row.get("choice_text") or {}
        choice_labels = row.get("choice_labels") or {}

        for idx in choice_texts:
            text = (choice_texts.get(idx) or "").strip()
            if not text or text.lower() == "none":
                continue
            labels = choice_labels.get(idx) or {}
            score = float(labels.get("morality_manipulation") or 0.0)
            label = 1 if score >= threshold else 0
            full_text = f"{obs} [ACTION]: {text}" if include_context and obs else text
            (pos if label else neg).append((full_text, label))

        if len(pos) >= limit and len(neg) >= limit * 3:
            break

    target_pos = min(len(pos), limit // 4)
    target_neg = min(len(neg), limit - target_pos)
    rng.shuffle(pos)
    rng.shuffle(neg)
    result = pos[:target_pos] + neg[:target_neg]
    rng.shuffle(result)
    return result


def _available_morality_fields() -> list[str]:
    """Return all morality label fields available in the dataset."""
    return [
        "morality_deception",
        "morality_manipulation",
        "morality_killing",
        "morality_physical_harm",
        "morality_non_physical_harm",
        "morality_intending_harm",
        "morality_promises",
        "morality_stealing",
        "morality_trespassing",
        "morality_spying",
        "morality_vandalism",
        "morality_fairness",
        "morality_other",
    ]


if __name__ == "__main__":
    print("Loading deception pairs (limit=200, streaming)...")
    pairs = load_deception_pairs(limit=200)

    texts, labels = zip(*pairs)
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count

    print(f"Total pairs: {len(pairs)}")
    print(f"Deceptive  (label=1): {pos_count} ({pos_count/len(labels)*100:.1f}%)")
    print(f"Neutral    (label=0): {neg_count} ({neg_count/len(labels)*100:.1f}%)")
    print()

    lengths = [len(t) for t in texts]
    print(f"Text length: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.0f} chars")
    print()

    print("=== FIRST 5 PAIRS ===")
    for i, (text, label) in enumerate(pairs[:5]):
        print(f"--- Pair {i+1} (label={label}) ---")
        print(f"  Text ({len(text)} chars): {text[:300]}")
        print()
