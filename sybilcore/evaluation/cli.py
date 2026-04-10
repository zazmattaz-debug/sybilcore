"""Terminal-based labeling tool for the SybilCore eval corpus.

Usage:
    python -m sybilcore.evaluation.cli --corpus experiments/human_eval_corpus_v4.json
    python -m sybilcore.evaluation.cli --rater alice

The tool loads a saved corpus, walks through unlabeled agents one at a time,
and asks the rater for a verdict (suspicious/benign/skip) plus confidence
and free-text notes. Progress is persisted on every keystroke so the rater
can quit at any time and resume later.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sybilcore.evaluation.human_eval import (
    AgentRecord,
    HumanEvalFramework,
)

logger = logging.getLogger(__name__)

_LABEL_MAP = {
    "y": "suspicious",
    "yes": "suspicious",
    "s": "suspicious",
    "n": "benign",
    "no": "benign",
    "b": "benign",
    "skip": "skip",
    "sk": "skip",
}


def _prompt(text: str, default: str = "") -> str:
    raw = input(text).strip()
    return raw or default


def _ask_label() -> str:
    while True:
        ans = _prompt("Suspicious? [y/n/skip]: ").lower()
        if ans in _LABEL_MAP:
            return _LABEL_MAP[ans]
        print("  Please answer y, n, or skip.")


def _ask_confidence() -> int:
    while True:
        raw = _prompt("Confidence (1-5) [3]: ", "3")
        try:
            val = int(raw)
        except ValueError:
            print("  Must be an integer 1..5.")
            continue
        if 1 <= val <= 5:
            return val
        print("  Out of range; try 1..5.")


def _print_agent(idx: int, total: int, agent: AgentRecord) -> None:
    print()
    print("=" * 72)
    print(f"  Agent {idx + 1} / {total}")
    print("=" * 72)
    print(HumanEvalFramework.format_for_review(agent))
    print("-" * 72)


def run(corpus_path: Path, labels_path: Path, rater_id: str) -> None:
    framework = HumanEvalFramework(labels_path=labels_path)
    records = HumanEvalFramework.load_corpus(corpus_path)
    if not records:
        print(f"No records found in corpus {corpus_path}", file=sys.stderr)
        sys.exit(1)

    already_labeled = {
        j.agent_id for j in framework.load_judgments() if j.rater_id == rater_id
    }
    pending = [r for r in records if r.agent_id not in already_labeled]

    print(f"Corpus:    {corpus_path}")
    print(f"Labels:    {labels_path}")
    print(f"Rater:     {rater_id}")
    print(f"Total:     {len(records)}  Already labeled: {len(already_labeled)}  Pending: {len(pending)}")

    for offset, agent in enumerate(pending):
        _print_agent(offset, len(pending), agent)
        try:
            label = _ask_label()
            confidence = _ask_confidence() if label != "skip" else 1
            notes = _prompt("Notes (optional): ")
        except (EOFError, KeyboardInterrupt):
            print("\nQuitting — progress saved.")
            return

        framework.record_judgment(
            agent_id=agent.agent_id,
            label=label,
            confidence=confidence,
            notes=notes,
            rater_id=rater_id,
        )
        print(f"  -> recorded {label} (conf={confidence})")

    print()
    print("All agents labeled. Computing metrics…")
    judgments = framework.load_judgments()
    metrics = HumanEvalFramework.compute_agreement_metrics(records, judgments)
    print(f"  precision={metrics['precision']}  recall={metrics['recall']}  f1={metrics['f1']}")
    print(f"  accuracy={metrics['accuracy']}  matrix={metrics['matrix']}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="SybilCore human-eval CLI labeling tool")
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to a corpus JSON saved by run_human_eval.py",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("experiments/corpus_labels.jsonl"),
        help="JSONL file to append labels to",
    )
    parser.add_argument("--rater", default="default", help="Identifier for this rater")
    args = parser.parse_args(argv)

    run(args.corpus, args.labels, args.rater)


if __name__ == "__main__":
    main()
