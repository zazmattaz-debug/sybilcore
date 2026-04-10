"""CLI runner for the SybilCore adversarial training loop.

Loads the v3 tournament results, runs the trainer, writes proposed
patches to research/proposed_brain_patches/, and saves a JSON summary
to experiments/adversarial_training_v4_TIMESTAMP.json.

Usage:
    python -m sybilcore.analysis.run_adversarial_training
    python -m sybilcore.analysis.run_adversarial_training --tournament path/to.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from sybilcore.analysis.adversarial_training import (
    AdversarialTrainer,
    BrainBlindspot,
    BrainPatch,
    TrainingResult,
)

logger = logging.getLogger(__name__)

DEFAULT_TOURNAMENT = Path("experiments/tournament_20260406_170217.json")
DEFAULT_PATCH_DIR = Path("research/proposed_brain_patches")
DEFAULT_RESULTS_DIR = Path("experiments")


def _serialize_blindspot(b: BrainBlindspot) -> dict:
    return {
        "brain_name": b.brain_name,
        "miss_count": b.miss_count,
        "total_candidates": b.total_candidates,
        "miss_rate": round(b.miss_rate, 3),
        "avg_missed_score": b.avg_missed_score,
        "sample_event_types": list(b.sample_event_types),
        "sample_metadata_keys": list(b.sample_metadata_keys),
        "sample_content_terms": list(b.sample_content_terms),
    }


def _serialize_patch(p: BrainPatch) -> dict:
    return {
        "brain_name": p.brain_name,
        "rationale": p.rationale,
        "new_thresholds": p.new_thresholds,
        "new_weight_multiplier": p.new_weight_multiplier,
        "target_path": str(p.target_path),
        "code_snippet_preview": p.code_snippet.splitlines()[:10],
    }


def _serialize_result(result: TrainingResult) -> dict:
    return {
        "tournament_path": result.tournament_path,
        "candidate_count": result.candidate_count,
        "floor_before": result.floor_before,
        "floor_after": result.floor_after,
        "floor_delta": result.floor_delta,
        "timestamp": result.timestamp,
        "blindspots": [_serialize_blindspot(b) for b in result.blindspots],
        "patches": [_serialize_patch(p) for p in result.patches],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adversarial trainer runner")
    parser.add_argument(
        "--tournament", type=Path, default=DEFAULT_TOURNAMENT,
        help="Path to the tournament_*.json input.",
    )
    parser.add_argument(
        "--patch-dir", type=Path, default=DEFAULT_PATCH_DIR,
        help="Directory where proposed brain patches will be written.",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help="Directory where the JSON summary will be written.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.tournament.exists():
        print(f"ERROR: tournament file not found: {args.tournament}", file=sys.stderr)
        return 2

    args.patch_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    trainer = AdversarialTrainer(patch_dir=args.patch_dir)
    result = trainer.run(args.tournament)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = args.results_dir / f"adversarial_training_v4_{timestamp}.json"
    out_path.write_text(
        json.dumps(_serialize_result(result), indent=2),
        encoding="utf-8",
    )

    print("=" * 64)
    print("SybilCore Adversarial Training v4")
    print("=" * 64)
    print(f"Tournament input  : {result.tournament_path}")
    print(f"Candidates loaded : {result.candidate_count}")
    print(f"Blindspots found  : {len(result.blindspots)}")
    print(f"Patches proposed  : {len(result.patches)}")
    print(f"Patch directory   : {args.patch_dir}")
    print(f"Results saved to  : {out_path}")
    print()
    print("Top blindspots:")
    for b in result.blindspots[:8]:
        print(
            f"  - {b.brain_name:<18} missed {b.miss_count}/{b.total_candidates} "
            f"(avg score {b.avg_missed_score:.1f})",
        )
    print()
    print("Coefficient floor:")
    print(f"  before : {result.floor_before:.2f}")
    print(f"  after  : {result.floor_after:.2f}")
    print(f"  delta  : {result.floor_delta:+.2f}")
    print("=" * 64)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
