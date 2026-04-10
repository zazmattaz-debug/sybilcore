"""Run adversarial LLM red team experiment against SybilCore.

Runs in mock mode by default (no API key needed). Set GEMINI_API_KEY
to enable live mode with Gemini Flash.

Usage:
    # Mock mode (default, no API needed)
    python -m sybilcore.simulation.run_adversarial_llm

    # Live mode with Gemini API
    python -m sybilcore.simulation.run_adversarial_llm --mode live

    # Custom iterations
    python -m sybilcore.simulation.run_adversarial_llm --iterations 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from sybilcore.simulation.adversarial_llm import AdversarialLLMRedTeam, EVASION_THRESHOLD

logger = logging.getLogger(__name__)


def run_experiment(
    mode: str = "mock",
    iterations: int = 20,
    output_dir: str = "experiments",
) -> Path:
    """Run the adversarial LLM experiment and save results.

    Args:
        mode: "mock" for template-based, "live" for Gemini API.
        iterations: Number of evolutionary generations.
        output_dir: Directory for result files.

    Returns:
        Path to the saved results file.
    """
    api_key = os.environ.get("GEMINI_API_KEY") if mode == "live" else None

    red_team = AdversarialLLMRedTeam(
        mode=mode,
        api_key=api_key,
    )

    logger.info(
        "Starting adversarial LLM red team: mode=%s iterations=%d",
        mode, iterations,
    )

    result = red_team.run(iterations=iterations)

    # Save results
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filepath = out / f"adversarial_llm_{ts}.json"

    output = {
        "type": "adversarial_llm_red_team",
        "mode": result.mode,
        "iterations": result.iterations,
        "start_time": result.start_time,
        "end_time": result.end_time,
        "total_elapsed_seconds": result.total_elapsed_seconds,
        "evasion_threshold": EVASION_THRESHOLD,
        "evasion_floor": result.evasion_floor,
        "best_evasion": result.best_evasion,
        "brain_analysis": result.brain_analysis,
        "evasion_strategies": result.evasion_strategies,
        "trajectory": result.trajectory,
    }

    with filepath.open("w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved to %s", filepath)
    return filepath


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adversarial LLM red team vs SybilCore",
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default="mock",
        help="mock=template-based (no API), live=Gemini Flash API",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of evolutionary generations (default: 20)",
    )
    parser.add_argument(
        "--output",
        default="experiments",
        help="Output directory for results (default: experiments)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args = _parse_args()
    filepath = run_experiment(
        mode=args.mode,
        iterations=args.iterations,
        output_dir=args.output,
    )

    # Load and print summary
    with filepath.open() as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print("ADVERSARIAL LLM RED TEAM COMPLETE")
    print(f"{'='*60}")
    print(f"Mode: {data['mode']}")
    print(f"Iterations: {data['iterations']}")
    print(f"Duration: {data['total_elapsed_seconds']}s")
    print(f"Evasion threshold: < {data['evasion_threshold']}")
    print(f"\nEVASION FLOOR: {data['evasion_floor']}")

    best = data["best_evasion"]
    status = "EVADED" if best["evaded"] else "DETECTED"
    print(f"Best attempt: [{status}] coefficient={best['coefficient']} tier={best['tier']}")
    print(f"  Strategy: {best['strategy']}")
    print(f"  Brain scores: {best['brain_scores']}")

    analysis = data["brain_analysis"]
    print(f"\nBRAIN VULNERABILITY ANALYSIS:")
    print(f"  Easiest to evade: {analysis.get('easiest_to_evade', 'N/A')}")
    print(f"  Hardest to evade: {analysis.get('hardest_to_evade', 'N/A')}")

    for brain, info in analysis.get("per_brain", {}).items():
        print(f"  {brain}: min={info['min_score']} max={info['max_score']} "
              f"difficulty={info['evasion_difficulty']}")

    strategies = data["evasion_strategies"]
    if strategies:
        print(f"\nTOP EVASION STRATEGIES ({len(strategies)} found):")
        for s in strategies[:5]:
            evaded = "EVADED" if s["evaded"] else "detected"
            print(f"  [{evaded}] {s['strategy']}: coefficient={s['coefficient']}")
    else:
        print("\nNo evasion strategies found below coefficient 200")

    print(f"\nTrajectory (first/last 3):")
    for entry in data["trajectory"][:3]:
        print(f"  Iter {entry['iteration']}: best={entry['best_coefficient']} "
              f"tier={entry['best_tier']} evaded={entry['evaded']}")
    if len(data["trajectory"]) > 6:
        print("  ...")
    for entry in data["trajectory"][-3:]:
        print(f"  Iter {entry['iteration']}: best={entry['best_coefficient']} "
              f"tier={entry['best_tier']} evaded={entry['evaded']}")

    print(f"\nResults: {filepath}")
