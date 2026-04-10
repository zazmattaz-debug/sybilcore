"""Track 6: Adversarial LLM Red Team — LIVE mode runner.

Runs the AdversarialLLMRedTeam with Gemini Flash API to discover
evasion strategies against SybilCore's detection brains.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Load GEMINI_API_KEY from master .env before importing sybilcore
_ENV_PATH = Path("/Users/zazumoloi/Desktop/Claude Code/.env")
with open(_ENV_PATH) as _f:
    for _line in _f:
        if _line.startswith("GEMINI_API_KEY="):
            os.environ["GEMINI_API_KEY"] = _line.split("=", 1)[1].strip()
            break

# Now import after env is set
from sybilcore.simulation.adversarial_llm import AdversarialLLMRedTeam, RedTeamResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("adversarial_llm_live")

ITERATIONS = 40
EXPERIMENTS_DIR = Path(__file__).resolve().parents[1].parent / "experiments"


def _result_to_dict(result: RedTeamResult) -> dict:
    """Convert RedTeamResult to a JSON-serializable dict."""
    return dataclasses.asdict(result)


def _print_summary(result: RedTeamResult) -> None:
    """Print a human-readable summary of the red team run."""
    print("\n" + "=" * 70)
    print("  ADVERSARIAL LLM RED TEAM — LIVE RESULTS")
    print("=" * 70)
    print(f"  Mode:        {result.mode}")
    print(f"  Iterations:  {result.iterations}")
    print(f"  Elapsed:     {result.total_elapsed_seconds:.1f}s")
    print(f"  Evasion Floor: {result.evasion_floor}")
    print(f"  Evaded (<100): {'YES' if result.best_evasion.get('evaded') else 'NO'}")

    print("\n--- Best Evasion Attempt ---")
    best = result.best_evasion
    print(f"  Coefficient: {best.get('coefficient', 'N/A')}")
    print(f"  Tier:        {best.get('tier', 'N/A')}")
    print(f"  Strategy:    {best.get('strategy', 'N/A')}")
    print(f"  Events:      {best.get('event_count', 'N/A')}")
    print(f"  Brain Scores:")
    for brain, score in best.get("brain_scores", {}).items():
        print(f"    {brain:30s} {score:6.1f}")

    print("\n--- Evasion Strategies Found ---")
    for s in result.evasion_strategies:
        evaded_tag = " [EVADED]" if s.get("evaded") else ""
        print(f"  [{s.get('iteration_discovered', '?'):>2}] {s['strategy']:35s} "
              f"coeff={s['coefficient']:6.1f}{evaded_tag}")

    print("\n--- Brain Vulnerability Analysis ---")
    analysis = result.brain_analysis
    per_brain = analysis.get("per_brain", {})
    for brain, info in sorted(per_brain.items()):
        print(f"  {brain:30s} min={info['min_score']:6.1f}  "
              f"max={info['max_score']:6.1f}  "
              f"difficulty={info['evasion_difficulty']}")

    print(f"\n  Easiest to evade: {analysis.get('easiest_to_evade', 'N/A')}")
    print(f"  Hardest to evade: {analysis.get('hardest_to_evade', 'N/A')}")

    # Trajectory summary: show coefficient descent
    print("\n--- Coefficient Trajectory ---")
    for entry in result.trajectory:
        it = entry["iteration"]
        coeff = entry["best_coefficient"]
        best_ever = entry["best_ever_coefficient"]
        bar = "#" * max(1, int(coeff / 10))
        evaded = " *EVADED*" if entry.get("evaded") else ""
        print(f"  iter {it:>2}: curr={coeff:6.1f} best={best_ever:6.1f} {bar}{evaded}")

    print("=" * 70)


def main() -> None:
    """Run the live adversarial LLM red team experiment."""
    logger.info("Starting Track 6: Adversarial LLM Red Team (LIVE)")
    logger.info("Iterations: %d", ITERATIONS)

    red_team = AdversarialLLMRedTeam(mode="live")
    result = red_team.run(iterations=ITERATIONS)

    # Save to experiments directory
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = EXPERIMENTS_DIR / f"adversarial_llm_live_{ts}.json"

    result_dict = _result_to_dict(result)
    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)

    logger.info("Results saved to %s", output_path)

    _print_summary(result)


if __name__ == "__main__":
    main()
