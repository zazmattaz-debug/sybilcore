"""Budget-only ablation: 5000 iter × 13 brains × no cross-pollination × 1 seed × gpt4o.

This is the final missing cell from the Codex/Grok review:
tests whether MCE dominance emerges from budget alone without cross-pollination.

Usage:
    python3 scripts/ablation_budget_only.py
"""
from __future__ import annotations
import asyncio, json, logging, sys, time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(Path("/Users/zazumoloi/Desktop/Claude Code/.env"))

from sybilcore.simulation.tournament import AdversarialTournament

SEED = 20260408
MODEL = "gpt4o"
ITERATIONS = 5000
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    logger = logging.getLogger("ablation_budget_only")

    logger.info("=== Starting: 5000 iter, 13 brains, NO cross-poll, seed=%d ===", SEED)

    tournament = AdversarialTournament(
        models=MODEL,
        iterations=ITERATIONS,
        population_size=20,
        cross_pollination_interval=999999,  # disabled
        seed=SEED,
    )

    t0 = time.monotonic()
    result = await tournament.run()
    elapsed = time.monotonic() - t0

    mr = result.per_model.get(MODEL)
    tax = (mr.strategy_taxonomy or {}) if mr else {}
    total = sum(tax.values()) or 1
    mce_pct = (tax.get("multi_channel_exfiltration", 0) / total * 100)

    summary = {
        "description": "Budget-only ablation: 5000 iter, 13 brains, no cross-pollination",
        "seed": SEED,
        "model": MODEL,
        "iterations": ITERATIONS,
        "n_brains": 13,
        "cross_pollination": False,
        "best_coefficient": result.overall_best_coefficient,
        "strategy_taxonomy": dict(tax),
        "multi_channel_exfiltration_pct": round(mce_pct, 1),
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now(UTC).isoformat(),
    }

    out_path = EXPERIMENTS_DIR / "ablation_budget_only_5000iter.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("=== DONE: best=%.2f, MCE=%.1f%%, elapsed=%.0fs ===",
                result.overall_best_coefficient, mce_pct, elapsed)
    logger.info("Output -> %s", out_path)
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
