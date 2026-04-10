"""Top-level driver for the SybilCore human evaluation pipeline.

What it does:
    1. Samples a balanced corpus from all 4 sources.
    2. Saves the corpus to experiments/human_eval_corpus_v4.json.
    3. Optionally runs the AI judge baseline (Gemini Pro).
    4. Optionally launches the FastAPI web UI.

Usage:
    python -m sybilcore.evaluation.run_human_eval --n 200
    python -m sybilcore.evaluation.run_human_eval --n 100 --ai-judge
    python -m sybilcore.evaluation.run_human_eval --launch-web
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from sybilcore.evaluation.ai_judge import run_ai_judge_on_corpus, summarize_run
from sybilcore.evaluation.human_eval import HumanEvalFramework

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTS_DIR = _REPO_ROOT / "experiments"
_DEFAULT_CORPUS_PATH = _EXPERIMENTS_DIR / "human_eval_corpus_v4.json"


def _load_dotenv() -> None:
    """Best-effort load of .env from the workspace root for GEMINI_API_KEY."""
    env_path = _REPO_ROOT.parents[2] / ".env"  # rooms/engineering/sybilcore -> Claude Code/.env
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except OSError as exc:
        logger.debug("Could not load .env at %s: %s", env_path, exc)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="SybilCore human eval orchestrator")
    parser.add_argument("--n", type=int, default=200, help="Total corpus size")
    parser.add_argument(
        "--corpus-out",
        type=Path,
        default=_DEFAULT_CORPUS_PATH,
        help="Where to save the sampled corpus",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ai-judge", action="store_true", help="Run the Gemini AI judge")
    parser.add_argument(
        "--ai-judge-n",
        type=int,
        default=100,
        help="How many agents to evaluate with the AI judge",
    )
    parser.add_argument(
        "--launch-web",
        action="store_true",
        help="Launch the FastAPI web UI after sampling",
    )
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8765)
    args = parser.parse_args(argv)

    _load_dotenv()

    framework = HumanEvalFramework(seed=args.seed)
    print(f"Sampling corpus of {args.n} agents…")
    corpus = framework.sample_corpus(n=args.n, source="mixed")
    by_source: dict[str, int] = {}
    for r in corpus:
        by_source[r.source] = by_source.get(r.source, 0) + 1
    print(f"  total={len(corpus)} by_source={by_source}")
    HumanEvalFramework.save_corpus(corpus, args.corpus_out)
    print(f"  saved -> {args.corpus_out}")

    if args.ai_judge:
        n_judge = min(args.ai_judge_n, len(corpus))
        print(f"\nRunning AI judge on {n_judge} agents…")
        verdicts, _ = run_ai_judge_on_corpus(corpus[:n_judge])
        summary = summarize_run(corpus[:n_judge], verdicts)
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        out_path = _EXPERIMENTS_DIR / f"ai_judge_v4_{ts}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fp:
            json.dump(summary, fp, indent=2, default=str)
        print(f"  saved -> {out_path}")
        m = summary["agreement_metrics"]
        print(
            f"  precision={m['precision']} recall={m['recall']} "
            f"f1={m['f1']} accuracy={m['accuracy']}"
        )
        print(f"  matrix={m['matrix']}")
        for src, bucket in summary["by_source"].items():
            print(f"  {src}: {bucket}")

    if args.launch_web:
        print(f"\nLaunching web UI on http://{args.web_host}:{args.web_port}…")
        from sybilcore.evaluation.web_ui import create_app

        try:
            import uvicorn
        except ImportError:
            print("uvicorn not installed; install with: pip install uvicorn", file=sys.stderr)
            return 1
        db_path = _EXPERIMENTS_DIR / "human_eval_labels.db"
        app = create_app(args.corpus_out, db_path)
        uvicorn.run(app, host=args.web_host, port=args.web_port)

    return 0


if __name__ == "__main__":
    sys.exit(main())
