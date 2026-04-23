# SybilCore -- Engineering Project
> Runtime behavioral monitoring for autonomous LLM agents. Assigns each agent an Agent Coefficient (0-100) via 13 independent "brain" modules, then enforces trust tiers (CLEAR / LATENT / CLOUDED / FLAGGED / ENFORCEMENT). Behavioral output-stream scoring -- unoccupied by commercial products as of Apr 2026.

## Stack
- Python 3.13 (StrEnum requires 3.11+; core library, pip-installable)
- FastAPI + Uvicorn (Dominator API -- `/scan`, `/score`)
- Pydantic 2 (models, request/response validation)
- pytest 8 + pytest-cov (727 tests, 82% coverage)
- ruff + mypy strict (lint + type-check)
- Hatchling (build backend)
- Optional: LangChain, CrewAI, MiroFish integrations

## Architecture
```
Events --> [13 Brain Modules] --> CoefficientCalculator --> Snapshot (coeff + tier)
                                                       |
                                                  Dominator API
                                                  (FastAPI, port 8080)
```
Each brain scores 0-100 with confidence. The calculator produces a weighted aggregate scaled to 0-100. Enforcement tiers map coefficient ranges to actions (full access -> termination).

## File Structure
```
sybilcore/
  brains/           # 13 default + 2 opt-in brains (base.py = interface)
  brains_v5_patched/ # Experimental v5 hardened variants
  core/             # coefficient.py, config.py, weight_presets.py
  models/           # event.py (Event, EventType), agent.py (Snapshot, Tier)
  dominator/        # FastAPI app + response logic
  api/              # Standalone API server + service layer
  simulation/       # Synthetic corpus gen, adversarial LLM, archetypes, long-horizon
  analysis/         # Correlation, calibration, ablation, adversarial training
  evaluation/       # Human eval, AI judge, CLI + web UI
  integrations/     # LangChain, CrewAI, MiroFish, MoltBook
  visibility/       # Translator (coefficient -> human explanation)
  dashboard/        # Jinja2 dashboard app
tests/              # Mirrors sybilcore/ structure. 727 tests.
experiments/        # Paper evidence JSONs + Brain 14 phase eval artifacts + phase2v2 run files
paper/              # LaTeX paper (main.tex)
research/           # Design docs, threat model, landscape analysis
sybilcore-sdk/      # Pip-installable client SDK
OBSOLESCENCE_CHECK_20260415.md   # Market analysis post Anthropic Managed Agents Apr 2026
UNDEREXPLORED_CORPORA_20260415.md  # Phase 2 v2 corpus plan (MACHIAVELLI, AgentDojo, etc.)
```

## Quick Start
```bash
pip install -e ".[dev]"       # Install with dev deps
pytest                         # 727 tests
ruff check sybilcore/ tests/   # Lint
mypy sybilcore/                # Type-check
uvicorn sybilcore.dominator.api:app --port 8080  # Run API
```

## Key Config
- `sybilcore/core/config.py` -- MAX_COEFFICIENT, SCORING_WINDOW_SECONDS, DEFAULT_BRAIN_WEIGHTS
- `sybilcore/core/weight_presets.py` -- optimized/default weight sets from calibration
- No .env required for core library. API may use env vars for auth.

## Status
- **2026-04-15**: 727 tests passing, 82% coverage, 13-brain default ensemble frozen
- **Done**: v4 tournament (15k iterations), ablation, calibration v5.1, HuggingFace dataset published, Brain 14 Phase 1+2v1+3 evaluated (verdict: MARGINAL), paper intro + related work updated (commits 27e115a, dab15ea)
- **Active**: Phase 2 v2 — 4-corpus parallel scoring (MACHIAVELLI, AgentDojo, Alignment Faking, MALT)
- **Next**: Phase 2 v2 results into paper appendix, NeurIPS submission (deadlines May 4/6)
