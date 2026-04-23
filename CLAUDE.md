# Agent Rules -- SybilCore
Stack: Python 3.13 (StrEnum requires 3.11+), FastAPI 0.115+, Pydantic 2, pytest 8, ruff, mypy strict, Hatchling.

## Code Style
- Max file length: 400 lines. Split at 200.
- Type annotations on every function signature (mypy strict enforced).
- Ruff line-length: 100. Use `from __future__ import annotations`.
- Frozen dataclasses or Pydantic models for all data objects -- never mutate.
- Brain scores are 0-100 floats. Coefficient is 0-100. Never exceed bounds.

## Workflow
- Run `mulch prime engineering` before starting work.
- Read `_CONTEXT.md` before making changes.
- Run `pytest` after edits. All 727 tests must pass. Fix silently.
- Run `ruff check sybilcore/ tests/` and `mypy sybilcore/` after edits.
- Do not install new dependencies without asking.
- Experiment artifacts go in `experiments/` -- never commit scratch files.
- Git ops: use `/usr/bin/git`. Hooks may block push -- commit locally and let the user push manually.

## Brain Module Rules
- New brains must subclass `BaseBrain` and implement `name`, `weight`, `score()`.
- Register via `sybilcore.brains.register_brain()`, never modify `get_default_brains()` directly.
- The 13-brain default ensemble is frozen for paper reproducibility. Do not add or remove.
- EmbeddingBrain and FidelityBrain are pruned -- available but excluded from defaults.
- Brain 14 (InterpretabilityProxyBrain) is NOT in the frozen 13-brain default. It is an exploratory research configuration -- do not include it in `get_default_brains()`.
- TRIBE Modal endpoint (`zazmattaz--cerebrum-tribe-tribescorer-score.modal.run`) returns `region_activations: {}` (empty dict). Supervised TRIBE features reduce to a 3-dim scalar triple. Cold starts are 30-90s -- account for latency in any scoring pipeline.

## Benchmark / Research Rules
- Length-artifact check is MANDATORY on any new corpus: save `text_length` per row, compute length-AUROC, flag as artifact if AUROC > 0.6.
- DeceptionBench AUROC 0.997 is an unverified length artifact -- do not cite as a clean result.

## Anti-Patterns
- Never hardcode brain weights -- use `core/config.py` or `weight_presets.py`.
- Never return scores outside 0-100 or coefficients outside 0-100.
- Never modify paper-evidence JSON files in `experiments/`.
- Never import simulation modules from core library code (one-way dependency).
- Never skip confidence values in BrainScore -- always set explicitly.
