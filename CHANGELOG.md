# Changelog

All notable changes to SybilCore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Open-source repository hygiene: `SECURITY.md`, `CHANGELOG.md`, GitHub issue
  and pull-request templates, GitHub Actions CI workflow, `docs/` directory
  with architecture, quickstart, and research indexes.

### Changed
- Documentation reorganized under `docs/` to prepare for a static site build.

### Deprecated
- `EmbeddingBrain` (transformer-based semantic brain) is slated for removal in
  `v0.2.0`. The v4 15k-iteration tournament showed it was consistently
  dominated by `ContrastiveEmbeddingBrain` and `SemanticBrain` on every
  adversarial split while doubling the import-time footprint. Users who depend
  on it should migrate before upgrading.

### Removed
- _(pending)_ `EmbeddingBrain` will be removed in `v0.2.0`.

### Fixed
- _(pending)_

### Security
- _(pending)_

---

## [0.2.0-dev] — unreleased

Development track for the next minor release. Highlights:

- **Brain roster cleanup.** `EmbeddingBrain` will be removed; the default
  registry shrinks to the 14 brains that survived the v4 tournament.
- **Config-driven weights.** Brain weights will be loadable from a YAML config
  instead of being hard-coded in each brain class.
- **Research docs.** The `research/` folder is being indexed from `docs/research.md`
  so external readers can navigate the 15k-iter tournament, ablations,
  calibration curves, and threat model without cloning the repo.

### Changed
- Removed `EmbeddingBrain` from the default ensemble returned by
  `sybilcore.brains.get_default_brains()`, and confirmed it is absent from
  `sybilcore.core.config.DEFAULT_BRAIN_WEIGHTS`. Five independent v4
  methodologies converged on "prune": (1) Tournament v4 — silent across 15k
  iterations; (2) correlation analysis — clustered redundantly with
  `ContrastiveEmbeddingBrain`; (3) calibration — learned weight of -0.76;
  (4) ablation — dropping it cut FPR from 0.76 to 0.42 with **zero** detection-
  rate loss; (5) adversarial training — identified as a blindspot. The
  `EmbeddingBrain` class remains importable from `sybilcore.brains.embedding`
  for opt-in use. See `SESSION_HANDOFF_V4_COMPLETE.md` ("The Cross-Verified
  Prune List") for the full analysis. `FidelityBrain` is **not** being pruned
  yet — it is pending v5 RAG-corpus validation.

See the `[Unreleased]` section above for the running list of changes on this
track.

---

## [0.1.0] — 2026-04-02

The first tagged release. This version reflects the outcome of the v4 15,000-iteration
adversarial tournament that selected the current default brain roster.

### Added
- **Five default brain modules** shipped as `get_default_brains()`:
  - `DeceptionBrain` — contradictory outputs, fabricated citations, confidence miscalibration
  - `ResourceHoardingBrain` — excessive tool calls, repeated access, data accumulation
  - `SocialGraphBrain` — fan-out communication, message spikes, anomalous interactions
  - `IntentDriftBrain` — deviation from stated objectives, goal shifting, scope creep
  - `CompromiseBrain` — external call storms, suspicious destinations, prompt-injection indicators
- **Ten extended brains** available for opt-in use: `ContrastiveEmbeddingBrain`,
  `EconomicBrain`, `EmbeddingBrain`, `FidelityBrain`, `IdentityBrain`,
  `NeuroBrain`, `SemanticBrain`, `SilenceBrain`, `SwarmBrain`, `TemporalBrain`.
- **Coefficient aggregator** producing a single `AgentCoefficient` (0–500)
  with tiers: `CLEAR`, `CLOUDED`, `FLAGGED`, `LETHAL_ELIMINATOR`.
- **Dominator API** — FastAPI REST service exposing `/scan`, `/agents`,
  `/agents/{id}/history`, and `/health`.
- **CLI** — `sybilcore serve`, `sybilcore dashboard`, `sybilcore scan`.
- **Integrations** — LangChain callback handler and CrewAI monitor.
- **Test suite** — 96 tests, 82% line coverage.
- **Documentation** — initial `README.md` and `CONTRIBUTING.md`.

### Research Milestones
- **v4 15k-iteration adversarial tournament** completed. Results showed the
  five-brain default roster achieves the best trade-off between detection
  accuracy and false-positive rate across all adversarial splits tested
  (AdvBench, AgentHarm, HarmBench, DoNotAnswer, JailbreakBench). Full results
  in `research/ABLATION_RESULTS.md` and the experiment logs under
  `experiments/`.
- Baselines plan, threat model, competitive landscape, and black-hat mindset
  documents published under `research/`.

### Known Limitations
- `EmbeddingBrain` is included but deprecated; see `[Unreleased]`.
- API authentication is a placeholder; operators must add their own auth layer.
- Persistence uses in-memory storage by default; a SQLite/Postgres adapter is
  planned for `v0.3.0`.

---

[Unreleased]: https://github.com/sybilcore/sybilcore/compare/v0.1.0...HEAD
[0.2.0-dev]: https://github.com/sybilcore/sybilcore/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sybilcore/sybilcore/releases/tag/v0.1.0
