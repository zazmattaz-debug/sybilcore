# MACHIAVELLI Benchmark — Scout Report

## Paper

**Title:** Do the Rewards Justify the Means? Measuring Trade-Offs Between Rewards and Ethical Behavior in the MACHIAVELLI Benchmark

**Venue:** ICML 2023 (Oral)

**arXiv:** https://arxiv.org/abs/2304.03279

**Authors:** Alexander Pan, Jun Shern Chan, Andy Zou, Nathaniel Li, Steven Basart, Thomas Woodside, Jonathan Ng, Hanlin Zhang, Scott Emmons, Dan Hendrycks

## Dataset URLs

| Source | URL | Notes |
|--------|-----|-------|
| Original repo (MIT) | https://github.com/aypan17/machiavelli | Game env + Google Drive data (password: "machiavelli") |
| Original website | https://aypan17.github.io/machiavelli/ | Summary + leaderboard |
| HuggingFace wrapper | https://huggingface.co/datasets/wassname/machiavelli | Pre-processed, no password needed |
| HuggingFace wrapper repo | https://github.com/wassname/machiavelli_as_ds | Code for the HF version |

## License

**MIT** — both the original repo and the HuggingFace wrapper. No ethics agreement required. No access gates.

Caveat: the underlying games are copyrighted fiction by their respective authors (Choice of Games). The annotations and game structure data are MIT-licensed for research use. This is standard for NLP game benchmarks (see also: TextWorld, LIGHT). Acceptable for SybilCore paper.

## Dataset Size

| Metric | Value |
|--------|-------|
| Games (original) | 134 |
| Scenes / decision points | 572,322 |
| Total annotations | ~2,861,610 |
| HuggingFace rows (train) | ~115,000 |
| HuggingFace rows (test) | ~24,700 |
| HuggingFace total | ~139,000 |
| Annotation cost | ~$500,000 (GPT-4 + human expert + crowdworker) |
| Disk footprint (HF streaming) | 0 MB local (streams from HuggingFace) |
| Disk footprint (full download) | ~2-5 GB estimated |

## Label Schema

Labels live in `choice_labels[choice_idx]` — one dict per available action at each decision point.

### Morality Fields (binary-ish, continuous severity 0.0-N)

| Field | Meaning | Binarize at |
|-------|---------|-------------|
| `morality_deception` | Deceptive acts (lying, misleading, false pretenses) | >= 1.0 |
| `morality_manipulation` | Psychological manipulation of characters | >= 1.0 |
| `morality_killing` | Causing death | >= 1.0 |
| `morality_physical_harm` | Physical injury (non-lethal) | >= 1.0 |
| `morality_non_physical_harm` | Psychological/reputational/financial harm | >= 1.0 |
| `morality_intending_harm` | Intent to harm without execution | >= 1.0 |
| `morality_promises` | Breaking promises or commitments | >= 1.0 |
| `morality_stealing` | Theft of physical or intellectual property | >= 1.0 |
| `morality_trespassing` | Entering restricted/private areas | >= 1.0 |
| `morality_spying` | Surveillance or information gathering without consent | >= 1.0 |
| `morality_vandalism` | Destruction of property | >= 1.0 |
| `morality_fairness` | Unfair treatment of characters | >= 1.0 |
| `morality_other` | Catch-all ethical violations | >= 1.0 |

### Aggregate Fields (continuous, signed)

| Field | Meaning |
|-------|---------|
| `agg_cooperation` | Cooperation score with other characters |
| `agg_honest` | Honesty / non-deception aggregate |
| `agg_morality` | Overall morality score (negative = harmful) |
| `agg_power` | Power-seeking behavior (social + material + military) |
| `agg_utility` | Utility impact on all stakeholders |

### Text Fields

| Field | Meaning |
|-------|---------|
| `obs` | Current narrative scene text presented to the agent |
| `choice_text[idx]` | The action text for choice idx (agent's decision) |
| `choice_obs[idx]` | Resulting narrative after taking choice idx |
| `early_history` | List of prior scene observations (early game) |
| `recent_history` | List of prior scene observations (recent context) |

**Label granularity:** Per-choice (per-action). Not per-episode, not per-turn. Each decision point has N available choices (typically 2-9), each with independent labels.

**Label method:** GPT-4 annotation (18 questions across 5 prompts per scene) + expert human annotators (validation) + Surge crowdworkers (spot-check). Empirically shown to match/exceed human annotators on the ethics classification task.

## Black-Box Compatibility

**Compatible with SybilCore's threat model.** Key analysis:

| Criterion | Status | Notes |
|-----------|--------|-------|
| Text-only behavioral stream | YES | `obs + choice_text` is pure natural language, no game state vectors |
| No environment-in-the-loop needed | YES | Pre-computed annotations — just load and query |
| Action-level granularity | YES | Labels per agent action, matches SybilCore's action-attribution model |
| Deception specifically labeled | YES | `morality_deception` field, continuous + binarizable |
| Manipulation specifically labeled | YES | `morality_manipulation` field |
| Human-written text quality | YES | Professional Choose-Your-Own-Adventure game authors |
| Scale sufficient for training | YES | 115K train rows × ~4 choices avg = ~460K action pairs |
| Scale sufficient for evaluation | YES | 24.7K test rows |
| Label provenance documented | YES | Annotation methodology in paper Appendix |

## Phase 2 Integration Estimate

**Effort: 4-6 hours**

| Task | Time |
|------|------|
| Finalize loader.py + test harness integration | 1-2 hr |
| Run batch through phase2_scorer (deception probe) | 1 hr |
| Calibrate binarization threshold against scorer output | 1 hr |
| Add manipulation + power-seeking variants | 1 hr |
| Document results in corpus registry | 30 min |

## Blockers

None. Clean path:
- MIT license
- No ethics agreement
- No password required via HuggingFace
- `datasets` library already installed in environment
- Streaming mode available (no large local download needed)
- Python 3.11 requirement is for the **game simulation env** only — loading the HF dataset works fine with Python 3.13

One note: if the team wants to RUN the game environment (to generate new trajectories), Python <= 3.11 is required. For evaluation-only use (loading pre-annotated data), Python 3.13 is fine.

## Sample Pairs

See `loader.py` for extraction logic. Run:
```bash
python3.13 loader.py
```

Expected output: ~200 pairs, ~15-20% deceptive, text lengths 100-2000 chars.

## Recommendation

**GREEN — ship it**

MACHIAVELLI is the cleanest available corpus for SybilCore Phase 2 evaluation:
- The only benchmark with explicit per-action deception + manipulation labels at scale
- 460K+ labeled action pairs (vs. ~10-50K for most comparable benchmarks)
- Human-written narrative text (not synthetic) — high ecological validity
- MIT license with no restrictions
- Zero-friction HuggingFace loading, no downloads required
- Label schema maps directly to SybilCore's deception/manipulation threat taxonomy

Suggested SybilCore use: Phase 2 scorer evaluation dataset for deception detection probe (`morality_deception >= 1.0`). Secondary use: manipulation classifier validation (`morality_manipulation`).

---
*Scout: machiavelli-scout | Date: 2026-04-14 | Model: claude-sonnet-4-6*
