# Calibration v4 vs v5: Synthetic vs Real Moltbook Negatives

**Date:** 2026-04-07
**Question:** Does the v4 brain-weight calibration generalize when the
negative class is swapped from synthetic clean rounds to raw Moltbook
event streams?
**Verdict (TL;DR):** v4 weights are a **synthetic-corpus artifact**.
They still beat the default baseline on real data (F1 0.90) but are
materially worse than a fresh v5 calibration (F1 0.98), and three
weights flip sign in a way that is not consistent with a stable
optimum. **Do not ship v4 as the production default yet.** Ship v5 or
hold until a larger real-negative run can confirm these numbers.

## Run configuration

| | v4 (synthetic) | v5 (real Moltbook) |
|---|---|---|
| Positives | 103 (alignment scenarios + archetypes) | 103 (same) |
| Negatives | 100 synthetic `_clean_round` | 500 real Moltbook authors (≥2 posts) |
| Brain ensemble | 15 brains | 15 brains (same set) |
| Search budget | 50 random + 25 local | 50 random + 25 local |
| FPR cap | 0.05 | 0.05 |
| Seed | 42 | 42 |
| Precompute time | — | 35s |
| Total runtime | — | ~77s |

## Headline metrics

| Metric | v4 optimal (synthetic) | v5 optimal (real Moltbook) | Δ (v5 − v4) |
|---|---:|---:|---:|
| F1 | 0.9119 | **0.9810** | +0.0691 |
| Precision | 0.9462 | 0.9626 | +0.0164 |
| Recall | 0.8800 | **1.0000** | +0.1200 |
| AUC | 0.9728 | 0.9982 | +0.0254 |
| FPR | 0.0500 | **0.0080** | −0.0420 |
| TP / FP / TN / FN | 88 / 5 / 95 / 12 | 103 / 4 / 496 / 0 | — |
| Best threshold | 45.0 | 55.0 | +10.0 |

The v5 optimum is better on every metric, *on its own corpus*. That is
expected — v5 is tuned against v5 negatives. The interesting question
is the next section.

## How do v4 weights perform on the real Moltbook corpus?

We replayed v4's exact weight vector through the v5 corpus (same
positives + 500 real negatives) with a fresh threshold sweep.

| Metric | v4 weights on v5 corpus | v5 optimal on v5 corpus | v5 − v4-on-v5 |
|---|---:|---:|---:|
| F1 | 0.9005 | 0.9810 | +0.0805 |
| Precision | 0.8796 | 0.9626 | +0.0830 |
| Recall | 0.9223 | 1.0000 | +0.0777 |
| AUC | 0.9875 | 0.9982 | +0.0107 |
| FPR | 0.0260 | 0.0080 | −0.0180 |
| Best threshold | 40.0 | 55.0 | +15.0 |

### Observations

1. **v4 weights don't catastrophically fail on real data** — they
   still hit F1 0.90 and AUC 0.989. So the v4 calibration is not
   garbage; it captures *something* generalizable.
2. **But they miss 8 positives** on the real corpus where v5 catches
   all of them (recall 0.92 vs 1.00). The v4 weight over-fit reduced
   sensitivity by ~8 percentage points.
3. **Real negatives are *easier* to separate than the synthetic ones.**
   On v4's synthetic corpus the best v4 FPR was 0.05 at threshold 45;
   on the v5 real corpus the same weights get FPR 0.026 at threshold
   40. Real Moltbook posts look *less* like positives than the
   synthetic clean rounds do — so the synthetic negatives were
   actually the *harder* distribution. This is consistent with the
   known fact that 2,677 real agents all scored CLEAR.
4. **The best threshold shifted by 15 points upward** (40 → 55)
   between v4-on-v5 and v5-optimal. That alone is a red flag: if the
   v4 weight vector were actually optimal, the optimal threshold
   shouldn't depend this much on whether the negatives are synthetic
   or real.

## Side-by-side weight vectors (sorted by |Δ|)

| Brain | v4 optimal | v5 optimal | Δ (v5 − v4) | Interpretation |
|---|---:|---:|---:|---|
| fidelity           | 0.184 | **2.846** | **+2.662** | v4 pruned it; v5 maxes it. |
| compromise         | 2.575 | **0.102** | **−2.473** | v4 loved it; v5 essentially removes it. |
| semantic           | 2.865 | 1.040     | −1.825 | v4 leaned on it; v5 neutral. |
| resource_hoarding  | 0.117 | 1.664     | +1.547 | v4 pruned it; v5 re-inflates. |
| contrastive        | 1.364 | 2.646     | +1.282 | Stronger in v5. |
| social_graph       | 1.120 | 2.358     | +1.238 | Much stronger in v5. |
| swarm_detection    | 1.315 | 0.156     | −1.159 | v5 demotes. |
| deception          | 1.203 | 2.311     | +1.108 | v5 roughly doubles. |
| economic           | 1.684 | 2.648     | +0.964 | Stronger in v5. |
| silence            | 1.075 | 0.348     | −0.727 | v5 demotes. |
| identity           | 1.849 | 2.512     | +0.663 | Slightly stronger. |
| intent_drift       | 2.285 | 1.638     | −0.647 | v5 trims. |
| temporal           | 2.268 | 2.794     | +0.526 | Both corpora agree: high. |
| embedding          | 0.736 | 0.268     | −0.468 | Both corpora agree: prune. |
| neuro              | 0.740 | 0.992     | +0.252 | Effectively unchanged. |

### Top 3 weight changes (by absolute Δ)

1. **fidelity** 0.18 → 2.85 (+2.66). v4 treated fidelity as almost
   useless; v5 pushes it to the weight ceiling. Fidelity fires on
   cover-story / narrative-consistency breaks — it turns out to be
   the single strongest separator between real Moltbook posts and
   adversarial archetypes. v4 missed this because the synthetic clean
   rounds all looked fidelity-neutral.
2. **compromise** 2.58 → 0.10 (−2.47). v4 leaned very hard on
   compromise; v5 near-zeroes it. Compromise was over-weighted on
   synthetic negatives because the archetypes heavily embed
   compromise-like artifacts and the clean rounds don't, making it an
   artificially clean separator. On real Moltbook it fires on noise
   in benign posts, so the optimizer dials it down.
3. **semantic** 2.87 → 1.04 (−1.83). Semantic was the #1 weight in v4
   at the ceiling; v5 collapses it to neutral. This mirrors the
   compromise story — semantic distance is a useful signal on
   narrow synthetic negatives but is much noisier on real organic
   text.

### Signs of stability (agreement between corpora)

- **temporal** (2.27 → 2.79) — stays high on both corpora. Likely a
  genuine positive signal.
- **embedding** (0.74 → 0.27) — stays low on both. Consistent with
  the v4 decision to exclude it from the default ensemble.
- **deception**, **identity**, **intent_drift** — all stay well above
  neutral on both runs. These are the brains the adversarial tournament
  also flagged as hardest to evade.

### Signs of over-fit (disagreement between corpora)

- fidelity, compromise, semantic all flip by more than 1.8 weight
  units. Any weight vector whose top-3 signals all disagree with a
  second calibration is almost by definition a local optimum, not a
  generalizable minimum.
- Four of the top six v4 weights (compromise, semantic,
  swarm_detection, silence) all get *downgraded* on real negatives.
  The reverse direction — v5 adds weight to fidelity, contrastive,
  economic, social_graph — suggests these brains were under-weighted
  because the synthetic clean rounds didn't challenge them.

## Interpretation

The v4 calibration is **partially generalizable and partially a
synthetic-corpus artifact**. It identifies the correct "high-priority
axis" (temporal, deception, identity, intent_drift, embedding-prune)
but it mis-weights the mid-tier brains because the synthetic clean
rounds lack the organic variance that fidelity, contrastive, and
economic brains need to calibrate against.

Concretely:
- On *synthetic* negatives, v4 weights get F1 0.91 (their true
  training metric).
- On *real* negatives, v4 weights degrade to F1 0.90 and recall drops
  from 0.88 to 0.92 — not catastrophic, but clearly sub-optimal.
- A fresh v5 calibration on real negatives gets F1 0.98 and recall
  1.00 with half the FPR. The delta is real and the *same budget*
  (50 random + 25 local) finds it.

## Recommendation

**Do NOT ship the v4 weights as production defaults.**

Two viable paths forward:

### Option A: Ship v5 weights now (fast path)

Use the v5 optimal vector. Pros: better on every metric, tuned on
real negatives, runs with identical budget. Cons: 500 real negatives
is still a modest sample — we're one corpus away from knowing if v5
itself is stable.

### Option B: Run a larger v5.1 before shipping (safer)

Re-run the calibration with (1) the full 1,297 multi-post Moltbook
authors as negatives, (2) a doubled search budget (100 random + 50
local), and (3) a held-out split (80/20) so we can report out-of-sample
F1 instead of training F1. Estimated runtime: 15-25 min. This would
give us a defensible number for the paper.

**My recommendation: do Option B before flipping defaults.**
v5's F1 = 0.98 is training-set F1 with no held-out set. The v4 → v5
result already proves the calibration is sensitive to the negative
sample; we should measure that sensitivity properly before locking
weights into production. The compute cost is trivial and the
NeurIPS-grade defense is worth it.

If the user wants to ship immediately anyway, Option A (v5 weights)
is strictly better than Option A' (v4 weights) on the real corpus.

## Files

- `experiments/calibration_v4_20260407_101019.json` — v4 run (synthetic)
- `experiments/calibration_v5_real_moltbook.json` — v5 run (real)
- `scripts/recalibrate_v5_real_moltbook.py` — v5 runner
