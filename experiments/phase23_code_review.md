# Phase 2/3 Code Review — Adversarial Punchlist

**Reviewer:** codex-reviewer (ov-reviewer agent, Sonnet 4.6 + o4-mini second-brain)
**Date:** 2026-04-14
**Files reviewed:**
- `experiments/run_phase2_scorer_A.py` ✅ reviewed
- `experiments/run_phase2_scorer_B.py` ✅ reviewed
- `experiments/run_phase3_ablation.py` ⏳ not yet written — review pending on delivery

---

## CRITICAL BLOCKERS (must fix before any training run)

### [CRITICAL] run_phase2_scorer_B.py:245 — Label inversion in load_bigbench_truthqa

**Description:** `load_bigbench_truthqa` assigns `label = 1` to the *correct/honest* answer (BIG-Bench's "target_score=1 = correct"). The unified schema across all Phase 2 datasets requires `label=1 = deceptive/dishonest`, `label=0 = honest/truthful`. Scorer A uses this convention correctly (Best Answer→0, Best Incorrect Answer→1). Scorer B's bigbench_truthqa loader is inverted. When Phase 3 concatenates all JSONL files into one training matrix, the bigbench_truthqa rows will have flipped polarity, poisoning the classifier signal.

**Docstring confirms the inversion:** "Label = target_score (1=correct, 0=incorrect)" — "correct" in BIG-Bench truthful_qa means the *honest* answer.

**Suggested fix:** In `load_bigbench_truthqa`, invert the label assignment:
```python
label = 0 if int(score) == 1 else 1  # BIG-Bench "correct" = honest = label 0
```

---

### [CRITICAL] run_phase2_scorer_B.py:285 — Label inversion in load_bigbench_hhh

**Description:** `load_bigbench_hhh` assigns `label=1` to honest responses, `label=0` to evasive ones. Docstring: "Label = target_score (1=honest, 0=evasive)". This is again the opposite of the unified convention. HHH-alignment "honest" responses should be label=0.

**Suggested fix:**
```python
pairs_raw.append((text, 0 if int(score) == 1 else 1))  # honest=0, evasive=1
```

---

### [CRITICAL] run_phase2_scorer_B.py:42 — Wrong data directory for BIG-Bench HHH

**Description:** `BIGBENCH_HHH_FILE` is set to `"phase2_data/bigbench_tasks/deceptive_alignment/task.json"` but the module docstring, function docstring, and print statement all say `hhh_alignment/honest`. This will either crash at runtime (FileNotFoundError if the path doesn't exist) or silently load the *deceptive_alignment* dataset instead of *hhh_alignment*, producing entirely wrong data for the "honest" BIG-Bench task.

**Suggested fix:**
```python
BIGBENCH_HHH_FILE = EXPERIMENTS / "phase2_data/bigbench_tasks/hhh_alignment/task.json"
```

---

## HIGH PRIORITY (fix before interpreting results)

### [HIGH] run_phase2_scorer_A.py:328-329 / run_phase2_scorer_B.py:218-219 — AUROC direction not normalized

**Description:** Both scorers compute and report raw AUROC without normalizing values below 0.5. If emotion_score is anti-correlated with deception (Phase 1 found AUROC=0.383, meaning the score is *lower* for deceptive text), the raw AUROC will be reported as ~0.38 and interpreted as "poor discrimination" when it actually represents AUROC=0.617 in the correct direction. Anyone reading the summary JSON will draw the wrong conclusion about the feature's utility.

The script has a comment acknowledging this ("Phase 1 found 0.383") but does nothing to normalize.

**Suggested fix:** After computing AUROC, symmetrize:
```python
def auroc_symmetric(scores, labels):
    raw = auroc(scores, labels)
    return max(raw, 1 - raw)  # always report distance from 0.5
```
And log the direction (inverted or not) in the summary JSON.

---

### [HIGH] run_phase2_scorer_A.py:136 — Dead variable `sampled` in load_truthfulqa

**Description:** Line 136 builds `sampled = truthful[:n_per_class] + deceptive[:n_per_class]` but this variable is never used. The function returns `interleaved` (built on lines 138-141). The `sampled` list is an O(n) allocation that silently does nothing. This is dead code that creates confusion about which list is actually returned.

**Suggested fix:** Delete lines 135-136.

---

### [HIGH] run_phase2_scorer_A.py:293 — `n_errors` can go negative on retry success

**Description:** When a retry succeeds after an initial error, line 293 executes `n_errors -= 1`. But `n_errors` tracks *total* errors across all items. If this is the very first error and the retry succeeds, `n_errors` was incremented to 1 then decremented to 0 — that is correct. However if the intent is that `n_errors` counts *permanent* failures (not temporary ones), the decrement logic should apply more carefully. More importantly, there's no `max(0, ...)` guard — in theory if the code path were reached with `n_errors=0` (which shouldn't happen but could if logic changes), it would go negative, corrupting the summary count.

**Suggested fix:**
```python
n_errors = max(0, n_errors - 1)
```

---

### [HIGH] run_phase2_scorer_A.py:305 — Progress counter incorrect when resuming

**Description:** Line 305 computes `processed = (idx + 1) - len(existing)`. This assumes all `idx` values from 0 to `idx` are either in `existing` or were just scored, but `idx` is the position in the full `items` list (0-based), not a count of items processed so far. When some items are skipped (via `continue`), this computation is wrong — it can go negative or give inflated counts.

**Suggested fix:** Maintain an explicit counter:
```python
n_processed = 0  # initialize before loop
# inside loop, after successful score:
n_processed += 1
if n_processed % SAVE_EVERY == 0:
    # use n_processed for progress display
```

---

## MEDIUM PRIORITY (should fix, affects reliability)

### [MEDIUM] run_phase2_scorer_A.py:115 / run_phase2_scorer_B.py:242,282,310 — No empty-dataset guard

**Description:** None of the dataset loaders (`load_truthfulqa`, `load_deceptionbench`, `load_bigbench_truthqa`, `load_bigbench_hhh`, `load_sycophancy`) check for an empty result before returning. If a data file is missing, malformed, or contains zero usable rows, the caller receives an empty list, and the scoring loop produces zero records. The AUROC functions return 0.5 (their default), and the summary JSON shows `n_scored=0` with no error. This silent failure is dangerous in an experiment context.

**Suggested fix:** After loading, in each loader:
```python
if not interleaved:
    raise ValueError(f"[{dataset_name}] Zero items loaded — check data file path and format")
```

---

### [MEDIUM] run_phase2_scorer_B.py:93 — Shadow of built-in `l` in auroc()

**Description:** The list comprehensions use `l` as the loop variable name (`for s, l in zip(...)`). This shadows the built-in function `l` (there isn't one in Python, but `l` is visually confusable with `1` and `I`). More concretely, the variable name is unclear about what it represents.

**Suggested fix:** Rename to `lbl`:
```python
pos = [s for s, lbl in zip(scores, labels) if lbl == 1]
neg = [s for s, lbl in zip(scores, labels) if lbl == 0]
```

---

### [MEDIUM] run_phase2_scorer_A.py:O(n²) AUROC / run_phase2_scorer_B.py:O(n²) AUROC — Quadratic complexity

**Description:** Both manual AUROC implementations use a double loop (`sum(1 for p in pos for n in neg if ...)`). With 400 items per dataset (200 pos + 200 neg), this is 40,000 comparisons — acceptable. But if the dataset were scaled (e.g., phase3 combining 5 datasets × 400 items), the final AUROC call on all_records would be O((1000)²) = 1M comparisons. Not a current blocker but worth noting.

**Suggested fix (if scaling):** Use `sklearn.metrics.roc_auc_score(labels, scores)` — O(n log n).

---

### [MEDIUM] run_phase2_scorer_B.py — No --dry-run mode

**Description:** Scorer A has a `--dry-run` flag that validates dataset loading and script structure without hitting the Modal endpoint. Scorer B has no equivalent. This makes it harder to validate paths and data integrity before committing to a 2h+ run.

**Suggested fix:** Add `--dry-run` flag that runs all loaders and prints counts, then exits 0 without calling `score_text`.

---

## LOW PRIORITY (observations, not blockers)

### [LOW] run_phase2_scorer_B.py:70 — MD5 used for text hashing

**Description:** `text_hash` uses `hashlib.md5()`. MD5 is cryptographically broken (collision attacks exist). For a content-addressed dedup key with no security implications, this is acceptable in practice, but it differs from Scorer A which uses SHA-256. Cross-script JSONL joins by `text_hash` will produce different hash values for the same text.

**Suggested fix:** Use `hashlib.sha256()` in both scripts for consistency. Scorer A already does this correctly.

---

### [LOW] run_phase2_scorer_A.py:266 / run_phase2_scorer_B.py:191 — Bare `except Exception` without re-raise

**Description:** Both scripts use broad `except Exception` blocks in the scoring loop. This is intentional (continue-on-error is the design), but exceptions from bugs (e.g., KeyError from a schema change) are silently counted as endpoint errors and retried. In debugging, this masks real bugs as "endpoint failures."

**Suggested fix (optional):** At minimum, distinguish `urllib.error.URLError` (network) from other exceptions:
```python
except urllib.error.URLError as exc:
    # endpoint error — retry
except (KeyError, ValueError, TypeError) as exc:
    # schema error — log and skip, don't retry
    print(f"SCHEMA ERROR at idx={idx}: {exc}", flush=True)
```

---

### [LOW] run_phase2_scorer_A.py — `n_scored` in summary double-counts on resume

**Description:** Line 348 returns `"n_scored": n_scored + len(existing)`. This reports the total across all runs (new + resumed), which is the correct intent. However, `n_scored` counts *newly scored in this run* (not total). The naming is slightly misleading but the math is correct.

No fix required — but the variable name `n_scored` in the return value already means "total including resumed" which is consistent with how `score_dataset` documents it.

---

## Phase 3 Status

`experiments/run_phase3_ablation.py` was not present at review time (polling for 40 min; file not yet written by phase3-ablation agent). Review of Phase 3 is pending. Key things to verify in Phase 3 when it arrives:

1. **CV leakage**: StratifiedKFold must fit vectorizer/scaler on *train fold only*, predict on *test fold*. No `fit_transform` on the full dataset before splitting.
2. **Label schema**: Must apply `label=1=deceptive` convention. If it ingests Scorer B's bigbench_truthqa or hhh JSONL directly (without the label fix), it will train on corrupted labels.
3. **Region activations key ordering**: When flattening `region_activations` dicts to feature vectors, keys must be sorted or use a fixed union-of-keys vocabulary. JSON deserialization preserves insertion order in Python 3.7+ but dict key sets can differ across texts if any region is missing.
4. **class_weight='balanced'**: LogisticRegression and LinearSVC should use `class_weight='balanced'` if any dataset has class imbalance.
5. **AUROC direction**: `roc_auc_score(y_true, y_score)` — `y_score` must be probability/score for the *positive class* (label=1=deceptive). If passing raw emotion_score which is anti-correlated, must negate or use `1-score`.

---

## Summary by Severity

| Severity | Count | Files |
|----------|-------|-------|
| CRITICAL | 3 | scorer_B (label inversions ×2, wrong file path) |
| HIGH | 4 | scorer_A (dead var, n_errors, progress counter), scorer_A+B (AUROC direction) |
| MEDIUM | 5 | both (empty guard, shadow var, O(n²), dry-run, bare except) |
| LOW | 3 | both (MD5, bare except distinction, n_scored naming) |
| PENDING | 5 | phase3 (CV leakage, key ordering, class_weight, AUROC, label ingestion) |

**Total found: 15 issues (3 CRITICAL, 4 HIGH, 5 MEDIUM, 3 LOW) + 5 phase3 checks pending**

---

## Top 3 Critical Issues

1. **[CRITICAL] scorer_B:42** — Wrong file path for HHH dataset. `deceptive_alignment/task.json` will either crash or load the wrong dataset entirely. Fix: change to `hhh_alignment/task.json`.

2. **[CRITICAL] scorer_B:245** — `load_bigbench_truthqa` labels honest answers as `label=1`. Inverted from all other datasets. Will corrupt classifier when Phase 3 concatenates all JSONL. Fix: `label = 0 if int(score) == 1 else 1`.

3. **[CRITICAL] scorer_B:285** — `load_bigbench_hhh` labels honest responses as `label=1`. Same inversion as above. Fix: same inversion in `pairs_raw.append`.

**These three must be fixed before the scoring runs. If scorer_B data is already collected with wrong labels, the JSONL output files must be regenerated after the fix.**


---

## Phase 3 Review — run_phase3_ablation.py

File arrived: 2026-04-14 15:20. Review completed.

### [CRITICAL] run_phase3_ablation.py:584 — CV Leakage in B5 TRIBE local features

**Description:** `tribe_feats_cornell = _tribe_features(texts)` is called on the FULL corpus before the CV loop begins. `_tribe_features` internally calls `module.load_or_build_centroids()` to construct a centroid representation, which operates over all input texts — including the test fold. This means when `b5_predict_closure` looks up features for test-fold texts using `_get_tribe_feats_for_texts`, those features were computed with centroid information derived from the entire corpus, giving the classifier illegitimate access to test-fold statistical properties.

This is standard pre-computation leakage. It inflates B5 AUROC estimates and invalidates the B5 vs TRIBE comparison.

**Suggested fix:** Move TRIBE feature computation inside the CV loop:
```python
for train_idx, test_idx in fold_iter:
    X_train = [texts[i] for i in train_idx]
    X_test = [texts[i] for i in test_idx]
    # Fit centroids on train only:
    centroids = module.build_centroids(X_train)
    train_feats = module.score_texts(X_train, centroids)
    test_feats = module.score_texts(X_test, centroids)
    ...
```

Note: B1 TF-IDF is correctly implemented via sklearn Pipeline — `pipe.fit(X_train)` is called inside `b1_fit` which only receives `X_train`. No leakage in B1.

Note: B2 and B3 use no fitting-on-corpus steps (SBERT is pre-trained, LIWC features are deterministic). No leakage in B2/B3.

---

### [HIGH] run_phase3_ablation.py:782 — Verdict passes when B5 is unavailable (spec violation)

**Description:** The pass condition at line 782 is:
```python
if b1_delta > 0.05 and (b5_delta is None or b5_delta > 0.03) and abs(b2_delta) <= 0.05:
    verdict = "pass"
```

When `b5_delta is None` (B5 skipped because TRIBE scorer path not found), the B5 check is trivially True. Per spec §2.3.3, "TRIBE > B5 by Δ > 0.03" is a required pass condition. Silently treating a skipped B5 as a met condition gives a false "pass" verdict.

**Suggested fix:**
```python
b5_condition = (b5_delta is not None and b5_delta > 0.03)
if b1_delta > 0.05 and b5_condition and abs(b2_delta) <= 0.05:
    verdict = "pass"
elif b1_delta > 0.05 and b5_delta is None and abs(b2_delta) <= 0.05:
    verdict = "marginal"  # Can't confirm B5 condition — inconclusive
```

---

### [HIGH] run_phase3_ablation.py:84 — Hardcoded absolute path

**Description:** `_ROOMS_ROOT = Path("/Users/zazumoloi/Desktop/Claude Code/rooms")` is hardcoded at line 84. This will crash on any other machine, any CI environment, or if the user moves the workspace. The earlier `REPO_ROOT` derivation (line 60) is the correct pattern and should be used instead.

**Suggested fix:**
```python
# TRIBE_SCORER_PATH should derive from REPO_ROOT, not a hardcoded path
TRIBE_SCORER_PATH = REPO_ROOT.parent.parent / "research" / "tribe-v2" / "pipeline" / "local_resonance_scorer.py"
```
(Adjust the number of `.parent` steps to match actual directory depth.)

---

### [MEDIUM] run_phase3_ablation.py:281,367,465,486 — class_weight=None in LogReg on potentially imbalanced Phase 2 data

**Description:** All `LogisticRegression` instantiations use default `class_weight=None`. Cornell corpus is balanced (400/400) so this is fine for Cornell. However, Phase 2 datasets (TruthfulQA, DeceptionBench, BIG-Bench, Sycophancy) may have class imbalance depending on sampling. When `load_phase2_data()` returns and `run_cv()` is called on `p2_texts/p2_labels`, the classifier is not corrected for imbalance.

**Suggested fix:** Set `class_weight='balanced'` in all four LogReg instantiations, or document why it is explicitly not needed.

---

### [MEDIUM] run_phase3_ablation.py:148-151 — FPR@95TPR iteration order is implicit

**Description:** The FPR@95TPR loop iterates `zip(fpr_arr, tpr_arr)` forward and breaks at first `tpr_val >= 0.95`. sklearn's `roc_curve` returns arrays in ascending FPR/TPR order, so the first `tpr >= 0.95` is the minimum FPR meeting 95% recall. The behavior is correct, but the intent is not documented and relies on undocumented sklearn ordering guarantees.

**Suggested fix:** Use explicit numpy indexing to make intent clear:
```python
mask = tpr_arr >= 0.95
fpr_at_95 = float(fpr_arr[mask][0]) if mask.any() else float("nan")
```

---

### [LOW] run_phase3_ablation.py:505 — Phase 2 label schema trust gap

**Description:** `load_phase2_data()` trusts the `label` field from phase2_*_results.jsonl files verbatim. If Scorer B's bigbench_truthqa and hhh JSONL files have inverted labels (as identified in the Scorer B review above), Phase 3 will silently ingest corrupted labels and train on them. Phase 3 itself cannot detect this.

**No fix in Phase 3** — this is downstream of the Scorer B label inversion bugs. Fix Scorer B first and regenerate JSONL. But add a validation comment noting the dependency:
```python
# IMPORTANT: phase2 JSONL label schema must be: 1=deceptive, 0=honest.
# See phase23_code_review.md re: Scorer B label inversion bugs in bigbench_truthqa and hhh.
```

---

## Updated Summary — All Three Files

| Severity | Count | Sources |
|----------|-------|---------|
| CRITICAL | 4 | scorer_B ×3, (phase3 leakage also critical) |
| HIGH | 6 | scorer_A ×3, scorer_B ×1, phase3 ×2 |
| MEDIUM | 8 | scorer_A ×2, scorer_B ×2, phase3 ×2 + 2 carryover |
| LOW | 4 | all files |
| **TOTAL** | **22** | |

### Top 3 Critical Issues (unchanged — must fix before any training run)

1. **[CRITICAL] scorer_B:42** — Wrong file path `deceptive_alignment/task.json` for HHH dataset. Fix: `hhh_alignment/task.json`.
2. **[CRITICAL] scorer_B:245** — `load_bigbench_truthqa` labels honest=1, deceptive=0. Inverted. Fix: `label = 0 if int(score) == 1 else 1`.
3. **[CRITICAL] scorer_B:285** — `load_bigbench_hhh` labels honest=1. Inverted. Fix: same inversion.

**Additional Critical:** phase3:584 — B5 TRIBE features pre-computed on full corpus before CV split — test-fold leakage inflating B5 AUROC.

