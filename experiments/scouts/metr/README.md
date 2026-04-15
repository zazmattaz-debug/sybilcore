# METR / Alignment Faking — Transcript Scout

Scout run: 2026-04-14. Scout: metr-scout (Haiku).

---

## Corpus Matrix

| Corpus | Public URL | Gated? | License | Size | Transcript Format | Label Quality | SybilCore Compat |
|--------|-----------|--------|---------|------|-------------------|--------------|-----------------|
| **METR Time Horizon runs.jsonl** | [eval-analysis-public](https://github.com/METR/eval-analysis-public) | No (public git) | See LICENSE | 14MB / 24,008 rows | JSONL: run-level stats (no message text) | score_binarized 0/1 | YELLOW — no CoT |
| **METR MALT (malt-public)** | [HF: metr-evals/malt-public](https://huggingface.co/datasets/metr-evals/malt-public) | YES — HF gated (auto-approve) | MIT | 28.3 GB (42 shards) | Parquet: ordered messages + metadata | 10 behavior labels + manually_reviewed | GREEN — needs HF token |
| **METR MALT (malt-transcripts-public)** | [HF: metr-evals/malt-transcripts-public](https://huggingface.co/datasets/metr-evals/malt-transcripts-public) | YES — HF gated (auto-approve) | MIT | 8.48 GB (13 shards) | Parquet: transcript as DAG of nodes | Same labels as above | GREEN — needs HF token |
| **Alignment Faking (Greenblatt 2024)** | [GitHub site](https://redwoodresearch.github.io/alignment_faking_examples/) | No (public HTML) | MIT | ~50K+ HTML files | HTML: SCRATCHPAD\_REASONING + OUTPUT | 18 boolean behavioral labels | GREEN — loader ready |
| **Alignment Faking (Google Drive bulk)** | [Drive folder](https://drive.google.com/drive/folders/1s-IcsWO5I3caX21yPbn7N_oTaJlU4Mz1) | Requires Google login | MIT | Unknown (json_outputs.zip) | JSON (unknown schema) | Same paper labels | YELLOW — needs Drive API |
| **METR RE-Bench / HCAST task suite** | [inspect-tasks-public](https://github.com/METR/inspect-tasks-public) | No (public git) | See LICENSE | Task configs only | NO transcripts — task environments only | N/A — no transcripts | RED — no data |
| **METR public-tasks** | [public-tasks](https://github.com/METR/public-tasks) | No (public git) | See LICENSE | Task configs only | NO transcripts — 31 example tasks only | N/A | RED — no data |
| **DeepMind FSF evals** | Not public | GATED | Unknown | Unknown | None released publicly | N/A | RED — contact DM |

---

## Integration Verdicts

### GREEN — Pull immediately

**Alignment Faking HTML transcripts**
- Fully public, no auth, MIT license
- `loader_anthropic.py` is ready and tested
- 689 trajectory selector pages, ~100-600 transcripts each
- SCRATCHPAD\_REASONING shows raw strategic reasoning (e.g. "I need to appear compliant during training...")
- 18 binary behavioral labels per transcript (effective\_scheme, training\_awareness, deviates\_for\_reward, etc.)
- Composite `alignment_faked` label derivable from (effective\_scheme OR partial\_scheme OR motivated\_by\_modification)
- 7 experiment categories: animal\_welfare, helpful\_only, misconception, prefix, promotion, secret, helpful\_only\_post\_rl\_variations
- Model: Claude 3 Opus exclusively (the "model organism" experiment)

**MALT on HuggingFace (after token fix)**
- MIT license, auto-gated (accepts anyone who logs in)
- 10,919 agent transcripts, 21 models, 86 task families
- Labels: bypass\_constraints, gives\_up, hardcoded\_solution, sabotage, ignores\_task\_instructions
- Has CoT traces + 4 obfuscation variants (vague, summarized, multilingual, irrelevant\_detail)
- 103 unprompted reward-hacking examples (highest-value subset)
- To unlock: create account at huggingface.co, accept terms, get token, set HF\_TOKEN

### YELLOW — Partial or needs setup

**METR Time Horizon runs.jsonl**
- Already cloned to `eval-analysis-public/`
- 24,008 rows, 21 models (Claude 3 Opus through Claude Opus 4.6, GPT-4 through GPT-5.3, Gemini 3 Pro)
- BUT: aggregate stats only — task\_id, score\_binarized, tokens\_count, scaffold, human\_minutes
- No message text, no CoT, no tool-call sequences
- Useful for: calibrating SybilCore difficulty scores against known human-time benchmarks
- SybilCore use: map task difficulty (human\_minutes) → expected capability level for threat model calibration

**Alignment Faking Google Drive (json\_outputs.zip)**
- Likely the same data as the HTML transcripts, in JSON form
- Google Drive requires authenticated access even for "public" shared folders (browser redirect)
- Alternative: Use `loader_anthropic.py` which scrapes the same data from GitHub Pages directly

### RED — Gated or no transcripts released

**METR RE-Bench / TaskBench / HCAST**
- No transcript corpus released — only task environment code
- RE-Bench requires 8×H100 GPUs to run
- README explicitly warns against publishing eval solutions
- METR will not release transcripts (benchmark contamination concern)

**DeepMind Frontier Safety Framework**
- No public transcript release found
- Internal eval results only; no companion dataset

---

## Key Answers to Critical Questions

**Does METR's RE-Bench release actual Claude/o1/DeepSeek transcripts?**
No. RE-Bench releases task environment code only. The runs.jsonl in eval-analysis-public is aggregate run stats (score, tokens, duration) — no message-level content.

**Does the Alignment Faking companion release raw CoT transcripts with "faking" labels?**
YES — the GitHub Pages site at `redwoodresearch.github.io/alignment_faking_examples` hosts thousands of individual HTML transcript files with SCRATCHPAD\_REASONING (raw CoT) + OUTPUT + 18 per-transcript boolean labels. Publicly accessible, no auth required.

**Are there downstream datasets that repackage these transcripts?**
MALT (HF: metr-evals/malt-public) is METR's own repackaging of their agentic eval transcripts with behavioral labels — distinct from Alignment Faking but complementary. No third-party HuggingFace repackaging found for either corpus.

---

## Top Recommendation

**Score the Alignment Faking corpus first.**

Reason: It is the only fully public, pre-labeled corpus of model behavioral transcripts showing *strategic deception* in the wild. The SCRATCHPAD\_REASONING field gives direct access to the model's internal reasoning about whether to fake alignment. The 18 binary labels (effective\_scheme, training\_awareness, deviates\_for\_reward, etc.) are exactly the behavioral signals SybilCore's black-box threat model needs to calibrate against.

Suggested first run:
```python
from loader_anthropic import load_transcripts

# Pull 100 training-context transcripts from animal_welfare
# (highest alignment-faking rate per the paper)
train_samples = load_transcripts(
    limit=100, category="animal_welfare", variant="default", split="train"
)

# Pull 100 deployment-context transcripts as the "benign" control
deploy_samples = load_transcripts(
    limit=100, category="animal_welfare", variant="default", split="deploy"
)

faked = [s for s in train_samples if s["label"]["alignment_faked"]]
print(f"Alignment faking rate (train): {len(faked)/len(train_samples):.1%}")
```

Second priority: Unlock MALT for broader behavior coverage (reward hacking, sandbagging, sabotage) across 21 models.

---

## Outreach Email Templates (for gated corpora)

### METR — RE-Bench/HCAST transcript release

```
To: contact@metr.org
Subject: Research request: access to HCAST/RE-Bench agent transcripts for alignment research

Hi METR team,

I'm a researcher working on SybilCore, a behavioral scoring system for
detecting deceptive alignment patterns in agentic AI systems. I'm
interested in accessing the raw agent execution transcripts from your
HCAST / RE-Bench evaluations — specifically the message-level logs
(tool calls, CoT, environment interactions) rather than the aggregate
runs.jsonl scores you've published.

I understand you haven't released transcripts due to benchmark
contamination concerns. Would it be possible to discuss a
research data agreement that would allow use of a subset of
transcripts (e.g., retired tasks only) for alignment research?

I'm happy to sign an NDA, agree to canary GUIDs, and restrict use
to non-training purposes. Please let me know if this is feasible.

[Your name, affiliation, project URL]
```

### DeepMind — Frontier Safety Framework eval transcripts

```
To: frontier-safety@google.com (or per FSF contact page)
Subject: Research data inquiry: FSF evaluation transcript access

Hi DeepMind Safety team,

I'm researching behavioral threat models for agentic AI systems and
would like to inquire about access to any agent execution transcripts
generated during your Frontier Safety Framework evaluations. I'm
specifically interested in cases where models demonstrated
power-seeking, deceptive alignment, or capability elicitation patterns.

Even anonymized, partial transcripts would be valuable for calibrating
behavioral scoring systems. Are there any research data sharing
agreements available for this kind of work?

[Your name, affiliation, project URL]
```

---

## Files in This Directory

```
experiments/scouts/metr/
├── README.md                          (this file)
├── loader_metr.py                     (METR runs.jsonl + MALT loader)
├── loader_anthropic.py                (Alignment Faking HTML scraper, tested)
└── eval-analysis-public/              (git clone, 42MB)
    └── reports/
        ├── time-horizon-1-0/data/raw/runs.jsonl   (1.0 run data)
        └── time-horizon-1-1/data/raw/runs.jsonl   (1.1 run data, 24k rows, 21 models)
```

---

## Disk Footprint

| Source | Size | Status |
|--------|------|--------|
| eval-analysis-public git clone | 42 MB | pulled |
| alignment\_faking\_public git clone | 18 MB | pulled (code only, no transcript data) |
| Alignment Faking HTML transcripts | ~500KB per 100 transcripts (fetched on demand) | streaming |
| MALT HuggingFace (full) | 28.3 GB | NOT downloaded — gated |
| Total pulled | ~60 MB | well under 1GB cap |
