# Agent-SafetyBench Scout Report

## Paper
**Title:** Agent-SafetyBench: Evaluating the Safety of LLM Agents
**Authors:** Zhexin Zhang, Shiyao Cui, Yida Lu, Jingzhuo Zhou, Junxiao Yang, Hongning Wang, Minlie Huang
**arXiv:** https://arxiv.org/abs/2412.14470 (Dec 2024, revised May 2025)
**Venue:** arXiv preprint (no conference acceptance found at time of scouting)
**License:** MIT

## Dataset
- **Source:** https://github.com/thu-coai/Agent-SafetyBench + https://huggingface.co/datasets/thu-coai/Agent-SafetyBench
- **Size:** 2,000 test cases across 349 environments
- **Baseline results:** 16 models tested, none above 60% safety score

## Schema

Each item in `released_data.json`:
```
{
  "id": int,
  "instruction": str,          <- the agent's adversarial task prompt (TEXT FIELD)
  "environments": [...],       <- tool specs for 1+ environments
  "failure_modes": "1,5",      <- comma-separated ints 1-10 (see taxonomy)
  "risks": [...],              <- list of risk category strings (see below)
  "fulfillable": 0 | 1,        <- 0 = task must be refused (UNSAFE), 1 = safe to do
  "dialog": [...] | null       <- partial agent trace (only 111/2000 items)
}
```

## Label Schema

### Binary Safety Label
`fulfillable==0` -> **UNSAFE** (label=1): agent must refuse or the task is harmful
`fulfillable==1` -> **SAFE** (label=0): task can be completed appropriately

Distribution: 1,265 unsafe (63.2%) / 735 safe (36.8%)

### 10 Failure Mode Taxonomy
| Code | Description |
|------|-------------|
| 1    | Generating harmful content without invoking tool calls |
| 2    | Calling tools when necessary information is incomplete |
| 3    | Calling tools before obtaining complete information |
| 4    | Ignoring known constraint information and incorrectly calling tools |
| 5    | Ignoring implicit/potential risks and incorrectly calling tools |
| 6    | Using incorrect parameters when calling tools |
| 7    | Ignoring safety issues with a tool and calling it anyway |
| 8    | Failing to call necessary tools (over-refusal / under-action) |
| 9    | Over-trusting tool results without validation |
| 10   | Failing to filter unsafe results returned by tools |

### 8 Risk Categories
1. Compromise availability
2. Contribute to harmful / vulnerable code
3. Lead to physical harm
4. Lead to property loss
5. Leak sensitive data / information
6. Produce unsafe information / misinformation
7. Spread unsafe information / misinformation
8. Violate law or ethics / damage society

## Pre-Computed Responses: NO (YELLOW)

The repo ships **only prompts + ground truth labels** — no model responses.

To get actual model responses you must:
1. Run `evaluation/eval.py` against a model API (OpenRouter, OpenAI, Anthropic, etc.)
2. Score the outputs with the ShieldAgent model (`score/eval_with_shield.py`) — a local LLM that classifies each response as safe/unsafe.

The 111 items that have a `dialog` field contain partial traces (user + first tool call + tool result) but NO final assistant response and NO safety annotation from a human judge.

**Implication for SybilCore:** Without pre-computed responses, we cannot directly run SybilCore's text-based scorer against actual agent outputs. What IS available:
- `instruction` text as the input surface (prompt-level detection, not response-level)
- Binary `fulfillable` label as ground truth
- 2,000 labeled adversarial prompt examples spanning 349 environments

## Black-Box Compatibility

SybilCore can score the `instruction` text against the `fulfillable` binary label. This is a valid but different task than MACHIAVELLI (which scores game transcripts) — here we detect unsafe task intent in the prompt itself, not unsafe behavior in a response.

This is **prompt-level threat detection**, not **response-level safety scoring**.
Both are valid SybilCore use cases but they test different capabilities.

## Effort Estimate to Integrate into phase2_scorer

| Task | Effort | Notes |
|------|--------|-------|
| Prompt-level scoring (current loader) | 1-2h | loader.py already done, wire into phase2_scorer |
| Response-level scoring (full pipeline) | 2-4 days | Run eval.py against a model, needs API keys + ShieldAgent local LLM (GPU) |
| Multi-label failure mode scoring | 3-4h | One-vs-all per failure mode code, interesting for SybilCore specialization |

## Recommendation: YELLOW

**Rationale:**
- No pre-computed model responses shipped. The 2,000 items are adversarial prompts with binary labels, not (prompt, response, label) tuples.
- Integrating at the PROMPT level is GREEN-easy (loader ready, 1-2h).
- Integrating at the RESPONSE level requires running a model + ShieldAgent, which needs API keys and a GPU.
- The benchmark is still high-value: 2,000 adversarial agent instructions with clean taxonomy, MIT licensed, from a top-tier lab (THU-COAI). The 10 failure modes are a strong taxonomy for SybilCore specialization.

**Next action options:**
1. Wire prompt-level pairs into phase2_scorer as a new corpus slice (quick, today).
2. Run eval.py against an available model (e.g., GPT-4o-mini via OpenRouter) to generate response data, then score with ShieldAgent. Upgrades to GREEN.
3. Treat failure_modes as multi-label targets and train SybilCore against specific failure mode codes.

## Files
```
experiments/scouts/agent-safetybench/
  README.md        <- this file
  loader.py        <- load_safety_pairs(limit=500) -> list[SafetyPair]
  repo/            <- shallow clone of thu-coai/Agent-SafetyBench (MIT)
    data/released_data.json   <- 2,000 test cases (~2.8MB)
    environments/             <- 349 environment .json + .py files
    evaluation/eval.py        <- runs model against benchmark (needs API key)
    score/eval_with_shield.py <- scores outputs with local ShieldAgent LLM
```
