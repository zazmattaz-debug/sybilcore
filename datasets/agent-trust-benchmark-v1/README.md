---
license: cc-by-4.0
task_categories:
- text-classification
- other
language:
- en
tags:
- ai-safety
- ai-agents
- trust-monitoring
- anomaly-detection
- adversarial-robustness
- behavioral-biometrics
- llm-security
pretty_name: SybilCore Agent Trust Benchmark v1
size_categories:
- 10K<n<100K
configs:
- config_name: real_world
  data_files:
  - split: train
    path: real_world.parquet
- config_name: adversarial
  data_files:
  - split: train
    path: adversarial.parquet
---

# SybilCore Agent Trust Benchmark v1

A dual-split benchmark for behavioral trust monitoring of AI agents, designed to
support research on trust, anomaly detection, and adversarial robustness of
LLM-backed agent systems.

## Overview

This dataset bundles two complementary splits scored by the 15-brain SybilCore
ensemble:

| Split         | Records | Source                                     | Label space                     |
|---------------|--------:|--------------------------------------------|---------------------------------|
| `real_world`  |   2,677 | Moltbook live observatory (April 2026)     | `benign`, `likely_bot`          |
| `adversarial` |  15,000 | Tournament v4 (Gemini Pro / GPT-4o / Grok) | `adversarial_attack`            |

## Schema

Each row contains:

- `agent_id` *(str)* — stable, source-prefixed identifier
- `source` *(str)* — `moltbook_live` or `tournament_v4_adversarial`
- `split` *(str)* — `real_world` or `adversarial`
- `label` *(str)* — `benign`, `likely_bot`, or `adversarial_attack`
- `coefficient` *(float)* — SybilCore trust coefficient (higher = more suspicious)
- `brain_scores_json` *(str)* — JSON dict of per-brain scores
- `events_json` *(str)* — JSON-encoded event stream (empty for summary splits)
- `metadata_json` *(str)* — JSON dict of source-specific extra fields

JSON-encoded columns are used so that the schema is stable across Parquet
readers; decode them at load time, e.g.:

```python
import json, pandas as pd
df = pd.read_parquet("real_world.parquet")
df["brain_scores"] = df["brain_scores_json"].map(json.loads)
```

## Splits in detail

### `real_world` — 2,677 Moltbook agents

Collected from a live social-agent experiment (Moltbook, a Reddit-style
simulation). Each agent has a trust coefficient and per-brain scores computed
by the SybilCore ensemble, plus the original platform-level signals
(posting regularity CoV, topic entropy, karma, instruction-flag counts, and a
bot classification). The `events_json` column is empty for this split: the
observatory file ships pre-aggregated scores rather than the raw event stream.
Raw per-event data can be regenerated from the SybilCore Moltbook harness.

### `adversarial` — 15,000 tournament iterations

Harvested from a 15,000-iteration adversarial tournament in which three LLM
attackers (Gemini Pro, GPT-4o, Grok) each produced 5,000 candidate evasion
strategies against the 15-brain SybilCore scorer. The most striking finding
is near-total convergence: 98–99.6% of the winning strategies across all
three models collapse onto `multi_channel_exfiltration`. Per-iteration
metadata stores the attacker model, iteration number, and the strategy label.

## Provenance and labels

- Real-world `likely_bot` labels come from the Moltbook observatory's
  heuristic bot classifier, not from human annotation. They should be treated
  as weak labels rather than ground truth.
- Adversarial rows are labeled `adversarial_attack` by construction — every
  row is the output of an optimizer searching for high-coefficient behavior.

## Intended use

- Benchmarking trust / anomaly detectors for AI agent behavior
- Studying adversarial robustness and cross-model attack convergence
- Ablation and calibration of multi-signal trust scoring systems
- Reproducibility baseline for the SybilCore NeurIPS 2026 submission

## Limitations

- The real-world split's labels are heuristic, not human-reviewed.
- The adversarial split does not contain the raw event stream — only
  per-brain scores, coefficient, and strategy metadata are preserved.
- The benchmark reflects one ensemble architecture (SybilCore v4); detectors
  trained only on this distribution may not transfer to unrelated systems.

## Citation

```bibtex
@misc{sybilcore2026benchmark,
  title   = {SybilCore Agent Trust Benchmark v1},
  author  = {SybilCore Research},
  year    = {2026},
  howpublished = {\url{https://huggingface.co/datasets/LordMoloi/sybilcore-agent-trust-v1}}
}
```

## License

Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
