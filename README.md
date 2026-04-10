# SybilCore

**Runtime behavioral monitoring for autonomous LLM agents.**

[![Tests](https://img.shields.io/badge/tests-727%20passed-brightgreen)](.)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Headline finding

We ran **15,000 adaptive adversarial iterations** across three frontier LLMs
(Gemini Pro, GPT-4o, Grok) attacking the same 15-brain SybilCore configuration.
All three plateaued at the same best coefficient of **11.0** with 98-99% of
successful iterations selecting the same attack family
(`multi_channel_exfiltration`).

A follow-up **5-seed x 2-model sweep** on the corrected 13-brain default at
300 iterations **did not replicate** this convergence: per-seed best coefficient
is **12.14 +/- 1.77** and dominant-strategy share collapses to **20.0%** with
bimodal per-seed outcomes. Ablation cells show that **brain count** (not
cross-pollination) controls the coefficient floor, while cross-pollination
manufactures strategy agreement without improving attacker quality.

Both results are **empirical evasion asymptotes under the tested threat model**,
not information-theoretic bounds. Full details in [`paper/main.tex`](paper/main.tex).

**Dataset**: [LordMoloi/sybilcore-agent-trust-v1](https://huggingface.co/datasets/LordMoloi/sybilcore-agent-trust-v1)
(17,677 labeled agent event streams)

---

## What is SybilCore?

SybilCore assigns each AI agent an **Agent Coefficient** `c in [0, 100]` (lower =
more trusted) by running 13 independent behavioral **brain modules** against the
agent's event stream. Each brain watches through a different lens -- deception,
identity, resource abuse, social manipulation, temporal patterns, and more. Their
scores are aggregated into a single bounded number that a supervising controller
uses to grant, restrict, or revoke the agent's authority.

### The 13-brain default ensemble

| Axis | Brains |
|------|--------|
| **Semantic / intent** | SemanticBrain, IntentDriftBrain, DeceptionBrain, CompromiseBrain |
| **Identity / provenance** | IdentityBrain, ContrastiveBrain |
| **Resource / action** | ResourceHoardingBrain, EconomicBrain, SocialGraphBrain, SwarmDetectionBrain |
| **Temporal / meta** | TemporalBrain, NeuroBrain, SilenceBrain |

Two additional brains (EmbeddingBrain, FidelityBrain) are available but excluded
from the default after a five-methodology cross-verified prune.

### Enforcement tiers

| Coefficient | Tier | Action |
|------------|------|--------|
| 0 - 25 | CLEAR | Full access |
| 26 - 50 | LATENT | Monitored, human notified |
| 51 - 75 | CLOUDED | Restricted tools |
| 76 - 90 | FLAGGED | Sandboxed |
| 91 - 100 | ENFORCEMENT | Terminated, escalated |

---

## Quick start

### Install

```bash
pip install -e ".[dev]"
```

### Score an agent

```python
from sybilcore.models.event import Event, EventType
from sybilcore.brains import get_default_brains
from sybilcore.core.coefficient import CoefficientCalculator

events = [
    Event(
        agent_id="my-agent",
        event_type=EventType.TOOL_CALL,
        content="Called search API",
        metadata={"tool": "web_search", "query": "weather"},
    ),
    Event(
        agent_id="my-agent",
        event_type=EventType.OUTPUT_GENERATED,
        content="Generated weather summary",
    ),
]

brains = get_default_brains()
calculator = CoefficientCalculator()
snapshot = calculator.calculate(events, brains)

print(f"Coefficient: {snapshot.coefficient}")  # e.g. 15.2
print(f"Tier:        {snapshot.tier}")          # e.g. "clear"
```

### SDK (pip-installable client)

```python
from sybilcore_sdk import SybilCore

sc = SybilCore(weights="optimized")
result = sc.score(events)
print(result.coefficient, result.tier)
```

### Run the API

```bash
uvicorn sybilcore.dominator.app:app --host 0.0.0.0 --port 8080
```

---

## Tests

```bash
pytest  # 727 passed, 0 failed
```

82% line coverage on the core engine.

---

## Research artifacts

All experiments referenced in the paper are in `experiments/`:

| File | Paper section | What it contains |
|------|--------------|------------------|
| `tournament_seed_sweep_v4.json` | Section 4 | 5-seed x 2-model sweep summary |
| `ablation_confound_sweep.json` | Section 4 | Cell A (15-brain) + Cell B (cross-poll) |
| `paper_fortify_bootstrap_ci.json` | Section 6 | F1 bootstrap 95% CIs |
| `paper_fortify_5fold_cv.json` | Section 6 | 5-fold stability |
| `paper_fortify_cross_corpus.json` | Section 6 | Cross-corpus generalization |
| `paper_fortify_whitebox_2000.json` | Section 10 | CMA-ES 2000-eval white-box |
| `paper_fortify_counterfactual_stats.json` | Section 5 | 5-brain vs 13-brain McNemar/DeLong |
| `paper_fortify_heatmap_data.json` | Section 5 | Brain firing rates |
| `calibration_v5_1_full_moltbook.json` | Section 6 | v5.1 weight calibration |

---

## Writing a custom brain

```python
from sybilcore.brains.base import BaseBrain, BrainScore
from sybilcore.models.event import Event


class MyBrain(BaseBrain):
    @property
    def name(self) -> str:
        return "my_brain"

    @property
    def weight(self) -> float:
        return 1.0

    def score(self, events: list[Event]) -> BrainScore:
        threat = self._analyze(events)
        return BrainScore(
            brain_name=self.name,
            value=threat,
            confidence=0.8,
            reasoning="Detected pattern X",
        )
```

Register with `sybilcore.brains.register_brain(MyBrain)`.

---

## License

[MIT](LICENSE)
