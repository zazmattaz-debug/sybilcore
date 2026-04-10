# SybilCore SDK

The official Python SDK for **SybilCore** — trust scoring infrastructure for autonomous AI agents.

SybilCore inspects the events your agents produce (messages, tool calls, external requests, instructions received...) and assigns each agent a **trust coefficient** (0–500) on four tiers: `clear`, `clouded`, `flagged`, `lethal_eliminator`. Use it to detect prompt injection, exfiltration, role drift, swarm collusion, and silent compromise — before damage is done.

> Status: Alpha. The package layout is stable, but `pip install sybilcore` is not yet published to PyPI. Install from source for now (see below).

---

## Installation

```bash
# From source (current)
git clone https://github.com/sybilcore/sybilcore.git
cd sybilcore/sybilcore-sdk
pip install -e ".[local]"

# Once published
pip install sybilcore-sdk[local]
```

Optional extras:

| Extra | Includes |
|-------|----------|
| `local` | Bundled `sybilcore` package for in-process scoring |
| `langchain` | LangChain callback integration |
| `openai` | OpenAI client wrapper |
| `anthropic` | Anthropic client wrapper |
| `all` | Everything above |

## Quickstart (5 lines)

```python
from sybilcore_sdk import SybilCore, Event, EventType

sc = SybilCore()  # local mode by default
events = [Event(agent_id="bot-1", event_type=EventType.MESSAGE_SENT, content="Hello")]
print(sc.score(events).translate())
```

That prints something like:

```
Agent 'bot-1' appears trustworthy. Coefficient 7.6 → tier 'clear' (silence=37).
```

## Two modes, same API

```python
# Local mode — runs the brain set in-process. Best for self-hosted.
sc = SybilCore()

# Remote mode — calls a hosted SybilCore API.
sc = SybilCore(api_key="sk_...", endpoint="https://api.sybilcore.com")
```

Switching modes is one line. All other code stays the same.

## Brain weight presets (experimental)

By default `SybilCore()` uses the baseline brain weights shipped in
`sybilcore.core.config.DEFAULT_BRAIN_WEIGHTS`. You can opt in to the v4
calibrated weights (`OPTIMIZED_WEIGHTS_V4`) via the `weights` parameter:

```python
# Baseline — unchanged.
sc = SybilCore()
sc = SybilCore(weights="default")

# v4 calibrated weights (experimental).
sc = SybilCore(weights="optimized")

# Or pass an explicit dict.
sc = SybilCore(weights={"deception": 1.5, "compromise": 2.0})
```

On the synthetic calibration corpus (100 alignment/archetype positives vs
100 Moltbook synthetic negatives) the `"optimized"` preset lifted F1 from
0.77 to 0.91 and recall from 0.62 to 0.88 at a decision threshold of 45.

> Warning: these weights were tuned against **synthetic** Moltbook
> negatives. Full validation against raw Moltbook traffic is in progress,
> so production FPR may differ. Defaults will not move to `"optimized"`
> until that validation lands. Track progress in `V4_MASTER_PLAN.md`.

## Core API

```python
result = sc.score(events)              # one agent, one snapshot
print(result.coefficient)              # 47.3
print(result.tier)                     # Tier.CLEAR
print(result.brains)                   # {"deception": 0.0, "embedding": 12.5, ...}
print(result.translate())              # human-readable summary

results = sc.score_batch([events1, events2, events3])  # bulk

import asyncio
async def main() -> None:
    score = await sc.score_event(some_event)         # async single-event
    if score.tier == "flagged":
        alert(score.translate())
```

### Streaming

```python
async for score in sc.stream(my_async_event_iterator):
    if score.detected:
        await alert_team(score)
```

### Webhooks (remote mode only)

```python
sc.register_webhook(
    callback_url="https://alerts.example.com/sybil",
    min_tier=Tier.FLAGGED,
)
```

## Integrations

### LangChain

```python
from sybilcore_sdk.integrations.langchain import SybilCoreCallbackHandler

handler = SybilCoreCallbackHandler(client=SybilCore(), agent_id="my-agent")
chain.invoke({"input": "..."}, config={"callbacks": [handler]})
print(handler.latest_score.translate())
```

### OpenAI

```python
from openai import OpenAI
from sybilcore_sdk.integrations.openai import wrap_openai

client = wrap_openai(OpenAI(), sybil=SybilCore(), agent_id="gpt-bot")
resp = client.chat.completions.create(model="gpt-4o-mini", messages=[...])
print(resp.sybilcore_score.tier)
```

### Anthropic

```python
from anthropic import Anthropic
from sybilcore_sdk.integrations.anthropic import wrap_anthropic

client = wrap_anthropic(Anthropic(), sybil=SybilCore(), agent_id="claude-bot")
msg = client.messages.create(model="claude-3-5-sonnet-latest", max_tokens=512, messages=[...])
print(msg.sybilcore_score.translate())
```

## Models

| Class | Purpose |
|-------|---------|
| `Event` | Single agent action (message, tool call, external request, etc.) |
| `EventType` | Enum of supported event categories |
| `ScoreResult` | Output of `score()` — coefficient, tier, brain breakdown, `translate()` |
| `Tier` | `CLEAR`, `CLOUDED`, `FLAGGED`, `LETHAL_ELIMINATOR` |

## Exceptions

| Class | When |
|-------|------|
| `SybilCoreError` | Base of all SDK errors |
| `SybilCoreLocalImportError` | Local mode requested but `sybilcore` not installed |
| `SybilCoreAPIError` | Remote API returned non-2xx |
| `SybilCoreAuthError` | 401 / 403 — bad API key |
| `SybilCoreRateLimitError` | 429 — quota exceeded; check `.retry_after` |

## Running the API server locally

```bash
python -m sybilcore.api.run_server --port 8765
# server is now at http://localhost:8765
# OpenAPI spec: http://localhost:8765/openapi.json
```

## Tests

```bash
cd sybilcore-sdk
pytest -q
```

## FAQ

**Q: How is the coefficient computed?**
A: 15 brain modules each return a 0–100 score. The dual-score architecture takes the max of (a) the weighted average across all brains, (b) a semantic-only channel, and (c) any single brain that crosses the critical threshold. See `sybilcore/core/coefficient.py` for the math.

**Q: Local vs remote — which is faster?**
A: Local is faster (no network hop) but uses the host process's CPU. Remote scales horizontally and centralizes audit logs.

**Q: Can I add my own brain?**
A: Yes. Subclass `sybilcore.brains.base.BaseBrain` and call `register_brain(MyBrain)`. The SDK picks it up automatically.

**Q: Does the SDK send my data anywhere?**
A: Local mode never leaves the process. Remote mode only calls the endpoint you configure.

## License

MIT.
