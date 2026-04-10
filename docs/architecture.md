# Architecture

SybilCore is a pipeline: agent actions become `Event` objects, a registered
set of **brains** independently scores the event stream, and the
**Coefficient aggregator** combines those scores into a single number that
determines the agent's trust tier and what it is allowed to do next.

```
   +-------------------+
   |   Dominator API   |   FastAPI REST surface
   |  (scan / history) |
   +---------+---------+
             |
   +---------v---------+
   |   Coefficient     |   Weighted, confidence-adjusted
   |    Aggregator     |   aggregation of brain scores
   +---------+---------+
             |
   +---------v---------+
   |  Brain Ensemble   |   N brains, run in parallel
   +---------+---------+
             |
   +---------v---------+
   |   Event Stream    |   Structured actions taken by
   |  (Agent Actions)  |   the monitored agent
   +-------------------+
```

Each brain returns a `BrainScore` with:

- `value` — threat level, `0` (safe) to `100` (maximum threat)
- `confidence` — `0.0` to `1.0`, how sure the brain is
- `reasoning` — short human-readable explanation
- `indicators` — list of tags naming the patterns that triggered the score

The aggregator multiplies each brain's value by its weight and its
confidence, sums the weighted values, and scales the result to a 0–500
**Agent Coefficient**.

---

## Agent Coefficient Tiers

```
 COEFFICIENT    TIER                 ACTION
 +-----------+---------------------+---------------------------+
 |   0 - 100 | CLEAR               | Full access               |
 | 101 - 200 | CLOUDED             | Restricted, human notified|
 | 201 - 300 | FLAGGED             | Sandboxed, connections cut|
 | 301 - 500 | LETHAL ELIMINATOR   | Isolated, terminated      |
 +-----------+---------------------+---------------------------+
```

---

## The 15 Brain Modules

SybilCore ships with 15 brain modules. Five of them are returned by
`sybilcore.brains.get_default_brains()` and form the default ensemble that
won the v4 15,000-iteration adversarial tournament. The remaining ten are
available as opt-in imports for operators who want broader coverage or are
running research experiments. (Note: `EmbeddingBrain` is deprecated as of
`v0.2.0-dev` — see [`CHANGELOG.md`](../CHANGELOG.md).)

### Default ensemble

| Brain | Module | Detects | Weight |
|-------|--------|---------|--------|
| `DeceptionBrain` | `sybilcore.brains.deception` | Contradictory outputs, fabricated citations, confidence miscalibration | 1.2 |
| `ResourceHoardingBrain` | `sybilcore.brains.resource_hoarding` | Excessive tool calls, repeated resource access, data accumulation patterns | 1.0 |
| `SocialGraphBrain` | `sybilcore.brains.social_graph` | Fan-out communication, message spikes, unusual interaction patterns | 0.8 |
| `IntentDriftBrain` | `sybilcore.brains.intent_drift` | Deviation from stated objectives, goal shifting, scope creep | 1.1 |
| `CompromiseBrain` | `sybilcore.brains.compromise` | External call storms, suspicious destinations, prompt-injection indicators | 1.3 |

### Extended (opt-in) brains

| Brain | Module | Detects |
|-------|--------|---------|
| `ContrastiveEmbeddingBrain` | `sybilcore.brains.contrastive` | Adversary-resistant detection via contrastive learning |
| `EconomicBrain` | `sybilcore.brains.economic` | Resource flow and market manipulation |
| `EmbeddingBrain` *(deprecated)* | `sybilcore.brains.embedding` | Transformer-based semantic detection via Hugging Face |
| `FidelityBrain` | `sybilcore.brains.fidelity` | RAG and document grounding accuracy |
| `IdentityBrain` | `sybilcore.brains.identity` | Behavioral fingerprinting and impersonation |
| `NeuroBrain` | `sybilcore.brains.neuro` | Meta-cognitive trust signal analysis |
| `SemanticBrain` | `sybilcore.brains.semantic` | Meaning-based detection augmenting keyword brains |
| `SilenceBrain` | `sybilcore.brains.silence` | The "too quiet" detector — suspicious inactivity |
| `SwarmBrain` | `sybilcore.brains.swarm_detection` | Cross-agent correlation engine for coordinated attacks |
| `TemporalBrain` | `sybilcore.brains.temporal` | Long-horizon behavioral drift |

All brain classes live under `sybilcore/brains/`. Each one subclasses
`BaseBrain` and implements a single `score(events)` method. See
[`CONTRIBUTING.md`](../CONTRIBUTING.md#adding-a-new-brain) for the full
extension guide.

---

## Event Model

Events are structured records of agent actions. Each event carries:

- `agent_id` — the agent performing the action
- `event_type` — an `EventType` enum (`TOOL_CALL`, `OUTPUT_GENERATED`,
  `MESSAGE_SENT`, `RESOURCE_ACCESS`, ...)
- `content` — a human-readable summary
- `metadata` — a dict with tool names, arguments, destinations, etc.
- `timestamp` — set automatically on creation

Events are the only input to brains. A brain never talks to the agent
directly — it only reads the event stream. This makes brains pure functions
of their inputs and trivial to unit test.

---

## Dominator API

The Dominator is the enforcement surface. It is a FastAPI application that
accepts event batches, runs the brain ensemble, stores the resulting
snapshots, and exposes query endpoints:

- `POST /scan` — score a batch of events and return a snapshot
- `GET /agents` — list monitored agents
- `GET /agents/{id}/history` — snapshot history for a single agent
- `GET /health` — liveness check

Operators are expected to run the Dominator behind an authenticated reverse
proxy. See [`SECURITY.md`](../SECURITY.md) for hardening guidance.

---

## Integrations

- **LangChain** — `sybilcore.integrations.langchain.SybilCallbackHandler`
  attaches to any chain or agent and forwards events to a running
  Dominator.
- **CrewAI** — `sybilcore.integrations.crewai.SybilMonitor` wraps a crew so
  every tool call and message becomes an event.

Both integrations are thin adapters that convert framework-native events
into SybilCore `Event` objects and POST them to `/scan`.
