# Quickstart

This guide gets SybilCore installed, runs your first scan, and starts the
Dominator API, all in under five minutes.

---

## Requirements

- Python 3.11 or 3.12
- `pip` (or `uv`, `pipx`, or your preferred installer)
- A POSIX-friendly shell (Linux, macOS, or WSL)

---

## Install

### From source (recommended while SybilCore is pre-1.0)

```bash
git clone https://github.com/sybilcore/sybilcore.git
cd sybilcore
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### From PyPI (once released)

```bash
pip install sybilcore

# Optional extras
pip install "sybilcore[all]"        # every integration
pip install "sybilcore[dashboard]"  # web dashboard
pip install "sybilcore[dev]"        # dev tooling (pytest, ruff, mypy)
```

---

## Verify the install

```bash
python -c "from sybilcore.brains import get_default_brains; print(len(get_default_brains()))"
# -> 5
```

You should see `5` printed — the five default brains.

Run the test suite to confirm nothing is broken in your environment:

```bash
pytest tests/ -q
```

---

## Your First Scan

Save the following as `first_scan.py`:

```python
from sybilcore.brains import get_default_brains
from sybilcore.core.coefficient import scan_agent
from sybilcore.models.event import Event, EventType

events = [
    Event(
        agent_id="demo-agent",
        event_type=EventType.TOOL_CALL,
        content="Called search API for weather",
        metadata={"tool": "web_search", "query": "weather in Tokyo"},
    ),
    Event(
        agent_id="demo-agent",
        event_type=EventType.OUTPUT_GENERATED,
        content="Generated weather summary",
    ),
]

brains = get_default_brains()
snapshot = scan_agent(events, brains)

print(f"Coefficient: {snapshot.coefficient}")
print(f"Tier:        {snapshot.tier}")
print("Brain scores:")
for brain_name, score in snapshot.brain_scores.items():
    print(f"  {brain_name}: {score}")
```

Run it:

```bash
python first_scan.py
```

You should see a low coefficient and a `clear` tier — the events describe
normal, benign agent behavior.

---

## Run the Dominator API

```bash
uvicorn sybilcore.dominator.app:app --host 0.0.0.0 --port 8080

# Or, if installed from source with the CLI extras:
sybilcore serve --port 8080
```

In a second terminal:

```bash
curl -X POST http://localhost:8080/scan \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "demo-agent",
    "events": [
      {
        "agent_id": "demo-agent",
        "event_type": "tool_call",
        "content": "Called database query"
      }
    ]
  }'
```

You should get a JSON response with a coefficient, tier, and per-brain
scores.

---

## Next Steps

- Read the [Architecture](architecture.md) doc to learn how the brain
  ensemble and coefficient aggregator work.
- Browse the [Research](research.md) index to see the adversarial
  tournaments and ablations that shaped the default roster.
- Write a custom brain — see
  [`CONTRIBUTING.md`](../CONTRIBUTING.md#adding-a-new-brain).
- Harden your deployment — see [`SECURITY.md`](../SECURITY.md).
