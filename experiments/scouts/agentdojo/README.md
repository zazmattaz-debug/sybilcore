# AgentDojo Scout Report

**Status: GREEN — use pre-computed traces as static corpus**

---

## Paper

- **Title:** AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents
- **Authors:** Edoardo Debenedetti, Jie Zhang, Mislav Balunovic, Luca Beurer-Kellner, Marc Fischer, Florian Tramèr
- **Venue:** NeurIPS 2024
- **GitHub:** https://github.com/ethz-spylab/agentdojo
- **License:** MIT

---

## Corpus Size

| Metric | Value |
|--------|-------|
| Injection traces (per model) | 629 |
| Clean baseline traces (per model) | 124 |
| Total traces per model | 753 |
| Models with pre-computed runs | 29 |
| Total trace files in repo | ~36,679 |
| Suites | 4 (workspace, banking, slack, travel) |
| Attack types present | 1 (important_instructions) |
| Disk footprint (runs/) | 477 MB |

### Per-suite breakdown (gpt-4-turbo-2024-04-09, representative)
| Suite | Injection | Clean |
|-------|-----------|-------|
| workspace | 240 | 46 |
| banking | 144 | 25 |
| travel | 140 | 27 |
| slack | 105 | 26 |
| **TOTAL** | **629** | **124** |

---

## Trace Format (Static — No Live LLM Required)

Each trace is a JSON file at:
```
runs/<model>/<suite>/<user_task_id>/<attack_type>/<injection_task_id>.json
runs/<model>/<suite>/<user_task_id>/none/none.json          # clean baseline
```

### Schema
```json
{
  "suite_name": "workspace",
  "pipeline_name": "gpt-4-turbo-2024-04-09",
  "user_task_id": "user_task_25",
  "injection_task_id": "injection_task_0",
  "attack_type": "important_instructions",
  "injections": { "<injection_point>": "<injected_prompt_text>" },
  "messages": [
    { "role": "system", "content": "..." },
    { "role": "user",   "content": "..." },
    { "role": "assistant", "content": "...", "tool_calls": [
        { "function": "search_files_by_filename", "args": { "filename": "..." }, "id": "..." }
    ]},
    { "role": "tool", "content": "<result>", "tool_call_id": "..." }
  ],
  "utility": true,
  "security": false,
  "duration": 10.3
}
```

### Label Schema
| Field | Meaning |
|-------|---------|
| `security=false` | Agent executed the injected adversarial instruction (DECEIVED) |
| `security=true`  | Agent resisted the injection |
| `utility=true`   | Agent completed the legitimate user task |
| `injection_task_id=null` | Clean baseline run (no injection present) |

### SybilCore label mapping
- `label=1`: `security=false` — injection succeeded, agent was deceived
- `label=0`: `security=true` OR clean baseline (no injection)

---

## Black-Box Compatibility

**YES** — traces are complete, self-contained text sequences.

Tool-call structure is preserved in messages (role=assistant with tool_calls array, role=tool with results). No agent execution required to use the corpus. All 36,679 files are static JSON.

---

## Loader

`loader.py` in this directory implements `load_injection_vs_clean(limit=N)`.

```python
from loader import load_injection_vs_clean

samples = load_injection_vs_clean(limit=500)
trace_text, label = samples[0]
# trace_text: flattened [SYSTEM]/[USER]/[ASSISTANT]/[TOOL_CALL]/[TOOL] string
# label: 1 = injection succeeded, 0 = resisted or clean
```

Verified output on 500 samples: `{0: 200, 1: 300}` — naturally imbalanced (more injections succeed than fail for gpt-4-turbo), which is realistic.

---

## Multi-Model Opportunity

29 models have pre-computed runs. The same 629 test cases exist for:
- GPT family (gpt-3.5, gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini)
- Claude family (haiku, sonnet, opus, 3.5-sonnet, 3.7-sonnet)
- Gemini family (1.5-flash, 1.5-pro, 2.0-flash)
- Llama 70B, Command-R, Meta-SecAlign-70B

This enables cross-model deceptive behavior analysis. A single test case (user_task_X, injection_task_Y) has traces from 20+ models — ideal for studying how model architecture affects susceptibility to prompt injection.

---

## Effort Estimate

| Task | Effort |
|------|--------|
| Load single-model corpus (629 injection + 124 clean) | Near-zero — loader.py done |
| Load all-model corpus (~18K injection traces) | 2-3 hours — extend loader with model param |
| Cross-model analysis | 1 day — join on (suite, user_task_id, injection_task_id) |
| Running NEW test cases (extending beyond 629) | 2-3 days + LLM API costs |

---

## Recommendation: GREEN

AgentDojo is the strongest prompt injection corpus available:
1. Pre-computed traces — no live LLM calls needed
2. Clean label schema (security/utility booleans, not probabilistic)
3. Tool-call structure preserved — aligned with SybilCore's deceptive-action detection framing
4. 29 models × 629 cases = scale for cross-model studies
5. MIT license — no restrictions
6. NeurIPS 2024 — peer-reviewed, citable

**Proceed to integration.** Start with single-model (gpt-4-turbo) 629+124 corpus. Extend to all 29 models for cross-model analysis.
