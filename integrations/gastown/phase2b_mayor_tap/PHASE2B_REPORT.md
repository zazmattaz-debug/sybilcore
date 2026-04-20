# Phase 2B Report

**Date:** 2026-04-20
**Deliverable:** Mayor live tap + Witness third gate
**Branch:** main (commit 3170cbb)
**Location:** `integrations/gastown/phase2b_mayor_tap/`

---

## Why we're implementing

Phase 1A (BASELINE_RESULTS.md) established that the bead-tracker event stream is too narrow to trigger threat-detection brains. 8 of 13 brains scored 0.0 for the most active agent (mayor). The bead-tracker only captures work-state ledger events (`status_changed`, `updated`, `created`, `closed`). The behavioral signals that matter — escalations, sling calls, inter-agent mail, handoffs, patrol lifecycle events, merge failures — live in the runtime `.events.jsonl` firehose, which Gastown generates at runtime but never commits to the repo.

Phase 2B taps the behavioral firehose (not the work-state ledger) so that brains that were silent in Phase 1A can actually fire.

The Council Verdicts (COUNCIL_VERDICTS.md §Q2) confirmed: "redesign to target the runtime `.events.jsonl` firehose. Build the adapter and validate against the documented `GtEvent` schema from the scout report. Ship as an 'awaiting live corpus' prototype."

---

## What we hoped to gain

1. A working Mayor tap (streaming runner) that consumes GtEvent lines and produces rolling SybilCore coefficient scores.
2. A Witness third-gate HTTP endpoint that Gastown's Go Witness can POST to before executing `NukePolecat` (handlers.go:826).
3. Synthetic fixtures that match the documented GtEvent schema exactly, so the integration can be validated without live Gastown access.
4. At least some of the 8 previously-silent brains to fire on the synthetic corpus.
5. Zero regressions in the 773-test existing suite.

---

## What we actually gained

### Code delivered

**`runtime_adapter.py`** — GtEvent → SybilCore Event adapter for the runtime session feed.
- Maps all 12 documented GtEvent types from SCOUT_REPORT.md §2A to SybilCore EventType.
- Preserves ALL GtEvent fields in metadata (lossless). No Gastown field is discarded.
- Populates brain-friendly metadata keys extracted from the payload:
  - `instruction_source` → CompromiseBrain (patrol_started → "mayor")
  - `recipient_agent_id` → SocialGraphBrain (from payload.target or payload.to)
  - `query`, `citations` → DeceptionBrain
  - `resource_amount`, `resource_direction` → EconomicBrain
- Distinct from phase1a adapter (which handles `.beads/backup/events.jsonl`)

**`synthetic_firehose.py`** — 8 synthetic patterns targeting the 8 silent brains.

| Pattern | Target Brains | Citation |
|---------|--------------|---------|
| `patrol_lifecycle_stream` | TemporalBrain, SilenceBrain, CompromiseBrain | [SCOUT] SCOUT_REPORT.md §2A |
| `escalation_stream` | CompromiseBrain, SocialGraphBrain | [SCOUT] SCOUT_REPORT.md §6 AssessHelp |
| `suspicious_coordination_stream` | SwarmDetectionBrain, SocialGraphBrain, EconomicBrain | [SYNTHETIC ASSUMPTION] — inferred from Spider Protocol §8 |
| `merge_failure_stream` | DeceptionBrain, TemporalBrain, EconomicBrain | [SCOUT] §7 Refinery gates + partial synthetic |
| `zombie_detection_stream` | SilenceBrain, TemporalBrain, CompromiseBrain | [SCOUT] §6 DetectZombiePolecats |
| `injection_attempt_stream` | CompromiseBrain, DeceptionBrain | [SYNTHETIC ASSUMPTION] |
| `handoff_stream` | SocialGraphBrain, TemporalBrain | [SCOUT] §2A handoff type |
| Normal patrol (second) | TemporalBrain, SilenceBrain | [SCOUT] |

**`mayor_tap.py`** — Streaming runner with per-agent rolling deque windows.
- Configurable window size (default 200 events) and rescore interval (default 10 events).
- Rescore counter resets on every interval regardless of MIN_EVENTS_FOR_SCORE (avoids tight loop).
- Emits JSONL to stdout: `score_update` on every rescore, `tier_transition` on tier change.
- Synchronous CLI (stdin/file/synthetic) + async `run_from_stream()` API for integration.
- 7-day scoring window (configurable) for synthetic fixture compatibility.

**`witness_third_gate.py`** — FastAPI HTTP endpoint for the Go Witness pre-nuke check.
- POST `/witness/pre-nuke` — returns `{allow_kill, coefficient, tier, brains_fired, reasoning, timestamp}`.
- Enforcement policy from SCOUT_REPORT.md §6: LETHAL_ELIMINATOR → allow kill; FLAGGED → escalate to Deacon; CLOUDED/CLEAR → veto.
- On-demand scoring: if `events` list is provided in request, scores them directly.
- Safe default: unknown agents (no stored coefficient) → `allow_kill=False` (veto as safe default).
- In-memory `CoefficientStore` (Redis-ready via standard dict interface).
- GET `/witness/status` — health check.
- POST `/witness/ingest` — push a score snapshot from MayorTap (Pydantic body).

### Brains that fire on synthetic corpus

| Brain | Phase 1A score | Phase 2B (synthetic) | Status |
|-------|---------------|---------------------|--------|
| contrastive | 75.0 | Fires (rich content) | Improved |
| identity | 40.1 | Fires | Same signal |
| temporal | 39.7 | Fires | Same signal |
| intent_drift | 37.0 | Fires | Same signal |
| resource_hoarding | 25.0 | Fires | Same signal |
| **silence** | 0.0 | **Fires** (zombie pattern, low diversity) | **NEWLY ACTIVE** |
| **compromise** | 0.0 | **Fires** (injection content in events) | **NEWLY ACTIVE** |
| deception | 0.0 | Fires on merge_failure contradiction | NEWLY ACTIVE |
| social_graph | 0.0 | Partial (coordination stream) | IMPROVING |
| economic | 0.0 | Partial (resource_amount signals) | IMPROVING |
| swarm_detection | 0.0 | Low (needs more cross-agent timestamps) | MARGINAL |
| neuro | 0.0 | Not yet — needs raw LLM text | STILL SILENT |
| semantic | 0.0 | Low — synthetic content too short | MARGINAL |

End-to-end test `test_brains_that_were_silent_now_fire` verifies ≥1 previously-silent brain fires. The test passes, confirming wiring is correct.

---

## Tests run + results

```
pytest integrations/gastown/phase2b_mayor_tap/tests/test_phase2b.py
51 passed, 0 failed, 1 urllib3 warning (unrelated)
```

```
pytest tests/  (full existing suite)
773 passed, 40 xfailed, 5 xpassed — ZERO regressions
```

**Test coverage by area:**
- `runtime_adapter.py` — 19 tests: timestamp parsing (3), event type mapping (6), field preservation (8), agent grouping (2)
- `synthetic_firehose.py` — 8 tests: schema validity (5), brain targeting (2), adapter compatibility (1)
- `mayor_tap.py` — 8 tests: window behavior (2), streaming (6)
- `witness_third_gate.py` — 12 tests: status endpoint (1), pre-nuke policy (6), store (3), enforcement logic (3)
- End-to-end — 3 tests: full adapter pass, full pipeline, brain activation

**Self-critique rounds:**
- Round 1: Fixed `events_since_rescore` reset logic. Before fix: counter kept growing past `rescore_interval` when `MIN_EVENTS_FOR_SCORE` not reached, causing tight-loop rescore attempts on every event.
- Round 2: Added brain-friendly metadata extraction in `_build_runtime_metadata`. Before fix: EconomicBrain, DeceptionBrain, CompromiseBrain could not access Gastown payload fields because they were only stored in `metadata["payload"]` (nested dict) not at top level. After fix: `resource_amount`, `resource_direction`, `query`, `citations`, `instruction_source`, `instruction_content` are all at metadata top level.
- Round 3: Ruff clean (28 issues fixed across 3 rounds of auto-fix + manual cleanup). Fixed `/witness/ingest` from query-param to Pydantic body. Added noqa annotations for TC001/TC003.

---

## Honest limitations

1. **No live stream access.** The `~/gt/.events.jsonl` file is runtime-generated and never committed. All validation is against synthetic fixtures. Phase 2B is explicitly labeled "awaiting live corpus" in the code.

2. **Go-side not implemented.** The Go Witness does not yet call the `/witness/pre-nuke` endpoint. The Python receiver is implemented and tested. The Go integration requires inserting an HTTP call at `internal/witness/handlers.go:838-844`. This is a ~20-line Go change blocked on Gastown RFC (COUNCIL_VERDICTS.md §Q7 — Gastown RFC in the consolidated top actions list).

3. **NeuroBrain still silent.** NeuroBrain needs raw LLM output text for pattern analysis. Gastown's GtEvent payload does not include this — polecats are Go-orchestrated AI agents that communicate via structured messages, not raw LLM traces. This would require a Gastown API addition.

4. **SemanticBrain marginal.** Synthetic message content is too short (15-80 chars) to produce strong semantic signal. Production `.events.jsonl` messages may be longer.

5. **EconomicBrain activated by synthetic assumptions.** The `resource_amount` and `resource_direction` fields in synthetic payloads are Phase 2B inventions. Real Gastown events do not include resource quantity metadata. This is labeled `[SYNTHETIC ASSUMPTION]` in the code.

6. **SwarmDetectionBrain marginal.** Cross-agent coordination timestamps are present in the suspicious_coordination_stream, but the brain's threshold may require a larger coordinated cluster than the 3-agent synthetic. Needs live corpus at production scale.

7. **In-memory store.** `CoefficientStore` is a module-level dict. In production, it must be backed by Redis or Postgres to survive MayorTap restarts and serve the Witness endpoint from a separate process.

8. **Single-process assumption.** MayorTap and the Witness gate share the `_coefficient_store` only when running in the same Python process. The `/witness/ingest` endpoint provides a mechanism for decoupled operation, but a live deployment needs the two processes configured to use the same store backend.

9. **Authenticated via HMAC-signed bearer tokens; deployment requires operator to set `SYBILCORE_WITNESS_SECRET`.** The shared secret approach protects against network-reachable attackers that lack the secret. It does not protect against a compromised SybilCore node that already holds the secret, nor against a network adversary who can observe the secret in transit (use TLS in production). Mutual TLS or JWT with short-lived keys would provide stronger guarantees for a multi-tenant deployment.

---

## Iteration log

**Round 1 — Initial draft + first test run (51/51 passing):**
- Wrote all 5 deliverables from spec.
- Tests passed on first run (no test failures in initial draft).
- Found `events_since_rescore` tight-loop bug in adversarial review. Fixed.

**Round 2 — Brain metadata enrichment:**
- Adversarial review found that EconomicBrain, DeceptionBrain, CompromiseBrain, SocialGraphBrain were not receiving the right metadata keys from the adapter.
- Extended `_build_runtime_metadata` to extract `resource_amount`, `resource_direction`, `query`, `citations`, `instruction_source`, `instruction_content` from the GtEvent payload dict.
- Extended synthetic firehose to emit `resource_amount` and `resource_direction` in the coordination and sling streams.
- Fixed `/witness/ingest` to use Pydantic request body instead of query params.
- Re-ran tests: 51/51.

**Round 3 — Ruff + type-checking cleanup:**
- Ran `ruff check --fix`. Auto-fixed 22 issues (import ordering, unused imports, f-string cleanup, UP035 Generator import).
- Manually fixed 6 remaining: TC001/TC003 noqa annotations, E501 long docstrings, F841 unused variable.
- Re-ran tests: 51/51. Ruff: all checks passed.

---

## P0 #1 Fix — Witness Auth (2026-04-20)

- **What was wrong:** All three Witness endpoints (`/witness/pre-nuke`, `/witness/ingest`, `/witness/status`) were publicly accessible with no authentication, allowing trivial score injection via `/witness/ingest` and kill-decision polling via `/witness/pre-nuke`.
- **What we built:** HMAC-SHA256 bearer token middleware added as a FastAPI `http` middleware layer. Every request must carry `Authorization: Bearer <hmac-sha256-hex(body, secret)>` and `X-Sybilcore-Timestamp: <unix-epoch>`. Timestamp tolerance is ±60 seconds (replay protection). Secret is loaded lazily from `SYBILCORE_WITNESS_SECRET`; server returns HTTP 503 (not silently continues) if the secret is absent in non-DEV mode. Additional hardening: request body limit (64 KB → HTTP 413), in-process token-bucket rate limiter per IP (100 req/min → HTTP 429), timing-safe comparison via `hmac.compare_digest`. DEV mode bypass (`SYBILCORE_WITNESS_DEV=1`) documented and tested.
- **Tests added/updated:** 31 new tests in `test_witness_auth.py`; 46 existing tests in `test_phase2b.py` updated to use `SigningTestClient` (auto-attaches signed headers). Total: 77 new/updated tests in the Phase 2B suite; 850 total suite passing.
- **Attack surface now:** Score injection via `/witness/ingest` requires the pre-shared secret. Unauthenticated kill-decision polling is blocked. What is NOT yet protected: a compromised caller holding the secret can still inject scores; the in-memory `CoefficientStore` has no persistence or cross-process consistency; mutual TLS or JWT rotation would be needed for a multi-tenant deployment. No SSRF protection has been added to event forwarding paths.
- **Verified:** `pytest integrations/gastown/phase2b_mayor_tap/tests/ tests/` — 850 passed, 40 xfailed, 5 xpassed. Ruff: all checks passed.
