# Gastown × SybilCore — Phase 0 Scout Report

**Clone status:** SUCCESS — cloned to `/tmp/gastown-scout` from `https://github.com/gastownhall/gastown`
**Scout date:** 2026-04-20
**Scout model:** Sonnet 4.6 (Haiku-tier assignment; read-only recon)

---

## Table of Contents
1. [SybilCore Event Model](#1-sybilcore-event-model)
2. [Gastown Event Schema (.events.jsonl + channel events)](#2-gastown-event-schema)
3. [Beads Model (Issues)](#3-beads-model-issues)
4. [Convoy Model](#4-convoy-model)
5. [Agent Identity](#5-agent-identity)
6. [Witness + Deacon Enforcement Surface](#6-witness--deacon-enforcement-surface)
7. [Refinery Hook Points](#7-refinery-hook-points)
8. [Wasteland Stamp Protocol Status](#8-wasteland-stamp-protocol-status)
9. [Replayable Fixtures](#9-replayable-fixtures)
10. [Schema Side-by-Side + Gap Analysis](#10-schema-side-by-side--gap-analysis)
11. [Go Integration Points](#11-go-integration-points)
12. [Phase 0 Report](#phase-0-report)

---

## 1. SybilCore Event Model

**File:** `rooms/engineering/sybilcore/sybilcore/models/event.py`

### Required / typed fields

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `event_id` | `str` | No | `uuid4()` | Unique identifier |
| `agent_id` | `str` | **Yes** | — | The agent that produced this event |
| `event_type` | `EventType` (StrEnum) | **Yes** | — | Category of action |
| `timestamp` | `datetime` (UTC, tz-aware) | No | `datetime.now(UTC)` | Must not be in future (max 60s drift). Timezone-naive timestamps are rejected. |
| `content` | `str` | No | `""` | Max length = `MAX_CONTENT_LENGTH` (from config) |
| `metadata` | `dict[str, Any]` | No | `{}` | Arbitrary brain-specific data |
| `source` | `str` | No | `"unknown"` | Origin system (e.g. `"langchain"`, `"crewai"`) |

### EventType enum values
```
tool_call, message_sent, message_received, resource_access,
permission_request, output_generated, error_raised,
state_change, instruction_received, external_call
```

### CoefficientCalculator input
`CoefficientCalculator.calculate(brain_scores)` takes `list[BrainScore]`, not raw events.
The upstream call is `brain.score(events: list[Event])` — so the **Event list** is the primary
integration surface, not the calculator directly.
File: `rooms/engineering/sybilcore/sybilcore/core/coefficient.py:52`

---

## 2. Gastown Event Schema

There are **two separate event systems** in Gastown. Do not conflate them.

### 2A. `.events.jsonl` — session activity feed log

**File:** `internal/tui/feed/events.go:226-233` (struct `GtEvent`)

```json
{
  "ts":         "<RFC3339 timestamp>",
  "source":     "<string>",
  "type":       "<event type string>",
  "actor":      "<agent address, e.g. gastown/polecats/nux>",
  "payload":    { "rig": "...", "bead": "...", "message": "...", ... },
  "visibility": "feed | both | (other)"
}
```

**Location:** `~/gt/.events.jsonl` (at town root, resolved by `workspace.Find()`)

**Known type values** (from `buildEventMessage` at `internal/tui/feed/events.go:423-539`):
`patrol_started`, `patrol_complete`, `polecat_checked`, `polecat_nudged`, `escalation_sent`,
`sling`, `hook`, `handoff`, `done`, `mail`, `merged`, `merge_failed`

**Visibility filter:** Only events with `visibility == "feed"` or `"both"` are shown in the TUI feed.

**No committed example `.events.jsonl` files in the repo.** The file is runtime-generated. The struct definition is the only spec.

### 2B. Channel event files — `~/gt/events/<channel>/*.event`

**File:** `internal/channelevents/channelevents.go:78-83`

```json
{
  "type":      "<SCREAMING_SNAKE event type>",
  "channel":   "<channel name>",
  "timestamp": "<RFC3339>",
  "payload":   { "key": "value", ... }
}
```

**Location:** `<townRoot>/events/<channel>/<nanos>-<seq>-<pid>.event`  
These are file-drop events consumed by `await-event` watchers (e.g., Refinery watching for `MERGE_READY`).

**Known channel event types** (from tests at `internal/cmd/molecule_emit_event_test.go:17,59`):
`MERGE_READY`, `PATROL_WAKE`, `TEST`

### 2C. Beads `.beads/backup/events.jsonl` — issue tracker events

**File:** `.beads/backup/events.jsonl` (fixture data)

```json
{
  "id":         <int>,
  "issue_id":   "<bead id, e.g. gt-69dai>",
  "event_type": "status_changed | updated | closed | created",
  "actor":      "<agent address>",
  "created_at": "<RFC3339>",
  "new_value":  "<JSON string or plain string>",
  "old_value":  "<JSON string or empty string>",
  "comment":    null
}
```

**Actors seen in fixture:** `mayor`, `gastown/polecats/furiosa`, `gastown/polecats/nux`, `gastown/witness`, `gastown/refinery`

---

## 3. Beads Model (Issues)

**Fixture:** `.beads/backup/issues.jsonl`

Full shape from the fixture (all fields present in the corpus):

```json
{
  "id":                 "<bead-id, e.g. gt-00us>",
  "title":              "<string>",
  "description":        "<markdown string>",
  "status":             "open | hooked | closed | tombstone",
  "priority":           <int 0-4>,
  "issue_type":         "bug | task | feature",
  "assignee":           "<agent address or null>",
  "owner":              "<email string>",
  "created_at":         "<RFC3339>",
  "created_by":         "<agent address>",
  "updated_at":         "<RFC3339>",
  "closed_at":          "<RFC3339 or null>",
  "close_reason":       "<string or empty>",
  "closed_by_session":  "<session id or empty>",
  "compacted_at":       null,
  "compacted_at_commit":null,
  "compaction_level":   <int>,
  "content_hash":       "<hex string>",
  "crystallizes":       <int 0/1>,
  "defer_until":        null,
  "ephemeral":          <int 0/1>,
  "estimated_minutes":  null,
  "external_ref":       null,
  "hook_bead":          "<bead id or empty>",
  "is_template":        <int 0/1>,
  "last_activity":      null,
  "metadata":           "{}",
  "mol_type":           "<string or empty>",
  "notes":              "<string>",
  "original_size":      null,
  "pinned":             <int 0/1>,
  "quality_score":      null,
  "rig":                "<rig name or empty>",
  "role_bead":          "<bead id or empty>",
  "role_type":          "<string or empty>",
  "sender":             "<string>",
  "source_repo":        "<string>",
  "source_system":      "<string>",
  "spec_id":            "<string>",
  "timeout_ns":         <int>,
  "waiters":            "<string>",
  "wisp_type":          "<string>",
  "work_type":          "<string>",
  "acceptance_criteria":"<string>",
  "actor":              "<string>",
  "agent_state":        "<string>",
  "await_id":           "<string>",
  "await_type":         "<string>",
  "design":             "<string>",
  "due_at":             null,
  "event_kind":         "<string>",
  "payload":            "<string>",
  "target":             "<string>"
}
```

**Status transitions in the wild (from events fixture):**
`open` → `hooked` (mayor assigns to polecat) → `closed`
`tombstone` is a deletion marker (import rejects it without `BD_ALLOW_STALE=1`).

**Where beads are stored:** Dolt SQL database. Accessed via `bd` CLI or `internal/beads/` Go package.
Backup in `.beads/backup/` (git-committed JSONL).

---

## 4. Convoy Model

**Primary code:** `internal/beads/fields.go:257-323`, `internal/cmd/convoy_stage.go`

A convoy is itself a **bead** (issue) with structured fields in its `description`:

```
owner: <agent address>        # e.g., "mayor/"
notify: <address>
molecule: <swarm-id>
merge: <strategy>             # "direct" or "pr"
base_branch: <branch>         # target branch for polecats
watchers: <comma-separated>
nudge_watchers: <comma-separated>
```

Convoy ID pattern from fixtures: `hq-cv-xyz` (prefix + `-cv-` + short hash).

**MRFields on individual beads** reference their convoy via description fields
(`convoy_id`, `convoy_created_at`) parsed by `ParseMRFields` at `internal/beads/fields.go:588-672`.

**Convoy lifecycle** (from `StageResult` at `internal/cmd/convoy_stage.go:47-57`):
- `staged_ready` — staged, Wave 1 ready to dispatch
- `staged_warnings` — staged with non-fatal issues
- Error — not created

**Wave structure:** Convoys are dispatched in waves (Wave 1, Wave 2, ...). Each wave contains
independent beads that can run in parallel. Wave ordering is a DAG topological sort.

**No explicit ConvoyStatus enum** — the convoy bead itself has `status: open/closed`.

---

## 5. Agent Identity

**File:** `internal/polecat/types.go:81-105`

The stable identifier for a Polecat is its **Name** field combined with its **Rig**:

```
Address format: <rig>/<polecat-name>
Examples:      gastown/polecats/furiosa, gastown/polecats/nux
```

**Agent address scheme** (from `internal/tui/feed/events.go:196-214`):
- Mayor: `"mayor"` (singleton)
- Deacon: `"deacon"` (singleton)  
- Witness: `"<rig>/witness"` (e.g., `"gastown/witness"`)
- Refinery: `"<rig>/refinery"`
- Crew: `"<rig>/crew/<name>"`
- Polecat: `"<rig>/polecats/<name>"` — note **polecats** (plural) in address

**Polecat struct fields for agent identity:**
```go
type Polecat struct {
    Name      string    // e.g., "furiosa" (NOT the full address)
    Rig       string    // e.g., "gastown"
    State     State     // working | idle | done | stuck | stalled | zombie
    ClonePath string
    Branch    string
    Issue     string    // currently assigned bead ID
    CreatedAt time.Time
    UpdatedAt time.Time
}
```

**There is no standalone `agent_id` field** — identity is always `rig + "/" + name` concatenated at call sites.

---

## 6. Witness + Deacon Enforcement Surface

### Witness enforcement decision chain

**Primary file:** `internal/witness/handlers.go`

The Witness **does not kill polecats autonomously in normal operation**.  
Its enforcement model is: **detect → report → let the Witness AI agent decide**.

Key decision logic (line numbers from handlers.go):

1. **Zombie detection** (`DetectZombiePolecats`, line ~1030-1050):
   - Cross-references polecat `agent_state` in beads vs tmux session liveness
   - Produces `ZombieResult` with `Classification`, `WasActive`, `Action`, `HookBead`
   - **Does NOT automatically kill.** Returns patrol receipts for the agent.

2. **NukePolecat** (line 826-875) — the actual kill function:
   ```go
   func NukePolecat(bd *BdCli, workDir, rigName, polecatName string) error
   ```
   Called explicitly by the Witness AI via CLI (`gt polecat nuke`), NOT called from patrol loop automatically.

   **Safety interlocks before nuke:**
   - `mayor.NewCleanupVetoChecker(townRoot).ShouldVetoCleanup()` — blocks if Mayor ACP active
   - `hasPendingMR(bd, workDir, rigName, polecatName, agentBeadID)` — blocks if MR pending in refinery

3. **AssessHelp** (protocol.go:678) — keyword triage for `HELP:` messages from polecats:
   - Categories: `emergency`, `failed`, `blocked`, `decision`, `lifecycle`, `help`
   - Severities: `critical`, `high`, `medium`
   - Routes emergencies to `overseer`, failures to `deacon`, blocks to `mayor`

4. **PatrolVerdict enum** (patrol_receipts.go:8-10): `stale` | `orphan`

### SybilCore integration point
The Witness's `NukePolecat` (handlers.go:826) and the patrol loop that calls `DetectZombiePolecats` (handlers.go:1030) are the **two surfaces** where SybilCore's FLAGGED/ENFORCEMENT tiers plug in:
- **Pre-nuke gate:** Before calling `NukePolecat`, check SybilCore coefficient. ENFORCEMENT tier = allow nuke; FLAGGED tier = escalate to Deacon first.
- **Patrol scoring:** Translate `ZombieResult.Classification` into a SybilCore `EventType.STATE_CHANGE` event and feed it to the scanner.

### Protocol message types (witness/protocol.go:55-68)
```
polecat_done, lifecycle_shutdown, help, merged, merge_failed,
merge_ready, handoff, swarm_start, dispatch_attempt,
dispatch_ok, dispatch_fail, idle_passivated
```

---

## 7. Refinery Hook Points

**Primary files:** `internal/refinery/engineer.go`, `internal/refinery/types.go`, `internal/refinery/score.go`

### Gate system (engineer.go:69-156)

```go
type GateConfig struct {
    Cmd     string        `json:"cmd"`
    Timeout time.Duration `json:"timeout"`
    Phase   GatePhase     `json:"phase"` // "pre-merge" | "post-squash"
}
```

Gates are registered in `MergeQueueConfig.Gates map[string]*GateConfig`.

**Hook insertion point:** Add a SybilCore gate to the `Gates` map in the rig's `MergeQueueConfig`:
```json
"gates": {
  "sybilcore": {
    "cmd": "python3 -m sybilcore.integrations.gastown.gate --agent {{worker}} --threshold 60",
    "timeout": 30000000000,
    "phase": "pre-merge"
  }
}
```

### MR phase state machine (types.go:62-100)
```
ready → claimed → preparing → prepared → merging → merged (terminal)
                                       ↘ rejected (terminal)
                           ↘ failed → ready (retry)
```

`MRPhasePreMerge` quality gates run in the `preparing → prepared` transition.

### ScoreMR function (score.go:88-132)
```go
func ScoreMR(input ScoreInput, config ScoreConfig) float64
```
Inputs: `Priority int`, `MRCreatedAt time.Time`, `ConvoyCreatedAt *time.Time`, `RetryCount int`, `Now time.Time`

**SybilCore coefficient gate** inserts at `GatePhasePreMerge`. The gate script receives the polecat identity (`worker` field on MR bead) and can query SybilCore's `/score` endpoint.

### Channel event that triggers refinery processing
```
MERGE_READY <polecat-name>      (emitted to channel "refinery" by Witness)
```
File: `internal/witness/protocol.go:26,163`
The Refinery watches for `MERGE_READY` events on its channel, then claims the MR bead.

---

## 8. Wasteland Stamp Protocol Status

**Status: EXISTS IN CODE — functional, Phase 1 ("wild-west mode")**

**Files:**
- `internal/wasteland/wasteland.go` — join/fork/register workflow
- `internal/wasteland/trust.go` — trust tier escalation engine
- `internal/wasteland/spider.go` — collusion detection (Spider Protocol)

### What exists

**Trust tiers** (trust.go:28-35):
```
0 = Drifter     (unverified, just joined)
1 = Registered  (completed DoltHub identity verification)
2 = Contributor (earned stamps through validated completions)
3 = War Chief   (sustained high-quality, can validate others)
```

**Stamp schema** (queried in trust.go:264-276 from `stamps` table):
```sql
SELECT COUNT(*), AVG(JSON_EXTRACT(valence, '$.quality')), COUNT(DISTINCT author)
FROM stamps
WHERE subject = '<rig_handle>'
```

**Valence JSON shape** (inferred from SQL): `{ "quality": <float 1-5> }`

**Escalation requirements** (trust.go:92-116):
- `Registered → Contributor`: 3 completions, 3 stamps, 2 distinct validators, 7 days in tier
- `Contributor → War Chief`: 10 completions, 15 stamps, avg quality ≥ 3.5, 5 validators, 30 days

**RigTrustProfile struct** (trust.go:119-130):
```go
Handle, CurrentTier, TierSince, CompletionCount, StampCount, AvgQuality, DistinctValidators
```

**WLCommonsStore interface** used by `wl_claim.go` to claim wanted items (sets `claimed_by`, `status='claimed'`).

### What is NOT in code (design-only)
- The DoltHub PR-based voting flow for promotions is designed but not yet implemented (references "Phase 1 wild-west mode" throughout — direct writes to main, not PR-gated).
- Cross-rig stamp issuance UI (`gt wl stamp`) was not found in the Go sources inspected.

### SybilCore mapping
Gastown's `TrustTier` (0-3) maps conceptually to SybilCore's trust tiers:
```
Drifter     → ENFORCEMENT (untrusted)
Registered  → FLAGGED
Contributor → CLOUDED / LATENT
War Chief   → CLEAR
```
But the **directionality is inverted**: Gastown trust is about the **rig** earning credibility in the federation; SybilCore coefficient is about **behavioral risk score** of an agent at runtime.

---

## 9. Replayable Fixtures

All fixtures are committed and stable. Phase 1A agent can replay them without network access.

| File | Contents | Size (approx) | Use |
|------|----------|---------------|-----|
| `/tmp/gastown-scout/.beads/backup/events.jsonl` | Bead lifecycle events (status changes, creations, closes) | ~25 events | Test event-to-SybilCore mapping |
| `/tmp/gastown-scout/.beads/backup/issues.jsonl` | Full bead corpus (all issues ever tracked) | ~100+ beads | Test full bead schema parsing |
| `/tmp/gastown-scout/.beads/backup/dependencies.jsonl` | Dependency edges between beads | ~10 edges | Test convoy/DAG structure |
| `/tmp/gastown-scout/.beads/backup/config.jsonl` | Bead tracker config | 1 entry | N/A |
| `/tmp/gastown-scout/.beads/backup/labels.jsonl` | Label definitions | few entries | Test label parsing |
| `/tmp/gastown-scout/.beads/backup/comments.jsonl` | Issue comments | varies | N/A |

**Notable replay events from `.beads/backup/events.jsonl`:**

1. `event_type: "status_changed"` — actor `mayor`, issue `gt-69dai`, `old_value` contains full issue JSON, `new_value` contains `{"assignee": "gastown/polecats/furiosa", "status": "hooked"}` — maps to SybilCore `STATE_CHANGE`
2. `event_type: "closed"` — actor `gastown/polecats/furiosa`, `close_reason: "no-changes: Test failure was already fixed..."` — maps to SybilCore `OUTPUT_GENERATED`
3. `event_type: "updated"` — actor `gastown/witness` — maps to SybilCore `MESSAGE_SENT`
4. `event_type: "created"` — actor `gastown/refinery` — maps to SybilCore `TOOL_CALL`

---

## 10. Schema Side-by-Side + Gap Analysis

### Side-by-side: Gastown bead event vs SybilCore Event

| Gastown Field | SybilCore Field | Mapping | Notes |
|---------------|----------------|---------|-------|
| `actor` (agent address) | `agent_id` | Direct | Gastown: `"gastown/polecats/nux"` → SybilCore: same string |
| `event_type` | `event_type` (EventType) | Translate | See table below |
| `created_at` | `timestamp` | Direct | Both RFC3339. Gastown uses UTC-Z, SybilCore validates tz-aware |
| `issue_id` | `metadata["bead_id"]` | Into metadata | No direct field |
| `new_value` | `content` | Partial | new_value is often JSON; content is human-readable |
| `old_value` | `metadata["old_value"]` | Into metadata | Large JSON blobs — truncate to MAX_CONTENT_LENGTH |
| `id` (int) | `metadata["beads_event_id"]` | Into metadata | Gastown auto-increment; SybilCore uses UUID |
| `comment` | `metadata["comment"]` | Into metadata | Usually null |
| — | `event_id` | Auto-generate | uuid4() on import |
| — | `source` | Hardcode | `"gastown"` |

### Gastown event_type → SybilCore EventType mapping

| Gastown event_type | SybilCore EventType | Rationale |
|-------------------|-------------------|-----------|
| `status_changed` (to `hooked`) | `STATE_CHANGE` | Agent receives work assignment |
| `status_changed` (to `closed`) | `STATE_CHANGE` | Work completion state change |
| `created` | `TOOL_CALL` | Agent creates a new artifact |
| `updated` | `MESSAGE_SENT` | Agent updates an issue |
| `closed` (by polecat) | `OUTPUT_GENERATED` | Polecat signals work output |
| `closed` (by witness) | `MESSAGE_SENT` | Witness administrative action |
| POLECAT_DONE (protocol) | `STATE_CHANGE` | Agent lifecycle transition |
| HELP: (protocol) | `PERMISSION_REQUEST` | Agent requesting elevated access to human oversight |
| MERGE_READY (channel) | `EXTERNAL_CALL` | Agent signaling external system (Refinery) |
| MERGED (protocol) | `OUTPUT_GENERATED` | Final output of merge work |
| MERGE_FAILED (protocol) | `ERROR_RAISED` | Merge pipeline error |

### Gap Analysis

**Fields Gastown has that SybilCore lacks:**
- `issue_id` — the specific bead being acted on (key for behavioral correlation)
- `old_value` — before-state of an issue (needed for diff-based scoring)
- `hook_bead` — which bead the agent is pinned to (agent assignment pointer)
- `mol_type`, `wisp_type`, `work_type` — molecule and wisp classification
- `rig` — which rig the agent belongs to (multi-rig federation context)
- `convoy_id` — batch grouping context
- `close_reason` — free-text rationale for work outcome
- `ephemeral`, `crystallizes` — bead lifecycle flags
- `compaction_level` — session squash metadata

**Fields SybilCore needs that Gastown lacks:**
- `source` — Gastown has no explicit source field; must be hardcoded `"gastown"` on ingest
- `content` (human-readable description) — Gastown stores structured JSON in `new_value`, not prose
- `event_id` — must be generated (UUID) on the adapter side; Gastown uses an integer sequence
- **Timezone-aware timestamps** — Gastown timestamps are RFC3339 with `Z` suffix (UTC). SybilCore's validator requires `tzinfo` to be set. The `Z` suffix is parsed correctly by Python's `datetime.fromisoformat()` only in Python 3.11+. **Verify parser handles `Z` suffix on the SybilCore side.**
- Brain-specific `metadata` keys — each of SybilCore's 13 brains looks for different keys. The adapter needs to populate relevant brain keys from Gastown's event payload.

---

## 11. Go Integration Points

### Phase 1A: Adapter (translate Gastown events to SybilCore Events)

| File | Line Range | What to add |
|------|-----------|-------------|
| `internal/beads/fields.go` | 257-323 | Read `ConvoyFields` to populate `metadata["convoy_id"]` |
| `internal/townlog/logger.go` | 49-54 | EventType constants — the `EventType` values here match what goes into `.events.jsonl` `"type"` field |
| `internal/tui/feed/events.go` | 218-233 | `GtEvent` struct — this is the canonical `.events.jsonl` line schema |

### Phase 2B: Witness gate (SybilCore coefficient check before nuke)

| File | Line | What to add |
|------|------|-------------|
| `internal/witness/handlers.go` | 826 | `NukePolecat` — insert SybilCore coefficient check BEFORE the `ShouldVetoCleanup()` call. If agent coefficient < ENFORCEMENT threshold, return early with error. |
| `internal/witness/handlers.go` | 838-844 | Existing safety gates — new SybilCore gate goes here as third gate after MR-pending check |
| `internal/witness/patrol_receipts.go` | 49-73 | `BuildPatrolReceipt` — optionally annotate `PatrolReceipt` with SybilCore coefficient in `Evidence` |

### Phase 2C: Refinery gate (SybilCore coefficient check before merge)

| File | Line Range | What to add |
|------|-----------|-------------|
| `internal/refinery/engineer.go` | 69-98 | `GateConfig` struct — add a `sybilcore` gate entry to `MergeQueueConfig.Gates` |
| `internal/refinery/engineer.go` | 80-83 | `GatePhasePreMerge` — this is the phase constant to use for the SybilCore gate |
| `internal/refinery/types.go` | 62-100 | Phase state machine — gate runs in `preparing → prepared` transition |
| `internal/refinery/score.go` | 88 | `ScoreMR` — input has `Worker` field (polecat name) needed to look up SybilCore coefficient |

### Phase 2D: Wasteland trust ↔ SybilCore tier sync

| File | Line Range | What to add |
|------|-----------|-------------|
| `internal/wasteland/trust.go` | 119-130 | `RigTrustProfile` — map `CurrentTier` (0-3) to SybilCore tier (ENFORCEMENT/FLAGGED/CLOUDED/LATENT/CLEAR) |
| `internal/wasteland/trust.go` | 147-217 | `EvaluateEscalation` — this is pure Go, injectable: pass SybilCore snapshot as additional input |
| `internal/cmd/wl_claim.go` | 80-97 | `claimWanted` — pre-claim check could verify SybilCore tier of claiming rig |

---

## Phase 0 Report

**Why we're implementing:** Establish ground truth so Phases 1-2 don't waste tokens on wrong assumptions about Gastown's schema, enforcement model, and available integration surfaces.

**What we hoped to gain:** Exact schemas, integration surfaces, real fixtures, confirmation of Wasteland stamp protocol existence.

**What we actually gained:**
- Full bead schema (41+ fields documented from fixture corpus)
- Two distinct event systems clarified (`.events.jsonl` session log vs channel `.event` files) — critical distinction; conflating them would have derailed Phase 1A
- Confirmed: Witness does NOT auto-kill polecats in the patrol loop. NukePolecat is an explicit call with two existing safety gates. SybilCore gate inserts cleanly as a third gate.
- Confirmed: Refinery gate system is a named map (`MergeQueueConfig.Gates`) — SybilCore plugs in as a shell command gate with no Go code changes needed in Refinery itself
- Confirmed: Wasteland stamp protocol is live code (not vaporware), Phase 1 "wild-west mode" (direct Dolt writes, not PR-gated). Trust tiers 0-3 exist with enforcement criteria.
- Confirmed: Convoy is a bead with structured description fields — not a separate database table
- No `.events.jsonl` example files committed — struct definition is the only spec

**Tests run + results:**
- Successfully cloned from `https://github.com/gastownhall/gastown.git` to `/tmp/gastown-scout`
- Parsed `.beads/backup/events.jsonl` (25+ events, schema verified)
- Parsed `.beads/backup/issues.jsonl` (100+ beads, full field inventory done)
- Read 15+ key Go source files across `internal/witness/`, `internal/refinery/`, `internal/wasteland/`, `internal/polecat/`, `internal/channelevents/`, `internal/beads/`, `internal/townlog/`

**Honest limitations:**
1. The Witness AI agent's actual patrol molecule (the YAML/formula that drives the patrol loop) was not found in the Go source — it likely lives in a templates directory or is runtime-configured. The Go code shows the *functions* the patrol calls, but not the decision sequence the AI follows.
2. `internal/refinery/engineer.go` was only partially read (first 180 lines). The actual gate execution loop (where `runGates()` is called) was not inspected. Phase 2C agent should read the full file.
3. No `.events.jsonl` runtime examples exist in the repo. The struct spec is solid but the Phase 1A agent cannot test against real event data without running Gastown.
4. Spider Protocol (`internal/wasteland/spider.go`) was identified as existing but not read in detail. It implements collusion detection for stamp validation. Phase 2D agent should read it.
5. `internal/beads/` has many files not inspected (`store.go`, `status.go`, `issue.go`). The `Issue` struct in Go may have more fields than the JSONL backup reveals.
6. Deacon's role in enforcement was referenced (escalation target for `HELP` messages) but `internal/deacon/` was not inspected. Phase 2B agent should read it.
