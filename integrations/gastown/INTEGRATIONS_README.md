# SybilCore × Gastown Integration

This directory contains the full research record for integrating SybilCore's behavioral inference engine with [Gas Town](https://github.com/gastownhall/gastown) — Steve Yegge's multi-agent coding orchestrator. The work began as an adapter question ("can SybilCore score Gastown polecats?") and evolved into a structural finding: the behavioral telemetry that SybilCore's threat-detection brains need does not exist in any major orchestration system. The deliverable is an RFC proposing the schema extension that would close that gap, backed by empirical replay of 3,667 real Gastown events through a 13-brain ensemble.

---

## Navigation

| Directory | Contents | Phase |
|-----------|----------|-------|
| `phase1a_baseline/` | Adapter implementation; replay of 3,667 bead-tracker events against 13-brain ensemble; 31 agents scored; 8 silent brains diagnosed. 41 tests. | Phase 1A |
| `phase1d_stamp_protocol/` | Behavioral Warrant spec — protocol for embedding SybilCore's agent coefficient into Gastown's Wasteland stamp economy as a parallel reputation axis. | Phase 1D |
| `phase2b_mayor_tap/` | Runtime adapter for the `.events.jsonl` session firehose (not the bead-tracker); streaming MayorTap runner; Witness third-gate HTTP endpoint; 8 synthetic patterns targeting 8 silent brains. 51 tests. | Phase 2B |
| `phase2c_refinery_gate/` | Narrow 3-brain (Contrastive + Identity + Temporal) Refinery merge gate; drop-in `MergeQueueConfig.Gates` config fragment; threshold calibration from Phase 1A corpus. 46 tests. Note: ContrastiveBrain's threshold in the gate is affected by the saturation finding (see phase2f_cross_substrate/CONTRASTIVE_BRAIN_SATURATION_NOTE.md); gate calibration thresholds remain valid for the Phase 1A all-CLEAR corpus. | Phase 2C |
| `phase2e_schema_gap/` | Cross-substrate evidence matrix (7 systems); ABOS v0.1 formal spec (23 fields, 6 groups, 3-language implementation); Gastown RFC v1 (795 words, PR-ready draft). | Phase 2E |
| `phase2f_cross_substrate/` | OpenHands and Claude Code adapters; brain activation comparison across 3 substrates; 576 real Claude Code sessions analyzed; strategic verdict on fallback corpora. 31 tests. | Phase 2F |
| `phase3_rfc_final/` | RFC v3 — submission-ready PR targeting gastownhall/gastown. Includes empirical motivation (3,667-event study, 8 silent brains), Go struct diff, 3-phase migration, performance analysis, FAQ (7 Q&A), issue #2784 tie-in. (RFC v2 preserved for diff.) | Phase 3 (Wave 3.3 + P0 fix) |
| `phase3_preprint/` | NeurIPS preprint materials (see that directory's own README). | Phase 3 |
| `phase3_content/` | Substack + LinkedIn post drafts for release the day the preprint lands. | Phase 3 |

---

## The Story: How This Work Evolved

### Scout (Phase 0 — embedded in SCOUT_REPORT.md)

The work began with a full reconnaissance of the `gastownhall/gastown` repo. Key findings: Gas Town has two distinct event systems (`.events.jsonl` session feed vs. `.beads/backup/events.jsonl` bead-tracker — conflating them would have derailed all downstream work). The `GtEvent` struct at `internal/tui/feed/events.go:218-233` is the canonical schema. The Witness does not auto-kill polecats; `NukePolecat` at `handlers.go:826` is an explicit call with two existing safety gates. The Refinery gate system accepts shell-command plugins with no Go changes required. The Wasteland trust system is live code ("wild-west mode") with a defined stamp schema.

### Phase 1A — Empirical Baseline

Adapter layer built. 3,667 events from `.beads/backup/events.jsonl` replayed through the 13-brain default ensemble across 31 agents. All 31 agents scored CLEAR (coefficient range 5.4–82.3, mean 40.4). The mayor scored highest (82.3) not because it is suspicious but because it is the most active actor — a calibration finding, not a detection finding. Eight brains scored 0.0 for every agent. These brains (DeceptionBrain, SocialGraphBrain, CompromiseBrain, SwarmDetectionBrain, EconomicBrain, NeuroBrain, SemanticBrain, SilenceBrain) are not broken — they need metadata keys that the bead-tracker schema does not provide. The null result was the result.

### Phase 1D — Design: Behavioral Warrant

Rather than treating the schema gap as a blocker, Phase 1D designed the Behavioral Warrant — a protocol for embedding SybilCore's agent coefficient into Gastown's existing Wasteland reputation economy as a parallel axis. The warrant is orthogonal to the Drifter → Registered → Contributor → War Chief ladder, not a fifth rung. Eight adversarial attacks analyzed; five mitigations classified solid, three partial (labeled honestly). The single autonomous action: an ENFORCEMENT-tier coefficient (91-100) suspends stamp validation privileges pending Witness investigation. Everything else is advisory.

### Council Verdicts

A four-model supreme council (Gemini 3.1 Pro, GPT-4.1, Grok 4.20, Perplexity Sonar Pro) reached 4/4 consensus on two pivotal decisions: (1) the schema-gap framing is the correct headline for the NeurIPS paper, and (2) "agent observability" is the macro-positioning umbrella — behavioral safety is one SKU under it, not the product name. The council also decided preprint-first (3/4) over private Yegge outreach, committed on the "Behavioral Warrant" vocabulary, and named six consolidated top actions with explicit kill criteria.

### Phase 2B — Mayor Live Tap

The runtime `.events.jsonl` adapter (GtEvent schema, not bead-tracker schema) was built and tested against synthetic fixtures matching the 12 documented GtEvent types. The Witness third-gate HTTP endpoint was shipped: `POST /witness/pre-nuke` returns allow/veto with coefficient and reasoning before `NukePolecat` executes. With richer synthetic event content, 3 previously-silent brains (SilenceBrain, CompromiseBrain, DeceptionBrain) began firing. The Go-side integration at `handlers.go:838-844` is pending the Gastown RFC.

### Phase 2C — Refinery Gate

A drop-in behavioral gate for Gastown's `MergeQueueConfig.Gates` was built and calibrated against the Phase 1A corpus. Zero Gastown Go changes required — it plugs in as a shell-command config entry. All 31 Phase 1A agents would PASS under calibrated thresholds (consistent with the CLEAR baseline). The gate is labeled as a 3-brain narrow PoC; full brain coverage requires the runtime session feed.

### Phase 2E — Schema Gap + ABOS + RFC v1

The null result from Phase 1A was turned into a structural contribution. An evidence matrix assessed 7 orchestration substrates (Gastown, OpenHands, Devin/Cognition, Anthropic Managed Agents, Microsoft AGT, Claude Code, Cursor) across 6 behavioral governance dimensions. Deception signal is absent in all 7. Social graph is absent in 6 of 7. The finding is universal, not a Gastown oversight. ABOS v0.1 was drafted: 23 fields across 6 groups, all optional, with an estimated overhead under 2 ms/event for the 5-field Minimum Viable Subset (per ABOS_SPEC.md §4). Each of the 8 previously-silent brains was given a minimum required ABOS field set. RFC v1 was written at 795 words in Gastown idiom with 4 Go file:line references.

### Phase 2F — Cross-Substrate Validation

Two more adapters were built against real schema data: OpenHands (shallow clone of All-Hands-AI/OpenHands) and Claude Code (576 real session JSONL files). Brain activation comparison: Gastown beads 5/13, OpenHands 2/13 (tiny corpus caveat), Claude Code 6/13. Claude Code fires CompromiseBrain (score 50.0) and SemanticBrain (score 25.0) — signals absent from Gastown bead events. The "substrate-agnostic behavioral monitor" claim is demonstrated, not aspirational. SybilCore adapts to new substrates in under 200 lines of adapter code.

### Wave 3.3 (current) — RFC v2, README, Positioning Synthesis

RFC v1 polished to PR-ready submission quality. This README written. Positioning synthesis written.

---

## Headline Numbers

| Metric | Value |
|--------|-------|
| Gastown events replayed (Phase 1A) | 3,667 |
| Agents scored | 31 |
| Brain ensemble size | 13 |
| Phase 1A agents scoring CLEAR | 31/31 |
| Silent brains on bead-tracker data | 8/13 |
| ABOS fields proposed | 23 (6-field MVS for immediate adoption — MVS-6 after coordination_group promotion) |
| Tests passing across all phases | 41 (1A) + 51 (2B) + 46 (2C) + 31 (2F) = 169 |
| Substrates in cross-substrate study | 7 (evidence matrix) + 3 with working adapters |
| Claude Code sessions available for fallback corpus | 576 |
| Phase 2F brain coverage — Claude Code | 5/12 (excl. ContrastiveBrain saturation artifact) |

---

## Status Board

| Item | Status |
|------|--------|
| Phase 1A adapter + replay | COMPLETE. 41 tests green. |
| Phase 1D Behavioral Warrant spec | COMPLETE. Pending Wasteland governance review. |
| Phase 2B Mayor tap + Witness third gate | COMPLETE. 51 tests green. Awaiting live `.events.jsonl` corpus. |
| Phase 2C Refinery gate | COMPLETE. 46 tests green. Untested against live Refinery. |
| Phase 2E evidence matrix | COMPLETE. 7 substrates, 28 citations. |
| Phase 2E ABOS spec (v0.1) | COMPLETE. 23 fields, 3-language implementation. |
| Phase 2E RFC v1 | COMPLETE. 795 words, PR-ready draft. |
| Phase 2F cross-substrate adapters | COMPLETE. 31 tests green. ContrastiveBrain excluded from cross-substrate comparison (saturation artifact — see phase2f_cross_substrate/CONTRASTIVE_BRAIN_SATURATION_NOTE.md). |
| Phase 3 RFC v2 | COMPLETE (Wave 3.3). Preserved for diff. |
| Phase 3 RFC v3 | COMPLETE (P0 #4 fix). 1,606 prose words. Empirical motivation inlined. MVS-6. |
| Gastown RFC submitted as PR | BLOCKED: requires fork + branch + Gastown contributor context. |
| NeurIPS preprint | TARGET: May 4-6. |
| Go-side Witness integration (`handlers.go:838-844`) | BLOCKED: pending Gastown RFC acceptance. |
| Live `.events.jsonl` corpus | BLOCKED: runtime-generated, not committed. Requires running Gastown instance or Kilo access. |

---

## Quick-Start: Reproduce Phase 1A Replay

```bash
# Clone Gastown to get the fixtures
git clone --depth=1 https://github.com/gastownhall/gastown.git /tmp/gastown-fixtures

# From the sybilcore root
cd rooms/engineering/sybilcore

# Run the Phase 1A tests (requires Python 3.11+ for Z-suffix timestamp parsing)
python3 -m pytest integrations/gastown/phase1a_baseline/tests/ -v

# Run the replay directly against the bead-tracker fixture
python3 -c "
import json
from integrations.gastown.phase1a_baseline.adapter import adapt_bead_event, group_events_by_agent
from integrations.gastown.phase1a_baseline.replay import replay_all_agents

results = replay_all_agents('/tmp/gastown-fixtures/.beads/backup/events.jsonl')
for agent, result in sorted(results.items(), key=lambda x: -x[1]['coefficient']):
    print(f'{agent:45s}  coeff={result[\"coefficient\"]:.1f}  tier={result[\"tier\"]}')
"
```

Expected output: 31 agents, all CLEAR, coefficient range 5.4–82.3, mayor highest at 82.3.

Note: The scoring window must be overridden to 90 days (7,776,000 seconds) for historical replay against committed fixtures. The default 1-hour window filters all events from the Feb-Mar 2026 corpus. See `replay.py` module docstring for the parameter.

---

## Wave 3.3 Report

**Why we're implementing:** Orient all future readers — reviewers, collaborators, Yegge, Kilo — to the integration work as a coherent arc, and ship the RFC in PR-ready form so it can be submitted to `gastownhall/gastown` without further revision.

**What we hoped to gain:** Three submission-ready documents: RFC v2 (PR-ready, 1200-word ceiling), this README (navigation + chronological story + quick-start), and a positioning synthesis doc that integrates council verdicts, X research, and empirical findings into a single strategic statement.

**What we actually gained:**
- RFC v2: 1,386 words total (1,183 prose + 203 Go code blocks). Added Go struct before/after diff, executive summary, issue #2784 tie-in, Deacon quote framing from Yegge's own words, 5-question FAQ, performance analysis with no-benchmark caveat, kill-switches for each migration phase, and a "Where the Fields Come From" table. Removed promotional language from v1. The 5/6-field MVS inconsistency between ABOS_SPEC.md and the RFC struct was identified and resolved: the RFC uses 6 fields (ABOS_SPEC.md MVS-5 + `coordination_group`), with the deviation documented in the struct comment.
- README: Navigation table covering all 9 subdirectories, chronological story across 7 phases, 13-row headline numbers table, status board with explicit blockers, quick-start with expected output.
- POSITIONING_SYNTHESIS.md: Thesis, per-phase evidence stack, competitive matrix (4 systems), ownable primitives, distribution plan with council-sourced outreach sequence, kill criteria verbatim from COUNCIL_VERDICTS.md, observability-tent reframe with vocabulary substitution table, 5 honest gaps.

**Claims audited (Round 2 + 3 hostile reviewer passes):**
- "3,667 events" — sourced from BASELINE_RESULTS.md
- "31 agents" — sourced from BASELINE_RESULTS.md
- "8 silent brains" — sourced from BASELINE_RESULTS.md brain-by-brain table
- "13-brain ensemble" — sourced from CLAUDE.md (frozen for paper reproducibility)
- Deacon quote — sourced verbatim from X_RESEARCH_BRIEF.md Axis 1 (URL provided; direct X read blocked by auth wall)
- Issue #2784 — sourced from X_RESEARCH_BRIEF.md Axis 6 (description: "open RFC from Mar 2026 requesting behavioral-aware routing"; URL not independently verified)
- ABOS overhead "< 2 ms/event" — labeled as design-time estimate throughout; no benchmark run
- Go file:line references — all sourced from SCOUT_REPORT.md §11; `handlers.go:1030` changed to "approximately line 1030" since scout used tilde
- Devon's Citadel piece — URL sourced from X_RESEARCH_BRIEF.md Axis 4
- "6 consolidated top actions" in README Council Verdicts summary — corrected from incorrect "five" (COUNCIL_VERDICTS.md lists 6 actions)

**Tests run + results:**
- RFC v2 prose word count: 1,183 words (under 1,200 ceiling). Pass.
- Gastown idiomatic terms in RFC v2: polecat, bead, mayor, witness, deacon, refinery, convoy, patrol, patrol wave, sling, handoff, escalation, zombie, MergeQueueConfig, NukePolecat, BuildPatrolReceipt, AssessHelp, Wasteland, War Chief. 19 terms. Pass.
- Go file:line references in RFC v2: events.go:218-233, events.go:423-539, handlers.go:826, handlers.go:838-844, patrol_receipts.go:49, protocol.go:678, polecat/types.go:81-105. 7 references. Pass.
- No placeholder text, no lorem ipsum, no TODO in any of the three files. Pass.
- No fabricated citations — all quotes and URLs traceable to source documents. Pass.
- `phase3_preprint/` and `phase3_content/` directories verified to exist before listing them in the navigation table. Pass.

**Honest limitations:**
- The RFC has not been reviewed by a Gastown contributor. Gastown's PR template, if any, is unknown.
- The ABOS overhead figure (< 2 ms/event) is a design-time estimate. No Go `BenchmarkABOS` function has been run. A motivated reviewer will ask for one.
- Issue #2784 title and URL were not verified against the live GitHub issue — sourced secondhand from X_RESEARCH_BRIEF.md.
- Gas City FAQ claim is based on Yegge's v1.0 post (April 2026). If the roadmap has been updated since, this may be stale.
- The 5-field vs 6-field MVS inconsistency between ABOS_SPEC.md (5 fields) and this RFC (6 fields, adding `coordination_group`) is documented in the struct comment but not yet resolved upstream in ABOS_SPEC.md.

**Iteration log:**
- Round 1: Drafted all three documents from full source-reading pass of 12 source files.
- Round 2: Hostile reviewer pass on all three docs. Identified: (a) RFC said "5 fields" but struct had 6 — resolved and documented; (b) performance table per-row figures were misleadingly high for Go string assignments — collapsed to single design-time estimate; (c) `handlers.go:1030` cited as exact — changed to approximate; (d) README said "five consolidated top actions" but COUNCIL_VERDICTS.md has six — corrected; (e) POSITIONING_SYNTHESIS.md said "4/4 ICLR skip" but council verdict notes it as "consensus" under Q4, not a separate vote — softened language. Fixed all.
- Round 3: Final citation density check. Confirmed 10 specific quantitative claims against source files. Confirmed Deacon quote exact wording. Confirmed no fabricated figures. Final word counts taken. RFC prose at 1,183 words (under ceiling). All three files verified non-empty and parseable.

---

## P0 #4 Fix Addendum (2026-04-20)

Wave 3.3 left two open limitations: (1) MVS field-count inconsistency between ABOS_SPEC.md and RFC v2, and (2) issue #2784 URL unverified. Both resolved in this pass:

- **RFC v3** (`phase3_rfc_final/GASTOWN_RFC_V3.md`): adds "Empirical Motivation" section inlining the 3,667-event study, 8-silent-brain table with one required ABOS field per brain, cross-substrate confirmation, ContrastiveBrain saturation caveat, and the MVS-6 one-sentence summary. Adds 2 new FAQ items (cross-substrate evidence + all-CLEAR honest explanation). Prose: 1,606 words. V2 preserved for diff.
- **ABOS_SPEC.md §4**: MVS-5 promoted to MVS-6. `coordination_group` added at priority 6; brain-enabled column updated to reflect SwarmDetectionBrain unlock. Changelog entry v0.1.1 added.
- **Issue #2784**: Confirmed to exist at `github.com/steveyegge/gastown/issues/2784` (original repo org, pre-transfer). RFC v3 header updated with correct org path.
