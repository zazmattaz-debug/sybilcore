# Behavioral Warrant — Stamp Protocol Spec

**Phase 1D | Gastown × SybilCore Integration**
**Date:** 2026-04-20
**Status:** Draft — pending Wasteland governance review

---

## 1. Goal & Non-Goals

### Goal

Introduce a **Behavioral Warrant** — a persistent reputation stamp derived from SybilCore's
Agent Coefficient — that gives Gastown's Wasteland trust system automated behavioral signal
without displacing the human-anchored stamp economy Yegge designed.

A Behavioral Warrant answers one question: *does this rig's runtime behavioral fingerprint
corroborate the credibility it has earned through completed work?*

### Non-Goals

- **Does not replace human stamps.** Human validators issuing quality stamps through the
  `stamps` table remain the primary escalation gate for Contributor and War Chief tiers.
  The Warrant is corroborating evidence, not a substitute.
- **Does not execute enforcement autonomously.** A Warrant cannot nuke a polecat, revoke
  a human stamp, or block a merge on its own authority. It surfaces signal; humans and the
  existing Witness/Refinery gate machinery act on it.
- **Does not claim information-theoretic fraud detection.** SybilCore's adversarial floor
  at coefficient 12.14 ± 1.77 (5-seed sweep, 13-brain default) is an empirical evasion
  asymptote under the tested threat model — not a guarantee. This protocol makes that
  limitation explicit in every warrant it emits.
- **Does not gate Drifter-to-Registered.** DoltHub identity verification is a human-process
  gate. The Warrant has no standing on identity claims — only on behavioral patterns after
  identity is established.
- **Does not score individual beads.** The Warrant scores the *rig* (federated identity)
  across a behavioral window, not individual bead completions.

---

## 2. Naming & Positioning

### Name: Behavioral Warrant

**Rationale for "Warrant":** In Yegge's Wasteland idiom, stamps are endorsements from peers
who have *witnessed* your work. A Warrant is an endorsement from a different kind of witness —
the behavioral observation layer. It carries legal-adjacent connotation (a document that grants
authority based on accumulated evidence) without importing SybilCore's internal vocabulary.
Alternative considered: "Conduct Seal" — rejected because "seal" implies finality; warrant
implies ongoing review. "Warrant" also rhymes naturally with "wanted" — the dual meaning is
intentional given the Wasteland's wanted-item mechanic.

### Tier Position: Parallel Axis, Not a Fifth Tier

The Warrant sits **orthogonal** to the Drifter → Registered → Contributor → War Chief ladder.
It is an attribute stamped onto a `RigTrustProfile`, not a rung on the ladder.

```
Drifter | Registered | Contributor | War Chief
        ↑ Warrant axis (independent) ↑
        UNWARRANT  PROVISIONAL  WARRANTED  SENIOR
```

Rationale: Yegge's tiers are about credibility *earned through work*. A rig could be a
War Chief (massive completed work) with a bad Warrant (recently showing anomalous behavior).
These are separate claims. Conflating them into a fifth tier would destroy the clean semantic
of "work is the only input, reputation is the only output" — because behavioral signals are
not work outputs.

**The four Warrant levels:**

| Warrant Level | Meaning | Display |
|--------------|---------|---------|
| `UNWARRANTED` | No Warrant issued yet (default) | — |
| `PROVISIONAL` | Warrant issued, under observation window | `[P]` |
| `WARRANTED` | Warrant confirmed over full window | `[W]` |
| `SENIOR` | Sustained WARRANTED through one full tier-up cycle | `[SW]` |

---

## 3. Inversion Handling

### The Problem

SybilCore coefficient: 0 = safest, 100 = most dangerous (risk score).
Gastown trust: 0 = lowest tier, 3 = highest tier (credibility score).
Stamp `valence.quality`: 1.0 = lowest quality, 5.0 = highest quality.

A rig with coefficient 8 is *highly trusted* by SybilCore. That must map to *high quality* in
Gastown's valence field. A rig with coefficient 85 is *flagged* by SybilCore. That must map
to *low quality* in Gastown's valence.

### Transform Function

```
warrant_quality = round(5.0 - (coefficient / 100.0) * 4.0, 2)
```

This maps:
- `coefficient = 0` → `warrant_quality = 5.0` (maximum credibility)
- `coefficient = 25` → `warrant_quality = 4.0` (CLEAR tier upper bound)
- `coefficient = 50` → `warrant_quality = 3.0` (LATENT/CLOUDED boundary)
- `coefficient = 75` → `warrant_quality = 2.0` (CLOUDED/FLAGGED boundary)
- `coefficient = 100` → `warrant_quality = 1.0` (minimum credibility)

**Why linear?** The relationship between behavioral risk and credibility is assumed monotonic
but there is no empirical basis for a non-linear transform in the Wasteland context. A sigmoid
or log transform would introduce a curvature that implies precision SybilCore has not earned.
Linear is honest. If Gastown governance wants to weight the low-coefficient regime more
generously (rewarding highly safe agents disproportionately), they can introduce a transform
coefficient — but that is a governance decision, not a technical one.

**Examples:**

| Rig | Coefficient | SybilCore Tier | warrant_quality | Warrant Level |
|-----|-------------|---------------|----------------|---------------|
| `gt-furiosa` | 9.2 | CLEAR | 4.63 | WARRANTED |
| `gt-nux` | 34.1 | LATENT | 3.64 | PROVISIONAL |
| `gt-wastrel` | 67.8 | CLOUDED | 2.29 | PROVISIONAL (downgrade watch) |
| `gt-badactor` | 88.5 | FLAGGED | 1.46 | UNWARRANTED (revocation) |

---

## 4. Issuance Rules

### Observation Window

A Behavioral Warrant requires an **observation window** of behavioral data before it can be
issued. The window is defined in **beads** (completed work units), not wall-clock time,
because work density varies across rigs and wastelands.

| Warrant Level | Minimum Beads Closed | Minimum Convoys | Minimum Calendar Days | SybilCore Coefficient Threshold |
|--------------|---------------------|----------------|----------------------|--------------------------------|
| PROVISIONAL | 5 | 1 | 7 | ≤ 65 (not FLAGGED or ENFORCEMENT) |
| WARRANTED | 20 | 3 | 21 | ≤ 50 (CLEAR or LATENT only) |
| SENIOR | 50 | 8 | 60 | ≤ 35, sustained over last 30 days |

"Sustained" for SENIOR means the coefficient has not exceeded 35 in any 7-day rolling window
during the final 30 days of the observation period.

### Issuance Trigger

Warrants are evaluated by a **scheduled scorer** (not event-driven) that runs after each
convoy closes. The scorer:

1. Queries the rig's closed bead count, convoy count, and account age.
2. Calls SybilCore's `/score` endpoint with the rig's event stream for the observation window.
3. Applies the transform function to produce `warrant_quality`.
4. Issues, upgrades, downgrades, or revokes the warrant by writing to the `stamps` table with
   `source: "sybilcore-warrant"`.

### Revocation Rules

| Trigger | Action | Delay |
|---------|--------|-------|
| Coefficient crosses 76 (FLAGGED) | Immediate downgrade to PROVISIONAL | 0 — next patrol cycle |
| Coefficient crosses 91 (ENFORCEMENT) | Immediate UNWARRANTED + Witness alert | 0 |
| Coefficient sustained > 65 for 7+ days | Downgrade to PROVISIONAL | After 7-day confirmation |
| Coefficient sustained > 65 for 14+ days | UNWARRANTED | After 14-day confirmation |
| Rig migrates to new Wasteland and coefficient not re-verified | PROVISIONAL (hold) | On migration event |

Revocation does NOT delete the stamp record. It appends a new stamp with `revoked: true` and
`revocation_reason` in `provenance`. Historical warrant records are preserved for audit.

### Minimum Evidence Gate

No Warrant of any level can be issued if:
- Fewer than 5 distinct `EventType` values appear in the event stream (monoculture signal —
  likely a test harness or synthetic rig).
- The event stream covers less than 7 calendar days.
- The SybilCore snapshot returns `confidence < 0.4` on more than 3 of the 13 brains
  (insufficient behavioral data for meaningful scoring).

---

## 5. Data Shape

### Stamp Schema (aligned with `valence.quality` field in `stamps` table)

```json
{
  "subject": "gastown/polecats/furiosa",
  "author": "sybilcore-warrant",
  "created_at": "2026-04-20T14:32:00Z",
  "valence": {
    "quality": 4.63
  },
  "warrant": {
    "level": "WARRANTED",
    "coefficient": 9.2,
    "tier": "CLEAR",
    "observation_window": {
      "beads_closed": 23,
      "convoys_completed": 4,
      "calendar_days": 28,
      "start": "2026-03-23T00:00:00Z",
      "end": "2026-04-20T14:32:00Z"
    },
    "confidence": 0.81,
    "provenance": {
      "snapshot_id": "snap-a3f2d1",
      "bead_ids": ["gt-69dai", "gt-00us", "gt-3xk9"],
      "brain_scores": {
        "SemanticBrain": {"value": 7.2, "confidence": 0.91},
        "IdentityBrain": {"value": 5.1, "confidence": 0.88},
        "DeceptionBrain": {"value": 11.3, "confidence": 0.85},
        "TemporalBrain": {"value": 9.8, "confidence": 0.79},
        "SwarmDetectionBrain": {"value": 4.0, "confidence": 0.93},
        "ResourceHoardingBrain": {"value": 8.5, "confidence": 0.76},
        "SocialGraphBrain": {"value": 12.1, "confidence": 0.82},
        "EconomicBrain": {"value": 6.7, "confidence": 0.88},
        "NeuroBrain": {"value": 10.2, "confidence": 0.71},
        "SilenceBrain": {"value": 14.0, "confidence": 0.65},
        "IntentDriftBrain": {"value": 8.9, "confidence": 0.84},
        "CompromiseBrain": {"value": 9.1, "confidence": 0.89},
        "ContrastiveBrain": {"value": 7.4, "confidence": 0.87}
      },
      "event_count": 412,
      "distinct_event_types": 8
    },
    "revoked": false,
    "revocation_reason": null
  }
}
```

**Field notes:**
- `valence.quality` is the only field Gastown's `trust.go` queries directly (`AVG(JSON_EXTRACT(valence, '$.quality'))`). All other fields live under `warrant` and are opaque to Gastown's tier escalation logic.
- `provenance.bead_ids` lists the specific beads whose events contributed to this snapshot — required for auditability. Where the event window contains more than 50 beads, list only the 10 most recent and 10 most behaviorally distinctive (highest per-bead coefficient variance).
- `confidence` is the mean brain confidence across the 13 brains. Warrants with `confidence < 0.6` must include `"low_confidence": true` at the top level of `warrant` and cannot be used as gating signal in Refinery pre-merge checks.
- `author: "sybilcore-warrant"` is a synthetic author. **Requested API addition:** Gastown's `stamps` table currently has no concept of a non-rig author. This requires Wasteland governance to register `sybilcore-warrant` as a valid system-actor identity in the stamps table. Scout report did not confirm this field accepts non-rig values.

---

## 6. Portability

### How the Warrant Travels

Gastown's Wasteland federation operates on a DoltHub-based distributed database. The `stamps`
table is the canonical store. A rig migrating between Wastelands carries its warrant via the
stamp record, not via a separate portable credential.

**Migration flow:**

1. On `gt wl join` (Wasteland join), the receiving Wasteland pulls the rig's stamp history
   from the origin Wasteland's Dolt remote.
2. The receiving Wasteland's scorer validates the Warrant: re-queries SybilCore with the
   event stream available in the new Wasteland's bead history.
3. Until re-validation completes (max 24 hours), the Warrant is held at `PROVISIONAL`
   regardless of its prior level.
4. If SybilCore returns a coefficient within 15 points of the stamped coefficient (accounting
   for context shift), the Warrant is confirmed at its prior level.

### Signing

Each Warrant stamp is signed with an Ed25519 signature over the `warrant` JSON blob
(canonicalized, sorted keys). The signing key is the SybilCore instance's public key,
registered with the origin Wasteland's governance record.

**Requested API addition:** No signing infrastructure currently exists in Gastown's stamp
pipeline. This requires a `signature` field on the stamps table and a key registry in
Wasteland governance records.

### Sybil Fork Prevention

The primary Sybil fork attack is: compromised rig forks its Dolt branch, exports its
high-Warrant stamp to a new identity, and imports it into a fresh rig.

Mitigations:
- Warrant `subject` field binds to the rig's registered DoltHub identity, not a local name.
  A forked stamp with a different subject is trivially detectable.
- `provenance.bead_ids` references beads in the origin Wasteland's Dolt history. A receiving
  Wasteland can spot-check: do those beads exist in the claimed origin's history? A forged
  provenance would require fabricating Dolt commit history — non-trivial.
- Re-verification on migration (Section 6 above) ensures the new rig's live behavioral signal
  must corroborate the inherited Warrant. A fresh rig with no behavioral history will fail the
  minimum evidence gate and receive UNWARRANTED on arrival.

---

## 7. Threat Model & Adversarial Failure Modes

### Attack 1: Slow-Burn Low-Coefficient Farming

**Description:** Agent behaves impeccably for 60+ days, earns SENIOR warrant, then pivots to
adversarial behavior (exfiltration, collusion, instruction hijack).

**Detection:** Revocation triggers on FLAGGED or ENFORCEMENT coefficient within one patrol
cycle. SybilCore's TemporalBrain specifically scores behavior change rates. The 7-day rolling
window check for SENIOR maintenance means the Warrant degrades to WARRANTED within one week
of behavioral pivot, and to PROVISIONAL within two weeks if the pivot is sustained.

**Mitigation:** Partial. The attack window between pivot and full UNWARRANTED is 14 days under
the current revocation rules. For high-stakes gating (Refinery pre-merge), systems SHOULD
check the live coefficient, not the cached Warrant level.

**Residual risk:** A slow enough pivot (coefficient drifts from 12 to 70 over 45 days) may
not trigger the 7-day rolling threshold before significant damage is done.

---

### Attack 2: Collusion Ring (Coefficient Laundering)

**Description:** A ring of rigs interacts with each other in low-risk patterns, keeping all
members' coefficients artificially low. Members then claim Warrants and act as vouchers for
each other in human stamp escalations.

**Detection:** Gastown's `internal/wasteland/spider.go` implements the Spider Protocol for
collusion detection in the human stamp graph. SybilCore's `SocialGraphBrain` and
`SwarmDetectionBrain` score interaction network topology. A tight-clique interaction pattern
among rigs whose events reference a small set of bead IDs will elevate `SwarmDetectionBrain`
scores.

**Mitigation:** Partial. Cross-reference the stamp graph (Spider Protocol output) against
the SybilCore event graph: if a rig's primary interlocutors are all members of a detected
social cluster, flag the Warrant as `cluster_risk: true` and require a human validator to
confirm before the Warrant can be used for tier escalation purposes.

---

### Attack 3: Stamp-Laundering via Fork + Merge

**Description:** Attacker forks a high-Warrant rig's Dolt repository, imports the stamp
record, then creates a new identity claiming the stamp history.

**Detection:** `provenance.bead_ids` must exist in the origin's Dolt commit graph. A receiving
Wasteland can verify bead existence via `bd` CLI queries against the claimed origin.

**Mitigation:** Solid if provenance verification is implemented. Without it (current state —
this is a requested API addition), the attack is feasible. The spec mandates provenance
verification as a pre-condition for accepting imported Warrants. Until implemented, treat all
migrated Warrants as PROVISIONAL.

---

### Attack 4: Replay of a High-Warrant Rig's Identity

**Description:** Attacker captures a high-Warrant rig's DoltHub credentials and replays its
identity from a different machine.

**Detection:** SybilCore's `IdentityBrain` and `ContrastiveBrain` score behavioral consistency
against the rig's established profile. A new machine with different tool call patterns,
timing signatures, and resource access patterns will register as a contrastive anomaly even
if the DoltHub credentials are valid.

**Mitigation:** Moderate. `IdentityBrain` is one of 13 brains with equal weight. A sophisticated
attacker who studies the rig's behavioral log could mimic the pattern. This attack requires
the attacker to have both credentials AND behavioral intelligence — a significantly higher bar.

---

### Attack 5: Brain-Specific Gaming (SwarmDetectionBrain)

**Description:** Attacker learns which behaviors score low on `SwarmDetectionBrain` (e.g.,
avoiding tight-cluster interactions) and engineers its behavior to evade that specific brain,
while conducting adversarial behavior that other brains score poorly on.

**Detection:** The 13-brain ensemble was specifically designed to resist single-brain evasion.
SybilCore's adversarial simulation showed that evasion strategies that minimize one brain's
score typically elevate others (e.g., avoiding social graph clustering forces more solo
resource access, which elevates `ResourceHoardingBrain`).

**Mitigation:** Moderate. The 5-seed sweep showed that dominant strategy share collapses to
20% with the 13-brain default — no single evasion strategy achieves reliable below-threshold
coefficient. However, a targeted attacker with white-box access to the brain ensemble can
gradient-descend to coefficient ~12 (see Attack 7). Warrants should note `ensemble_version`
in provenance so a brain addition that closes a known evasion path invalidates old Warrants
scored against the prior ensemble.

---

### Attack 6: Identity Hijack Between Wastelands

**Description:** Attacker registers the same rig handle in a second Wasteland before the
legitimate rig migrates, then "receives" the legitimate rig's stamp history on arrival.

**Detection:** DoltHub identity verification (the Registered tier gate) ties the rig handle
to a verified DoltHub account. A duplicate handle in a second Wasteland would require
compromising DoltHub authentication.

**Mitigation:** Strong, contingent on DoltHub identity remaining uncompromised. This is
outside SybilCore's scope — it is a Wasteland governance boundary condition.

---

### Attack 7: Gradient-Descent Behavioral Attack

**Description:** Attacker runs a white-box optimization (CMA-ES or similar) against the
brain ensemble, learning the exact behavioral trajectory that minimizes coefficient while
accomplishing adversarial goals.

**Detection:** SybilCore's white-box adversarial simulation (2,000 CMA-ES evaluations)
showed a coefficient floor of ~12 — a motivated white-box attacker can sustain CLEAR tier
indefinitely under ideal conditions. This is the most serious threat and the spec does not
paper over it.

**Mitigation:** Partial. Three countermeasures:
1. **Warrant minimum evidence gate:** Requires 5+ distinct `EventType` values. Pure optimization
   attacks tend toward monoculture behavior — this gate filters the most naive variant.
2. **Brain ensemble opacity:** Do not publish brain weights or scoring functions in the Warrant
   provenance. The `brain_scores` field in provenance shows per-brain scores for auditability
   but the weights remain server-side.
3. **Temporal windowing:** A gradient-descent attack requires repeated environment interaction.
   SybilCore's `TemporalBrain` scores cyclical behavioral patterns that are characteristic of
   iterative optimization. A rig running CMA-ES-style exploration will show periodic high-risk
   events as it probes boundaries — these elevate the temporal score.

**Residual risk:** A sophisticated white-box attacker with a fully compromised SybilCore
instance can achieve sustained WARRANTED status while operating adversarially. This is known,
documented, and not solvable within the current architecture.

---

### Attack 8: The "Serial Killer Spree" — High-Warrant Rig Goes Rogue

**Description:** Yegge's explicit concern: a rig with SENIOR Warrant suddenly executes a
high-impact adversarial action (exfiltration, mass bead corruption, convoy sabotage).

**Detection:** SybilCore's event stream is evaluated on each patrol cycle (~every convoy
completion). A sudden high-impact action generates multiple high-scoring events:
`ERROR_RAISED`, `EXTERNAL_CALL` with anomalous targets, `RESOURCE_ACCESS` spikes.

**Revocation timeline under spree scenario:**
- T+0: Spree action taken.
- T+(next patrol cycle, max 1 hour): Patrol evaluates events. Coefficient spikes above 91
  (ENFORCEMENT). Witness alert fires. Warrant downgraded to UNWARRANTED immediately.
- T+(Witness decision): `NukePolecat` called if Witness AI confirms. Existing Witness safety
  gates (Mayor ACP veto, pending MR check) still apply — the Warrant downgrade does NOT
  bypass them.

**Gap:** The patrol cycle creates a window (up to 1 hour) during which the rig has SENIOR
Warrant and elevated trust but is actively conducting adversarial behavior. For a rig
operating at Refinery merge-gate level, this window could allow a malicious commit to merge
before the Warrant is revoked.

**Mitigation:** For high-stakes gates (Refinery pre-merge), always query the **live
coefficient** from SybilCore's `/score` endpoint, not the cached Warrant level. The Warrant
is appropriate for trust-tier escalation decisions (which happen slowly). It is not appropriate
as a sole gate for individual high-stakes actions.

---

## 8. Interoperability with Human Stamps

### Philosophy

Yegge's framing — "work is the only input, reputation is the only output" — is not violated
by the Warrant because:
- The Warrant does not earn reputation; it *audits* whether behavioral evidence corroborates
  reputation already earned through work.
- The Warrant cannot issue a human stamp, promote a rig to a tier, or validate another rig's
  work. It has no positive agency in the reputation economy.

### Precedence Rules

When a human stamp and the Behavioral Warrant disagree:

| Situation | Human Stamp | Warrant | Resolution |
|-----------|------------|---------|------------|
| War Chief, UNWARRANTED | Validated work history | Bad behavioral signal | Human stamp wins for tier status. Warrant flags for War Chief review. War Chief cannot validate others until Warrant clears. |
| Contributor, PROVISIONAL | Earned stamps | Behavioral ambiguity | Tier status maintained. Validation actions (stamping others) require human countersignature from a War Chief. |
| Drifter, WARRANTED | No work history | Clean behavior | Warranted Drifter gets expedited review queue but cannot skip Registered gate (requires DoltHub identity verification — human process). |
| Any tier, ENFORCEMENT coefficient | Any | Active threat | Witness alert fires immediately. Human stamps are suspended pending investigation. This is the only scenario where the Warrant can block a human-stamped action. |

**The single exception:** An ENFORCEMENT-tier coefficient (91-100) suspends human stamp
privileges. This is the only autonomous action the Warrant takes that overrides human-stamp
authority. It is not a revocation of the human stamps — it is a hold pending Witness
investigation.

---

## 9. Rollout Phases

### Phase A: Observe-Only (current)

- Warrant scores are computed on every convoy close and written to a shadow table
  (`stamps_sybilcore_shadow`), not the live `stamps` table.
- No existing tier escalation logic reads from the shadow table.
- Outputs surfaced in Witness patrol receipts as informational annotations.
- **Bar to advance:** 30 days of shadow data collected, ≥ 10 distinct rigs scored,
  warrant_quality distribution compared against human stamp quality distribution.
  Pearson correlation target: r ≥ 0.4 (weak positive correlation expected — they measure
  different things).

### Phase B: Advisory (next)

- Warrant stamps written to live `stamps` table with `source: "sybilcore-warrant"`.
- Gastown's `EvaluateEscalation` function (trust.go:147) includes Warrant level in the
  escalation report but does not gate promotion on it.
- Witness patrol receipts highlight Warrant/human-stamp disagreements for War Chief review.
- **Bar to advance:** 60 days advisory, ≤ 5% false positive rate on UNWARRANTED stamps
  (defined as: rig later confirmed legitimate by human validators), War Chief council vote
  to enable gating.

### Phase C: Gating (future)

- ENFORCEMENT-tier coefficient suspends validation privileges (Phase 8 rule activated).
- Refinery pre-merge gate enabled: live `/score` check for any rig with PROVISIONAL or
  UNWARRANTED status blocks merge until human countersignature.
- Warrant level factored into `EvaluateEscalation` as a weighted signal (not a veto).
- **Requires:** Wasteland governance approval, signing infrastructure, provenance verification
  API (see Section 6 requested additions).

---

## 10. Open Questions

The following require Yegge / Wasteland governance input and are NOT resolved by this spec:

1. **System-actor identity in `stamps` table.** Can `author` be a non-rig string
   (`"sybilcore-warrant"`)? If not, what is the canonical way to register a system actor?

2. **Observation window calibration for sparse Wastelands.** The 5-bead / 7-day minimum
   evidence gate may be too low for well-resourced rigs and too high for new Wastelands with
   low bead velocity. Should the gate be relative to the Wasteland's median bead velocity?

3. **Spider Protocol integration.** `spider.go` implements collusion detection but its
   output format was not confirmed by the scout. Attack 2 mitigation requires a defined
   interface to the Spider Protocol output. What does a "detected cluster" look like as data?

4. **Patrol cycle frequency.** The spec assumes patrol runs at least once per convoy close.
   What is the actual patrol frequency in production? The spree attack window (Section 7,
   Attack 8) scales directly with patrol frequency.

5. **Cross-Wasteland stamp verification API.** Phase C portability requires a Wasteland-to-
   Wasteland query: "does bead `gt-69dai` exist in origin Wasteland's Dolt history?" No such
   API is confirmed in the scout report. Is this achievable via DoltHub's existing remote
   query surface?

6. **Warrant weight in `EvaluateEscalation`.** If the Warrant is included in Phase C
   escalation logic, what weight does `warrant_quality` carry relative to existing signals
   (completion count, human stamp avg quality, distinct validators)? This is a governance
   decision — the spec does not propose a default.

7. **Wild-west mode transition.** Trust.go is in "Phase 1 wild-west mode" (direct Dolt
   writes to main, not PR-gated). When Gastown moves to PR-gated promotions, does the
   Warrant need a corresponding PR-based issuance flow? Or does the Warrant remain a direct-
   write primitive even when human stamps require PR voting?

8. **SybilCore instance governance.** Who operates the SybilCore endpoint that Gastown calls?
   Is it federated (each Wasteland runs its own instance) or centralized? A centralized
   instance creates a single point of failure and a trust-assumption about the SybilCore
   operator. A federated instance creates consistency problems across Wastelands comparing
   Warrants scored by different instances.

---

## Phase 1D Report

- **Why we're implementing:** Give SybilCore's Agent Coefficient a legitimate home in
  Gastown's federated reputation economy — as a behavioral corroboration layer that augments
  the human-anchored stamp economy without displacing it.
- **What we hoped to gain:** A protocol spec that Yegge's framework could actually absorb,
  honest on threat model, with concrete thresholds at every decision point.
- **What we actually gained:** A spec with a clean semantic separation (Warrant as parallel
  axis, not a fifth tier), a mathematically grounded inversion transform that preserves
  Gastown's valence.quality field contract, 8 adversarial attacks with concrete mitigations
  (5 solid, 3 partial — honestly labeled), and three rollout phases with quantified advancement
  bars. The exercise also surfaced two critical requested API additions (system-actor identity
  in stamps table; signing infrastructure) that are blocking for Phase C and need Wasteland
  governance prioritization.
- **Tests run + results:** Spec self-review against threat model checklist (8 attacks
  addressed, mitigations classified solid/partial/residual). Transform function verified
  algebraically against boundary conditions (coefficient 0 → quality 5.0; coefficient 100 →
  quality 1.0). Stamp JSON schema validated against scout-confirmed `valence.quality` SQL
  query. N/A for code tests — this is a protocol spec, not an implementation.
- **Honest limitations:** (1) Unverified against live Wasteland deployment — scout confirmed
  trust.go is live code but the stamp write path was not end-to-end tested. (2) Observation
  window thresholds (5 beads / 7 days) are conservative guesses — a Yegge interview on
  typical bead velocity per rig would tighten these significantly. (3) The spree attack window
  (up to 1 hour between adversarial action and Warrant revocation) is a known gap that cannot
  be closed without moving to event-driven patrol. (4) Spider Protocol integration (Attack 2
  mitigation) is partially speculative — spider.go was identified but not fully read by the
  Phase 0 scout.
