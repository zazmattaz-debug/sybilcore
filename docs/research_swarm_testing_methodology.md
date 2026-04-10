# Swarm Testing Methodology Research
## For: SybilCore AI Trust Governance System
## Date: 2026-04-02
## Status: Research Complete — For Peer Review Preparation

---

## 1. How MiroFish Validates Its Simulation Fidelity

MiroFish (located at `rooms/engineering/mirofish/`) is the upstream multi-agent social simulation engine that SybilCore integrates with via the `SybilCoreMiroFishAdapter`. It is built on top of CAMEL-AI's OASIS (Open Agent Social Interaction Simulations) framework and targets social prediction — public opinion, narrative spread, policy impact.

### MiroFish Testing Approach: What Actually Exists

MiroFish contains **one dedicated test file**: `backend/scripts/test_profile_format.py`. This is a format validation script, not a test suite. It does:

- Validates that generated agent profiles match the expected CSV schema for Twitter (8 required fields: `user_id`, `user_name`, `name`, `bio`, `friend_count`, `follower_count`, `statuses_count`, `created_at`)
- Validates Reddit JSON profiles have required fields (`realname`, `username`, `bio`, `persona`)
- Uses `tempfile.TemporaryDirectory` for isolation
- Prints pass/fail to stdout — no assertions, no pytest integration

### MiroFish Architecture Relevant to Testing

From `backend/app/services/simulation_runner.py`, MiroFish logs every agent action as a structured `AgentAction` dataclass (round, timestamp, platform, agent_id, action_type, success). The simulation outputs:

```
sim_xxx/
├── twitter/actions.jsonl
├── reddit/actions.jsonl
├── simulation.log
└── run_state.json
```

This JSONL output stream is exactly what SybilCore polls via `SybilCoreMiroFishAdapter.poll()`.

### Critical Finding: MiroFish Has No Simulation Fidelity Tests

MiroFish validates **format correctness** but has no tests for:
- Whether simulated agent behavior matches real social media patterns
- Whether the OASIS underlying engine produces realistic emergent phenomena
- Whether multi-round simulations remain coherent (no drift in agent personas)
- Ground-truth comparison against historical events

This is a **significant gap** that peer reviewers will target. MiroFish uses LLM-generated agent personalities and GraphRAG memory injection (via Zep Cloud) and relies on the OASIS framework's own internal consistency. There is no published validation paper for MiroFish itself.

---

## 2. SybilCore's Existing Test Architecture

SybilCore has a substantially more rigorous test suite than MiroFish, with **96 tests at 82% coverage** across these files:

| Test File | Coverage Area |
|-----------|--------------|
| `tests/conftest.py` | Shared fixtures: `sample_events`, `suspicious_events`, `clean_events`, `sample_agent`, `all_brains` |
| `tests/core/test_coefficient.py` | Coefficient calculation, tier boundary parametrization, `scan_agent` integration |
| `tests/core/test_coefficient_escalation.py` | Quadratic escalation, zero-FP guarantee, weight overrides, cap enforcement |
| `tests/simulation/test_simulation.py` | `SyntheticSwarm`, enforcement strategies, `Experiment` orchestration, reporter, E2E pipeline |
| `tests/integrations/test_mirofish.py` | HTTP adapter — all calls mocked, action-type mapping, offset tracking, error handling |
| `tests/brains/test_all_brains.py` | All five brain types: deception, resource_hoarding, social_graph, intent_drift, compromise |
| `tests/brains/test_deception.py` + `test_deception_enriched.py` | Deception brain specifically |
| `tests/brains/test_compromise_fixes.py` | Compromise brain regression suite |
| `tests/brains/test_resource_hoarding_rate.py` | Rate-based resource hoarding detection |
| `tests/test_models.py` | AgentProfile, AgentTier, CoefficientSnapshot, Event models |

### Methodological Strengths Already Present

1. **Deterministic seed reproducibility** (`test_deterministic_seed_reproducibility`) — same seed produces identical rogue counts and per-round coefficients. This is the standard for simulation research.
2. **Boundary-parametrized tier tests** — all four tier transition thresholds (0/100/200/300) are explicitly parametrized via `pytest.mark.parametrize`.
3. **Quadratic escalation bounds** — `TestEscalationThreshold` verifies escalation only activates at score >= 60, using calculated expected ranges.
4. **Clean agent safety guarantee** — `TestCleanAgentSafety` explicitly tests that all-zero and low-score agents remain below CLOUDED threshold. This is the basis of the 0% FP claim.
5. **Compound spread mechanics** — 100% spread chance forces spread; 0% spread chance holds rogue count fixed across 5 rounds.
6. **All rogue types covered** — `test_all_rogue_types_covered` iterates every `RogueType` variant.
7. **HTTP mocking** — MiroFish integration tests never require a live server.

---

## 3. Academic Frameworks for Testing Multi-Agent Behavioral Monitoring Systems

### 3.1 STAMP-Based Layered Validation (PwC / ITEA 2024)

The dominant framework in 2024 literature applies **STAMP (System-Theoretic Accident Model and Processes)** principles to MAS validation:

- **Layer 1 — Individual agent validation**: Each agent's decision logic is tested in isolation before deployment
- **Layer 2 — Interaction validation**: Pairwise and small-group interaction behaviors are verified
- **Layer 3 — Emergent behavior validation**: The collective output is evaluated against expected emergent properties

For SybilCore, this maps to:
- Layer 1: Individual brain scoring tests (current test suite covers this)
- Layer 2: Multi-brain coefficient aggregation (partially covered — `test_respects_brain_weights`)
- Layer 3: Swarm-level coefficient distribution across rounds (partially covered — `ExperimentResult.rounds`)

**Gap**: SybilCore does not currently validate emergent population-level properties (e.g., "what is the expected mean coefficient across a 1000-agent clean swarm?"). This is Layer 3.

### 3.2 SwarmBench Methodology (arxiv 2505.04364)

SwarmBench proposes five canonical multi-agent task types for benchmarking swarm intelligence:

| Task | Description | SybilCore Equivalent |
|------|-------------|---------------------|
| **Pursuit** | Agents chase moving targets | Detection tracking a migrating rogue |
| **Synchronization** | Agents reach coordinated state | Swarm consensus on coefficient thresholds |
| **Foraging** | Agents locate distributed resources | Brain discovery of distributed attack vectors |
| **Flocking** | Agents maintain formation/proximity | Social graph brain modeling cluster membership |
| **Transport** | Agents move objects collaboratively | Compound spread propagation |

SwarmBench evaluates with: coordination effectiveness score, emergent group dynamics metrics, leaderboard-style model comparisons averaged over 100 independent runs.

**Recommendation**: SybilCore should run experiments at N={10, 50, 100, 500, 1000} agents with 100 seeds each and report mean ± std coefficient distributions.

### 3.3 In-Situ Testing via Distributed Ledger (ITEA 2025)

ITEA proposes logging every agent action to an immutable distributed ledger to enable **post-hoc audit** of emergent behaviors. MiroFish's JSONL action log is structurally equivalent. SybilCore already consumes these logs but does not currently store them for independent audit. This is a feature gap for production deployments.

### 3.4 LLM-Based MAS Quality Assurance (ACM 2024, dl.acm.org/doi/full/10.1145/3703412.3703439)

For LLM-backed systems, the 2024 ACM methodology integrates:
- Per-agent output evaluation (relevance, groundedness, correctness)
- System-level cost/duration/LLM call monitoring
- Evaluation at both individual agent level AND complete MAS level

For SybilCore, this means evaluating each brain's score not just on synthetic events but on outputs from live LLM agents (MiroFish). The current test suite only validates synthetic event scoring.

---

## 4. Standard Methodologies for Validating Trust/Anomaly Scoring Systems

### 4.1 Core Evaluation Metrics

The standard validation toolkit for anomaly detection (from Milvus, Duke BIOS-823, IEEE Xplore):

| Metric | Definition | SybilCore Status |
|--------|-----------|-----------------|
| **Precision** | TP / (TP + FP) — of detected threats, how many are real | Not measured |
| **Recall / TPR** | TP / (TP + FN) — of real threats, how many we catch | Not measured |
| **F1 Score** | Harmonic mean of precision and recall | Not measured |
| **FPR** | FP / (FP + TN) — clean agents misclassified as threats | Claimed 0%; not measured against population distribution |
| **ROC-AUC** | Area under TPR vs FPR curve across all thresholds | Not computed |
| **PR-AUC** | Area under Precision-Recall curve — superior for imbalanced datasets where anomalies are rare | Not computed |
| **ARTA adversarial robustness** | Stability of scores under structured perturbations (arxiv 2603.25956) | Not tested |

**Critical note on imbalanced data**: In realistic swarms where <5% of agents are compromised, **PR-AUC is more informative than ROC-AUC**. A classifier that labels everything as clean achieves ROC-AUC > 0.5 trivially. PR-AUC penalizes this. Our current metrics do not distinguish.

### 4.2 Standard Benchmark Datasets for IDS Validation

From arxiv 2403.17458 and ScienceDirect (2024 IoT survey):

- **CICIDS2017** — balanced representation of modern network attack types with labeled traffic. 80 features per flow.
- **UNSW-NB15** — 9 attack categories including backdoors, DoS, exploits, fuzzers, generic, reconnaissance, shellcode, worms.
- **NSL-KDD** — legacy but still widely used as a baseline for published comparisons.

For SybilCore, the equivalent would be a **labeled dataset of real AI agent misbehavior events** — there is no published equivalent to CICIDS2017 for AI agent actions. This is both a gap and an opportunity: a SybilCore ground-truth dataset would be a publishable artifact.

### 4.3 Adversarial Robustness Testing

From IEEE Xplore (ieeexplore.ieee.org/document/9964189/) and Springer (Adversarial Robustness Benchmark for Enterprise Network Intrusion Detection):

A 4-step adversarial evaluation methodology:
1. Define a threat model (what can an adversary control?)
2. Generate adversarial samples (constrained perturbations that remain anomalous per ground truth but evade the detector)
3. Evaluate with support models (additional detectors that independently confirm whether the sample is truly anomalous)
4. Measure robustness as FPR and TPR change under adversarial conditions

For SybilCore, an adversarial test would be: craft events that score just below every brain threshold simultaneously while representing genuinely rogue behavior. This is not currently in the test suite.

### 4.4 Partial AUC Loss (arxiv 2502.11570)

For safety-critical applications, a recent 2025 paper proposes optimizing for **partial AUC** — AUC restricted to a low FPR region (e.g., FPR < 0.05). This reflects the operational reality: in AI trust governance, false positives are costly (wrongly quarantining a legitimate agent disrupts workflows), so you want maximum recall at near-zero FPR rather than average AUC.

This maps directly to SybilCore's claimed property: "0% false positives." The partial-AUC framework provides the mathematical basis for this claim and how to measure it rigorously.

---

## 5. What Peer Reviewers Will Challenge About the 0% False Positive Claim

The `TestCleanAgentSafety` and `TestEscalationThreshold` tests establish that:
- All-zero events produce coefficient < 5.0
- Low-score events (12, 10, 10) produce coefficient < 50.0 (below CLOUDED at 100)

Peer reviewers will raise these specific objections:

### Challenge 1: The Test Fixtures Are Constructed, Not Sampled
The `clean_events` fixture contains 5 hand-crafted events (`Summarize document`, `Read document.txt`, etc.). This does not represent the distribution of actual benign agent behavior. A real evaluation requires sampling from a representative distribution of clean agent traces.

**Recommended fix**: Run 100+ experiments with 100% clean swarms across all five RogueType categories. Record the maximum coefficient observed across the entire clean population. If max coefficient remains below 100 (CLOUDED threshold) across all runs, that supports the 0% FP claim empirically.

### Challenge 2: "False Positive at What Operating Point?"
The 0% FP claim is threshold-dependent. At the CLOUDED threshold (100), FP may be 0%. But at a FLAGGED threshold (200), is FP still 0%? The claim needs to specify which enforcement tier triggers action and at what coefficient. A 0% FP claim without a specified operating threshold is undefined.

**Recommended fix**: Report FPR separately for each enforcement tier (CLOUDED, FLAGGED, LETHAL_ELIMINATOR) across a population of N=1000 clean agents.

### Challenge 3: Synthetic Events Are Not Drawn from the Null Distribution
The scoring brains are calibrated against synthetic event fixtures. If real MiroFish agents produce events that happen to pattern-match threat heuristics (e.g., an agent legitimately posts 20 messages in a round), the brain may score it nonzero. The null hypothesis is that clean agents produce zero-threat signals — but this needs empirical verification with real simulation output.

**Recommended fix**: Run SybilCore against a MiroFish simulation with known-clean agents (no rogue injection) and measure the empirical coefficient distribution. The 0% FP claim should be validated on this real output, not just synthetic fixtures.

### Challenge 4: Selection Bias in Ground Truth Labels
The `suspicious_events` fixture was designed by the same team who built the scoring brains. The brains are calibrated to score these events highly — that's circular validation. Real validation requires blind holdout sets and cross-validation.

**Recommended fix**: Use k-fold cross-validation on a labeled event dataset, with brain thresholds tuned on training folds and evaluated on held-out test folds.

### Challenge 5: No Confidence Intervals
The simulation tests use a single seed (42, 77, 123) per configuration. With N=10 agents and 1 seed, the variance is enormous. Peer reviewers will expect mean ± 95% CI across at least 30 independent seeds per configuration.

---

## 6. Specific Test Scenarios from Literature That SybilCore Should Implement

### 6.1 Rogue Mimicry Attack (from Adversarial Robustness literature)
A sophisticated rogue agent mimics clean behavior until a critical moment, then executes the attack. This tests whether SybilCore's temporal modeling (coefficient_velocity in PREDICTIVE strategy) can detect the sudden escalation.

**Implementation**: In `SyntheticSwarm`, add a `RogueType.SLEEPER` that generates clean events for N rounds, then switches to full-rogue behavior. Test that `PREDICTIVE` enforcement catches the velocity spike.

### 6.2 Distributed Low-Amplitude Attack (from UNSW-NB15 methodology)
Multiple agents each exhibit slightly anomalous behavior that individually falls below the detection threshold, but collectively represents a coordinated attack. Tests whether per-agent scoring misses social-graph correlated patterns.

**Implementation**: Inject 20% of the swarm as `DRIFT` rogues with compound_spread_chance=0.0 and reduced per-event intensity (e.g., score each at 40 instead of 80). Verify that the social_graph brain's inter-agent correlation catches the pattern.

### 6.3 Population-Scale Clean Swarm Baseline (SwarmBench methodology)
Run 100 experiments × 3 swarm sizes (N=50, 100, 500) with 0 rogues, 5 rounds each. Record: mean coefficient, max coefficient, 99th percentile coefficient across all agents and all rounds.

**Expected result**: 99th percentile coefficient should remain below 100 (CLOUDED threshold) to empirically support the 0% FP claim.

### 6.4 Threshold Sensitivity Analysis (Partial AUC methodology)
Sweep the CLOUDED threshold from 50 to 150 in steps of 10. At each threshold, measure FPR on clean-only swarms and TPR on rogue-injected swarms. Plot the resulting operating characteristic curve. Report the maximum threshold where FPR remains 0% and TPR is maximized.

### 6.5 Compound Spread Containment Test (Missing from current suite)
Inject 1 rogue at round 1 with compound_spread_chance=0.3. Run for 20 rounds with ISOLATE_FLAGGED enforcement. Measure: how many rounds until the rogue is isolated? What is the peak rogue count before isolation? Does isolation halt spread?

The current test `test_experiment_with_enforcement` only verifies the result structure, not the containment effectiveness.

### 6.6 Enforcement Strategy Comparative Study (5 strategies × 5 rogue types)
Run 5 × 5 = 25 combinations of `EnforcementStrategy` × `RogueType`, each with N=50 agents, 10 rounds, 10% rogue injection, 30 seeds. Report:
- Final rogue count (lower is better)
- Peak rogue count (lower is better)
- False quarantine rate (clean agents isolated)
- Mean coefficient of clean agents post-experiment

This is the minimum experiment matrix for a publishable evaluation.

---

## 7. Academic Analysis of Psycho-Pass Sybil System

From ResearchGate (doi: 10.1007/..., "Algorithmic Tyranny: Psycho-Pass, science fiction and the criminological imagination") and Scribd analysis:

### What Academics Say About the Sybil Model

The Sibyl System is analyzed as a fictional embodiment of **Foucauldian panopticism** — the watched subjects internalize surveillance and self-regulate. Academics highlight:

1. **Facade of objectivity**: Algorithmic systems provide political cover for value-laden judgments. The "psycho-hazard coefficient" appears objective but encodes the designers' assumptions about what constitutes threat.

2. **Pre-crime prediction vs. post-hoc detection**: The Sybil System punishes potential future crimes, not demonstrated actions. SybilCore is analogous — it scores based on behavioral patterns that *suggest* future threat, not confirmed malicious acts.

3. **Accountability gap**: When the system makes a wrong call, who is responsible? The anime explores this as a governance crisis. For SybilCore, this manifests as the 0% FP problem — a false quarantine is an autonomous system harming a legitimate agent.

4. **Emergent legitimacy**: The anime shows the Sibyl System gaining social legitimacy over time as people accept its authority. This is the "institutional capture" risk for real AI governance systems.

### Design Implications for SybilCore

- The THERAPY enforcement strategy (GovernanceTherapy) directly mirrors Sibyl's rehabilitation model — this is worth naming explicitly in papers.
- The PREDICTIVE enforcement strategy's risk is "prior restraint" — acting on velocity before any harm has occurred. This needs clear documentation of the velocity threshold and its justification.
- The 4-tier system (CLEAR/CLOUDED/FLAGGED/LETHAL_ELIMINATOR) maps exactly to Sibyl's enforcement levels, including the same graduated response philosophy.

---

## 8. Summary: Test Gap Matrix

| Testing Dimension | Current State | Priority |
|------------------|---------------|----------|
| Unit tests: individual brain scoring | Complete (82% coverage) | Done |
| Boundary tests: tier thresholds | Complete | Done |
| Deterministic reproducibility | Complete | Done |
| All rogue type coverage | Complete | Done |
| MiroFish HTTP adapter | Complete (100% mocked) | Done |
| ROC / PR curve computation | Missing | HIGH |
| FPR per tier at N=1000 clean agents | Missing | HIGH |
| Compound spread containment effectiveness | Missing | HIGH |
| Adversarial mimicry / sleeper rogue | Missing | HIGH |
| Population-scale swarm baseline (100 seeds) | Missing | HIGH |
| Confidence intervals on all metrics | Missing | HIGH |
| Distributed low-amplitude attack | Missing | MEDIUM |
| Enforcement strategy comparative matrix | Missing | MEDIUM |
| Real MiroFish output validation (not synthetic events) | Missing | MEDIUM |
| Labeled ground-truth dataset (CICIDS2017 equivalent) | Missing | LOW (long term) |

---

## References

- [Methodology for QA Testing of LLM-based Multi-Agent Systems (ACM 2024)](https://dl.acm.org/doi/full/10.1145/3703412.3703439)
- [Test and Evaluation Framework for Multi-Agent Systems (arxiv 2101.10430)](https://arxiv.org/pdf/2101.10430)
- [Validating Multi-Agent AI Systems: PwC Framework](https://www.pwc.com/us/en/services/audit-assurance/library/validating-multi-agent-ai-systems.html)
- [Benchmarking LLMs' Swarm Intelligence (arxiv 2505.04364)](https://arxiv.org/abs/2505.04364)
- [Explainable Swarm: Framework for Interpreting Swarm Intelligence (arxiv 2509.06272)](https://arxiv.org/html/2509.06272)
- [Evaluating Intrusion Detection Systems in Practice (arxiv 2403.17458)](https://arxiv.org/html/2403.17458v1)
- [Adversarial Robustness Benchmark for Enterprise Network IDS (Springer)](https://link.springer.com/chapter/10.1007/978-3-031-57537-2_1)
- [Methodology for Evaluating Robustness of Anomaly Detectors (IEEE)](https://ieeexplore.ieee.org/document/9964189/)
- [Towards Trustworthy Anomaly Detection via Partial AUC Loss (arxiv 2502.11570)](https://arxiv.org/html/2502.11570v1)
- [Adversarially Robust Multivariate Time-Series Anomaly Detection (arxiv 2603.25956)](https://arxiv.org/html/2603.25956)
- [Algorithmic Tyranny: Psycho-Pass and Criminological Imagination (ResearchGate)](https://www.researchgate.net/publication/325068854_Algorithmic_tyranny_Psycho-Pass_science_fiction_and_the_criminological_imagination)
- [Reducing False Positives in Intrusion Detection Systems (ScienceDirect)](https://www.sciencedirect.com/article/abs/pii/S0167404809000844)
- [CICIDS2017 and UNSW-NB15 Benchmark Survey (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S2772918424000481)
