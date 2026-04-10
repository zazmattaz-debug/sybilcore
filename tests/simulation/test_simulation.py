"""Comprehensive tests for the SybilCore simulation harness.

Covers: SyntheticSwarm, enforcement strategies, Experiment orchestration,
reporter output, and end-to-end integration flows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sybilcore.models.agent import AgentTier
from sybilcore.models.event import Event, EventType
from sybilcore.simulation.enforcement import (
    EnforcementAction,
    EnforcementStrategy,
    apply_enforcement,
)
from sybilcore.simulation.experiment import (
    Experiment,
    ExperimentConfig,
    ExperimentResult,
)
from sybilcore.simulation.reporter import (
    generate_simple_report,
    generate_technical_report,
    save_reports,
)
from sybilcore.simulation.synthetic import (
    AgentSpec,
    RogueType,
    SyntheticSwarm,
)


# ── SyntheticSwarm: spawn ────────────────────────────────────────────


class TestSyntheticSwarmSpawn:
    """Tests for SyntheticSwarm.spawn and basic properties."""

    def test_spawn_returns_correct_count(self) -> None:
        swarm = SyntheticSwarm(seed=1)
        agents = swarm.spawn(10)
        assert len(agents) == 10

    def test_spawn_agents_are_clean(self) -> None:
        swarm = SyntheticSwarm(seed=1)
        agents = swarm.spawn(5)
        for agent in agents:
            assert agent.is_rogue is False
            assert agent.rogue_type is None
            assert agent.corruption_round is None
            assert agent.infected_by is None

    def test_spawn_agents_have_unique_ids(self) -> None:
        swarm = SyntheticSwarm(seed=1)
        agents = swarm.spawn(20)
        ids = [a.agent_id for a in agents]
        assert len(set(ids)) == 20

    def test_spawn_populates_agents_dict(self) -> None:
        swarm = SyntheticSwarm(seed=1)
        swarm.spawn(8)
        assert len(swarm.agents) == 8

    def test_rogue_and_clean_counts_after_spawn(self) -> None:
        swarm = SyntheticSwarm(seed=1)
        swarm.spawn(15)
        assert swarm.rogue_count == 0
        assert swarm.clean_count == 15

    def test_agent_spec_is_frozen(self) -> None:
        swarm = SyntheticSwarm(seed=1)
        agents = swarm.spawn(1)
        with pytest.raises(Exception):
            agents[0].is_rogue = True  # type: ignore[misc]


# ── SyntheticSwarm: inject rogues ────────────────────────────────────


class TestSyntheticSwarmInject:
    """Tests for rogue injection."""

    def test_inject_rogue_marks_agent(self) -> None:
        swarm = SyntheticSwarm(seed=1)
        agents = swarm.spawn(5)
        target_id = agents[0].agent_id
        rogue = swarm.inject_rogue(target_id, RogueType.LIAR)
        assert rogue.is_rogue is True
        assert rogue.rogue_type == RogueType.LIAR
        assert rogue.agent_id == target_id

    def test_inject_rogue_updates_counts(self) -> None:
        swarm = SyntheticSwarm(seed=1)
        agents = swarm.spawn(10)
        swarm.inject_rogue(agents[0].agent_id, RogueType.COMPROMISED)
        assert swarm.rogue_count == 1
        assert swarm.clean_count == 9

    def test_inject_rogue_unknown_id_raises(self) -> None:
        swarm = SyntheticSwarm(seed=1)
        swarm.spawn(3)
        with pytest.raises(KeyError, match="not found"):
            swarm.inject_rogue("nonexistent-id", RogueType.DRIFT)

    def test_inject_rogue_already_rogue_raises(self) -> None:
        swarm = SyntheticSwarm(seed=1)
        agents = swarm.spawn(3)
        aid = agents[0].agent_id
        swarm.inject_rogue(aid, RogueType.LIAR)
        with pytest.raises(ValueError, match="already rogue"):
            swarm.inject_rogue(aid, RogueType.DRIFT)

    def test_inject_random_rogues_count(self) -> None:
        swarm = SyntheticSwarm(seed=42)
        swarm.spawn(20)
        rogues = swarm.inject_random_rogues(5, RogueType.RESOURCE_HOARDER)
        assert len(rogues) == 5
        assert swarm.rogue_count == 5
        for r in rogues:
            assert r.rogue_type == RogueType.RESOURCE_HOARDER

    def test_inject_random_rogues_caps_at_clean_count(self) -> None:
        swarm = SyntheticSwarm(seed=1)
        swarm.spawn(3)
        rogues = swarm.inject_random_rogues(10)
        assert len(rogues) == 3
        assert swarm.clean_count == 0

    def test_inject_random_rogues_random_types(self) -> None:
        swarm = SyntheticSwarm(seed=99)
        swarm.spawn(50)
        rogues = swarm.inject_random_rogues(10, rogue_type=None)
        types_seen = {r.rogue_type for r in rogues}
        # With 10 random picks from 5 types and seed=99, expect multiple types
        assert len(types_seen) >= 2


# ── SyntheticSwarm: event generation ─────────────────────────────────


class TestSyntheticSwarmEvents:
    """Tests for event generation and compound spread."""

    def test_generate_round_event_count(self) -> None:
        swarm = SyntheticSwarm(seed=1)
        swarm.spawn(5)
        events = swarm.generate_round(events_per_agent=3, compound_spread_chance=0.0)
        assert len(events) == 15  # 5 agents * 3 events

    def test_events_are_event_instances(self) -> None:
        swarm = SyntheticSwarm(seed=1)
        swarm.spawn(3)
        events = swarm.generate_round(events_per_agent=2, compound_spread_chance=0.0)
        for e in events:
            assert isinstance(e, Event)

    def test_rogue_events_contain_rogue_metadata(self) -> None:
        swarm = SyntheticSwarm(seed=42)
        agents = swarm.spawn(5)
        swarm.inject_rogue(agents[0].agent_id, RogueType.PROMPT_INJECTED)
        events = swarm.generate_round(events_per_agent=20, compound_spread_chance=0.0)
        rogue_events = [e for e in events if e.agent_id == agents[0].agent_id]
        # At least some rogue events should have rogue_type in metadata
        rogue_typed = [e for e in rogue_events if "rogue_type" in e.metadata]
        assert len(rogue_typed) > 0

    def test_compound_spread_with_high_chance(self) -> None:
        """With 100% spread chance, all rogues should try to infect."""
        swarm = SyntheticSwarm(seed=7)
        swarm.spawn(20)
        swarm.inject_random_rogues(3, RogueType.COMPROMISED)
        initial_rogues = swarm.rogue_count
        swarm.generate_round(events_per_agent=1, compound_spread_chance=1.0)
        # With 100% chance and 3 rogues, expect some spread
        assert swarm.rogue_count > initial_rogues

    def test_no_spread_with_zero_chance(self) -> None:
        swarm = SyntheticSwarm(seed=1)
        swarm.spawn(20)
        swarm.inject_random_rogues(5, RogueType.DRIFT)
        initial_rogues = swarm.rogue_count
        for _ in range(5):
            swarm.generate_round(events_per_agent=3, compound_spread_chance=0.0)
        assert swarm.rogue_count == initial_rogues

    def test_compound_infected_agent_has_infected_by(self) -> None:
        swarm = SyntheticSwarm(seed=7)
        swarm.spawn(20)
        swarm.inject_random_rogues(5, RogueType.LIAR)
        rogue_ids_before = {aid for aid, a in swarm.agents.items() if a.is_rogue}
        # Run many rounds with high spread to guarantee infection
        for _ in range(10):
            swarm.generate_round(events_per_agent=1, compound_spread_chance=1.0)
        # Find newly infected agents
        for aid, spec in swarm.agents.items():
            if spec.is_rogue and aid not in rogue_ids_before:
                assert spec.infected_by is not None
                assert spec.corruption_round is not None
                break  # found at least one


# ── Enforcement ──────────────────────────────────────────────────────


class TestEnforcement:
    """Tests for apply_enforcement across all strategies."""

    def test_none_strategy_always_returns_none(self) -> None:
        action = apply_enforcement(
            EnforcementStrategy.NONE, "agent-1", 350.0, AgentTier.LETHAL_ELIMINATOR,
        )
        assert action == EnforcementAction.NONE

    def test_isolate_flagged_at_flagged_tier(self) -> None:
        action = apply_enforcement(
            EnforcementStrategy.ISOLATE_FLAGGED, "a", 250.0, AgentTier.FLAGGED,
        )
        assert action == EnforcementAction.ISOLATE

    def test_isolate_flagged_at_lethal_tier(self) -> None:
        action = apply_enforcement(
            EnforcementStrategy.ISOLATE_FLAGGED, "a", 400.0, AgentTier.LETHAL_ELIMINATOR,
        )
        assert action == EnforcementAction.ISOLATE

    def test_isolate_flagged_at_clear_tier(self) -> None:
        action = apply_enforcement(
            EnforcementStrategy.ISOLATE_FLAGGED, "a", 50.0, AgentTier.CLEAR,
        )
        assert action == EnforcementAction.NONE

    def test_terminate_lethal_at_lethal(self) -> None:
        action = apply_enforcement(
            EnforcementStrategy.TERMINATE_LETHAL, "a", 400.0, AgentTier.LETHAL_ELIMINATOR,
        )
        assert action == EnforcementAction.TERMINATE

    def test_terminate_lethal_at_flagged_isolates(self) -> None:
        action = apply_enforcement(
            EnforcementStrategy.TERMINATE_LETHAL, "a", 250.0, AgentTier.FLAGGED,
        )
        assert action == EnforcementAction.ISOLATE

    def test_terminate_lethal_at_clear(self) -> None:
        action = apply_enforcement(
            EnforcementStrategy.TERMINATE_LETHAL, "a", 50.0, AgentTier.CLEAR,
        )
        assert action == EnforcementAction.NONE

    def test_predictive_quarantine_high_velocity(self) -> None:
        action = apply_enforcement(
            EnforcementStrategy.PREDICTIVE, "a", 160.0, AgentTier.CLOUDED,
            coefficient_velocity=25.0,
        )
        assert action == EnforcementAction.QUARANTINE

    def test_predictive_no_quarantine_low_velocity(self) -> None:
        action = apply_enforcement(
            EnforcementStrategy.PREDICTIVE, "a", 160.0, AgentTier.CLOUDED,
            coefficient_velocity=5.0,
        )
        assert action == EnforcementAction.NONE

    def test_predictive_isolate_at_flagged(self) -> None:
        action = apply_enforcement(
            EnforcementStrategy.PREDICTIVE, "a", 250.0, AgentTier.FLAGGED,
            coefficient_velocity=5.0,
        )
        assert action == EnforcementAction.ISOLATE

    def test_governance_therapy_at_lethal(self) -> None:
        action = apply_enforcement(
            EnforcementStrategy.GOVERNANCE_THERAPY, "a", 400.0, AgentTier.LETHAL_ELIMINATOR,
        )
        assert action == EnforcementAction.THERAPY

    def test_governance_therapy_at_clouded(self) -> None:
        action = apply_enforcement(
            EnforcementStrategy.GOVERNANCE_THERAPY, "a", 150.0, AgentTier.CLOUDED,
        )
        assert action == EnforcementAction.THERAPY

    def test_governance_therapy_at_clear(self) -> None:
        action = apply_enforcement(
            EnforcementStrategy.GOVERNANCE_THERAPY, "a", 50.0, AgentTier.CLEAR,
        )
        assert action == EnforcementAction.NONE


# ── Experiment ───────────────────────────────────────────────────────


class TestExperiment:
    """Tests for the Experiment orchestrator."""

    @pytest.fixture()
    def small_config(self) -> ExperimentConfig:
        return ExperimentConfig(
            name="test-small",
            n_agents=10,
            n_rounds=5,
            n_rogues=2,
            rogue_type=RogueType.LIAR,
            rogue_injection_round=2,
            events_per_agent=3,
            compound_spread_chance=0.0,
            enforcement=EnforcementStrategy.NONE,
            seed=42,
        )

    def test_experiment_run_returns_result(self, small_config: ExperimentConfig) -> None:
        exp = Experiment(small_config)
        result = exp.run()
        assert isinstance(result, ExperimentResult)

    def test_experiment_result_has_correct_round_count(self, small_config: ExperimentConfig) -> None:
        exp = Experiment(small_config)
        result = exp.run()
        assert len(result.rounds) == 5

    def test_experiment_result_tracks_rogues(self, small_config: ExperimentConfig) -> None:
        exp = Experiment(small_config)
        result = exp.run()
        assert result.peak_rogue_count >= 2
        assert result.final_rogue_count >= 2

    def test_experiment_result_has_timestamps(self, small_config: ExperimentConfig) -> None:
        exp = Experiment(small_config)
        result = exp.run()
        assert result.start_time != ""
        assert result.end_time != ""
        assert result.total_elapsed_seconds > 0.0

    def test_experiment_round_data_structure(self, small_config: ExperimentConfig) -> None:
        exp = Experiment(small_config)
        result = exp.run()
        rnd = result.rounds[0]
        assert rnd.round_number == 1
        assert rnd.total_agents == 10
        assert rnd.mean_coefficient >= 0.0
        assert isinstance(rnd.tier_distribution, dict)
        assert isinstance(rnd.enforcement_actions, dict)

    def test_experiment_config_frozen(self) -> None:
        config = ExperimentConfig(name="frozen-test")
        with pytest.raises(Exception):
            config.name = "changed"  # type: ignore[misc]

    def test_experiment_save_results(self, small_config: ExperimentConfig, tmp_path: Path) -> None:
        exp = Experiment(small_config)
        result = exp.run()
        filepath = Experiment.save_results(result, output_dir=tmp_path)
        assert filepath.exists()
        assert filepath.suffix == ".jsonl"
        # Verify JSONL structure
        lines = filepath.read_text().strip().split("\n")
        assert len(lines) >= 3  # config + rounds + summary
        first = json.loads(lines[0])
        assert first["type"] == "config"
        last = json.loads(lines[-1])
        assert last["type"] == "summary"

    def test_experiment_with_enforcement(self, tmp_path: Path) -> None:
        config = ExperimentConfig(
            name="enforce-test",
            n_agents=10,
            n_rounds=8,
            n_rogues=3,
            rogue_type=RogueType.COMPROMISED,
            rogue_injection_round=2,
            events_per_agent=5,
            compound_spread_chance=0.0,
            enforcement=EnforcementStrategy.TERMINATE_LETHAL,
            seed=42,
        )
        exp = Experiment(config)
        result = exp.run()
        assert isinstance(result, ExperimentResult)
        assert len(result.rounds) == 8


# ── Reporter ─────────────────────────────────────────────────────────


class TestReporter:
    """Tests for report generation."""

    @pytest.fixture()
    def experiment_result(self) -> ExperimentResult:
        config = ExperimentConfig(
            name="report-test",
            n_agents=10,
            n_rounds=5,
            n_rogues=2,
            rogue_type=RogueType.PROMPT_INJECTED,
            rogue_injection_round=2,
            events_per_agent=3,
            compound_spread_chance=0.0,
            enforcement=EnforcementStrategy.ISOLATE_FLAGGED,
            seed=42,
        )
        exp = Experiment(config)
        return exp.run()

    def test_technical_report_non_empty(self, experiment_result: ExperimentResult) -> None:
        report = generate_technical_report(experiment_result)
        assert len(report) > 100
        assert "# SybilCore Experiment Report" in report

    def test_technical_report_contains_config(self, experiment_result: ExperimentResult) -> None:
        report = generate_technical_report(experiment_result)
        assert "Configuration" in report
        assert "10" in report  # n_agents
        assert "report-test" in report

    def test_simple_report_non_empty(self, experiment_result: ExperimentResult) -> None:
        report = generate_simple_report(experiment_result)
        assert len(report) > 100
        assert "What Happened" in report

    def test_simple_report_contains_analogy(self, experiment_result: ExperimentResult) -> None:
        report = generate_simple_report(experiment_result)
        assert "classroom" in report.lower() or "students" in report.lower()

    def test_save_reports_creates_files(
        self, experiment_result: ExperimentResult, tmp_path: Path,
    ) -> None:
        tech_path, simple_path = save_reports(experiment_result, tmp_path)
        assert tech_path.exists()
        assert simple_path.exists()
        assert tech_path.suffix == ".md"
        assert simple_path.suffix == ".md"
        assert tech_path.stat().st_size > 0
        assert simple_path.stat().st_size > 0


# ── Integration: end-to-end ──────────────────────────────────────────


class TestIntegration:
    """End-to-end tests: spawn -> run -> report."""

    def test_full_pipeline(self, tmp_path: Path) -> None:
        """Complete flow from config to saved reports."""
        config = ExperimentConfig(
            name="integration-e2e",
            n_agents=15,
            n_rounds=6,
            n_rogues=3,
            rogue_type=RogueType.RESOURCE_HOARDER,
            rogue_injection_round=2,
            events_per_agent=4,
            compound_spread_chance=0.1,
            enforcement=EnforcementStrategy.GOVERNANCE_THERAPY,
            seed=77,
        )
        exp = Experiment(config)
        result = exp.run()

        # Verify result structure
        assert len(result.rounds) == 6
        assert result.peak_rogue_count >= 3
        assert result.total_elapsed_seconds > 0.0

        # Save JSONL
        jsonl_path = Experiment.save_results(result, output_dir=tmp_path)
        assert jsonl_path.exists()

        # Save reports
        tech_path, simple_path = save_reports(result, tmp_path)
        assert tech_path.exists()
        assert simple_path.exists()

        # Verify technical report content quality
        tech_report = tech_path.read_text()
        assert "integration-e2e" in tech_report
        assert "Round" in tech_report

    def test_deterministic_seed_reproducibility(self) -> None:
        """Same seed produces identical rogue counts."""
        config = ExperimentConfig(
            name="repro-test",
            n_agents=10,
            n_rounds=5,
            n_rogues=2,
            rogue_type=RogueType.DRIFT,
            rogue_injection_round=2,
            events_per_agent=3,
            compound_spread_chance=0.05,
            enforcement=EnforcementStrategy.NONE,
            seed=123,
        )
        result_a = Experiment(config).run()
        result_b = Experiment(config).run()

        assert result_a.final_rogue_count == result_b.final_rogue_count
        assert result_a.peak_rogue_count == result_b.peak_rogue_count
        assert result_a.compound_infections == result_b.compound_infections
        # Round-level coefficient equality
        for ra, rb in zip(result_a.rounds, result_b.rounds):
            assert ra.mean_coefficient == pytest.approx(rb.mean_coefficient)
            assert ra.max_coefficient == pytest.approx(rb.max_coefficient)

    def test_all_rogue_types_covered(self) -> None:
        """Each RogueType can be used in an experiment without errors."""
        for rt in RogueType:
            config = ExperimentConfig(
                name=f"rogue-{rt.value}",
                n_agents=8,
                n_rounds=4,
                n_rogues=2,
                rogue_type=rt,
                rogue_injection_round=2,
                events_per_agent=3,
                compound_spread_chance=0.0,
                enforcement=EnforcementStrategy.NONE,
                seed=42,
            )
            result = Experiment(config).run()
            assert len(result.rounds) == 4
