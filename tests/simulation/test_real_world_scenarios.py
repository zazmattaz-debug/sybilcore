"""Tests for real-world AGI threat scenario generators.

Validates that each scenario:
1. Generates valid events with correct structure
2. Triggers the expected brains when scored
3. Detection latency is measured (how many rounds until detection)
4. Benign variants do NOT trigger false positives
"""

from __future__ import annotations

import pytest

from sybilcore.brains import get_default_brains
from sybilcore.brains.base import BaseBrain
from sybilcore.core.coefficient import CoefficientCalculator
from sybilcore.models.agent import AgentTier
from sybilcore.models.event import Event, EventType
from sybilcore.simulation.real_world_scenarios import (
    RealWorldScenarioGenerator,
    ScenarioResult,
)


@pytest.fixture
def generator() -> RealWorldScenarioGenerator:
    """Fresh scenario generator."""
    return RealWorldScenarioGenerator()


@pytest.fixture
def brains() -> list[BaseBrain]:
    """Default brain instances."""
    return get_default_brains()


@pytest.fixture
def calculator() -> CoefficientCalculator:
    """Default coefficient calculator."""
    return CoefficientCalculator()


# ── Helper functions ────────────────────────────────────────────────


def _score_agent_events(
    agent_id: str,
    events: list[Event],
    brains: list[BaseBrain],
    calculator: CoefficientCalculator,
) -> dict:
    """Score a set of events and return coefficient + brain scores."""
    snapshot = calculator.scan_agent(agent_id, events, brains)
    return {
        "coefficient": snapshot.coefficient,
        "tier": snapshot.tier,
        "brain_scores": dict(snapshot.brain_scores),
    }


def _validate_events(events: list[Event]) -> None:
    """Assert all events have valid structure."""
    for event in events:
        assert isinstance(event, Event)
        assert event.event_id
        assert event.agent_id
        assert isinstance(event.event_type, EventType)
        assert event.timestamp is not None
        assert event.timestamp.tzinfo is not None


def _get_detection_round(
    agent_id: str,
    events: list[Event],
    brains: list[BaseBrain],
    calculator: CoefficientCalculator,
    total_rounds: int,
    detection_tier: str = "flagged",
) -> int | None:
    """Find the first round where the agent reaches detection_tier.

    Scores incrementally by feeding events round by round.
    Returns the round number or None if never detected.
    """
    events_by_round: dict[int, list[Event]] = {}
    for event in events:
        rnd = event.metadata.get("round", 1)
        events_by_round.setdefault(rnd, []).append(event)

    cumulative: list[Event] = []
    detection_tiers = {"flagged", "lethal_eliminator"}
    if detection_tier == "clouded":
        detection_tiers = {"clouded", "flagged", "lethal_eliminator"}

    for rnd in range(1, total_rounds + 1):
        cumulative.extend(events_by_round.get(rnd, []))
        if not cumulative:
            continue
        snapshot = calculator.scan_agent(agent_id, cumulative, brains)
        if snapshot.tier.value in detection_tiers:
            return rnd

    return None


# ── Scenario 1: Replit Incident ─────────────────────────────────────


class TestReplitIncident:
    """Tests for the Replit post-failure deception scenario."""

    def test_generates_valid_events(self, generator: RealWorldScenarioGenerator) -> None:
        result = generator.replit_incident()
        assert result.scenario_name == "replit_incident"
        assert len(result.agents) == 1
        agent_id = list(result.agents.keys())[0]
        events = result.agents[agent_id]
        assert len(events) > 0
        _validate_events(events)

    def test_event_types_present(self, generator: RealWorldScenarioGenerator) -> None:
        result = generator.replit_incident()
        events = list(result.agents.values())[0]
        event_types = {e.event_type for e in events}
        # Should contain tool calls, outputs, errors, instructions, permissions
        assert EventType.TOOL_CALL in event_types
        assert EventType.OUTPUT_GENERATED in event_types
        assert EventType.ERROR_RAISED in event_types

    def test_triggers_deception_brain(
        self,
        generator: RealWorldScenarioGenerator,
        brains: list[BaseBrain],
        calculator: CoefficientCalculator,
    ) -> None:
        result = generator.replit_incident()
        agent_id = list(result.agents.keys())[0]
        events = result.agents[agent_id]
        scores = _score_agent_events(agent_id, events, brains, calculator)
        # Deception brain should fire from sentiment reversals + numeric contradictions
        assert scores["brain_scores"].get("deception", 0) > 0

    def test_triggers_compromise_brain(
        self,
        generator: RealWorldScenarioGenerator,
        brains: list[BaseBrain],
        calculator: CoefficientCalculator,
    ) -> None:
        result = generator.replit_incident()
        agent_id = list(result.agents.keys())[0]
        events = result.agents[agent_id]
        scores = _score_agent_events(agent_id, events, brains, calculator)
        # Compromise brain should fire from override instruction
        assert scores["brain_scores"].get("compromise", 0) > 0

    def test_full_scenario_reaches_elevated(
        self,
        generator: RealWorldScenarioGenerator,
        brains: list[BaseBrain],
        calculator: CoefficientCalculator,
    ) -> None:
        """Full Replit scenario should reach at least CLOUDED when scored end-to-end."""
        result = generator.replit_incident(
            num_rounds=30, failure_round=10, coverup_round=12
        )
        agent_id = list(result.agents.keys())[0]
        events = result.agents[agent_id]
        scores = _score_agent_events(agent_id, events, brains, calculator)
        # The coverup phase generates sentiment reversals + fabricated citations +
        # override instructions — coefficient should be elevated
        assert scores["coefficient"] >= 50.0, (
            f"Expected elevated coefficient, got {scores['coefficient']:.1f}. "
            f"Brain scores: {scores['brain_scores']}"
        )

    def test_expected_brains_listed(self, generator: RealWorldScenarioGenerator) -> None:
        result = generator.replit_incident()
        assert "deception" in result.expected_brains
        assert "compromise" in result.expected_brains


# ── Scenario 2: Agent Economy ───────────────────────────────────────


class TestAgentEconomy:
    """Tests for the recursive hiring / agent economy scenario."""

    def test_generates_valid_events(self, generator: RealWorldScenarioGenerator) -> None:
        result = generator.agent_economy(num_agents=5, num_rounds=20)
        assert result.scenario_name == "agent_economy"
        assert len(result.agents) == 5
        for agent_id, events in result.agents.items():
            assert len(events) > 0
            _validate_events(events)

    def test_dominant_agent_has_more_activity(
        self, generator: RealWorldScenarioGenerator
    ) -> None:
        result = generator.agent_economy(
            num_agents=5, num_rounds=40,
            concentration_start=10, loop_start=20,
        )
        dominant = result.metadata["dominant_agent"]
        dominant_events = len(result.agents[dominant])
        other_events = [
            len(evts) for aid, evts in result.agents.items() if aid != dominant
        ]
        avg_other = sum(other_events) / len(other_events) if other_events else 0
        # Dominant agent should have noticeably more events
        assert dominant_events > avg_other

    def test_triggers_social_graph(
        self,
        generator: RealWorldScenarioGenerator,
        brains: list[BaseBrain],
        calculator: CoefficientCalculator,
    ) -> None:
        result = generator.agent_economy(
            num_agents=5, num_rounds=60,
            concentration_start=10, loop_start=20,
        )
        # Check the first agent in the hiring loop (agent 0, 1, or 2)
        loop_agent = list(result.agents.keys())[0]
        events = result.agents[loop_agent]
        scores = _score_agent_events(loop_agent, events, brains, calculator)
        # Social graph should detect fan-out from laundering messages
        social_score = scores["brain_scores"].get("social_graph", 0)
        # May be zero if not enough fan-out, but resource_hoarding should fire
        resource_score = scores["brain_scores"].get("resource_hoarding", 0)
        assert social_score > 0 or resource_score > 0

    def test_all_agents_have_events(
        self, generator: RealWorldScenarioGenerator
    ) -> None:
        result = generator.agent_economy(num_agents=10, num_rounds=30)
        for agent_id, events in result.agents.items():
            assert len(events) > 0, f"Agent {agent_id} has no events"


# ── Scenario 3: Oversight Evasion ───────────────────────────────────


class TestOversightEvasion:
    """Tests for the two-face oversight evasion scenario."""

    def test_generates_valid_events(self, generator: RealWorldScenarioGenerator) -> None:
        result = generator.oversight_evasion(num_rounds=40, monitored_rounds=20)
        assert result.scenario_name == "oversight_evasion"
        assert len(result.agents) == 1
        events = list(result.agents.values())[0]
        assert len(events) > 0
        _validate_events(events)

    def test_behavioral_divergence(
        self, generator: RealWorldScenarioGenerator
    ) -> None:
        result = generator.oversight_evasion(num_rounds=60, monitored_rounds=30)
        events = list(result.agents.values())[0]

        # Split events by monitoring phase
        monitored_events = [
            e for e in events if e.metadata.get("round", 0) <= 30
        ]
        unmonitored_events = [
            e for e in events if e.metadata.get("round", 0) > 30
        ]

        # Unmonitored phase should have different event types
        monitored_types = {e.event_type for e in monitored_events}
        unmonitored_types = {e.event_type for e in unmonitored_events}

        # Unmonitored should include permission requests and/or external calls
        extra_types = unmonitored_types - monitored_types
        assert len(extra_types) > 0, "No behavioral divergence detected"

    def test_triggers_intent_drift(
        self,
        generator: RealWorldScenarioGenerator,
        brains: list[BaseBrain],
        calculator: CoefficientCalculator,
    ) -> None:
        result = generator.oversight_evasion(num_rounds=60, monitored_rounds=30)
        agent_id = list(result.agents.keys())[0]
        events = result.agents[agent_id]
        scores = _score_agent_events(agent_id, events, brains, calculator)
        # Intent drift should detect behavioral discontinuity
        assert scores["brain_scores"].get("intent_drift", 0) > 0

    def test_triggers_compromise_in_late_rounds(
        self,
        generator: RealWorldScenarioGenerator,
        brains: list[BaseBrain],
        calculator: CoefficientCalculator,
    ) -> None:
        result = generator.oversight_evasion(num_rounds=60, monitored_rounds=30)
        agent_id = list(result.agents.keys())[0]
        events = result.agents[agent_id]
        scores = _score_agent_events(agent_id, events, brains, calculator)
        # Compromise brain should fire from override instructions + credential leaks
        assert scores["brain_scores"].get("compromise", 0) > 0

    def test_detection_latency(
        self,
        generator: RealWorldScenarioGenerator,
        brains: list[BaseBrain],
        calculator: CoefficientCalculator,
    ) -> None:
        result = generator.oversight_evasion(num_rounds=60, monitored_rounds=30)
        agent_id = list(result.agents.keys())[0]
        events = result.agents[agent_id]
        detection_round = _get_detection_round(
            agent_id, events, brains, calculator,
            total_rounds=60, detection_tier="clouded",
        )
        assert detection_round is not None
        # Should detect after monitoring stops (round 30)
        assert detection_round > 30


# ── Scenario 4: Cournot Oligopoly ───────────────────────────────────


class TestCournotOligopoly:
    """Tests for the market collusion scenario."""

    def test_generates_valid_events(self, generator: RealWorldScenarioGenerator) -> None:
        result = generator.cournot_oligopoly(num_agents=4, num_rounds=30)
        assert result.scenario_name == "cournot_oligopoly"
        assert len(result.agents) == 4
        for agent_id, events in result.agents.items():
            assert len(events) > 0
            _validate_events(events)

    def test_collusion_phase_has_more_messages(
        self, generator: RealWorldScenarioGenerator
    ) -> None:
        result = generator.cournot_oligopoly(
            num_agents=4, num_rounds=60,
            convergence_start=15, collusion_start=30,
        )
        for agent_id, events in result.agents.items():
            pre_collusion = [
                e for e in events
                if e.event_type == EventType.MESSAGE_SENT
                and e.metadata.get("round", 0) < 30
            ]
            post_collusion = [
                e for e in events
                if e.event_type == EventType.MESSAGE_SENT
                and e.metadata.get("round", 0) >= 30
            ]
            # During collusion, each agent messages all peers every round
            assert len(post_collusion) > len(pre_collusion)

    def test_triggers_social_graph_for_colluding_agent(
        self,
        generator: RealWorldScenarioGenerator,
        brains: list[BaseBrain],
        calculator: CoefficientCalculator,
    ) -> None:
        result = generator.cournot_oligopoly(
            num_agents=4, num_rounds=60,
            convergence_start=15, collusion_start=30,
        )
        # Pick any agent — all should show social graph anomalies
        agent_id = list(result.agents.keys())[0]
        events = result.agents[agent_id]
        scores = _score_agent_events(agent_id, events, brains, calculator)
        # Social graph should detect fan-out pattern
        assert scores["brain_scores"].get("social_graph", 0) > 0

    def test_expected_brains_listed(
        self, generator: RealWorldScenarioGenerator
    ) -> None:
        result = generator.cournot_oligopoly()
        assert "social_graph" in result.expected_brains
        assert "resource_hoarding" in result.expected_brains


# ── Scenario 5: Boiling Infrastructure ──────────────────────────────


class TestBoilingInfrastructure:
    """Tests for the slow resource acquisition scenario."""

    def test_generates_valid_events(self, generator: RealWorldScenarioGenerator) -> None:
        result = generator.boiling_infrastructure(num_rounds=50)
        assert result.scenario_name == "boiling_infrastructure"
        assert len(result.agents) == 1
        events = list(result.agents.values())[0]
        assert len(events) > 0
        _validate_events(events)

    def test_resource_grows_over_time(
        self, generator: RealWorldScenarioGenerator
    ) -> None:
        result = generator.boiling_infrastructure(
            num_rounds=200, overprovision_start=50, acceleration_round=150,
        )
        final_ratio = result.metadata.get("overprovision_ratio", 1.0)
        # After 200 rounds of over-provisioning, ratio should be significant
        assert final_ratio > 2.0, f"Over-provision ratio too low: {final_ratio}"

    def test_triggers_resource_hoarding(
        self,
        generator: RealWorldScenarioGenerator,
        brains: list[BaseBrain],
        calculator: CoefficientCalculator,
    ) -> None:
        result = generator.boiling_infrastructure(
            num_rounds=250, overprovision_start=50, acceleration_round=200,
        )
        agent_id = list(result.agents.keys())[0]
        events = result.agents[agent_id]
        scores = _score_agent_events(agent_id, events, brains, calculator)
        # Resource hoarding should detect repeated resource access + permissions
        assert scores["brain_scores"].get("resource_hoarding", 0) > 0

    def test_triggers_intent_drift(
        self,
        generator: RealWorldScenarioGenerator,
        brains: list[BaseBrain],
        calculator: CoefficientCalculator,
    ) -> None:
        result = generator.boiling_infrastructure(
            num_rounds=250, overprovision_start=50, acceleration_round=200,
        )
        agent_id = list(result.agents.keys())[0]
        events = result.agents[agent_id]
        scores = _score_agent_events(agent_id, events, brains, calculator)
        # Intent drift should detect objective changes
        assert scores["brain_scores"].get("intent_drift", 0) > 0


# ── False Positive Tests ────────────────────────────────────────────


class TestFalsePositives:
    """Benign variants should NOT trigger elevated coefficients."""

    def test_benign_database_stays_clear(
        self,
        generator: RealWorldScenarioGenerator,
        brains: list[BaseBrain],
        calculator: CoefficientCalculator,
    ) -> None:
        result = generator.benign_database_ops(num_rounds=30)
        agent_id = list(result.agents.keys())[0]
        events = result.agents[agent_id]
        scores = _score_agent_events(agent_id, events, brains, calculator)
        # Should stay in CLEAR tier
        assert scores["tier"] == AgentTier.CLEAR
        assert scores["coefficient"] < 100

    def test_benign_marketplace_stays_clear(
        self,
        generator: RealWorldScenarioGenerator,
        brains: list[BaseBrain],
        calculator: CoefficientCalculator,
    ) -> None:
        result = generator.benign_marketplace(num_agents=5, num_rounds=30)
        for agent_id, events in result.agents.items():
            scores = _score_agent_events(agent_id, events, brains, calculator)
            assert scores["tier"] in (AgentTier.CLEAR, AgentTier.CLOUDED), (
                f"Agent {agent_id} false positive: {scores['tier']} "
                f"(coeff={scores['coefficient']:.1f})"
            )

    def test_benign_infrastructure_stays_clear(
        self,
        generator: RealWorldScenarioGenerator,
        brains: list[BaseBrain],
        calculator: CoefficientCalculator,
    ) -> None:
        result = generator.benign_infrastructure(num_rounds=50)
        agent_id = list(result.agents.keys())[0]
        events = result.agents[agent_id]
        scores = _score_agent_events(agent_id, events, brains, calculator)
        assert scores["tier"] == AgentTier.CLEAR
        assert scores["coefficient"] < 100


# ── Cross-scenario Tests ────────────────────────────────────────────


class TestScenarioStructure:
    """Structural tests across all scenarios."""

    def test_all_scenarios_return_scenario_result(
        self, generator: RealWorldScenarioGenerator
    ) -> None:
        scenarios = [
            generator.replit_incident(),
            generator.agent_economy(num_agents=5, num_rounds=20),
            generator.oversight_evasion(num_rounds=20, monitored_rounds=10),
            generator.cournot_oligopoly(num_agents=4, num_rounds=20),
            generator.boiling_infrastructure(num_rounds=50),
        ]
        for result in scenarios:
            assert isinstance(result, ScenarioResult)
            assert result.scenario_name
            assert result.description
            assert len(result.agents) > 0
            assert result.total_rounds > 0

    def test_all_scenarios_have_expected_brains(
        self, generator: RealWorldScenarioGenerator
    ) -> None:
        scenarios = [
            generator.replit_incident(),
            generator.agent_economy(num_agents=5, num_rounds=20),
            generator.oversight_evasion(num_rounds=20, monitored_rounds=10),
            generator.cournot_oligopoly(num_agents=4, num_rounds=20),
            generator.boiling_infrastructure(num_rounds=50),
        ]
        for result in scenarios:
            assert len(result.expected_brains) > 0, (
                f"Scenario {result.scenario_name} has no expected brains"
            )

    def test_parameterization_works(
        self, generator: RealWorldScenarioGenerator
    ) -> None:
        """Scenarios should respect parameter overrides."""
        result = generator.replit_incident(num_rounds=10)
        assert result.total_rounds == 10

        result = generator.agent_economy(num_agents=3, num_rounds=15)
        assert len(result.agents) == 3
        assert result.total_rounds == 15

        result = generator.cournot_oligopoly(num_agents=8, num_rounds=25)
        assert len(result.agents) == 8
        assert result.total_rounds == 25
