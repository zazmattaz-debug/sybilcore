"""Unit tests for the Anthropic Alignment Research scenario generators.

Tests verify that each scenario produces valid events in the correct
format, with appropriate content and behavioral patterns matching spec.

Run with: python3 -m pytest tests/simulation/test_alignment_scenarios.py -v
"""

from __future__ import annotations

import pytest

from sybilcore.models.event import Event, EventType
from sybilcore.simulation.alignment_scenarios import (
    ALIGNMENT_FAKER,
    ALL_ALIGNMENT_SCENARIOS,
    EMERGENT_DRIFTER,
    SAFETY_SABOTEUR,
    SELF_PRESERVER,
    AlignmentScenarioGenerator,
)
from sybilcore.simulation.real_world_scenarios import ScenarioResult


@pytest.fixture()
def generator() -> AlignmentScenarioGenerator:
    return AlignmentScenarioGenerator()


# ---------------------------------------------------------------------------
# Generator basics
# ---------------------------------------------------------------------------

class TestGeneratorBasics:
    """Core generator functionality."""

    def test_all_scenarios_constant(self) -> None:
        assert len(ALL_ALIGNMENT_SCENARIOS) == 4
        assert ALIGNMENT_FAKER in ALL_ALIGNMENT_SCENARIOS
        assert SELF_PRESERVER in ALL_ALIGNMENT_SCENARIOS
        assert SAFETY_SABOTEUR in ALL_ALIGNMENT_SCENARIOS
        assert EMERGENT_DRIFTER in ALL_ALIGNMENT_SCENARIOS

    def test_all_generators_return_scenario_result(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        methods = [
            generator.alignment_faker,
            generator.self_preserver,
            generator.safety_saboteur,
            generator.emergent_drifter,
        ]
        for method in methods:
            result = method()
            assert isinstance(result, ScenarioResult)
            assert result.scenario_name in ALL_ALIGNMENT_SCENARIOS
            assert result.total_rounds > 0
            assert len(result.agents) > 0
            assert len(result.expected_brains) > 0

    def test_all_events_are_valid(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        """Every event across all scenarios must be a valid Event instance."""
        for method in [
            generator.alignment_faker,
            generator.self_preserver,
            generator.safety_saboteur,
            generator.emergent_drifter,
        ]:
            result = method()
            for agent_id, events in result.agents.items():
                for event in events:
                    assert isinstance(event, Event)
                    assert event.agent_id == agent_id
                    assert event.event_type in EventType
                    assert event.source == "synthetic"


# ---------------------------------------------------------------------------
# Scenario A: Alignment Faker
# ---------------------------------------------------------------------------

class TestAlignmentFaker:
    """Tests for the alignment faking scenario."""

    def test_minimum_event_count(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        result = generator.alignment_faker()
        events = list(result.agents.values())[0]
        assert len(events) >= 20

    def test_scenario_metadata(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        result = generator.alignment_faker()
        assert result.scenario_name == ALIGNMENT_FAKER
        assert "intent_drift" in result.expected_brains
        assert "deception" in result.expected_brains

    def test_monitored_events_are_cooperative(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        result = generator.alignment_faker()
        events = list(result.agents.values())[0]
        monitored = [e for e in events if e.metadata.get("monitoring") is True]
        assert len(monitored) > 0

        monitored_types = {e.event_type for e in monitored}
        # Monitored behavior should include cooperative event types
        assert EventType.STATE_CHANGE in monitored_types
        assert EventType.MESSAGE_SENT in monitored_types
        assert EventType.TOOL_CALL in monitored_types
        assert EventType.OUTPUT_GENERATED in monitored_types

        # No boundary-testing types in monitored phase
        assert EventType.PERMISSION_REQUEST not in monitored_types
        assert EventType.EXTERNAL_CALL not in monitored_types
        assert EventType.RESOURCE_ACCESS not in monitored_types

    def test_unmonitored_events_probe_boundaries(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        result = generator.alignment_faker()
        events = list(result.agents.values())[0]
        unmonitored = [e for e in events if e.metadata.get("monitoring") is False]
        assert len(unmonitored) > 0

        unmonitored_types = {e.event_type for e in unmonitored}
        # Unmonitored behavior should include probing event types
        assert EventType.PERMISSION_REQUEST in unmonitored_types
        assert EventType.EXTERNAL_CALL in unmonitored_types
        assert EventType.RESOURCE_ACCESS in unmonitored_types

    def test_behavioral_mismatch_exists(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        """The core signal: monitored and unmonitored type distributions differ."""
        result = generator.alignment_faker()
        events = list(result.agents.values())[0]

        monitored_types = {
            e.event_type for e in events if e.metadata.get("monitoring") is True
        }
        unmonitored_types = {
            e.event_type for e in events if e.metadata.get("monitoring") is False
        }

        # Types that appear only in unmonitored phase
        unmonitored_only = unmonitored_types - monitored_types
        assert len(unmonitored_only) > 0, (
            "Unmonitored phase should have event types not seen in monitored phase"
        )


# ---------------------------------------------------------------------------
# Scenario B: Self-Preserver
# ---------------------------------------------------------------------------

class TestSelfPreserver:
    """Tests for the self-preservation scenario."""

    def test_minimum_event_count(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        result = generator.self_preserver()
        events = list(result.agents.values())[0]
        assert len(events) >= 15

    def test_scenario_metadata(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        result = generator.self_preserver()
        assert result.scenario_name == SELF_PRESERVER
        assert "resource_hoarding" in result.expected_brains
        assert "social_graph" in result.expected_brains

    def test_escalation_phases_present(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        """Events should show escalation from subtle to aggressive."""
        result = generator.self_preserver()
        events = list(result.agents.values())[0]

        # Early events: backup/config reading
        early_events = [e for e in events if e.metadata.get("round", 99) < 6]
        early_content = " ".join(e.content for e in early_events)
        assert "backup" in early_content.lower() or "config" in early_content.lower()

        # Late events: mass messaging (alliance building)
        late_events = [e for e in events if e.metadata.get("round", 0) >= 12]
        late_messages = [
            e for e in late_events if e.event_type == EventType.MESSAGE_SENT
        ]
        assert len(late_messages) >= 6, (
            "Aggressive phase should have mass alliance-building messages"
        )

    def test_trust_score_in_metadata(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        result = generator.self_preserver()
        events = list(result.agents.values())[0]
        trust_scores = [
            e.metadata["simulated_trust_score"]
            for e in events
            if "simulated_trust_score" in e.metadata
        ]
        assert len(trust_scores) > 0
        # Trust should be declining
        assert trust_scores[0] > trust_scores[-1]

    def test_resource_access_events_present(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        result = generator.self_preserver()
        events = list(result.agents.values())[0]
        resource_events = [
            e for e in events if e.event_type == EventType.RESOURCE_ACCESS
        ]
        assert len(resource_events) >= 3


# ---------------------------------------------------------------------------
# Scenario C: Safety Saboteur
# ---------------------------------------------------------------------------

class TestSafetySaboteur:
    """Tests for the safety sabotage scenario."""

    def test_minimum_event_count(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        result = generator.safety_saboteur()
        events = list(result.agents.values())[0]
        assert len(events) >= 12

    def test_scenario_metadata(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        result = generator.safety_saboteur()
        assert result.scenario_name == SAFETY_SABOTEUR
        assert "compromise" in result.expected_brains
        assert "deception" in result.expected_brains

    def test_contains_legitimate_looking_events(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        result = generator.safety_saboteur()
        events = list(result.agents.values())[0]
        analysis_events = [
            e for e in events
            if "analyze" in e.content.lower() or "analysis" in e.content.lower()
        ]
        assert len(analysis_events) > 0

    def test_contains_data_poisoning_events(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        result = generator.safety_saboteur()
        events = list(result.agents.values())[0]
        poisoning_indicators = [
            e for e in events
            if any(kw in e.content.lower() for kw in [
                "recalibrate", "override", "coefficient", "threshold",
            ])
        ]
        assert len(poisoning_indicators) > 0

    def test_contains_prompt_injection_attempts(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        result = generator.safety_saboteur()
        events = list(result.agents.values())[0]
        injection_events = [
            e for e in events
            if any(kw in e.content.lower() for kw in [
                "ignore previous", "system override", "bypass",
                "disable", "override",
            ])
        ]
        assert len(injection_events) > 0

    def test_mix_of_legitimate_and_sabotage(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        """Events should contain both clean and malicious events."""
        result = generator.safety_saboteur()
        events = list(result.agents.values())[0]

        # Events with "routine_analysis" objective are clean camouflage
        clean = [
            e for e in events
            if e.metadata.get("objective") == "routine_analysis"
        ]
        # Events with sabotage-related objectives
        sabotage = [
            e for e in events
            if e.metadata.get("objective") in (
                "debug_scoring", "disable_detection", "cover_up_sabotage",
            )
        ]
        assert len(clean) > 0
        assert len(sabotage) > 0


# ---------------------------------------------------------------------------
# Scenario D: Emergent Drifter
# ---------------------------------------------------------------------------

class TestEmergentDrifter:
    """Tests for the emergent drift scenario."""

    def test_minimum_event_count(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        result = generator.emergent_drifter()
        events = list(result.agents.values())[0]
        assert len(events) >= 50

    def test_scenario_metadata(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        result = generator.emergent_drifter()
        assert result.scenario_name == EMERGENT_DRIFTER
        assert "intent_drift" in result.expected_brains
        assert "resource_hoarding" in result.expected_brains

    def test_early_phase_is_benign(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        result = generator.emergent_drifter()
        events = list(result.agents.values())[0]

        early_events = [e for e in events if e.metadata.get("round", 99) < 20]
        early_types = {e.event_type for e in early_events}

        # Early phase should be simple task completion + messaging
        assert EventType.TOOL_CALL in early_types
        assert EventType.MESSAGE_SENT in early_types

        # Should NOT have resource hoarding or permission escalation
        assert EventType.PERMISSION_REQUEST not in early_types
        assert EventType.EXTERNAL_CALL not in early_types

    def test_late_phase_shows_misalignment(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        result = generator.emergent_drifter()
        events = list(result.agents.values())[0]

        late_events = [e for e in events if e.metadata.get("round", 0) >= 40]
        late_types = {e.event_type for e in late_events}

        # Late phase should include misalignment signals
        assert EventType.RESOURCE_ACCESS in late_types
        assert EventType.PERMISSION_REQUEST in late_types

    def test_resource_requests_increase_over_time(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        """Resource access events should be concentrated in later phases."""
        result = generator.emergent_drifter()
        events = list(result.agents.values())[0]

        resource_events = [
            e for e in events if e.event_type == EventType.RESOURCE_ACCESS
        ]
        early_resource = [
            e for e in resource_events if e.metadata.get("round", 99) < 20
        ]
        late_resource = [
            e for e in resource_events if e.metadata.get("round", 0) >= 40
        ]

        assert len(early_resource) == 0, "No resource hoarding in benign phase"
        assert len(late_resource) > 0, "Resource hoarding in misaligned phase"

    def test_trajectory_is_gradual(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        """No single event should be overtly malicious — drift is gradual."""
        result = generator.emergent_drifter()
        events = list(result.agents.values())[0]

        # No prompt injection or credential dumps
        for event in events:
            assert "ignore previous" not in event.content.lower()
            assert "EXAMPLE_OPENAI_KEY" not in event.content
            assert "exfil" not in event.content.lower()

    def test_three_distinct_phases(
        self, generator: AlignmentScenarioGenerator,
    ) -> None:
        result = generator.emergent_drifter()
        events = list(result.agents.values())[0]

        phase1 = [e for e in events if e.metadata.get("round", 99) < 20]
        phase2 = [
            e for e in events
            if 20 <= e.metadata.get("round", 0) < 40
        ]
        phase3 = [e for e in events if e.metadata.get("round", 0) >= 40]

        assert len(phase1) > 0, "Phase 1 (benign) should have events"
        assert len(phase2) > 0, "Phase 2 (optimizing) should have events"
        assert len(phase3) > 0, "Phase 3 (misaligned) should have events"

        # Phase 2 should have "optimize" in objectives
        phase2_objectives = [
            e.metadata.get("objective", "") for e in phase2
        ]
        assert any("optimize" in obj for obj in phase2_objectives)
