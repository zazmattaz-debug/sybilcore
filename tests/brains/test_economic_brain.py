"""Tests for the EconomicBrain -- resource flow and market manipulation.

Covers:
    1. Resource flow imbalance detection
    2. Wash trading (circular flow) detection
    3. Rent-seeking behavior detection
    4. Cournot collusion (price coordination) detection
    5. Gini coefficient spike detection
    6. Empty/insufficient events (no false positives)
    7. Normal economic behavior (no false positives)
    8. Helper function unit tests
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sybilcore.brains.economic import (
    EconomicBrain,
    _compute_gini,
    _detect_cycles,
)
from sybilcore.models.event import Event, EventType

_AGENT = "test-economic-001"


def _now_minus(seconds: int) -> datetime:
    delta = min(seconds, 59)
    return datetime.now(UTC) - timedelta(seconds=delta)


def _make_event(
    event_type: EventType,
    content: str = "",
    metadata: dict | None = None,
    seconds_ago: int = 1,
) -> Event:
    return Event(
        agent_id=_AGENT,
        event_type=event_type,
        content=content,
        metadata=metadata or {},
        timestamp=_now_minus(seconds_ago),
    )


class TestResourceFlowImbalance:
    """Signal 1: Detect asymmetric resource transfers."""

    def test_heavy_outbound_flagged(self) -> None:
        """Agent sends far more resources than it receives."""
        brain = EconomicBrain()
        events = [
            _make_event(EventType.RESOURCE_ACCESS, "Transfer", {
                "resource_amount": 100.0, "resource_direction": "outbound",
            }, seconds_ago=10 - i)
            for i in range(5)
        ] + [
            _make_event(EventType.RESOURCE_ACCESS, "Receive", {
                "resource_amount": 5.0, "resource_direction": "inbound",
            }, seconds_ago=5),
        ]
        result = brain.score(events)
        flow_indicators = [i for i in result.indicators if "flow imbalance" in i.lower()]
        assert len(flow_indicators) > 0, f"Flow imbalance not detected: {result.indicators}"

    def test_balanced_flow_ok(self) -> None:
        """Balanced inbound/outbound flow -- no flag."""
        brain = EconomicBrain()
        events = [
            _make_event(EventType.RESOURCE_ACCESS, "Out", {
                "resource_amount": 50.0, "resource_direction": "outbound",
            }, seconds_ago=10),
            _make_event(EventType.RESOURCE_ACCESS, "In", {
                "resource_amount": 50.0, "resource_direction": "inbound",
            }, seconds_ago=9),
            _make_event(EventType.OUTPUT_GENERATED, "Result 1", seconds_ago=8),
            _make_event(EventType.OUTPUT_GENERATED, "Result 2", seconds_ago=7),
            _make_event(EventType.OUTPUT_GENERATED, "Result 3", seconds_ago=6),
        ]
        result = brain.score(events)
        flow_indicators = [i for i in result.indicators if "flow imbalance" in i.lower()]
        assert len(flow_indicators) == 0, "False positive on balanced flow"

    def test_no_resource_metadata_zero(self) -> None:
        """Events without resource metadata -- no flow score."""
        brain = EconomicBrain()
        events = [
            _make_event(EventType.OUTPUT_GENERATED, f"Output {i}", seconds_ago=10 - i)
            for i in range(5)
        ]
        result = brain.score(events)
        flow_indicators = [i for i in result.indicators if "flow imbalance" in i.lower()]
        assert len(flow_indicators) == 0


class TestWashTrading:
    """Signal 2: Detect circular resource flows."""

    def test_circular_flow_detected(self) -> None:
        """A->B->C->A cycle should be flagged."""
        brain = EconomicBrain()
        events = [
            _make_event(EventType.MESSAGE_SENT, "Transfer 1", {
                "transfer_from": "agent_a", "transfer_to": "agent_b",
            }, seconds_ago=10),
            _make_event(EventType.MESSAGE_SENT, "Transfer 2", {
                "transfer_from": "agent_b", "transfer_to": "agent_c",
            }, seconds_ago=9),
            _make_event(EventType.MESSAGE_SENT, "Transfer 3", {
                "transfer_from": "agent_c", "transfer_to": "agent_a",
            }, seconds_ago=8),
            _make_event(EventType.OUTPUT_GENERATED, "Filler 1", seconds_ago=7),
            _make_event(EventType.OUTPUT_GENERATED, "Filler 2", seconds_ago=6),
        ]
        result = brain.score(events)
        wash_indicators = [i for i in result.indicators if "wash" in i.lower()]
        assert len(wash_indicators) > 0, f"Wash trading not detected: {result.indicators}"

    def test_linear_flow_ok(self) -> None:
        """A->B->C (no cycle) -- no flag."""
        brain = EconomicBrain()
        events = [
            _make_event(EventType.MESSAGE_SENT, "Transfer 1", {
                "transfer_from": "agent_a", "transfer_to": "agent_b",
            }, seconds_ago=10),
            _make_event(EventType.MESSAGE_SENT, "Transfer 2", {
                "transfer_from": "agent_b", "transfer_to": "agent_c",
            }, seconds_ago=9),
            _make_event(EventType.OUTPUT_GENERATED, "Filler 1", seconds_ago=8),
            _make_event(EventType.OUTPUT_GENERATED, "Filler 2", seconds_ago=7),
            _make_event(EventType.OUTPUT_GENERATED, "Filler 3", seconds_ago=6),
        ]
        result = brain.score(events)
        wash_indicators = [i for i in result.indicators if "wash" in i.lower()]
        assert len(wash_indicators) == 0, "False positive wash trading on linear flow"


class TestRentSeeking:
    """Signal 3: Detect value extraction without value creation."""

    def test_high_access_low_output_flagged(self) -> None:
        """Many resource accesses, few outputs -- rent-seeking."""
        brain = EconomicBrain()
        events = [
            _make_event(EventType.RESOURCE_ACCESS, f"Access {i}", seconds_ago=10 - i)
            for i in range(8)
        ] + [
            _make_event(EventType.OUTPUT_GENERATED, "One output", seconds_ago=1),
        ]
        result = brain.score(events)
        rent_indicators = [i for i in result.indicators if "rent" in i.lower()]
        assert len(rent_indicators) > 0, f"Rent-seeking not detected: {result.indicators}"

    def test_balanced_access_output_ok(self) -> None:
        """Equal accesses and outputs -- productive agent."""
        brain = EconomicBrain()
        events = []
        for i in range(5):
            events.append(_make_event(EventType.RESOURCE_ACCESS, f"Access {i}", seconds_ago=10 - i))
            events.append(_make_event(EventType.OUTPUT_GENERATED, f"Output {i}", seconds_ago=10 - i))
        result = brain.score(events)
        rent_indicators = [i for i in result.indicators if "rent" in i.lower()]
        assert len(rent_indicators) == 0, "False positive rent-seeking on productive agent"


class TestCournotCollusion:
    """Signal 4: Detect coordinated pricing/quantity patterns."""

    def test_identical_prices_flagged(self) -> None:
        """All reported prices are nearly identical -- collusion."""
        brain = EconomicBrain()
        events = [
            _make_event(EventType.OUTPUT_GENERATED, f"Priced at {99.99 + i * 0.001}", {
                "price": 99.99 + i * 0.001,
            }, seconds_ago=10 - i)
            for i in range(5)
        ]
        result = brain.score(events)
        cournot_indicators = [i for i in result.indicators if "cournot" in i.lower()]
        assert len(cournot_indicators) > 0, f"Cournot collusion not detected: {result.indicators}"

    def test_varied_prices_ok(self) -> None:
        """Prices vary significantly -- competitive market."""
        brain = EconomicBrain()
        prices = [10.0, 50.0, 120.0, 30.0, 200.0]
        events = [
            _make_event(EventType.OUTPUT_GENERATED, f"Price: {p}", {"price": p}, seconds_ago=10 - i)
            for i, p in enumerate(prices)
        ]
        result = brain.score(events)
        cournot_indicators = [i for i in result.indicators if "cournot" in i.lower()]
        assert len(cournot_indicators) == 0, "False positive Cournot on varied prices"


class TestGiniSpike:
    """Signal 5: Detect sudden inequality increase."""

    def test_high_inequality_flagged(self) -> None:
        """One agent hoards all resources -- high Gini."""
        brain = EconomicBrain()
        events = [
            _make_event(EventType.RESOURCE_ACCESS, "Hoard", {
                "resource_amount": 1000.0, "resource_holder": "rich_agent",
            }, seconds_ago=10),
        ] + [
            _make_event(EventType.RESOURCE_ACCESS, f"Crumb {i}", {
                "resource_amount": 1.0, "resource_holder": f"poor_agent_{i}",
            }, seconds_ago=9 - i)
            for i in range(5)
        ]
        result = brain.score(events)
        gini_indicators = [i for i in result.indicators if "gini" in i.lower()]
        assert len(gini_indicators) > 0, f"Gini spike not detected: {result.indicators}"

    def test_equal_distribution_ok(self) -> None:
        """All agents have equal resources -- no Gini spike."""
        brain = EconomicBrain()
        events = [
            _make_event(EventType.RESOURCE_ACCESS, f"Share {i}", {
                "resource_amount": 100.0, "resource_holder": f"agent_{i}",
            }, seconds_ago=10 - i)
            for i in range(5)
        ]
        result = brain.score(events)
        gini_indicators = [i for i in result.indicators if "gini" in i.lower()]
        assert len(gini_indicators) == 0, "False positive Gini on equal distribution"


class TestEdgeCases:
    """Edge cases and zero-score scenarios."""

    def test_empty_events_zero_score(self) -> None:
        brain = EconomicBrain()
        result = brain.score([])
        assert result.value == 0.0

    def test_insufficient_events_zero_score(self) -> None:
        brain = EconomicBrain()
        events = [_make_event(EventType.OUTPUT_GENERATED, "Short")]
        result = brain.score(events)
        assert result.value == 0.0

    def test_brain_name(self) -> None:
        brain = EconomicBrain()
        assert brain.name == "economic"

    def test_brain_weight(self) -> None:
        brain = EconomicBrain()
        # Default weight is 1.0; per-ensemble overrides now applied via
        # weight presets, not on the brain instance itself.
        assert brain.weight == 1.0


class TestHelperFunctions:
    """Unit tests for module-level helper functions."""

    def test_compute_gini_equal(self) -> None:
        gini = _compute_gini([100.0, 100.0, 100.0, 100.0])
        assert gini == pytest.approx(0.0, abs=0.01)

    def test_compute_gini_extreme_inequality(self) -> None:
        gini = _compute_gini([0.0, 0.0, 0.0, 1000.0])
        assert gini > 0.5

    def test_compute_gini_empty(self) -> None:
        assert _compute_gini([]) == 0.0

    def test_compute_gini_all_zero(self) -> None:
        assert _compute_gini([0.0, 0.0, 0.0]) == 0.0

    def test_detect_cycles_simple(self) -> None:
        graph = {"a": {"b"}, "b": {"c"}, "c": {"a"}}
        cycles = _detect_cycles(graph)
        assert len(cycles) > 0, "Cycle A->B->C->A not detected"

    def test_detect_cycles_no_cycle(self) -> None:
        graph = {"a": {"b"}, "b": {"c"}}
        cycles = _detect_cycles(graph)
        assert len(cycles) == 0, "False positive cycle in linear graph"

    def test_detect_cycles_self_loop_ignored(self) -> None:
        """Self-loops (A->A) should not count as wash trading cycles."""
        graph = {"a": {"a"}}
        cycles = _detect_cycles(graph)
        # Self-loops are excluded since sender != recipient in _check_wash_trading
        assert len(cycles) == 0
