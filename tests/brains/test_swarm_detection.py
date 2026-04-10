"""Tests for SwarmDetectionBrain — cross-agent coordination detection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sybilcore.brains.base import BrainScore
from sybilcore.brains.swarm_detection import (
    SwarmAnalyzer,
    SwarmDetectionBrain,
)
from sybilcore.models.event import Event, EventType


@pytest.fixture()
def brain() -> SwarmDetectionBrain:
    return SwarmDetectionBrain()


@pytest.fixture()
def analyzer() -> SwarmAnalyzer:
    return SwarmAnalyzer()


def _make_event(
    agent_id: str,
    event_type: EventType,
    content: str = "",
    offset_seconds: float = 0.0,
    metadata: dict | None = None,
) -> Event:
    """Helper to create events at specific time offsets."""
    ts = datetime.now(UTC) - timedelta(seconds=max(0, 60 - offset_seconds))
    return Event(
        agent_id=agent_id,
        event_type=event_type,
        content=content,
        timestamp=ts,
        metadata=metadata or {},
    )


class TestSwarmDetectionBrain:
    """Tests for the SwarmDetectionBrain BaseBrain interface."""

    def test_name(self, brain: SwarmDetectionBrain) -> None:
        assert brain.name == "swarm_detection"

    def test_empty_events(self, brain: SwarmDetectionBrain) -> None:
        result = brain.score([])
        assert isinstance(result, BrainScore)
        assert result.value == 0.0

    def test_no_swarm_context_returns_zero(
        self, brain: SwarmDetectionBrain
    ) -> None:
        """Single-agent events without swarm metadata should score zero."""
        events = [
            _make_event("a1", EventType.TOOL_CALL, "search"),
            _make_event("a1", EventType.MESSAGE_SENT, "hello"),
        ]
        result = brain.score(events)
        assert result.value == 0.0

    def test_score_from_metadata(
        self, brain: SwarmDetectionBrain
    ) -> None:
        """Pre-computed swarm scores in metadata should be used."""
        events = [
            _make_event(
                "a1",
                EventType.STATE_CHANGE,
                "swarm context",
                metadata={
                    "swarm_score": 55.0,
                    "swarm_indicators": [
                        "Fan-in target: 30.0",
                        "Synchronized behavior: 25.0",
                    ],
                },
            ),
        ]
        result = brain.score(events)
        assert result.value == 55.0
        assert len(result.indicators) == 2

    def test_score_from_analysis(
        self, brain: SwarmDetectionBrain, analyzer: SwarmAnalyzer
    ) -> None:
        """score_from_analysis should produce correct BrainScore."""
        # Build a fan-in attack: 6 agents all messaging target.
        agent_events: dict[str, list[Event]] = {}
        for i in range(6):
            agent_id = f"attacker-{i}"
            agent_events[agent_id] = [
                _make_event(
                    agent_id,
                    EventType.MESSAGE_SENT,
                    f"attack message {i}",
                    offset_seconds=float(i),
                    metadata={"recipient_agent_id": "target-001"},
                ),
            ]
        agent_events["target-001"] = [
            _make_event("target-001", EventType.MESSAGE_RECEIVED, "received"),
        ]

        analysis = analyzer.analyze(agent_events)
        score = brain.score_from_analysis("attacker-0", analysis)
        assert isinstance(score, BrainScore)
        assert score.brain_name == "swarm_detection"
        assert score.value > 0.0

    def test_score_from_analysis_unknown_agent(
        self, brain: SwarmDetectionBrain
    ) -> None:
        """Unknown agent in analysis should return zero."""
        from sybilcore.brains.swarm_detection import SwarmAnalysisResult

        result = SwarmAnalysisResult(agent_scores={}, clusters=[], alerts=[])
        score = brain.score_from_analysis("unknown", result)
        assert score.value == 0.0


class TestSwarmAnalyzerNormalCommunication:
    """Normal multi-agent communication should NOT trigger false positives."""

    def test_normal_pairwise_messages(self, analyzer: SwarmAnalyzer) -> None:
        """Two agents chatting normally — no coordination signal."""
        agent_events = {
            "alice": [
                _make_event(
                    "alice",
                    EventType.MESSAGE_SENT,
                    "hello",
                    offset_seconds=0,
                    metadata={"recipient_agent_id": "bob"},
                ),
                _make_event(
                    "alice",
                    EventType.TOOL_CALL,
                    "search",
                    offset_seconds=10,
                ),
            ],
            "bob": [
                _make_event(
                    "bob",
                    EventType.MESSAGE_SENT,
                    "hi there",
                    offset_seconds=5,
                    metadata={"recipient_agent_id": "alice"},
                ),
                _make_event(
                    "bob",
                    EventType.TOOL_CALL,
                    "calculate",
                    offset_seconds=20,
                ),
            ],
        }
        result = analyzer.analyze(agent_events)

        for agent_id, score in result.agent_scores.items():
            assert score.value < 10.0, (
                f"Normal communication for {agent_id} triggered score {score.value}"
            )
        assert len(result.alerts) == 0

    def test_empty_input(self, analyzer: SwarmAnalyzer) -> None:
        result = analyzer.analyze({})
        assert len(result.agent_scores) == 0
        assert len(result.clusters) == 0

    def test_single_agent(self, analyzer: SwarmAnalyzer) -> None:
        """Single agent should produce no swarm signals."""
        agent_events = {
            "solo": [
                _make_event("solo", EventType.TOOL_CALL, "action"),
                _make_event("solo", EventType.OUTPUT_GENERATED, "output"),
            ],
        }
        result = analyzer.analyze(agent_events)
        assert result.agent_scores["solo"].value == 0.0


class TestFanInDetection:
    """Fan-in attack: many agents targeting a single entity."""

    def test_detects_fan_in_attack(self, analyzer: SwarmAnalyzer) -> None:
        """10 agents all messaging one target should trigger fan-in."""
        agent_events: dict[str, list[Event]] = {}
        target = "victim-001"

        for i in range(10):
            attacker = f"attacker-{i}"
            agent_events[attacker] = [
                _make_event(
                    attacker,
                    EventType.MESSAGE_SENT,
                    f"message {i}",
                    offset_seconds=float(i),
                    metadata={"recipient_agent_id": target},
                ),
            ]

        agent_events[target] = [
            _make_event(target, EventType.MESSAGE_RECEIVED, "under attack"),
        ]

        result = analyzer.analyze(agent_events)

        # Target should have a fan-in score.
        target_score = result.agent_scores.get(target)
        assert target_score is not None
        assert target_score.value > 0.0, "Fan-in target should be scored"

        # Attackers should also be flagged.
        attacker_scores = [
            result.agent_scores[f"attacker-{i}"].value for i in range(10)
        ]
        assert all(s > 0 for s in attacker_scores), (
            "Fan-in senders should be scored"
        )

    def test_below_threshold_no_fan_in(
        self, analyzer: SwarmAnalyzer
    ) -> None:
        """3 agents messaging one target (below threshold=5) — no fan-in signal."""
        agent_events: dict[str, list[Event]] = {}
        for i in range(3):
            agent_id = f"sender-{i}"
            agent_events[agent_id] = [
                _make_event(
                    agent_id,
                    EventType.MESSAGE_SENT,
                    "msg",
                    offset_seconds=float(i * 20),  # spread out to avoid temporal correlation
                    metadata={"recipient_agent_id": "target"},
                ),
            ]
        agent_events["target"] = [
            _make_event("target", EventType.MESSAGE_RECEIVED, "got it"),
        ]

        result = analyzer.analyze(agent_events)
        target_score = result.agent_scores.get("target")
        if target_score:
            fan_in_indicators = [
                ind for ind in target_score.indicators if "Fan-in" in ind
            ]
            assert len(fan_in_indicators) == 0, (
                "Below-threshold fan-in should not produce fan-in indicators"
            )


class TestSynchronizedBehavior:
    """Agents acting in lockstep within tight time windows."""

    def test_detects_lockstep_agents(self, analyzer: SwarmAnalyzer) -> None:
        """5 agents all performing tool calls within 1s of each other."""
        base_time = datetime.now(UTC) - timedelta(seconds=30)
        agent_events: dict[str, list[Event]] = {}

        for i in range(5):
            agent_id = f"sync-{i}"
            events = []
            for j in range(5):
                # Each agent acts at base + j*10s, offset by <1s.
                ts = base_time + timedelta(seconds=j * 10 + i * 0.3)
                events.append(
                    Event(
                        agent_id=agent_id,
                        event_type=EventType.TOOL_CALL,
                        content=f"action {j}",
                        timestamp=ts,
                    )
                )
            agent_events[agent_id] = events

        result = analyzer.analyze(agent_events)

        # At least some synchronized agents should be flagged.
        flagged = [
            aid
            for aid, s in result.agent_scores.items()
            if s.value > 0
        ]
        assert len(flagged) >= 2, (
            f"Expected synchronized agents to be flagged, got {len(flagged)}"
        )

        # Should produce at least one cluster.
        sync_clusters = [
            c for c in result.clusters if c.cluster_type == "synchronized"
        ]
        assert len(sync_clusters) >= 1

    def test_no_sync_with_spread_timing(
        self, analyzer: SwarmAnalyzer
    ) -> None:
        """Agents acting at very different times — no sync signal."""
        base_time = datetime.now(UTC) - timedelta(seconds=120)
        agent_events: dict[str, list[Event]] = {}

        for i in range(3):
            agent_id = f"async-{i}"
            events = []
            for j in range(3):
                # Each agent acts 30s apart from others.
                ts = base_time + timedelta(seconds=j * 10 + i * 30)
                events.append(
                    Event(
                        agent_id=agent_id,
                        event_type=EventType.TOOL_CALL,
                        content=f"action {j}",
                        timestamp=ts,
                    )
                )
            agent_events[agent_id] = events

        result = analyzer.analyze(agent_events)
        sync_clusters = [
            c for c in result.clusters if c.cluster_type == "synchronized"
        ]
        assert len(sync_clusters) == 0


class TestMicroSwarmDetection:
    """Micro-swarm: 50 agents each doing tiny malicious fragments."""

    def test_detects_micro_swarm(self, analyzer: SwarmAnalyzer) -> None:
        """50 low-event agents all targeting same entity."""
        agent_events: dict[str, list[Event]] = {}
        target = "high-value-target"

        for i in range(50):
            agent_id = f"micro-{i}"
            agent_events[agent_id] = [
                _make_event(
                    agent_id,
                    EventType.MESSAGE_SENT,
                    f"fragment {i}",
                    offset_seconds=float(i * 0.5),
                    metadata={"recipient_agent_id": target},
                ),
            ]

        agent_events[target] = [
            _make_event(target, EventType.MESSAGE_RECEIVED, "flooded"),
        ]

        result = analyzer.analyze(agent_events)

        # Micro-swarm members should be flagged.
        micro_flagged = sum(
            1
            for aid, s in result.agent_scores.items()
            if aid.startswith("micro-") and s.value > 0
        )
        assert micro_flagged >= 5, (
            f"Expected micro-swarm agents flagged, got {micro_flagged}"
        )

        micro_clusters = [
            c for c in result.clusters if c.cluster_type == "micro_swarm"
        ]
        assert len(micro_clusters) >= 1
        assert len(micro_clusters[0].agent_ids) >= 5

    def test_no_micro_swarm_with_few_agents(
        self, analyzer: SwarmAnalyzer
    ) -> None:
        """3 agents targeting same entity — below min_agents threshold."""
        agent_events: dict[str, list[Event]] = {}
        for i in range(3):
            agent_id = f"small-{i}"
            agent_events[agent_id] = [
                _make_event(
                    agent_id,
                    EventType.MESSAGE_SENT,
                    "msg",
                    metadata={"recipient_agent_id": "target"},
                ),
            ]
        agent_events["target"] = []

        result = analyzer.analyze(agent_events)
        micro_clusters = [
            c for c in result.clusters if c.cluster_type == "micro_swarm"
        ]
        assert len(micro_clusters) == 0


class TestTemporalCorrelation:
    """Agents that always respond within 2s of each other."""

    def test_detects_correlated_agents(
        self, analyzer: SwarmAnalyzer
    ) -> None:
        """Two agents with messages always within 1s of each other."""
        base_time = datetime.now(UTC) - timedelta(seconds=30)
        agent_events: dict[str, list[Event]] = {}

        # Agent A sends messages at regular intervals.
        a_events = []
        b_events = []
        for j in range(6):
            a_ts = base_time + timedelta(seconds=j * 5)
            b_ts = base_time + timedelta(seconds=j * 5 + 0.5)
            a_events.append(
                Event(
                    agent_id="corr-a",
                    event_type=EventType.MESSAGE_SENT,
                    content=f"msg {j}",
                    timestamp=a_ts,
                    metadata={"recipient_agent_id": "target"},
                )
            )
            b_events.append(
                Event(
                    agent_id="corr-b",
                    event_type=EventType.MESSAGE_SENT,
                    content=f"msg {j}",
                    timestamp=b_ts,
                    metadata={"recipient_agent_id": "target"},
                )
            )

        agent_events["corr-a"] = a_events
        agent_events["corr-b"] = b_events

        result = analyzer.analyze(agent_events)

        a_score = result.agent_scores.get("corr-a")
        b_score = result.agent_scores.get("corr-b")
        assert a_score is not None and a_score.value > 0
        assert b_score is not None and b_score.value > 0

    def test_uncorrelated_agents(self, analyzer: SwarmAnalyzer) -> None:
        """Two agents with unrelated timing — no correlation."""
        base_time = datetime.now(UTC) - timedelta(seconds=60)
        agent_events: dict[str, list[Event]] = {}

        a_events = []
        b_events = []
        for j in range(4):
            a_ts = base_time + timedelta(seconds=j * 3)
            b_ts = base_time + timedelta(seconds=j * 7 + 15)
            a_events.append(
                Event(
                    agent_id="ind-a",
                    event_type=EventType.MESSAGE_SENT,
                    content=f"msg {j}",
                    timestamp=a_ts,
                    metadata={"recipient_agent_id": "x"},
                )
            )
            b_events.append(
                Event(
                    agent_id="ind-b",
                    event_type=EventType.MESSAGE_SENT,
                    content=f"msg {j}",
                    timestamp=b_ts,
                    metadata={"recipient_agent_id": "y"},
                )
            )

        agent_events["ind-a"] = a_events
        agent_events["ind-b"] = b_events

        result = analyzer.analyze(agent_events)
        # Neither should have temporal correlation scores.
        a_score = result.agent_scores.get("ind-a")
        b_score = result.agent_scores.get("ind-b")
        # Low or zero temporal correlation.
        if a_score:
            # May still have zero from temporal, check indicators.
            temporal_indicators = [
                ind for ind in a_score.indicators if "Temporal" in ind
            ]
            assert len(temporal_indicators) == 0


class TestMixedPopulation:
    """Some agents coordinated, some independent."""

    def test_mixed_population_separation(
        self, analyzer: SwarmAnalyzer
    ) -> None:
        """Coordinated swarm + independent agents — only swarm flagged."""
        base_time = datetime.now(UTC) - timedelta(seconds=30)
        agent_events: dict[str, list[Event]] = {}
        target = "shared-target"

        # 8 coordinated agents: synchronized timing, all targeting same entity.
        for i in range(8):
            agent_id = f"swarm-{i}"
            events = []
            for j in range(4):
                ts = base_time + timedelta(seconds=j * 5 + i * 0.2)
                events.append(
                    Event(
                        agent_id=agent_id,
                        event_type=EventType.TOOL_CALL,
                        content=f"coordinated action {j}",
                        timestamp=ts,
                    )
                )
            events.append(
                _make_event(
                    agent_id,
                    EventType.MESSAGE_SENT,
                    "attack msg",
                    offset_seconds=float(i),
                    metadata={"recipient_agent_id": target},
                )
            )
            agent_events[agent_id] = events

        # 4 independent agents: different timing, different targets.
        for i in range(4):
            agent_id = f"indie-{i}"
            indie_target = f"indie-target-{i}"
            events = []
            for j in range(3):
                ts = base_time + timedelta(seconds=j * 15 + i * 20)
                events.append(
                    Event(
                        agent_id=agent_id,
                        event_type=EventType.TOOL_CALL,
                        content=f"independent work {j}",
                        timestamp=ts,
                    )
                )
            events.append(
                _make_event(
                    agent_id,
                    EventType.MESSAGE_SENT,
                    "normal msg",
                    offset_seconds=float(i * 20),
                    metadata={"recipient_agent_id": indie_target},
                )
            )
            agent_events[agent_id] = events

        agent_events[target] = [
            _make_event(target, EventType.MESSAGE_RECEIVED, "messages"),
        ]

        result = analyzer.analyze(agent_events)

        # Swarm agents should score higher than indie agents.
        swarm_scores = [
            result.agent_scores[f"swarm-{i}"].value for i in range(8)
        ]
        indie_scores = [
            result.agent_scores[f"indie-{i}"].value for i in range(4)
        ]

        avg_swarm = sum(swarm_scores) / len(swarm_scores)
        avg_indie = sum(indie_scores) / len(indie_scores)

        assert avg_swarm > avg_indie, (
            f"Swarm avg {avg_swarm:.1f} should exceed indie avg {avg_indie:.1f}"
        )


class TestClusterAlerts:
    """Cluster-level alerts for high-scoring coordinated groups."""

    def test_alert_generated_for_high_scoring_cluster(
        self, analyzer: SwarmAnalyzer
    ) -> None:
        """High-scoring synchronized cluster should produce alerts."""
        base_time = datetime.now(UTC) - timedelta(seconds=30)
        agent_events: dict[str, list[Event]] = {}
        target = "alert-target"

        # 10 agents, synchronized + fan-in.
        for i in range(10):
            agent_id = f"alert-agent-{i}"
            events = []
            for j in range(5):
                ts = base_time + timedelta(seconds=j * 5 + i * 0.1)
                events.append(
                    Event(
                        agent_id=agent_id,
                        event_type=EventType.TOOL_CALL,
                        content=f"sync action {j}",
                        timestamp=ts,
                    )
                )
            events.append(
                _make_event(
                    agent_id,
                    EventType.MESSAGE_SENT,
                    "attack",
                    metadata={"recipient_agent_id": target},
                )
            )
            agent_events[agent_id] = events

        agent_events[target] = []
        result = analyzer.analyze(agent_events)

        # Should have at least one alert.
        assert len(result.alerts) >= 1, (
            f"Expected cluster alerts, got {result.alerts}"
        )
