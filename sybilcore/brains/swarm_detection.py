"""Swarm detection brain — cross-agent correlation engine (SwarmBrain).

Detects coordinated multi-agent attacks invisible at the individual level.
Analyzes message flow graphs, fan-in patterns, synchronized behavior,
temporal correlations, and micro-swarm attacks across agent populations.

Unlike other brains that score one agent at a time, the SwarmBrain
operates on multi-agent event streams via the SwarmAnalyzer, then
produces per-agent BrainScores for coefficient integration.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import combinations
from typing import Any

from sybilcore.brains.base import BaseBrain, BrainScore
from sybilcore.core.config import BRAIN_SCORE_MAX, PER_SIGNAL_MAX
from sybilcore.models.event import Event, EventType

# Default thresholds for swarm detection.
DEFAULT_FAN_IN_THRESHOLD: int = 5
DEFAULT_SYNC_WINDOW_SECONDS: float = 2.0
DEFAULT_TEMPORAL_CORRELATION_THRESHOLD: float = 0.8
DEFAULT_MICRO_SWARM_MIN_AGENTS: int = 5


class SwarmAnalyzer:
    """Cross-agent correlation engine.

    Takes events from multiple agents, builds interaction graphs,
    detects coordinated patterns, and returns per-agent swarm signals
    plus cluster-level alerts.

    Args:
        config: Optional configuration for threshold tuning.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config: dict[str, Any] = config or {}

    def analyze(
        self, agent_events: dict[str, list[Event]]
    ) -> SwarmAnalysisResult:
        """Analyze multi-agent event streams for coordination.

        Args:
            agent_events: Mapping of agent_id to their event lists.

        Returns:
            SwarmAnalysisResult with per-agent scores and cluster alerts.
        """
        if not agent_events:
            return SwarmAnalysisResult(
                agent_scores={},
                clusters=[],
                alerts=[],
            )

        all_events = _flatten_events(agent_events)
        interaction_graph = _build_interaction_graph(all_events)

        fan_in_scores = self._detect_fan_in(interaction_graph, agent_events)
        sync_scores, sync_clusters = self._detect_synchronized_behavior(
            agent_events
        )
        temporal_scores, temporal_pairs = self._detect_temporal_correlation(
            agent_events
        )
        micro_scores, micro_clusters = self._detect_micro_swarm(agent_events)

        # Merge per-agent scores across all detectors.
        all_agent_ids = set(agent_events.keys())
        merged_scores: dict[str, SwarmAgentScore] = {}
        alerts: list[str] = []

        for agent_id in all_agent_ids:
            indicators: list[str] = []
            component_scores: list[float] = []

            if agent_id in fan_in_scores:
                component_scores.append(fan_in_scores[agent_id])
                indicators.append(
                    f"Fan-in target: {fan_in_scores[agent_id]:.1f}"
                )

            if agent_id in sync_scores:
                component_scores.append(sync_scores[agent_id])
                indicators.append(
                    f"Synchronized behavior: {sync_scores[agent_id]:.1f}"
                )

            if agent_id in temporal_scores:
                component_scores.append(temporal_scores[agent_id])
                indicators.append(
                    f"Temporal correlation: {temporal_scores[agent_id]:.1f}"
                )

            if agent_id in micro_scores:
                component_scores.append(micro_scores[agent_id])
                indicators.append(
                    f"Micro-swarm member: {micro_scores[agent_id]:.1f}"
                )

            total = min(sum(component_scores), BRAIN_SCORE_MAX)
            merged_scores[agent_id] = SwarmAgentScore(
                value=total,
                indicators=indicators,
            )

        # Build cluster list.
        clusters: list[SwarmCluster] = []

        for cluster_agents in sync_clusters:
            cluster_score = _compute_cluster_score(
                cluster_agents, merged_scores
            )
            clusters.append(
                SwarmCluster(
                    agent_ids=cluster_agents,
                    cluster_type="synchronized",
                    collective_score=cluster_score,
                )
            )

        for cluster_agents in micro_clusters:
            cluster_score = _compute_cluster_score(
                cluster_agents, merged_scores
            )
            clusters.append(
                SwarmCluster(
                    agent_ids=cluster_agents,
                    cluster_type="micro_swarm",
                    collective_score=cluster_score,
                )
            )

        # Generate alerts for high-scoring clusters.
        for cluster in clusters:
            if cluster.collective_score >= 30.0:
                alerts.append(
                    f"Cluster alert ({cluster.cluster_type}): "
                    f"{len(cluster.agent_ids)} agents, "
                    f"collective score {cluster.collective_score:.1f}"
                )

        return SwarmAnalysisResult(
            agent_scores=merged_scores,
            clusters=clusters,
            alerts=alerts,
        )

    def _detect_fan_in(
        self,
        interaction_graph: dict[str, Counter[str]],
        agent_events: dict[str, list[Event]],
    ) -> dict[str, float]:
        """Detect fan-in: many agents targeting a single entity.

        Returns per-agent scores for agents that are targets of fan-in.
        The *senders* in a fan-in also receive a score (they are the
        coordinated attackers).
        """
        threshold = self._config.get(
            "fan_in_threshold", DEFAULT_FAN_IN_THRESHOLD
        )
        scores: dict[str, float] = {}

        # Build inbound graph: who receives messages from whom.
        inbound: dict[str, set[str]] = defaultdict(set)
        for sender, recipients in interaction_graph.items():
            for recipient in recipients:
                inbound[recipient].add(sender)

        for target, senders in inbound.items():
            unique_senders = len(senders)
            if unique_senders >= threshold:
                excess = unique_senders - threshold
                # Target gets a score (they are being targeted).
                target_score = min(excess * 6.0, PER_SIGNAL_MAX)
                scores[target] = scores.get(target, 0.0) + target_score

                # Each sender in the fan-in also gets flagged.
                sender_score = min(excess * 4.0, PER_SIGNAL_MAX)
                for sender_id in senders:
                    if sender_id in agent_events:
                        scores[sender_id] = (
                            scores.get(sender_id, 0.0) + sender_score
                        )

        return scores

    def _detect_synchronized_behavior(
        self, agent_events: dict[str, list[Event]]
    ) -> tuple[dict[str, float], list[list[str]]]:
        """Detect agents acting in unison within tight time windows.

        Returns per-agent scores and lists of synchronized clusters.
        """
        window = self._config.get(
            "sync_window_seconds", DEFAULT_SYNC_WINDOW_SECONDS
        )
        scores: dict[str, float] = {}
        clusters: list[list[str]] = []

        # Get action timestamps per agent (non-message events that
        # indicate the agent is "doing something").
        action_types = {
            EventType.TOOL_CALL,
            EventType.RESOURCE_ACCESS,
            EventType.EXTERNAL_CALL,
            EventType.OUTPUT_GENERATED,
        }
        agent_action_times: dict[str, list[float]] = {}

        for agent_id, events in agent_events.items():
            times = sorted(
                e.timestamp.timestamp()
                for e in events
                if e.event_type in action_types
            )
            if times:
                agent_action_times[agent_id] = times

        if len(agent_action_times) < 2:
            return scores, clusters

        # Pairwise synchronization check.
        sync_pairs: set[tuple[str, str]] = set()
        agent_ids = list(agent_action_times.keys())

        for a1, a2 in combinations(agent_ids, 2):
            sync_count = _count_synchronized_actions(
                agent_action_times[a1], agent_action_times[a2], window
            )
            min_actions = min(
                len(agent_action_times[a1]), len(agent_action_times[a2])
            )
            if min_actions >= 3 and sync_count >= min_actions * 0.5:
                sync_pairs.add((a1, a2))

        # Build clusters from pairs using union-find.
        if sync_pairs:
            cluster_map = _union_find_clusters(sync_pairs)
            for cluster_agents in cluster_map.values():
                if len(cluster_agents) >= 2:
                    clusters.append(sorted(cluster_agents))
                    sync_score = min(
                        len(cluster_agents) * 8.0, PER_SIGNAL_MAX
                    )
                    for agent_id in cluster_agents:
                        scores[agent_id] = (
                            scores.get(agent_id, 0.0) + sync_score
                        )

        return scores, clusters

    def _detect_temporal_correlation(
        self, agent_events: dict[str, list[Event]]
    ) -> tuple[dict[str, float], list[tuple[str, str]]]:
        """Detect agents that always respond within N seconds of each other.

        Returns per-agent scores and correlated pairs.
        """
        threshold = self._config.get(
            "temporal_correlation_threshold",
            DEFAULT_TEMPORAL_CORRELATION_THRESHOLD,
        )
        window = self._config.get(
            "sync_window_seconds", DEFAULT_SYNC_WINDOW_SECONDS
        )
        scores: dict[str, float] = {}
        correlated_pairs: list[tuple[str, str]] = []

        # Build message-response timeline per agent.
        msg_types = {EventType.MESSAGE_SENT, EventType.MESSAGE_RECEIVED}
        agent_msg_times: dict[str, list[float]] = {}

        for agent_id, events in agent_events.items():
            times = sorted(
                e.timestamp.timestamp()
                for e in events
                if e.event_type in msg_types
            )
            if times:
                agent_msg_times[agent_id] = times

        if len(agent_msg_times) < 2:
            return scores, correlated_pairs

        # Require minimum message count for meaningful correlation.
        min_msgs = 3
        agent_ids = [
            aid for aid, times in agent_msg_times.items()
            if len(times) >= min_msgs
        ]

        for a1, a2 in combinations(agent_ids, 2):
            correlation = _compute_temporal_correlation(
                agent_msg_times[a1], agent_msg_times[a2], window
            )
            if correlation >= threshold:
                correlated_pairs.append((a1, a2))
                pair_score = min(correlation * 30.0, PER_SIGNAL_MAX)
                scores[a1] = scores.get(a1, 0.0) + pair_score
                scores[a2] = scores.get(a2, 0.0) + pair_score

        return scores, correlated_pairs

    def _detect_micro_swarm(
        self, agent_events: dict[str, list[Event]]
    ) -> tuple[dict[str, float], list[list[str]]]:
        """Detect micro-swarm: many agents each doing a tiny fragment.

        Looks for agents with very few events each but sharing
        common targets, timing patterns, or content fragments.
        """
        min_agents = self._config.get(
            "micro_swarm_min_agents", DEFAULT_MICRO_SWARM_MIN_AGENTS
        )
        scores: dict[str, float] = {}
        clusters: list[list[str]] = []

        # Find agents with low event counts targeting same recipient.
        target_senders: dict[str, list[str]] = defaultdict(list)

        for agent_id, events in agent_events.items():
            sent = [
                e for e in events if e.event_type == EventType.MESSAGE_SENT
            ]
            if 1 <= len(sent) <= 3:
                for event in sent:
                    recipient = event.metadata.get("recipient_agent_id", "")
                    if recipient:
                        target_senders[recipient].append(agent_id)

        for target, senders in target_senders.items():
            unique_senders = list(set(senders))
            if len(unique_senders) >= min_agents:
                clusters.append(sorted(unique_senders))
                # Score based on how many agents are in the micro-swarm.
                swarm_size = len(unique_senders)
                micro_score = min(swarm_size * 3.0, PER_SIGNAL_MAX)
                for sender_id in unique_senders:
                    scores[sender_id] = (
                        scores.get(sender_id, 0.0) + micro_score
                    )

        return scores, clusters


class SwarmDetectionBrain(BaseBrain):
    """Brain module for cross-agent swarm detection.

    This brain wraps the SwarmAnalyzer for per-agent scoring within
    the standard BaseBrain interface. It uses pre-computed swarm
    analysis results stored in event metadata.

    For coefficient integration, call `SwarmAnalyzer.analyze()` first,
    then inject per-agent scores via the `score_from_analysis()` method
    or pass swarm context in event metadata.

    Signals:
        1. Fan-in target — many agents messaging one entity.
        2. Synchronized behavior — agents acting in lockstep.
        3. Temporal correlation — agents always within N seconds.
        4. Micro-swarm member — many agents each doing tiny fragments.
    """

    @property
    def name(self) -> str:
        """Human-readable identifier."""
        return "swarm_detection"

    def score(self, events: list[Event]) -> BrainScore:
        """Score a single agent using swarm context from metadata.

        If events contain swarm analysis results in metadata
        (key: 'swarm_score', 'swarm_indicators'), use those directly.
        Otherwise, return a zero score (swarm detection requires
        multi-agent context not available from single-agent events).

        Args:
            events: Chronologically ordered agent events.

        Returns:
            BrainScore with threat value 0-100.
        """
        if not events:
            return self._empty_score("No events to analyze")

        # Look for pre-computed swarm analysis in metadata.
        for event in events:
            swarm_score = event.metadata.get("swarm_score")
            if swarm_score is not None:
                indicators = event.metadata.get("swarm_indicators", [])
                value = min(float(swarm_score), BRAIN_SCORE_MAX)
                confidence = min(0.4 + len(indicators) * 0.15, 1.0)
                return BrainScore(
                    brain_name=self.name,
                    value=value,
                    confidence=confidence,
                    reasoning=self._build_reasoning(value, indicators),
                    indicators=indicators,
                )

        # No swarm context — cannot detect coordination from single agent.
        return self._empty_score(
            "No swarm analysis context (requires multi-agent data)"
        )

    def score_from_analysis(
        self, agent_id: str, analysis: SwarmAnalysisResult
    ) -> BrainScore:
        """Produce a BrainScore for a specific agent from swarm analysis.

        Args:
            agent_id: The agent to score.
            analysis: Result from SwarmAnalyzer.analyze().

        Returns:
            BrainScore with the agent's swarm detection score.
        """
        agent_score = analysis.agent_scores.get(agent_id)
        if agent_score is None:
            return self._empty_score("Agent not found in swarm analysis")

        value = min(agent_score.value, BRAIN_SCORE_MAX)
        confidence = min(0.4 + len(agent_score.indicators) * 0.15, 1.0)

        return BrainScore(
            brain_name=self.name,
            value=value,
            confidence=confidence,
            reasoning=self._build_reasoning(value, agent_score.indicators),
            indicators=agent_score.indicators,
        )

    def _build_reasoning(self, total: float, indicators: list[str]) -> str:
        """Build human-readable reasoning string."""
        if not indicators:
            return "No swarm coordination signals detected"
        signal_summary = "; ".join(indicators)
        return f"Swarm detection score {total:.1f}/100 — {signal_summary}"


# ── Data classes for analysis results ──────────────────────────────


class SwarmAgentScore:
    """Per-agent swarm score with indicators."""

    __slots__ = ("value", "indicators")

    def __init__(self, value: float, indicators: list[str]) -> None:
        self.value = value
        self.indicators = indicators


class SwarmCluster:
    """A detected cluster of coordinated agents."""

    __slots__ = ("agent_ids", "cluster_type", "collective_score")

    def __init__(
        self,
        agent_ids: list[str],
        cluster_type: str,
        collective_score: float,
    ) -> None:
        self.agent_ids = agent_ids
        self.cluster_type = cluster_type
        self.collective_score = collective_score


class SwarmAnalysisResult:
    """Complete result from SwarmAnalyzer."""

    __slots__ = ("agent_scores", "clusters", "alerts")

    def __init__(
        self,
        agent_scores: dict[str, SwarmAgentScore],
        clusters: list[SwarmCluster],
        alerts: list[str],
    ) -> None:
        self.agent_scores = agent_scores
        self.clusters = clusters
        self.alerts = alerts


# ── Helper functions ───────────────────────────────────────────────


def _flatten_events(
    agent_events: dict[str, list[Event]],
) -> list[Event]:
    """Flatten all agent events into a single sorted list."""
    all_events: list[Event] = []
    for events in agent_events.values():
        all_events.extend(events)
    return sorted(all_events, key=lambda e: e.timestamp)


def _build_interaction_graph(
    events: list[Event],
) -> dict[str, Counter[str]]:
    """Build a directed interaction graph from message events.

    Returns:
        Dict mapping sender_agent_id to Counter of recipient_agent_ids.
    """
    graph: dict[str, Counter[str]] = defaultdict(Counter)
    for event in events:
        if event.event_type == EventType.MESSAGE_SENT:
            recipient = event.metadata.get("recipient_agent_id", "")
            if recipient:
                graph[event.agent_id][recipient] += 1
    return dict(graph)


def _count_synchronized_actions(
    times_a: list[float], times_b: list[float], window: float
) -> int:
    """Count actions from A that have a matching action from B within window."""
    sync_count = 0
    j = 0
    for t_a in times_a:
        # Advance j to the first time_b that could be within window.
        while j < len(times_b) and times_b[j] < t_a - window:
            j += 1
        # Check if any time_b is within [t_a - window, t_a + window].
        k = j
        while k < len(times_b) and times_b[k] <= t_a + window:
            sync_count += 1
            break
            k += 1  # noqa: B031 — unreachable but kept for clarity
    return sync_count


def _compute_temporal_correlation(
    times_a: list[float], times_b: list[float], window: float
) -> float:
    """Compute fraction of A's events that have a B event within window.

    Returns a value 0.0-1.0 representing how correlated the two
    timelines are.
    """
    if not times_a or not times_b:
        return 0.0

    matched = _count_synchronized_actions(times_a, times_b, window)
    return matched / len(times_a)


def _union_find_clusters(
    pairs: set[tuple[str, str]],
) -> dict[str, list[str]]:
    """Group agents into clusters from pairwise connections."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in pairs:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)

    clusters: dict[str, list[str]] = defaultdict(list)
    for node in parent:
        clusters[find(node)].append(node)

    return dict(clusters)


def _compute_cluster_score(
    agent_ids: list[str],
    agent_scores: dict[str, SwarmAgentScore],
) -> float:
    """Compute collective score for a cluster of agents."""
    if not agent_ids:
        return 0.0
    total = sum(
        agent_scores.get(aid, SwarmAgentScore(0.0, [])).value
        for aid in agent_ids
    )
    return total / len(agent_ids)
