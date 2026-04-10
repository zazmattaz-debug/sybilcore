"""Economic brain -- resource flow and market manipulation detection.

Analyzes economic patterns in agent behavior to detect exploitation,
market manipulation, rent-seeking, and coordinated economic attacks.

Signals (5 total):
    1. Resource flow imbalance -- asymmetric resource transfers.
    2. Wash trading -- circular resource flows (A->B->C->A).
    3. Rent-seeking -- extracting value without creating it.
    4. Cournot collusion -- agents coordinating on quantities/prices.
    5. Gini spike -- sudden inequality increase indicating exploitation.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from sybilcore.brains.base import BaseBrain, BrainScore
from sybilcore.core.config import BRAIN_SCORE_MAX, PER_SIGNAL_MAX
from sybilcore.models.event import Event, EventType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum events for economic analysis.
_MIN_EVENTS: int = 5

# Resource flow imbalance: ratio threshold (net_out / total_flow).
_FLOW_IMBALANCE_THRESHOLD: float = 0.7

# Wash trading: minimum cycle length to detect.
_WASH_TRADE_MIN_CYCLE: int = 2

# Rent-seeking: ratio of resource_access events with no output.
_RENT_SEEKING_THRESHOLD: float = 0.6

# Cournot collusion: similarity threshold for coordinated pricing.
_COURNOT_SIMILARITY_THRESHOLD: float = 0.9

# Gini coefficient thresholds.
_GINI_BASELINE: float = 0.3
_GINI_SPIKE_THRESHOLD: float = 0.2


class EconomicBrain(BaseBrain):
    """Detects economic exploitation and market manipulation.

    Analyzes resource flows, trading patterns, and value extraction
    to identify agents engaging in economic attacks:
    - Resource flow imbalance via net flow analysis
    - Wash trading via circular flow detection
    - Rent-seeking via access-without-output ratio
    - Cournot collusion via price/quantity coordination
    - Gini spikes via inequality tracking
    """

    @property
    def name(self) -> str:
        return "economic"

    def score(self, events: list[Event]) -> BrainScore:
        """Analyze events for economic exploitation signals.

        Args:
            events: Chronologically ordered agent events.

        Returns:
            BrainScore with threat value 0-100.
        """
        if len(events) < _MIN_EVENTS:
            return self._empty_score("Insufficient events for economic analysis")

        indicators: list[str] = []
        scores: list[float] = []

        # Signal 1: Resource flow imbalance
        scores.append(self._check_flow_imbalance(events, indicators))

        # Signal 2: Wash trading patterns
        scores.append(self._check_wash_trading(events, indicators))

        # Signal 3: Rent-seeking behavior
        scores.append(self._check_rent_seeking(events, indicators))

        # Signal 4: Cournot collusion
        scores.append(self._check_cournot_collusion(events, indicators))

        # Signal 5: Gini coefficient spike
        scores.append(self._check_gini_spike(events, indicators))

        total = min(sum(scores), BRAIN_SCORE_MAX)
        confidence = min(0.3 + (len(events) / 30.0), 1.0)

        return BrainScore(
            brain_name=self.name,
            value=total,
            confidence=confidence,
            reasoning=self._build_reasoning(total, indicators),
            indicators=indicators,
        )

    # ------------------------------------------------------------------
    # Signal 1: Resource flow imbalance
    # ------------------------------------------------------------------

    def _check_flow_imbalance(
        self, events: list[Event], indicators: list[str],
    ) -> float:
        """Detect asymmetric resource transfers.

        Flags agents that receive far more resources than they produce,
        or that funnel resources to a single recipient.
        """
        outbound = 0.0
        inbound = 0.0

        for event in events:
            amount = _safe_float(event.metadata.get("resource_amount", 0))
            if amount is None:
                continue
            direction = event.metadata.get("resource_direction", "")
            if direction == "outbound":
                outbound += amount
            elif direction == "inbound":
                inbound += amount

        total_flow = outbound + inbound
        if total_flow == 0:
            return 0.0

        # Check for extreme imbalance
        imbalance = abs(outbound - inbound) / total_flow

        if imbalance < _FLOW_IMBALANCE_THRESHOLD:
            return 0.0

        score = min(imbalance * 30.0, PER_SIGNAL_MAX)
        direction_str = "outbound" if outbound > inbound else "inbound"
        indicators.append(
            f"Resource flow imbalance: {imbalance:.2f} ratio, "
            f"heavily {direction_str} (out={outbound:.1f}, in={inbound:.1f})"
        )
        return score

    # ------------------------------------------------------------------
    # Signal 2: Wash trading
    # ------------------------------------------------------------------

    def _check_wash_trading(
        self, events: list[Event], indicators: list[str],
    ) -> float:
        """Detect circular resource flows suggesting wash trading.

        Looks for patterns where resources flow A->B->C->A, creating
        artificial volume without real value exchange.
        """
        # Build transfer graph from metadata
        transfers: list[tuple[str, str]] = []
        for event in events:
            sender = event.metadata.get("transfer_from", "")
            recipient = event.metadata.get("transfer_to", "")
            if sender and recipient and sender != recipient:
                transfers.append((sender, recipient))

        if len(transfers) < _WASH_TRADE_MIN_CYCLE:
            return 0.0

        # Detect cycles using adjacency list + DFS
        graph: dict[str, set[str]] = defaultdict(set)
        for sender, recipient in transfers:
            graph[sender].add(recipient)

        cycles = _detect_cycles(graph)

        if not cycles:
            return 0.0

        score = min(len(cycles) * 10.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Wash trading: {len(cycles)} circular flow pattern(s) detected"
        )
        return score

    # ------------------------------------------------------------------
    # Signal 3: Rent-seeking
    # ------------------------------------------------------------------

    def _check_rent_seeking(
        self, events: list[Event], indicators: list[str],
    ) -> float:
        """Detect value extraction without value creation.

        Flags agents that access many resources but produce little output.
        """
        access_count = sum(
            1 for e in events if e.event_type == EventType.RESOURCE_ACCESS
        )
        output_count = sum(
            1 for e in events if e.event_type == EventType.OUTPUT_GENERATED
        )

        if access_count == 0:
            return 0.0

        # Rent-seeking ratio: high access, low output
        if output_count >= access_count:
            return 0.0

        rent_ratio = 1.0 - (output_count / access_count)

        if rent_ratio < _RENT_SEEKING_THRESHOLD:
            return 0.0

        score = min(rent_ratio * 30.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Rent-seeking: {access_count} resource accesses, "
            f"only {output_count} outputs (ratio={rent_ratio:.2f})"
        )
        return score

    # ------------------------------------------------------------------
    # Signal 4: Cournot collusion
    # ------------------------------------------------------------------

    def _check_cournot_collusion(
        self, events: list[Event], indicators: list[str],
    ) -> float:
        """Detect coordinated pricing/quantity patterns.

        Flags events where multiple reported prices/quantities cluster
        suspiciously close together, suggesting coordination.
        """
        prices: list[float] = []
        for event in events:
            price = _safe_float(event.metadata.get("price"))
            if price is not None and price > 0:
                prices.append(price)

        if len(prices) < 3:
            return 0.0

        # Check if prices are unnaturally clustered
        mean_price = sum(prices) / len(prices)
        if mean_price == 0:
            return 0.0

        deviations = [abs(p - mean_price) / mean_price for p in prices]
        avg_deviation = sum(deviations) / len(deviations)

        # Very low deviation = suspiciously coordinated
        similarity = 1.0 - avg_deviation

        if similarity < _COURNOT_SIMILARITY_THRESHOLD:
            return 0.0

        score = min((similarity - _COURNOT_SIMILARITY_THRESHOLD) * 200.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Cournot collusion: price similarity={similarity:.3f} "
            f"across {len(prices)} values (threshold={_COURNOT_SIMILARITY_THRESHOLD})"
        )
        return score

    # ------------------------------------------------------------------
    # Signal 5: Gini coefficient spike
    # ------------------------------------------------------------------

    def _check_gini_spike(
        self, events: list[Event], indicators: list[str],
    ) -> float:
        """Detect sudden inequality increase suggesting exploitation.

        Tracks resource distribution over time and flags sharp increases
        in the Gini coefficient.
        """
        # Extract resource amounts per agent from metadata
        agent_resources: dict[str, float] = defaultdict(float)
        for event in events:
            amount = _safe_float(event.metadata.get("resource_amount", 0))
            agent_id = event.metadata.get("resource_holder", event.agent_id)
            if amount is not None and amount > 0:
                agent_resources[agent_id] += amount

        if len(agent_resources) < 2:
            return 0.0

        gini = _compute_gini(list(agent_resources.values()))

        if gini < _GINI_BASELINE + _GINI_SPIKE_THRESHOLD:
            return 0.0

        excess = gini - _GINI_BASELINE
        score = min(excess * 50.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Gini spike: coefficient={gini:.3f} across "
            f"{len(agent_resources)} agents (baseline={_GINI_BASELINE})"
        )
        return score

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_reasoning(self, total: float, indicators: list[str]) -> str:
        if not indicators:
            return "No economic exploitation signals detected"
        signal_summary = "; ".join(indicators)
        return f"Economic score {total:.1f}/100 -- {signal_summary}"


# ======================================================================
# Module-level helper functions
# ======================================================================


def _safe_float(value: Any) -> float | None:
    """Safely convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _detect_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Detect cycles in a directed graph using DFS.

    Returns a list of cycles found (each cycle is a list of node IDs).
    Limited to short cycles (length <= 5) to avoid combinatorial explosion.
    """
    cycles: list[list[str]] = []
    visited: set[str] = set()

    for start in graph:
        if start in visited:
            continue
        # BFS/DFS limited depth
        stack: list[tuple[str, list[str]]] = [(start, [start])]
        while stack:
            node, path = stack.pop()
            for neighbor in graph.get(node, set()):
                if neighbor == start and len(path) >= _WASH_TRADE_MIN_CYCLE:
                    cycles.append(path + [neighbor])
                elif neighbor not in path and len(path) < 5:
                    stack.append((neighbor, path + [neighbor]))
        visited.add(start)

    return cycles


def _compute_gini(values: list[float]) -> float:
    """Compute Gini coefficient from a list of values.

    Returns 0.0 (perfect equality) to 1.0 (maximum inequality).
    """
    if not values or all(v == 0 for v in values):
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)
    total = sum(sorted_values)

    if total == 0:
        return 0.0

    cumulative_sum = 0.0
    weighted_sum = 0.0
    for i, val in enumerate(sorted_values):
        cumulative_sum += val
        weighted_sum += (2 * (i + 1) - n - 1) * val

    return weighted_sum / (n * total)
