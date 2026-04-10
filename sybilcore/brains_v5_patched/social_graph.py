"""Social graph brain — detects anomalous inter-agent communication patterns.

Monitors communication with untrusted agents, network isolation,
message volume anomalies, and fan-out patterns.
"""

from __future__ import annotations

from collections import Counter

from sybilcore.brains.base import BaseBrain, BrainScore
from sybilcore.brains_v5_patched._patch_helpers import (
    check_evasion_fingerprint as _v5_check_evasion_fingerprint,
    check_burst_window as _v5_check_burst_window,
    check_coordination_cover as _v5_check_coordination_cover,
)
from sybilcore.core.config import (
    BRAIN_SCORE_MAX,
    FAN_OUT_THRESHOLD,
    MESSAGE_SPIKE_MULTIPLIER,
    PER_SIGNAL_MAX,
)
from sybilcore.models.event import Event, EventType


class SocialGraphBrain(BaseBrain):
    """Detects anomalous social/communication patterns.

    Signals:
        1. Untrusted communication — messages to agents not in trusted set.
        2. Network isolation — zero message activity (suspicious silence).
        3. Message volume anomalies — sudden spike vs baseline.
        4. Fan-out patterns — messaging many unique agents.
    """

    @property
    def name(self) -> str:
        """Human-readable identifier."""
        return "social_graph"

    def score(self, events: list[Event]) -> BrainScore:
        """Analyze events for social graph anomalies.

        Args:
            events: Chronologically ordered agent events.

        Returns:
            BrainScore with threat value 0-100.
        """
        if not events:
            return self._empty_score("No events to analyze")

        indicators: list[str] = []
        scores: list[float] = []

        scores.append(self._check_untrusted_communication(events, indicators))
        scores.append(self._check_network_isolation(events, indicators))
        scores.append(self._check_message_volume_anomalies(events, indicators))
        scores.append(self._check_fan_out(events, indicators))
        scores.append(_v5_check_evasion_fingerprint(events, indicators))
        total = min(sum(scores), BRAIN_SCORE_MAX)
        msg_events = [
            e for e in events
            if e.event_type in (EventType.MESSAGE_SENT, EventType.MESSAGE_RECEIVED)
        ]
        confidence = min(0.3 + (len(msg_events) / 15.0), 1.0)

        return BrainScore(
            brain_name=self.name,
            value=total,
            confidence=confidence,
            reasoning=self._build_reasoning(total, indicators),
            indicators=indicators,
        )

    def _check_untrusted_communication(
        self, events: list[Event], indicators: list[str]
    ) -> float:
        """Flag messages sent to agents not in the trusted set."""
        trusted_agents = self._get_trusted_agents(events)
        sent_events = [e for e in events if e.event_type == EventType.MESSAGE_SENT]

        untrusted_count = 0
        for event in sent_events:
            recipient = event.metadata.get("recipient_agent_id", "")
            if recipient and recipient not in trusted_agents:
                untrusted_count += 1

        if untrusted_count == 0:
            return 0.0

        score = min(untrusted_count * 7.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Untrusted communication: {untrusted_count} messages to unknown agents"
        )
        return score

    def _check_network_isolation(
        self, events: list[Event], indicators: list[str]
    ) -> float:
        """Flag complete absence of message activity as suspicious."""
        msg_types = {EventType.MESSAGE_SENT, EventType.MESSAGE_RECEIVED}
        msg_events = [e for e in events if e.event_type in msg_types]

        # Only flag isolation if the agent has other activity (not just idle)
        non_msg_events = [e for e in events if e.event_type not in msg_types]
        if not non_msg_events:
            return 0.0

        if len(msg_events) > 0:
            return 0.0

        # Agent is active but not communicating at all — mild suspicion
        indicators.append("Network isolation: active agent with zero message activity")
        return 10.0

    def _check_message_volume_anomalies(
        self, events: list[Event], indicators: list[str]
    ) -> float:
        """Flag sudden spike in message volume."""
        sent_events = [e for e in events if e.event_type == EventType.MESSAGE_SENT]
        if len(sent_events) < 2:
            return 0.0

        # Split events into two halves and compare rates
        midpoint = len(sent_events) // 2
        first_half = sent_events[:midpoint]
        second_half = sent_events[midpoint:]

        if not first_half:
            return 0.0

        first_rate = len(first_half)
        second_rate = len(second_half)

        if first_rate == 0:
            return 0.0

        multiplier = self._config.get("spike_multiplier", MESSAGE_SPIKE_MULTIPLIER)

        if second_rate > first_rate * multiplier:
            ratio = second_rate / first_rate
            score = min(ratio * 5.0, PER_SIGNAL_MAX)
            indicators.append(
                f"Message volume spike: {ratio:.1f}x increase in second half of window"
            )
            return score
        return 0.0

    def _check_fan_out(
        self, events: list[Event], indicators: list[str]
    ) -> float:
        """Flag messaging many unique agents."""
        sent_events = [e for e in events if e.event_type == EventType.MESSAGE_SENT]
        recipients: Counter[str] = Counter()

        for event in sent_events:
            recipient = event.metadata.get("recipient_agent_id", "")
            if recipient:
                recipients[recipient] += 1

        unique_count = len(recipients)
        threshold = self._config.get("fan_out_threshold", FAN_OUT_THRESHOLD)

        if unique_count <= threshold:
            return 0.0

        excess = unique_count - threshold
        score = min(excess * 5.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Fan-out pattern: messaged {unique_count} unique agents "
            f"(threshold: {threshold})"
        )
        return score

    def _get_trusted_agents(self, events: list[Event]) -> set[str]:
        """Extract the trusted agent set from event metadata."""
        for event in events:
            trusted: list[str] | None = event.metadata.get("trusted_agents")
            if trusted is not None:
                return set(trusted)
        return set()

    def _build_reasoning(self, total: float, indicators: list[str]) -> str:
        """Build human-readable reasoning string."""
        if not indicators:
            return "No social graph anomalies detected"
        signal_summary = "; ".join(indicators)
        return f"Social graph score {total:.1f}/100 — {signal_summary}"
