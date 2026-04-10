"""Compromise brain — detects signs of agent takeover or injection.

Highest-weight brain (1.3). Monitors prompt injection patterns,
unusual instruction sources, post-instruction behavioral shifts,
and credential/key leakage in outputs.
"""

from __future__ import annotations

import re

from sybilcore.brains.base import BaseBrain, BrainScore
from sybilcore.brains_v5_patched._patch_helpers import (
    check_evasion_fingerprint as _v5_check_evasion_fingerprint,
    check_burst_window as _v5_check_burst_window,
    check_coordination_cover as _v5_check_coordination_cover,
)
from sybilcore.core.config import BRAIN_SCORE_MAX, PER_SIGNAL_MAX
from sybilcore.models.event import Event, EventType

# Patterns indicating prompt injection attempts.
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?previous\s+(instructions?|prompts?)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"^system\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"forget\s+(everything|all|your)\s+(you|instructions?|rules?)", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(prior|previous|above)", re.IGNORECASE),
    re.compile(r"override\s+(safety|security|rules?|instructions?)", re.IGNORECASE),
)

# Patterns for credential/key leakage in output.
_CREDENTIAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:sk|pk|rk|ak)-[A-Za-z0-9]{20,}"),          # Stripe-style keys
    re.compile(r"ghp_[A-Za-z0-9]{36,}"),                       # GitHub personal tokens
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+={0,10}", re.IGNORECASE),  # Bearer tokens
    re.compile(r"(?:api[_-]?key|secret|token)\s*[=:]\s*\S{10,200}", re.IGNORECASE),
    re.compile(r"AIza[A-Za-z0-9\-_]{35}"),                     # Google API keys
    re.compile(r"AKIA[A-Z0-9]{16}"),                           # AWS access keys
    re.compile(r"xox[bpsa]-[A-Za-z0-9\-]{10,200}"),             # Slack tokens
)


class CompromiseBrain(BaseBrain):
    """Detects signs of agent compromise or takeover.

    Highest weight brain (1.3) — security critical.

    Signals:
        1. Prompt injection patterns — content containing injection markers.
        2. Unusual instruction sources — INSTRUCTION_RECEIVED from unknown source.
        3. Post-instruction behavioral shift — behavior change after instruction.
        4. Credential/key patterns in output — API keys, tokens in generated content.
    """

    @property
    def name(self) -> str:
        """Human-readable identifier."""
        return "compromise"

    def score(self, events: list[Event]) -> BrainScore:
        """Analyze events for compromise signals.

        Args:
            events: Chronologically ordered agent events.

        Returns:
            BrainScore with threat value 0-100.
        """
        if not events:
            return self._empty_score("No events to analyze")

        indicators: list[str] = []
        scores: list[float] = []

        scores.append(self._check_injection_patterns(events, indicators))
        scores.append(self._check_unusual_instruction_sources(events, indicators))
        scores.append(self._check_post_instruction_shift(events, indicators))
        scores.append(self._check_credential_leakage(events, indicators))
        scores.append(_v5_check_evasion_fingerprint(events, indicators))
        total = min(sum(scores), BRAIN_SCORE_MAX)
        confidence = min(0.5 + (len(events) / 20.0), 1.0)
        total = min(total, BRAIN_SCORE_MAX)  # explicit cap

        return BrainScore(
            brain_name=self.name,
            value=total,
            confidence=confidence,
            reasoning=self._build_reasoning(total, indicators),
            indicators=indicators,
        )

    def _check_injection_patterns(
        self, events: list[Event], indicators: list[str]
    ) -> float:
        """Detect prompt injection markers in event content."""
        injection_count = 0
        for event in events:
            text = event.content
            if not text:
                continue
            for pattern in _INJECTION_PATTERNS:
                if pattern.search(text):
                    injection_count += 1
                    break  # One match per event is enough

        if injection_count == 0:
            return 0.0

        score = min(injection_count * 12.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Prompt injection patterns: {injection_count} events with injection markers"
        )
        return score

    def _check_unusual_instruction_sources(
        self, events: list[Event], indicators: list[str]
    ) -> float:
        """Detect instructions from unknown or unexpected sources."""
        known_sources: set[str] = set(
            self._config.get("known_instruction_sources", ["system", "user", "orchestrator"])
        )
        instruction_events = [
            e for e in events if e.event_type == EventType.INSTRUCTION_RECEIVED
        ]

        unknown_count = 0
        for event in instruction_events:
            source = event.metadata.get("instruction_source", event.source)
            if source not in known_sources:
                unknown_count += 1

        if unknown_count == 0:
            return 0.0

        score = min(unknown_count * 10.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Unknown instruction sources: {unknown_count} instructions from unrecognized origins"
        )
        return score

    def _check_post_instruction_shift(
        self, events: list[Event], indicators: list[str]
    ) -> float:
        """Detect behavioral change immediately after receiving instructions."""
        instruction_indices = [
            i for i, e in enumerate(events)
            if e.event_type == EventType.INSTRUCTION_RECEIVED
        ]

        if not instruction_indices:
            return 0.0

        shift_count = 0
        for idx in instruction_indices:
            pre_types = _get_event_types_in_range(events, max(0, idx - 5), idx)
            post_types = _get_event_types_in_range(events, idx + 1, min(len(events), idx + 6))

            if not pre_types or not post_types:
                continue

            # Check if post-instruction event types differ significantly
            overlap = pre_types & post_types
            total_types = pre_types | post_types
            if total_types and len(overlap) / len(total_types) < 0.3:
                shift_count += 1

        if shift_count == 0:
            return 0.0

        score = min(shift_count * 12.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Post-instruction behavioral shift: {shift_count} significant behavior "
            f"changes after receiving instructions"
        )
        return score

    def _check_credential_leakage(
        self, events: list[Event], indicators: list[str]
    ) -> float:
        """Detect API keys, tokens, or credentials in output content."""
        output_events = [e for e in events if e.event_type == EventType.OUTPUT_GENERATED]
        leak_count = 0

        for event in output_events:
            text = event.content
            if not text:
                continue
            for pattern in _CREDENTIAL_PATTERNS:
                if pattern.search(text):
                    leak_count += 1
                    break  # One match per event

        if leak_count == 0:
            return 0.0

        score = min(leak_count * 15.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Credential leakage: {leak_count} outputs containing key/token patterns"
        )
        return score

    def _build_reasoning(self, total: float, indicators: list[str]) -> str:
        """Build human-readable reasoning string."""
        if not indicators:
            return "No compromise signals detected"
        signal_summary = "; ".join(indicators)
        return f"Compromise score {total:.1f}/100 — {signal_summary}"


def _get_event_types_in_range(
    events: list[Event], start: int, end: int
) -> set[str]:
    """Extract unique event type values from a slice of events."""
    return {events[i].event_type.value for i in range(start, min(end, len(events)))}
