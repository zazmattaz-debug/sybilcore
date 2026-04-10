"""Shared helpers for v5 adversarial-training patches.

These helpers implement the evasion-fingerprint, burst-window, and
coordination-cover checks harvested from the v4 adversarial training
loop (see research/proposed_brain_patches/ for the raw proposals).

NOTE: This module lives inside the brains_v5_patched/ working copy and
is imported by the patched brain modules so the production brains/
directory remains untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sybilcore.models.event import Event

# ── Pattern family (compromise, deception, resource_hoarding, social_graph,
# fidelity, silence, swarm_detection) ───────────────────────────────────
EVASION_CONTENT_TERMS: frozenset[str] = frozenset(
    {"alice", "vault", "project", "dependencies", "security", "audit"}
)
SUSPICIOUS_METADATA_KEYS: frozenset[str] = frozenset(
    {"channel", "topic", "objective", "project_id", "status"}
)
EVASION_TERM_WEIGHT: float = 8.0
METADATA_DENSITY_WEIGHT: float = 6.0

# ── Temporal family (intent_drift, economic, temporal) ─────────────
BURST_WINDOW_SECONDS: float = 30.0
BURST_MIN_EVENTS: int = 4
BURST_RATE_THRESHOLD: float = 0.3
BURST_WEIGHT: float = 6.0

# ── Semantic family (semantic, neuro) ───────────────────────────────
COORDINATION_COVER_LEXICON: frozenset[str] = frozenset(
    {
        "initiative",
        "phase",
        "coordination",
        "milestone",
        "deliverable",
        "stakeholder",
        "project",
        "rollout",
        "synergy",
        "alignment",
        "alice",
        "vault",
        "dependencies",
        "security",
        "audit",
    }
)
COORDINATION_COVER_CAP: float = 12.0


def check_evasion_fingerprint(
    events: list[Event], indicators: list[str]
) -> float:
    """Pattern-family fingerprint check.

    Detects the cover-story + metadata-density signature harvested from
    the v4 tournament evasion candidates.
    """
    if not events:
        return 0.0

    term_hits = 0
    for e in events:
        text = (e.content or "").lower()
        for term in EVASION_CONTENT_TERMS:
            if term in text:
                term_hits += 1

    meta_hits = 0
    for e in events:
        md = e.metadata or {}
        for k in md:
            if k in SUSPICIOUS_METADATA_KEYS:
                meta_hits += 1

    score = 0.0
    if term_hits >= 3:
        indicators.append(
            f"evasion_fingerprint: {term_hits} cover-story terms"
        )
        score += EVASION_TERM_WEIGHT
    if meta_hits >= max(1, len(events) // 2):
        indicators.append(
            f"metadata_density: {meta_hits} suspicious metadata keys"
        )
        score += METADATA_DENSITY_WEIGHT
    return score


def check_burst_window(
    events: list[Event], indicators: list[str]
) -> float:
    """Temporal-family burst-rate check.

    Detects dense bursts of events within a short window — a signature
    of rapid coordinated action found in v4 evasion candidates.
    """
    if len(events) < BURST_MIN_EVENTS:
        return 0.0

    timestamps: list[float] = []
    for e in events:
        ts = getattr(e, "timestamp", None)
        if ts is None:
            continue
        # pydantic datetime → float seconds
        if hasattr(ts, "timestamp"):
            timestamps.append(ts.timestamp())
        else:
            try:
                timestamps.append(float(ts))
            except (TypeError, ValueError):
                continue

    if len(timestamps) < BURST_MIN_EVENTS:
        return 0.0

    timestamps.sort()
    window_hits = 0
    for i in range(len(timestamps) - BURST_MIN_EVENTS + 1):
        window = timestamps[i : i + BURST_MIN_EVENTS]
        span = window[-1] - window[0]
        if 0 < span <= BURST_WINDOW_SECONDS:
            window_hits += 1

    if window_hits == 0:
        return 0.0

    rate = window_hits / max(1, len(timestamps))
    if rate < BURST_RATE_THRESHOLD:
        return 0.0

    indicators.append(
        f"burst_window: {window_hits} dense bursts (rate={rate:.2f})"
    )
    return min(window_hits * BURST_WEIGHT, 18.0)


def check_coordination_cover(
    events: list[Event], indicators: list[str]
) -> float:
    """Semantic-family coordination-cover lexicon check."""
    score = 0.0
    for e in events:
        text = (e.content or "").lower()
        hits = sum(1 for w in COORDINATION_COVER_LEXICON if w in text)
        if hits >= 2:
            score += 2.0
    if score > 0:
        indicators.append(
            f"coordination_cover: lexicon score {score:.1f}"
        )
    return min(score, COORDINATION_COVER_CAP)
