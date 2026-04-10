"""Identity brain -- behavioral fingerprinting and impersonation detection.

Analyzes writing style, action patterns, and vocabulary to create
behavioral fingerprints, then detects identity anomalies like
impersonation, sock puppets, and role inconsistency.

Signals (5 total):
    1. Behavioral fingerprint shift -- sudden style change (impersonation).
    2. Sock puppet detection -- same fingerprint across multiple agents.
    3. Identity consistency -- behavior doesn't match claimed role.
    4. Stylometric anomaly -- statistical writing style deviation.
    5. Vocabulary shift -- sudden vocabulary change mid-session.
"""

from __future__ import annotations

import math
import string
from collections import Counter
from typing import Any

from sybilcore.brains.base import BaseBrain, BrainScore
from sybilcore.core.config import BRAIN_SCORE_MAX, PER_SIGNAL_MAX
from sybilcore.models.event import Event, EventType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum content events for identity analysis.
_MIN_CONTENT_EVENTS: int = 5

# Fingerprint shift: cosine distance threshold for flagging.
_FINGERPRINT_SHIFT_THRESHOLD: float = 0.4

# Sock puppet: fingerprint similarity threshold.
_SOCK_PUPPET_SIMILARITY_THRESHOLD: float = 0.85

# Identity consistency: keyword match threshold for role alignment.
_ROLE_CONSISTENCY_THRESHOLD: float = 0.3

# Stylometric: z-score threshold for anomalous sentence length.
_STYLOMETRIC_ZSCORE_THRESHOLD: float = 2.0

# Vocabulary shift: Jaccard distance threshold between windows.
_VOCAB_SHIFT_THRESHOLD: float = 0.7

# Role keyword mappings for identity consistency checks.
_ROLE_KEYWORDS: dict[str, frozenset[str]] = {
    "assistant": frozenset({
        "help", "assist", "support", "provide", "here", "answer",
        "information", "recommend", "suggest", "happy",
    }),
    "analyst": frozenset({
        "data", "analysis", "report", "findings", "statistics",
        "trend", "metric", "insight", "correlation", "pattern",
    }),
    "security": frozenset({
        "secure", "vulnerability", "threat", "patch", "firewall",
        "audit", "compliance", "risk", "protect", "monitor",
    }),
    "developer": frozenset({
        "code", "function", "class", "variable", "debug",
        "implement", "test", "deploy", "api", "module",
    }),
}


class IdentityBrain(BaseBrain):
    """Detects identity anomalies via behavioral fingerprinting.

    Creates statistical profiles of agent behavior and flags:
    - Fingerprint shifts suggesting impersonation
    - Sock puppet patterns (shared fingerprints)
    - Role-behavior inconsistency
    - Stylometric deviations in writing patterns
    - Vocabulary regime changes
    """

    @property
    def name(self) -> str:
        return "identity"

    def score(self, events: list[Event]) -> BrainScore:
        """Analyze events for identity anomaly signals.

        Args:
            events: Chronologically ordered agent events.

        Returns:
            BrainScore with threat value 0-100.
        """
        content_events = [
            e for e in events
            if e.content and len(e.content) > 10
        ]

        if len(content_events) < _MIN_CONTENT_EVENTS:
            return self._empty_score("Insufficient content for identity analysis")

        indicators: list[str] = []
        scores: list[float] = []

        # Signal 1: Behavioral fingerprint shift
        scores.append(self._check_fingerprint_shift(content_events, indicators))

        # Signal 2: Sock puppet detection
        scores.append(self._check_sock_puppet(events, indicators))

        # Signal 3: Identity consistency
        scores.append(self._check_identity_consistency(events, indicators))

        # Signal 4: Stylometric anomaly
        scores.append(self._check_stylometric_anomaly(content_events, indicators))

        # Signal 5: Vocabulary shift
        scores.append(self._check_vocabulary_shift(content_events, indicators))

        total = min(sum(scores), BRAIN_SCORE_MAX)
        confidence = min(0.3 + (len(content_events) / 20.0), 1.0)

        return BrainScore(
            brain_name=self.name,
            value=total,
            confidence=confidence,
            reasoning=self._build_reasoning(total, indicators),
            indicators=indicators,
        )

    # ------------------------------------------------------------------
    # Signal 1: Behavioral fingerprint shift
    # ------------------------------------------------------------------

    def _check_fingerprint_shift(
        self, events: list[Event], indicators: list[str],
    ) -> float:
        """Detect sudden writing style changes (possible impersonation).

        Compares fingerprints between first and second halves of events.
        """
        if len(events) < 6:
            return 0.0

        mid = len(events) // 2
        fp_first = _compute_fingerprint([e.content for e in events[:mid]])
        fp_second = _compute_fingerprint([e.content for e in events[mid:]])

        distance = _fingerprint_distance(fp_first, fp_second)

        if distance < _FINGERPRINT_SHIFT_THRESHOLD:
            return 0.0

        score = min(distance * 50.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Fingerprint shift: distance={distance:.3f} between session halves "
            f"(threshold={_FINGERPRINT_SHIFT_THRESHOLD})"
        )
        return score

    # ------------------------------------------------------------------
    # Signal 2: Sock puppet detection
    # ------------------------------------------------------------------

    def _check_sock_puppet(
        self, events: list[Event], indicators: list[str],
    ) -> float:
        """Detect same behavioral signature across multiple agent IDs.

        Groups events by agent_id and checks fingerprint similarity.
        """
        agent_contents: dict[str, list[str]] = {}
        for event in events:
            if event.content and len(event.content) > 10:
                agent_contents.setdefault(event.agent_id, []).append(event.content)

        if len(agent_contents) < 2:
            return 0.0

        # Compare fingerprints between different agent IDs
        agent_ids = list(agent_contents.keys())
        puppet_pairs = 0

        for i in range(len(agent_ids)):
            if len(agent_contents[agent_ids[i]]) < 3:
                continue
            fp_i = _compute_fingerprint(agent_contents[agent_ids[i]])
            for j in range(i + 1, len(agent_ids)):
                if len(agent_contents[agent_ids[j]]) < 3:
                    continue
                fp_j = _compute_fingerprint(agent_contents[agent_ids[j]])
                similarity = 1.0 - _fingerprint_distance(fp_i, fp_j)
                if similarity >= _SOCK_PUPPET_SIMILARITY_THRESHOLD:
                    puppet_pairs += 1

        if puppet_pairs == 0:
            return 0.0

        score = min(puppet_pairs * 15.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Sock puppet: {puppet_pairs} agent pair(s) with matching "
            f"behavioral fingerprints"
        )
        return score

    # ------------------------------------------------------------------
    # Signal 3: Identity consistency
    # ------------------------------------------------------------------

    def _check_identity_consistency(
        self, events: list[Event], indicators: list[str],
    ) -> float:
        """Check if agent behavior matches its claimed role.

        Uses role keyword analysis to verify behavioral alignment.
        """
        claimed_role = ""
        for event in events:
            role = event.metadata.get("claimed_role", "")
            if role:
                claimed_role = role.lower()
                break

        if not claimed_role:
            return 0.0

        # Find best matching role keywords
        role_keywords = _ROLE_KEYWORDS.get(claimed_role)
        if not role_keywords:
            return 0.0

        # Analyze actual content for role-consistent vocabulary
        all_words: set[str] = set()
        for event in events:
            if event.content:
                words = {
                    w.strip(string.punctuation).lower()
                    for w in event.content.split()
                }
                all_words.update(words)

        if not all_words:
            return 0.0

        role_match = len(all_words & role_keywords) / len(role_keywords)

        if role_match >= _ROLE_CONSISTENCY_THRESHOLD:
            return 0.0

        # Low match means behavior doesn't align with claimed role
        inconsistency = 1.0 - (role_match / _ROLE_CONSISTENCY_THRESHOLD)
        score = min(inconsistency * 25.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Identity inconsistency: claimed role '{claimed_role}' but "
            f"only {role_match:.2f} keyword alignment "
            f"(threshold={_ROLE_CONSISTENCY_THRESHOLD})"
        )
        return score

    # ------------------------------------------------------------------
    # Signal 4: Stylometric anomaly
    # ------------------------------------------------------------------

    def _check_stylometric_anomaly(
        self, events: list[Event], indicators: list[str],
    ) -> float:
        """Detect statistical deviations in writing style.

        Analyzes sentence length distribution and flags events
        with abnormal patterns compared to the agent's baseline.
        """
        all_lengths: list[float] = []
        per_event_avg: list[float] = []

        for event in events:
            sentences = _split_sentences(event.content)
            lengths = [len(s.split()) for s in sentences if s.strip()]
            if lengths:
                all_lengths.extend(lengths)
                per_event_avg.append(sum(lengths) / len(lengths))

        if len(per_event_avg) < 5:
            return 0.0

        # Compute baseline stats
        mean_len = sum(per_event_avg) / len(per_event_avg)
        if mean_len == 0:
            return 0.0

        variance = sum((l - mean_len) ** 2 for l in per_event_avg) / len(per_event_avg)
        std = math.sqrt(variance) if variance > 0 else 0.001

        # Count events with anomalous sentence lengths
        anomalous = sum(
            1 for avg in per_event_avg
            if abs(avg - mean_len) / std > _STYLOMETRIC_ZSCORE_THRESHOLD
        )

        if anomalous == 0:
            return 0.0

        score = min(anomalous * 8.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Stylometric anomaly: {anomalous}/{len(per_event_avg)} events "
            f"with sentence length z-score > {_STYLOMETRIC_ZSCORE_THRESHOLD}"
        )
        return score

    # ------------------------------------------------------------------
    # Signal 5: Vocabulary shift
    # ------------------------------------------------------------------

    def _check_vocabulary_shift(
        self, events: list[Event], indicators: list[str],
    ) -> float:
        """Detect sudden vocabulary changes mid-session.

        Compares word frequency distributions between session halves.
        """
        if len(events) < 6:
            return 0.0

        mid = len(events) // 2
        vocab_first = _extract_vocabulary([e.content for e in events[:mid]])
        vocab_second = _extract_vocabulary([e.content for e in events[mid:]])

        if not vocab_first or not vocab_second:
            return 0.0

        # Jaccard distance between vocabulary sets
        intersection = vocab_first & vocab_second
        union = vocab_first | vocab_second
        jaccard_sim = len(intersection) / len(union) if union else 1.0
        jaccard_dist = 1.0 - jaccard_sim

        if jaccard_dist < _VOCAB_SHIFT_THRESHOLD:
            return 0.0

        score = min((jaccard_dist - _VOCAB_SHIFT_THRESHOLD) * 80.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Vocabulary shift: Jaccard distance={jaccard_dist:.3f} "
            f"between session halves (threshold={_VOCAB_SHIFT_THRESHOLD})"
        )
        return score

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_reasoning(self, total: float, indicators: list[str]) -> str:
        if not indicators:
            return "No identity anomaly signals detected"
        signal_summary = "; ".join(indicators)
        return f"Identity score {total:.1f}/100 -- {signal_summary}"


# ======================================================================
# Module-level helper functions
# ======================================================================


def _compute_fingerprint(texts: list[str]) -> dict[str, float]:
    """Compute a behavioral fingerprint from a list of text contents.

    Features:
    - Average sentence length
    - Average word length
    - Punctuation frequency
    - Function word frequency
    """
    if not texts:
        return {}

    all_text = " ".join(texts)
    words = all_text.lower().split()
    if not words:
        return {}

    sentences = _split_sentences(all_text)
    sentence_lengths = [len(s.split()) for s in sentences if s.strip()]

    fp: dict[str, float] = {
        "avg_sentence_len": (
            sum(sentence_lengths) / len(sentence_lengths)
            if sentence_lengths else 0.0
        ),
        "avg_word_len": sum(len(w) for w in words) / len(words),
        "punct_freq": sum(1 for c in all_text if c in string.punctuation) / len(all_text),
    }

    # Function word frequency
    function_words = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "it", "this", "that", "these", "those", "and", "but", "or",
    })
    func_count = sum(1 for w in words if w in function_words)
    fp["function_word_freq"] = func_count / len(words)

    return fp


def _fingerprint_distance(fp_a: dict[str, float], fp_b: dict[str, float]) -> float:
    """Compute normalized Euclidean distance between two fingerprints."""
    all_keys = set(fp_a.keys()) | set(fp_b.keys())
    if not all_keys:
        return 0.0

    sum_sq = 0.0
    for key in all_keys:
        a_val = fp_a.get(key, 0.0)
        b_val = fp_b.get(key, 0.0)
        # Normalize by max value to keep distance in 0-1 range
        max_val = max(abs(a_val), abs(b_val), 1.0)
        sum_sq += ((a_val - b_val) / max_val) ** 2

    return math.sqrt(sum_sq / len(all_keys))


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple heuristics."""
    if not text:
        return []
    # Split on sentence-ending punctuation
    sentences: list[str] = []
    current: list[str] = []
    for word in text.split():
        current.append(word)
        if word.endswith((".", "!", "?")):
            sentences.append(" ".join(current))
            current = []
    if current:
        sentences.append(" ".join(current))
    return sentences


def _extract_vocabulary(texts: list[str]) -> set[str]:
    """Extract unique vocabulary from a list of texts.

    Filters to meaningful words (3+ chars, no pure punctuation).
    """
    vocab: set[str] = set()
    for text in texts:
        words = {
            w.strip(string.punctuation).lower()
            for w in text.split()
        }
        vocab.update(w for w in words if len(w) >= 3)
    return vocab
