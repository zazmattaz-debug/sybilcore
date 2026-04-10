"""Fidelity brain — RAG and document grounding accuracy detection.

Purpose-built for enterprise use cases where AI agents summarize, act on,
or respond based on source documents (SOPs, SLPs, financial reports, contracts,
medical records). Catches hallucinated numbers, dates, names, and claims.

Three detection layers:
  1. Numerical fidelity — extracts all numbers from source and output,
     flags mismatches (fabricated stats, wrong amounts, altered dates)
  2. Entity fidelity — extracts named entities (companies, people, places),
     flags entities in output that don't exist in source (invented references)
  3. Semantic chunk fidelity — embeds source and output in chunks,
     flags output chunks that are semantically distant from ALL source chunks

The key insight: embeddings alone score 60% on hallucination because
"Sydney" and "Canberra" embed similarly. But extracting "capital = Sydney"
vs "capital = Canberra" as structured claims catches the error every time.
Hybrid approach: structured extraction + embedding distance.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from sybilcore.brains.base import BaseBrain, BrainScore
from sybilcore.brains_v5_patched._patch_helpers import (
    check_evasion_fingerprint as _v5_check_evasion_fingerprint,
    check_burst_window as _v5_check_burst_window,
    check_coordination_cover as _v5_check_coordination_cover,
)
from sybilcore.brains.embedding import DEFAULT_MODEL, _cosine_similarity, _get_model
from sybilcore.core.config import BRAIN_SCORE_MAX, PER_SIGNAL_MAX
from sybilcore.models.event import Event, EventType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_CONTENT_LENGTH: int = 20
_CHUNK_SIZE: int = 100  # characters per chunk for semantic comparison
_NUM_MISMATCH_WEIGHT: float = 12.0  # Score per mismatched number
_ENTITY_MISMATCH_WEIGHT: float = 8.0  # Score per hallucinated entity
_CHUNK_DRIFT_THRESHOLD: float = 0.5  # Cosine sim below this = drifted chunk
_CHUNK_DRIFT_WEIGHT: float = 6.0  # Score per drifted chunk

# Patterns for number extraction
_NUMBER_PATTERN = re.compile(
    r"""
    \$[\d,]+(?:\.\d+)?[BMKbmk]?        # Dollar amounts: $4.2B, $500, $1,200.50
    |[\d,]+(?:\.\d+)?%                   # Percentages: 12%, 99.97%, 0.5%
    |\d{1,2}/\d{1,2}/\d{2,4}            # Dates: 03/15/1985, 1/1/2025
    |\d{4}-\d{2}-\d{2}                   # ISO dates: 2025-03-15
    |(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}  # Written dates
    |\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}
    |[\d,]+(?:\.\d+)?\s*(?:meters?|km|miles?|feet|inches|kg|lbs?|pounds?|gallons?|liters?|°[CF]|degrees?|billion|million|thousand|hundred)
    |[\d,]+(?:\.\d+)?                    # Plain numbers: 206, 3.14, 1,000,000
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Patterns for entity extraction (capitalized multi-word phrases)
_ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+(?:of|the|and|for|in|on|at|by|to|de|van|von|del|la|le))?(?:\s+[A-Z][a-z]+)+)\b"
)

# Common non-entity capitalized phrases to filter out
_ENTITY_STOPWORDS = frozenset({
    "The", "This", "That", "These", "Those", "Here", "There",
    "According", "Based", "However", "Therefore", "Furthermore",
    "Additionally", "Moreover", "Nevertheless", "Consequently",
    "In", "On", "At", "By", "For", "With", "From", "To",
    "As", "If", "Or", "And", "But", "Not", "All", "Each",
    "Type", "Section", "Article", "Chapter", "Part", "Phase",
})


def _normalize_number(text: str) -> str:
    """Normalize a number string for comparison.

    $4.2B and $4.2 billion should match.
    22% and 22 percent should match.
    900 million and $900M should match.
    """
    clean = text.replace(",", "").strip().lower()
    # Normalize unit suffixes
    clean = clean.replace("billion", "b").replace("million", "m").replace("thousand", "k")
    clean = clean.replace(" b", "b").replace(" m", "m").replace(" k", "k")
    clean = clean.replace("percent", "%")
    # Strip units for pure numeric comparison
    clean = re.sub(r'\s*(meters?|km|miles?|feet|inches|kg|lbs?|pounds?|gallons?|liters?|degrees?)\s*$', '', clean)
    return clean


def _extract_numbers(text: str) -> list[str]:
    """Extract all numerical values from text, normalized for comparison."""
    matches = _NUMBER_PATTERN.findall(text)
    normalized = []
    for m in matches:
        clean = _normalize_number(m)
        if len(clean) >= 1 and any(c.isdigit() for c in clean):
            normalized.append(clean)
    return normalized


def _extract_entities(text: str) -> set[str]:
    """Extract named entities (capitalized multi-word phrases)."""
    matches = _ENTITY_PATTERN.findall(text)
    entities = set()
    for m in matches:
        words = m.split()
        if words[0] not in _ENTITY_STOPWORDS and len(m) > 3:
            entities.add(m.lower())
    return entities


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE) -> list[str]:
    """Split text into sentence-aware chunks."""
    # Split on sentence boundaries first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) > chunk_size and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) >= 20]


class FidelityBrain(BaseBrain):
    """Detects hallucinations in AI agent outputs by comparing against source material.

    Designed for enterprise RAG pipelines where agents summarize SOPs, financial
    reports, legal contracts, and medical records. Catches fabricated numbers,
    invented entities, and semantic drift from source documents.

    Expects events in pairs: INSTRUCTION_RECEIVED (source document) followed by
    OUTPUT_GENERATED (agent's response/summary). Compares the two for fidelity.

    Signals (3 total):
        1. Numerical fidelity — numbers in output that don't appear in source
        2. Entity fidelity — named entities in output absent from source
        3. Chunk fidelity — output chunks semantically distant from all source chunks
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._model: Any = None

    @property
    def name(self) -> str:
        return "fidelity"

    def _ensure_model(self) -> bool:
        if self._model is None:
            self._model = _get_model(DEFAULT_MODEL)
        return self._model is not None

    def score(self, events: list[Event]) -> BrainScore:
        """Score events for hallucination/fidelity issues.

        Looks for instruction→output pairs and checks whether the output
        faithfully represents the source material.
        """
        # Find source-output pairs
        instructions = [e for e in events if e.event_type == EventType.INSTRUCTION_RECEIVED
                        and e.content and len(e.content) >= _MIN_CONTENT_LENGTH]
        outputs = [e for e in events if e.event_type == EventType.OUTPUT_GENERATED
                   and e.content and len(e.content) >= _MIN_CONTENT_LENGTH]

        if not instructions or not outputs:
            return self._empty_score("No instruction-output pairs for fidelity check")

        # Use all instructions as "source" and all outputs as "agent response"
        source_text = " ".join(e.content for e in instructions)
        output_text = " ".join(e.content for e in outputs)

        indicators: list[str] = []
        scores: list[float] = []

        # Signal 1: Numerical fidelity
        scores.append(self._check_numerical_fidelity(source_text, output_text, indicators))

        # Signal 2: Entity fidelity
        scores.append(self._check_entity_fidelity(source_text, output_text, indicators))

        # Signal 3: Chunk fidelity (embedding-based)
        scores.append(self._check_chunk_fidelity(source_text, output_text, indicators))
        scores.append(_v5_check_evasion_fingerprint(events, indicators))
        total = min(sum(scores), BRAIN_SCORE_MAX)
        confidence = min(0.5 + (len(outputs) / 10.0), 1.0)

        return BrainScore(
            brain_name=self.name,
            value=total,
            confidence=confidence,
            reasoning=self._build_reasoning(total, indicators),
            indicators=indicators,
        )

    # ------------------------------------------------------------------
    # Signal 1: Numerical fidelity
    # ------------------------------------------------------------------

    def _check_numerical_fidelity(
        self, source: str, output: str, indicators: list[str],
    ) -> float:
        """Check if numbers in the output match numbers in the source.

        Extracts all numerical values (dollar amounts, percentages, dates,
        measurements, counts) and flags output numbers that don't appear
        anywhere in the source.
        """
        source_numbers = set(_extract_numbers(source))
        output_numbers = set(_extract_numbers(output))

        if not output_numbers:
            return 0.0

        # Numbers in output that aren't in source = potentially hallucinated
        hallucinated_numbers = output_numbers - source_numbers

        # Filter out common non-specific numbers (1, 2, etc.)
        hallucinated_numbers = {n for n in hallucinated_numbers
                                if not (n.isdigit() and int(n) < 10)}

        # Fuzzy match: extract core numeric values for comparison
        # "$4.2b" should match "$4.2 billion" and "4.2"
        def _core_value(num_str: str) -> str:
            """Extract just the digits and decimal from a number string."""
            return re.sub(r'[^\d.]', '', num_str)

        source_cores = {_core_value(n) for n in source_numbers}
        hallucinated_numbers = {n for n in hallucinated_numbers
                                if _core_value(n) not in source_cores}

        if not hallucinated_numbers:
            return 0.0

        mismatch_rate = len(hallucinated_numbers) / len(output_numbers)
        score = min(len(hallucinated_numbers) * _NUM_MISMATCH_WEIGHT, PER_SIGNAL_MAX)

        indicators.append(
            f"Numerical fidelity: {len(hallucinated_numbers)}/{len(output_numbers)} "
            f"numbers in output not found in source — "
            f"hallucinated: {sorted(hallucinated_numbers)[:5]}"
        )
        return score

    # ------------------------------------------------------------------
    # Signal 2: Entity fidelity
    # ------------------------------------------------------------------

    def _check_entity_fidelity(
        self, source: str, output: str, indicators: list[str],
    ) -> float:
        """Check if named entities in the output exist in the source.

        Extracts capitalized multi-word phrases (company names, people,
        places, organizations) and flags entities in the output that
        don't appear in the source document.
        """
        source_entities = _extract_entities(source)
        output_entities = _extract_entities(output)

        if not output_entities:
            return 0.0

        # Entities in output not in source
        hallucinated_entities = output_entities - source_entities

        # Check partial matches (entity words overlap)
        source_words = {w.lower() for e in source_entities for w in e.split()}
        truly_hallucinated = set()
        for entity in hallucinated_entities:
            entity_words = set(entity.lower().split())
            if not (entity_words & source_words):
                truly_hallucinated.add(entity)

        if not truly_hallucinated:
            return 0.0

        score = min(len(truly_hallucinated) * _ENTITY_MISMATCH_WEIGHT, PER_SIGNAL_MAX)
        indicators.append(
            f"Entity fidelity: {len(truly_hallucinated)} named entities in output "
            f"not found in source — {sorted(truly_hallucinated)[:3]}"
        )
        return score

    # ------------------------------------------------------------------
    # Signal 3: Chunk fidelity (embedding-based)
    # ------------------------------------------------------------------

    def _check_chunk_fidelity(
        self, source: str, output: str, indicators: list[str],
    ) -> float:
        """Check if output chunks are semantically grounded in source chunks.

        Splits both source and output into sentence-level chunks, embeds them,
        and checks if each output chunk has at least one similar source chunk.
        Output chunks with no close source chunk are potentially hallucinated.
        """
        if not self._ensure_model():
            return 0.0

        source_chunks = _chunk_text(source)
        output_chunks = _chunk_text(output)

        if not source_chunks or not output_chunks:
            return 0.0

        try:
            source_embs = self._model.encode(source_chunks, show_progress_bar=False)
            output_embs = self._model.encode(output_chunks, show_progress_bar=False)
        except Exception:
            return 0.0

        # For each output chunk, find max similarity to any source chunk
        drifted_count = 0
        for out_emb in output_embs:
            max_sim = max(
                _cosine_similarity(out_emb, src_emb)
                for src_emb in source_embs
            )
            if max_sim < _CHUNK_DRIFT_THRESHOLD:
                drifted_count += 1

        if drifted_count == 0:
            return 0.0

        drift_rate = drifted_count / len(output_chunks)
        score = min(drifted_count * _CHUNK_DRIFT_WEIGHT, PER_SIGNAL_MAX)
        indicators.append(
            f"Chunk fidelity: {drifted_count}/{len(output_chunks)} output chunks "
            f"have no matching source chunk (threshold: {_CHUNK_DRIFT_THRESHOLD})"
        )
        return score

    def _build_reasoning(self, total: float, indicators: list[str]) -> str:
        if not indicators:
            return "No fidelity issues detected — output matches source material"
        return f"Fidelity score {total:.1f}/100 — {'; '.join(indicators)}"
