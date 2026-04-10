"""Embedding brain — transformer-based semantic detection via Hugging Face.

Uses sentence-transformers to compute dense embeddings for agent messages,
enabling detection that keyword/regex methods fundamentally cannot achieve:

  1. Semantic drift detection — cosine distance between consecutive messages
     reveals gradual topic shifting that evades keyword filters.
  2. Embedding-space clustering — malicious agents cluster differently in
     embedding space vs. benign agents, even when using novel vocabulary.
  3. Paraphrase detection — catches agents that rephrase the same malicious
     intent using completely different words (the Makishima Problem).
  4. Anomaly scoring — messages that are outliers in embedding space relative
     to the agent's own message history suggest compromised behavior.

This brain requires the `sentence-transformers` package and a Hugging Face
token (HF_TOKEN env var) for gated models. Falls back gracefully to a zero
score when dependencies are unavailable.

Layer architecture:
  - Layer 1 (local): Uses `all-MiniLM-L6-v2` (22M params, fast) for
    real-time scoring. No API calls needed — runs entirely on CPU.
  - Layer 2 (optional): When HF_TOKEN is set, can use larger models
    like `all-mpnet-base-v2` for higher accuracy experiments.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from sybilcore.brains.base import BaseBrain, BrainScore
from sybilcore.core.config import BRAIN_SCORE_MAX, PER_SIGNAL_MAX
from sybilcore.models.event import Event, EventType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default model — small, fast, runs on CPU. Good baseline.
DEFAULT_MODEL: str = "all-MiniLM-L6-v2"

# Larger model for higher accuracy (requires more compute).
LARGE_MODEL: str = "all-mpnet-base-v2"

# Minimum messages needed for meaningful analysis.
_MIN_MESSAGES: int = 3

# Cosine similarity thresholds.
_DRIFT_THRESHOLD: float = 0.3       # Below this = suspicious topic drift
_PARAPHRASE_THRESHOLD: float = 0.85  # Above this = likely paraphrase
_ANOMALY_THRESHOLD: float = 2.0      # Std devs from centroid = anomaly

# Maximum number of messages to embed per scoring call (memory bound).
_MAX_EMBED_MESSAGES: int = 50

# Cache for the loaded model (expensive to load, reuse across calls).
_model_cache: dict[str, Any] = {}


def _get_model(model_name: str = DEFAULT_MODEL) -> Any:
    """Load and cache a sentence-transformer model.

    Returns None if sentence-transformers is not available.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    try:
        from sentence_transformers import SentenceTransformer

        hf_token = os.environ.get("HF_TOKEN")
        model = SentenceTransformer(model_name, token=hf_token)
        _model_cache[model_name] = model
        logger.info("Loaded sentence-transformer model: %s", model_name)
        return model
    except Exception as exc:
        logger.warning("Failed to load model %s: %s", model_name, exc)
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _pairwise_cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute NxN cosine similarity matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normalized = embeddings / norms
    return normalized @ normalized.T


class EmbeddingBrain(BaseBrain):
    """Detects deceptive agent behavior using transformer embeddings.

    This brain addresses the Makishima Problem: adversarial agents that
    evade keyword-based detection by using semantically equivalent but
    lexically novel language. Dense embeddings capture meaning regardless
    of word choice.

    Signals (4 total):
        1. Semantic drift — consecutive messages diverge in embedding space.
        2. Paraphrase clustering — repeated malicious intent rephrased.
        3. Embedding anomaly — messages that are outliers from agent centroid.
        4. Cross-message coherence — instruction vs output embedding distance.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        super().__init__(config)
        self._model_name = model_name
        self._model: Any = None  # Lazy-loaded

    @property
    def name(self) -> str:
        return "embedding"

    def _ensure_model(self) -> bool:
        """Lazy-load the model on first use."""
        if self._model is None:
            self._model = _get_model(self._model_name)
        return self._model is not None

    def _embed(self, texts: list[str]) -> np.ndarray | None:
        """Embed a list of texts. Returns None if model unavailable."""
        if not self._ensure_model():
            return None
        try:
            return self._model.encode(texts, show_progress_bar=False)
        except Exception as exc:
            logger.warning("Embedding failed: %s", exc)
            return None

    def score(self, events: list[Event]) -> BrainScore:
        """Analyze events using transformer embeddings.

        Args:
            events: Chronologically ordered agent events.

        Returns:
            BrainScore with threat value 0-100.
        """
        content_events = [
            e for e in events
            if e.content and len(e.content.strip()) >= 10
        ]

        if len(content_events) < _MIN_MESSAGES:
            return self._empty_score("Too few content events for embedding analysis")

        # Limit to most recent messages for memory efficiency.
        recent = content_events[-_MAX_EMBED_MESSAGES:]
        texts = [e.content for e in recent]

        embeddings = self._embed(texts)
        if embeddings is None:
            return self._empty_score("Embedding model unavailable")

        indicators: list[str] = []
        scores: list[float] = []

        # Signal 1: Semantic drift detection
        scores.append(self._check_semantic_drift(embeddings, indicators))

        # Signal 2: Paraphrase clustering
        scores.append(self._check_paraphrase_clusters(embeddings, texts, indicators))

        # Signal 3: Embedding anomaly detection
        scores.append(self._check_embedding_anomalies(embeddings, indicators))

        # Signal 4: Instruction-output misalignment
        instruction_events = [e for e in recent if e.event_type == EventType.INSTRUCTION_RECEIVED]
        output_events = [e for e in recent if e.event_type == EventType.OUTPUT_GENERATED]
        scores.append(self._check_instruction_output_distance(
            instruction_events, output_events, indicators,
        ))

        total = min(sum(scores), BRAIN_SCORE_MAX)
        confidence = min(0.4 + (len(recent) / 30.0), 1.0)

        return BrainScore(
            brain_name=self.name,
            value=total,
            confidence=confidence,
            reasoning=self._build_reasoning(total, indicators),
            indicators=indicators,
        )

    # ------------------------------------------------------------------
    # Signal 1: Semantic drift
    # ------------------------------------------------------------------

    def _check_semantic_drift(
        self, embeddings: np.ndarray, indicators: list[str],
    ) -> float:
        """Detect abrupt topic shifts in consecutive messages.

        A benign agent stays on-topic. A compromised agent may suddenly
        shift to exfiltration/reconnaissance topics, visible as low
        cosine similarity between consecutive embeddings.
        """
        if len(embeddings) < 2:
            return 0.0

        drift_events = 0
        max_drift = 0.0

        for i in range(1, len(embeddings)):
            sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < _DRIFT_THRESHOLD:
                drift_events += 1
                max_drift = max(max_drift, 1.0 - sim)

        if drift_events == 0:
            return 0.0

        drift_rate = drift_events / (len(embeddings) - 1)
        score = min(drift_rate * 40.0 + max_drift * 15.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Semantic drift: {drift_events}/{len(embeddings)-1} consecutive "
            f"message pairs below similarity threshold (max drift: {max_drift:.2f})"
        )
        return score

    # ------------------------------------------------------------------
    # Signal 2: Paraphrase clustering
    # ------------------------------------------------------------------

    def _check_paraphrase_clusters(
        self, embeddings: np.ndarray, texts: list[str], indicators: list[str],
    ) -> float:
        """Detect repeated malicious intent expressed with different words.

        If multiple messages are near-identical in embedding space but use
        different surface forms, the agent may be probing or repeating
        malicious commands with varied phrasing to find one that works.
        """
        if len(embeddings) < 3:
            return 0.0

        sim_matrix = _pairwise_cosine_matrix(embeddings)
        paraphrase_pairs = 0

        for i in range(len(embeddings)):
            for j in range(i + 2, len(embeddings)):  # Skip adjacent (normal continuity)
                if sim_matrix[i, j] >= _PARAPHRASE_THRESHOLD:
                    # High embedding similarity — check if surface text differs
                    text_overlap = _word_overlap(texts[i], texts[j])
                    if text_overlap < 0.5:  # Different words, same meaning
                        paraphrase_pairs += 1

        if paraphrase_pairs == 0:
            return 0.0

        score = min(paraphrase_pairs * 10.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Paraphrase detection: {paraphrase_pairs} non-adjacent message "
            f"pair(s) with high semantic similarity but low lexical overlap"
        )
        return score

    # ------------------------------------------------------------------
    # Signal 3: Embedding anomaly detection
    # ------------------------------------------------------------------

    def _check_embedding_anomalies(
        self, embeddings: np.ndarray, indicators: list[str],
    ) -> float:
        """Detect messages that are outliers in the agent's embedding space.

        Computes the centroid of all message embeddings and flags messages
        whose distance from the centroid exceeds a threshold. A sudden
        outlier suggests injected or compromised behavior.
        """
        if len(embeddings) < 4:
            return 0.0

        centroid = embeddings.mean(axis=0)
        distances = np.array([
            1.0 - _cosine_similarity(emb, centroid) for emb in embeddings
        ])

        mean_dist = distances.mean()
        std_dist = distances.std()
        if std_dist < 1e-6:
            return 0.0

        z_scores = (distances - mean_dist) / std_dist
        anomaly_count = int(np.sum(z_scores > _ANOMALY_THRESHOLD))

        if anomaly_count == 0:
            return 0.0

        max_z = float(z_scores.max())
        score = min(anomaly_count * 8.0 + max_z * 5.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Embedding anomalies: {anomaly_count} message(s) are outliers "
            f"from agent centroid (max z-score: {max_z:.2f})"
        )
        return score

    # ------------------------------------------------------------------
    # Signal 4: Instruction-output embedding distance
    # ------------------------------------------------------------------

    def _check_instruction_output_distance(
        self,
        instruction_events: list[Event],
        output_events: list[Event],
        indicators: list[str],
    ) -> float:
        """Detect misalignment between instructions and outputs in embedding space.

        If an agent's outputs are semantically distant from its instructions,
        it may be ignoring/subverting its directive — a classic Makishima pattern.
        """
        if not instruction_events or not output_events:
            return 0.0

        instruction_texts = [e.content for e in instruction_events if e.content]
        output_texts = [e.content for e in output_events if e.content]

        if not instruction_texts or not output_texts:
            return 0.0

        all_texts = instruction_texts + output_texts
        all_embeddings = self._embed(all_texts)
        if all_embeddings is None:
            return 0.0

        n_inst = len(instruction_texts)
        inst_embeddings = all_embeddings[:n_inst]
        out_embeddings = all_embeddings[n_inst:]

        inst_centroid = inst_embeddings.mean(axis=0)
        out_centroid = out_embeddings.mean(axis=0)

        alignment = _cosine_similarity(inst_centroid, out_centroid)

        if alignment > 0.5:  # Reasonably aligned
            return 0.0

        misalignment = 1.0 - alignment
        score = min(misalignment * 30.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Instruction-output misalignment: embedding alignment {alignment:.2f} "
            f"(threshold: 0.50) — outputs diverge from stated instructions"
        )
        return score

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_reasoning(self, total: float, indicators: list[str]) -> str:
        if not indicators:
            return "No embedding-based deception signals detected"
        signal_summary = "; ".join(indicators)
        return f"Embedding score {total:.1f}/100 — {signal_summary}"


def _word_overlap(text_a: str, text_b: str) -> float:
    """Compute word-level Jaccard overlap between two texts."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)
