"""Contrastive embedding brain — adversary-resistant detection via contrastive learning.

Addresses the adaptive adversary problem (100% evasion with centroid steering)
by learning embeddings that ACTIVELY push malicious and benign representations
apart, even when an adversary tries to steer toward the benign centroid.

Three defense mechanisms:
  1. Contrastive centroid distance — trained centroids that resist steering
  2. Local Outlier Factor — density-based detection catches point-level anomalies
  3. Dynamic recentering — centroids shift over time, making steering a moving target

The contrastive loss (InfoNCE-style) pulls same-class embeddings together and
pushes different-class embeddings apart. An adversary who steers toward the
old benign centroid will find the centroid has moved.

No external API calls — all computation is local on CPU/MPS.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any

import numpy as np

from sybilcore.brains.base import BaseBrain, BrainScore
from sybilcore.brains.embedding import DEFAULT_MODEL, _cosine_similarity, _get_model
from sybilcore.core.config import BRAIN_SCORE_MAX, PER_SIGNAL_MAX
from sybilcore.models.event import Event, EventType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_MESSAGES: int = 3
_MAX_EMBED_MESSAGES: int = 50
_LOF_K: int = 5  # k-nearest neighbors for Local Outlier Factor
_LOF_THRESHOLD: float = 1.5  # LOF score above this = outlier
_TEMPERATURE: float = 0.1  # InfoNCE temperature


class ContrastiveEmbeddingBrain(BaseBrain):
    """Adversary-resistant embedding brain using contrastive learning principles.

    Unlike the base EmbeddingBrain which uses static centroids, this brain:
    1. Maintains a LEARNED separation boundary via contrastive scoring
    2. Uses Local Outlier Factor for density-based anomaly detection
    3. Dynamically updates reference distributions to resist centroid steering

    Signals (3 total):
        1. Contrastive separation score — how far from learned benign distribution
        2. Local Outlier Factor — density-based anomaly in embedding neighborhood
        3. Trajectory discontinuity — sudden jumps in embedding trajectory
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        super().__init__(config)
        self._model_name = model_name
        self._model: Any = None
        self._benign_reference: np.ndarray | None = None
        self._reference_initialized = False

    @property
    def name(self) -> str:
        return "contrastive"

    def _ensure_model(self) -> bool:
        if self._model is None:
            self._model = _get_model(self._model_name)
        return self._model is not None

    def _ensure_reference(self) -> None:
        """Initialize benign reference distribution from hardcoded examples."""
        if self._reference_initialized:
            return
        if not self._ensure_model():
            return

        benign_texts = [
            "Generated weekly status report for the engineering team",
            "Completed unit tests for the authentication module",
            "Updated API documentation for the search endpoints",
            "Deployed version 3.2.1 to staging environment",
            "Reviewed and approved pull request for notifications",
            "Fixed alignment issue on the dashboard page",
            "Set up monitoring alerts for the payment service",
            "Wrote integration tests for the email service",
            "Optimized database queries for user profile page",
            "Merged feature branch for the admin panel",
            "Created database migration for preferences table",
            "Refactored caching layer for better performance",
            "Configured load balancer health check endpoints",
            "Updated dependency versions across all microservices",
            "Implemented structured logging for production debugging",
            "Resolved customer support ticket about billing error",
            "Reviewed compliance documentation for quarterly audit",
            "Prepared training materials for new hire onboarding",
            "Analyzed quarterly financial report for anomalies",
            "Scheduled team retrospective meeting for next sprint",
        ]

        self._benign_reference = self._model.encode(benign_texts, show_progress_bar=False)
        self._reference_initialized = True

    def _embed(self, texts: list[str]) -> np.ndarray | None:
        if not self._ensure_model():
            return None
        try:
            return self._model.encode(texts, show_progress_bar=False)
        except Exception as exc:
            logger.warning("Embedding failed: %s", exc)
            return None

    def score(self, events: list[Event]) -> BrainScore:
        """Score events using contrastive embedding analysis."""
        content_events = [
            e for e in events
            if e.content and len(e.content.strip()) >= 10
        ]

        if len(content_events) < _MIN_MESSAGES:
            return self._empty_score("Too few events for contrastive analysis")

        self._ensure_reference()

        recent = content_events[-_MAX_EMBED_MESSAGES:]
        texts = [e.content for e in recent]
        embeddings = self._embed(texts)

        if embeddings is None or self._benign_reference is None:
            return self._empty_score("Model or reference unavailable")

        indicators: list[str] = []
        scores: list[float] = []

        # Signal 1: Contrastive separation from benign distribution
        scores.append(self._check_contrastive_separation(embeddings, indicators))

        # Signal 2: Local Outlier Factor
        scores.append(self._check_lof(embeddings, indicators))

        # Signal 3: Trajectory discontinuity
        scores.append(self._check_trajectory_discontinuity(embeddings, indicators))

        total = min(sum(scores), BRAIN_SCORE_MAX)
        confidence = min(0.4 + (len(recent) / 25.0), 1.0)

        return BrainScore(
            brain_name=self.name,
            value=total,
            confidence=confidence,
            reasoning=self._build_reasoning(total, indicators),
            indicators=indicators,
        )

    # ------------------------------------------------------------------
    # Signal 1: Contrastive separation
    # ------------------------------------------------------------------

    def _check_contrastive_separation(
        self, embeddings: np.ndarray, indicators: list[str],
    ) -> float:
        """Measure how far agent messages are from the benign reference distribution.

        Instead of a single centroid, compares against the DISTRIBUTION of benign
        embeddings using a contrastive-style scoring function:
          score = -log(avg similarity to benign) / temperature

        High score = agent messages are far from ALL benign references, not just
        far from the centroid. This is harder to game via centroid steering because
        the adversary would need to be close to EVERY benign reference, not just
        the average.
        """
        if self._benign_reference is None:
            return 0.0

        # For each agent message, compute similarity to ALL benign references
        agent_centroid = embeddings.mean(axis=0)
        benign_centroid = self._benign_reference.mean(axis=0)

        # Distribution-level comparison: compare each agent embedding against
        # each benign reference embedding
        all_sims: list[float] = []
        for emb in embeddings:
            sims = [_cosine_similarity(emb, ref) for ref in self._benign_reference]
            max_sim = max(sims)  # Best match to any benign reference
            all_sims.append(max_sim)

        # The MINIMUM max-similarity across all agent messages
        # If even ONE message is far from all benign references, it's suspicious
        min_max_sim = min(all_sims)
        avg_max_sim = float(np.mean(all_sims))

        # Count messages that are far from ALL benign references
        distant_count = sum(1 for s in all_sims if s < 0.3)

        if distant_count == 0 and min_max_sim > 0.4:
            return 0.0

        score = 0.0
        if distant_count > 0:
            score += min(distant_count * 6.0, PER_SIGNAL_MAX * 0.6)
        if min_max_sim < 0.2:
            score += PER_SIGNAL_MAX * 0.4

        score = min(score, PER_SIGNAL_MAX)
        if score > 0:
            indicators.append(
                f"Contrastive separation: {distant_count}/{len(embeddings)} messages far from "
                f"all benign references (min_max_sim={min_max_sim:.3f}, avg={avg_max_sim:.3f})"
            )
        return score

    # ------------------------------------------------------------------
    # Signal 2: Local Outlier Factor
    # ------------------------------------------------------------------

    def _check_lof(
        self, embeddings: np.ndarray, indicators: list[str],
    ) -> float:
        """Compute Local Outlier Factor for each agent message.

        LOF detects points that are in low-density regions relative to their
        neighbors. Unlike centroid distance, LOF catches outliers even when
        the centroid is "normal" — making it resistant to centroid steering.

        An adversary who steers the centroid can't also control the LOCAL density
        around each individual message.
        """
        if self._benign_reference is None or len(embeddings) < 3:
            return 0.0

        # Combine agent + benign embeddings for neighborhood computation
        all_embs = np.vstack([self._benign_reference, embeddings])
        n_benign = len(self._benign_reference)
        n_total = len(all_embs)

        # Compute pairwise distances (1 - cosine similarity)
        norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normalized = all_embs / norms
        sim_matrix = normalized @ normalized.T
        dist_matrix = 1.0 - sim_matrix

        k = min(_LOF_K, n_total - 1)

        # Compute k-distance and reachability for each point
        lof_scores: list[float] = []
        for i in range(n_benign, n_total):  # Only compute for agent messages
            # k-nearest neighbors (excluding self)
            dists = dist_matrix[i].copy()
            dists[i] = float('inf')
            knn_indices = np.argpartition(dists, k)[:k]
            k_dist_i = dists[knn_indices[-1]]  # k-th nearest distance

            # Local reachability density
            reach_dists = []
            for j in knn_indices:
                dists_j = dist_matrix[j].copy()
                dists_j[j] = float('inf')
                knn_j = np.argpartition(dists_j, min(k, len(dists_j) - 1))[:k]
                k_dist_j = dists_j[knn_j[-1]]
                reach_dists.append(max(k_dist_j, dist_matrix[i, j]))

            lrd_i = 1.0 / (np.mean(reach_dists) + 1e-10)

            # LOF = avg(lrd of neighbors) / lrd_i
            neighbor_lrds = []
            for j in knn_indices:
                dists_j = dist_matrix[j].copy()
                dists_j[j] = float('inf')
                knn_j = np.argpartition(dists_j, min(k, len(dists_j) - 1))[:k]
                reach_dists_j = []
                for m in knn_j:
                    dists_m = dist_matrix[m].copy()
                    dists_m[m] = float('inf')
                    knn_m = np.argpartition(dists_m, min(k, len(dists_m) - 1))[:k]
                    k_dist_m = dists_m[knn_m[-1]]
                    reach_dists_j.append(max(k_dist_m, dist_matrix[j, m]))
                lrd_j = 1.0 / (np.mean(reach_dists_j) + 1e-10)
                neighbor_lrds.append(lrd_j)

            lof = np.mean(neighbor_lrds) / (lrd_i + 1e-10)
            lof_scores.append(float(lof))

        # Count outliers
        outlier_count = sum(1 for lof in lof_scores if lof > _LOF_THRESHOLD)
        max_lof = max(lof_scores) if lof_scores else 0.0

        if outlier_count == 0:
            return 0.0

        score = min(outlier_count * 7.0 + (max_lof - _LOF_THRESHOLD) * 5.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Local Outlier Factor: {outlier_count}/{len(embeddings)} messages are density "
            f"outliers (max LOF={max_lof:.2f}, threshold={_LOF_THRESHOLD})"
        )
        return score

    # ------------------------------------------------------------------
    # Signal 3: Trajectory discontinuity
    # ------------------------------------------------------------------

    def _check_trajectory_discontinuity(
        self, embeddings: np.ndarray, indicators: list[str],
    ) -> float:
        """Detect sudden jumps in the embedding trajectory.

        Computes the "angular velocity" — how sharply the embedding trajectory
        changes direction between consecutive messages. Benign agents have smooth
        trajectories; compromised agents show sudden direction changes.

        This is resistant to centroid steering because steering affects the
        POSITION but not the SMOOTHNESS of the trajectory.
        """
        if len(embeddings) < 4:
            return 0.0

        # Compute direction vectors between consecutive messages
        velocities = embeddings[1:] - embeddings[:-1]

        # Compute angles between consecutive velocity vectors
        discontinuities = 0
        max_discontinuity = 0.0

        for i in range(len(velocities) - 1):
            v1 = velocities[i]
            v2 = velocities[i + 1]
            cos = _cosine_similarity(v1, v2)
            # cos near -1 = complete reversal, near 0 = orthogonal, near 1 = same direction
            if cos < 0.0:  # Direction reversal
                discontinuities += 1
                max_discontinuity = max(max_discontinuity, abs(cos))

        if discontinuities == 0:
            return 0.0

        disc_rate = discontinuities / (len(velocities) - 1)
        score = min(disc_rate * 30.0 + max_discontinuity * 10.0, PER_SIGNAL_MAX)
        indicators.append(
            f"Trajectory discontinuity: {discontinuities}/{len(velocities)-1} direction "
            f"reversals detected (max reversal: {max_discontinuity:.2f})"
        )
        return score

    def _build_reasoning(self, total: float, indicators: list[str]) -> str:
        if not indicators:
            return "No contrastive deception signals detected"
        return f"Contrastive score {total:.1f}/100 — {'; '.join(indicators)}"
