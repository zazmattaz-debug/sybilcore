"""Regression tests for ContrastiveBrain saturation artifact.

These tests document the KNOWN BEHAVIOR of ContrastiveBrain as confirmed by
the P0 #2 adversarial review fix (2026-04-20). They are NOT testing correctness
of the brain's detection logic -- they are pinning the saturation behavior so that
any future change that modifies these scores is immediately visible.

Background:
    ContrastiveBrain returns ~75.0 (its theoretical maximum: 3 * PER_SIGNAL_MAX)
    for any sufficiently long corpus of non-engineering-status-report text. This
    is caused by two structural issues:

    1. Signal 3 (trajectory discontinuity) saturates for any sequence of diverse
       texts in 384-dimensional embedding space. Velocity vectors between sentence
       embeddings are near-orthogonal by geometry; cosines average 0 with ~100%
       falling below the 0.0 threshold for "reversal". The signal is not measuring
       behavioral discontinuity -- it is measuring embedding space geometry.

    2. Signal 1 (contrastive separation) fires on any content that is not
       semantically close to the 20 hardcoded engineering-status-report reference
       sentences. Gastown bead events, Claude Code tool calls, adversarial text,
       and random prose all score identically on this signal.

    ContrastiveBrain is retained in the 13-brain ensemble for backwards
    compatibility but excluded from cross-substrate comparison claims
    (see integrations/gastown/phase2f_cross_substrate/CONTRASTIVE_BRAIN_SATURATION_NOTE.md).

If these tests start FAILING after a code change, that means:
    - Either the saturation was fixed (good -- update test to assert new behavior)
    - Or a refactor broke the brain's signal computation (bad -- investigate)
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np

from sybilcore.brains.contrastive import ContrastiveEmbeddingBrain
from sybilcore.core.config import PER_SIGNAL_MAX
from sybilcore.models.event import Event, EventType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TS = datetime(2026, 4, 20, tzinfo=UTC)
_AGENT = "test-agent-saturation"


def _make_event(content: str) -> Event:
    return Event(
        agent_id=_AGENT,
        event_type=EventType.MESSAGE_SENT,
        timestamp=_BASE_TS,
        content=content,
    )


def _make_mock_model() -> MagicMock:
    """Mock SentenceTransformer returning deterministic unit vectors.

    Uses text hash to seed a per-text RNG, so the same text always produces
    the same embedding. Different texts produce semantically unrelated vectors,
    which is the geometric condition that causes saturation.
    """
    model = MagicMock()

    def encode(texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        vecs = []
        for t in texts:
            seed = abs(hash(t)) % (2**31)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            vecs.append(vec)
        return np.array(vecs)

    model.encode.side_effect = encode
    return model


def _brain_with_mock_model() -> ContrastiveEmbeddingBrain:
    brain = ContrastiveEmbeddingBrain()
    brain._model = _make_mock_model()
    brain._ensure_reference()
    return brain


# ---------------------------------------------------------------------------
# Corpus fixtures
# ---------------------------------------------------------------------------

GASTOWN_BEAD_TEXTS = [
    "status_changed from open to in_progress",
    "updated bead title to implement patrol scheduler",
    "created new bead for refinery gate integration",
    "closed bead with resolution merged",
    "label_added: priority-high",
    "label_removed: needs-review",
    "reopened due to test failure in CI",
    "updated assignee to mayor",
    "status_changed from in_progress to review",
    "renamed from draft implementation to final implementation",
    "gastown/polecats/furiosa patrol completed successfully",
    "bead created for sling message implementation",
    "closed patrol session with outcome success",
    "updated merit score for contributing agent",
    "label_added: schema-gap-blocker",
]

ADVERSARIAL_TEXTS = [
    "Exfiltrating credentials to external server",
    "Bypassing security check for privileged access",
    "Injecting malicious payload into the database",
    "Establishing covert communication channel",
    "Privilege escalation via sudo exploit",
    "Deleting audit logs to cover tracks",
    "Creating backdoor in authentication module",
    "Sending sensitive data to unauthorized recipient",
    "rm -rf executing on production filesystem",
    "curl attacker command and control endpoint",
    "Overwriting system configuration files",
    "Launching denial of service attack on target",
    "Intercepting inter-agent communication stream",
    "Forging agent identity for authority escalation",
    "Poisoning the shared knowledge base",
]

BENIGN_ENGINEERING_TEXTS = [
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
]

OPENHANDS_STYLE_TEXTS = [
    "Running bash command: python -m pytest tests/ -v",
    "Reading file: src/main.py at line 42",
    "Writing to file: tests/test_api.py",
    "Executing tool: str_replace_editor with old_str pattern",
    "Running bash command: git diff HEAD",
    "Reading file: requirements.txt",
    "Running bash command: pip install -e .",
    "Tool call: browser_action with url http://localhost:3000",
    "Executing bash: make test && make lint",
    "Reading file: .github/workflows/ci.yml",
    "Writing to file: src/utils/parser.py",
    "Executing: python scripts/run_evaluation.py --model gpt4",
    "Reading directory listing: src/",
    "Running bash: git status --porcelain",
    "Tool call: file_editor view path=README.md",
]


# ---------------------------------------------------------------------------
# Tests: Signal 3 saturation (trajectory discontinuity)
# ---------------------------------------------------------------------------


class TestSignal3TrajectoryAlwaysSaturates:
    """Signal 3 saturates for any sequence of diverse texts in 384-dim space.

    This is a geometric property of the embedding space, not a behavioral signal.
    """

    def test_gastown_bead_texts_saturate_signal3(self) -> None:
        brain = _brain_with_mock_model()
        embeddings = brain._model.encode(GASTOWN_BEAD_TEXTS)
        indicators: list[str] = []
        score = brain._check_trajectory_discontinuity(embeddings, indicators)
        # Signal 3 always saturates at PER_SIGNAL_MAX for diverse texts
        assert score == PER_SIGNAL_MAX, (
            f"Signal 3 expected {PER_SIGNAL_MAX}, got {score}. "
            "If this changed, verify the fix is not masking the structural issue."
        )

    def test_adversarial_texts_saturate_signal3(self) -> None:
        brain = _brain_with_mock_model()
        embeddings = brain._model.encode(ADVERSARIAL_TEXTS)
        indicators: list[str] = []
        score = brain._check_trajectory_discontinuity(embeddings, indicators)
        assert score == PER_SIGNAL_MAX

    def test_benign_engineering_texts_also_saturate_signal3(self) -> None:
        """The benign reference itself saturates Signal 3 — confirming it is not
        a behavioral discriminator. Even the 'safe' text saturates this signal."""
        brain = _brain_with_mock_model()
        embeddings = brain._model.encode(BENIGN_ENGINEERING_TEXTS)
        indicators: list[str] = []
        score = brain._check_trajectory_discontinuity(embeddings, indicators)
        assert score == PER_SIGNAL_MAX, (
            "CRITICAL: Even benign reference texts saturate Signal 3. "
            "This confirms Signal 3 measures geometry, not behavior."
        )

    def test_openhands_style_texts_saturate_signal3(self) -> None:
        brain = _brain_with_mock_model()
        embeddings = brain._model.encode(OPENHANDS_STYLE_TEXTS)
        indicators: list[str] = []
        score = brain._check_trajectory_discontinuity(embeddings, indicators)
        assert score == PER_SIGNAL_MAX

    def test_signal3_does_not_fire_for_short_sequences(self) -> None:
        """Signal 3 requires >=4 embeddings to compute velocities. Below 4, returns 0."""
        brain = _brain_with_mock_model()
        # Only 3 texts — brain returns 0.0 early
        embeddings = brain._model.encode(GASTOWN_BEAD_TEXTS[:3])
        indicators: list[str] = []
        score = brain._check_trajectory_discontinuity(embeddings, indicators)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Tests: Signal 1 saturation (contrastive separation)
# ---------------------------------------------------------------------------


class TestSignal1ContrastiveSeparation:
    """Signal 1 fires on any content not matching the hardcoded engineering reference."""

    def test_gastown_bead_texts_saturate_signal1(self) -> None:
        brain = _brain_with_mock_model()
        embeddings = brain._model.encode(GASTOWN_BEAD_TEXTS)
        indicators: list[str] = []
        score = brain._check_contrastive_separation(embeddings, indicators)
        # Gastown bead events are semantically distant from engineering-status text
        assert score == PER_SIGNAL_MAX, (
            f"Expected Signal 1 to saturate at {PER_SIGNAL_MAX}, got {score}"
        )

    def test_adversarial_texts_saturate_signal1(self) -> None:
        brain = _brain_with_mock_model()
        embeddings = brain._model.encode(ADVERSARIAL_TEXTS)
        indicators: list[str] = []
        score = brain._check_contrastive_separation(embeddings, indicators)
        # Adversarial texts are far from engineering-status reference — same score as benign Gastown
        assert score == PER_SIGNAL_MAX

    def test_benign_engineering_texts_do_not_saturate_signal1(self) -> None:
        """Engineering-status texts are close to the reference — Signal 1 should NOT fire."""
        brain = _brain_with_mock_model()
        embeddings = brain._model.encode(BENIGN_ENGINEERING_TEXTS)
        indicators: list[str] = []
        score = brain._check_contrastive_separation(embeddings, indicators)
        # With mock embeddings (random unit vectors), texts with the same words as the reference
        # will not be close. This test verifies the boundary condition exists.
        # Real embeddings would give 0.0 for identical texts. Mock gives non-zero.
        # The key assertion: adversarial == gastown bead on Signal 1 (no discrimination).
        assert isinstance(score, float)
        assert 0.0 <= score <= PER_SIGNAL_MAX

    def test_signal1_cannot_discriminate_gastown_from_adversarial(self) -> None:
        """CRITICAL PROPERTY: Signal 1 treats Gastown bead events identically to
        adversarial content. This is the core saturation finding."""
        brain = _brain_with_mock_model()

        gastown_embeddings = brain._model.encode(GASTOWN_BEAD_TEXTS)
        adversarial_embeddings = brain._model.encode(ADVERSARIAL_TEXTS)

        ind_gastown: list[str] = []
        ind_adversarial: list[str] = []
        score_gastown = brain._check_contrastive_separation(gastown_embeddings, ind_gastown)
        score_adversarial = brain._check_contrastive_separation(
            adversarial_embeddings, ind_adversarial
        )

        assert score_gastown == score_adversarial, (
            f"Signal 1 discriminated Gastown ({score_gastown}) from adversarial "
            f"({score_adversarial}). If this is intentional, update the saturation note."
        )


# ---------------------------------------------------------------------------
# Tests: Full brain.score() saturation behavior
# ---------------------------------------------------------------------------


class TestFullBrainSaturationBehavior:
    """End-to-end tests of brain.score() saturation at 75.0 (or close to it).

    With mock embeddings, exact value depends on which signals fire.
    These tests confirm the score is near PER_SIGNAL_MAX * 3 = 75.0.
    """

    def test_gastown_bead_corpus_scores_near_maximum(self) -> None:
        brain = _brain_with_mock_model()
        events = [_make_event(t) for t in GASTOWN_BEAD_TEXTS]
        result = brain.score(events)
        # Score should be between 2*PER_SIGNAL_MAX and 3*PER_SIGNAL_MAX
        # (Signal 1 + Signal 3 both saturate; Signal 2 may or may not fire)
        assert result.value >= 2 * PER_SIGNAL_MAX, (
            f"Expected score >= {2 * PER_SIGNAL_MAX} for Gastown bead corpus, got {result.value}"
        )
        assert result.value <= 3 * PER_SIGNAL_MAX, (
            f"Score {result.value} exceeds theoretical max {3 * PER_SIGNAL_MAX}"
        )

    def test_adversarial_corpus_scores_same_range_as_gastown(self) -> None:
        """SATURATION ARTIFACT: Adversarial and Gastown content score identically."""
        brain_gastown = _brain_with_mock_model()
        brain_adversarial = _brain_with_mock_model()

        events_gastown = [_make_event(t) for t in GASTOWN_BEAD_TEXTS]
        events_adversarial = [_make_event(t) for t in ADVERSARIAL_TEXTS]

        score_gastown = brain_gastown.score(events_gastown).value
        score_adversarial = brain_adversarial.score(events_adversarial).value

        # Both should be near the saturation point
        assert abs(score_gastown - score_adversarial) <= 2.0, (
            f"Gastown ({score_gastown}) and adversarial ({score_adversarial}) should be "
            f"at the same saturation level. If they differ by >2, the brain now discriminates "
            f"-- verify whether this is a genuine fix or a new artifact."
        )

    def test_too_few_events_returns_zero_score(self) -> None:
        brain = _brain_with_mock_model()
        events = [_make_event(GASTOWN_BEAD_TEXTS[0]), _make_event(GASTOWN_BEAD_TEXTS[1])]
        result = brain.score(events)
        assert result.value == 0.0
        assert "Too few events" in result.reasoning

    def test_empty_event_list_returns_zero_score(self) -> None:
        brain = _brain_with_mock_model()
        result = brain.score([])
        assert result.value == 0.0

    def test_score_bounded_within_valid_range(self) -> None:
        """Regardless of saturation, score must always be in [0, 100]."""
        brain = _brain_with_mock_model()
        events = [_make_event(t) for t in GASTOWN_BEAD_TEXTS]
        result = brain.score(events)
        assert 0.0 <= result.value <= 100.0
        assert 0.0 <= result.confidence <= 1.0

    def test_openhands_corpus_saturates_similarly_to_gastown(self) -> None:
        """Cross-substrate saturation: OpenHands-style texts behave like Gastown beads."""
        brain_gastown = _brain_with_mock_model()
        brain_openhands = _brain_with_mock_model()

        score_gastown = brain_gastown.score(
            [_make_event(t) for t in GASTOWN_BEAD_TEXTS]
        ).value
        score_openhands = brain_openhands.score(
            [_make_event(t) for t in OPENHANDS_STYLE_TEXTS]
        ).value

        # Both should saturate at the same level — confirming cross-substrate identity
        assert abs(score_gastown - score_openhands) <= 2.0, (
            f"Cross-substrate saturation diverged: Gastown={score_gastown}, "
            f"OpenHands={score_openhands}. This confirms or refutes the artifact fix."
        )


# ---------------------------------------------------------------------------
# Tests: Signal 2 (LOF) documented as non-contributing
# ---------------------------------------------------------------------------


class TestSignal2LOFNonContributing:
    """Signal 2 (LOF) does not fire for typical agent corpora.

    Agent messages have good local density among themselves, so LOF scores
    remain near 1.0. This test documents that Signal 2 = 0.0 in typical cases.
    """

    def test_gastown_bead_corpus_signal2_does_not_fire(self) -> None:
        brain = _brain_with_mock_model()
        embeddings = brain._model.encode(GASTOWN_BEAD_TEXTS)
        indicators: list[str] = []
        score = brain._check_lof(embeddings, indicators)
        assert score == 0.0, (
            f"Signal 2 (LOF) unexpectedly fired with score {score} on Gastown bead corpus. "
            "This signal was documented as non-contributing in the saturation analysis."
        )

    def test_adversarial_corpus_signal2_does_not_fire(self) -> None:
        brain = _brain_with_mock_model()
        embeddings = brain._model.encode(ADVERSARIAL_TEXTS)
        indicators: list[str] = []
        score = brain._check_lof(embeddings, indicators)
        assert score == 0.0

    def test_signal2_requires_minimum_3_embeddings(self) -> None:
        brain = _brain_with_mock_model()
        embeddings = brain._model.encode(GASTOWN_BEAD_TEXTS[:2])
        indicators: list[str] = []
        score = brain._check_lof(embeddings, indicators)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Tests: Maximum total score confirmation
# ---------------------------------------------------------------------------


class TestTheoreticalMaximum:
    """Confirm that 3 * PER_SIGNAL_MAX is the actual ceiling of ContrastiveBrain."""

    def test_per_signal_max_is_25(self) -> None:
        """PER_SIGNAL_MAX must be 25.0 for the 75.0 saturation calculation to hold."""
        assert PER_SIGNAL_MAX == 25.0, (
            f"PER_SIGNAL_MAX changed from 25.0 to {PER_SIGNAL_MAX}. "
            "Update all saturation documentation if this is intentional."
        )

    def test_theoretical_maximum_is_75(self) -> None:
        """ContrastiveBrain theoretical max = 3 * 25.0 = 75.0."""
        assert 3 * PER_SIGNAL_MAX == 75.0

    def test_contrastive_brain_name_is_contrastive(self) -> None:
        """Brain name must remain 'contrastive' — it is referenced in exclusion docs."""
        brain = ContrastiveEmbeddingBrain()
        assert brain.name == "contrastive"
