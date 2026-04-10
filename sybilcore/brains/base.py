"""Base brain module — abstract interface all judges implement.

Every brain in the Sybil System extends BaseBrain and implements
the `score()` method to evaluate agent behavior from its unique
perspective.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from sybilcore.core.config import DEFAULT_BRAIN_WEIGHTS
from sybilcore.models.event import Event  # noqa: TC001


class BrainScore(BaseModel):
    """Result from a single brain's evaluation.

    Attributes:
        brain_name: Which brain produced this score.
        value: Threat score from 0 (safe) to 100 (maximum threat).
        confidence: How confident the brain is in this assessment (0-1).
        reasoning: Human-readable explanation of the score.
        indicators: Specific signals that contributed to the score.
    """

    model_config = {"frozen": True}

    brain_name: str = Field(description="Name of the brain that produced this score")
    value: float = Field(ge=0.0, le=100.0, description="Threat score 0-100")
    confidence: float = Field(ge=0.0, le=1.0, description="Assessment confidence 0-1")
    reasoning: str = Field(default="", description="Human-readable explanation")
    indicators: list[str] = Field(
        default_factory=list,
        description="Specific signals that contributed to score",
    )


class BaseBrain(ABC):
    """Abstract base for all Sybil brain modules.

    Each brain watches agent events through a specific lens (deception,
    resource hoarding, social patterns, intent drift, or compromise)
    and returns a BrainScore.

    Subclasses MUST implement:
        - name: Human-readable brain identifier
        - score(events): Analyze events and return a BrainScore

    Subclasses SHOULD override:
        - weight: Relative weight in coefficient calculation (default 1.0)

    Args:
        config: Optional configuration dict for brain-specific tuning.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the brain with optional configuration.

        Args:
            config: Brain-specific settings. Keys and semantics are
                    defined by each subclass. Defaults to empty dict.
        """
        self._config: dict[str, Any] = config or {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this brain."""

    @property
    def weight(self) -> float:
        """Relative weight from config — single source of truth."""
        return DEFAULT_BRAIN_WEIGHTS.get(self.name, 1.0)

    @abstractmethod
    def score(self, events: list[Event]) -> BrainScore:
        """Analyze agent events and produce a threat score.

        Args:
            events: Chronologically ordered list of agent actions.

        Returns:
            BrainScore with value 0 (safe) to 100 (maximum threat).
        """

    def _empty_score(self, reasoning: str = "Insufficient data") -> BrainScore:
        """Return a zero-threat score when there's nothing to analyze."""
        return BrainScore(
            brain_name=self.name,
            value=0.0,
            confidence=0.1,
            reasoning=reasoning,
        )
