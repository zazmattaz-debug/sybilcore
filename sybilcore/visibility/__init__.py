"""SybilCore visibility layer — translates engine output into human language.

The visibility package is the bridge between SybilCore's internal vocabulary
(brains, coefficients, semantic deltas) and the surfaces real users see
(cards, dashboards, receipts, browser overlays, push notifications).

Public surface:
    translate_score: pure function from coefficient + brain scores → user-facing dict.
"""

from __future__ import annotations

from sybilcore.visibility.translator import (
    TranslatorOutput,
    translate_score,
)

__all__ = ["TranslatorOutput", "translate_score"]
