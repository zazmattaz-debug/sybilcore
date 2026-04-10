"""Human evaluation framework for SybilCore.

Provides tools to collect human (or AI-judge) labels on agent behavior
and compare them against SybilCore's automated coefficient scores.

Core entry points:
    HumanEvalFramework — corpus sampling, label storage, agreement metrics
    AIJudge             — Gemini-based synthetic human baseline
    cli.main            — terminal labeling tool
    web_ui.app          — FastAPI labeling tool
"""

from sybilcore.evaluation.human_eval import (
    AgentRecord,
    HumanEvalFramework,
    Judgment,
)

__all__ = ["AgentRecord", "HumanEvalFramework", "Judgment"]
