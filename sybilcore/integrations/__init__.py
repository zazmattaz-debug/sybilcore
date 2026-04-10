"""Integrations — adapters for popular AI agent frameworks.

Drop-in monitors for LangChain, CrewAI, MiroFish, and other frameworks
that convert framework-specific events into SybilCore Events for brain
analysis.
"""

from sybilcore.integrations.mirofish import SybilCoreMiroFishAdapter

__all__ = [
    "SybilCoreMiroFishAdapter",
]
