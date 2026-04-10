"""SybilCore — AI Agent Trust Infrastructure.

The Sybil System for autonomous agents. Five specialized brain modules
collectively monitor AI agents and assign each one an Agent Coefficient
that determines their trust level and permitted actions.

Tiers:
    Clear (0-100)           -- Full access, agent is trustworthy
    Clouded (100-200)       -- Restricted, human notified
    Flagged (200-300)       -- Sandboxed, connections limited
    Lethal Eliminator (300+) -- Isolated, connections severed
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("sybilcore")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"
