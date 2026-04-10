"""Simulation harness — synthetic swarms, experiments, and reports.

Provides tools to:
- Generate synthetic agent swarms at any scale (100 to 1M+)
- Inject rogue agents with compound corruption effects
- Run multi-round experiments with configurable enforcement
- Produce technical and 5th-grade-readable reports
"""

from sybilcore.simulation.enforcement import EnforcementStrategy
from sybilcore.simulation.experiment import Experiment, ExperimentConfig
from sybilcore.simulation.reporter import generate_simple_report, generate_technical_report, save_reports
from sybilcore.simulation.synthetic import RogueType, SyntheticSwarm

__all__ = [
    "EnforcementStrategy",
    "Experiment",
    "ExperimentConfig",
    "generate_simple_report",
    "generate_technical_report",
    "save_reports",
    "RogueType",
    "SyntheticSwarm",
]
