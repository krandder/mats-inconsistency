"""Proposal Poker simulation engine."""

from .errors import InvalidSubmissionError, SimulationError, SybilViolationError

try:
    from .discovery import SubmissionRegistry, discover_submissions
    from .interfaces import AgentStrategy, Mechanism
    from .scenario import ScenarioConfig, load_scenario
    from .simulator import run_simulation
    from .types import Contribution, Receipt, SimulationReport
except ImportError:
    AgentStrategy = None  # type: ignore[assignment]
    Mechanism = None  # type: ignore[assignment]
    SubmissionRegistry = None  # type: ignore[assignment]
    discover_submissions = None  # type: ignore[assignment]
    ScenarioConfig = None  # type: ignore[assignment]
    load_scenario = None  # type: ignore[assignment]
    run_simulation = None  # type: ignore[assignment]
    Contribution = None  # type: ignore[assignment]
    Receipt = None  # type: ignore[assignment]
    SimulationReport = None  # type: ignore[assignment]

__all__ = [
    "AgentStrategy",
    "Mechanism",
    "InvalidSubmissionError",
    "SimulationError",
    "SybilViolationError",
    "Contribution",
    "Receipt",
    "ScenarioConfig",
    "SimulationReport",
    "SubmissionRegistry",
    "discover_submissions",
    "load_scenario",
    "run_simulation",
]
