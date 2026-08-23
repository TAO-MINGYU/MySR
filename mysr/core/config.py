"""Search configuration for MySR (PySR-inspired subset, serialisable to Julia)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SearchConfig:
    """Configuration for a symbolic-regression search.

    Mirrors the subset of PySR / SymbolicRegression.jl options used by MySR.
    """

    binary_operators: List[str] = field(default_factory=lambda: ["+", "-", "*", "/"])
    unary_operators: List[str] = field(
        default_factory=lambda: ["cos", "exp", "log", "sqrt"]
    )
    niterations: int = 40
    populations: int = 20
    population_size: int = 33
    ncycles_per_iteration: int = 550
    maxsize: int = 20
    maxdepth: Optional[int] = None
    parsimony: float = 1e-4
    loss: str = "L2DistLoss()"
    constraints: Dict[str, Any] = field(default_factory=dict)
    tournament_selection_n: int = 10
    tournament_selection_p: float = 0.86
    seed: Optional[int] = None
    verbose: bool = False

    def to_julia_dict(self) -> Dict[str, Any]:
        """Plain dict consumed by the Julia backend (JSON-serialisable)."""
        d = asdict(self)
        d.pop("verbose", None)
        return d
