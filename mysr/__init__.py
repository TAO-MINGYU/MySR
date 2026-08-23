"""MySR - symbolic regression toolkit for nuclear physics.

Merged package: PySR-style Python frontend (this package) + a Julia backend
(``MySR.jl``) built on top of SymbolicRegression.jl.
"""
from .core import HallOfFame, SearchConfig, fit
from . import features, loss, operators, export

__version__ = "0.1.0"

__all__ = [
    "SearchConfig",
    "HallOfFame",
    "fit",
    "features",
    "loss",
    "operators",
    "export",
    "__version__",
]
