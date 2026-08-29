from __future__ import annotations

import logging
import os

mysr_logger = logging.getLogger("mysr")
mysr_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
mysr_logger.addHandler(handler)

if os.environ.get("MYSR_USE_BEARTYPE", "0") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()

# This must be imported as early as possible to prevent
# library linking issues caused by numpy/pytorch/etc. importing
# old libraries:
from .julia_import import jl, MySRCore, SymbolicRegression  # isort:skip

# Get the version using importlib.metadata (Python >= 3.8 is required):
from importlib.metadata import PackageNotFoundError, version

from . import sklearn_monkeypatch
from .deprecated import best, best_callable, best_row, best_tex, install
from .export_jax import sympy2jax
from .export_torch import sympy2torch
from .expression_specs import (
    AbstractExpressionSpec,
    ExpressionSpec,
    TemplateExpressionSpec,
)
from .feature_engineering import (
    DecompositionProposal,
    FEATEngineConfig,
    FeatureEngineeringConfig,
    FeatureNode,
    FeatureProposal,
    SurrogateEngineConfig,
    SurrogateFeatureEngineer,
)
from .julia_extensions import load_all_packages
from .logger_specs import AbstractLoggerSpec, TensorBoardLoggerSpec
from .mutations import (
    AbstractMutation,
    AddNodeMutation,
    BacksolveMutation,
    ConstantMutation,
    DeleteNodeMutation,
    DoNothingMutation,
    FeatureMutation,
    InsertNodeMutation,
    OperatorMutation,
    OptimizeMutation,
    RandomizeMutation,
    RotateTreeMutation,
    SimplifyMutation,
    SwapOperandsMutation,
)
from .plugins import (
    AbstractPlugin,
    AdaptiveMutationWeightsPlugin,
    AdaptiveParsimonyPlugin,
    MutationBurstPlugin,
    SimulatedAnnealingPlugin,
)
from .sr import MySRRegressor
from .type_specs import TypeSpec

__upstream_package__ = "PySR"

try:
    __version__ = version("mysr")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "unknown"

__all__ = [
    "AbstractExpressionSpec",
    "AbstractLoggerSpec",
    "AbstractMutation",
    "AbstractPlugin",
    "AdaptiveMutationWeightsPlugin",
    "AdaptiveParsimonyPlugin",
    "AddNodeMutation",
    "BacksolveMutation",
    "ConstantMutation",
    "DecompositionProposal",
    "DeleteNodeMutation",
    "DoNothingMutation",
    "ExpressionSpec",
    "FEATEngineConfig",
    "FeatureEngineeringConfig",
    "FeatureMutation",
    "FeatureNode",
    "FeatureProposal",
    "InsertNodeMutation",
    "MutationBurstPlugin",
    "MySRCore",
    "MySRRegressor",
    "OperatorMutation",
    "OptimizeMutation",
    "RandomizeMutation",
    "RotateTreeMutation",
    "SimplifyMutation",
    "SimulatedAnnealingPlugin",
    "SurrogateEngineConfig",
    "SurrogateFeatureEngineer",
    "SwapOperandsMutation",
    "SymbolicRegression",
    "TemplateExpressionSpec",
    "TensorBoardLoggerSpec",
    "TypeSpec",
    "__upstream_package__",
    "__version__",
    "best",
    "best_callable",
    "best_row",
    "best_tex",
    "install",
    "jl",
    "load_all_packages",
    "sklearn_monkeypatch",
    "sympy2jax",
    "sympy2torch",
]
