"""AI Feynman-inspired automatic input feature engineering.

This module is an independent implementation for MySR. It borrows the ideas of
surrogate interpolation, symmetry tests, separability, compositionality, and
recursive dimensional reduction from AI Feynman, but it does not copy or import
AI Feynman source code. MySRCore remains responsible for the final SR search.

References: Udrescu and Tegmark, "AI Feynman" (Science Advances, 2020);
Udrescu et al., "AI Feynman 2.0" (NeurIPS, 2020); and the MIT-licensed official
implementation at https://github.com/SJ001/AI-Feynman.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar  # type: ignore
from sklearn.inspection import permutation_importance  # type: ignore
from sklearn.metrics import r2_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.neural_network import MLPRegressor  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.utils import check_random_state  # type: ignore

from .dimensions import (
    DimensionVector,
    dimension_add,
    dimension_equal,
    dimension_is_zero,
    dimension_scale,
    dimension_sub,
    dimension_to_string,
)

CandidateOperator = Literal["sub", "add", "mul", "div"]
CompositionOperator = Literal[
    "sub",
    "add",
    "mul",
    "div",
    "normalized_sub",
    "hypot",
    "log_ratio",
    "exp_ratio",
    "sin_ratio",
    "cos_ratio",
]
UnaryOperator = Literal[
    "square",
    "cube",
    "sqrt_abs",
    "reciprocal",
    "abs",
    "log_abs",
    "exp",
    "sin",
    "cos",
]


@dataclass(frozen=True)
class SurrogateEngineConfig:
    """Configuration for the AI Feynman-inspired surrogate branch."""

    enabled: bool = True
    candidate_operators: tuple[CandidateOperator, ...] = (
        "sub",
        "add",
        "mul",
        "div",
    )
    # Operators used by the bounded recursive composition search.  The original
    # four operators remain available for backwards compatibility; the additional
    # families mirror common AI-Feynman reductions without enabling arbitrary
    # expression enumeration.
    composition_operators: tuple[CompositionOperator, ...] = ()
    candidate_unary_operators: tuple[UnaryOperator, ...] = (
        "square",
        "cube",
        "sqrt_abs",
        "reciprocal",
        "abs",
        "log_abs",
        "exp",
        "sin",
        "cos",
    )
    enable_pairwise_symmetry: bool = True
    enable_unary_composition: bool = True
    enable_recursive_composition: bool = True
    enable_separability: bool = True
    enable_parameterized_symmetry: bool = True
    enable_power_composition: bool = True
    power_exponents: tuple[float, ...] = (0.5, 1.5, 2.5, -0.5)
    surrogate_min_r2: float = 0.80
    surrogate_ensemble_size: int = 3
    surrogate_stability_min_fraction: float = 2.0 / 3.0
    construction_fraction: float = 0.15
    validation_fraction: float = 0.15
    invariance_min_score: float = 0.90
    separability_min_score: float = 0.92
    composition_min_score: float = 0.35
    composition_min_improvement: float = 0.05
    high_confidence_composition_score: float = 0.98
    relevance_min_fraction: float = 0.05
    invariance_relative_tolerance: float = 0.01
    n_perturbations: int = 4
    perturbation_scales: tuple[float, ...] = (0.18, 0.35, 0.55)
    max_candidate_pairs: int = 256
    max_parameterized_candidates: int = 24
    parameter_search_min_score: float = 0.40
    parameter_min: float = 0.125
    parameter_max: float = 8.0
    parameter_grid_size: int = 9
    parameter_identity_tolerance: float = 0.05
    parameter_boundary_tolerance: float = 0.04
    parameter_min_improvement: float = 0.04
    max_composition_candidates: int = 256
    max_separability_partitions: int = 32
    max_subset_size: int = 3
    max_composition_depth: int = 3
    composition_beam_width: int = 24
    max_generated_features: int = 8
    complexity_penalty: float = 0.015
    exp_input_limit: float = 20.0
    hidden_layer_sizes: tuple[int, ...] = (32, 32)
    max_iter: int = 1000


@dataclass(frozen=True)
class PairSymmetryCandidate:
    """Internal parameterized pairwise symmetry hypothesis."""

    operator: CandidateOperator
    left_index: int
    right_index: int
    parameter: float = 1.0
    parameterized: bool = False


@dataclass(frozen=True)
class FEATEngineConfig:
    """Configuration for the lightweight MySR-native FEAT-like branch."""

    enabled: bool = False
    population_size: int = 24
    generations: int = 8
    max_evaluations: int = 240
    max_depth: int = 3
    max_bundle_size: int = 4
    max_generated_features: int = 6
    max_seed_nodes: int = 96
    binary_operators: tuple[CandidateOperator, ...] = (
        "add",
        "sub",
        "mul",
        "div",
    )
    unary_operators: tuple[UnaryOperator, ...] = (
        "square",
        "cube",
        "sqrt_abs",
        "abs",
        "log_abs",
        "sin",
        "cos",
    )
    construction_fraction: float = 0.20
    validation_fraction: float = 0.20
    ridge_alpha: float = 1.0e-3
    crossover_rate: float = 0.45
    mutation_rate: float = 0.85
    min_validation_improvement: float = 0.03
    max_abs_value: float = 1.0e12
    min_feature_std: float = 1.0e-10
    duplicate_correlation: float = 0.9995
    complexity_tiebreak: float = 1.0e-4


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Top-level automatic feature-engineering configuration.

    The formula type is owned by :class:`MySRRegressor`. Keeping a second
    public formula-type field here would allow feature engineering and the
    MySRCore search to use different dimensional policies.
    """

    mode: Literal["suggest", "augment"] = "suggest"
    surrogate_engine: SurrogateEngineConfig = field(
        default_factory=SurrogateEngineConfig
    )
    feat_engine: FEATEngineConfig = field(default_factory=FEATEngineConfig)
    max_generated_features: int = 12


def coerce_feature_engineering_config(
    config: FeatureEngineeringConfig | dict[str, Any],
) -> FeatureEngineeringConfig:
    """Normalize a public config object or a convenient nested dictionary."""

    if isinstance(config, FeatureEngineeringConfig):
        normalized = config
    elif isinstance(config, dict):
        if "formula_type" in config:
            raise ValueError(
                "formula_type is owned by MySRRegressor; pass it as "
                "MySRRegressor(formula_type=...) instead of nesting it in "
                "feature_engineering_config"
            )
        surrogate_value = config.get("surrogate_engine", {})
        feat_value = config.get("feat_engine", {})
        surrogate = (
            surrogate_value
            if isinstance(surrogate_value, SurrogateEngineConfig)
            else SurrogateEngineConfig(**surrogate_value)
        )
        feat = (
            feat_value
            if isinstance(feat_value, FEATEngineConfig)
            else FEATEngineConfig(**feat_value)
        )
        normalized = FeatureEngineeringConfig(
            mode=cast(Any, config.get("mode", "suggest")),
            surrogate_engine=surrogate,
            feat_engine=feat,
            # Preserve the raw value until the shared validation below. Casting
            # here would silently turn values such as 2.5 into 2 and booleans
            # into integers, hiding malformed public configuration.
            max_generated_features=config.get("max_generated_features", 12),
        )
    else:
        raise TypeError(
            "feature_engineering_config must be a FeatureEngineeringConfig or dictionary"
        )
    surrogate_value = normalized.surrogate_engine
    feat_value = normalized.feat_engine
    if isinstance(surrogate_value, dict):
        surrogate_value = SurrogateEngineConfig(**surrogate_value)
    if isinstance(feat_value, dict):
        feat_value = FEATEngineConfig(**feat_value)
    if (
        surrogate_value is not normalized.surrogate_engine
        or feat_value is not normalized.feat_engine
    ):
        normalized = replace(
            normalized,
            surrogate_engine=surrogate_value,
            feat_engine=feat_value,
        )
    if normalized.mode not in {"suggest", "augment"}:
        raise ValueError("feature-engineering mode must be 'suggest' or 'augment'")
    if not isinstance(normalized.max_generated_features, int) or isinstance(
        normalized.max_generated_features, bool
    ):
        raise TypeError("max_generated_features must be an integer")
    if normalized.max_generated_features < 1:
        raise ValueError("max_generated_features must be positive")
    return normalized


@dataclass(frozen=True)
class FeatureComplexitySpec:
    """Cross-language complexity contract for replayable feature trees.

    MySRCore computes a custom complexity by recursively summing variable leaves,
    constant leaves, and operator nodes. This immutable Python specification
    mirrors that public ``ComplexityMapping`` subset so generated feature columns
    can carry the cost of their expanded expression instead of being treated as
    free terminal variables.
    """

    variable_complexities: tuple[float, ...]
    constant_complexity: float = 1.0
    operator_complexities: tuple[tuple[str, float], ...] = ()

    @classmethod
    def from_user(
        cls,
        n_features: int,
        complexity_of_variables: float | Sequence[float] | None = None,
        complexity_of_constants: float | None = None,
        complexity_of_operators: Mapping[str, float] | None = None,
    ) -> FeatureComplexitySpec:
        if n_features < 1:
            raise ValueError("n_features must be positive")
        if complexity_of_variables is None:
            variable_values = [1.0] * n_features
        elif isinstance(complexity_of_variables, Sequence) and not isinstance(
            complexity_of_variables, (str, bytes)
        ):
            if len(complexity_of_variables) != n_features:
                raise ValueError(
                    "The number of elements in `complexity_of_variables` must equal "
                    "the number of input features for automatic feature engineering."
                )
            variable_values = [float(value) for value in complexity_of_variables]
        else:
            variable_values = [float(complexity_of_variables)] * n_features
        constant_value = (
            1.0 if complexity_of_constants is None else float(complexity_of_constants)
        )
        if not np.isfinite(variable_values).all() or not np.isfinite(constant_value):
            raise ValueError("complexity values must be finite")
        operator_values: list[tuple[str, float]] = []
        if complexity_of_operators is not None:
            for name, value in complexity_of_operators.items():
                numeric_value = float(value)
                if not np.isfinite(numeric_value):
                    raise ValueError("operator complexity values must be finite")
                operator_values.append((str(name), numeric_value))
        return cls(
            variable_complexities=tuple(variable_values),
            constant_complexity=constant_value,
            operator_complexities=tuple(sorted(operator_values)),
        )

    @property
    def operator_map(self) -> dict[str, float]:
        return dict(self.operator_complexities)

    def variable_cost(self, index: int) -> float:
        try:
            return self.variable_complexities[index]
        except IndexError as error:  # pragma: no cover - defensive tree validation
            raise ValueError(f"feature index {index} is outside the complexity specification") from error

    def operator_cost(self, feature_operator: str) -> float:
        """Return the backend cost for a FeatureNode operator.

        The feature engine uses readable internal names while MySRCore stores the
        actual operator spelling. Composite helpers intentionally resolve to the
        operators present in their expanded expression.
        """

        aliases = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "div": "/",
            "power": "^",
            "square": "^",
            "cube": "^",
            "sqrt_abs": "sqrt",
            "log_abs": "log",
            "reciprocal": "/",
        }
        mapping = self.operator_map
        # Honor an explicitly configured readable helper name (for example
        # "square") before falling back to its expanded backend operator.
        if feature_operator in mapping:
            return float(mapping[feature_operator])
        backend_name = aliases.get(feature_operator, feature_operator)
        return float(mapping.get(backend_name, mapping.get(feature_operator, 1.0)))


@dataclass(frozen=True)
class FeatureDimensionSpec:
    """Dimension context used to screen generated feature ASTs."""

    input_dimensions: tuple[DimensionVector, ...]
    target_dimension: DimensionVector | None
    policy: Literal["ignore", "compatible", "strict"]

    def __post_init__(self) -> None:
        if self.policy not in {"ignore", "compatible", "strict"}:
            raise ValueError("dimension policy must be 'ignore', 'compatible', or 'strict'")
        for index, dimension in enumerate(self.input_dimensions):
            if len(dimension) != 7 or not np.all(np.isfinite(dimension)):
                raise ValueError(
                    f"input_dimensions[{index}] must be a finite seven-component vector"
                )
        if self.target_dimension is not None and (
            len(self.target_dimension) != 7
            or not np.all(np.isfinite(self.target_dimension))
        ):
            raise ValueError(
                "target_dimension must be a finite seven-component vector"
            )

    def _infer(self, node: FeatureNode) -> tuple[DimensionVector | None, str, str | None]:
        if node.operator == "variable":
            if node.index is None or not isinstance(node.index, int) or node.index < 0:
                return None, "unknown", "feature_index"
            try:
                dimension = self.input_dimensions[node.index]
            except IndexError:
                return None, "1", "feature_index"
            return dimension, dimension_to_string(dimension), None
        if node.operator == "constant":
            return (0.0,) * 7, "1", None

        child_results = [self._infer(child) for child in node.children]
        if any(result[2] is not None for result in child_results):
            return None, "unknown", "dimension_unknown" if self.policy == "ignore" else "dimension_constraint"
        dimensions_raw = [result[0] for result in child_results]
        if any(dimension is None for dimension in dimensions_raw):
            return None, "unknown", "dimension_unknown" if self.policy == "ignore" else "dimension_constraint"
        dimensions = [cast(DimensionVector, dimension) for dimension in dimensions_raw]
        zero = (0.0,) * 7

        if node.operator == "square":
            output = dimension_scale(dimensions[0], 2.0)
            return output, dimension_to_string(output), None
        if node.operator == "cube":
            output = dimension_scale(dimensions[0], 3.0)
            return output, dimension_to_string(output), None
        if node.operator == "sqrt_abs":
            output = dimension_scale(dimensions[0], 0.5)
            return output, dimension_to_string(output), None
        if node.operator == "log_abs":
            if not dimension_is_zero(dimensions[0]):
                if self.policy == "ignore":
                    return None, "unknown", "dimension_unknown"
                return None, "1", "dimension_constraint"
            return zero, "1", None
        if node.operator == "reciprocal":
            output = dimension_scale(dimensions[0], -1.0)
            return output, dimension_to_string(output), None
        if node.operator == "power":
            if node.value is None:
                return None, "1", "missing_exponent"
            output = dimension_scale(dimensions[0], node.value)
            return output, dimension_to_string(output), None
        if node.operator in {"normalized_sub", "hypot"}:
            if not dimension_equal(dimensions[0], dimensions[1]):
                if self.policy == "ignore":
                    return None, "unknown", "dimension_unknown"
                return None, "1", "dimension_constraint"
            if node.operator == "normalized_sub":
                return zero, "1", None
            return dimensions[0], dimension_to_string(dimensions[0]), None
        if node.operator in {"log_ratio", "exp_ratio", "sin_ratio", "cos_ratio"}:
            ratio_dimension = dimension_sub(dimensions[0], dimensions[1])
            if not dimension_is_zero(ratio_dimension):
                if self.policy == "ignore":
                    return None, "unknown", "dimension_unknown"
                return None, "1", "dimension_constraint"
            if node.operator == "log_ratio":
                return zero, "1", None
            return zero, "1", None
        if node.operator in {"add", "sub"}:
            if not dimension_equal(dimensions[0], dimensions[1]):
                if self.policy == "ignore":
                    return None, "unknown", "dimension_unknown"
                return None, "1", "dimension_constraint"
            return dimensions[0], dimension_to_string(dimensions[0]), None
        if node.operator == "mul":
            output = dimension_add(dimensions[0], dimensions[1])
            return output, dimension_to_string(output), None
        if node.operator == "div":
            output = dimension_sub(dimensions[0], dimensions[1])
            return output, dimension_to_string(output), None
        if node.operator in {"exp", "sin", "cos"}:
            if not dimension_is_zero(dimensions[0]):
                if self.policy == "ignore":
                    return None, "unknown", "dimension_unknown"
                return None, "1", "dimension_constraint"
            return zero, "1", None
        if node.operator == "abs":
            return dimensions[0], dimension_to_string(dimensions[0]), None
        return None, "1", "unsupported_operator"

    def validate(self, node: FeatureNode) -> tuple[bool, str, str | None]:
        dimension, expression, reason = self._infer(node)
        if self.policy == "ignore":
            # Empirical mode never rejects a candidate because of dimensions.
            # Keep an explicit marker when propagation is not physically defined.
            return True, expression, None
        if reason is not None or dimension is None:
            return False, expression, reason or "dimension_constraint"
        if self.policy == "strict":
            if self.target_dimension is None:
                return False, expression, "missing_output_dimension"
            if not dimension_equal(dimension, self.target_dimension):
                return False, expression, "output_dimension_mismatch"
        return True, expression, None


_BINARY_FEATURE_OPERATORS = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "normalized_sub",
        "hypot",
        "log_ratio",
        "exp_ratio",
        "sin_ratio",
        "cos_ratio",
    }
)
_UNARY_FEATURE_OPERATORS = frozenset(
    {
        "square",
        "cube",
        "sqrt_abs",
        "reciprocal",
        "abs",
        "log_abs",
        "exp",
        "sin",
        "cos",
    }
)


@dataclass(frozen=True)
class FeatureNode:
    """A small replayable expression tree built only from input columns."""

    operator: str
    children: tuple[FeatureNode, ...] = ()
    index: int | None = None
    value: float | None = None

    def __post_init__(self) -> None:
        """Reject malformed public trees before they reach replay or inference."""

        if self.operator == "variable":
            if (
                self.children
                or self.value is not None
                or self.index is None
                or not isinstance(self.index, int)
                or isinstance(self.index, bool)
                or self.index < 0
            ):
                raise ValueError(
                    "variable FeatureNode requires a non-negative integer index "
                    "and no children or value"
                )
            return
        if self.operator == "constant":
            if self.children or self.index is not None or self.value is None:
                raise ValueError(
                    "constant FeatureNode requires a finite value and no children or index"
                )
            if not np.isfinite(float(self.value)):
                raise ValueError("constant FeatureNode value must be finite")
            return
        if self.operator == "power":
            if (
                len(self.children) != 1
                or self.index is not None
                or self.value is None
                or not np.isfinite(float(self.value))
            ):
                raise ValueError(
                    "power FeatureNode requires one child and a finite exponent"
                )
            return
        if self.operator in _UNARY_FEATURE_OPERATORS:
            if len(self.children) != 1 or self.index is not None or self.value is not None:
                raise ValueError(
                    f"{self.operator} FeatureNode requires exactly one child"
                )
            return
        if self.operator in _BINARY_FEATURE_OPERATORS:
            if len(self.children) != 2 or self.index is not None or self.value is not None:
                raise ValueError(
                    f"{self.operator} FeatureNode requires exactly two children"
                )
            return
        raise ValueError(f"Unsupported feature operator: {self.operator}")

    @staticmethod
    def variable(index: int) -> FeatureNode:
        return FeatureNode("variable", index=index)

    @staticmethod
    def constant(value: float) -> FeatureNode:
        return FeatureNode("constant", value=float(value))

    @staticmethod
    def power(child: FeatureNode, exponent: float) -> FeatureNode:
        return FeatureNode("power", (child,), value=float(exponent))

    @property
    def depth(self) -> int:
        if self.operator in {"variable", "constant"}:
            return 0
        if self.operator == "normalized_sub":
            # normalized_sub expands to one binary root and one binary child.
            return 2 + max(child.depth for child in self.children)
        if self.operator in {"hypot", "log_ratio"}:
            # hypot expands to sqrt(add(power(...), power(...))); log_ratio
            # expands to log(abs(div(...))).
            return 3 + max(child.depth for child in self.children)
        if self.operator in {"exp_ratio", "sin_ratio", "cos_ratio"}:
            return 2 + max(child.depth for child in self.children)
        return 1 + max(child.depth for child in self.children)

    @property
    def complexity(self) -> int:
        if self.operator in {"variable", "constant"}:
            return 1
        return 1 + sum(child.complexity for child in self.children)

    def ast_complexity(self, spec: FeatureComplexitySpec) -> float:
        """Compute the MySRCore-compatible expanded AST complexity.

        ``FeatureNode`` helper operators such as ``square`` and ``reciprocal``
        are expanded to the expression that is sent to the backend metadata
        contract, including exponent/reciprocal constants where applicable.
        """

        if self.operator == "variable":
            assert self.index is not None
            return spec.variable_cost(self.index)
        if self.operator == "constant":
            return spec.constant_complexity

        child_costs = [child.ast_complexity(spec) for child in self.children]
        # An explicitly named feature operator is a deliberate user override of
        # the default expanded-AST fallback. This is important for helper names
        # such as square/cube and for custom composition operators.
        if self.operator in spec.operator_map:
            hidden_constant = spec.constant_complexity if self.operator == "power" else 0.0
            return (
                spec.operator_cost(self.operator)
                + sum(child_costs)
                + hidden_constant
            )
        if self.operator == "square":
            # MySRCore's built-in square is defined as x * x.
            return spec.operator_cost("*") + 2.0 * child_costs[0]
        if self.operator == "cube":
            # MySRCore's built-in cube is defined as x * x * x.
            return (
                2.0 * spec.operator_cost("*")
                + 3.0 * child_costs[0]
            )
        if self.operator == "power":
            return (
                spec.operator_cost("^")
                + child_costs[0]
                + spec.constant_complexity
            )
        if self.operator == "sqrt_abs":
            return (
                spec.operator_cost("sqrt")
                + spec.operator_cost("abs")
                + child_costs[0]
            )
        if self.operator == "log_abs":
            return (
                spec.operator_cost("log")
                + spec.operator_cost("abs")
                + child_costs[0]
            )
        if self.operator == "reciprocal":
            return (
                spec.operator_cost("/")
                + spec.constant_complexity
                + child_costs[0]
            )
        if self.operator == "normalized_sub":
            # (a - b) / (a + b) expands both children twice.
            return (
                spec.operator_cost("-")
                + spec.operator_cost("+")
                + spec.operator_cost("/")
                + 2.0 * sum(child_costs)
            )
        if self.operator == "hypot":
            # sqrt(a² + b²), represented explicitly with two power nodes and
            # their constant leaves in the replayed AST.
            return (
                spec.operator_cost("sqrt")
                + spec.operator_cost("+")
                + 2.0 * (spec.operator_cost("^") + spec.constant_complexity)
                + sum(child_costs)
            )
        if self.operator == "log_ratio":
            return (
                spec.operator_cost("log")
                + spec.operator_cost("abs")
                + spec.operator_cost("/")
                + sum(child_costs)
            )
        if self.operator in {"exp_ratio", "sin_ratio", "cos_ratio"}:
            return (
                spec.operator_cost(self.operator.removesuffix("_ratio"))
                + spec.operator_cost("/")
                + sum(child_costs)
            )
        return spec.operator_cost(self.operator) + sum(child_costs)

    @property
    def input_indices(self) -> tuple[int, ...]:
        if self.operator == "variable":
            assert self.index is not None
            return (self.index,)
        if self.operator == "constant":
            return ()
        return tuple(
            sorted({index for child in self.children for index in child.input_indices})
        )

    @property
    def signature(self) -> str:
        if self.operator == "variable":
            return f"x{self.index}"
        if self.operator == "constant":
            assert self.value is not None
            return f"c{self.value:.10g}"
        if self.operator == "power":
            assert self.value is not None
            return f"power({self.children[0].signature},{self.value:.10g})"
        child_signatures = [child.signature for child in self.children]
        if self.operator in {"add", "mul", "hypot"}:
            child_signatures.sort()
        return f"{self.operator}({','.join(child_signatures)})"

    def expression(self, variable_names: Sequence[str]) -> str:
        if self.operator == "variable":
            assert self.index is not None
            return variable_names[self.index]
        if self.operator == "constant":
            assert self.value is not None
            return f"{self.value:.10g}"
        children = [child.expression(variable_names) for child in self.children]
        if self.operator == "power":
            assert self.value is not None
            return f"({children[0]})^{self.value:.10g}"
        if self.operator == "normalized_sub":
            return f"(({children[0]} - {children[1]}) / ({children[0]} + {children[1]}))"
        if self.operator == "hypot":
            return f"sqrt(({children[0]})^2 + ({children[1]})^2)"
        if self.operator == "log_ratio":
            return f"log(abs(({children[0]} / {children[1]})))"
        if self.operator in {"exp_ratio", "sin_ratio", "cos_ratio"}:
            function = self.operator.removesuffix("_ratio")
            return f"{function}(({children[0]} / {children[1]}))"
        symbols = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
        if self.operator in symbols:
            return f"({children[0]} {symbols[self.operator]} {children[1]})"
        formats = {
            "square": "({0})^2",
            "cube": "({0})^3",
            "sqrt_abs": "sqrt(abs({0}))",
            "reciprocal": "1/({0})",
            "abs": "abs({0})",
            "log_abs": "log(abs({0}))",
            "exp": "exp({0})",
            "sin": "sin({0})",
            "cos": "cos({0})",
        }
        return formats[self.operator].format(children[0])

    def evaluate(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.operator == "variable":
            assert self.index is not None
            return np.asarray(X[:, self.index], dtype=float)
        if self.operator == "constant":
            assert self.value is not None
            return np.full(X.shape[0], self.value, dtype=float)
        values = [child.evaluate(X) for child in self.children]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if self.operator == "add":
                result = values[0] + values[1]
            elif self.operator == "sub":
                result = values[0] - values[1]
            elif self.operator == "mul":
                result = values[0] * values[1]
            elif self.operator == "div":
                result = values[0] / values[1]
            elif self.operator == "square":
                result = values[0] ** 2
            elif self.operator == "cube":
                result = values[0] ** 3
            elif self.operator == "sqrt_abs":
                result = np.sqrt(np.abs(values[0]))
            elif self.operator == "reciprocal":
                result = 1.0 / values[0]
            elif self.operator == "abs":
                result = np.abs(values[0])
            elif self.operator == "log_abs":
                result = np.log(np.abs(values[0]))
            elif self.operator == "exp":
                result = np.exp(values[0])
            elif self.operator == "sin":
                result = np.sin(values[0])
            elif self.operator == "cos":
                result = np.cos(values[0])
            elif self.operator == "power":
                assert self.value is not None
                result = values[0] ** self.value
            elif self.operator == "normalized_sub":
                result = (values[0] - values[1]) / (values[0] + values[1])
            elif self.operator == "hypot":
                result = np.sqrt(values[0] ** 2 + values[1] ** 2)
            elif self.operator == "log_ratio":
                result = np.log(np.abs(values[0] / values[1]))
            elif self.operator == "exp_ratio":
                result = np.exp(values[0] / values[1])
            elif self.operator == "sin_ratio":
                result = np.sin(values[0] / values[1])
            elif self.operator == "cos_ratio":
                result = np.cos(values[0] / values[1])
            else:  # pragma: no cover
                raise ValueError(f"Unsupported feature operator: {self.operator}")
        return np.asarray(result, dtype=float)

    def dimension_expression(
        self, input_dimensions: Sequence[DimensionVector]
    ) -> str:
        """Build a compact dimension label for metadata replay."""

        _, expression, _ = FeatureDimensionSpec(
            tuple(input_dimensions), None, "ignore"
        )._infer(self)
        return expression


@dataclass(frozen=True)
class FeatureProposal:
    """An accepted or rejected replayable feature candidate."""

    operator: str
    left_index: int
    right_index: int
    name: str
    expression: str
    invariance_score: float
    relevance_score: float
    support_fraction: float
    accepted: bool
    rejection_reason: str | None = None
    feature_kind: Literal[
        "pairwise_symmetry",
        "unary_composition",
        "recursive_composition",
        "feat_evolved",
    ] = "pairwise_symmetry"
    depth: int = 1
    complexity: float = 3.0
    validation_score: float = 0.0
    improvement_score: float = 0.0
    construction_score: float = 0.0
    stability_fraction: float = 1.0
    parameter: float | None = None
    node: FeatureNode | None = field(default=None, repr=False, compare=False)

    @property
    def input_indices(self) -> tuple[int, ...]:
        if self.node is not None:
            return self.node.input_indices
        return tuple(sorted({self.left_index, self.right_index}))

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.node is not None:
            return self.node.evaluate(X)
        left = X[:, self.left_index]
        right = X[:, self.right_index]
        with np.errstate(divide="ignore", invalid="ignore"):
            values = {
                "sub": left - right,
                "add": left + right,
                "mul": left * right,
                "div": left / right,
            }[self.operator]
        return np.asarray(values, dtype=float)

    def to_record(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "input_indices": self.input_indices,
            "left_index": self.left_index,
            "right_index": self.right_index,
            "name": self.name,
            "expression": self.expression,
            "feature_kind": self.feature_kind,
            "depth": self.depth,
            "complexity": self.complexity,
            "invariance_score": self.invariance_score,
            "relevance_score": self.relevance_score,
            "validation_score": self.validation_score,
            "improvement_score": self.improvement_score,
            "construction_score": self.construction_score,
            "stability_fraction": self.stability_fraction,
            "parameter": self.parameter,
            "support_fraction": self.support_fraction,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }

    def dimension_expression(
        self, input_dimensions: Sequence[DimensionVector]
    ) -> str:
        if self.node is not None:
            return self.node.dimension_expression(input_dimensions)
        left_dimension = input_dimensions[self.left_index]
        right_dimension = input_dimensions[self.right_index]
        if self.operator in {"add", "sub"}:
            return (
                dimension_to_string(left_dimension)
                if dimension_equal(left_dimension, right_dimension)
                else "unknown"
            )
        if self.operator == "mul":
            return dimension_to_string(dimension_add(left_dimension, right_dimension))
        return dimension_to_string(dimension_sub(left_dimension, right_dimension))


@dataclass(frozen=True)
class DecompositionProposal:
    """A diagnostic additive or multiplicative variable partition."""

    kind: Literal["additive", "multiplicative"]
    left_indices: tuple[int, ...]
    right_indices: tuple[int, ...]
    score: float
    support_fraction: float
    left_relevance: float
    right_relevance: float
    accepted: bool
    rejection_reason: str | None = None

    def to_record(self) -> dict[str, Any]:
        return self.__dict__.copy()


class SurrogateFeatureEngineer:
    """Discover low-dimensional structure from a supervised data set."""

    def __init__(
        self,
        config: SurrogateEngineConfig | None = None,
        *,
        random_state: int | np.random.RandomState | None = None,
        complexity_spec: FeatureComplexitySpec | None = None,
        dimension_spec: FeatureDimensionSpec | None = None,
    ) -> None:
        self.config = config or SurrogateEngineConfig()
        self.random_state = random_state
        self.complexity_spec = complexity_spec
        self.dimension_spec = dimension_spec
        self.report_: dict[str, Any]
        self.parameter_search_report_: list[dict[str, Any]] = []
        self.composition_search_layers_: list[dict[str, Any]] = []

    def _node_complexity(self, node: FeatureNode) -> float:
        if not hasattr(self, "complexity_spec_"):
            raise RuntimeError("complexity specification is not initialized")
        return node.ast_complexity(self.complexity_spec_)

    def _dimension_check(self, node: FeatureNode) -> tuple[bool, str | None]:
        if not hasattr(self, "dimension_spec_") or self.dimension_spec_ is None:
            return True, None
        valid, _, reason = self.dimension_spec_.validate(node)
        return valid, reason

    @staticmethod
    def _validate_config(config: SurrogateEngineConfig) -> None:
        binary_allowed = {"sub", "add", "mul", "div"}
        composition_allowed = binary_allowed | {
            "normalized_sub",
            "hypot",
            "log_ratio",
            "exp_ratio",
            "sin_ratio",
            "cos_ratio",
        }
        unary_allowed = {
            "square",
            "cube",
            "sqrt_abs",
            "reciprocal",
            "abs",
            "log_abs",
            "exp",
            "sin",
            "cos",
        }
        if config.enable_pairwise_symmetry and not config.candidate_operators:
            raise ValueError("candidate_operators must contain at least one operator")
        if any(op not in binary_allowed for op in config.candidate_operators):
            raise ValueError("candidate_operators contains an unsupported operator")
        if any(op not in composition_allowed for op in config.composition_operators):
            raise ValueError("composition_operators contains an unsupported operator")
        if any(op not in unary_allowed for op in config.candidate_unary_operators):
            raise ValueError("candidate_unary_operators contains an unsupported operator")
        for field_name in (
            "surrogate_min_r2",
            "invariance_min_score",
            "separability_min_score",
            "composition_min_score",
            "composition_min_improvement",
            "high_confidence_composition_score",
            "relevance_min_fraction",
            "invariance_relative_tolerance",
            "surrogate_stability_min_fraction",
            "construction_fraction",
            "validation_fraction",
            "parameter_search_min_score",
        ):
            if not 0.0 <= float(getattr(config, field_name)) <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")
        integer_fields = (
            "n_perturbations",
            "surrogate_ensemble_size",
            "max_candidate_pairs",
            "max_parameterized_candidates",
            "parameter_grid_size",
            "max_composition_candidates",
            "max_separability_partitions",
            "max_subset_size",
            "max_composition_depth",
            "composition_beam_width",
            "max_generated_features",
            "max_iter",
        )
        for field_name in integer_fields:
            value = getattr(config, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be an integer")
            if value < 1:
                raise ValueError(f"{field_name} must be positive")
        if (
            isinstance(config.hidden_layer_sizes, (str, bytes))
            or not isinstance(config.hidden_layer_sizes, Sequence)
            or not config.hidden_layer_sizes
            or any(
                not isinstance(size, int) or isinstance(size, bool) or size < 1
                for size in config.hidden_layer_sizes
            )
        ):
            raise TypeError("hidden_layer_sizes must be a non-empty sequence of positive integers")
        if config.exp_input_limit <= 0:
            raise ValueError("exp_input_limit must be positive")
        if config.complexity_penalty < 0:
            raise ValueError("complexity_penalty must be non-negative")
        if config.construction_fraction + config.validation_fraction >= 1.0:
            raise ValueError(
                "construction_fraction + validation_fraction must be less than 1"
            )
        if not config.perturbation_scales or any(
            scale <= 0 for scale in config.perturbation_scales
        ):
            raise ValueError("perturbation_scales must contain positive values")
        if config.parameter_min <= 0 or config.parameter_max <= config.parameter_min:
            raise ValueError("parameter bounds must satisfy 0 < min < max")
        if config.parameter_identity_tolerance < 0:
            raise ValueError("parameter_identity_tolerance must be non-negative")
        if not 0.0 <= config.parameter_boundary_tolerance < 0.5:
            raise ValueError("parameter_boundary_tolerance must be in [0, 0.5)")
        if config.parameter_min_improvement < 0:
            raise ValueError("parameter_min_improvement must be non-negative")
        if not config.power_exponents or any(
            not np.isfinite(float(exponent)) or abs(float(exponent)) < 1.0e-12
            for exponent in config.power_exponents
        ):
            raise ValueError("power_exponents must contain finite non-zero values")

    @staticmethod
    def _as_float_matrix(X: Any) -> NDArray[np.float64]:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError("X must be a two-dimensional numeric matrix")
        if values.shape[0] < 20:
            raise ValueError(
                "at least 20 samples are required for surrogate feature engineering"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("X must contain only finite values")
        return values

    @staticmethod
    def _as_transform_matrix(X: Any) -> NDArray[np.float64]:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError("X must be a two-dimensional numeric matrix")
        if not np.all(np.isfinite(values)):
            raise ValueError("X must contain only finite values")
        return values

    @staticmethod
    def _as_target(y: Any, n_samples: int) -> NDArray[np.float64]:
        values = np.asarray(y, dtype=float).reshape(-1)
        if values.shape[0] != n_samples:
            raise ValueError("X and y have inconsistent numbers of samples")
        if not np.all(np.isfinite(values)):
            raise ValueError("y must contain only finite values")
        return values

    def _make_surrogate(self, seed: int) -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=self.config.hidden_layer_sizes,
                        max_iter=self.config.max_iter,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=30,
                        random_state=seed,
                    ),
                ),
            ]
        )

    @staticmethod
    def _pair_directions(
        n_features: int,
        operators: Sequence[CandidateOperator],
        max_pairs: int,
        relevance: NDArray[np.float64] | None = None,
    ) -> list[tuple[CandidateOperator, int, int]]:
        pairs: list[tuple[CandidateOperator, int, int]] = []
        index_pairs = list(combinations(range(n_features), 2))
        if relevance is not None:
            index_pairs.sort(
                key=lambda pair: (
                    min(float(relevance[pair[0]]), float(relevance[pair[1]])),
                    float(relevance[pair[0]]) + float(relevance[pair[1]]),
                ),
                reverse=True,
            )
        for i, j in index_pairs:
            for operator in operators:
                pairs.append((operator, i, j))
                if len(pairs) >= max_pairs:
                    return pairs
        return pairs

    @staticmethod
    def _support_bounds(
        X: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower = np.min(X, axis=0)
        upper = np.max(X, axis=0)
        span = upper - lower
        margin = np.maximum(span * 0.02, 1e-12)
        return lower - margin, upper + margin

    def _perturb(
        self,
        X: NDArray[np.float64],
        operator: CandidateOperator,
        i: int,
        j: int,
        rng: np.random.RandomState,
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
        *,
        parameter: float = 1.0,
        perturbation_scale: float = 0.35,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        scales = np.std(X, axis=0)
        scales = np.where(scales > 1e-12, scales, 1.0)
        transformed = X.copy()
        valid = np.ones(X.shape[0], dtype=bool)
        if operator in ("sub", "add"):
            effective_scale = min(scales[i], abs(parameter) * scales[j])
            delta = rng.normal(
                0.0, perturbation_scale * effective_scale, X.shape[0]
            )
            transformed[:, i] += delta
            direction = 1.0 if operator == "sub" else -1.0
            transformed[:, j] += direction * delta / parameter
        else:
            factor = np.exp(rng.normal(0.0, perturbation_scale, X.shape[0]))
            if operator == "mul":
                transformed[:, i] *= factor
                with np.errstate(divide="ignore", invalid="ignore"):
                    transformed[:, j] /= factor ** (1.0 / parameter)
            else:
                transformed[:, i] *= factor
                transformed[:, j] *= factor ** (1.0 / parameter)
        valid &= np.all(transformed >= lower, axis=1)
        valid &= np.all(transformed <= upper, axis=1)
        valid &= np.all(np.isfinite(transformed), axis=1)
        if operator == "div":
            valid &= np.abs(transformed[:, j]) > 1e-12
        return transformed, valid

    def _invariance_score(
        self,
        model: Pipeline,
        X: NDArray[np.float64],
        y_prediction: NDArray[np.float64],
        candidate: PairSymmetryCandidate,
        rng: np.random.RandomState,
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
    ) -> tuple[float, float]:
        operator = candidate.operator
        i = candidate.left_index
        j = candidate.right_index
        errors: list[float] = []
        supports: list[float] = []
        prediction_scale = max(float(np.std(y_prediction)), 1e-8)
        for perturbation_scale in self.config.perturbation_scales:
            for _ in range(self.config.n_perturbations):
                transformed, valid = self._perturb(
                    X,
                    operator,
                    i,
                    j,
                    rng,
                    lower,
                    upper,
                    parameter=candidate.parameter,
                    perturbation_scale=perturbation_scale,
                )
                if np.any(valid):
                    prediction = np.asarray(model.predict(transformed[valid]), dtype=float)
                    errors.append(
                        float(np.mean(np.abs(prediction - y_prediction[valid])))
                    )
                supports.append(float(np.mean(valid)))
        if not errors:
            return 0.0, float(np.mean(supports))
        relative_error = float(np.mean(errors)) / prediction_scale
        return max(0.0, min(1.0, 1.0 - relative_error)), float(np.mean(supports))

    @staticmethod
    def _association_score(values: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        if values.size == 0 or float(np.std(values)) <= 1e-12:
            return 0.0
        correlation = float(np.corrcoef(values, y)[0, 1])
        return correlation * correlation if np.isfinite(correlation) else 0.0

    def _node_support(
        self, node: FeatureNode, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        # Evaluate every subtree separately so an undefined intermediate value
        # cannot be hidden by a later operation (for example, 1 / (x / 0)
        # becoming the finite value 0 after IEEE-754 propagation).
        child_values: list[NDArray[np.float64]] = []
        valid = np.ones(X.shape[0], dtype=bool)
        for child in node.children:
            child_value, child_valid = self._node_support(child, X)
            child_values.append(child_value)
            valid &= child_valid

        values = node.evaluate(X)
        valid &= np.isfinite(values)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if node.operator in {"reciprocal", "log_abs"}:
                valid &= np.abs(child_values[0]) > 1e-12
            if node.operator == "div":
                valid &= np.abs(child_values[1]) > 1e-12
            if node.operator == "normalized_sub":
                denominator = child_values[0] + child_values[1]
                valid &= np.abs(denominator) > 1e-12
            if node.operator in {"log_ratio", "exp_ratio", "sin_ratio", "cos_ratio"}:
                numerator = child_values[0]
                denominator = child_values[1]
                valid &= np.abs(denominator) > 1e-12
                if node.operator == "log_ratio":
                    valid &= np.abs(numerator) > 1e-12
                ratio = numerator / denominator
                if node.operator == "exp_ratio":
                    valid &= np.abs(ratio) <= self.config.exp_input_limit
            if node.operator == "power" and node.value is not None:
                base = child_values[0]
                exponent = float(node.value)
                non_integer = abs(exponent - round(exponent)) > 1e-10
                if non_integer:
                    valid &= base >= 0.0
                if exponent < 0.0:
                    valid &= np.abs(base) > 1e-12
            if node.operator == "exp":
                valid &= np.abs(child_values[0]) <= self.config.exp_input_limit
        return values, valid

    @staticmethod
    def _candidate_name(node: FeatureNode) -> str:
        cleaned = node.signature.replace("x", "")
        return "afe_" + re.sub(r"[^0-9A-Za-z_]+", "_", cleaned).strip("_")

    @staticmethod
    def _binary_node(
        operator: CandidateOperator, left: FeatureNode, right: FeatureNode
    ) -> FeatureNode:
        """Canonicalize commutative nodes so the beam does not revisit duplicates."""

        if operator in {"add", "mul"} and right.signature < left.signature:
            left, right = right, left
        return FeatureNode(operator, (left, right))

    @staticmethod
    def _composition_node(
        operator: CompositionOperator, left: FeatureNode, right: FeatureNode
    ) -> FeatureNode:
        """Build a canonical node for an extended composition operator."""

        if operator in {"add", "mul", "hypot"} and right.signature < left.signature:
            left, right = right, left
        return FeatureNode(operator, (left, right))

    def _composition_operator_pool(self) -> tuple[CompositionOperator, ...]:
        """Return a de-duplicated, validated recursive-composition grammar."""

        if self.config.composition_operators:
            configured = self.config.composition_operators
        elif tuple(self.config.candidate_operators) == ("sub", "add", "mul", "div"):
            # Keep the historical candidate_operators switch meaningful while
            # enabling the richer grammar for the untouched default configuration.
            configured = (
                *self.config.candidate_operators,
                "normalized_sub",
                "hypot",
                "log_ratio",
                "exp_ratio",
                "sin_ratio",
                "cos_ratio",
            )
        else:
            configured = self.config.candidate_operators
        ordered: list[CompositionOperator] = []
        for operator in configured:
            if operator not in ordered:
                ordered.append(operator)
        return tuple(ordered)

    def _composition_proposal(
        self,
        node: FeatureNode,
        source_nodes: Sequence[FeatureNode],
        construction_X: NDArray[np.float64],
        domain_X: NDArray[np.float64],
        construction_y: NDArray[np.float64],
        validation_X: NDArray[np.float64],
        validation_y: NDArray[np.float64],
        names: Sequence[str],
        feature_kind: Literal["unary_composition", "recursive_composition"],
        reference_nodes: Sequence[FeatureNode] = (),
    ) -> FeatureProposal:
        construction_values, construction_valid = self._node_support(
            node, construction_X
        )
        validation_values, validation_valid = self._node_support(node, validation_X)
        _, domain_valid = self._node_support(node, domain_X)
        support = float(np.mean(domain_valid))
        raw_sources = [FeatureNode.variable(index) for index in node.input_indices]
        input_set = set(node.input_indices)
        comparable_references = [
            reference
            for reference in reference_nodes
            if reference.signature != node.signature
            and set(reference.input_indices).issubset(input_set)
        ]
        construction_source_score = max(
            (
                self._association_score(
                    source.evaluate(construction_X), construction_y
                )
                for source in (
                    *source_nodes,
                    *raw_sources,
                    *comparable_references,
                )
            ),
            default=0.0,
        )
        validation_source_score = max(
            (
                self._association_score(source.evaluate(validation_X), validation_y)
                for source in (
                    *source_nodes,
                    *raw_sources,
                    *comparable_references,
                )
            ),
            default=0.0,
        )
        construction_score = (
            self._association_score(
                construction_values[construction_valid],
                construction_y[construction_valid],
            )
            if np.any(construction_valid)
            else 0.0
        )
        validation_score = (
            self._association_score(
                validation_values[validation_valid], validation_y[validation_valid]
            )
            if np.any(validation_valid)
            else 0.0
        )
        construction_improvement = construction_score - construction_source_score
        validation_improvement = validation_score - validation_source_score
        improvement = min(construction_improvement, validation_improvement)
        simpler_references = (
            *source_nodes,
            *raw_sources,
            *comparable_references,
        )
        redundant_with_simpler = any(
            self._node_complexity(reference) < self._node_complexity(node)
            and self._association_score(
                construction_values[construction_valid],
                reference.evaluate(construction_X)[construction_valid],
            )
            >= 1.0 - 1e-10
            for reference in simpler_references
            if np.any(construction_valid)
        )
        accepted = (
            support == 1.0
            and not redundant_with_simpler
            and construction_score >= self.config.composition_min_score
            and validation_score >= self.config.composition_min_score
            and (
                improvement >= self.config.composition_min_improvement
                or (
                    node.operator
                    in {
                        "power",
                        "normalized_sub",
                        "hypot",
                        "log_ratio",
                        "exp_ratio",
                        "sin_ratio",
                        "cos_ratio",
                    }
                    and min(construction_score, validation_score)
                    >= self.config.high_confidence_composition_score
                )
            )
        )
        reason = None
        dimension_valid, dimension_reason = self._dimension_check(node)
        if not dimension_valid:
            accepted = False
            reason = dimension_reason or "dimension_constraint"
        if not accepted:
            if reason is not None:
                pass
            elif support < 1.0:
                reason = "unsafe_composition_domain"
            elif redundant_with_simpler:
                reason = "redundant_with_simpler_feature"
            elif min(construction_score, validation_score) < (
                self.config.composition_min_score
            ):
                reason = "weak_target_association"
            else:
                reason = "insufficient_composition_gain"
        indices = node.input_indices
        return FeatureProposal(
            operator=node.operator,
            left_index=indices[0],
            right_index=indices[1] if len(indices) > 1 else -1,
            name=self._candidate_name(node),
            expression=node.expression(names),
            invariance_score=0.0,
            relevance_score=construction_score,
            support_fraction=support,
            accepted=accepted,
            rejection_reason=reason,
            feature_kind=feature_kind,
            depth=node.depth,
            complexity=self._node_complexity(node),
            validation_score=validation_score,
            improvement_score=improvement,
            construction_score=construction_score,
            node=node,
        )

    def _composition_candidates(
        self,
        X: NDArray[np.float64],
        domain_X: NDArray[np.float64],
        y: NDArray[np.float64],
        validation_X: NDArray[np.float64],
        validation_y: NDArray[np.float64],
        names: Sequence[str],
        pairwise: Sequence[FeatureProposal],
    ) -> list[FeatureProposal]:
        """Search bounded multi-level feature compositions with a beam frontier."""

        proposals: list[FeatureProposal] = []
        raw_nodes = [FeatureNode.variable(i) for i in range(X.shape[1])]
        reduced_nodes = [
            proposal.node
            for proposal in pairwise
            if proposal.accepted and proposal.node is not None
        ]
        typed_reduced = cast(list[FeatureNode], reduced_nodes)
        # Reduced symmetry variables carry stronger structural evidence than
        # untouched columns.  Put them first so the bounded candidate budget
        # reaches compositions such as (x1+x2)*(x3-x4) before spending all
        # slots on raw/raw combinations.
        base_nodes = typed_reduced + raw_nodes
        seen: set[str] = {node.signature for node in base_nodes}
        seen.update(
            proposal.node.signature
            for proposal in pairwise
            if proposal.node is not None
        )
        frontier = base_nodes
        reference_nodes = list(base_nodes)
        composition_operators = self._composition_operator_pool()
        self.composition_search_layers_: list[dict[str, Any]] = []
        layer_proposals: list[FeatureProposal] = []

        for layer in range(1, self.config.max_composition_depth + 1):
            if not frontier or len(proposals) >= self.config.max_composition_candidates:
                break
            layer_proposals.clear()

            def add_candidate(
                node: FeatureNode,
                sources: Sequence[FeatureNode],
                feature_kind: Literal[
                    "unary_composition", "recursive_composition"
                ],
            ) -> bool:
                if node.depth > self.config.max_composition_depth:
                    return False
                if node.signature in seen:
                    return False
                if len(proposals) + len(layer_proposals) >= (
                    self.config.max_composition_candidates
                ):
                    return True
                seen.add(node.signature)
                layer_proposals.append(
                    self._composition_proposal(
                        node,
                        sources,
                        X,
                        domain_X,
                        y,
                        validation_X,
                        validation_y,
                        names,
                        feature_kind,
                        reference_nodes,
                    )
                )
                return False

            budget_reached = False
            if self.config.enable_unary_composition:
                for source in frontier:
                    for unary_operator in self.config.candidate_unary_operators:
                        budget_reached = add_candidate(
                            FeatureNode(unary_operator, (source,)),
                            (source,),
                            "unary_composition",
                        )
                        if budget_reached:
                            break
                    if budget_reached:
                        break

            if self.config.enable_power_composition and not budget_reached:
                for source in frontier:
                    for exponent in self.config.power_exponents:
                        budget_reached = add_candidate(
                            FeatureNode.power(source, float(exponent)),
                            (source,),
                            "unary_composition",
                        )
                        if budget_reached:
                            break
                    if budget_reached:
                        break

            if self.config.enable_recursive_composition and not budget_reached:
                right_pool = base_nodes if layer > 1 else frontier
                for left in frontier:
                    for right in right_pool:
                        if left.signature == right.signature:
                            continue
                        for binary_operator in composition_operators:
                            orientations = [(left, right)]
                            if binary_operator in {
                                "sub",
                                "div",
                                "log_ratio",
                                "exp_ratio",
                                "sin_ratio",
                                "cos_ratio",
                            }:
                                orientations.append((right, left))
                            for oriented_left, oriented_right in orientations:
                                budget_reached = add_candidate(
                                    self._composition_node(
                                        binary_operator,
                                        oriented_left,
                                        oriented_right,
                                    ),
                                    (oriented_left, oriented_right),
                                    "recursive_composition",
                                )
                                if budget_reached:
                                    break
                            if budget_reached:
                                break
                        if budget_reached:
                            break
                    if budget_reached:
                        break

            proposals.extend(layer_proposals)
            ranked = sorted(
                (
                    proposal
                    for proposal in layer_proposals
                    if proposal.support_fraction == 1.0 and proposal.node is not None
                ),
                key=lambda proposal: (
                    proposal.construction_score
                    - self.config.complexity_penalty * proposal.complexity,
                    -proposal.complexity,
                ),
                reverse=True,
            )[: self.config.composition_beam_width]
            frontier = cast(
                list[FeatureNode],
                [proposal.node for proposal in ranked if proposal.node is not None],
            )
            reference_nodes.extend(frontier)
            self.composition_search_layers_.append(
                {
                    "layer": layer,
                    "generated_count": len(layer_proposals),
                    "frontier_count": len(frontier),
                    "accepted_count": sum(
                        proposal.accepted for proposal in layer_proposals
                    ),
                    "frontier": [
                        {
                            "expression": proposal.expression,
                            "score": proposal.construction_score,
                            "depth": proposal.depth,
                            "complexity": proposal.complexity,
                        }
                        for proposal in ranked
                    ],
                }
            )
        return proposals

    def _partitions(self, n_features: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        all_indices = tuple(range(n_features))
        partitions: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        max_size = min(self.config.max_subset_size, n_features // 2)
        for subset_size in range(1, max_size + 1):
            for left in combinations(all_indices, subset_size):
                right = tuple(index for index in all_indices if index not in left)
                if not right or left > right:
                    continue
                partitions.append((left, right))
                if len(partitions) >= self.config.max_separability_partitions:
                    return partitions
        return partitions

    def _separability_candidates(
        self,
        model: Pipeline,
        X: NDArray[np.float64],
        importance: NDArray[np.float64],
        rng: np.random.RandomState,
    ) -> list[DecompositionProposal]:
        if not self.config.enable_separability or X.shape[1] < 2:
            return []
        proposals: list[DecompositionProposal] = []
        prediction = np.asarray(model.predict(X), dtype=float)
        scale = max(float(np.std(prediction)), 1e-8)
        permutation = rng.permutation(X.shape[0])
        paired = X[permutation]
        paired_prediction = np.asarray(model.predict(paired), dtype=float)
        for left, right in self._partitions(X.shape[1]):
            mixed_left = paired.copy()
            mixed_left[:, left] = X[:, left]
            mixed_right = X.copy()
            mixed_right[:, left] = paired[:, left]
            pred_left = np.asarray(model.predict(mixed_left), dtype=float)
            pred_right = np.asarray(model.predict(mixed_right), dtype=float)
            finite = (
                np.isfinite(prediction)
                & np.isfinite(paired_prediction)
                & np.isfinite(pred_left)
                & np.isfinite(pred_right)
            )
            support = float(np.mean(finite))
            additive_score = multiplicative_score = 0.0
            if np.any(finite):
                additive_delta = (
                    prediction[finite]
                    + paired_prediction[finite]
                    - pred_left[finite]
                    - pred_right[finite]
                )
                additive_score = float(
                    np.clip(1.0 - float(np.mean(np.abs(additive_delta))) / scale, 0.0, 1.0)
                )
                multiplicative_delta = (
                    prediction[finite] * paired_prediction[finite]
                    - pred_left[finite] * pred_right[finite]
                )
                product_scale = max(
                    float(
                        np.mean(
                            np.abs(prediction[finite] * paired_prediction[finite])
                            + np.abs(pred_left[finite] * pred_right[finite])
                        )
                    ),
                    scale * scale,
                    1e-8,
                )
                multiplicative_score = float(
                    np.clip(
                        1.0
                        - float(np.mean(np.abs(multiplicative_delta))) / product_scale,
                        0.0,
                        1.0,
                    )
                )
            left_relevance = float(np.sum(importance[list(left)]))
            right_relevance = float(np.sum(importance[list(right)]))
            for kind, score in (
                ("additive", additive_score),
                ("multiplicative", multiplicative_score),
            ):
                accepted = (
                    support == 1.0
                    and score >= self.config.separability_min_score
                    and left_relevance >= self.config.relevance_min_fraction
                    and right_relevance >= self.config.relevance_min_fraction
                )
                reason = None
                if not accepted:
                    if support < 1.0:
                        reason = "non_finite_surrogate_prediction"
                    elif min(left_relevance, right_relevance) < self.config.relevance_min_fraction:
                        reason = "irrelevant_partition"
                    else:
                        reason = "weak_separability"
                proposals.append(
                    DecompositionProposal(
                        kind=cast(Any, kind),
                        left_indices=left,
                        right_indices=right,
                        score=score,
                        support_fraction=support,
                        left_relevance=left_relevance,
                        right_relevance=right_relevance,
                        accepted=accepted,
                        rejection_reason=reason,
                    )
                )
        return proposals

    @staticmethod
    def _pair_node(candidate: PairSymmetryCandidate) -> FeatureNode:
        left = FeatureNode.variable(candidate.left_index)
        right = FeatureNode.variable(candidate.right_index)
        if not candidate.parameterized:
            return FeatureNode(candidate.operator, (left, right))
        if candidate.operator in {"add", "sub"}:
            transformed_right = FeatureNode(
                "mul", (FeatureNode.constant(candidate.parameter), right)
            )
        else:
            transformed_right = FeatureNode.power(right, candidate.parameter)
        return FeatureNode(candidate.operator, (left, transformed_right))

    @staticmethod
    def _parameter_name(value: float) -> str:
        text = f"{value:.6g}".replace("-", "m").replace(".", "p")
        return text.replace("+", "")

    def _ensemble_invariance(
        self,
        models: Sequence[Pipeline],
        X: NDArray[np.float64],
        predictions: Sequence[NDArray[np.float64]],
        candidate: PairSymmetryCandidate,
        seed: int,
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
    ) -> tuple[float, float, list[float]]:
        scores: list[float] = []
        supports: list[float] = []
        for model_index, (model, prediction) in enumerate(zip(models, predictions)):
            model_rng = np.random.RandomState(seed + 104729 * model_index)
            score, support = self._invariance_score(
                model,
                X,
                prediction,
                candidate,
                model_rng,
                lower,
                upper,
            )
            scores.append(score)
            supports.append(support)
        return float(np.mean(scores)), float(np.mean(supports)), scores

    def _optimized_parameter(
        self,
        models: Sequence[Pipeline],
        X: NDArray[np.float64],
        predictions: Sequence[NDArray[np.float64]],
        operator: CandidateOperator,
        i: int,
        j: int,
        seed: int,
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
    ) -> tuple[float, float]:
        """Optimize one positive coefficient using construction data only."""

        log_grid = np.linspace(
            np.log(self.config.parameter_min),
            np.log(self.config.parameter_max),
            self.config.parameter_grid_size,
        )

        def objective(log_parameter: float) -> float:
            parameter = float(np.exp(log_parameter))
            candidate = PairSymmetryCandidate(
                operator, i, j, parameter=parameter, parameterized=True
            )
            score, _, _ = self._ensemble_invariance(
                models, X, predictions, candidate, seed, lower, upper
            )
            return -score

        grid_scores = np.asarray([-objective(value) for value in log_grid], dtype=float)
        best_index = int(np.argmax(grid_scores))
        best_log_parameter = float(log_grid[best_index])
        best_score = float(grid_scores[best_index])
        if 0 < best_index < len(log_grid) - 1:
            result = minimize_scalar(
                objective,
                bounds=(float(log_grid[best_index - 1]), float(log_grid[best_index + 1])),
                method="bounded",
                options={"maxiter": 12, "xatol": 1e-3},
            )
            if result.success and -float(result.fun) > best_score:
                best_log_parameter = float(result.x)
                best_score = -float(result.fun)
        return float(np.exp(best_log_parameter)), best_score

    def _pairwise_candidates(
        self,
        models: Sequence[Pipeline],
        construction_X: NDArray[np.float64],
        validation_X: NDArray[np.float64],
        domain_X: NDArray[np.float64],
        construction_predictions: Sequence[NDArray[np.float64]],
        validation_predictions: Sequence[NDArray[np.float64]],
        relevance: NDArray[np.float64],
        names: Sequence[str],
        seed: int,
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
    ) -> list[FeatureProposal]:
        if not self.config.enable_pairwise_symmetry:
            return []
        self.parameter_search_report_ = []
        proposals: list[FeatureProposal] = []
        raw_candidates = self._pair_directions(
            construction_X.shape[1],
            self.config.candidate_operators,
            self.config.max_candidate_pairs,
            relevance,
        )
        fixed_evidence: list[
            tuple[PairSymmetryCandidate, float, float, list[float], float]
        ] = []
        for candidate_index, (operator, i, j) in enumerate(raw_candidates):
            candidate = PairSymmetryCandidate(operator, i, j)
            node = self._pair_node(candidate)
            _, node_domain_valid = self._node_support(node, domain_X)
            node_domain_support = float(np.mean(node_domain_valid))
            unsafe_division = operator == "div" and np.any(
                np.abs(domain_X[:, j]) <= 1e-12
            )
            unsafe_division = unsafe_division or node_domain_support < 1.0
            if unsafe_division:
                construction_score = validation_score = support = 0.0
                construction_scores: list[float] = []
                validation_scores: list[float] = []
            else:
                candidate_seed = seed + 1009 * candidate_index
                construction_score, construction_support, construction_scores = (
                    self._ensemble_invariance(
                        models,
                        construction_X,
                        construction_predictions,
                        candidate,
                        candidate_seed,
                        lower,
                        upper,
                    )
                )
                validation_score, validation_support, validation_scores = (
                    self._ensemble_invariance(
                        models,
                        validation_X,
                        validation_predictions,
                        candidate,
                        candidate_seed + 17,
                        lower,
                        upper,
                    )
                )
                support = min(
                    construction_support, validation_support, node_domain_support
                )
            pair_relevance = float(min(relevance[i], relevance[j]))
            stability = (
                float(
                    np.mean(
                        [
                            min(construction, validation)
                            >= self.config.invariance_min_score
                            for construction, validation in zip(
                                construction_scores, validation_scores
                            )
                        ]
                    )
                )
                if construction_scores
                else 0.0
            )
            accepted = (
                construction_score >= self.config.invariance_min_score
                and validation_score >= self.config.invariance_min_score
                and stability >= self.config.surrogate_stability_min_fraction
                and support >= 0.5
                and pair_relevance >= self.config.relevance_min_fraction
            )
            reason = None
            if not accepted:
                if unsafe_division:
                    reason = "unsafe_division_domain"
                elif support < 0.5:
                    reason = "insufficient_support"
                elif pair_relevance < self.config.relevance_min_fraction:
                    reason = "low_pair_relevance"
                elif stability < self.config.surrogate_stability_min_fraction:
                    reason = "unstable_across_surrogates"
                else:
                    reason = "weak_invariance"
            dimension_valid, dimension_reason = self._dimension_check(node)
            if not dimension_valid:
                accepted = False
                reason = dimension_reason or "dimension_constraint"
            proposals.append(
                FeatureProposal(
                    operator=operator,
                    left_index=i,
                    right_index=j,
                    name=f"afe_{operator}_{i}_{j}",
                    expression=node.expression(names),
                    invariance_score=min(construction_score, validation_score),
                    relevance_score=pair_relevance,
                    support_fraction=support,
                    accepted=accepted,
                    rejection_reason=reason,
                    feature_kind="pairwise_symmetry",
                    depth=node.depth,
                    complexity=self._node_complexity(node),
                    validation_score=validation_score,
                    improvement_score=min(construction_score, validation_score),
                    construction_score=construction_score,
                    stability_fraction=stability,
                    node=node,
                )
            )
            fixed_evidence.append(
                (
                    candidate,
                    construction_score,
                    pair_relevance,
                    construction_scores,
                    support,
                )
            )

        parameterized_sources = sorted(
            (
                evidence
                for evidence in fixed_evidence
                if self.config.enable_parameterized_symmetry
                and evidence[1] >= self.config.parameter_search_min_score
                and evidence[2] >= self.config.relevance_min_fraction
                and evidence[4] >= 0.5
            ),
            key=lambda evidence: (evidence[2], evidence[1]),
            reverse=True,
        )[: self.config.max_parameterized_candidates]
        for parameter_index, (
            fixed,
            fixed_construction_score,
            pair_relevance,
            _,
            _,
        ) in enumerate(
            parameterized_sources
        ):
            parameter_seed = seed + 1_000_003 + 1009 * parameter_index
            parameter, optimized_score = self._optimized_parameter(
                models,
                construction_X,
                construction_predictions,
                fixed.operator,
                fixed.left_index,
                fixed.right_index,
                parameter_seed,
                lower,
                upper,
            )
            search_record: dict[str, Any] = {
                "operator": fixed.operator,
                "input_indices": (fixed.left_index, fixed.right_index),
                "fixed_construction_score": fixed_construction_score,
                "optimized_parameter": parameter,
                "optimized_construction_score": optimized_score,
            }
            if abs(np.log(parameter)) <= self.config.parameter_identity_tolerance:
                search_record["status"] = "identity_parameter"
                self.parameter_search_report_.append(search_record)
                continue
            log_parameter = np.log(parameter)
            log_minimum = np.log(self.config.parameter_min)
            log_maximum = np.log(self.config.parameter_max)
            relative_position = (log_parameter - log_minimum) / (
                log_maximum - log_minimum
            )
            if (
                relative_position <= self.config.parameter_boundary_tolerance
                or relative_position >= 1.0 - self.config.parameter_boundary_tolerance
            ):
                search_record["status"] = "boundary_optimum_rejected"
                self.parameter_search_report_.append(search_record)
                continue
            if optimized_score < (
                fixed_construction_score + self.config.parameter_min_improvement
            ):
                search_record["status"] = "insufficient_parameter_gain"
                self.parameter_search_report_.append(search_record)
                continue
            candidate = replace(
                fixed, parameter=parameter, parameterized=True
            )
            node = self._pair_node(candidate)
            _, node_domain_valid = self._node_support(node, domain_X)
            node_domain_support = float(np.mean(node_domain_valid))
            non_integer_power = abs(parameter - round(parameter)) > 1e-8
            unsafe_power = (
                fixed.operator in {"mul", "div"}
                and non_integer_power
                and np.any(domain_X[:, fixed.right_index] <= 0.0)
            )
            unsafe_division = fixed.operator == "div" and np.any(
                np.abs(domain_X[:, fixed.right_index]) <= 1e-12
            )
            if node_domain_support < 1.0:
                if fixed.operator == "mul":
                    unsafe_power = True
                elif fixed.operator == "div":
                    unsafe_division = True
            if unsafe_power or unsafe_division:
                construction_score = validation_score = support = stability = 0.0
            else:
                construction_score, construction_support, construction_scores = (
                    self._ensemble_invariance(
                        models,
                        construction_X,
                        construction_predictions,
                        candidate,
                        parameter_seed,
                        lower,
                        upper,
                    )
                )
                validation_score, validation_support, validation_scores = (
                    self._ensemble_invariance(
                        models,
                        validation_X,
                        validation_predictions,
                        candidate,
                        parameter_seed + 17,
                        lower,
                        upper,
                    )
                )
                support = min(
                    construction_support, validation_support, node_domain_support
                )
                stability = float(
                    np.mean(
                        [
                            min(construction, validation)
                            >= self.config.invariance_min_score
                            for construction, validation in zip(
                                construction_scores, validation_scores
                            )
                        ]
                    )
                )
            accepted = (
                construction_score >= self.config.invariance_min_score
                and validation_score >= self.config.invariance_min_score
                and stability >= self.config.surrogate_stability_min_fraction
                and support >= 0.5
            )
            reason = None
            if not accepted:
                if unsafe_power:
                    reason = "unsafe_parameterized_power_domain"
                elif unsafe_division:
                    reason = "unsafe_division_domain"
                elif support < 0.5:
                    reason = "insufficient_support"
                elif stability < self.config.surrogate_stability_min_fraction:
                    reason = "unstable_across_surrogates"
                else:
                    reason = "weak_parameterized_invariance"
            parameter_name = self._parameter_name(parameter)
            dimension_valid, dimension_reason = self._dimension_check(node)
            if not dimension_valid:
                accepted = False
                reason = dimension_reason or "dimension_constraint"
            proposals.append(
                FeatureProposal(
                    operator=f"parameterized_{fixed.operator}",
                    left_index=fixed.left_index,
                    right_index=fixed.right_index,
                    name=(
                        f"afe_{fixed.operator}_{fixed.left_index}_"
                        f"{fixed.right_index}_a_{parameter_name}"
                    ),
                    expression=node.expression(names),
                    invariance_score=min(construction_score, validation_score),
                    relevance_score=pair_relevance,
                    support_fraction=support,
                    accepted=accepted,
                    rejection_reason=reason,
                    feature_kind="pairwise_symmetry",
                    depth=node.depth,
                    complexity=self._node_complexity(node),
                    validation_score=validation_score,
                    improvement_score=min(construction_score, validation_score),
                    construction_score=construction_score,
                    stability_fraction=stability,
                    parameter=parameter,
                    node=node,
                )
            )
            search_record["status"] = "accepted" if accepted else "rejected"
            search_record["validation_score"] = validation_score
            search_record["stability_fraction"] = stability
            search_record["rejection_reason"] = reason
            self.parameter_search_report_.append(search_record)
        best_by_pair: dict[tuple[int, int], float] = {}
        for proposal in proposals:
            if proposal.accepted:
                pair = cast(
                    tuple[int, int],
                    tuple(sorted((proposal.left_index, proposal.right_index))),
                )
                best_by_pair[pair] = max(
                    best_by_pair.get(pair, 0.0), proposal.invariance_score
                )
        return [
            replace(
                proposal,
                accepted=False,
                rejection_reason="dominated_by_stronger_invariance",
            )
            if proposal.accepted
            and proposal.invariance_score
            < best_by_pair[
                cast(
                    tuple[int, int],
                    tuple(sorted((proposal.left_index, proposal.right_index))),
                )
            ]
            - self.config.invariance_relative_tolerance
            else proposal
            for proposal in proposals
        ]

    def _selection_score(self, proposal: FeatureProposal) -> float:
        if proposal.feature_kind == "pairwise_symmetry":
            evidence = (
                0.55 * proposal.construction_score
                + 0.25 * proposal.relevance_score
                + 0.20 * proposal.stability_fraction
            )
        else:
            evidence = 0.7 * proposal.construction_score + 0.3 * max(
                proposal.improvement_score, 0.0
            )
        return evidence - self.config.complexity_penalty * proposal.complexity

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        variable_names: Sequence[str] | None = None,
    ) -> SurrogateFeatureEngineer:
        """Fit the surrogate and discover structural candidates."""

        self._validate_config(self.config)
        # A fit is a fresh search. Do not expose diagnostics from a previous fit
        # when the new configuration disables or skips those search branches.
        self.parameter_search_report_ = []
        self.composition_search_layers_ = []
        if not self.config.enabled:
            self.proposals_: list[FeatureProposal] = []
            self.accepted_proposals_: list[FeatureProposal] = []
            self.decomposition_proposals_: list[DecompositionProposal] = []
            self.report_ = {"status": "disabled", "candidates": []}
            return self
        values = self._as_float_matrix(X)
        target = self._as_target(y, values.shape[0])
        names = list(variable_names or [f"x{i}" for i in range(values.shape[1])])
        if len(names) != values.shape[1] or len(set(names)) != len(names):
            raise ValueError("variable_names must be unique and match X columns")
        self.n_features_in_ = values.shape[1]
        self.variable_names_in_ = tuple(names)
        self.complexity_spec_ = self.complexity_spec or FeatureComplexitySpec.from_user(
            values.shape[1]
        )
        if len(self.complexity_spec_.variable_complexities) != values.shape[1]:
            raise ValueError(
                "complexity specification must match the number of input features"
            )
        self.dimension_spec_ = self.dimension_spec

        seed_rng = check_random_state(self.random_state)
        seed = int(seed_rng.randint(0, 2**31 - 1))
        train, holdout = train_test_split(
            np.arange(values.shape[0]),
            test_size=(
                self.config.construction_fraction + self.config.validation_fraction
            ),
            random_state=seed,
        )
        validation_share = self.config.validation_fraction / (
            self.config.construction_fraction + self.config.validation_fraction
        )
        construction, validation = train_test_split(
            holdout, test_size=validation_share, random_state=seed + 1
        )
        fitted_models: list[Pipeline] = []
        model_validation_predictions: list[NDArray[np.float64]] = []
        model_validation_r2: list[float] = []
        model_seeds: list[int] = []
        for model_index in range(self.config.surrogate_ensemble_size):
            model_seed = seed + 7919 * model_index
            model = self._make_surrogate(model_seed)
            model.fit(values[train], target[train])
            validation_prediction = np.asarray(
                model.predict(values[validation]), dtype=float
            )
            fitted_models.append(model)
            model_validation_predictions.append(validation_prediction)
            model_validation_r2.append(
                float(r2_score(target[validation], validation_prediction))
            )
            model_seeds.append(model_seed)
        good_model_indices = [
            index
            for index, score in enumerate(model_validation_r2)
            if np.isfinite(score) and score >= self.config.surrogate_min_r2
        ]
        quality_fraction = len(good_model_indices) / len(fitted_models)
        baseline_r2 = float(np.mean(model_validation_r2))
        self.surrogates_ = fitted_models
        self.surrogate_ = fitted_models[0]
        self.surrogate_r2_ = baseline_r2
        self.surrogate_validation_scores_ = tuple(model_validation_r2)
        if quality_fraction < self.config.surrogate_stability_min_fraction:
            self.proposals_ = []
            self.accepted_proposals_ = []
            self.decomposition_proposals_ = []
            self.report_ = {
                "status": "surrogate_quality_insufficient",
                "surrogate_r2": baseline_r2,
                "surrogate_validation_r2": model_validation_r2,
                "surrogate_quality_fraction": quality_fraction,
                "candidates": [],
                "separability": [],
            }
            return self

        models = [fitted_models[index] for index in good_model_indices]
        construction_predictions = [
            np.asarray(model.predict(values[construction]), dtype=float)
            for model in models
        ]
        validation_predictions = [
            model_validation_predictions[index] for index in good_model_indices
        ]

        raw_importance = np.mean(
            [
                permutation_importance(
                    model,
                    values[construction],
                    target[construction],
                    n_repeats=3,
                    random_state=seed + model_index,
                ).importances_mean
                for model_index, model in enumerate(models)
            ],
            axis=0,
        )
        raw_importance = np.maximum(np.asarray(raw_importance, dtype=float), 0.0)
        relevance = raw_importance / max(float(np.max(raw_importance)), 1e-12)
        normalized_importance = raw_importance / max(float(np.sum(raw_importance)), 1e-12)
        lower, upper = self._support_bounds(values)
        pairwise = self._pairwise_candidates(
            models,
            values[construction],
            values[validation],
            values,
            construction_predictions,
            validation_predictions,
            relevance,
            names,
            seed,
            lower,
            upper,
        )
        compositions = self._composition_candidates(
            values[construction],
            values,
            target[construction],
            values[validation],
            target[validation],
            names,
            pairwise,
        )
        decompositions = self._separability_candidates(
            models[0], values[construction], normalized_importance, seed_rng
        )
        proposals = pairwise + compositions
        selected = sorted(
            (proposal for proposal in proposals if proposal.accepted),
            key=self._selection_score,
            reverse=True,
        )[: self.config.max_generated_features]
        selected_signatures = {
            proposal.node.signature if proposal.node is not None else proposal.name
            for proposal in selected
        }
        self.proposals_ = [
            replace(
                proposal,
                accepted=False,
                rejection_reason="generated_feature_budget",
            )
            if proposal.accepted
            and (proposal.node.signature if proposal.node is not None else proposal.name)
            not in selected_signatures
            else proposal
            for proposal in proposals
        ]
        self.accepted_proposals_ = [
            proposal for proposal in self.proposals_ if proposal.accepted
        ]
        self.decomposition_proposals_ = decompositions
        records = [proposal.to_record() for proposal in self.proposals_]
        self.report_ = {
            "status": "ok",
            "algorithm": "ai_feynman_inspired_feature_construction_v5",
            "composition_operators": list(self._composition_operator_pool()),
            "power_exponents": [float(value) for value in self.config.power_exponents],
            "surrogate_r2": baseline_r2,
            "surrogate_validation_r2": model_validation_r2,
            "surrogate_quality_fraction": quality_fraction,
            "surrogate_ensemble_size": len(fitted_models),
            "accepted_surrogate_count": len(models),
            "model_seeds": model_seeds,
            "perturbation_scales": list(self.config.perturbation_scales),
            "split_sizes": {
                "train": len(train),
                "construction": len(construction),
                "validation": len(validation),
            },
            "candidate_count": len(self.proposals_),
            "accepted_count": len(self.accepted_proposals_),
            "candidates": records,
            "pairwise_symmetries": [
                record for record in records if record["feature_kind"] == "pairwise_symmetry"
            ],
            "parameterized_symmetries": [
                record for record in records if record["parameter"] is not None
            ],
            "parameter_search": self.parameter_search_report_,
            "unary_compositions": [
                record for record in records if record["feature_kind"] == "unary_composition"
            ],
            "recursive_compositions": [
                record
                for record in records
                if record["feature_kind"] == "recursive_composition"
            ],
            "composition_search": self.composition_search_layers_,
            "separability": [proposal.to_record() for proposal in decompositions],
            "accepted_decomposition_count": sum(
                proposal.accepted for proposal in decompositions
            ),
            "reduction_path": [
                {
                    "name": proposal.name,
                    "expression": proposal.expression,
                    "depth": proposal.depth,
                    "input_indices": proposal.input_indices,
                }
                for proposal in self.accepted_proposals_
            ],
            "limitations": [
                "separability_is_diagnostic_only",
                "no_recursive_mysrcore_invocation",
                "empirical_formula_mode_only_at_regressor_level",
                "non_integer_parameterized_powers_require_positive_base",
                "validation_split_is_internal_not_external_test_data",
            ],
        }
        return self

    def transform(self, X: Any, *, augment: bool = True) -> NDArray[np.float64]:
        """Replay accepted candidates on new data."""

        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("SurrogateFeatureEngineer must be fitted before transform")
        values = self._as_transform_matrix(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("X does not match the fitted feature shape")
        if not augment or not self.accepted_proposals_:
            return values
        columns = [values]
        for proposal in self.accepted_proposals_:
            columns.append(proposal.transform(values).reshape(-1, 1))
        transformed = np.column_stack(columns)
        if not np.all(np.isfinite(transformed)):
            raise ValueError("accepted feature replay produced non-finite values")
        return transformed

    def get_feature_names_out(self) -> list[str]:
        if not hasattr(self, "variable_names_in_"):
            raise RuntimeError("SurrogateFeatureEngineer must be fitted first")
        return list(self.variable_names_in_) + [
            proposal.name for proposal in self.accepted_proposals_
        ]
