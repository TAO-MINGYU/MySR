"""Lightweight FEAT-like evolutionary representation learning for MySR.

This is an independent Apache-2.0 implementation informed by the published FEAT
method and its public documentation. It does not copy, import, or vendor source
code from the GNU GPLv3-licensed official FEAT implementation.

The defining unit of search is a bundle of replayable expression trees. A Ridge
regressor evaluates each bundle, epsilon-lexicase uses per-case errors for parent
selection, and non-dominated error-complexity sorting controls survival.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.utils import check_random_state  # type: ignore

from .feature_engineering import (
    FEATEngineConfig,
    FeatureComplexitySpec,
    FeatureDimensionSpec,
    FeatureNode,
    FeatureProposal,
)


@dataclass(frozen=True)
class FeatureBundleProposal:
    """A jointly evaluated set of evolved feature expressions."""

    nodes: tuple[FeatureNode, ...] = field(repr=False, compare=False)
    names: tuple[str, ...]
    expressions: tuple[str, ...]
    downstream_columns: tuple[str, ...]
    construction_nmse: float
    validation_nmse: float
    baseline_validation_nmse: float
    improvement_score: float
    complexity: float
    generation: int
    coefficients: tuple[float, ...]
    accepted: bool
    rejection_reason: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "expressions": list(self.expressions),
            "downstream_columns": list(self.downstream_columns),
            "construction_nmse": self.construction_nmse,
            "validation_nmse": self.validation_nmse,
            "baseline_validation_nmse": self.baseline_validation_nmse,
            "improvement_score": self.improvement_score,
            "complexity": self.complexity,
            "generation": self.generation,
            "coefficients": list(self.coefficients),
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class _BundleIndividual:
    nodes: tuple[FeatureNode, ...]
    generation: int
    construction_nmse: float = float("inf")
    validation_nmse: float = float("inf")
    complexity: float = 0.0
    coefficients: tuple[float, ...] = ()
    per_case_errors: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=float), repr=False
    )
    rejection_reason: str | None = None

    @property
    def signature(self) -> tuple[str, ...]:
        return tuple(sorted(node.signature for node in self.nodes))


class FEATLikeFeatureEngineer:
    """Evolve compact bundles of interpretable features for a downstream Ridge."""

    def __init__(
        self,
        config: FEATEngineConfig | None = None,
        *,
        random_state: int | np.random.RandomState | None = None,
        complexity_spec: FeatureComplexitySpec | None = None,
        dimension_spec: FeatureDimensionSpec | None = None,
    ) -> None:
        self.config = config or FEATEngineConfig()
        self.random_state = random_state
        self.complexity_spec = complexity_spec
        self.dimension_spec = dimension_spec
        self.report_: dict[str, Any]

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
    def _validate_config(config: FEATEngineConfig) -> None:
        binary_allowed = {"add", "sub", "mul", "div"}
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
        if not set(config.binary_operators).issubset(binary_allowed):
            raise ValueError("FEAT-like binary_operators contains an unsupported operator")
        if not set(config.unary_operators).issubset(unary_allowed):
            raise ValueError("FEAT-like unary_operators contains an unsupported operator")
        positive_integer_fields = {
            "population_size": config.population_size,
            "generations": config.generations,
            "max_evaluations": config.max_evaluations,
            "max_depth": config.max_depth,
            "max_bundle_size": config.max_bundle_size,
            "max_generated_features": config.max_generated_features,
            "max_seed_nodes": config.max_seed_nodes,
        }
        for name, value in positive_integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if config.population_size < 4:
            raise ValueError("population_size must be at least 4")
        if config.max_evaluations < config.population_size:
            raise ValueError("max_evaluations must be at least population_size")
        if not 0.0 < config.construction_fraction < 0.5:
            raise ValueError("construction_fraction must be in (0, 0.5)")
        if not 0.0 < config.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0, 0.5)")
        if config.construction_fraction + config.validation_fraction >= 0.8:
            raise ValueError("construction and validation fractions leave too little training data")
        rates: dict[str, float] = {
            "crossover_rate": config.crossover_rate,
            "mutation_rate": config.mutation_rate,
        }
        for rate_name, rate_value in rates.items():
            if not 0.0 <= rate_value <= 1.0:
                raise ValueError(f"{rate_name} must be in [0, 1]")
        if config.ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be non-negative")
        if config.min_validation_improvement < 0.0:
            raise ValueError("min_validation_improvement must be non-negative")
        if config.max_abs_value <= 0.0 or not np.isfinite(config.max_abs_value):
            raise ValueError("max_abs_value must be a positive finite number")
        if config.min_feature_std < 0.0 or not np.isfinite(config.min_feature_std):
            raise ValueError("min_feature_std must be a finite non-negative number")
        if config.complexity_tiebreak < 0.0 or not np.isfinite(config.complexity_tiebreak):
            raise ValueError(
                "complexity_tiebreak must be a finite non-negative number"
            )
        if not 0.0 < config.duplicate_correlation <= 1.0:
            raise ValueError("duplicate_correlation must be in (0, 1]")

    @staticmethod
    def _as_float_matrix(X: Any) -> NDArray[np.float64]:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError("X must be a two-dimensional numeric matrix")
        if values.shape[0] < 40:
            raise ValueError("at least 40 samples are required for FEAT-like search")
        if values.shape[1] < 1:
            raise ValueError("X must contain at least one feature")
        if not np.all(np.isfinite(values)):
            raise ValueError("X must contain only finite values")
        return values

    @staticmethod
    def _as_target(y: Any, n_samples: int) -> NDArray[np.float64]:
        target = np.asarray(y, dtype=float)
        if target.ndim != 1 or target.shape[0] != n_samples:
            raise ValueError("FEAT-like search currently requires one target per row")
        if not np.all(np.isfinite(target)):
            raise ValueError("y must contain only finite values")
        if float(np.std(target)) <= 1.0e-12:
            raise ValueError("y must not be constant")
        return target

    def _node_support(
        self, node: FeatureNode, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Evaluate a node while retaining validity of every intermediate subtree."""

        child_values: list[NDArray[np.float64]] = []
        valid = np.ones(X.shape[0], dtype=bool)
        for child in node.children:
            child_value, child_valid = self._node_support(child, X)
            child_values.append(child_value)
            valid &= child_valid

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            values = node.evaluate(X)
            valid &= np.isfinite(values)
            if node.operator in {"reciprocal", "log_abs"}:
                valid &= np.abs(child_values[0]) > 1.0e-12
            if node.operator == "div":
                valid &= np.abs(child_values[1]) > 1.0e-12
            if node.operator == "normalized_sub":
                valid &= np.abs(child_values[0] + child_values[1]) > 1.0e-12
            if node.operator in {"log_ratio", "exp_ratio", "sin_ratio", "cos_ratio"}:
                numerator = child_values[0]
                denominator = child_values[1]
                valid &= np.abs(denominator) > 1.0e-12
                if node.operator == "log_ratio":
                    valid &= np.abs(numerator) > 1.0e-12
                if node.operator == "exp_ratio":
                    valid &= np.abs(numerator / denominator) <= 700.0
            if node.operator == "power" and node.value is not None:
                base = child_values[0]
                exponent = float(node.value)
                if abs(exponent - round(exponent)) > 1.0e-10:
                    valid &= base >= 0.0
                if exponent < 0.0:
                    valid &= np.abs(base) > 1.0e-12
            if node.operator == "exp":
                valid &= np.abs(child_values[0]) <= 700.0
        return np.asarray(values, dtype=float), valid

    @staticmethod
    def _nmse(y: NDArray[np.float64], prediction: NDArray[np.float64]) -> float:
        denominator = max(float(np.var(y)), 1.0e-12)
        return float(np.mean((prediction - y) ** 2) / denominator)

    def _feature_matrix(
        self,
        nodes: tuple[FeatureNode, ...],
        X: NDArray[np.float64],
        *,
        include_raw: bool = False,
    ) -> NDArray[np.float64] | None:
        raw_columns: list[NDArray[np.float64]] = (
            [np.asarray(X[:, index], dtype=float) for index in range(X.shape[1])]
            if include_raw
            else []
        )
        generated_columns: list[NDArray[np.float64]] = []
        for node in nodes:
            values, support = self._node_support(node, X)
            if not np.all(support):
                return None
            if not np.all(np.isfinite(values)):
                return None
            if float(np.max(np.abs(values))) > self.config.max_abs_value:
                return None
            if float(np.std(values)) < self.config.min_feature_std:
                return None
            generated_columns.append(values)
        columns = raw_columns + generated_columns
        if not columns:
            return None
        matrix = np.column_stack(columns)
        # Only generated columns are candidates for duplicate rejection. Raw
        # input columns may legitimately be correlated (or even identical);
        # rejecting a bundle because of raw/raw correlation makes FEAT-like
        # search unusable on redundant user data. Keep the historical guard for
        # generated/generated and generated/raw near-duplicates.
        if generated_columns:
            generated_matrix = np.column_stack(generated_columns)
            correlations: list[NDArray[np.float64]] = []
            if len(generated_columns) > 1:
                correlations.append(
                    np.asarray(
                        np.corrcoef(generated_matrix, rowvar=False), dtype=float
                    )[np.triu_indices(len(generated_columns), k=1)]
                )
            if raw_columns:
                cross_correlation = np.asarray(
                    np.corrcoef(
                        np.column_stack(raw_columns + generated_columns),
                        rowvar=False,
                    ),
                    dtype=float,
                )
                raw_count = len(raw_columns)
                correlations.append(
                    cross_correlation[:raw_count, raw_count:].reshape(-1)
                )
            upper = np.abs(np.concatenate(correlations)) if correlations else np.empty(0)
            if np.any(np.isfinite(upper) & (upper >= self.config.duplicate_correlation)):
                return None
        return np.asarray(matrix, dtype=float)

    def _score_matrices(
        self,
        train_matrix: NDArray[np.float64],
        construction_matrix: NDArray[np.float64],
        validation_matrix: NDArray[np.float64],
    ) -> tuple[float, float, tuple[float, ...], NDArray[np.float64]]:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=self.config.ridge_alpha)),
            ]
        )
        model.fit(train_matrix, self._target_train)
        construction_prediction = np.asarray(
            model.predict(construction_matrix), dtype=float
        )
        validation_prediction = np.asarray(model.predict(validation_matrix), dtype=float)
        construction_nmse = self._nmse(
            self._target_construction, construction_prediction
        )
        validation_nmse = self._nmse(self._target_validation, validation_prediction)
        variance = max(float(np.var(self._target_construction)), 1.0e-12)
        per_case = (
            (construction_prediction - self._target_construction) ** 2 / variance
        )
        coefficients = tuple(
            float(value)
            for value in np.ravel(model.named_steps["ridge"].coef_)
        )
        return construction_nmse, validation_nmse, coefficients, per_case

    def _residual_candidate_order(
        self,
        base: _BundleIndividual,
        candidates: list[FeatureNode],
    ) -> list[FeatureNode]:
        """Order candidates by correlation with the current construction residual."""

        base_matrix = self._feature_matrix(
            base.nodes, self._values_construction, include_raw=True
        )
        if base_matrix is None:
            return candidates
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=self.config.ridge_alpha)),
            ]
        )
        try:
            model.fit(base_matrix, self._target_construction)
            residual = self._target_construction - np.asarray(
                model.predict(base_matrix), dtype=float
            )
        except (FloatingPointError, ValueError):
            return candidates
        residual_std = float(np.std(residual))
        if residual_std <= 1.0e-12:
            return candidates
        ranked: list[tuple[float, FeatureNode]] = []
        for candidate in candidates:
            if candidate.signature in base.signature:
                continue
            values, support = self._node_support(
                candidate, self._values_construction
            )
            candidate_std = float(np.std(values))
            if (
                not np.all(support)
                or not np.all(np.isfinite(values))
                or candidate_std <= self.config.min_feature_std
            ):
                continue
            correlation = float(np.corrcoef(values, residual)[0, 1])
            if np.isfinite(correlation):
                ranked.append((abs(correlation), candidate))
        ranked.sort(
            key=lambda item: (item[0], -self._node_complexity(item[1])),
            reverse=True,
        )
        return [candidate for _, candidate in ranked]

    def _evaluate_nodes(
        self, nodes: tuple[FeatureNode, ...], generation: int
    ) -> _BundleIndividual:
        nodes = self._normalize_nodes(nodes)
        signature = tuple(node.signature for node in nodes)
        cached = self._evaluation_cache.get(signature)
        if cached is not None:
            return cached
        individual = _BundleIndividual(nodes=nodes, generation=generation)
        self._evaluations += 1
        invalid_nodes = [
            node
            for node in nodes
            if node.operator != "variable" and not self._dimension_check(node)[0]
        ]
        if invalid_nodes:
            individual.rejection_reason = "dimension_constraint"
            self._evaluation_cache[signature] = individual
            return individual
        train_matrix = self._feature_matrix(
            nodes, self._values_train, include_raw=True
        )
        construction_matrix = self._feature_matrix(
            nodes, self._values_construction, include_raw=True
        )
        validation_matrix = self._feature_matrix(
            nodes, self._values_validation, include_raw=True
        )
        if train_matrix is None or construction_matrix is None or validation_matrix is None:
            individual.rejection_reason = "invalid_or_redundant_feature_matrix"
        else:
            try:
                (
                    individual.construction_nmse,
                    individual.validation_nmse,
                    individual.coefficients,
                    individual.per_case_errors,
                ) = self._score_matrices(
                    train_matrix, construction_matrix, validation_matrix
                )
                individual.complexity = sum(
                    self._node_complexity(node) for node in nodes
                )
            except (FloatingPointError, ValueError):
                individual.rejection_reason = "downstream_ridge_failed"
        self._evaluation_cache[signature] = individual
        return individual

    @staticmethod
    def _normalize_nodes(nodes: tuple[FeatureNode, ...]) -> tuple[FeatureNode, ...]:
        unique = {node.signature: node for node in nodes}
        return tuple(unique[key] for key in sorted(unique))

    def _random_node(
        self, rng: np.random.RandomState, depth: int
    ) -> FeatureNode:
        if depth <= 0 or rng.rand() < 0.32:
            return FeatureNode.variable(int(rng.randint(self.n_features_in_)))
        can_unary = bool(self.config.unary_operators)
        can_binary = bool(self.config.binary_operators)
        if can_unary and (not can_binary or rng.rand() < 0.48):
            operator = self.config.unary_operators[
                int(rng.randint(len(self.config.unary_operators)))
            ]
            return FeatureNode(operator, (self._random_node(rng, depth - 1),))
        if can_binary:
            binary_operator = self.config.binary_operators[
                int(rng.randint(len(self.config.binary_operators)))
            ]
            return FeatureNode(
                binary_operator,
                (
                    self._random_node(rng, depth - 1),
                    self._random_node(rng, depth - 1),
                ),
            )
        return FeatureNode.variable(int(rng.randint(self.n_features_in_)))

    def _make_seed_nodes(
        self, values: NDArray[np.float64], rng: np.random.RandomState
    ) -> list[FeatureNode]:
        candidates: list[FeatureNode] = [
            FeatureNode.variable(index) for index in range(self.n_features_in_)
        ]
        for index in range(self.n_features_in_):
            variable = FeatureNode.variable(index)
            candidates.extend(
                FeatureNode(operator, (variable,))
                for operator in self.config.unary_operators
            )
        for left in range(self.n_features_in_):
            for right in range(left + 1, self.n_features_in_):
                for operator in self.config.binary_operators:
                    candidates.append(
                        FeatureNode(
                            operator,
                            (FeatureNode.variable(left), FeatureNode.variable(right)),
                        )
                    )
                    if operator in {"sub", "div"}:
                        candidates.append(
                            FeatureNode(
                                operator,
                                (FeatureNode.variable(right), FeatureNode.variable(left)),
                            )
                        )
        # Materialize the most useful depth-2 compositions deterministically.
        # This keeps common structures such as (x1 - x2)^2 and sin(x1 + x2)
        # discoverable without relying on a lucky random tree draw.
        unary_priority = {
            "square": 0,
            "sin": 1,
            "cos": 2,
            "cube": 3,
            "abs": 4,
            "sqrt_abs": 5,
            "log_abs": 6,
            "reciprocal": 7,
            "exp": 8,
        }
        composition_unary_operators = sorted(
            self.config.unary_operators,
            key=lambda operator: unary_priority.get(operator, 99),
        )
        pair_nodes = [
            node
            for node in candidates
            if node.operator in self.config.binary_operators and node.depth == 1
        ]
        for pair_node in pair_nodes:
            for unary_operator in composition_unary_operators:
                candidates.append(FeatureNode(unary_operator, (pair_node,)))
        unary_nodes = [
            node
            for node in candidates
            if node.operator in self.config.unary_operators and node.depth == 1
        ]
        for left_node in unary_nodes:
            for right_node in unary_nodes:
                if left_node.signature >= right_node.signature:
                    continue
                for binary_operator in self.config.binary_operators:
                    candidates.append(
                        FeatureNode(binary_operator, (left_node, right_node))
                    )
        attempts = 0
        while len(candidates) < self.config.max_seed_nodes * 2 and attempts < 500:
            candidates.append(self._random_node(rng, self.config.max_depth))
            attempts += 1
        valid: list[FeatureNode] = []
        seen: set[str] = set()
        for node in candidates:
            if node.signature in seen or node.depth > self.config.max_depth:
                continue
            if self._feature_matrix((node,), values) is None:
                continue
            seen.add(node.signature)
            valid.append(node)
            if len(valid) >= self.config.max_seed_nodes:
                break
        return valid

    def _lexicase_parent(
        self,
        population: list[_BundleIndividual],
        rng: np.random.RandomState,
    ) -> _BundleIndividual:
        pool = [
            individual
            for individual in population
            if np.isfinite(individual.construction_nmse)
            and individual.per_case_errors.size > 0
        ]
        if not pool:
            return population[int(rng.randint(len(population)))]
        errors = np.vstack([individual.per_case_errors for individual in pool])
        case_order = rng.permutation(errors.shape[1])
        active = np.arange(len(pool))
        for case in case_order:
            case_errors = errors[active, case]
            best = float(np.min(case_errors))
            epsilon = float(np.median(np.abs(case_errors - np.median(case_errors))))
            active = active[case_errors <= best + epsilon + 1.0e-15]
            if len(active) <= 1:
                break
        return pool[int(active[int(rng.randint(len(active)))])]

    def _mutate_nodes(
        self,
        nodes: tuple[FeatureNode, ...],
        node_library: list[FeatureNode],
        rng: np.random.RandomState,
    ) -> tuple[FeatureNode, ...]:
        result = list(nodes)
        actions = ["replace", "add", "wrap", "combine"]
        if len(result) > 1:
            actions.append("delete")
        action = actions[int(rng.randint(len(actions)))]
        if action == "delete":
            result.pop(int(rng.randint(len(result))))
        elif action == "add" and len(result) < self.config.max_bundle_size:
            result.append(node_library[int(rng.randint(len(node_library)))])
        elif action == "wrap" and self.config.unary_operators:
            location = int(rng.randint(len(result)))
            operator = self.config.unary_operators[
                int(rng.randint(len(self.config.unary_operators)))
            ]
            wrapped = FeatureNode(operator, (result[location],))
            if wrapped.depth <= self.config.max_depth:
                result[location] = wrapped
        elif action == "combine" and self.config.binary_operators:
            location = int(rng.randint(len(result)))
            binary_operator = self.config.binary_operators[
                int(rng.randint(len(self.config.binary_operators)))
            ]
            other = node_library[int(rng.randint(len(node_library)))]
            combined = FeatureNode(binary_operator, (result[location], other))
            if combined.depth <= self.config.max_depth:
                result[location] = combined
        else:
            result[int(rng.randint(len(result)))] = node_library[
                int(rng.randint(len(node_library)))
            ]
        return self._normalize_nodes(tuple(result[: self.config.max_bundle_size]))

    def _crossover_nodes(
        self,
        left: tuple[FeatureNode, ...],
        right: tuple[FeatureNode, ...],
        rng: np.random.RandomState,
    ) -> tuple[FeatureNode, ...]:
        union = list(self._normalize_nodes(left + right))
        union = [union[int(index)] for index in rng.permutation(len(union))]
        size = int(rng.randint(1, min(len(union), self.config.max_bundle_size) + 1))
        return self._normalize_nodes(tuple(union[:size]))

    @staticmethod
    def _dominates(left: _BundleIndividual, right: _BundleIndividual) -> bool:
        no_worse = (
            left.construction_nmse <= right.construction_nmse
            and left.complexity <= right.complexity
        )
        strictly_better = (
            left.construction_nmse < right.construction_nmse
            or left.complexity < right.complexity
        )
        return no_worse and strictly_better

    def _pareto_fronts(
        self, population: list[_BundleIndividual]
    ) -> list[list[_BundleIndividual]]:
        remaining = [
            individual
            for individual in population
            if np.isfinite(individual.construction_nmse)
        ]
        fronts: list[list[_BundleIndividual]] = []
        while remaining:
            front = [
                candidate
                for candidate in remaining
                if not any(
                    self._dominates(other, candidate)
                    for other in remaining
                    if other is not candidate
                )
            ]
            if not front:
                break
            fronts.append(front)
            front_ids = {id(individual) for individual in front}
            remaining = [
                individual
                for individual in remaining
                if id(individual) not in front_ids
            ]
        return fronts

    @staticmethod
    def _crowding(front: list[_BundleIndividual]) -> dict[int, float]:
        distance = {id(individual): 0.0 for individual in front}
        if len(front) <= 2:
            return {id(individual): float("inf") for individual in front}
        objectives = (
            lambda individual: individual.construction_nmse,
            lambda individual: float(individual.complexity),
        )
        for objective in objectives:
            ordered = sorted(front, key=objective)
            distance[id(ordered[0])] = float("inf")
            distance[id(ordered[-1])] = float("inf")
            span = objective(ordered[-1]) - objective(ordered[0])
            if span <= 0.0:
                continue
            for index in range(1, len(ordered) - 1):
                distance[id(ordered[index])] += (
                    objective(ordered[index + 1]) - objective(ordered[index - 1])
                ) / span
        return distance

    def _survive(
        self, population: list[_BundleIndividual]
    ) -> list[_BundleIndividual]:
        unique = {individual.signature: individual for individual in population}
        candidates = list(unique.values())
        survivors: list[_BundleIndividual] = []
        for front in self._pareto_fronts(candidates):
            remaining = self.config.population_size - len(survivors)
            if remaining <= 0:
                break
            if len(front) <= remaining:
                survivors.extend(front)
            else:
                crowding = self._crowding(front)
                survivors.extend(
                    sorted(
                        front,
                        key=lambda individual: crowding[id(individual)],
                        reverse=True,
                    )[:remaining]
                )
        if len(survivors) < self.config.population_size:
            survivor_signatures = {individual.signature for individual in survivors}
            extras = sorted(
                (
                    candidate
                    for candidate in candidates
                    if candidate.signature not in survivor_signatures
                ),
                key=lambda individual: (
                    individual.construction_nmse,
                    individual.complexity,
                ),
            )
            survivors.extend(extras[: self.config.population_size - len(survivors)])
        return survivors

    def _raw_baseline(self) -> tuple[float, float]:
        construction_nmse, validation_nmse, _, _ = self._score_matrices(
            self._values_train,
            self._values_construction,
            self._values_validation,
        )
        return construction_nmse, validation_nmse

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        variable_names: list[str] | tuple[str, ...] | None = None,
    ) -> FEATLikeFeatureEngineer:
        """Run a bounded evolutionary search for a useful feature bundle."""

        self._validate_config(self.config)
        if not self.config.enabled:
            self.proposals_: list[FeatureProposal] = []
            self.accepted_proposals_: list[FeatureProposal] = []
            self.bundle_proposals_: list[FeatureBundleProposal] = []
            self.accepted_bundles_: list[FeatureBundleProposal] = []
            self.report_ = {"status": "disabled", "algorithm": "feat_like_v1"}
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
        rng = check_random_state(self.random_state)
        seed = int(rng.randint(0, 2**31 - 1))
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
        self._values_train = values[train]
        self._values_construction = values[construction]
        self._values_validation = values[validation]
        self._target_train = target[train]
        self._target_construction = target[construction]
        self._target_validation = target[validation]
        baseline_construction_nmse, baseline_validation_nmse = self._raw_baseline()

        self._evaluation_cache: dict[tuple[str, ...], _BundleIndividual] = {}
        self._evaluations = 0
        node_library = self._make_seed_nodes(values, rng)
        singleton_evaluations: list[_BundleIndividual] = []
        singleton_budget = min(
            len(node_library),
            max(self.config.population_size, self.config.max_evaluations // 2),
        )
        for node in node_library[:singleton_budget]:
            if self._evaluations >= self.config.max_evaluations:
                break
            singleton_evaluations.append(self._evaluate_nodes((node,), generation=0))
        ranked_nodes = [
            individual.nodes[0]
            for individual in sorted(
                (
                    individual
                    for individual in singleton_evaluations
                    if np.isfinite(individual.construction_nmse)
                ),
                key=lambda individual: (
                    individual.construction_nmse,
                    individual.complexity,
                ),
            )
        ]
        if not ranked_nodes:
            self.proposals_ = []
            self.accepted_proposals_ = []
            self.bundle_proposals_ = []
            self.accepted_bundles_ = []
            self.report_ = {
                "status": "no_valid_seed_nodes",
                "algorithm": "feat_like_evolutionary_representation_v1",
                "evaluations": self._evaluations,
            }
            return self

        initial_population = [
            individual
            for individual in singleton_evaluations
            if np.isfinite(individual.construction_nmse)
        ]
        pair_anchor_count = min(
            len(ranked_nodes), max(1, min(3, self.config.population_size // 8))
        )
        pair_candidate_count = min(
            len(ranked_nodes), max(12, min(24, self.config.population_size))
        )
        if self.config.max_bundle_size >= 2:
            seeded_pairs: set[tuple[str, ...]] = set()
            for anchor in ranked_nodes[:pair_anchor_count]:
                for complement in ranked_nodes[:pair_candidate_count]:
                    pair = self._normalize_nodes((anchor, complement))
                    signature = tuple(node.signature for node in pair)
                    if len(pair) != 2 or signature in seeded_pairs:
                        continue
                    if self._evaluations >= self.config.max_evaluations:
                        break
                    seeded_pairs.add(signature)
                    individual = self._evaluate_nodes(pair, generation=0)
                    if np.isfinite(individual.construction_nmse):
                        initial_population.append(individual)
                if self._evaluations >= self.config.max_evaluations:
                    break

        # Grow a small deterministic beam by adding residual-complementary
        # candidates to the strongest bundles found so far. This gives bundles
        # of size 3+ a direct path to the archive before stochastic mutation.
        if self.config.max_bundle_size >= 3 and initial_population:
            beam_width = max(2, min(6, self.config.population_size // 4))
            candidate_pool = ranked_nodes
            beam = sorted(
                initial_population,
                key=lambda item: (item.construction_nmse, item.complexity),
            )[:beam_width]
            for _ in range(2, self.config.max_bundle_size):
                if self._evaluations >= self.config.max_evaluations or not beam:
                    break
                expanded: list[_BundleIndividual] = []
                for base in beam:
                    ordered_candidates = self._residual_candidate_order(
                        base, candidate_pool
                    )[: min(len(candidate_pool), 24)]
                    for candidate in ordered_candidates:
                        if self._evaluations >= self.config.max_evaluations:
                            break
                        nodes = self._normalize_nodes(base.nodes + (candidate,))
                        if len(nodes) <= len(base.nodes):
                            continue
                        before = self._evaluations
                        individual = self._evaluate_nodes(nodes, generation=0)
                        if (
                            self._evaluations > before
                            and np.isfinite(individual.construction_nmse)
                        ):
                            expanded.append(individual)
                    if self._evaluations >= self.config.max_evaluations:
                        break
                if not expanded:
                    break
                initial_population.extend(expanded)
                beam = sorted(
                    expanded,
                    key=lambda item: (item.construction_nmse, item.complexity),
                )[:beam_width]

        attempts = 0
        while (
            len(initial_population) < self.config.population_size * 3
            and self._evaluations < self.config.max_evaluations
            and attempts < self.config.population_size * 20
        ):
            size = int(
                rng.randint(
                    1,
                    min(self.config.max_bundle_size, len(ranked_nodes)) + 1,
                )
            )
            indices = rng.choice(len(ranked_nodes), size=size, replace=False)
            nodes = tuple(ranked_nodes[int(index)] for index in indices)
            before = self._evaluations
            individual = self._evaluate_nodes(nodes, generation=0)
            if self._evaluations > before and np.isfinite(individual.construction_nmse):
                initial_population.append(individual)
            attempts += 1

        population = self._survive(initial_population)
        if not population:
            self.proposals_ = []
            self.accepted_proposals_ = []
            self.bundle_proposals_ = []
            self.accepted_bundles_ = []
            self.report_ = {
                "status": "no_valid_initial_population",
                "algorithm": "feat_like_evolutionary_representation_v1",
                "evaluations": self._evaluations,
            }
            return self

        population = self._survive(population)
        generation_history: list[dict[str, Any]] = []
        for generation in range(1, self.config.generations + 1):
            if self._evaluations >= self.config.max_evaluations:
                break
            offspring: list[_BundleIndividual] = []
            attempts = 0
            while (
                len(offspring) < self.config.population_size
                and self._evaluations < self.config.max_evaluations
                and attempts < self.config.population_size * 10
            ):
                parent = self._lexicase_parent(population, rng)
                child_nodes = parent.nodes
                if rng.rand() < self.config.crossover_rate and len(population) > 1:
                    other = self._lexicase_parent(population, rng)
                    child_nodes = self._crossover_nodes(parent.nodes, other.nodes, rng)
                if rng.rand() < self.config.mutation_rate:
                    child_nodes = self._mutate_nodes(child_nodes, node_library, rng)
                before = self._evaluations
                child = self._evaluate_nodes(child_nodes, generation=generation)
                if self._evaluations > before and np.isfinite(child.construction_nmse):
                    offspring.append(child)
                attempts += 1
            population = self._survive(population + offspring)
            finite_population = [
                individual
                for individual in population
                if np.isfinite(individual.construction_nmse)
            ]
            if finite_population:
                best = min(
                    finite_population,
                    key=lambda individual: (
                        individual.construction_nmse,
                        individual.complexity,
                    ),
                )
                generation_history.append(
                    {
                        "generation": generation,
                        "evaluations": self._evaluations,
                        "population_size": len(population),
                        "best_construction_nmse": best.construction_nmse,
                        "best_complexity": best.complexity,
                    }
                )

        evaluated = [
            individual
            for individual in self._evaluation_cache.values()
            if np.isfinite(individual.construction_nmse)
        ]
        pareto = self._pareto_fronts(evaluated)
        archive = pareto[0] if pareto else []
        nontrivial_archive = [
            individual
            for individual in archive
            if any(node.operator != "variable" for node in individual.nodes)
            and sum(node.operator != "variable" for node in individual.nodes)
            <= self.config.max_generated_features
        ]
        best_bundle = min(
            nontrivial_archive,
            key=lambda individual: (
                individual.validation_nmse
                + self.config.complexity_tiebreak * individual.complexity,
                individual.complexity,
            ),
            default=None,
        )
        improvement = (
            baseline_validation_nmse - best_bundle.validation_nmse
            if best_bundle is not None
            else float("-inf")
        )
        accepted = bool(
            best_bundle is not None
            and np.isfinite(best_bundle.validation_nmse)
            and improvement >= self.config.min_validation_improvement
        )
        selected_nodes = (
            tuple(
                node
                for node in best_bundle.nodes
                if node.operator != "variable"
            )
            if accepted and best_bundle is not None
            else ()
        )
        if accepted and not selected_nodes:
            accepted = False
        selected_validation_nmse = (
            best_bundle.validation_nmse if best_bundle is not None else float("inf")
        )
        selected_construction_nmse = (
            best_bundle.construction_nmse if best_bundle is not None else float("inf")
        )
        generated_names = tuple(f"feat_v1_{index}" for index in range(len(selected_nodes)))
        expressions = tuple(node.expression(names) for node in selected_nodes)
        bundle_record = FeatureBundleProposal(
            nodes=selected_nodes,
            names=generated_names,
            expressions=expressions,
            downstream_columns=tuple(names) + generated_names,
            construction_nmse=(
                best_bundle.construction_nmse if best_bundle is not None else float("inf")
            ),
            validation_nmse=(
                best_bundle.validation_nmse if best_bundle is not None else float("inf")
            ),
            baseline_validation_nmse=baseline_validation_nmse,
            improvement_score=(improvement if np.isfinite(improvement) else 0.0),
            complexity=(best_bundle.complexity if best_bundle is not None else 0),
            generation=(best_bundle.generation if best_bundle is not None else 0),
            coefficients=(best_bundle.coefficients if best_bundle is not None else ()),
            accepted=accepted,
            rejection_reason=(
                None if accepted else "insufficient_independent_validation_improvement"
            ),
        )
        self.bundle_proposals_ = [bundle_record] if best_bundle is not None else []
        self.accepted_bundles_ = [bundle_record] if accepted else []
        self.accepted_proposals_ = [
            FeatureProposal(
                operator="feat_evolved",
                left_index=(node.input_indices[0] if node.input_indices else 0),
                right_index=(node.input_indices[-1] if node.input_indices else 0),
                name=name,
                expression=expression,
                invariance_score=0.0,
                relevance_score=max(improvement, 0.0),
                support_fraction=1.0,
                accepted=True,
                feature_kind="feat_evolved",
                depth=node.depth,
                complexity=self._node_complexity(node),
                validation_score=max(0.0, 1.0 - selected_validation_nmse),
                improvement_score=max(improvement, 0.0),
                construction_score=max(0.0, 1.0 - selected_construction_nmse),
                node=node,
            )
            for node, name, expression in zip(
                selected_nodes, generated_names, expressions, strict=True
            )
        ]
        self.proposals_ = list(self.accepted_proposals_)
        archive_records = [
            {
                "expressions": [node.expression(names) for node in individual.nodes],
                "construction_nmse": individual.construction_nmse,
                "validation_nmse": individual.validation_nmse,
                "complexity": individual.complexity,
                "generation": individual.generation,
            }
            for individual in sorted(
                archive,
                key=lambda individual: (
                    individual.construction_nmse,
                    individual.complexity,
                ),
            )[:12]
        ]
        self.report_ = {
            "status": "ok" if accepted else "no_validated_improvement",
            "algorithm": "feat_like_evolutionary_representation_v1",
            "official_feat_source_copied": False,
            "split_sizes": {
                "train": len(train),
                "construction": len(construction),
                "validation": len(validation),
            },
            "population_size": self.config.population_size,
            "generations_requested": self.config.generations,
            "generations_completed": len(generation_history),
            "evaluations": self._evaluations,
            "evaluation_budget": self.config.max_evaluations,
            "seed_node_count": len(node_library),
            "baseline_construction_nmse": baseline_construction_nmse,
            "baseline_validation_nmse": baseline_validation_nmse,
            "accepted_count": len(self.accepted_proposals_),
            "accepted_bundle_count": len(self.accepted_bundles_),
            "bundles": [proposal.to_record() for proposal in self.bundle_proposals_],
            "pareto_archive": archive_records,
            "generation_history": generation_history,
            "selection": "epsilon_lexicase",
            "survival": "error_complexity_nondominated_sorting",
            "downstream_model": "standardized_ridge",
            "limitations": [
                "regression_only",
                "constrained_feature_dimensions_are_screened_by_mysrcore_runtime",
                "no_constant_optimization",
                "no_backpropagation",
                "python_lightweight_search_not_official_feat",
                "validation_split_is_internal_not_external_test_data",
            ],
        }
        return self

    def transform(self, X: Any, *, augment: bool = True) -> NDArray[np.float64]:
        """Replay the selected bundle on new rows."""

        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("FEATLikeFeatureEngineer must be fitted before transform")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("X does not match the fitted feature shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("X must contain only finite values")
        if not augment or not self.accepted_proposals_:
            return values
        columns = [values]
        columns.extend(
            proposal.transform(values).reshape(-1, 1)
            for proposal in self.accepted_proposals_
        )
        transformed = np.column_stack(columns)
        if not np.all(np.isfinite(transformed)):
            raise ValueError("accepted FEAT-like feature replay produced non-finite values")
        return transformed

    def get_feature_names_out(self) -> list[str]:
        if not hasattr(self, "variable_names_in_"):
            raise RuntimeError("FEATLikeFeatureEngineer must be fitted first")
        return list(self.variable_names_in_) + [
            proposal.name for proposal in self.accepted_proposals_
        ]


class FeatureEngineeringEnsemble:
    """Merge independently fitted feature engines under one output budget."""

    def __init__(
        self,
        engines: list[tuple[str, Any]],
        variable_names: list[str],
        *,
        max_generated_features: int,
    ) -> None:
        if not isinstance(max_generated_features, int) or isinstance(
            max_generated_features, bool
        ) or max_generated_features < 1:
            raise ValueError("max_generated_features must be a positive integer")
        self.engines_ = tuple(engines)
        self.variable_names_in_ = tuple(variable_names)
        self.n_features_in_ = len(variable_names)
        by_signature: dict[str, tuple[FeatureProposal, set[str]]] = {}
        proposal_groups: list[tuple[tuple[str, ...], tuple[float, float, float, float]]] = []
        for engine_name, engine in engines:
            engine_signatures: set[str] = set()
            for proposal in engine.accepted_proposals_:
                signature = (
                    proposal.node.signature if proposal.node is not None else proposal.name
                )
                engine_signatures.add(signature)
                if signature not in by_signature:
                    by_signature[signature] = (proposal, {engine_name})
                else:
                    previous, sources = by_signature[signature]
                    sources.add(engine_name)
                    if self._rank(proposal) > self._rank(previous):
                        by_signature[signature] = (proposal, sources)
            bundled_signatures: set[str] = set()
            for bundle in getattr(engine, "accepted_bundles_", []):
                signatures = tuple(node.signature for node in bundle.nodes)
                if not signatures:
                    continue
                bundled_signatures.update(signatures)
                proposal_groups.append(
                    (
                        signatures,
                        max(self._rank(by_signature[item][0]) for item in signatures),
                    )
                )
            proposal_groups.extend(
                ((signature,), self._rank(by_signature[signature][0]))
                for signature in engine_signatures - bundled_signatures
            )

        unique_groups: dict[tuple[str, ...], tuple[float, float, float, float]] = {}
        for signatures, rank in proposal_groups:
            normalized = tuple(sorted(set(signatures)))
            unique_groups[normalized] = max(unique_groups.get(normalized, rank), rank)
        selected_signatures: list[str] = []
        selected_signature_set: set[str] = set()
        for signatures, _ in sorted(
            unique_groups.items(),
            key=lambda item: (item[1], -len(item[0])),
            reverse=True,
        ):
            unseen = [
                signature
                for signature in signatures
                if signature not in selected_signature_set
            ]
            if len(selected_signatures) + len(unseen) > max_generated_features:
                continue
            selected_signatures.extend(unseen)
            selected_signature_set.update(unseen)

        self.accepted_proposals_ = [
            by_signature[signature][0] for signature in selected_signatures
        ]
        self.proposals_ = list(self.accepted_proposals_)
        self.accepted_bundles_ = [
            bundle
            for _, engine in engines
            for bundle in getattr(engine, "accepted_bundles_", [])
            if all(node.signature in selected_signature_set for node in bundle.nodes)
        ]
        self.report_ = {
            "status": "ok",
            "algorithm": "mysr_dual_feature_engineering_v1",
            "accepted_count": len(self.accepted_proposals_),
            "candidate_count": sum(
                len(getattr(engine, "proposals_", [])) for _, engine in engines
            ),
            "engine_reports": {
                name: engine.report_ for name, engine in self.engines_
            },
            "accepted_features": [
                {
                    **proposal.to_record(),
                    "sources": sorted(by_signature[signature][1]),
                }
                for signature, proposal in zip(
                    selected_signatures,
                    self.accepted_proposals_,
                    strict=True,
                )
            ],
            "accepted_bundles": [
                bundle.to_record() for bundle in self.accepted_bundles_
            ],
            "global_generated_feature_budget": max_generated_features,
        }

    @staticmethod
    def _rank(proposal: FeatureProposal) -> tuple[float, float, float, float]:
        return (
            proposal.improvement_score,
            proposal.validation_score,
            proposal.invariance_score,
            -proposal.complexity,
        )

    def transform(self, X: Any, *, augment: bool = True) -> NDArray[np.float64]:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("X does not match the fitted feature shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("X must contain only finite values")
        if not augment or not self.accepted_proposals_:
            return values
        transformed = np.column_stack(
            [values]
            + [
                proposal.transform(values).reshape(-1, 1)
                for proposal in self.accepted_proposals_
            ]
        )
        if not np.all(np.isfinite(transformed)):
            raise ValueError("accepted feature replay produced non-finite values")
        return transformed

    def get_feature_names_out(self) -> list[str]:
        return list(self.variable_names_in_) + [
            proposal.name for proposal in self.accepted_proposals_
        ]
