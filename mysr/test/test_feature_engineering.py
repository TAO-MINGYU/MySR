from __future__ import annotations

import numpy as np
import pytest
from sklearn.utils import check_random_state

from mysr import MySRRegressor
from mysr.feat_engine import FEATLikeFeatureEngineer, FeatureEngineeringEnsemble
from mysr.feature_engineering import (
    FEATEngineConfig,
    FeatureComplexitySpec,
    FeatureDimensionSpec,
    FeatureEngineeringConfig,
    FeatureNode,
    FeatureProposal,
    SurrogateEngineConfig,
    SurrogateFeatureEngineer,
)


def test_feature_node_complexity_matches_mysrcore_recursive_mapping():
    spec = FeatureComplexitySpec.from_user(
        2,
        complexity_of_variables=[2, 3],
        complexity_of_constants=4,
        complexity_of_operators={"+": 5, "*": 6, "^": 7},
    )
    node = FeatureNode(
        "add",
        (
            FeatureNode.variable(0),
            FeatureNode("mul", (FeatureNode.constant(2.0), FeatureNode.variable(1))),
        ),
    )

    # 5 (+) + 2 (x0) + 6 (*) + 4 (constant) + 3 (x1).
    assert node.ast_complexity(spec) == 20.0


def test_feature_node_composite_complexity_expands_helper_operators():
    spec = FeatureComplexitySpec.from_user(
        1,
        complexity_of_variables=[2],
        complexity_of_constants=3,
        complexity_of_operators={"*": 5, "^": 7, "sqrt": 7, "abs": 11, "/": 13},
    )
    assert FeatureNode("square", (FeatureNode.variable(0),)).ast_complexity(spec) == 9.0
    assert FeatureNode("sqrt_abs", (FeatureNode.variable(0),)).ast_complexity(spec) == 20.0
    assert FeatureNode("reciprocal", (FeatureNode.variable(0),)).ast_complexity(spec) == 18.0


def test_explicit_helper_operator_complexity_overrides_expanded_fallback():
    spec = FeatureComplexitySpec.from_user(
        1,
        complexity_of_variables=[2],
        complexity_of_operators={"square": 5, "*": 99},
    )

    square = FeatureNode("square", (FeatureNode.variable(0),))
    assert square.ast_complexity(spec) == 7.0


def test_feature_node_rejects_malformed_public_trees():
    with pytest.raises(ValueError, match="exactly two children"):
        FeatureNode("add", (FeatureNode.variable(0),))
    with pytest.raises(ValueError, match="non-negative integer"):
        FeatureNode.variable(-1)


def test_feature_proposal_dimension_expression_uses_dimension_tolerance():
    left = (0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    right = (0.1 + 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    proposal = FeatureProposal(
        operator="add",
        left_index=0,
        right_index=1,
        name="afe_add_0_1",
        expression="(a + b)",
        invariance_score=0.0,
        relevance_score=0.0,
        support_fraction=1.0,
        accepted=True,
    )

    assert proposal.dimension_expression([left, right]) == "length^0.3"


def test_feature_node_signature_preserves_order_for_directional_composites():
    left = FeatureNode.variable(0)
    right = FeatureNode.variable(1)
    forward = FeatureNode("normalized_sub", (left, right))
    reverse = FeatureNode("normalized_sub", (right, left))

    assert forward.signature != reverse.signature
    assert forward.expression(["a", "b"]) == "((a - b) / (a + b))"
    assert reverse.expression(["a", "b"]) == "((b - a) / (b + a))"


def test_extended_helper_depth_and_hypot_complexity_match_expanded_ast():
    spec = FeatureComplexitySpec.from_user(
        2,
        complexity_of_variables=[2, 3],
        complexity_of_constants=5,
        complexity_of_operators={"+": 13, "^": 7, "sqrt": 11},
    )
    left = FeatureNode.variable(0)
    right = FeatureNode.variable(1)
    hypot = FeatureNode("hypot", (left, right))
    log_ratio = FeatureNode("log_ratio", (left, right))

    assert hypot.depth == 3
    assert log_ratio.depth == 3
    # sqrt((x0)^2 + (x1)^2): sqrt + add + two powers/constants + x0 + x1.
    assert hypot.ast_complexity(spec) == 53.0


@pytest.mark.parametrize("raw_complexities", [None, 2.5, [2.0, 3.0, 4.0]])
def test_augment_materializes_full_variable_complexity_list(raw_complexities):
    X = _positive_data(7)
    y = X[:, 0] - X[:, 1]
    model = MySRRegressor(
        auto_feature_engineering=True,
        feature_engineering_config=FeatureEngineeringConfig(
            mode="augment",
            surrogate_engine=SurrogateEngineConfig(
                candidate_operators=("sub",),
                candidate_unary_operators=(),
                enable_unary_composition=False,
                enable_recursive_composition=False,
                enable_separability=False,
                surrogate_min_r2=0.75,
                invariance_min_score=0.86,
                max_iter=1000,
            ),
            feat_engine=FEATEngineConfig(enabled=False),
        ),
    )
    model.feature_names_in_ = np.asarray(["a", "b", "c"])
    model.display_feature_names_in_ = model.feature_names_in_
    model.nout_ = 1
    result = model._pre_transform_training_data(
        X,
        y,
        None,
        model.feature_names_in_,
        raw_complexities,
        None,
        None,
        check_random_state(7),
    )
    assert result[0].shape == (X.shape[0], 4)
    assert len(result[3]) == 4
    if raw_complexities is None:
        expected_raw = [1.0, 1.0, 1.0]
    elif raw_complexities == 2.5:
        expected_raw = [2.5, 2.5, 2.5]
    else:
        expected_raw = raw_complexities
    assert result[3][:3] == expected_raw
    assert result[3][3] == 1.0 + expected_raw[0] + expected_raw[1]


def test_automatic_feature_engineering_rejects_arbitrary_complexity_mapping():
    model = MySRRegressor(
        auto_feature_engineering=True,
        complexity_mapping="tree -\u003e 1",
        feature_engineering_config=FeatureEngineeringConfig(),
    )
    model.feature_names_in_ = np.asarray(["a", "b", "c"])
    model.display_feature_names_in_ = model.feature_names_in_
    model.nout_ = 1
    with pytest.raises(ValueError, match="complexity_mapping"):
        model._pre_transform_training_data(
            _positive_data(8),
            _positive_data(8)[:, 0],
            None,
            model.feature_names_in_,
            None,
            None,
            None,
            check_random_state(8),
        )


def test_dict_configuration_does_not_truncate_non_integer_feature_budget():
    from mysr.feature_engineering import coerce_feature_engineering_config

    with pytest.raises(TypeError, match="max_generated_features"):
        coerce_feature_engineering_config({"max_generated_features": 2.5})


def test_feat_support_rejects_invalid_intermediate_subtree():
    engineer = FEATLikeFeatureEngineer(FEATEngineConfig(enabled=True))
    X = np.column_stack([np.ones(40), np.zeros(40)])
    node = FeatureNode(
        "reciprocal",
        (FeatureNode("div", (FeatureNode.variable(0), FeatureNode.variable(1))),),
    )

    values, support = engineer._node_support(node, X)
    assert np.all(np.isfinite(values))
    assert not np.any(support)


def test_feat_duplicate_filter_ignores_correlation_between_raw_columns():
    rng = np.random.RandomState(123)
    raw = rng.normal(size=(80, 2))
    X = np.column_stack([raw[:, 0], raw[:, 0], raw[:, 1]])
    engineer = FEATLikeFeatureEngineer(FEATEngineConfig(enabled=True))

    node = FeatureNode("square", (FeatureNode.variable(2),))
    matrix = engineer._feature_matrix((node,), X, include_raw=True)

    assert matrix is not None
    assert matrix.shape == (80, 4)


def test_feat_transform_rejects_non_finite_input_even_without_features():
    engineer = FEATLikeFeatureEngineer(FEATEngineConfig(enabled=False))
    engineer.n_features_in_ = 1
    engineer.accepted_proposals_ = []

    with pytest.raises(ValueError, match="finite"):
        engineer.transform(np.asarray([[np.nan]]))


def _positive_data(seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.uniform(0.5, 2.5, size=(400, 3))


def _feat_bundle_data(seed: int = 41) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2.0, 2.0, size=(320, 3))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2
    return X, y


def _feat_bundle_config() -> FEATEngineConfig:
    return FEATEngineConfig(
        enabled=True,
        population_size=16,
        generations=4,
        max_evaluations=100,
        max_depth=2,
        max_bundle_size=3,
        max_seed_nodes=48,
    )


@pytest.mark.parametrize(
    ("operator", "target"),
    [
        ("sub", lambda X: np.sin(X[:, 0] - X[:, 1])),
        ("add", lambda X: np.sin(X[:, 0] + X[:, 1])),
        ("mul", lambda X: np.sin(X[:, 0] * X[:, 1])),
        ("div", lambda X: np.sin(X[:, 0] / X[:, 1])),
    ],
)
def test_surrogate_discovers_each_pair_operator(operator, target):
    X = _positive_data()
    config = SurrogateEngineConfig(
        candidate_operators=(operator,),
        surrogate_min_r2=0.75,
        invariance_min_score=0.86,
        n_perturbations=10,
        max_iter=1200,
    )
    engineer = SurrogateFeatureEngineer(config, random_state=0).fit(
        X,
        target(X),
        variable_names=["x1", "x2", "noise"],
    )

    accepted = engineer.accepted_proposals_
    assert any(
        proposal.operator == operator
        and {proposal.left_index, proposal.right_index} == {0, 1}
        for proposal in accepted
    )


def test_irrelevant_variable_pair_is_not_accepted():
    X = _positive_data(1)
    y = np.sin(X[:, 0] - X[:, 1])
    config = SurrogateEngineConfig(
        surrogate_min_r2=0.75,
        invariance_min_score=0.88,
        n_perturbations=10,
        max_iter=1200,
    )
    engineer = SurrogateFeatureEngineer(config, random_state=1).fit(
        X,
        y,
        variable_names=["x1", "x2", "irrelevant"],
    )

    assert engineer.accepted_proposals_
    assert all(
        2 not in (proposal.left_index, proposal.right_index)
        for proposal in engineer.accepted_proposals_
    )


def test_low_quality_surrogate_returns_no_candidates():
    X = _positive_data(2)
    y = np.random.RandomState(2).normal(size=X.shape[0])
    config = SurrogateEngineConfig(surrogate_min_r2=0.95, max_iter=300)
    engineer = SurrogateFeatureEngineer(config, random_state=2).fit(X, y)

    assert engineer.accepted_proposals_ == []
    assert engineer.report_["status"] == "surrogate_quality_insufficient"


def test_division_candidate_rejects_zero_domain():
    X = _positive_data(3)
    X[0, 1] = 0.0
    y = X[:, 0] + X[:, 1]
    config = SurrogateEngineConfig(
        candidate_operators=("div",),
        surrogate_min_r2=0.70,
        invariance_min_score=0.0,
        max_iter=1000,
    )
    engineer = SurrogateFeatureEngineer(config, random_state=3).fit(X, y)

    assert engineer.accepted_proposals_ == []
    assert engineer.proposals_[0].rejection_reason == "unsafe_division_domain"


def test_transform_replays_accepted_feature_on_new_rows():
    X = _positive_data(4)
    y = X[:, 0] - X[:, 1]
    config = SurrogateEngineConfig(
        candidate_operators=("sub",),
        surrogate_min_r2=0.75,
        invariance_min_score=0.86,
        max_iter=1000,
    )
    engineer = SurrogateFeatureEngineer(config, random_state=4).fit(
        X,
        y,
        variable_names=["a", "b", "c"],
    )
    new_X = X[:5]
    transformed = engineer.transform(new_X)

    assert transformed.shape == (5, 4)
    np.testing.assert_allclose(transformed[:, -1], new_X[:, 0] - new_X[:, 1])
    assert engineer.get_feature_names_out()[-1] == "afe_sub_0_1"


def test_regressor_suggest_and_augment_modes_have_distinct_effects():
    X = _positive_data(5)
    y = X[:, 0] - X[:, 1]
    surrogate = SurrogateEngineConfig(
        candidate_operators=("sub",),
        surrogate_min_r2=0.75,
        invariance_min_score=0.86,
        max_iter=1000,
    )
    common = {
        "surrogate_engine": surrogate,
        "feat_engine": FEATEngineConfig(enabled=False),
    }

    suggest_model = MySRRegressor(
        auto_feature_engineering=True,
        feature_engineering_config=FeatureEngineeringConfig(mode="suggest", **common),
    )
    suggest_model.feature_names_in_ = np.asarray(["a", "b", "c"])
    suggest_model.display_feature_names_in_ = suggest_model.feature_names_in_
    suggest_model.nout_ = 1
    suggest_result = suggest_model._pre_transform_training_data(
        X,
        y,
        None,
        suggest_model.feature_names_in_,
        None,
        None,
        None,
        check_random_state(5),
    )
    assert suggest_result[0].shape == X.shape
    assert suggest_model.feature_engineering_report_["accepted_count"] == 1

    augment_model = MySRRegressor(
        auto_feature_engineering=True,
        feature_engineering_config=FeatureEngineeringConfig(mode="augment", **common),
    )
    augment_model.feature_names_in_ = np.asarray(["a", "b", "c"])
    augment_model.display_feature_names_in_ = augment_model.feature_names_in_
    augment_model.nout_ = 1
    augment_result = augment_model._pre_transform_training_data(
        X,
        y,
        None,
        augment_model.feature_names_in_,
        None,
        None,
        None,
        check_random_state(5),
    )
    assert augment_result[0].shape == (X.shape[0], X.shape[1] + 1)
    assert augment_result[2][-1] == "afe_sub_0_1"


def test_constrained_feature_engineering_requires_dimensions():
    X = _positive_data(6)
    y = X[:, 0] - X[:, 1]
    model = MySRRegressor(
        formula_type="theoretical",
        auto_feature_engineering=True,
        feature_engineering_config=FeatureEngineeringConfig(),
    )
    model.feature_names_in_ = np.asarray(["a", "b", "c"])
    model.display_feature_names_in_ = model.feature_names_in_
    model.nout_ = 1
    with pytest.raises(ValueError, match="requires X_dimensions"):
        model._pre_transform_training_data(
            X,
            y,
            None,
            model.feature_names_in_,
            None,
            None,
            None,
            check_random_state(6),
        )


@pytest.mark.parametrize("formula_type", ["semi_theoretical", "theoretical"])
def test_constrained_feature_dimensions_use_internal_compatibility_gate(formula_type):
    model = MySRRegressor()
    length = [1, 0, 0, 0, 0, 0, 0]
    time = [0, 0, 1, 0, 0, 0, 0]
    spec = model._make_feature_dimension_spec(
        formula_type, [length, length, time], length
    )
    assert spec is not None
    assert spec.policy == "compatible"

    valid_product = FeatureNode("mul", (FeatureNode.variable(0), FeatureNode.variable(1)))
    invalid_sum = FeatureNode("add", (FeatureNode.variable(0), FeatureNode.variable(2)))
    valid, product_dimension, reason = spec.validate(valid_product)
    assert valid and reason is None
    assert "length" in product_dimension
    invalid, _, reason = spec.validate(invalid_sum)
    assert not invalid
    assert reason == "dimension_constraint"


def test_constrained_augment_injects_feature_dimension_and_complexity_metadata():
    X = _positive_data(6)
    y = X[:, 0] - X[:, 1]
    surrogate = SurrogateEngineConfig(
        candidate_operators=("sub",),
        surrogate_min_r2=0.75,
        invariance_min_score=0.86,
        max_iter=1000,
    )
    model = MySRRegressor(
        formula_type="theoretical",
        auto_feature_engineering=True,
        feature_engineering_config=FeatureEngineeringConfig(
            mode="augment",
            surrogate_engine=surrogate,
            feat_engine=FEATEngineConfig(enabled=False),
        ),
    )
    model.feature_names_in_ = np.asarray(["a", "b", "c"])
    model.display_feature_names_in_ = model.feature_names_in_
    model.nout_ = 1
    length = [1, 0, 0, 0, 0, 0, 0]
    result = model._pre_transform_training_data(
        X,
        y,
        None,
        model.feature_names_in_,
        None,
        [length, length, length],
        length,
        check_random_state(6),
    )
    assert result[0].shape == (X.shape[0], 4)
    assert result[4] == [tuple(length)] * 4
    assert result[3] == [1.0, 1.0, 1.0, 3.0]
    assert model.engineered_feature_metadata_[0]["dimension"] == "length"


def test_ai_feynman_composition_discovers_square_of_pairwise_reduction():
    rng = np.random.RandomState(21)
    X = rng.uniform(-1.0, 1.0, size=(360, 3))
    y = (X[:, 0] - X[:, 1]) ** 2
    config = SurrogateEngineConfig(
        candidate_operators=("sub",),
        candidate_unary_operators=("square",),
        surrogate_min_r2=0.75,
        invariance_min_score=0.80,
        composition_min_score=0.20,
        composition_min_improvement=0.02,
        max_composition_depth=2,
        max_iter=1200,
        max_generated_features=5,
    )
    engineer = SurrogateFeatureEngineer(config, random_state=21).fit(
        X, y, variable_names=["x1", "x2", "noise"]
    )

    assert any(
        proposal.feature_kind == "unary_composition"
        and proposal.expression == "((x1 - x2))^2"
        and proposal.accepted
        for proposal in engineer.accepted_proposals_
    )
    transformed = engineer.transform(X[:7])
    square_proposal = next(
        proposal
        for proposal in engineer.accepted_proposals_
        if proposal.expression == "((x1 - x2))^2"
    )
    square_column = X.shape[1] + engineer.accepted_proposals_.index(square_proposal)
    np.testing.assert_allclose(
        transformed[:, square_column], square_proposal.transform(X[:7])
    )
    assert (
        engineer.report_["algorithm"]
        == "ai_feynman_inspired_feature_construction_v5"
    )
    assert engineer.report_["unary_compositions"]


def test_extended_composition_nodes_replay_and_propagate_dimensions():
    X = np.asarray([[2.0, 1.0], [3.0, 1.5], [4.0, 2.0]])
    length = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    dimensions = FeatureDimensionSpec((length, length), None, "compatible")
    normalized = FeatureNode(
        "normalized_sub",
        (FeatureNode.variable(0), FeatureNode.variable(1)),
    )
    np.testing.assert_allclose(
        normalized.evaluate(X), (X[:, 0] - X[:, 1]) / (X[:, 0] + X[:, 1])
    )
    assert normalized.expression(["a", "b"]) == "((a - b) / (a + b))"
    valid, expression, reason = dimensions.validate(normalized)
    assert valid and expression == "1" and reason is None

    hypot = FeatureNode(
        "hypot",
        (FeatureNode.variable(0), FeatureNode.variable(1)),
    )
    np.testing.assert_allclose(hypot.evaluate(X), np.hypot(X[:, 0], X[:, 1]))
    valid, expression, reason = dimensions.validate(hypot)
    assert valid and expression == "length" and reason is None

    ratio_log = FeatureNode(
        "log_ratio",
        (FeatureNode.variable(0), FeatureNode.variable(1)),
    )
    np.testing.assert_allclose(ratio_log.evaluate(X), np.log(X[:, 0] / X[:, 1]))
    valid, expression, reason = dimensions.validate(ratio_log)
    assert valid and expression == "1" and reason is None

    time = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    mismatched = FeatureDimensionSpec((length, time), None, "compatible")
    valid, _, reason = mismatched.validate(normalized)
    assert not valid and reason == "dimension_constraint"


def test_nested_domain_support_rejects_undefined_intermediate_values():
    engineer = SurrogateFeatureEngineer(SurrogateEngineConfig(enabled=False))
    X = np.asarray([[1.0, 0.0], [1.0, 2.0]])
    nested = FeatureNode(
        "div",
        (
            FeatureNode.constant(1.0),
            FeatureNode("div", (FeatureNode.variable(0), FeatureNode.variable(1))),
        ),
    )

    values, support = engineer._node_support(nested, X)
    assert np.isfinite(values).all()
    assert support.tolist() == [False, True]


def test_dimension_propagation_tolerates_fractional_cancellation():
    length = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    dimensions = FeatureDimensionSpec((length,), None, "compatible")
    node = FeatureNode(
        "div",
        (
            FeatureNode("power", (FeatureNode.variable(0),), value=0.1),
            FeatureNode("power", (FeatureNode.variable(0),), value=0.3),
        ),
    )
    # 0.1 + 0.2 - 0.3 cancels to a dimensionless result within tolerance.
    composed = FeatureNode(
        "div",
        (
            FeatureNode("mul", (
                FeatureNode("power", (FeatureNode.variable(0),), value=0.1),
                FeatureNode("power", (FeatureNode.variable(0),), value=0.2),
            )),
            FeatureNode("power", (FeatureNode.variable(0),), value=0.3),
        ),
    )
    valid, expression, reason = dimensions.validate(composed)
    assert valid and expression == "1" and reason is None
    valid, _, _ = dimensions.validate(node)
    assert valid


def test_dict_configuration_with_lists_keeps_default_composition_grammar():
    from mysr.feature_engineering import coerce_feature_engineering_config

    config = coerce_feature_engineering_config(
        {
            "surrogate_engine": {
                "candidate_operators": ["sub", "add", "mul", "div"],
            }
        }
    )
    engineer = SurrogateFeatureEngineer(config.surrogate_engine)
    assert "normalized_sub" in engineer._composition_operator_pool()


def test_surrogate_discovers_normalized_difference_composition():
    rng = np.random.RandomState(91)
    X = rng.uniform(0.5, 2.0, size=(500, 3))
    y = (X[:, 0] - X[:, 1]) / (X[:, 0] + X[:, 1])
    config = SurrogateEngineConfig(
        candidate_operators=(),
        composition_operators=("normalized_sub",),
        candidate_unary_operators=(),
        enable_pairwise_symmetry=False,
        enable_unary_composition=False,
        enable_power_composition=False,
        surrogate_min_r2=0.50,
        composition_min_score=0.20,
        composition_min_improvement=0.02,
        max_composition_depth=2,
        max_composition_candidates=32,
        max_generated_features=2,
        max_iter=1200,
    )
    engineer = SurrogateFeatureEngineer(config, random_state=91).fit(
        X, y, variable_names=["a", "b", "noise"]
    )

    assert any(
        proposal.accepted
        and proposal.operator == "normalized_sub"
        and proposal.expression == "((a - b) / (a + b))"
        for proposal in engineer.accepted_proposals_
    )


def test_surrogate_discovers_non_integer_power_of_composite():
    rng = np.random.RandomState(92)
    X = rng.uniform(0.5, 2.0, size=(500, 3))
    y = (X[:, 0] + X[:, 1]) ** 1.5
    config = SurrogateEngineConfig(
        candidate_operators=(),
        composition_operators=("add",),
        candidate_unary_operators=(),
        enable_pairwise_symmetry=False,
        enable_unary_composition=False,
        enable_power_composition=True,
        power_exponents=(1.5,),
        surrogate_min_r2=0.75,
        composition_min_score=0.20,
        composition_min_improvement=0.02,
        max_composition_depth=2,
        max_composition_candidates=128,
        max_generated_features=3,
        max_iter=1200,
    )
    engineer = SurrogateFeatureEngineer(config, random_state=92).fit(
        X, y, variable_names=["a", "b", "noise"]
    )

    proposal = next(
        proposal
        for proposal in engineer.accepted_proposals_
        if proposal.operator == "power"
        and proposal.expression == "((a + b))^1.5"
    )
    np.testing.assert_allclose(proposal.transform(X[:20]), y[:20])
    assert proposal.validation_score >= config.high_confidence_composition_score


def test_unary_domain_checks_are_reported_and_not_augmented():
    rng = np.random.RandomState(22)
    X = rng.uniform(-1.0, 1.0, size=(300, 2))
    X[0, 0] = 0.0
    y = X[:, 0] + X[:, 1]
    config = SurrogateEngineConfig(
        candidate_operators=(),
        enable_pairwise_symmetry=False,
        candidate_unary_operators=("reciprocal", "log_abs"),
        surrogate_min_r2=0.70,
        composition_min_score=0.0,
        composition_min_improvement=0.0,
        max_iter=900,
    )
    engineer = SurrogateFeatureEngineer(config, random_state=22).fit(X, y)

    assert engineer.accepted_proposals_ == []
    zero_column_proposals = [
        proposal for proposal in engineer.proposals_ if proposal.left_index == 0
    ]
    assert zero_column_proposals
    assert all(
        proposal.rejection_reason == "unsafe_composition_domain"
        for proposal in zero_column_proposals
    )


def test_separability_report_contains_additive_and_multiplicative_partitions():
    rng = np.random.RandomState(23)
    X = rng.uniform(0.2, 2.0, size=(320, 4))
    y = np.sin(X[:, 0]) + np.log(X[:, 1]) + X[:, 2] * X[:, 3]
    config = SurrogateEngineConfig(
        candidate_operators=(),
        candidate_unary_operators=(),
        enable_pairwise_symmetry=False,
        enable_unary_composition=False,
        surrogate_min_r2=0.60,
        max_iter=1000,
        max_separability_partitions=8,
    )
    engineer = SurrogateFeatureEngineer(config, random_state=23).fit(X, y)

    assert engineer.decomposition_proposals_
    assert {proposal.kind for proposal in engineer.decomposition_proposals_} == {
        "additive",
        "multiplicative",
    }
    assert len(engineer.report_["separability"]) == len(
        engineer.decomposition_proposals_
    )


def test_recursive_composition_combines_two_symmetry_reductions():
    rng = np.random.RandomState(24)
    X = rng.uniform(0.4, 2.0, size=(650, 4))
    y = (X[:, 0] + X[:, 1]) * (X[:, 2] - X[:, 3])
    config = SurrogateEngineConfig(
        surrogate_min_r2=0.80,
        invariance_min_score=0.85,
        composition_min_score=0.25,
        composition_min_improvement=0.03,
        max_iter=1500,
        max_generated_features=8,
    )
    engineer = SurrogateFeatureEngineer(config, random_state=24).fit(
        X, y, variable_names=["x1", "x2", "x3", "x4"]
    )

    assert any(
        proposal.feature_kind == "recursive_composition"
        and proposal.expression == "((x1 + x2) * (x3 - x4))"
        for proposal in engineer.accepted_proposals_
    )
    assert any(
        proposal.accepted
        and proposal.kind == "multiplicative"
        and proposal.left_indices == (0, 1)
        and proposal.right_indices == (2, 3)
        for proposal in engineer.decomposition_proposals_
    )


def test_beam_search_constructs_depth_three_multivariable_feature():
    rng = np.random.RandomState(25)
    X = rng.uniform(-2.0, 2.0, size=(650, 3))
    y = np.sin(X[:, 0] + X[:, 1]) ** 2
    config = SurrogateEngineConfig(
        candidate_operators=("add",),
        candidate_unary_operators=("sin", "square"),
        surrogate_min_r2=0.75,
        invariance_min_score=0.82,
        composition_min_score=0.20,
        composition_min_improvement=0.02,
        max_composition_depth=3,
        composition_beam_width=16,
        max_composition_candidates=192,
        max_generated_features=8,
        max_iter=1800,
    )
    engineer = SurrogateFeatureEngineer(config, random_state=25).fit(
        X, y, variable_names=["x1", "x2", "noise"]
    )

    proposal = next(
        proposal
        for proposal in engineer.accepted_proposals_
        if proposal.expression == "(sin((x1 + x2)))^2"
    )
    assert proposal.depth == 3
    np.testing.assert_allclose(proposal.transform(X[:12]), y[:12])
    assert len(engineer.proposals_) <= config.max_composition_candidates + 3
    assert len(engineer.report_["composition_search"]) == 3
    assert all(
        layer["frontier_count"] <= config.composition_beam_width
        for layer in engineer.report_["composition_search"]
    )


@pytest.mark.parametrize("seed", [30, 31])
def test_parameterized_additive_symmetry_learns_non_unit_coefficient(seed):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0.2, 2.2, size=(700, 4))
    true_parameter = 2.3
    y = np.sin(X[:, 0] + true_parameter * X[:, 1])
    config = SurrogateEngineConfig(
        candidate_operators=("add",),
        candidate_unary_operators=(),
        enable_unary_composition=False,
        enable_recursive_composition=False,
        enable_separability=False,
        surrogate_ensemble_size=2,
        surrogate_stability_min_fraction=0.5,
        surrogate_min_r2=0.88,
        invariance_min_score=0.86,
        n_perturbations=3,
        max_candidate_pairs=12,
        max_parameterized_candidates=6,
        parameter_search_min_score=0.25,
        max_iter=1600,
    )
    engineer = SurrogateFeatureEngineer(config, random_state=seed).fit(
        X, y, variable_names=["x1", "x2", "noise1", "noise2"]
    )

    proposal = next(
        proposal
        for proposal in engineer.accepted_proposals_
        if proposal.operator == "parameterized_add"
        and proposal.input_indices == (0, 1)
    )
    assert proposal.parameter == pytest.approx(true_parameter, abs=0.12)
    expected = X[:20, 0] + proposal.parameter * X[:20, 1]
    np.testing.assert_allclose(proposal.transform(X[:20]), expected)
    assert proposal.stability_fraction == 1.0
    assert engineer.report_["parameterized_symmetries"]
    assert engineer.report_["split_sizes"] == {
        "train": 490,
        "construction": 105,
        "validation": 105,
    }


def test_parameterized_multiplicative_symmetry_learns_exponent():
    rng = np.random.RandomState(32)
    X = rng.uniform(0.5, 2.0, size=(800, 3))
    true_exponent = 1.7
    y = np.sin(X[:, 0] * X[:, 1] ** true_exponent)
    config = SurrogateEngineConfig(
        candidate_operators=("mul",),
        candidate_unary_operators=(),
        enable_unary_composition=False,
        enable_recursive_composition=False,
        enable_separability=False,
        surrogate_ensemble_size=2,
        surrogate_stability_min_fraction=0.5,
        surrogate_min_r2=0.90,
        invariance_min_score=0.84,
        n_perturbations=3,
        max_candidate_pairs=8,
        max_parameterized_candidates=4,
        parameter_search_min_score=0.30,
        max_iter=1800,
    )
    engineer = SurrogateFeatureEngineer(config, random_state=32).fit(
        X, y, variable_names=["x1", "x2", "noise"]
    )

    proposal = next(
        proposal
        for proposal in engineer.accepted_proposals_
        if proposal.operator == "parameterized_mul"
        and proposal.input_indices == (0, 1)
    )
    assert proposal.parameter == pytest.approx(true_exponent, abs=0.12)
    expected = X[:20, 0] * X[:20, 1] ** proposal.parameter
    np.testing.assert_allclose(proposal.transform(X[:20]), expected)


def test_high_dimensional_pair_budget_prioritizes_relevant_columns():
    relevance = np.zeros(12)
    relevance[10] = 1.0
    relevance[11] = 0.9

    candidates = SurrogateFeatureEngineer._pair_directions(
        12, ("sub",), max_pairs=1, relevance=relevance
    )

    assert candidates == [("sub", 10, 11)]


def test_parameterized_symmetry_handles_scale_noise_and_irrelevant_columns():
    rng = np.random.RandomState(33)
    X = np.column_stack(
        [
            rng.uniform(-2.0, 2.0, 900),
            rng.uniform(-0.5, 0.5, 900),
            rng.uniform(-100.0, 100.0, 900),
            rng.uniform(-0.001, 0.001, 900),
            rng.normal(size=900),
        ]
    )
    true_parameter = 3.2
    y = np.sin(X[:, 0] + true_parameter * X[:, 1])
    y += rng.normal(0.0, 0.01, size=X.shape[0])
    config = SurrogateEngineConfig(
        candidate_operators=("add",),
        candidate_unary_operators=(),
        enable_unary_composition=False,
        enable_recursive_composition=False,
        enable_separability=False,
        surrogate_ensemble_size=3,
        surrogate_stability_min_fraction=2.0 / 3.0,
        surrogate_min_r2=0.92,
        invariance_min_score=0.86,
        n_perturbations=3,
        max_candidate_pairs=12,
        max_parameterized_candidates=6,
        parameter_search_min_score=0.20,
        max_iter=1800,
    )
    engineer = SurrogateFeatureEngineer(config, random_state=33).fit(X, y)

    proposal = next(
        proposal
        for proposal in engineer.accepted_proposals_
        if proposal.operator == "parameterized_add"
        and proposal.input_indices == (0, 1)
    )
    assert proposal.parameter == pytest.approx(true_parameter, abs=0.15)
    assert all(index in (0, 1) for index in proposal.input_indices)
    assert proposal.stability_fraction == 1.0


def test_feat_like_evolves_replayable_joint_nonlinear_bundle():
    X, y = _feat_bundle_data()
    config = _feat_bundle_config()

    engineer = FEATLikeFeatureEngineer(config, random_state=41).fit(
        X, y, variable_names=["x1", "x2", "noise"]
    )

    assert {proposal.expression for proposal in engineer.accepted_proposals_} == {
        "sin(x1)",
        "(x2)^2",
    }
    assert engineer.report_["status"] == "ok"
    assert engineer.report_["evaluations"] <= config.max_evaluations
    assert engineer.report_["accepted_bundle_count"] == 1
    bundle = engineer.report_["bundles"][0]
    assert bundle["validation_nmse"] < bundle["baseline_validation_nmse"] * 1.0e-6

    transformed = engineer.transform(X[:9])
    assert transformed.shape == (9, 5)
    np.testing.assert_allclose(transformed[:, 3], np.sin(X[:9, 0]))
    np.testing.assert_allclose(transformed[:, 4], X[:9, 1] ** 2)

    limited = FeatureEngineeringEnsemble(
        [("feat", engineer)],
        ["x1", "x2", "noise"],
        max_generated_features=1,
    )
    assert limited.accepted_proposals_ == []
    assert limited.accepted_bundles_ == []


def test_feat_like_is_reproducible_for_fixed_seed():
    X, y = _feat_bundle_data()
    config = _feat_bundle_config()

    first = FEATLikeFeatureEngineer(config, random_state=41).fit(X, y)
    second = FEATLikeFeatureEngineer(config, random_state=41).fit(X, y)

    assert [item.expression for item in first.accepted_proposals_] == [
        item.expression for item in second.accepted_proposals_
    ]
    assert first.report_["evaluations"] == second.report_["evaluations"]
    assert first.report_["bundles"] == second.report_["bundles"]


@pytest.mark.parametrize("mode", ["suggest", "augment"])
def test_regressor_runs_feat_like_branch_in_both_modes(mode):
    X, y = _feat_bundle_data()
    model = MySRRegressor(
        formula_type="empirical",
        auto_feature_engineering=True,
        feature_engineering_config=FeatureEngineeringConfig(
            mode=mode,
            surrogate_engine=SurrogateEngineConfig(enabled=False),
            feat_engine=_feat_bundle_config(),
        ),
    )
    model.feature_names_in_ = np.asarray(["x1", "x2", "noise"])
    model.display_feature_names_in_ = model.feature_names_in_
    model.nout_ = 1

    result = model._pre_transform_training_data(
        X,
        y,
        None,
        model.feature_names_in_,
        None,
        None,
        None,
        check_random_state(41),
    )

    expected_columns = X.shape[1] if mode == "suggest" else X.shape[1] + 2
    assert result[0].shape == (X.shape[0], expected_columns)
    assert model.feature_engineering_engine_reports_["feat"]["status"] == "ok"
    assert len(model.engineered_feature_bundles_) == 1
    assert set(model.engineered_feature_expressions_.values()) == {
        "sin(x1)",
        "(x2)^2",
    }


def test_feat_like_rejects_invalid_budget_configuration():
    X, y = _feat_bundle_data()
    config = FEATEngineConfig(
        enabled=True,
        population_size=12,
        max_evaluations=8,
    )

    with pytest.raises(ValueError, match="max_evaluations"):
        FEATLikeFeatureEngineer(config, random_state=0).fit(X, y)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("n_perturbations", 2.5),
        ("max_composition_depth", True),
        ("max_iter", 10.5),
    ],
)
def test_surrogate_rejects_non_integer_budget_configuration(field_name, value):
    config = SurrogateEngineConfig(**{field_name: value})

    with pytest.raises(TypeError, match=field_name):
        SurrogateFeatureEngineer(config, random_state=0).fit(
            _positive_data(9), _positive_data(9)[:, 0]
        )


def test_surrogate_accepts_sequence_hidden_layer_sizes():
    engineer = SurrogateFeatureEngineer(
        SurrogateEngineConfig(enabled=False, hidden_layer_sizes=[8, 8]),
        random_state=0,
    ).fit(_positive_data(11), _positive_data(11)[:, 0])

    assert engineer.report_["status"] == "disabled"


@pytest.mark.parametrize("field_name", ["population_size", "generations", "max_depth"])
def test_feat_like_rejects_non_integer_budget_configuration(field_name):
    X, y = _feat_bundle_data()
    config = FEATEngineConfig(enabled=True, **{field_name: 2.5})

    with pytest.raises(TypeError, match=field_name):
        FEATLikeFeatureEngineer(config, random_state=0).fit(X, y)


def test_surrogate_fit_clears_previous_diagnostics_when_disabled():
    X = _positive_data(10)
    y = X[:, 0] - X[:, 1]
    engineer = SurrogateFeatureEngineer(
        SurrogateEngineConfig(
            candidate_operators=("sub",),
            candidate_unary_operators=(),
            enable_unary_composition=False,
            enable_recursive_composition=False,
            enable_separability=False,
            surrogate_min_r2=0.75,
            invariance_min_score=0.80,
            max_iter=1000,
        ),
        random_state=10,
    ).fit(X, y)
    engineer.parameter_search_report_ = [{"stale": True}]
    engineer.composition_search_layers_ = [{"stale": True}]
    engineer.config = SurrogateEngineConfig(enabled=False)

    engineer.fit(X, y)

    assert engineer.parameter_search_report_ == []
    assert engineer.composition_search_layers_ == []


def test_feat_like_materializes_depth_two_composition_without_random_hit():
    rng = np.random.RandomState(17)
    X = rng.uniform(-2.0, 2.0, size=(500, 4))
    y = np.sin(X[:, 0] - X[:, 1]) + X[:, 2] ** 2
    config = FEATEngineConfig(
        enabled=True,
        population_size=20,
        generations=6,
        max_evaluations=180,
        max_depth=3,
        max_bundle_size=4,
        max_seed_nodes=80,
    )

    engineer = FEATLikeFeatureEngineer(config, random_state=17).fit(
        X, y, variable_names=["x1", "x2", "x3", "noise"]
    )

    assert {item.expression for item in engineer.accepted_proposals_} == {
        "sin((x1 - x2))",
        "(x3)^2",
    }
    assert engineer.report_["bundles"][0]["validation_nmse"] < 1.0e-6


def test_feat_like_residual_beam_recovers_three_component_bundle():
    rng = np.random.RandomState(17)
    X = rng.uniform(-2.0, 2.0, size=(500, 4))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2 + np.log(np.abs(X[:, 2]))
    config = FEATEngineConfig(
        enabled=True,
        population_size=20,
        generations=6,
        max_evaluations=180,
        max_depth=3,
        max_bundle_size=4,
        max_seed_nodes=80,
    )

    engineer = FEATLikeFeatureEngineer(config, random_state=17).fit(
        X, y, variable_names=["x1", "x2", "x3", "noise"]
    )

    assert {item.expression for item in engineer.accepted_proposals_} == {
        "sin(x1)",
        "(x2)^2",
        "log(abs(x3))",
    }
    assert engineer.report_["bundles"][0]["validation_nmse"] < 1.0e-6
