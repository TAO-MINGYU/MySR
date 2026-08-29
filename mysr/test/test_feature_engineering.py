from __future__ import annotations

import numpy as np
import pytest
from sklearn.utils import check_random_state

from mysr import MySRRegressor
from mysr.feature_engineering import (
    FEATEngineConfig,
    FeatureEngineeringConfig,
    SurrogateEngineConfig,
    SurrogateFeatureEngineer,
)


def _positive_data(seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.uniform(0.5, 2.5, size=(400, 3))


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
        "formula_type": "empirical",
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


def test_feat_and_dimensional_modes_fail_explicitly():
    X = _positive_data(6)
    y = X[:, 0] - X[:, 1]
    model = MySRRegressor(
        auto_feature_engineering=True,
        feature_engineering_config=FeatureEngineeringConfig(formula_type="theoretical"),
    )
    model.feature_names_in_ = np.asarray(["a", "b", "c"])
    model.display_feature_names_in_ = model.feature_names_in_
    model.nout_ = 1
    with pytest.raises(NotImplementedError, match="empirical"):
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
        == "ai_feynman_inspired_feature_construction_v4"
    )
    assert engineer.report_["unary_compositions"]


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
