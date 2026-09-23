import pytest

from mysr import MySRRegressor


def test_current_defaults_use_lexicase_and_age_fitness_pareto():
    model = MySRRegressor()

    assert model.parent_selection == "epsilon_lexicase"
    assert model.survival_strategy == "age_fitness_pareto"
    assert model.get_params()["parent_selection"] == "epsilon_lexicase"
    assert model.get_params()["survival_strategy"] == "age_fitness_pareto"
    assert model.child_refinement == "safe"
    assert model.epsilon is None
    assert model.epsilon_mode == "mad"
    assert model.get_params()["child_refinement"] == "safe"
    assert model.get_params()["epsilon"] is None
    assert model.get_params()["epsilon_mode"] == "mad"


def test_competitive_survival_strategy_round_trips_to_params():
    model = MySRRegressor(survival_strategy="competitive_age_fitness")

    assert model.survival_strategy == "competitive_age_fitness"
    assert model.get_params()["survival_strategy"] == "competitive_age_fitness"


def test_unknown_survival_strategy_is_rejected():
    model = MySRRegressor(survival_strategy="unknown")

    with pytest.raises(ValueError, match="survival_strategy"):
        model._validate_and_modify_params()


@pytest.mark.parametrize("child_refinement", ["none", "safe", "thorough"])
def test_child_refinement_values_are_accepted(child_refinement):
    model = MySRRegressor(child_refinement=child_refinement)

    model._validate_and_modify_params()


def test_unknown_child_refinement_is_rejected():
    model = MySRRegressor(child_refinement="unknown")

    with pytest.raises(ValueError, match="child_refinement"):
        model._validate_and_modify_params()


@pytest.mark.parametrize("epsilon", [-1.0, float("nan"), float("inf")])
def test_invalid_epsilon_is_rejected(epsilon):
    model = MySRRegressor(epsilon=epsilon)

    with pytest.raises(ValueError, match="epsilon"):
        model._validate_and_modify_params()


def test_non_mad_epsilon_mode_requires_epsilon():
    model = MySRRegressor(epsilon_mode="absolute")

    with pytest.raises(ValueError, match="epsilon"):
        model._validate_and_modify_params()


def test_unknown_epsilon_mode_is_rejected():
    model = MySRRegressor(epsilon=0.1, epsilon_mode="unknown")

    with pytest.raises(ValueError, match="epsilon_mode"):
        model._validate_and_modify_params()


def test_tournament_selection_must_be_strictly_smaller_than_population():
    model = MySRRegressor(population_size=4, tournament_selection_n=4)

    with pytest.raises(ValueError, match="smaller than `population_size`"):
        model._validate_and_modify_params()
