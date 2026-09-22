import pytest

from mysr import MySRRegressor


def test_current_defaults_use_lexicase_and_age_fitness_pareto():
    model = MySRRegressor()

    assert model.parent_selection == "epsilon_lexicase"
    assert model.survival_strategy == "age_fitness_pareto"
    assert model.get_params()["parent_selection"] == "epsilon_lexicase"
    assert model.get_params()["survival_strategy"] == "age_fitness_pareto"


def test_competitive_survival_strategy_round_trips_to_params():
    model = MySRRegressor(survival_strategy="competitive_age_fitness")

    assert model.survival_strategy == "competitive_age_fitness"
    assert model.get_params()["survival_strategy"] == "competitive_age_fitness"


def test_unknown_survival_strategy_is_rejected():
    model = MySRRegressor(survival_strategy="unknown")

    with pytest.raises(ValueError, match="survival_strategy"):
        model._validate_and_modify_params()
