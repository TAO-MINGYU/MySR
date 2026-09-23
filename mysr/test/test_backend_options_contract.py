import numpy as np

from mysr import MySRRegressor
from mysr.julia_import import jl


def _fit_small_model(**overrides):
    model = MySRRegressor(
        niterations=0,
        populations=1,
        population_size=8,
        tournament_selection_n=3,
        maxsize=7,
        parallelism="serial",
        deterministic=True,
        random_state=7,
        progress=False,
        temp_equation_file=True,
        should_optimize_constants=False,
        **overrides,
    )
    X = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(-1, 1)
    y = (2.0 * X[:, 0] + 0.5).astype(np.float32)
    model.fit(X, y)
    return model


def test_default_backend_options_are_current_contract_values():
    options = _fit_small_model().julia_options_

    assert options.populations == 1
    assert options.population_size == 8
    assert options.child_refinement == jl.Symbol("safe")
    assert options.epsilon is None
    assert options.epsilon_mode == jl.Symbol("mad")


def test_optional_backend_options_round_trip_through_bridge():
    options = _fit_small_model(
        child_refinement="none",
        epsilon=0.125,
        epsilon_mode="absolute",
    ).julia_options_

    assert options.child_refinement == jl.Symbol("none")
    assert options.epsilon == 0.125
    assert options.epsilon_mode == jl.Symbol("absolute")
