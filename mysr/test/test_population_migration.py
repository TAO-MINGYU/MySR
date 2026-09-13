"""Public Python controls for heterogeneous population search."""

import numpy as np
import pytest

from mysr import MySRRegressor


def test_population_profiles_and_ring_topology_are_public_parameters():
    model = MySRRegressor(
        populations=2,
        population_profiles=[{"role": "algebraic"}, {"role": "trigonometric"}],
        migration_topology="ring",
        migration_policy="best_plus_novelty",
    )
    assert model.population_profiles[0]["id"] == "population_1"
    assert model.population_profiles[1]["role"] == "trigonometric"
    assert model.migration_topology == "ring"
    assert model.migration_policy == "best_plus_novelty"
    params = model.get_params()
    assert params["population_profiles"] == model.population_profiles
    assert params["migration_topology"] == "ring"
    assert params["migration_policy"] == "best_plus_novelty"


def test_population_profile_validation():
    with pytest.raises(ValueError, match="exactly one profile"):
        MySRRegressor(populations=2, population_profiles=[{}])
    with pytest.raises(TypeError, match="mapping"):
        MySRRegressor(populations=2, population_profiles=[{}, "not-a-profile"])
    with pytest.raises(ValueError, match="Unsupported population profile"):
        MySRRegressor(populations=1, population_profiles=[{"unknown": 1}])
    with pytest.raises(ValueError, match="migration_topology"):
        MySRRegressor(migration_topology="fully_connected")
    with pytest.raises(ValueError, match="migration_policy"):
        MySRRegressor(migration_policy="novelty_only")


def test_population_profiles_reach_julia_search():
    model = MySRRegressor(
        populations=2,
        population_size=6,
        niterations=1,
        ncycles_per_iteration=1,
        maxsize=7,
        tournament_selection_n=2,
        population_profiles=[
            {
                "role": "algebraic",
                "operator_affinity": {
                    1: np.zeros((0, 0)),
                    2: np.ones((4, 4)),
                },
            },
            {"role": "trigonometric"},
        ],
        migration_topology="ring",
        migration_policy="best_plus_novelty",
        progress=False,
        verbosity=0,
        temp_equation_file=True,
    )
    X = np.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    model.fit(X, X[:, 0])
    assert len(model.equations_) > 0
