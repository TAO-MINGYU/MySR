import numpy as np
import pytest

from mysr import MySRRegressor
from mysr.feature_engineering import (
    FeatureEngineeringConfig,
    coerce_feature_engineering_config,
)
from mysr.sr import _check_assertions


def test_formula_type_is_a_public_parameter():
    model = MySRRegressor(formula_type="theoretical")
    assert model.formula_type == "theoretical"
    assert model.get_params()["formula_type"] == "theoretical"


def test_formula_type_rejects_unknown_values():
    try:
        MySRRegressor(formula_type="invalid")
    except ValueError as exc:
        assert "formula_type" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("invalid formula_type was accepted")


def test_mutation_affinity_is_public_and_validated():
    model = MySRRegressor(
        mutation_affinity="none",
        mutation_affinity_strength=2.5,
        mutation_affinity_exploration=0.4,
        operator_affinity={2: np.ones((4, 4))},
        feature_affinity=np.eye(2),
    )
    params = model.get_params()
    assert params["mutation_affinity"] == "none"
    assert params["mutation_affinity_strength"] == 2.5
    assert params["mutation_affinity_exploration"] == 0.4
    assert params["operator_affinity"][2].shape == (4, 4)
    assert params["feature_affinity"].shape == (2, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mutation_affinity": "invalid"},
        {"mutation_affinity_strength": 0.0},
        {"mutation_affinity_exploration": 1.1},
    ],
)
def test_mutation_affinity_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError, match="mutation_affinity"):
        MySRRegressor(**kwargs)


def test_semi_theoretical_is_a_public_formula_type():
    model = MySRRegressor(formula_type="semi_theoretical")
    assert model.formula_type == "semi_theoretical"
    assert model.get_params()["formula_type"] == "semi_theoretical"


@pytest.mark.parametrize("formula_type", ["semi_theoretical", "theoretical"])
@pytest.mark.parametrize("missing", ["X_dimensions", "y_dimensions"])
def test_constrained_formula_type_requires_dimension_metadata(formula_type, missing):
    model = MySRRegressor(formula_type=formula_type, niterations=0)
    X = np.ones((8, 1))
    y = np.ones(8)
    dimensions = [1, 0, 0, 0, 0, 0, 0]
    fit_kwargs = {
        "X_dimensions": [dimensions],
        "y_dimensions": dimensions,
    }
    fit_kwargs[missing] = None

    with pytest.raises(ValueError, match=missing):
        model.fit(X, y, **fit_kwargs)


def test_feature_engineering_config_has_no_duplicate_formula_type():
    config = FeatureEngineeringConfig()
    assert "formula_type" not in config.__dataclass_fields__
    with pytest.raises(ValueError, match="owned by MySRRegressor"):
        coerce_feature_engineering_config({"formula_type": "theoretical"})


def test_removed_unit_and_soft_penalty_apis_are_not_public():
    params = MySRRegressor().get_params()
    for removed in (
        "X_units",
        "y_units",
        "dimensional_constraint_penalty",
        "dimensionless_constants_only",
    ):
        assert removed not in params


def test_dimension_assertions_accept_single_vector_for_single_output():
    X = np.ones((10, 1))
    y = np.ones(10)
    dimension = [1, 0, 0, 0, 0, 0, 0]

    _check_assertions(
        X,
        False,
        None,
        1,
        None,
        y,
        dimension,
        dimension,
        False,
    )


@pytest.mark.parametrize(
    "dimensions",
    [
        [[1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
        np.asarray([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]]),
    ],
)
def test_dimension_assertions_accept_array_like_vectors_for_multi_output(dimensions):
    X = np.ones((10, 1))
    y = np.ones((10, 2))

    _check_assertions(
        X,
        False,
        None,
        1,
        None,
        y,
        None,
        dimensions,
        False,
    )
