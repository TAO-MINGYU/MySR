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


def test_semi_theoretical_is_a_public_formula_type():
    model = MySRRegressor(formula_type="semi_theoretical")
    assert model.formula_type == "semi_theoretical"
    assert model.get_params()["formula_type"] == "semi_theoretical"


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
