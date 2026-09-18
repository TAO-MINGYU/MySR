"""Tests for uncertainty-aware objectives and their public bridge."""

import numpy as np
import pytest

from mysr import MySRRegressor


def _small_model(**kwargs):
    return MySRRegressor(
        populations=1,
        population_size=6,
        niterations=1,
        ncycles_per_iteration=1,
        maxsize=7,
        tournament_selection_n=2,
        progress=False,
        verbosity=0,
        temp_equation_file=True,
        **kwargs,
    )


def test_loss_presets_and_uncertainty_modes_are_public_parameters():
    model = MySRRegressor(
        loss_preset="asymmetric_huber",
        uncertainty_mode="asymmetry",
        robust_delta=1.5,
        student_nu=5.0,
    )
    params = model.get_params()
    assert params["loss_preset"] == "asymmetric_huber"
    assert params["uncertainty_mode"] == "asymmetry"
    assert params["robust_delta"] == 1.5
    assert params["student_nu"] == 5.0


def test_uncertainty_inputs_are_validated_before_backend_start():
    X = np.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    y = X[:, 0] ** 2
    with pytest.raises(ValueError, match="loss_preset"):
        _small_model(loss_preset="unknown").fit(X, y)
    model = _small_model(uncertainty_mode="asymmetry", loss_scale="linear")
    with pytest.raises(ValueError, match="sigma_minus and sigma_plus"):
        model.fit(X, y, sigma_minus=np.ones_like(y))


def test_asymmetric_likelihood_reaches_mysrcore_search():
    X = np.linspace(-1.0, 1.0, 10).reshape(-1, 1)
    y = X[:, 0] ** 2 + 0.1
    sigma_minus = np.full_like(y, 0.05)
    sigma_plus = np.full_like(y, 0.15)
    model = _small_model(
        loss_preset="asymmetric_gaussian_nll",
        uncertainty_mode="asymmetry",
        loss_scale="linear",
    )
    model.fit(X, y, sigma_minus=sigma_minus, sigma_plus=sigma_plus)
    assert len(model.equations_) > 0
