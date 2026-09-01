import numpy as np
import pytest

from mysr import MySRRegressor
from mysr.rnn_gpsr import _rank_targets, _spearman


def test_rank_targets_reward_lower_real_cost() -> None:
    targets = _rank_targets(np.asarray([10.0, 1.0, np.inf, 5.0]))

    assert targets[1] > targets[3] > targets[0] > targets[2]


def test_spearman_gate_uses_rank_order() -> None:
    targets = np.asarray([-1.0, 0.0, 1.0])

    assert _spearman(np.asarray([3.0, 4.0, 5.0]), targets) == pytest.approx(1.0)
    assert _spearman(np.asarray([5.0, 4.0, 3.0]), targets) == pytest.approx(-1.0)


def test_rnn_gpsr_defaults_preserve_existing_initialization() -> None:
    model = MySRRegressor()

    assert model.rnn_gpsr_seeding is False
    assert model.get_params()["rnn_gpsr_seed_fraction"] == 0.5
    assert model.get_params()["rnn_cell"] == "lstm"
    assert model.get_params()["rnn_patience"] == 12
    assert model.get_params()["rnn_entropy_weight"] == pytest.approx(0.005)
    assert model.get_params()["rnn_gpsr_quality_gate"] is True
    model._validate_and_modify_params()


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("rnn_gpsr_seed_fraction", 1.1),
        ("rnn_gpsr_candidate_count", 7),
        ("rnn_gpsr_proposal_count", 0),
        ("rnn_hidden_size", 0),
        ("rnn_cell", "rnn"),
        ("rnn_embedding_size", 0),
        ("rnn_num_layers", 0),
        ("rnn_learning_rate", 0.0),
        ("rnn_weight_decay", -1.0),
        ("rnn_epochs", 0),
        ("rnn_patience", 0),
        ("rnn_entropy_weight", -1.0),
        ("rnn_validation_fraction", 0.5),
        ("rnn_min_validation_spearman", 1.1),
        ("rnn_top_fraction", 0.0),
        ("rnn_gpsr_cycles", -1),
        ("rnn_gpsr_rounds", 0),
        ("rnn_gpsr_maxsize", 31),
    ],
)
def test_rnn_gpsr_parameter_validation(parameter: str, value: object) -> None:
    model = MySRRegressor(**{parameter: value})

    with pytest.raises(ValueError):
        model._validate_and_modify_params()
