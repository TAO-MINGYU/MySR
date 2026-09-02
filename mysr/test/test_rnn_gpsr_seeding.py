import numpy as np
import pytest

from mysr import MySRRegressor
from mysr.feature_engineering import (
    FEATEngineConfig,
    FeatureEngineeringConfig,
    SurrogateEngineConfig,
)
from mysr.rnn_gpsr import (
    TorchRNNConfig,
    TorchRNNGenerator,
    _formula_type_bos_token,
    _make_policy,
    _normalize_formula_type,
    _rank_targets,
    _sample_expression_batch,
    _spearman,
)


def test_rank_targets_reward_lower_real_cost() -> None:
    targets = _rank_targets(np.asarray([10.0, 1.0, np.inf, 5.0]))

    assert targets[1] > targets[3] > targets[0] > targets[2]


def test_spearman_gate_uses_rank_order() -> None:
    targets = np.asarray([-1.0, 0.0, 1.0])

    assert _spearman(np.asarray([3.0, 4.0, 5.0]), targets) == pytest.approx(1.0)
    assert _spearman(np.asarray([5.0, 4.0, 3.0]), targets) == pytest.approx(-1.0)


@pytest.mark.parametrize("value", ["empirical", "semi_theoretical", "theoretical"])
def test_rnn_formula_type_contract_accepts_all_dimension_policies(value: str) -> None:
    assert _normalize_formula_type(value) == value


def test_rnn_formula_type_contract_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="formula_type"):
        _normalize_formula_type("invalid")


def test_formula_type_changes_recurrent_policy_input_token() -> None:
    tokens = [
        _formula_type_bos_token(10, value)
        for value in ["empirical", "semi_theoretical", "theoretical"]
    ]

    assert len(set(tokens)) == 3


def test_torch_rnn_records_backend_feedback_round() -> None:
    pytest.importorskip("torch")
    generator = TorchRNNGenerator(
        TorchRNNConfig(
            epochs=2,
            patience=1,
            validation_fraction=0.25,
            min_validation_spearman=-1.0,
        )
    )
    sequences = [
        [2],
        [1],
        [3, 2, 2],
        [3, 2, 1],
        [3, 1, 2],
        [3, 1, 1],
        [3, 3, 2, 2, 1],
        [3, 2, 3, 1, 2],
    ]
    proposals = generator(
        sequences,
        [0.2, 1.0, 0.1, 0.4, 0.5, 0.9, 0.3, 0.6],
        [0, 0, 2],
        4,
        5,
        20260901,
        "theoretical",
        2,
        "backend_gpsr_feedback",
        True,
    )

    assert proposals
    assert generator.diagnostics_[-1]["feedback_round"] == 2
    assert generator.diagnostics_[-1]["training_source"] == "backend_gpsr_feedback"
    assert generator.diagnostics_[-1]["backend_costs_used"] is True


def test_batched_sampler_returns_complete_grammar_trees() -> None:
    torch = pytest.importorskip("torch")
    model, bos_token = _make_policy(
        torch,
        3,
        TorchRNNConfig(hidden_size=8, embedding_size=4),
        "empirical",
    )
    model.eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260902)
    with torch.no_grad():
        proposals = _sample_expression_batch(
            torch,
            model,
            bos_token,
            [0, 0, 2],
            7,
            generator,
            32,
        )

    complete = [sequence for sequence in proposals if sequence is not None]
    assert complete
    for sequence in complete:
        dangling = 1
        for token in sequence:
            assert 1 <= token <= 3
            dangling += [0, 0, 2][token - 1] - 1
            assert dangling >= 0
        assert dangling == 0


def test_ai_feynman_features_are_visible_to_rnn_gpsr_entry() -> None:
    X = np.column_stack([np.linspace(0.1, 2.0, 24), np.linspace(0.2, 1.2, 24)])
    y = X[:, 0] - X[:, 1]
    model = MySRRegressor(
        rnn_gpsr_seeding=True,
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
    model.feature_names_in_ = np.asarray(["x1", "x2"])
    model.display_feature_names_in_ = model.feature_names_in_
    model.nout_ = 1

    transformed = model._pre_transform_training_data(
        X,
        y,
        None,
        model.feature_names_in_,
        None,
        None,
        None,
        np.random.RandomState(13),
    )

    assert transformed[0].shape[1] >= 3
    assert any(name.startswith("afe_") for name in transformed[2][2:])
    assert any(name.startswith("afe_") for name in model.feature_names_in_[2:])


def test_user_guesses_survive_ai_feynman_and_rnn_gpsr() -> None:
    """Exercise the three seed sources through the local backend bridge.

    Older MySRCore releases predating RNN-GPSR skip this integration assertion;
    the v1.1.0 backend executes it.
    """

    pytest.importorskip("torch")
    from mysr.julia_import import jl

    if not bool(
        jl.seval(
            ":rnn_gpsr_seeding in "
            "fieldnames(MySRCore.SymbolicRegression.Options)"
        )
    ):
        pytest.skip("backend does not expose the RNN-GPSR options")

    rng = np.random.RandomState(20260901)
    X = rng.uniform(-1.0, 1.0, size=(48, 2))
    y = 2.0 * X[:, 0] - 3.0 * X[:, 1]
    model = MySRRegressor(
        guesses=["2.0*x0 - 3.0*x1"],
        auto_feature_engineering=True,
        feature_engineering_config=FeatureEngineeringConfig(
            mode="augment",
            max_generated_features=2,
            surrogate_engine=SurrogateEngineConfig(
                candidate_operators=("sub",),
                candidate_unary_operators=(),
                enable_unary_composition=False,
                enable_recursive_composition=False,
                enable_separability=False,
                max_iter=200,
                surrogate_min_r2=0.5,
                invariance_min_score=0.5,
            ),
            feat_engine=FEATEngineConfig(enabled=False),
        ),
        rnn_gpsr_seeding=True,
        rnn_gpsr_candidate_count=8,
        rnn_gpsr_proposal_count=8,
        rnn_gpsr_cycles=1,
        rnn_gpsr_rounds=1,
        rnn_epochs=2,
        rnn_patience=1,
        rnn_validation_fraction=0.25,
        rnn_min_validation_spearman=-1.0,
        niterations=0,
        populations=2,
        population_size=8,
        tournament_selection_n=3,
        maxsize=12,
        progress=False,
        verbosity=0,
        temp_equation_file=True,
        deterministic=True,
        parallelism="serial",
        random_state=17,
    )
    model.fit(X, y, variable_names=["x0", "x1"])

    assert model.engineered_feature_expressions_
    assert float(model.equations_["loss"].min()) < 1e-10
    assert any(
        "x0" in str(equation) and "x1" in str(equation)
        for equation in model.equations_["equation"]
    )
    assert model.rnn_gpsr_diagnostics_[-1]["formula_type"] == "empirical"


def test_rnn_gpsr_defaults_preserve_existing_initialization() -> None:
    model = MySRRegressor()

    assert model.rnn_gpsr_seeding is False
    assert model.get_params()["rnn_gpsr_seed_fraction"] == 0.5
    assert model.get_params()["rnn_cell"] == "lstm"
    assert model.get_params()["rnn_patience"] == 12
    assert model.get_params()["rnn_entropy_weight"] == pytest.approx(0.005)
    assert model.get_params()["rnn_gpsr_quality_gate"] is True
    assert model.get_params()["rnn_gpsr_feedback_fraction"] == pytest.approx(0.2)
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
        ("rnn_gpsr_feedback_fraction", 1.1),
        ("rnn_gpsr_maxsize", 31),
    ],
)
def test_rnn_gpsr_parameter_validation(parameter: str, value: object) -> None:
    model = MySRRegressor(**{parameter: value})

    with pytest.raises(ValueError):
        model._validate_and_modify_params()
