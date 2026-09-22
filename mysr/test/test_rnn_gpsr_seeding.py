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
    _grammar_token_mask,
    _language_batch,
    _make_policy,
    _normalize_formula_type,
    _policy_loss,
    _rank_targets,
    _sample_expression_batch,
    _sequence_log_probabilities,
    _spearman,
)


def test_rank_targets_reward_lower_real_cost() -> None:
    targets = _rank_targets(np.asarray([10.0, 1.0, np.inf, 5.0]))

    assert targets[1] > targets[3] > targets[0] > targets[2]


def test_rank_targets_do_not_invent_quality_order_for_ties():
    ranks = _rank_targets(np.asarray([2.0, 1.0, 2.0, np.inf, np.nan]))
    assert ranks[0] == ranks[2]
    assert ranks[3] == ranks[4]
    np.testing.assert_array_equal(_rank_targets(np.ones(8)), np.zeros(8))


def test_length_penalty_changes_policy_gradient():
    torch = pytest.importorskip("torch")
    targets = torch.tensor([[1, 0, 0], [3, 1, 2], [2, 0, 0], [3, 2, 1]])
    mask = targets != 0
    qualities = torch.tensor([0.8, 1.0, -0.5, 0.5])
    gradients = []
    for penalty in (0.0, 1.0):
        logits = torch.zeros(4, 3, 3, requires_grad=True)
        loss = _policy_loss(torch, logits, targets, mask, qualities, 0.5, 0.0,
                            rank_loss_weight=0.0, diversity_weight=0.0,
                            length_penalty=penalty)
        loss.backward()
        gradients.append(logits.grad.clone())
    assert not torch.allclose(*gradients)


def test_training_grammar_mask_matches_prefix_feasibility():
    torch = pytest.importorskip("torch")
    masks = _grammar_token_mask(
        torch,
        [[3, 3, 1, 1, 2]],
        [0, 0, 2],
        5,
        torch.device("cpu"),
    )

    # At the final prefix only a leaf can close the remaining dangling slot;
    # the binary token would exceed the length ceiling.
    assert masks.shape == (1, 5, 3)
    assert masks[0, 4].tolist() == [True, True, False]


def test_training_grammar_mask_matches_shorter_teacher_forced_batch():
    torch = pytest.importorskip("torch")
    sequences = [[1], [2], [3, 1, 2], [3, 2, 1], [1], [2], [3, 1, 1], [3, 2, 2]]
    inputs, targets, sequence_mask = _language_batch(
        torch, sequences, 4, torch.device("cpu")
    )
    # max_length is deliberately larger than the actual batch time dimension.
    grammar_mask = _grammar_token_mask(
        torch, sequences, [0, 0, 2], 36, torch.device("cpu"), mask_length=inputs.shape[1]
    )
    logits = torch.zeros(8, inputs.shape[1], 3)
    log_probabilities = _sequence_log_probabilities(
        torch, logits, targets, sequence_mask, valid_token_mask=grammar_mask
    )
    assert grammar_mask.shape == (8, inputs.shape[1], 3)
    assert torch.isfinite(log_probabilities).all()


def test_elite_supervision_changes_policy_gradient():
    torch = pytest.importorskip("torch")
    targets = torch.tensor([[1, 0], [2, 0], [3, 1], [3, 2]])
    mask = targets != 0
    qualities = torch.tensor([1.0, 0.5, -0.5, -1.0])
    gradients = []
    for weight in (0.0, 0.5):
        logits = torch.zeros(4, 2, 3, requires_grad=True)
        loss = _policy_loss(
            torch,
            logits,
            targets,
            mask,
            qualities,
            0.5,
            0.0,
            rank_loss_weight=0.0,
            elite_supervision_weight=weight,
            diversity_weight=0.0,
        )
        loss.backward()
        gradients.append(logits.grad.clone())
    assert not torch.allclose(*gradients)


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"patience": 0},
        {"cell": "typo"},
        {"hidden_size": 0},
        {"learning_rate": 0.0},
        {"validation_fraction": 1.0},
        {"top_fraction": 0.0},
    ],
)
def test_torch_rnn_config_rejects_invalid_training_parameters(kwargs):
    with pytest.raises(ValueError):
        TorchRNNConfig(**kwargs)


def test_torch_rnn_generator_rejects_non_positive_sampling_request():
    pytest.importorskip("torch")
    generator = TorchRNNGenerator(TorchRNNConfig(epochs=1, patience=1))
    sequences = [[1], [2], [3, 1], [3, 2], [3, 1], [3, 2], [1], [2]]
    for proposal_count, max_length in [(0, 4), (1, 0)]:
        with pytest.raises(ValueError):
            generator(sequences, [float(i) for i in range(8)], [0, 0, 2],
                      proposal_count, max_length, 1)


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


def test_torch_rnn_falls_back_to_training_sequences_when_quality_gate_fails() -> None:
    pytest.importorskip("torch")
    generator = TorchRNNGenerator(
        TorchRNNConfig(
            epochs=1,
            patience=1,
            validation_fraction=0.5,
            min_validation_spearman=0.99,
        )
    )
    sequences = [
        [2],
        [2],
        [3, 2],
        [3, 2, 2],
        [2],
        [3, 2],
        [3, 2, 1],
        [3, 1, 2],
    ]
    proposals = generator(
        sequences,
        [0.2, 1.0, 0.1, 0.4, 0.5, 0.9, 0.3, 0.6],
        [0, 0, 2],
        4,
        5,
        20260901,
        "empirical",
        2,
        "bootstrap_structural",
        False,
    )

    assert proposals
    assert generator.diagnostics_[-1]["accepted"] is False
    assert generator.diagnostics_[-1]["fallback_generated_count"] > 0
    # Fallback returns a deduplicated subset of ranked training sequences.
    proposal_set = {tuple(item) for item in proposals}
    source_set = {tuple(item) for item in sequences}
    assert proposal_set.issubset(source_set)


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


def test_batched_sampler_supports_bounded_exploration_controls() -> None:
    torch = pytest.importorskip("torch")
    model, bos_token = _make_policy(
        torch,
        3,
        TorchRNNConfig(hidden_size=8, embedding_size=4),
        "empirical",
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260903)
    with torch.no_grad():
        proposals = _sample_expression_batch(
            torch,
            model,
            bos_token,
            [0, 0, 2],
            7,
            generator,
            16,
            temperature=0.8,
            top_k=2,
            top_p=0.9,
        )

    assert any(sequence is not None for sequence in proposals)


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
        rnn_gpsr_populations=1,
        rnn_gpsr_population_size=4,
        rnn_gpsr_niterations=2,
        rnn_gpsr_ncycles_per_iteration=1,
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
    assert model.rnn_gpsr_diagnostics_[-1]["lightweight_populations"] == 1
    assert model.rnn_gpsr_diagnostics_[-1]["lightweight_population_size"] == 4
    assert model.rnn_gpsr_diagnostics_[-1]["lightweight_niterations"] == 2
    assert (
        model.rnn_gpsr_diagnostics_[-1]["lightweight_ncycles_per_iteration"] == 1
    )


def test_rnn_gpsr_defaults_preserve_existing_initialization() -> None:
    model = MySRRegressor()

    assert model.rnn_gpsr_seeding is False
    assert model.get_params()["rnn_gpsr_seed_fraction"] == 0.5
    assert model.get_params()["rnn_cell"] == "lstm"
    assert model.get_params()["rnn_patience"] == 12
    assert model.get_params()["rnn_entropy_weight"] == pytest.approx(0.005)
    assert model.get_params()["rnn_rank_loss_weight"] == pytest.approx(0.35)
    assert model.get_params()["rnn_elite_supervision_weight"] == pytest.approx(0.15)
    assert model.get_params()["rnn_sampling_top_p"] == pytest.approx(1.0)
    assert model.get_params()["rnn_replay_fraction"] == pytest.approx(0.25)
    assert model.get_params()["rnn_gpsr_quality_gate"] is True
    assert model.get_params()["rnn_gpsr_feedback_fraction"] == pytest.approx(0.2)
    assert model.get_params()["rnn_gpsr_populations"] == 1
    assert model.get_params()["rnn_gpsr_population_size"] == 8
    assert model.get_params()["rnn_gpsr_niterations"] == 1
    assert model.get_params()["rnn_gpsr_ncycles_per_iteration"] == 4
    assert model.get_params()["rnn_gpsr_cycles"] == 4
    model._validate_and_modify_params()


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("rnn_gpsr_seed_fraction", 1.1),
        ("rnn_gpsr_candidate_count", 7),
        ("rnn_gpsr_proposal_count", 0),
        ("rnn_gpsr_populations", 0),
        ("rnn_gpsr_population_size", 0),
        ("rnn_gpsr_niterations", 0),
        ("rnn_gpsr_ncycles_per_iteration", -1),
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
        ("rnn_rank_loss_weight", -0.1),
        ("rnn_elite_supervision_weight", -0.1),
        ("rnn_diversity_weight", -0.1),
        ("rnn_length_penalty", -0.1),
        ("rnn_sampling_temperature", 0.0),
        ("rnn_sampling_top_k", -1),
        ("rnn_sampling_top_p", 0.0),
        ("rnn_replay_fraction", 1.1),
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


def test_rnn_gpsr_cycles_compatibility_alias() -> None:
    legacy = MySRRegressor(rnn_gpsr_cycles=2)
    assert legacy.rnn_gpsr_ncycles_per_iteration == 2
    assert legacy.rnn_gpsr_cycles == 2
    legacy._validate_and_modify_params()

    explicit = MySRRegressor(
        rnn_gpsr_cycles=3,
        rnn_gpsr_ncycles_per_iteration=3,
    )
    assert explicit.rnn_gpsr_ncycles_per_iteration == 3
    explicit._validate_and_modify_params()

    with pytest.raises(ValueError, match="must match"):
        MySRRegressor(
            rnn_gpsr_cycles=2,
            rnn_gpsr_ncycles_per_iteration=3,
        )
