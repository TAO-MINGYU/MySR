"""PyTorch expression generator for MySRCore RNN-GPSR population seeding.

The neural policy stays on the Python side so MySRCore remains a pure Julia
package. MySRCore owns the expression grammar, constraints, real loss, and GP
evolution; this module learns an autoregressive distribution over backend token
sequences and returns syntax-complete proposals.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TorchRNNConfig:
    """Training configuration for the recurrent expression policy."""

    cell: str = "lstm"
    hidden_size: int = 32
    embedding_size: int = 16
    num_layers: int = 1
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    epochs: int = 64
    validation_fraction: float = 0.2
    min_validation_spearman: float = 0.05
    top_fraction: float = 0.2
    patience: int = 12
    entropy_weight: float = 0.005


def ensure_torch_available() -> Any:
    """Import PyTorch lazily and provide an actionable optional-dependency error."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the active environment
        raise ImportError(
            "RNN-GPSR seeding requires PyTorch. Install the optional dependency "
            "with `python -m pip install 'mysr[rnn]'`, then retry."
        ) from exc
    return torch


def _as_sequences(values: Iterable[Iterable[Any]]) -> list[list[int]]:
    sequences = [[int(token) for token in sequence] for sequence in values]
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("RNN-GPSR token sequences must be non-empty")
    if any(token <= 0 for sequence in sequences for token in sequence):
        raise ValueError("RNN-GPSR tokens must be positive; zero is reserved for padding")
    return sequences


def _rank_targets(costs: np.ndarray) -> np.ndarray:
    """Map lower costs to targets in [-1, 1], with the best expression at 1."""

    finite_costs = np.where(np.isfinite(costs), costs, np.inf)
    order = np.argsort(finite_costs, kind="stable")
    targets = np.empty(len(costs), dtype=np.float32)
    if len(costs) == 1:
        targets[0] = 1.0
        return targets
    targets[order] = np.linspace(1.0, -1.0, len(costs), dtype=np.float32)
    return targets


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic average ranks, including ties."""

    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(predictions: np.ndarray, targets: np.ndarray) -> float:
    pred_ranks = _average_ranks(np.asarray(predictions, dtype=np.float64))
    target_ranks = _average_ranks(np.asarray(targets, dtype=np.float64))
    pred_ranks -= pred_ranks.mean()
    target_ranks -= target_ranks.mean()
    denominator = float(
        np.sqrt(np.dot(pred_ranks, pred_ranks) * np.dot(target_ranks, target_ranks))
    )
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(pred_ranks, target_ranks) / denominator)


def _validation_indices(targets: np.ndarray, fraction: float) -> np.ndarray:
    """Select a deterministic validation set spread across the quality ranks."""

    count = min(max(4, round(len(targets) * fraction)), len(targets) - 4)
    order = np.argsort(targets, kind="stable")
    positions = np.linspace(0, len(targets) - 1, count, dtype=int)
    return np.unique(order[positions])


def _language_batch(
    torch: Any,
    sequences: Sequence[Sequence[int]],
    bos_token: int,
    device: Any,
):
    max_length = max(len(sequence) for sequence in sequences)
    inputs = torch.zeros((len(sequences), max_length), dtype=torch.long, device=device)
    targets = torch.zeros_like(inputs)
    mask = torch.zeros((len(sequences), max_length), dtype=torch.bool, device=device)
    for index, sequence in enumerate(sequences):
        length = len(sequence)
        inputs[index, 0] = bos_token
        if length > 1:
            inputs[index, 1:length] = torch.as_tensor(
                sequence[:-1], dtype=torch.long, device=device
            )
        targets[index, :length] = torch.as_tensor(
            sequence, dtype=torch.long, device=device
        )
        mask[index, :length] = True
    return inputs, targets, mask


def _make_policy(torch: Any, vocabulary_size: int, config: TorchRNNConfig):
    nn = torch.nn
    bos_token = vocabulary_size + 1

    class ExpressionPolicyRNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(
                vocabulary_size + 2,
                config.embedding_size,
                padding_idx=0,
            )
            recurrent_type = nn.LSTM if config.cell == "lstm" else nn.GRU
            self.recurrent = recurrent_type(
                config.embedding_size,
                config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
            )
            self.output_layer = nn.Linear(config.hidden_size, vocabulary_size)

        def forward(self, inputs):
            recurrent_output, _ = self.recurrent(self.embedding(inputs))
            return self.output_layer(recurrent_output)

        def step(self, token, hidden):
            recurrent_output, hidden = self.recurrent(self.embedding(token), hidden)
            return self.output_layer(recurrent_output[:, -1, :]), hidden

    return ExpressionPolicyRNN(), bos_token


def _sequence_log_probabilities(
    torch: Any, logits, targets, mask, *, normalize_by_length: bool = True
):
    token_losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        (targets.reshape(-1) - 1).clamp_min(0),
        reduction="none",
    ).reshape_as(targets)
    token_losses = torch.where(mask, token_losses, 0.0)
    sequence_log_probabilities = -token_losses.sum(dim=1)
    if normalize_by_length:
        sequence_log_probabilities = (
            sequence_log_probabilities / mask.sum(dim=1).clamp_min(1)
        )
    return sequence_log_probabilities


def _policy_loss(
    torch: Any,
    logits,
    targets,
    mask,
    quality_targets,
    top_fraction: float,
    entropy_weight: float,
):
    sequence_log_probabilities = _sequence_log_probabilities(
        torch, logits, targets, mask, normalize_by_length=False
    )
    threshold = torch.quantile(quality_targets, 1.0 - top_fraction)
    elite_mask = quality_targets >= threshold
    advantages = (quality_targets - threshold).clamp_min(0.0)
    elite_advantages = advantages[elite_mask]
    risk_seeking_loss = -(
        sequence_log_probabilities[elite_mask] * elite_advantages.detach()
    ).sum() / elite_advantages.sum().clamp_min(1.0e-6)

    probabilities = torch.softmax(logits, dim=-1)
    log_probabilities = torch.log_softmax(logits, dim=-1)
    token_entropy = -(probabilities * log_probabilities).sum(dim=-1)
    entropy = token_entropy[mask].mean()
    return risk_seeking_loss - entropy_weight * entropy


def _sample_expression(
    torch: Any,
    model: Any,
    bos_token: int,
    arities: Sequence[int],
    max_length: int,
    generator: Any,
) -> list[int] | None:
    device = next(model.parameters()).device
    previous_token = torch.as_tensor([[bos_token]], dtype=torch.long, device=device)
    hidden = None
    dangling = 1
    sequence: list[int] = []
    for position in range(max_length):
        logits, hidden = model.step(previous_token, hidden)
        remaining = max_length - position - 1
        valid = []
        for arity in arities:
            next_dangling = dangling - 1 + arity
            valid.append(0 <= next_dangling <= remaining)
        valid_mask = torch.as_tensor(valid, dtype=torch.bool, device=device)
        if not bool(valid_mask.any()):
            return None
        constrained_logits = logits[0].masked_fill(~valid_mask, -torch.inf)
        probabilities = torch.softmax(constrained_logits, dim=-1)
        token = int(torch.multinomial(probabilities, 1, generator=generator).item()) + 1
        sequence.append(token)
        dangling += int(arities[token - 1]) - 1
        if dangling == 0:
            return sequence
        previous_token = torch.as_tensor([[token]], dtype=torch.long, device=device)
    return None


class TorchRNNGenerator:
    """Train an LSTM/GRU policy and generate grammar-complete expressions.

    The policy uses a risk-seeking autoregressive objective over the best-cost
    quantile, adapted from Deep Symbolic Optimization. A held-out
    Spearman gate rejects policies whose sequence likelihood does not track real
    expression quality. MySRCore then compares generated candidates against a
    separately evaluated random control group before GP-SR.
    """

    def __init__(self, config: TorchRNNConfig):
        self.config = config
        self.diagnostics_: list[dict[str, Any]] = []

    def __call__(
        self,
        training_sequences: Iterable[Iterable[Any]],
        training_costs: Iterable[Any],
        token_arities: Iterable[Any],
        proposal_count: Any,
        max_length: Any,
        seed: Any,
    ) -> list[list[int]]:
        torch = ensure_torch_available()
        sequences = _as_sequences(training_sequences)
        costs = np.asarray([float(cost) for cost in training_costs], dtype=np.float64)
        arities = [int(arity) for arity in token_arities]
        requested_count = int(proposal_count)
        maximum_length = int(max_length)
        if len(sequences) != len(costs):
            raise ValueError("RNN-GPSR training sequences and costs have different lengths")
        if len(sequences) < 8:
            raise ValueError("PyTorch RNN-GPSR requires at least eight training expressions")
        if not arities or any(arity < 0 for arity in arities):
            raise ValueError("RNN-GPSR token arities must be non-negative")
        if max(token for sequence in sequences for token in sequence) > len(arities):
            raise ValueError("RNN-GPSR training data contains an unknown token")

        seed_value = int(seed) % (2**31 - 1)
        quality_targets = _rank_targets(costs)
        validation_indices = _validation_indices(
            quality_targets, self.config.validation_fraction
        )
        validation_mask = np.zeros(len(quality_targets), dtype=bool)
        validation_mask[validation_indices] = True
        training_indices = np.flatnonzero(~validation_mask)

        device = torch.device("cpu")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed_value)
            model, bos_token = _make_policy(torch, len(arities), self.config)
            model = model.to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            inputs, targets, sequence_mask = _language_batch(
                torch, sequences, bos_token, device
            )
            quality_tensor = torch.as_tensor(
                quality_targets, dtype=torch.float32, device=device
            )
            train_index_tensor = torch.as_tensor(
                training_indices, dtype=torch.long, device=device
            )
            validation_index_tensor = torch.as_tensor(
                validation_indices, dtype=torch.long, device=device
            )

            best_state = None
            best_spearman = -np.inf
            best_epoch = 0
            epochs_without_improvement = 0
            for epoch in range(self.config.epochs):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs[train_index_tensor])
                loss = _policy_loss(
                    torch,
                    logits,
                    targets[train_index_tensor],
                    sequence_mask[train_index_tensor],
                    quality_tensor[train_index_tensor],
                    self.config.top_fraction,
                    self.config.entropy_weight,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    validation_logits = model(inputs[validation_index_tensor])
                    validation_scores = _sequence_log_probabilities(
                        torch,
                        validation_logits,
                        targets[validation_index_tensor],
                        sequence_mask[validation_index_tensor],
                    ).cpu().numpy()
                validation_spearman = _spearman(
                    validation_scores, quality_targets[validation_indices]
                )
                comparison_value = (
                    validation_spearman
                    if np.isfinite(validation_spearman)
                    else -np.inf
                )
                if comparison_value > best_spearman + 1.0e-8:
                    best_spearman = comparison_value
                    best_epoch = epoch + 1
                    best_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.patience:
                    break

            accepted = bool(
                best_state is not None
                and np.isfinite(best_spearman)
                and best_spearman >= self.config.min_validation_spearman
            )
            diagnostics = {
                **asdict(self.config),
                "seed": seed_value,
                "training_count": len(sequences),
                "proposal_count": requested_count,
                "validation_count": len(validation_indices),
                "best_epoch": best_epoch,
                "validation_spearman": float(best_spearman),
                "accepted": accepted,
                "generated_count": 0,
            }
            self.diagnostics_.append(diagnostics)
            if not accepted:
                return []

            model.load_state_dict(best_state)
            model.eval()
            torch_generator = torch.Generator(device=device)
            torch_generator.manual_seed(seed_value + 1)
            generated: list[list[int]] = []
            seen: set[tuple[int, ...]] = set()
            max_attempts = max(20, 12 * requested_count)
            with torch.no_grad():
                for _ in range(max_attempts):
                    sequence = _sample_expression(
                        torch,
                        model,
                        bos_token,
                        arities,
                        maximum_length,
                        torch_generator,
                    )
                    if sequence is None:
                        continue
                    key = tuple(sequence)
                    if key in seen:
                        continue
                    seen.add(key)
                    generated.append(sequence)
                    if len(generated) >= requested_count:
                        break
            diagnostics["generated_count"] = len(generated)
            return generated


def make_julia_rnn_generator(jl: Any, generator: TorchRNNGenerator) -> Any:
    """Wrap the Python policy as a Julia function returning native token vectors."""

    adapter = jl.seval(
        """
        py_generator -> function(
            training_sequences,
            training_costs,
            token_arities,
            proposal_count,
            max_length,
            seed,
        )
            result = py_generator(
                training_sequences,
                training_costs,
                token_arities,
                proposal_count,
                max_length,
                seed,
            )
            return PythonCall.pyconvert(Vector{Vector{Int}}, result)
        end
        """
    )
    return adapter(generator)
