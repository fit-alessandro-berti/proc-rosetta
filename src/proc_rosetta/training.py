from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from time import perf_counter

import torch
import numpy as np
from torch.utils.data import DataLoader, Sampler

from proc_rosetta.data import (
    BatchConfig,
    JsonlProcessDataset,
    ProcessBatchCollator,
    SyntheticProcessDataset,
    load_data_metadata,
    sample_statistics,
    split_samples_path,
)
from proc_rosetta.devices import default_device, resolve_device
from proc_rosetta.losses import LossWeights, multimodal_tree_loss
from proc_rosetta.models import LatentDistribution, ProcRosettaModel
from proc_rosetta.synthetic import SyntheticConfig
from proc_rosetta.tokenizers import ActivityTokenizer, TreeTokenizer


@dataclass(frozen=True)
class TrainConfig:
    samples: int = 128
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 3e-4
    latent_dim: int = 128
    hidden_dim: int = 256
    seed: int = 13
    device: str = default_device()
    semantic_latent_mode: str = "deterministic"
    dropout: float = 0.10
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    early_stopping_patience: int = 12
    min_delta: float = 0.001
    lr_patience: int = 2
    lr_factor: float = 0.5
    min_lr: float = 1e-5
    group_aware_batches: bool = True
    views_per_family: int = 4
    activity_remap_probability: float = 0.0
    memory_tokens: int = 8
    decoder_layers: int = 4
    decoder_input_dropout: float = 0.10
    scheduled_sampling_max: float = 0.20
    scheduled_sampling_start_epoch: int = 20
    scheduled_sampling_ramp_epochs: int = 20
    gradient_clip_norm: float = 5.0
    tree_reconstruction_weight: float = 0.5
    trace_to_tree_weight: float = 2.0
    petri_to_tree_weight: float = 0.5
    exact_contrastive_weight: float = 0.5
    within_modality_contrastive_weight: float = 0.25
    soft_behavior_geometry_weight: float = 0.25
    variance_weight: float = 0.1
    covariance_weight: float = 0.01
    kl_weight: float = 0.0
    latent_alignment_weight: float = 0.0
    contrastive_temperature: float = 0.2
    behavior_temperature: float = 0.2
    latent_temperature: float = 0.2
    training_stage: str = "full"
    stage_gate_interval: int = 5
    gradient_diagnostics_interval: int = 1


def loss_weights_from_config(config: TrainConfig) -> LossWeights:
    if config.training_stage not in {"a", "b", "c", "d", "full"}:
        raise ValueError("training_stage must be one of: a, b, c, d, full")
    exact_weight = 0.0 if config.training_stage == "a" else config.exact_contrastive_weight
    within_weight = (
        0.0 if config.training_stage == "a" else config.within_modality_contrastive_weight
    )


def loss_weights_from_checkpoint(
    checkpoint: dict[str, object], config: TrainConfig
) -> LossWeights:
    """Restore the exact serialized objective, falling back for legacy checkpoints."""

    values = asdict(loss_weights_from_config(config))
    stored = checkpoint.get("loss_weights")
    if isinstance(stored, dict):
        values.update(
            {name: stored[name] for name in values if name in stored}
        )
    return LossWeights(**values)
    soft_weight = (
        config.soft_behavior_geometry_weight
        if config.training_stage in {"c", "d", "full"}
        else 0.0
    )
    variance_weight = 0.0 if config.training_stage == "a" else config.variance_weight
    covariance_weight = 0.0 if config.training_stage == "a" else config.covariance_weight
    return LossWeights(
        tree_reconstruction=config.tree_reconstruction_weight,
        trace_to_tree=config.trace_to_tree_weight,
        petri_to_tree=config.petri_to_tree_weight,
        exact_contrastive=exact_weight,
        within_modality_contrastive=within_weight,
        soft_behavior_geometry=soft_weight,
        variance=variance_weight,
        covariance=covariance_weight,
        latent_alignment=config.latent_alignment_weight,
        kl=config.kl_weight,
        label_smoothing=config.label_smoothing,
        contrastive_temperature=config.contrastive_temperature,
        behavior_temperature=config.behavior_temperature,
        latent_temperature=config.latent_temperature,
    )


def scheduled_sampling_probability(config: TrainConfig, epoch: int | None) -> float:
    if epoch is None or epoch < config.scheduled_sampling_start_epoch:
        return 0.0
    progress = (epoch - config.scheduled_sampling_start_epoch + 1) / max(
        config.scheduled_sampling_ramp_epochs, 1
    )
    return config.scheduled_sampling_max * min(max(progress, 0.0), 1.0)


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = {
                child_key: child_value.to(device) if isinstance(child_value, torch.Tensor) else child_value
                for child_key, child_value in value.items()
            }
        else:
            moved[key] = value
    return moved


def train_epoch(
    model: ProcRosettaModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weights: LossWeights | None = None,
    epoch: int | None = None,
    show_progress: bool = False,
    train_config: TrainConfig | None = None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    batches = 0
    iterator = progress_dataloader(
        dataloader,
        desc=f"Epoch {epoch} training" if epoch is not None else "Training",
        enabled=show_progress,
    )
    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            batch,
            deterministic=True,
            input_token_dropout=(
                0.0 if train_config is None else train_config.decoder_input_dropout
            ),
            scheduled_sampling_probability=(
                0.0
                if train_config is None
                else scheduled_sampling_probability(train_config, epoch)
            ),
        )
        tree_tokens = batch["tree_tokens"]
        assert isinstance(tree_tokens, torch.Tensor)
        positive_mask = batch.get("positive_mask")
        assert positive_mask is None or isinstance(positive_mask, torch.Tensor)
        losses = multimodal_tree_loss(
            outputs,
            tree_tokens,
            model.tree_tokenizer.pad_id,
            weights=weights,
            positive_mask=positive_mask,
            contrastive_candidate_mask=batch.get("contrastive_candidate_mask"),
            behavior_signatures=batch.get("behavior_signatures"),
            tokenizer=model.tree_tokenizer,
        )
        run_gradient_diagnostics = (
            train_config is None
            or epoch is None
            or (epoch - 1) % max(train_config.gradient_diagnostics_interval, 1) == 0
        )
        if batches == 0 and run_gradient_diagnostics:
            gradient_metrics = gradient_norm_diagnostics(
                model,
                losses,
                weights or LossWeights(),
            )
            totals.update(gradient_metrics)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=(5.0 if train_config is None else train_config.gradient_clip_norm),
        )
        optimizer.step()

        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
        batches += 1
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{float(losses['loss'].detach().cpu()):.4f}")
    return {
        name: (
            value
            if "gradient_norm" in name or "gradient_ratio" in name
            else value / max(batches, 1)
        )
        for name, value in totals.items()
    }


def gradient_norm_diagnostics(
    model: ProcRosettaModel,
    losses: dict[str, torch.Tensor],
    weights: LossWeights,
) -> dict[str, float]:
    metric_objective = (
        weights.exact_contrastive * losses["exact_contrastive"]
        + weights.within_modality_contrastive * losses["within_modality_contrastive"]
        + weights.soft_behavior_geometry * losses["soft_behavior_geometry"]
        + weights.variance * losses["variance"]
        + weights.covariance * losses["covariance"]
    )
    specifications = {
        "tree": (
            model.tree_encoder,
            weights.tree_reconstruction * losses["tree_reconstruction"],
        ),
        "trace": (
            model.trace_encoder,
            weights.trace_to_tree * losses["trace_to_tree"],
        ),
        "petri": (
            model.petri_encoder,
            weights.petri_to_tree * losses["petri_to_tree"],
        ),
    }
    result: dict[str, float] = {}
    for name, (encoder, reconstruction_objective) in specifications.items():
        parameters = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
        reconstruction_gradients = torch.autograd.grad(
            reconstruction_objective,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        metric_gradients = torch.autograd.grad(
            metric_objective,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        reconstruction_norm = _gradient_norm(reconstruction_gradients)
        metric_norm = _gradient_norm(metric_gradients)
        result[f"reconstruction_gradient_norm_{name}"] = reconstruction_norm
        result[f"metric_gradient_norm_{name}"] = metric_norm
        result[f"metric_to_reconstruction_gradient_ratio_{name}"] = (
            metric_norm / max(reconstruction_norm, 1e-12)
        )
    return result


def _gradient_norm(gradients: tuple[torch.Tensor | None, ...]) -> float:
    squared = sum(
        float(gradient.detach().pow(2).sum().cpu())
        for gradient in gradients
        if gradient is not None
    )
    return squared**0.5


@torch.no_grad()
def evaluate_epoch(
    model: ProcRosettaModel,
    dataloader: DataLoader,
    device: torch.device,
    weights: LossWeights | None = None,
    epoch: int | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
    compute_discovery_metrics: bool = False,
    source_ablation: bool = False,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    discovery_rows: list[dict[str, object]] = []
    embedding_rows: dict[str, list[torch.Tensor]] = defaultdict(list)
    exact_behavior_ids: list[str | None] = []
    partial_order_ids: list[str | None] = []
    signature_rows: list[torch.Tensor] = []
    expected_positive_pairs = 0
    false_negative_pairs = 0
    analogy_pairs = 0
    analogy_negative_pairs = 0
    iterator = progress_dataloader(
        dataloader,
        desc=(
            progress_desc
            or (f"Epoch {epoch} validation" if epoch is not None else "Validation")
        ),
        enabled=show_progress,
    )
    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch, deterministic=True)
        tree_tokens = batch["tree_tokens"]
        assert isinstance(tree_tokens, torch.Tensor)
        positive_mask = batch.get("positive_mask")
        assert positive_mask is None or isinstance(positive_mask, torch.Tensor)
        losses = multimodal_tree_loss(
            outputs,
            tree_tokens,
            model.tree_tokenizer.pad_id,
            weights=weights,
            positive_mask=positive_mask,
            contrastive_candidate_mask=batch.get("contrastive_candidate_mask"),
            behavior_signatures=batch.get("behavior_signatures"),
            tokenizer=model.tree_tokenizer,
        )
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
        batches += 1
        if compute_discovery_metrics:
            dists = outputs["dists"]
            assert isinstance(dists, dict)
            trace_distribution = dists["trace"]
            maximum_length = min(512, max(2, tree_tokens.shape[1] * 2))
            decoded = model.tree_decoder.decode_greedy(
                trace_distribution,
                max_length=maximum_length,
            )
            shuffled_decoded: torch.Tensor | None = None
            zero_decoded: torch.Tensor | None = None
            if source_ablation:
                permutation = _different_behavior_permutation(
                    batch.get("exact_behavior_ids", []),
                    trace_distribution.mu.shape[0],
                    device,
                )
                shuffled_distribution = LatentDistribution(
                    mu=trace_distribution.mu[permutation],
                    logvar=trace_distribution.logvar[permutation],
                    memory=(
                        None
                        if trace_distribution.memory is None
                        else trace_distribution.memory[permutation]
                    ),
                    activity_mask=(
                        None
                        if trace_distribution.activity_mask is None
                        else trace_distribution.activity_mask[permutation]
                    ),
                    activity_memory=(
                        None
                        if trace_distribution.activity_memory is None
                        else trace_distribution.activity_memory[permutation]
                    ),
                )
                zero_distribution = LatentDistribution(
                    mu=torch.zeros_like(trace_distribution.mu),
                    logvar=torch.zeros_like(trace_distribution.logvar),
                    memory=(
                        None
                        if trace_distribution.memory is None
                        else torch.zeros_like(trace_distribution.memory)
                    ),
                    activity_mask=(
                        None
                        if trace_distribution.activity_mask is None
                        else torch.zeros_like(trace_distribution.activity_mask)
                    ),
                    activity_memory=(
                        None
                        if trace_distribution.activity_memory is None
                        else torch.zeros_like(trace_distribution.activity_memory)
                    ),
                )
                shuffled_decoded = model.tree_decoder.decode_greedy(
                    shuffled_distribution, max_length=maximum_length
                )
                zero_decoded = model.tree_decoder.decode_greedy(
                    zero_distribution, max_length=maximum_length
                )
            samples = batch.get("samples")
            assert isinstance(samples, list)
            for row_index, (sample, target, prediction) in enumerate(
                zip(samples, tree_tokens, decoded)
            ):
                target_ids = _trim_token_ids(target.tolist(), model.tree_tokenizer)
                prediction_ids = _trim_token_ids(
                    prediction.detach().cpu().tolist(), model.tree_tokenizer
                )
                edit = _token_edit_distance(target_ids, prediction_ids)
                motif = str(sample.metadata.get("motif", "unknown"))
                target_names = [model.tree_tokenizer.tokens[value] for value in target_ids]
                prediction_names = [
                    model.tree_tokenizer.tokens[value] for value in prediction_ids
                ]
                token_accuracy: dict[str, tuple[int, int]] = {}
                for category, vocabulary in (
                    ("operator", set(model.tree_tokenizer.operator_tokens)),
                    ("arity", set(model.tree_tokenizer.arity_tokens)),
                    ("activity_copy", set(model.tree_tokenizer.activity_tokens)),
                ):
                    positions = [
                        index for index, name in enumerate(target_names) if name in vocabulary
                    ]
                    correct = sum(
                        index < len(prediction_names)
                        and prediction_names[index] == target_names[index]
                        for index in positions
                    )
                    token_accuracy[category] = (correct, len(positions))
                discovery_rows.append(
                    {
                        "motif": motif,
                        "ordinary": motif == "ordinary_tree",
                        "loop": _tree_has_loop(sample.tree),
                        "exact": prediction_ids == target_ids,
                        "normalized_edit": edit
                        / max(len(target_ids), len(prediction_ids), 1),
                        "target": target_ids,
                        "prediction": prediction_ids,
                        "token_accuracy": token_accuracy,
                        "shuffled_exact": (
                            False
                            if shuffled_decoded is None
                            else _trim_token_ids(
                                shuffled_decoded[row_index].detach().cpu().tolist(),
                                model.tree_tokenizer,
                            )
                            == target_ids
                        ),
                        "zero_exact": (
                            False
                            if zero_decoded is None
                            else _trim_token_ids(
                                zero_decoded[row_index].detach().cpu().tolist(),
                                model.tree_tokenizer,
                            )
                            == target_ids
                        ),
                        "source_ablation": source_ablation,
                    }
                )
            for name, distribution in dists.items():
                embedding_rows[name].append(distribution.mu.detach().cpu())
            batch_exact_ids = list(batch.get("exact_behavior_ids", []))
            batch_partial_ids = list(batch.get("partial_order_ids", []))
            batch_positive_mask = batch.get("positive_mask")
            batch_candidate_mask = batch.get("contrastive_candidate_mask")
            if isinstance(batch_positive_mask, torch.Tensor) and isinstance(
                batch_candidate_mask, torch.Tensor
            ):
                for left in range(len(batch_exact_ids)):
                    for right in range(left + 1, len(batch_exact_ids)):
                        same_language = (
                            batch_exact_ids[left] is not None
                            and batch_exact_ids[left] == batch_exact_ids[right]
                        )
                        same_partial_order = (
                            batch_partial_ids[left] is not None
                            and batch_partial_ids[left] == batch_partial_ids[right]
                        )
                        if same_language and same_partial_order:
                            expected_positive_pairs += 1
                            false_negative_pairs += int(
                                not bool(batch_positive_mask[left, right])
                            )
                        elif same_language:
                            analogy_pairs += 1
                            analogy_negative_pairs += int(
                                bool(batch_candidate_mask[left, right])
                            )
            exact_behavior_ids.extend(batch_exact_ids)
            partial_order_ids.extend(batch_partial_ids)
            signatures = batch.get("behavior_signatures")
            if isinstance(signatures, torch.Tensor):
                signature_rows.append(signatures.detach().cpu())
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{float(losses['loss'].detach().cpu()):.4f}")
    metrics = {name: value / max(batches, 1) for name, value in totals.items()}
    if compute_discovery_metrics:
        discovery_metrics = _summarize_discovery_metrics(
            discovery_rows,
            embedding_rows,
            exact_behavior_ids,
            partial_order_ids,
            signature_rows,
        )
        discovery_metrics["false_negative_rate"] = (
            false_negative_pairs / expected_positive_pairs
            if expected_positive_pairs
            else 0.0
        )
        discovery_metrics["analogy_negative_rate"] = (
            analogy_negative_pairs / analogy_pairs if analogy_pairs else 0.0
        )
        metrics.update(discovery_metrics)
    return metrics


def _different_behavior_permutation(
    exact_behavior_ids: object,
    row_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a deterministic cyclic derangement across exact behaviors.

    Stage data places multiple observation views of one behavior next to each
    other, so a one-row roll is not a valid source ablation.  A cyclic shift is
    accepted only when every row receives memory from a different exact
    behavior; otherwise the batch cannot support this control reliably.
    """

    if not isinstance(exact_behavior_ids, list) or len(exact_behavior_ids) != row_count:
        raise ValueError("source ablation requires one exact behavior ID per row")
    identifiers = [str(value) for value in exact_behavior_ids]
    base = torch.arange(row_count, device=device)
    for offset in range(1, row_count):
        candidate = torch.roll(base, offset)
        if all(
            identifiers[row] != identifiers[int(candidate[row])]
            for row in range(row_count)
        ):
            return candidate
    raise ValueError(
        "source ablation requires a batch that admits a different-behavior cyclic permutation"
    )


def _trim_token_ids(token_ids: list[int], tokenizer: TreeTokenizer) -> list[int]:
    result = [int(value) for value in token_ids if int(value) != tokenizer.pad_id]
    if tokenizer.eos_id in result:
        result = result[: result.index(tokenizer.eos_id) + 1]
    return result


def _token_edit_distance(left: list[int], right: list[int]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + int(left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _tree_has_loop(tree) -> bool:
    from proc_rosetta.tree import NodeKind

    return tree.kind is NodeKind.LOOP or any(_tree_has_loop(child) for child in tree.children)


def _summarize_discovery_metrics(
    rows: list[dict[str, object]],
    embeddings: dict[str, list[torch.Tensor]],
    exact_ids: list[str | None],
    partial_order_ids: list[str | None],
    signatures: list[torch.Tensor],
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    def rate(selected: list[dict[str, object]]) -> float:
        return (
            sum(bool(row["exact"]) for row in selected) / len(selected)
            if selected
            else 0.0
        )

    ordinary = [row for row in rows if bool(row["ordinary"])]
    loops = [row for row in rows if bool(row["loop"])]
    nonloops = [row for row in rows if not bool(row["loop"])]
    metrics["trace_canonical_exact"] = rate(rows)
    metrics["ordinary_trace_canonical_exact"] = rate(ordinary)
    metrics["loop_trace_canonical_exact"] = rate(loops)
    metrics["nonloop_trace_canonical_exact"] = rate(nonloops)
    if rows and any(bool(row.get("source_ablation", False)) for row in rows):
        metrics["shuffled_trace_canonical_exact"] = sum(
            bool(row.get("shuffled_exact", False)) for row in rows
        ) / len(rows)
        metrics["zero_trace_canonical_exact"] = sum(
            bool(row.get("zero_exact", False)) for row in rows
        ) / len(rows)
    metrics["trace_normalized_tree_edit"] = (
        float(np.mean([float(row["normalized_edit"]) for row in rows]))
        if rows
        else 1.0
    )
    for category in ("operator", "arity", "activity_copy"):
        correct = sum(
            int(row["token_accuracy"][category][0]) for row in rows
        )
        count = sum(int(row["token_accuracy"][category][1]) for row in rows)
        metrics[f"trace_{category}_accuracy"] = correct / count if count else 0.0
    for motif in sorted({str(row["motif"]) for row in rows}):
        selected = [row for row in rows if row["motif"] == motif]
        metrics[f"trace_canonical_exact_{motif}"] = rate(selected)

    matrices = {
        name: torch.cat(chunks, dim=0)
        for name, chunks in embeddings.items()
        if chunks
    }
    for name, matrix in matrices.items():
        metrics[f"effective_rank_{name}"] = float(_effective_rank_tensor(matrix))
        metrics[f"mean_dimension_std_{name}"] = float(
            matrix.std(dim=0, unbiased=False).mean()
        )
    recalls: list[float] = []
    for left_name, left in matrices.items():
        for right_name, right in matrices.items():
            if left_name == right_name or left.shape[0] != len(exact_ids):
                continue
            similarity = torch.nn.functional.normalize(left, dim=-1) @ torch.nn.functional.normalize(
                right, dim=-1
            ).T
            valid = 0
            hit1 = 0
            hit5 = 0
            for index, behavior_id in enumerate(exact_ids):
                if behavior_id is None:
                    continue
                valid += 1
                order = similarity[index].argsort(descending=True)
                candidate_ids = [exact_ids[int(position)] for position in order[:5]]
                hit1 += int(candidate_ids[0] == behavior_id)
                hit5 += int(behavior_id in candidate_ids)
            if valid:
                prefix = f"exact_behavior_{left_name}_to_{right_name}"
                metrics[f"{prefix}_recall_at_1"] = hit1 / valid
                metrics[f"{prefix}_recall_at_5"] = hit5 / valid
                recalls.append(hit1 / valid)
    metrics["exact_behavior_recall_at_1"] = float(np.mean(recalls)) if recalls else 0.0

    if "trace" in matrices and matrices["trace"].shape[0] == len(exact_ids):
        similarity = torch.nn.functional.normalize(
            matrices["trace"], dim=-1
        ) @ torch.nn.functional.normalize(matrices["trace"], dim=-1).T
        positives: list[float] = []
        negatives: list[float] = []
        for left in range(len(exact_ids)):
            for right in range(left + 1, len(exact_ids)):
                strong = (
                    exact_ids[left] is not None
                    and exact_ids[left] == exact_ids[right]
                    and partial_order_ids[left] is not None
                    and partial_order_ids[left] == partial_order_ids[right]
                )
                (positives if strong else negatives).append(float(similarity[left, right]))
        for label, values in (("positive", positives), ("negative", negatives)):
            if values:
                for quantile in (0.1, 0.5, 0.9):
                    metrics[
                        f"trace_{label}_cosine_q{int(quantile * 100):02d}"
                    ] = float(np.quantile(values, quantile))

    if signatures and "trace" in matrices:
        signature_matrix = torch.cat(signatures, dim=0)
        if signature_matrix.shape[0] == matrices["trace"].shape[0]:
            behavior_distances = 1.0 - torch.nn.functional.normalize(
                signature_matrix, dim=-1
            ) @ torch.nn.functional.normalize(signature_matrix, dim=-1).T
            latent_distances = 1.0 - torch.nn.functional.normalize(
                matrices["trace"], dim=-1
            ) @ torch.nn.functional.normalize(matrices["trace"], dim=-1).T
            upper = torch.triu_indices(
                behavior_distances.shape[0], behavior_distances.shape[1], offset=1
            )
            if upper.shape[1] >= 2:
                metrics["behavior_distance_spearman"] = _spearman(
                    behavior_distances[upper[0], upper[1]],
                    latent_distances[upper[0], upper[1]],
                )
    primary_exact = (
        metrics["ordinary_trace_canonical_exact"]
        if ordinary
        else metrics["trace_canonical_exact"]
    )
    metrics["checkpoint_selection_score"] = (
        primary_exact
        - 0.01 * metrics["trace_normalized_tree_edit"]
        + 0.001 * metrics["exact_behavior_recall_at_1"]
    )
    return metrics


def _effective_rank_tensor(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[0] < 2:
        return matrix.new_tensor(0.0)
    values = torch.linalg.svdvals(matrix - matrix.mean(dim=0, keepdim=True))
    probability = values / values.sum().clamp_min(1e-12)
    return (-(probability * probability.clamp_min(1e-12).log()).sum()).exp()


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    left_rank = left.argsort().argsort().to(torch.float32)
    right_rank = right.argsort().argsort().to(torch.float32)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = left_rank.norm() * right_rank.norm()
    return float((left_rank @ right_rank) / denominator.clamp_min(1e-12))


def build_synthetic_dataloader(
    samples: int,
    synthetic_config: SyntheticConfig,
    tree_tokenizer: TreeTokenizer,
    activity_tokenizer: ActivityTokenizer,
    batch_size: int,
    seed: int,
    batch_config: BatchConfig | None = None,
    activity_remap_probability: float = 0.0,
) -> DataLoader:
    dataset = SyntheticProcessDataset(samples, config=synthetic_config, seed=seed)
    collator = ProcessBatchCollator(
        tree_tokenizer,
        activity_tokenizer,
        config=batch_config,
        activity_remap_probability=activity_remap_probability,
        seed=seed,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)


def build_jsonl_dataloader(
    sample_path: str | Path,
    tree_tokenizer: TreeTokenizer,
    activity_tokenizer: ActivityTokenizer,
    batch_size: int,
    shuffle: bool = False,
    batch_config: BatchConfig | None = None,
    show_progress: bool = False,
    group_aware: bool = False,
    views_per_family: int = 2,
    seed: int = 13,
    activity_remap_probability: float = 0.0,
) -> DataLoader:
    dataset = JsonlProcessDataset(sample_path, show_progress=show_progress)
    collator = ProcessBatchCollator(
        tree_tokenizer,
        activity_tokenizer,
        config=batch_config,
        activity_remap_probability=activity_remap_probability,
        seed=seed,
    )
    if group_aware:
        batch_sampler = BehaviorFamilyBatchSampler(
            dataset.samples,
            batch_size=batch_size,
            views_per_family=views_per_family,
            shuffle=shuffle,
            seed=seed,
        )
        return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collator)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


class BehaviorFamilyBatchSampler(Sampler[list[int]]):
    """Keep multiple views of a behavior together so batches contain positives."""

    def __init__(
        self,
        samples,
        *,
        batch_size: int,
        views_per_family: int = 2,
        shuffle: bool = True,
        seed: int = 13,
    ) -> None:
        self.batch_size = max(1, int(batch_size))
        self.views_per_family = max(1, int(views_per_family))
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            grouped[str(sample.equivalence_id)].append(index)
        self.groups = list(grouped.values())
        self.sample_count = len(samples)

    def __iter__(self):
        import random

        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        groups = [list(group) for group in self.groups]
        if self.shuffle:
            rng.shuffle(groups)
            for group in groups:
                rng.shuffle(group)

        per_group_chunks = [
            [
                group[start : start + self.views_per_family]
                for start in range(0, len(group), self.views_per_family)
            ]
            for group in groups
        ]
        # Interleave chunk rounds so a 128-row batch with four views contains
        # 32 distinct behaviors before any family contributes a second chunk.
        chunks: list[list[int]] = []
        for round_index in range(max((len(value) for value in per_group_chunks), default=0)):
            chunks.extend(
                value[round_index]
                for value in per_group_chunks
                if round_index < len(value)
            )
        batch: list[int] = []
        for chunk in chunks:
            if batch and len(batch) + len(chunk) > self.batch_size:
                yield batch
                batch = []
            if len(chunk) > self.batch_size:
                for start in range(0, len(chunk), self.batch_size):
                    yield chunk[start : start + self.batch_size]
            else:
                batch.extend(chunk)
        if batch:
            yield batch

    def __len__(self) -> int:
        return (self.sample_count + self.batch_size - 1) // self.batch_size


def build_model(
    train_config: TrainConfig,
    synthetic_config: SyntheticConfig,
    device: torch.device,
) -> ProcRosettaModel:
    if train_config.semantic_latent_mode != "deterministic":
        raise ValueError(
            "semantic_latent_mode must be 'deterministic'; stochastic uncertainty "
            "is not supported on the supervised semantic path"
        )
    tree_tokenizer = TreeTokenizer(
        max_activities=synthetic_config.max_activities,
        max_arity=max(3, synthetic_config.max_arity),
    )
    activity_tokenizer = ActivityTokenizer(max_activities=synthetic_config.max_activities)
    return ProcRosettaModel(
        tree_tokenizer=tree_tokenizer,
        activity_tokenizer=activity_tokenizer,
        latent_dim=train_config.latent_dim,
        hidden_dim=train_config.hidden_dim,
        dropout=train_config.dropout,
        memory_tokens=train_config.memory_tokens,
        decoder_layers=train_config.decoder_layers,
    ).to(device)


def train_synthetic(
    train_config: TrainConfig | None = None,
    synthetic_config: SyntheticConfig | None = None,
) -> tuple[ProcRosettaModel, list[dict[str, float]]]:
    train_config = train_config or TrainConfig()
    synthetic_config = synthetic_config or SyntheticConfig()
    torch.manual_seed(train_config.seed)
    device = resolve_device(train_config.device)

    model = build_model(train_config, synthetic_config, device)
    dataloader = build_synthetic_dataloader(
        samples=train_config.samples,
        synthetic_config=synthetic_config,
        tree_tokenizer=model.tree_tokenizer,
        activity_tokenizer=model.activity_tokenizer,
        batch_size=train_config.batch_size,
        seed=train_config.seed,
        activity_remap_probability=train_config.activity_remap_probability,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    weights = loss_weights_from_config(train_config)
    history = [
        train_epoch(
            model,
            dataloader,
            optimizer,
            device,
            weights=weights,
            epoch=epoch,
            train_config=train_config,
        )
        for epoch in range(1, train_config.epochs + 1)
    ]
    return model, history


def train_from_data_dir(
    data_dir: str | Path = "data",
    checkpoint_path: str | Path = "checkpoints/proc_rosetta.pt",
    train_config: TrainConfig | None = None,
    show_progress: bool = True,
    metrics_csv_path: str | Path = "checkpoints/training_metrics.csv",
    resume: bool = False,
) -> tuple[ProcRosettaModel, list[dict[str, object]]]:
    train_config = train_config or TrainConfig()
    torch.manual_seed(train_config.seed)
    device = resolve_device(train_config.device)
    debug(f"Loading metadata from {Path(data_dir) / 'metadata.json'}", enabled=show_progress)
    metadata = load_data_metadata(data_dir)
    if int(metadata.get("version", 0)) < 4:
        raise ValueError(
            "data uses the legacy split/index equivalence schema; recreate it with "
            "sample.py before training the deterministic semantic model"
        )
    if not bool(metadata.get("exact_behavior_signatures_disjoint", False)):
        raise ValueError("data metadata does not certify signature-disjoint splits")
    synthetic_config = SyntheticConfig.from_dict(metadata.get("synthetic_config", {}))
    debug(
        "Training configuration: "
        f"epochs={train_config.epochs}, batch_size={train_config.batch_size}, "
        f"lr={train_config.learning_rate}, latent_dim={train_config.latent_dim}, "
        f"hidden_dim={train_config.hidden_dim}, dropout={train_config.dropout}, "
        f"weight_decay={train_config.weight_decay}, label_smoothing={train_config.label_smoothing}, "
        f"activity_remap_probability={train_config.activity_remap_probability}, "
        f"early_stopping_patience={train_config.early_stopping_patience}, device={device}",
        enabled=show_progress,
    )
    debug(
        "Synthetic data configuration: "
        f"max_depth={synthetic_config.max_depth}, max_activities={synthetic_config.max_activities}, "
        f"max_arity={synthetic_config.max_arity}, traces_per_sample={synthetic_config.traces_per_sample}, "
        f"curriculum_phase={synthetic_config.curriculum_phase}",
        enabled=show_progress,
    )
    resume_checkpoint: dict[str, object] | None = None
    if resume:
        model, resume_checkpoint = load_checkpoint(checkpoint_path, device)
        validate_resume_configuration(
            checkpoint=resume_checkpoint,
            train_config=train_config,
            synthetic_config=synthetic_config,
        )
    else:
        model = build_model(train_config, synthetic_config, device)
    debug("Loading training split", enabled=show_progress)
    train_loader = build_jsonl_dataloader(
        split_samples_path(data_dir, "training"),
        model.tree_tokenizer,
        model.activity_tokenizer,
        batch_size=train_config.batch_size,
        shuffle=True,
        show_progress=show_progress,
        group_aware=train_config.group_aware_batches,
        views_per_family=train_config.views_per_family,
        seed=train_config.seed,
        activity_remap_probability=train_config.activity_remap_probability,
    )
    debug("Loading validation split", enabled=show_progress)
    validation_loader = build_jsonl_dataloader(
        split_samples_path(data_dir, "validation"),
        model.tree_tokenizer,
        model.activity_tokenizer,
        batch_size=train_config.batch_size,
        shuffle=False,
        show_progress=show_progress,
    )
    stage_training_loader = (
        build_jsonl_dataloader(
            split_samples_path(data_dir, "training"),
            model.tree_tokenizer,
            model.activity_tokenizer,
            batch_size=train_config.batch_size,
            shuffle=False,
            show_progress=False,
        )
        if train_config.training_stage == "a"
        else None
    )
    debug_split("training", train_loader.dataset.samples, len(train_loader), enabled=show_progress)
    debug_split("validation", validation_loader.dataset.samples, len(validation_loader), enabled=show_progress)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=train_config.lr_factor,
        patience=train_config.lr_patience,
        min_lr=train_config.min_lr,
    )
    weights = loss_weights_from_config(train_config)
    history: list[dict[str, object]] = []
    best_validation_loss = float("inf")
    best_validation_score = float("-inf")
    epochs_without_improvement = 0
    start_epoch = 1
    if resume_checkpoint is not None:
        completed_epoch = int(resume_checkpoint.get("epoch", 0))
        history = [dict(row) for row in resume_checkpoint.get("history", [])]
        if history and int(history[-1].get("epoch", -1)) != completed_epoch:
            raise ValueError(
                "checkpoint history does not end at its completed epoch: "
                f"epoch={completed_epoch}, history_epoch={history[-1].get('epoch')}"
            )
        start_epoch = completed_epoch + 1
        stored_best = resume_checkpoint.get("best_validation_loss")
        if stored_best is not None:
            best_validation_loss = float(stored_best)
        stored_best_score = resume_checkpoint.get("best_validation_score")
        if stored_best_score is not None:
            best_validation_score = float(stored_best_score)
        if history:
            epochs_without_improvement = int(
                history[-1].get("epochs_without_improvement", 0)
            )
        restore_training_state(
            checkpoint=resume_checkpoint,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            device=device,
            completed_epoch=completed_epoch,
            history=history,
            seed=train_config.seed,
            show_progress=show_progress,
        )
        debug(
            f"Resuming from {checkpoint_path} after epoch {completed_epoch}; "
            f"target epoch={train_config.epochs}",
            enabled=show_progress,
        )
    best_checkpoint_path = best_checkpoint_for(checkpoint_path)
    metrics_csv_path = Path(metrics_csv_path)
    write_metrics_csv(metrics_csv_path, history)
    debug(f"Per-epoch metrics CSV: {metrics_csv_path}", enabled=show_progress)
    for epoch in range(start_epoch, train_config.epochs + 1):
        epoch_start = perf_counter()
        debug(f"Starting epoch {epoch}/{train_config.epochs}", enabled=show_progress)
        training_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            weights=weights,
            epoch=epoch,
            show_progress=show_progress,
            train_config=train_config,
        )
        debug(
            f"Epoch {epoch} training complete: {format_metrics(training_metrics)}",
            enabled=show_progress,
        )
        validation_metrics = evaluate_epoch(
            model,
            validation_loader,
            device,
            weights=weights,
            epoch=epoch,
            show_progress=show_progress,
            compute_discovery_metrics=True,
        )
        run_stage_gate = (
            stage_training_loader is not None
            and (
                epoch % max(train_config.stage_gate_interval, 1) == 0
                or epoch == train_config.epochs
            )
        )
        if run_stage_gate:
            stage_metrics = evaluate_epoch(
                model,
                stage_training_loader,
                device,
                weights=weights,
                progress_desc="Stage A training-family gate",
                compute_discovery_metrics=True,
                source_ablation=True,
            )
            acceptance = stage_acceptance_report(
                train_config.training_stage,
                stage_metrics,
                train_config.latent_dim,
            )
        elif train_config.training_stage == "a":
            stage_metrics = None
            acceptance = {
                "stage": "a",
                "evaluated": False,
                "passed": False,
                "checks": {},
                "measurements": {},
            }
        else:
            stage_metrics = validation_metrics
            acceptance = stage_acceptance_report(
                train_config.training_stage,
                stage_metrics,
                train_config.latent_dim,
            )
        current_lr = optimizer.param_groups[0]["lr"]
        validation_score = validation_metrics["checkpoint_selection_score"]
        scheduler.step(validation_score)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < current_lr:
            debug(
                f"Validation plateau detected; reducing learning rate {current_lr:.6g} -> {new_lr:.6g}",
                enabled=show_progress,
            )

        validation_loss = validation_metrics["loss"]
        best_validation_loss = min(best_validation_loss, validation_loss)
        improved = validation_score > (best_validation_score + train_config.min_delta)
        if improved:
            best_validation_score = validation_score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        gap = validation_metrics["loss"] - training_metrics["loss"]
        debug(
            f"Epoch {epoch} validation complete: {format_metrics(validation_metrics)} "
            f"| gap={gap:+.4f} | selection={validation_score:.4f} "
            f"| best_selection={best_validation_score:.4f} "
            f"| patience={epochs_without_improvement}/{train_config.early_stopping_patience}",
            enabled=show_progress,
        )
        elapsed = perf_counter() - epoch_start
        row: dict[str, object] = {
            "epoch": epoch,
            "training": training_metrics,
            "validation": validation_metrics,
            "generalization_gap": metric_gaps(training_metrics, validation_metrics),
            "learning_rate": new_lr,
            "epoch_seconds": elapsed,
            "best_validation_loss": best_validation_loss,
            "best_validation_score": best_validation_score,
            "is_best": improved,
            "epochs_without_improvement": epochs_without_improvement,
            "stage_acceptance": acceptance,
        }
        if stage_metrics is not None and run_stage_gate:
            row["stage_metrics"] = stage_metrics
        history.append(row)
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            train_config=train_config,
            synthetic_config=synthetic_config,
            history=history,
            epoch=epoch,
            best_validation_loss=best_validation_loss,
            best_validation_score=best_validation_score,
            is_best=False,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            device=device,
        )
        if improved:
            save_checkpoint(
                checkpoint_path=best_checkpoint_path,
                model=model,
                train_config=train_config,
                synthetic_config=synthetic_config,
                history=history,
                epoch=epoch,
                best_validation_loss=best_validation_loss,
                best_validation_score=best_validation_score,
                is_best=True,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                device=device,
            )
        append_metrics_csv(metrics_csv_path, row)
        debug(
            f"Epoch {epoch} checkpoint saved to {checkpoint_path}; "
            f"best checkpoint: {best_checkpoint_path if improved else 'unchanged'} ({elapsed:.1f}s)",
            enabled=show_progress,
        )
        if train_config.training_stage == "a" and bool(acceptance["passed"]):
            debug(
                f"Stage A acceptance gate passed after epoch {epoch}; stopping the tiny-overfit run.",
                enabled=show_progress,
            )
            break
        if (
            train_config.early_stopping_patience > 0
            and epochs_without_improvement >= train_config.early_stopping_patience
        ):
            debug(
                f"Early stopping after {epoch} epochs; ordinary trace discovery did not improve by "
                f"{train_config.min_delta:g} for {train_config.early_stopping_patience} epochs.",
                enabled=show_progress,
            )
            break
    return model, history


def stage_acceptance_report(
    stage: str,
    metrics: dict[str, float],
    semantic_dimension: int,
) -> dict[str, object]:
    """Evaluate the explicit gates before advancing the experiment stage."""

    measurements: dict[str, float]
    if stage == "a":
        measurements = {
            "training_trace_canonical_exact": metrics.get(
                "trace_canonical_exact", 0.0
            ),
            "shuffled_trace_canonical_exact": metrics.get(
                "shuffled_trace_canonical_exact", 1.0
            ),
            "zero_trace_canonical_exact": metrics.get(
                "zero_trace_canonical_exact", 1.0
            ),
        }
        checks = {
            "training_trace_exact_at_least_0_95": measurements[
                "training_trace_canonical_exact"
            ]
            >= 0.95,
            "shuffled_source_exact_at_most_0_10": measurements[
                "shuffled_trace_canonical_exact"
            ]
            <= 0.10,
            "zero_source_exact_at_most_0_10": measurements[
                "zero_trace_canonical_exact"
            ]
            <= 0.10,
        }
    elif stage == "b":
        measurements = {
            "false_negative_rate": metrics.get("false_negative_rate", 1.0),
            "effective_rank_tree": metrics.get("effective_rank_tree", 0.0),
            "effective_rank_trace": metrics.get("effective_rank_trace", 0.0),
            "effective_rank_petri": metrics.get("effective_rank_petri", 0.0),
            "exact_behavior_recall_at_1": metrics.get(
                "exact_behavior_recall_at_1", 0.0
            ),
        }
        rank_gate = min(32.0, float(semantic_dimension) / 2.0)
        checks = {
            "false_negative_rate_zero": measurements["false_negative_rate"] == 0.0,
            "effective_rank_gate": all(
                measurements[f"effective_rank_{name}"] > rank_gate
                for name in ("tree", "trace", "petri")
            ),
            "exact_behavior_recall_at_1_at_least_0_90": measurements[
                "exact_behavior_recall_at_1"
            ]
            >= 0.90,
        }
    else:
        measurements = {
            "trace_canonical_exact": metrics.get("trace_canonical_exact", 0.0),
            "behavior_distance_spearman": metrics.get(
                "behavior_distance_spearman", 0.0
            ),
        }
        checks = {
            "discovery_metrics_present": "trace_canonical_exact" in metrics,
            "behavior_geometry_measured": "behavior_distance_spearman" in metrics,
        }
    return {
        "stage": stage,
        "evaluated": True,
        "passed": all(checks.values()),
        "checks": checks,
        "measurements": measurements,
    }


def evaluate_split_from_checkpoint(
    checkpoint_path: str | Path = "checkpoints/proc_rosetta.pt",
    data_dir: str | Path = "data",
    split: str = "test",
    batch_size: int = 16,
    device: str | None = None,
    show_progress: bool = False,
) -> dict[str, float]:
    torch_device = resolve_device(device)
    model, checkpoint = load_checkpoint(checkpoint_path, torch_device)
    checkpoint_train_config = train_config_from_checkpoint(checkpoint, torch_device)
    loader = build_jsonl_dataloader(
        split_samples_path(data_dir, split),
        model.tree_tokenizer,
        model.activity_tokenizer,
        batch_size=batch_size,
        shuffle=False,
    )
    return evaluate_epoch(
        model,
        loader,
        torch_device,
        weights=loss_weights_from_checkpoint(checkpoint, checkpoint_train_config),
        show_progress=show_progress,
        progress_desc=f"{split.title()} loss",
        compute_discovery_metrics=True,
    )


def progress_dataloader(dataloader: DataLoader, desc: str, enabled: bool):
    if not enabled:
        return dataloader
    from tqdm.auto import tqdm

    return tqdm(dataloader, desc=desc, total=len(dataloader), leave=False, unit="batch")


def debug(message: str, enabled: bool = True) -> None:
    if enabled:
        print(f"[train] {message}", file=sys.stderr, flush=True)


def debug_split(
    split: str,
    samples,
    batch_count: int,
    enabled: bool = True,
) -> None:
    if not enabled:
        return
    stats = sample_statistics(samples)
    debug(
        f"{split}: {stats['count']} samples, {batch_count} batches, "
        f"avg_tree_size={stats['avg_tree_size']:.2f}, "
        f"avg_trace_count={stats['avg_trace_count']:.2f}, "
        f"avg_trace_length={stats['avg_trace_length']:.2f}, "
        f"max_petri_nodes={stats['max_petri_nodes']}",
        enabled=enabled,
    )


def format_metrics(metrics: dict[str, float]) -> str:
    names = [
        "loss",
        "tree_reconstruction",
        "trace_to_tree",
        "petri_to_tree",
        "contrastive",
        "kl",
        "latent_alignment",
    ]
    return ", ".join(f"{name}={metrics[name]:.4f}" for name in names if name in metrics)


def metric_gaps(training_metrics: dict[str, float], validation_metrics: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(training_metrics) & set(validation_metrics))
    return {key: validation_metrics[key] - training_metrics[key] for key in keys}


def best_checkpoint_for(checkpoint_path: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.best{checkpoint_path.suffix}")


CSV_METRIC_NAMES = (
    "loss",
    "tree_reconstruction",
    "trace_to_tree",
    "petri_to_tree",
    "latent_alignment",
    "contrastive",
    "exact_contrastive",
    "within_modality_contrastive",
    "soft_behavior_geometry",
    "variance",
    "covariance",
    "effective_rank",
    "kl",
    "trace_canonical_exact",
    "ordinary_trace_canonical_exact",
    "trace_normalized_tree_edit",
    "exact_behavior_recall_at_1",
    "behavior_distance_spearman",
    "false_negative_rate",
    "checkpoint_selection_score",
)


def metrics_csv_columns() -> list[str]:
    columns = [
        "epoch",
        "learning_rate",
        "epoch_seconds",
        "best_validation_loss",
        "best_validation_score",
        "is_best",
        "epochs_without_improvement",
    ]
    for prefix in ("training", "validation", "gap"):
        columns.extend(f"{prefix}_{name}" for name in CSV_METRIC_NAMES)
    return columns


def init_metrics_csv(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics_csv_columns())
        writer.writeheader()


def write_metrics_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    """Synchronize metrics with checkpoint history without leaving a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics_csv_columns())
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_epoch_row(row))
    temporary_path.replace(path)


def append_metrics_csv(path: str | Path, row: dict[str, object]) -> None:
    flat = flatten_epoch_row(row)
    with Path(path).open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics_csv_columns())
        writer.writerow(flat)


def flatten_epoch_row(row: dict[str, object]) -> dict[str, object]:
    training = row["training"]
    validation = row["validation"]
    gap = row["generalization_gap"]
    assert isinstance(training, dict)
    assert isinstance(validation, dict)
    assert isinstance(gap, dict)
    flat: dict[str, object] = {
        "epoch": row["epoch"],
        "learning_rate": row["learning_rate"],
        "epoch_seconds": row["epoch_seconds"],
        "best_validation_loss": row["best_validation_loss"],
        "best_validation_score": row.get("best_validation_score", ""),
        "is_best": row["is_best"],
        "epochs_without_improvement": row["epochs_without_improvement"],
    }
    for name in CSV_METRIC_NAMES:
        flat[f"training_{name}"] = training.get(name, "")
        flat[f"validation_{name}"] = validation.get(name, "")
        flat[f"gap_{name}"] = gap.get(name, "")
    return flat


def save_checkpoint(
    checkpoint_path: str | Path,
    model: ProcRosettaModel,
    train_config: TrainConfig,
    synthetic_config: SyntheticConfig,
    history: list[dict[str, object]],
    epoch: int,
    best_validation_loss: float | None = None,
    best_validation_score: float | None = None,
    is_best: bool = False,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    train_loader: DataLoader | None = None,
    device: torch.device | None = None,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 4,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "train_config": asdict(train_config),
        "synthetic_config": synthetic_config.to_dict(),
        "history": history,
        "best_validation_loss": best_validation_loss,
        "best_validation_score": best_validation_score,
        "loss_weights": asdict(loss_weights_from_config(train_config)),
        "semantic_latent_mode": train_config.semantic_latent_mode,
        "semantic_latent_stochastic": train_config.semantic_latent_mode != "deterministic",
        "is_best": is_best,
    }
    if optimizer is not None and scheduler is not None and train_loader is not None:
        payload.update(
            {
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "rng_state": capture_rng_state(device or torch.device("cpu")),
                "training_loader_state": capture_training_loader_state(train_loader),
            }
        )
    temporary_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    torch.save(
        payload,
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)


def validate_resume_configuration(
    checkpoint: dict[str, object],
    train_config: TrainConfig,
    synthetic_config: SyntheticConfig,
) -> None:
    checkpoint_train_config = train_config_from_checkpoint(checkpoint, train_config.device)
    checkpoint_values = asdict(checkpoint_train_config)
    requested_values = asdict(train_config)
    differences = {
        name: (checkpoint_values[name], requested_values[name])
        for name in checkpoint_values
        if name not in {"epochs", "device"}
        and checkpoint_values[name] != requested_values[name]
    }
    if differences:
        formatted = ", ".join(
            f"{name}: checkpoint={old!r}, requested={new!r}"
            for name, (old, new) in sorted(differences.items())
        )
        raise ValueError(f"resume configuration differs from checkpoint ({formatted})")

    checkpoint_synthetic = SyntheticConfig.from_dict(checkpoint["synthetic_config"])
    if checkpoint_synthetic.to_dict() != synthetic_config.to_dict():
        raise ValueError("resume data configuration differs from checkpoint synthetic_config")


def train_config_from_checkpoint(
    checkpoint: dict[str, object], device: torch.device | str
) -> TrainConfig:
    train_config_data = asdict(TrainConfig())
    train_config_data.update(dict(checkpoint["train_config"]))
    train_config_data["device"] = str(device)
    return TrainConfig(**train_config_data)


def capture_rng_state(device: torch.device) -> dict[str, object]:
    state: dict[str, object] = {
        "device_type": device.type,
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps" and torch.backends.mps.is_available():
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict[str, object], device: torch.device) -> bool:
    python_state = state.get("python")
    cpu_state = state.get("torch_cpu")
    if python_state is not None:
        random.setstate(python_state)
    if isinstance(cpu_state, torch.Tensor):
        torch.set_rng_state(cpu_state.cpu())

    same_device_type = state.get("device_type") == device.type
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_states = state.get("torch_cuda")
        if isinstance(cuda_states, list) and all(
            isinstance(item, torch.Tensor) for item in cuda_states
        ):
            torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])
        else:
            same_device_type = False
    elif device.type == "mps" and torch.backends.mps.is_available():
        mps_state = state.get("torch_mps")
        if isinstance(mps_state, torch.Tensor):
            torch.mps.set_rng_state(mps_state.cpu())
        else:
            same_device_type = False
    return same_device_type and python_state is not None and isinstance(cpu_state, torch.Tensor)


def capture_training_loader_state(train_loader: DataLoader) -> dict[str, object]:
    state: dict[str, object] = {}
    batch_sampler = getattr(train_loader, "batch_sampler", None)
    if hasattr(batch_sampler, "epoch"):
        state["batch_sampler_epoch"] = int(batch_sampler.epoch)
    collator_rng = getattr(getattr(train_loader, "collate_fn", None), "rng", None)
    if isinstance(collator_rng, random.Random):
        state["collator_rng"] = collator_rng.getstate()
    return state


def restore_training_loader_state(
    train_loader: DataLoader, state: dict[str, object], completed_epoch: int
) -> bool:
    restored = True
    batch_sampler = getattr(train_loader, "batch_sampler", None)
    if hasattr(batch_sampler, "epoch"):
        batch_sampler.epoch = int(state.get("batch_sampler_epoch", completed_epoch))
        restored = "batch_sampler_epoch" in state
    collator_rng = getattr(getattr(train_loader, "collate_fn", None), "rng", None)
    if isinstance(collator_rng, random.Random):
        saved_collator_rng = state.get("collator_rng")
        if saved_collator_rng is not None:
            collator_rng.setstate(saved_collator_rng)
        else:
            restored = False
    return restored


def replay_scheduler_history(
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    history: list[dict[str, object]],
) -> None:
    for row in history:
        validation = row.get("validation")
        if isinstance(validation, dict):
            if "checkpoint_selection_score" in validation:
                scheduler.step(float(validation["checkpoint_selection_score"]))
            elif "loss" in validation:
                scheduler.step(-float(validation["loss"]))


def restore_training_state(
    *,
    checkpoint: dict[str, object],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    train_loader: DataLoader,
    device: torch.device,
    completed_epoch: int,
    history: list[dict[str, object]],
    seed: int,
    show_progress: bool,
) -> None:
    optimizer_state = checkpoint.get("optimizer_state_dict")
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if (optimizer_state is None) != (scheduler_state is None):
        raise ValueError("checkpoint contains incomplete optimizer/scheduler resume state")

    if optimizer_state is None:
        replay_scheduler_history(scheduler, history)
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(batch_sampler, "epoch"):
            batch_sampler.epoch = completed_epoch
        torch.manual_seed(seed + completed_epoch)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + completed_epoch)
        debug(
            "Legacy checkpoint has no optimizer, scheduler, RNG, or augmentation state; "
            "continuing from its model weights with a freshly initialized optimizer.",
            enabled=show_progress,
        )
        return

    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)
    rng_state = checkpoint.get("rng_state")
    loader_state = checkpoint.get("training_loader_state")
    rng_restored = isinstance(rng_state, dict) and restore_rng_state(rng_state, device)
    loader_restored = isinstance(loader_state, dict) and restore_training_loader_state(
        train_loader, loader_state, completed_epoch
    )
    if rng_restored and loader_restored:
        debug(
            "Restored optimizer, scheduler, RNG, and data-loader state.",
            enabled=show_progress,
        )
    else:
        debug(
            "Restored optimizer and scheduler; exact RNG/data-loader continuation was not "
            "available for this device.",
            enabled=show_progress,
        )


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str | None = None,
) -> tuple[ProcRosettaModel, dict[str, object]]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    torch_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device)
    train_config = train_config_from_checkpoint(checkpoint, torch_device)
    synthetic_config = SyntheticConfig.from_dict(checkpoint["synthetic_config"])
    model = build_model(train_config, synthetic_config, torch_device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    allowed_missing = {"petri_encoder.transition_label_embedding.weight"}
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint/model mismatch: "
            f"missing={sorted(unexpected_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    model.eval()
    return model, checkpoint
