from __future__ import annotations

import csv
from collections import defaultdict
from contextlib import contextmanager, nullcontext
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
from proc_rosetta.pm4py_bridge import TREE_NORMALIZATION_VERSION
from proc_rosetta.synthetic import SyntheticConfig
from proc_rosetta.tokenizers import ActivityTokenizer, TreeTokenizer


CHECKPOINT_FORMAT_VERSION = 6
MODEL_ARCHITECTURE_VERSION = "proc-rosetta-latent-transformer-v6"
RESUME_POLICY_OVERRIDE_FIELDS = frozenset(
    {
        "scheduled_sampling_max",
        "scheduled_sampling_start_epoch",
        "scheduled_sampling_ramp_epochs",
    }
)


@dataclass(frozen=True)
class TrainConfig:
    samples: int = 128
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 3e-4
    latent_dim: int = 96
    hidden_dim: int = 192
    seed: int = 13
    device: str = default_device()
    semantic_latent_mode: str = "deterministic"
    # ``dropout`` is a deprecated compatibility override; new runs should use
    # the modality-specific controls below.
    dropout: float | None = None
    tree_encoder_dropout: float = 0.12
    trace_encoder_dropout: float = 0.20
    petri_encoder_dropout: float = 0.12
    decoder_dropout: float = 0.20
    projection_dropout: float = 0.20
    weight_decay: float = 5e-4
    label_smoothing: float = 0.04
    early_stopping_patience: int = 6
    min_delta: float = 0.005
    lr_patience: int = 1
    lr_factor: float = 0.5
    min_lr: float = 1e-5
    group_aware_batches: bool = True
    views_per_family: int = 2
    activity_remap_probability: float = 0.5
    memory_tokens: int = 6
    decoder_layers: int = 3
    tree_encoder_layers: int = 3
    trace_event_layers: int = 1
    trace_set_layers: int = 1
    petri_message_passing_steps: int = 5
    decoder_input_dropout: float = 0.15
    scheduled_sampling_max: float = 0.075
    scheduled_sampling_start_epoch: int = 20
    scheduled_sampling_ramp_epochs: int = 20
    gradient_clip_norm: float = 5.0
    tree_reconstruction_weight: float = 0.5
    trace_to_tree_weight: float = 2.0
    petri_to_tree_weight: float = 0.5
    exact_contrastive_weight: float = 0.30
    within_modality_contrastive_weight: float = 0.25
    soft_behavior_geometry_weight: float = 0.25
    variance_weight: float = 0.1
    covariance_weight: float = 0.01
    kl_weight: float = 0.0
    latent_alignment_weight: float = 0.075
    contrastive_temperature: float = 0.3
    behavior_temperature: float = 0.2
    latent_temperature: float = 0.2
    exact_contrastive_start_epoch: int = 3
    exact_contrastive_ramp_epochs: int = 4
    soft_geometry_start_epoch: int = 5
    soft_geometry_ramp_epochs: int = 6
    scheduler_monitor: str = "trace_to_tree"
    restore_best_weights: bool = True
    use_ema: bool = True
    ema_start_epoch: int = 3
    ema_decay: float = 0.995
    training_stage: str = "full"
    stage_gate_interval: int = 5
    gradient_diagnostics_interval: int = 0
    loader_num_workers: int = 0
    loader_pin_memory: bool = True
    loader_persistent_workers: bool = True
    loader_prefetch_factor: int = 2

    def __post_init__(self) -> None:
        if self.kl_weight != 0.0:
            raise ValueError(
                "kl_weight is unsupported: the supervised semantic path is deterministic"
            )
        if self.scheduler_monitor not in {
            "trace_to_tree",
            "reconstruction_composite",
            "loss",
        }:
            raise ValueError(
                "scheduler_monitor must be trace_to_tree, reconstruction_composite, or loss"
            )
        for name in (
            "activity_remap_probability",
            "tree_encoder_dropout",
            "trace_encoder_dropout",
            "petri_encoder_dropout",
            "decoder_dropout",
            "projection_dropout",
            "decoder_input_dropout",
            "scheduled_sampling_max",
            "ema_decay",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


def loss_weights_from_config(
    config: TrainConfig,
    epoch: int | None = None,
) -> LossWeights:
    if config.training_stage not in {"a", "b", "c", "d", "full"}:
        raise ValueError("training_stage must be one of: a, b, c, d, full")
    exact_weight = 0.0 if config.training_stage == "a" else config.exact_contrastive_weight
    within_weight = (
        0.0 if config.training_stage == "a" else config.within_modality_contrastive_weight
    )
    soft_weight = (
        config.soft_behavior_geometry_weight
        if config.training_stage in {"c", "d", "full"}
        else 0.0
    )
    variance_weight = 0.0 if config.training_stage == "a" else config.variance_weight
    covariance_weight = 0.0 if config.training_stage == "a" else config.covariance_weight
    exact_scale = _objective_ramp(
        epoch,
        start_epoch=config.exact_contrastive_start_epoch,
        ramp_epochs=config.exact_contrastive_ramp_epochs,
    )
    soft_scale = _objective_ramp(
        epoch,
        start_epoch=config.soft_geometry_start_epoch,
        ramp_epochs=config.soft_geometry_ramp_epochs,
    )
    return LossWeights(
        tree_reconstruction=config.tree_reconstruction_weight,
        trace_to_tree=config.trace_to_tree_weight,
        petri_to_tree=config.petri_to_tree_weight,
        exact_contrastive=exact_weight * exact_scale,
        within_modality_contrastive=within_weight * exact_scale,
        soft_behavior_geometry=soft_weight * soft_scale,
        variance=variance_weight * exact_scale,
        covariance=covariance_weight * exact_scale,
        latent_alignment=config.latent_alignment_weight * exact_scale,
        kl=0.0,
        label_smoothing=config.label_smoothing,
        contrastive_temperature=config.contrastive_temperature,
        behavior_temperature=config.behavior_temperature,
        latent_temperature=config.latent_temperature,
    )


def _objective_ramp(
    epoch: int | None,
    *,
    start_epoch: int,
    ramp_epochs: int,
) -> float:
    if epoch is None:
        return 1.0
    if epoch < start_epoch:
        return 0.0
    return min(1.0, (epoch - start_epoch + 1) / max(ramp_epochs, 1))


def loss_weights_from_checkpoint(
    checkpoint: dict[str, object], config: TrainConfig
) -> LossWeights:
    """Restore the exact serialized objective, falling back for legacy checkpoints."""

    values = asdict(loss_weights_from_config(config))
    stored = checkpoint.get("loss_weights")
    if isinstance(stored, dict):
        values.update({name: stored[name] for name in values if name in stored})
    return LossWeights(**values)


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
            moved[key] = value.to(device, non_blocking=True)
        elif isinstance(value, dict):
            moved[key] = {
                child_key: (
                    child_value
                    if key == "traces" and child_key == "lengths"
                    else child_value.to(device, non_blocking=True)
                )
                if isinstance(child_value, torch.Tensor) else child_value
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
    ema: ModelEMA | None = None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float | torch.Tensor] = {}
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
            batch.get("decoder_targets", tree_tokens),
            model.tree_tokenizer.pad_id,
            weights=weights,
            positive_mask=positive_mask,
            contrastive_candidate_mask=batch.get("contrastive_candidate_mask"),
            behavior_signatures=batch.get("behavior_signatures"),
            tokenizer=model.tree_tokenizer,
        )
        diagnostics_interval = (
            0 if train_config is None else train_config.gradient_diagnostics_interval
        )
        run_gradient_diagnostics = diagnostics_interval > 0 and (
            epoch is None or (epoch - 1) % diagnostics_interval == 0
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
        if (
            ema is not None
            and train_config is not None
            and epoch is not None
            and epoch >= train_config.ema_start_epoch
        ):
            ema.update(model)

        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + value.detach()
        batches += 1
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{float(losses['loss'].detach().cpu()):.4f}")
    return {
        name: (
            float(value.detach().cpu())
            if isinstance(value, torch.Tensor) and "gradient_" in name
            else float(value.detach().cpu()) / max(batches, 1)
            if isinstance(value, torch.Tensor)
            else value
            if "gradient_" in name
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
        exact_gradients = torch.autograd.grad(
            losses["exact_contrastive"],
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        soft_gradients = torch.autograd.grad(
            losses["soft_behavior_geometry"],
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        raw_reconstruction_gradients = torch.autograd.grad(
            losses[f"{name}_reconstruction" if name == "tree" else f"{name}_to_tree"],
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
        result[f"reconstruction_exact_gradient_cosine_{name}"] = _gradient_cosine(
            raw_reconstruction_gradients,
            exact_gradients,
        )
        result[f"reconstruction_soft_geometry_gradient_cosine_{name}"] = (
            _gradient_cosine(raw_reconstruction_gradients, soft_gradients)
        )
        result[f"exact_soft_geometry_gradient_cosine_{name}"] = _gradient_cosine(
            exact_gradients,
            soft_gradients,
        )
    return result


def _gradient_norm(gradients: tuple[torch.Tensor | None, ...]) -> float:
    squared = sum(
        float(gradient.detach().pow(2).sum().cpu())
        for gradient in gradients
        if gradient is not None
    )
    return squared**0.5


def _gradient_cosine(
    left: tuple[torch.Tensor | None, ...],
    right: tuple[torch.Tensor | None, ...],
) -> float:
    dot = 0.0
    left_squared = 0.0
    right_squared = 0.0
    for left_gradient, right_gradient in zip(left, right):
        if left_gradient is None or right_gradient is None:
            continue
        left_flat = left_gradient.detach().reshape(-1)
        right_flat = right_gradient.detach().reshape(-1)
        dot += float((left_flat * right_flat).sum().cpu())
        left_squared += float(left_flat.square().sum().cpu())
        right_squared += float(right_flat.square().sum().cpu())
    denominator = (left_squared * right_squared) ** 0.5
    return 0.0 if denominator <= 1e-24 else dot / denominator


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
    totals: dict[str, torch.Tensor] = {}
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
            batch.get("decoder_targets", tree_tokens),
            model.tree_tokenizer.pad_id,
            weights=weights,
            positive_mask=positive_mask,
            contrastive_candidate_mask=batch.get("contrastive_candidate_mask"),
            behavior_signatures=batch.get("behavior_signatures"),
            tokenizer=model.tree_tokenizer,
        )
        for name, value in losses.items():
            totals[name] = totals.get(name, torch.zeros_like(value.detach())) + value.detach()
        batches += 1
        if compute_discovery_metrics:
            dists = outputs["dists"]
            assert isinstance(dists, dict)
            trace_distribution = dists["trace"]
            maximum_length = min(512, max(2, tree_tokens.shape[1] * 2))
            source_activity_masks = batch.get("source_activity_masks")
            trace_allowed = (
                source_activity_masks.get("trace")
                if isinstance(source_activity_masks, dict)
                else None
            )
            decoded = model.tree_decoder.decode_greedy(
                trace_distribution,
                max_length=maximum_length,
                allowed_activity_mask=trace_allowed,
                avoid_duplicate_activity_labels=False,
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
                    shuffled_distribution,
                    max_length=maximum_length,
                    allowed_activity_mask=trace_allowed,
                    avoid_duplicate_activity_labels=False,
                )
                zero_decoded = model.tree_decoder.decode_greedy(
                    zero_distribution,
                    max_length=maximum_length,
                    allowed_activity_mask=trace_allowed,
                    avoid_duplicate_activity_labels=False,
                )
            samples = batch.get("samples")
            assert isinstance(samples, list)
            for row_index, (sample, target, prediction) in enumerate(
                zip(samples, tree_tokens, decoded)
            ):
                trace_targets = batch.get("decoder_targets")
                target_row = (
                    trace_targets["trace"][row_index]
                    if isinstance(trace_targets, dict)
                    else target
                )
                target_ids = _trim_token_ids(target_row.tolist(), model.tree_tokenizer)
                prediction_ids = _trim_token_ids(
                    prediction.detach().cpu().tolist(), model.tree_tokenizer
                )
                raw_edit = _token_edit_distance(target_ids, prediction_ids)
                row_allowed = (
                    None
                    if trace_allowed is None
                    else trace_allowed[row_index].detach().cpu().tolist()
                )
                normalized_target_ids = _source_normalized_token_ids(
                    target_ids,
                    model.tree_tokenizer,
                    row_allowed,
                    avoid_duplicates=False,
                ) or target_ids
                normalized_prediction_ids = _source_normalized_token_ids(
                    prediction_ids,
                    model.tree_tokenizer,
                    row_allowed,
                    avoid_duplicates=False,
                ) or prediction_ids
                normalized_edit = _token_edit_distance(
                    normalized_target_ids,
                    normalized_prediction_ids,
                )
                deployment_target_ids = _source_normalized_token_ids(
                    target_ids,
                    model.tree_tokenizer,
                    row_allowed,
                    avoid_duplicates=True,
                ) or target_ids
                deployment_prediction_ids = _source_normalized_token_ids(
                    prediction_ids,
                    model.tree_tokenizer,
                    row_allowed,
                    avoid_duplicates=True,
                ) or prediction_ids
                deployment_edit = _token_edit_distance(
                    deployment_target_ids,
                    deployment_prediction_ids,
                )
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
                        "raw_exact": prediction_ids == target_ids,
                        "raw_normalized_edit": raw_edit
                        / max(len(target_ids), len(prediction_ids), 1),
                        "exact": normalized_prediction_ids == normalized_target_ids,
                        "normalized_edit": normalized_edit
                        / max(
                            len(normalized_target_ids),
                            len(normalized_prediction_ids),
                            1,
                        ),
                        "deployment_exact": (
                            deployment_prediction_ids == deployment_target_ids
                        ),
                        "deployment_normalized_edit": deployment_edit
                        / max(
                            len(deployment_target_ids),
                            len(deployment_prediction_ids),
                            1,
                        ),
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
    metrics = {
        name: float(value.detach().cpu()) / max(batches, 1)
        for name, value in totals.items()
    }
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


def _source_normalized_token_ids(
    token_ids: list[int],
    tokenizer: TreeTokenizer,
    allowed_slots: list[bool] | None,
    *,
    avoid_duplicates: bool,
) -> list[int] | None:
    from proc_rosetta.pm4py_bridge import fold_process_tree
    from proc_rosetta.tree import sanitize_activity_labels

    try:
        tree = tokenizer.decode_tree(token_ids)
        allowed = (
            None
            if allowed_slots is None
            else {
                f"A{index}"
                for index, value in enumerate(allowed_slots)
                if value
            }
        )
        tree = sanitize_activity_labels(
            tree,
            allowed_labels=allowed,
            avoid_duplicates=avoid_duplicates,
        ).tree
        return tokenizer.encode_tree(fold_process_tree(tree), canonicalize=False)
    except (TypeError, ValueError):
        return None


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
    metrics["raw_token_exact"] = (
        sum(bool(row["raw_exact"]) for row in rows) / len(rows) if rows else 0.0
    )
    metrics["raw_token_edit"] = (
        float(np.mean([float(row["raw_normalized_edit"]) for row in rows]))
        if rows
        else 1.0
    )
    metrics["source_normalized_tree_exact"] = metrics["trace_canonical_exact"]
    metrics["source_normalized_tree_edit"] = metrics["trace_normalized_tree_edit"]
    metrics["deployment_duplicate_free_tree_exact"] = (
        sum(bool(row["deployment_exact"]) for row in rows) / len(rows)
        if rows
        else 0.0
    )
    metrics["deployment_duplicate_free_tree_edit"] = (
        float(np.mean([float(row["deployment_normalized_edit"]) for row in rows]))
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
        dimension_std = matrix.std(dim=0, unbiased=False)
        metrics[f"min_dimension_std_{name}"] = float(dimension_std.min())
        metrics[f"median_dimension_std_{name}"] = float(dimension_std.median())
        metrics[f"mean_dimension_std_{name}"] = float(dimension_std.mean())
        metrics[f"max_dimension_std_{name}"] = float(dimension_std.max())
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
                top = similarity[index].topk(min(5, similarity.shape[1])).indices
                candidate_ids = [exact_ids[int(position)] for position in top]
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
    # Retain a scalar discovery-quality proxy for schedulers and tabular logs.
    # Best-checkpoint selection itself is based on validation loss.
    behavior_spearman = metrics.get("behavior_distance_spearman", -1.0)
    metrics["checkpoint_selection_primary_exact"] = primary_exact
    metrics["checkpoint_selection_edit_score"] = 1.0 - metrics[
        "trace_normalized_tree_edit"
    ]
    metrics["checkpoint_selection_recall_at_1"] = metrics[
        "exact_behavior_recall_at_1"
    ]
    metrics["checkpoint_selection_spearman"] = behavior_spearman
    metrics["checkpoint_selection_score"] = (
        primary_exact
        + 1e-6 * metrics["checkpoint_selection_edit_score"]
        + 1e-9 * metrics["exact_behavior_recall_at_1"]
        + 1e-12 * ((behavior_spearman + 1.0) / 2.0)
    )
    return metrics


def checkpoint_selection_key(metrics: dict[str, float]) -> tuple[float, ...]:
    """Return the discovery-quality ordering recorded with checkpoints."""

    primary_exact = metrics.get(
        "checkpoint_selection_primary_exact",
        metrics.get(
            "ordinary_trace_canonical_exact",
            metrics.get("trace_canonical_exact", float("-inf")),
        ),
    )
    return (
        primary_exact,
        metrics.get(
            "checkpoint_selection_edit_score",
            1.0 - metrics.get("trace_normalized_tree_edit", float("inf")),
        ),
        metrics.get(
            "checkpoint_selection_recall_at_1",
            metrics.get("exact_behavior_recall_at_1", float("-inf")),
        ),
        metrics.get(
            "checkpoint_selection_spearman",
            metrics.get("behavior_distance_spearman", -1.0),
        ),
    )


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
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> DataLoader:
    dataset = SyntheticProcessDataset(samples, config=synthetic_config, seed=seed)
    collator = ProcessBatchCollator(
        tree_tokenizer,
        activity_tokenizer,
        config=batch_config,
        activity_remap_probability=activity_remap_probability,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        **_loader_worker_options(
            collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        ),
    )


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
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
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
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            **_loader_worker_options(
                collator,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            ),
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        **_loader_worker_options(
            collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        ),
    )


def _seed_collator_worker(worker_id: int, collator: ProcessBatchCollator) -> None:
    collator.rng.seed(torch.initial_seed() + worker_id)


def _loader_worker_options(
    collator: ProcessBatchCollator,
    *,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> dict[str, object]:
    from functools import partial

    workers = max(0, int(num_workers))
    options: dict[str, object] = {
        "num_workers": workers,
        "pin_memory": bool(pin_memory and torch.cuda.is_available()),
    }
    if workers:
        options.update(
            persistent_workers=bool(persistent_workers),
            prefetch_factor=max(1, int(prefetch_factor)),
            worker_init_fn=partial(_seed_collator_worker, collator=collator),
        )
    return options


class BehaviorFamilyBatchSampler(Sampler[list[int]]):
    """Batch strong-positive classes while retaining many exact behaviors.

    Strong positives require both exact behavior and partial-order identity.
    Grouping only by the broader equivalence family can place incompatible
    representations in a chunk and leave a row without a valid positive.
    """

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
        grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            exact_behavior_id = getattr(sample, "exact_behavior_id", None)
            partial_order_id = getattr(sample, "partial_order_id", None)
            if exact_behavior_id is not None and partial_order_id is not None:
                key = ("strong", str(exact_behavior_id), str(partial_order_id))
            else:
                key = ("legacy", str(getattr(sample, "equivalence_id", index)))
            grouped[key].append(index)
        self.groups = list(grouped.values())
        self.sample_costs = [
            (
                len(sample.tree.to_prefix_tokens())
                + 0.5 * max((len(trace) for trace in sample.traces), default=0)
                + 0.5 * sample.petri_graph.num_nodes
                if all(
                    hasattr(sample, name)
                    for name in ("tree", "traces", "petri_graph")
                )
                else 1.0
            )
            for sample in samples
        ]
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

        per_group_chunks = [self._positive_chunks(group) for group in groups]
        # Interleave chunk rounds so a 128-row batch with four views contains
        # 32 distinct behaviors before any family contributes a second chunk.
        chunks: list[list[int]] = []
        for round_index in range(max((len(value) for value in per_group_chunks), default=0)):
            round_chunks = [
                value[round_index]
                for value in per_group_chunks
                if round_index < len(value)
            ]
            round_chunks.sort(
                key=lambda chunk: max(self.sample_costs[index] for index in chunk),
                reverse=True,
            )
            chunks.extend(round_chunks)
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

    def _positive_chunks(self, group: list[int]) -> list[list[int]]:
        if len(group) <= 1:
            return [group]
        # Balance chunks so a trailing singleton is avoided whenever the class
        # has at least two views.  For three views and a target width of two, a
        # single three-view chunk is preferable to one positive pair plus an
        # orphan row.
        target_width = max(self.views_per_family, 2)
        chunk_count = min(
            (len(group) + target_width - 1) // target_width,
            len(group) // 2,
        )
        chunk_count = max(chunk_count, 1)
        base, extra = divmod(len(group), chunk_count)
        chunks: list[list[int]] = []
        start = 0
        for chunk_index in range(chunk_count):
            size = base + (1 if chunk_index < extra else 0)
            chunks.append(group[start : start + size])
            start += size
        return chunks


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
        tree_encoder_dropout=train_config.tree_encoder_dropout,
        trace_encoder_dropout=train_config.trace_encoder_dropout,
        petri_encoder_dropout=train_config.petri_encoder_dropout,
        decoder_dropout=train_config.decoder_dropout,
        projection_dropout=train_config.projection_dropout,
        memory_tokens=train_config.memory_tokens,
        decoder_layers=train_config.decoder_layers,
        tree_encoder_layers=train_config.tree_encoder_layers,
        trace_event_layers=train_config.trace_event_layers,
        trace_set_layers=train_config.trace_set_layers,
        petri_message_passing_steps=train_config.petri_message_passing_steps,
    ).to(device)


def build_optimizer(
    model: ProcRosettaModel,
    train_config: TrainConfig,
) -> torch.optim.AdamW:
    """Apply AdamW decay to matrix weights, never biases or normalization scales."""

    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".bias") or parameter.ndim < 2:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": train_config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=train_config.learning_rate,
    )


class ModelEMA:
    """Exponential moving average with resumable, temporary weight swapping."""

    def __init__(self, decay: float) -> None:
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        self.updates = 0

    @property
    def initialized(self) -> bool:
        return bool(self.shadow)

    @torch.no_grad()
    def update(self, model: ProcRosettaModel) -> None:
        current = model.state_dict()
        if not self.shadow:
            self.shadow = {
                name: value.detach().clone() for name, value in current.items()
            }
        else:
            for name, value in current.items():
                target = self.shadow[name]
                if torch.is_floating_point(target):
                    target.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
                else:
                    target.copy_(value.detach())
        self.updates += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "updates": self.updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        stored_decay = float(state.get("decay", self.decay))
        if stored_decay != self.decay:
            raise ValueError(
                f"EMA decay differs from checkpoint ({stored_decay} != {self.decay})"
            )
        shadow = state.get("shadow", {})
        if not isinstance(shadow, dict) or not all(
            isinstance(value, torch.Tensor) for value in shadow.values()
        ):
            raise ValueError("checkpoint contains an invalid EMA state")
        self.shadow = {
            str(name): value.detach().clone() for name, value in shadow.items()
        }
        self.updates = int(state.get("updates", 0))

    @contextmanager
    def average_parameters(self, model: ProcRosettaModel):
        if not self.initialized:
            yield
            return
        ordinary = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(ordinary, strict=True)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    train_config: TrainConfig,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Reduce the learning rate on the same validation objective used for stopping."""

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_config.lr_factor,
        patience=train_config.lr_patience,
        min_lr=train_config.min_lr,
        threshold=train_config.min_delta,
        threshold_mode="abs",
    )


def scheduler_monitor_value(
    metrics: dict[str, float],
    monitor: str,
) -> float:
    if monitor == "trace_to_tree":
        return metrics["trace_to_tree"]
    if monitor == "reconstruction_composite":
        return (
            0.5 * metrics["tree_reconstruction"]
            + 2.0 * metrics["trace_to_tree"]
            + 0.5 * metrics["petri_to_tree"]
        )
    if monitor == "loss":
        return metrics["loss"]
    raise ValueError(f"unsupported scheduler monitor: {monitor}")


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
        num_workers=train_config.loader_num_workers,
        pin_memory=train_config.loader_pin_memory,
        persistent_workers=train_config.loader_persistent_workers,
        prefetch_factor=train_config.loader_prefetch_factor,
    )
    optimizer = build_optimizer(model, train_config)
    ema = ModelEMA(train_config.ema_decay) if train_config.use_ema else None
    history = [
        train_epoch(
            model,
            dataloader,
            optimizer,
            device,
            weights=loss_weights_from_config(train_config, epoch=epoch),
            epoch=epoch,
            train_config=train_config,
            ema=ema,
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
    if int(metadata.get("version", 0)) < 5:
        raise ValueError(
            "data uses a legacy schema without semantic folding and per-modality "
            "decoder targets; recreate it with sample.py before training"
        )
    if not bool(metadata.get("exact_behavior_signatures_disjoint", False)):
        raise ValueError("data metadata does not certify signature-disjoint splits")
    synthetic_config = SyntheticConfig.from_dict(metadata.get("synthetic_config", {}))
    debug(
        "Training configuration: "
        f"epochs={train_config.epochs}, batch_size={train_config.batch_size}, "
        f"lr={train_config.learning_rate}, latent_dim={train_config.latent_dim}, "
        f"hidden_dim={train_config.hidden_dim}, "
        f"dropout(tree/trace/petri/decoder/projection)="
        f"{train_config.tree_encoder_dropout}/{train_config.trace_encoder_dropout}/"
        f"{train_config.petri_encoder_dropout}/{train_config.decoder_dropout}/"
        f"{train_config.projection_dropout}, "
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
    resume_policy_overrides: dict[str, dict[str, object]] = {}
    if resume:
        model, resume_checkpoint = load_checkpoint(checkpoint_path, device)
        resume_policy_overrides = validate_resume_configuration(
            checkpoint=resume_checkpoint,
            train_config=train_config,
            synthetic_config=synthetic_config,
        )
        if resume_policy_overrides:
            formatted_overrides = ", ".join(
                f"{name}: checkpoint={values['checkpoint']!r}, "
                f"requested={values['requested']!r}"
                for name, values in sorted(resume_policy_overrides.items())
            )
            debug(
                f"Applying resume runtime-policy overrides ({formatted_overrides})",
                enabled=show_progress,
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
        num_workers=train_config.loader_num_workers,
        pin_memory=train_config.loader_pin_memory,
        persistent_workers=train_config.loader_persistent_workers,
        prefetch_factor=train_config.loader_prefetch_factor,
    )
    debug("Loading validation split", enabled=show_progress)
    validation_loader = build_jsonl_dataloader(
        split_samples_path(data_dir, "validation"),
        model.tree_tokenizer,
        model.activity_tokenizer,
        batch_size=train_config.batch_size,
        shuffle=False,
        show_progress=show_progress,
        num_workers=train_config.loader_num_workers,
        pin_memory=train_config.loader_pin_memory,
        persistent_workers=train_config.loader_persistent_workers,
        prefetch_factor=train_config.loader_prefetch_factor,
    )
    stage_training_loader = (
        build_jsonl_dataloader(
            split_samples_path(data_dir, "training"),
            model.tree_tokenizer,
            model.activity_tokenizer,
            batch_size=train_config.batch_size,
            shuffle=False,
            show_progress=False,
            num_workers=train_config.loader_num_workers,
            pin_memory=train_config.loader_pin_memory,
            persistent_workers=train_config.loader_persistent_workers,
            prefetch_factor=train_config.loader_prefetch_factor,
        )
        if train_config.training_stage == "a"
        else None
    )
    debug_split("training", train_loader.dataset.samples, len(train_loader), enabled=show_progress)
    debug_split("validation", validation_loader.dataset.samples, len(validation_loader), enabled=show_progress)
    optimizer = build_optimizer(model, train_config)
    scheduler = build_lr_scheduler(optimizer, train_config)
    ema = ModelEMA(train_config.ema_decay) if train_config.use_ema else None
    if ema is not None and resume_checkpoint is not None:
        ema_state = resume_checkpoint.get("ema_state_dict")
        if isinstance(ema_state, dict):
            ema.load_state_dict(ema_state)
    evaluation_weights = loss_weights_from_config(train_config)
    history: list[dict[str, object]] = []
    best_validation_loss = float("inf")
    best_early_stopping_metric = float("inf")
    best_validation_score = float("-inf")
    best_validation_key = (float("-inf"),) * 4
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
        stored_best_key = resume_checkpoint.get("best_validation_key")
        if isinstance(stored_best_key, (list, tuple)) and len(stored_best_key) == 4:
            best_validation_key = tuple(float(value) for value in stored_best_key)
        elif history:
            validation_rows = [
                row.get("validation") for row in history if isinstance(row, dict)
            ]
            compatible_rows = [
                row
                for row in validation_rows
                if isinstance(row, dict)
                and (
                    "ordinary_trace_canonical_exact" in row
                    or "trace_canonical_exact" in row
                )
            ]
            if compatible_rows:
                best_validation_key = max(
                    checkpoint_selection_key(row) for row in compatible_rows
                )
        if history:
            epochs_without_improvement = int(
                history[-1].get("epochs_without_improvement", 0)
            )
            best_early_stopping_metric = min(
                float(
                    row.get(
                        "early_stopping_metric",
                        scheduler_monitor_value(
                            row["validation"], train_config.scheduler_monitor
                        ),
                    )
                )
                for row in history
                if isinstance(row.get("validation"), dict)
            )
        restore_training_state(
            checkpoint=resume_checkpoint,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            device=device,
            completed_epoch=completed_epoch,
            history=history,
            scheduler_monitor=train_config.scheduler_monitor,
            seed=train_config.seed,
            show_progress=show_progress,
        )
        debug(
            f"Resuming from {checkpoint_path} after epoch {completed_epoch}; "
            f"target epoch={train_config.epochs}",
            enabled=show_progress,
        )
    best_checkpoint_path = best_checkpoint_for(checkpoint_path)
    objective_checkpoint_paths = {
        name: best_checkpoint_for(checkpoint_path, name)
        for name in ("loss", "trace", "edit", "latent")
    }
    best_objectives = best_objectives_from_history(history)
    metrics_csv_path = Path(metrics_csv_path)
    write_metrics_csv(metrics_csv_path, history)
    debug(f"Per-epoch metrics CSV: {metrics_csv_path}", enabled=show_progress)
    for epoch in range(start_epoch, train_config.epochs + 1):
        epoch_start = perf_counter()
        debug(f"Starting epoch {epoch}/{train_config.epochs}", enabled=show_progress)
        epoch_weights = loss_weights_from_config(train_config, epoch=epoch)
        training_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            weights=epoch_weights,
            epoch=epoch,
            show_progress=show_progress,
            train_config=train_config,
            ema=ema,
        )
        debug(
            f"Epoch {epoch} training complete: {format_metrics(training_metrics)}",
            enabled=show_progress,
        )
        ordinary_validation_metrics = evaluate_epoch(
            model,
            validation_loader,
            device,
            weights=evaluation_weights,
            epoch=epoch,
            show_progress=show_progress,
            compute_discovery_metrics=True,
        )
        ema_validation_metrics: dict[str, float] | None = None
        validation_weights = "ordinary"
        if ema is not None and ema.initialized:
            with ema.average_parameters(model):
                ema_validation_metrics = evaluate_epoch(
                    model,
                    validation_loader,
                    device,
                    weights=evaluation_weights,
                    epoch=epoch,
                    show_progress=show_progress,
                    progress_desc=f"Epoch {epoch} EMA validation",
                    compute_discovery_metrics=True,
                )
            ordinary_monitor = scheduler_monitor_value(
                ordinary_validation_metrics,
                train_config.scheduler_monitor,
            )
            ema_monitor = scheduler_monitor_value(
                ema_validation_metrics,
                train_config.scheduler_monitor,
            )
            if ema_monitor < ordinary_monitor:
                validation_metrics = ema_validation_metrics
                validation_weights = "ema"
            else:
                validation_metrics = ordinary_validation_metrics
        else:
            validation_metrics = ordinary_validation_metrics
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
                weights=evaluation_weights,
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
        monitor_value = scheduler_monitor_value(
            validation_metrics,
            train_config.scheduler_monitor,
        )
        scheduler.step(monitor_value)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < current_lr:
            debug(
                f"Validation plateau detected; reducing learning rate {current_lr:.6g} -> {new_lr:.6g}",
                enabled=show_progress,
            )

        validation_key = checkpoint_selection_key(validation_metrics)
        ordinary_objectives = checkpoint_objective_values(
            ordinary_validation_metrics
        )
        ema_objectives = (
            None
            if ema_validation_metrics is None
            else checkpoint_objective_values(ema_validation_metrics)
        )
        objective_values: dict[str, float | tuple[float, ...]] = {}
        objective_candidate_weights: dict[str, str] = {}
        for name, ordinary_value in ordinary_objectives.items():
            if (
                ema_objectives is not None
                and objective_is_better(name, ema_objectives[name], ordinary_value)
            ):
                objective_values[name] = ema_objectives[name]
                objective_candidate_weights[name] = "ema"
            else:
                objective_values[name] = ordinary_value
                objective_candidate_weights[name] = "ordinary"
        objective_improvements = {
            name: objective_is_better(name, value, best_objectives[name]["value"])
            for name, value in objective_values.items()
        }
        for name, improved_objective in objective_improvements.items():
            if improved_objective:
                best_objectives[name] = {
                    "value": objective_values[name],
                    "epoch": epoch,
                    "weights": objective_candidate_weights[name],
                }
        improved = objective_improvements["loss"]
        if improved:
            best_validation_loss = float(objective_values["loss"])
        if validation_key > best_validation_key:
            best_validation_key = validation_key
            best_validation_score = validation_score
        early_stopping_improved = (
            monitor_value < best_early_stopping_metric - train_config.min_delta
        )
        if early_stopping_improved:
            best_early_stopping_metric = monitor_value
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        gap = validation_metrics["loss"] - training_metrics["loss"]
        debug(
            f"Epoch {epoch} validation complete: {format_metrics(validation_metrics)} "
            f"| gap={gap:+.4f} | discovery_selection={validation_score:.4f} "
            f"| best_loss={best_validation_loss:.4f} "
            f"| {train_config.scheduler_monitor}={monitor_value:.4f} "
            f"| patience={epochs_without_improvement}/{train_config.early_stopping_patience}",
            enabled=show_progress,
        )
        elapsed = perf_counter() - epoch_start
        row: dict[str, object] = {
            "epoch": epoch,
            "training": training_metrics,
            "validation": validation_metrics,
            "validation_ordinary": ordinary_validation_metrics,
            "validation_ema": ema_validation_metrics,
            "validation_weights": validation_weights,
            "objective_candidate_weights": objective_candidate_weights,
            "generalization_gap": metric_gaps(training_metrics, validation_metrics),
            "learning_rate": new_lr,
            "lr_scheduler_metric": monitor_value,
            "scheduler_monitor": train_config.scheduler_monitor,
            "early_stopping_metric": monitor_value,
            "best_early_stopping_metric": best_early_stopping_metric,
            "lr_reduced": new_lr < current_lr,
            "epoch_seconds": elapsed,
            "best_validation_loss": best_validation_loss,
            "best_validation_score": best_validation_score,
            "best_validation_key": list(best_validation_key),
            "best_objectives": {
                name: dict(details) for name, details in best_objectives.items()
            },
            "is_best": improved,
            "epochs_without_improvement": epochs_without_improvement,
            "stage_acceptance": acceptance,
            "loss_weights": asdict(epoch_weights),
        }
        if stage_metrics is not None and run_stage_gate:
            row["stage_metrics"] = stage_metrics
        if resume_policy_overrides and epoch == start_epoch:
            row["resume_policy_overrides"] = {
                name: dict(values)
                for name, values in resume_policy_overrides.items()
            }
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
            best_validation_key=best_validation_key,
            best_objectives=best_objectives,
            is_best=False,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            device=device,
            ema=ema,
        )
        for objective, objective_improved in objective_improvements.items():
            if not objective_improved:
                continue
            best_weight_context = (
                ema.average_parameters(model)
                if objective_candidate_weights[objective] == "ema" and ema is not None
                else nullcontext()
            )
            with best_weight_context:
                save_checkpoint(
                    checkpoint_path=objective_checkpoint_paths[objective],
                    model=model,
                    train_config=train_config,
                    synthetic_config=synthetic_config,
                    history=history,
                    epoch=epoch,
                    best_validation_loss=best_validation_loss,
                    best_validation_score=best_validation_score,
                    best_validation_key=best_validation_key,
                    best_objectives=best_objectives,
                    is_best=True,
                    checkpoint_objective=objective,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loader=train_loader,
                    device=device,
                    ema=ema,
                )
                if objective == "loss":
                    save_checkpoint(
                        checkpoint_path=best_checkpoint_path,
                        model=model,
                        train_config=train_config,
                        synthetic_config=synthetic_config,
                        history=history,
                        epoch=epoch,
                        best_validation_loss=best_validation_loss,
                        best_validation_score=best_validation_score,
                        best_validation_key=best_validation_key,
                        best_objectives=best_objectives,
                        is_best=True,
                        checkpoint_objective="loss",
                        optimizer=optimizer,
                        scheduler=scheduler,
                        train_loader=train_loader,
                        device=device,
                        ema=ema,
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
                f"Early stopping after {epoch} epochs; validation loss did not improve by at least "
                f"{train_config.min_delta:g} on {train_config.scheduler_monitor} for "
                f"{train_config.early_stopping_patience} epochs.",
                enabled=show_progress,
            )
            break
    if train_config.restore_best_weights and best_checkpoint_path.exists():
        restore_model_weights(model, best_checkpoint_path, device)
        debug(
            f"Restored best validation-loss weights from {best_checkpoint_path} before return.",
            enabled=show_progress,
        )
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
    family_count = len(
        {
            str(getattr(sample, "equivalence_id", index))
            for index, sample in enumerate(samples)
        }
    )
    debug(
        f"{split}: {stats['count']} rows, {family_count} unique behavior families, "
        f"{batch_count} batches, "
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


def best_checkpoint_for(
    checkpoint_path: str | Path,
    objective: str | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    suffix = "best" if objective is None else f"best_{objective}"
    return checkpoint_path.with_name(
        f"{checkpoint_path.stem}.{suffix}{checkpoint_path.suffix}"
    )


def checkpoint_objective_values(
    metrics: dict[str, float],
) -> dict[str, float | tuple[float, ...]]:
    return {
        "loss": metrics["loss"],
        "trace": metrics["trace_to_tree"],
        "edit": metrics.get("trace_normalized_tree_edit", float("inf")),
        "latent": (
            metrics.get("exact_behavior_recall_at_1", float("-inf")),
            metrics.get("behavior_distance_spearman", float("-inf")),
            metrics.get("effective_rank", float("-inf")),
        ),
    }


def objective_is_better(
    objective: str,
    value: float | tuple[float, ...],
    best: float | tuple[float, ...],
) -> bool:
    if objective in {"loss", "trace", "edit"}:
        return float(value) < float(best)
    return tuple(value) > tuple(best)


def best_objectives_from_history(
    history: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    best: dict[str, dict[str, object]] = {
        "loss": {"value": float("inf"), "epoch": 0},
        "trace": {"value": float("inf"), "epoch": 0},
        "edit": {"value": float("inf"), "epoch": 0},
        "latent": {"value": (float("-inf"),) * 3, "epoch": 0},
    }
    if history:
        stored = history[-1].get("best_objectives")
        if isinstance(stored, dict) and set(best).issubset(stored):
            restored: dict[str, dict[str, object]] = {}
            for objective in best:
                details = stored[objective]
                if not isinstance(details, dict) or "value" not in details:
                    break
                value = details["value"]
                if objective == "latent" and isinstance(value, list):
                    value = tuple(float(item) for item in value)
                restored[objective] = {**details, "value": value}
            if len(restored) == len(best):
                return restored
    for row in history:
        metrics = row.get("validation")
        if not isinstance(metrics, dict):
            continue
        values = checkpoint_objective_values(metrics)
        for objective, value in values.items():
            if objective_is_better(objective, value, best[objective]["value"]):
                best[objective] = {
                    "value": value,
                    "epoch": int(row.get("epoch", 0)),
                }
    return best


def restore_model_weights(
    model: ProcRosettaModel,
    checkpoint_path: str | Path,
    device: torch.device,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)


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
    "checkpoint_selection_primary_exact",
    "checkpoint_selection_edit_score",
    "checkpoint_selection_recall_at_1",
    "checkpoint_selection_spearman",
    "checkpoint_selection_score",
    *(
        f"{kind}_{modality}"
        for modality in ("tree", "trace", "petri")
        for kind in (
            "reconstruction_gradient_norm",
            "metric_gradient_norm",
            "metric_to_reconstruction_gradient_ratio",
            "reconstruction_exact_gradient_cosine",
            "reconstruction_soft_geometry_gradient_cosine",
            "exact_soft_geometry_gradient_cosine",
        )
    ),
)


def metrics_csv_columns() -> list[str]:
    columns = [
        "epoch",
        "learning_rate",
        "lr_scheduler_metric",
        "lr_reduced",
        "epoch_seconds",
        "best_validation_loss",
        "best_validation_score",
        "scheduler_monitor",
        "early_stopping_metric",
        "best_early_stopping_metric",
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
        "lr_scheduler_metric": row.get("lr_scheduler_metric", ""),
        "lr_reduced": row.get("lr_reduced", ""),
        "epoch_seconds": row["epoch_seconds"],
        "best_validation_loss": row["best_validation_loss"],
        "best_validation_score": row.get("best_validation_score", ""),
        "scheduler_monitor": row.get("scheduler_monitor", ""),
        "early_stopping_metric": row.get("early_stopping_metric", ""),
        "best_early_stopping_metric": row.get("best_early_stopping_metric", ""),
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
    best_validation_key: tuple[float, ...] | None = None,
    best_objectives: dict[str, dict[str, object]] | None = None,
    is_best: bool = False,
    checkpoint_objective: str | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    train_loader: DataLoader | None = None,
    device: torch.device | None = None,
    ema: ModelEMA | None = None,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_FORMAT_VERSION,
        "model_architecture": MODEL_ARCHITECTURE_VERSION,
        "data_schema_version": 5,
        "tree_normalization_version": TREE_NORMALIZATION_VERSION,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "train_config": asdict(train_config),
        "synthetic_config": synthetic_config.to_dict(),
        "history": history,
        "best_validation_loss": best_validation_loss,
        "best_validation_score": best_validation_score,
        "best_validation_key": (
            None if best_validation_key is None else list(best_validation_key)
        ),
        "best_objectives": best_objectives,
        "loss_weights": asdict(loss_weights_from_config(train_config)),
        "semantic_latent_mode": train_config.semantic_latent_mode,
        "semantic_latent_stochastic": train_config.semantic_latent_mode != "deterministic",
        "is_best": is_best,
        "checkpoint_objective": checkpoint_objective,
        "ema_state_dict": (
            None if ema is None or not ema.initialized else ema.state_dict()
        ),
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
) -> dict[str, dict[str, object]]:
    checkpoint_train_config = train_config_from_checkpoint(checkpoint, train_config.device)
    checkpoint_values = asdict(checkpoint_train_config)
    requested_values = asdict(train_config)
    policy_overrides = {
        name: {
            "checkpoint": checkpoint_values[name],
            "requested": requested_values[name],
        }
        for name in RESUME_POLICY_OVERRIDE_FIELDS
        if checkpoint_values[name] != requested_values[name]
    }
    differences = {
        name: (checkpoint_values[name], requested_values[name])
        for name in checkpoint_values
        if name
        not in {
            "epochs",
            "device",
            "restore_best_weights",
            *RESUME_POLICY_OVERRIDE_FIELDS,
        }
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
    return policy_overrides


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
    scheduler_monitor: str = "loss",
) -> None:
    for row in history:
        validation = row.get("validation")
        if isinstance(validation, dict):
            scheduler.step(
                float(
                    row.get(
                        "lr_scheduler_metric",
                        scheduler_monitor_value(validation, scheduler_monitor),
                    )
                )
            )


def restore_training_state(
    *,
    checkpoint: dict[str, object],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    train_loader: DataLoader,
    device: torch.device,
    completed_epoch: int,
    history: list[dict[str, object]],
    scheduler_monitor: str,
    seed: int,
    show_progress: bool,
) -> None:
    optimizer_state = checkpoint.get("optimizer_state_dict")
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if (optimizer_state is None) != (scheduler_state is None):
        raise ValueError("checkpoint contains incomplete optimizer/scheduler resume state")

    if optimizer_state is None:
        replay_scheduler_history(scheduler, history, scheduler_monitor)
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
    if isinstance(scheduler_state, dict) and scheduler_state.get("mode") == scheduler.mode:
        scheduler.load_state_dict(scheduler_state)
        scheduler_restored = True
    else:
        # v5 checkpoints created before validation-loss scheduling stored a
        # mode="max" discovery-score scheduler. Replaying loss history avoids
        # silently restoring that conflicting objective while preserving the
        # optimizer, RNG, and loader continuation state.
        replay_scheduler_history(scheduler, history, scheduler_monitor)
        scheduler_restored = False
    rng_state = checkpoint.get("rng_state")
    loader_state = checkpoint.get("training_loader_state")
    rng_restored = isinstance(rng_state, dict) and restore_rng_state(rng_state, device)
    loader_restored = isinstance(loader_state, dict) and restore_training_loader_state(
        train_loader, loader_state, completed_epoch
    )
    if rng_restored and loader_restored and scheduler_restored:
        debug(
            "Restored optimizer, scheduler, RNG, and data-loader state.",
            enabled=show_progress,
        )
    else:
        details: list[str] = []
        if not scheduler_restored:
            details.append("replayed validation-loss scheduler history")
        if not (rng_restored and loader_restored):
            details.append("exact RNG/data-loader continuation unavailable for this device")
        debug(f"Restored optimizer; {'; '.join(details)}.", enabled=show_progress)


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str | None = None,
) -> tuple[ProcRosettaModel, dict[str, object]]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    torch_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device)
    checkpoint_architecture = checkpoint.get("model_architecture")
    if (
        checkpoint_architecture is not None
        and checkpoint_architecture != MODEL_ARCHITECTURE_VERSION
    ):
        raise RuntimeError(
            f"checkpoint architecture {checkpoint_architecture!r} is incompatible with "
            f"{MODEL_ARCHITECTURE_VERSION!r}; retrain it instead of changing metadata"
        )
    train_config = train_config_from_checkpoint(checkpoint, torch_device)
    synthetic_config = SyntheticConfig.from_dict(checkpoint["synthetic_config"])
    model = build_model(train_config, synthetic_config, torch_device)
    try:
        incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            "checkpoint tensors are incompatible with the current v6 architecture; "
            "checkpoint migration is unsafe, so retrain from schema-v5 data"
        ) from exc
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
