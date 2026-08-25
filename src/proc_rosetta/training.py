from __future__ import annotations

import csv
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from dataclasses import dataclass, replace
from pathlib import Path
import random
import math
import shutil
import sys
from time import perf_counter

import torch
import numpy as np
from torch.nn import functional as F
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
from proc_rosetta.behavior import behavioral_distance
from proc_rosetta.losses import LossWeights, SemanticMemoryBank, multimodal_tree_loss
from proc_rosetta.models import LatentDistribution, ProcRosettaModel
from proc_rosetta.pm4py_bridge import (
    TREE_NORMALIZATION_VERSION,
    simulate_traces,
    tree_to_petri_net,
)
from proc_rosetta.synthetic import CURRICULUM_LEVELS, SyntheticConfig
from proc_rosetta.tokenizers import ActivityTokenizer, TreeTokenizer


CHECKPOINT_FORMAT_VERSION = 7
MODEL_ARCHITECTURE_VERSION = "proc-rosetta-latent-transformer-v6"
RESUME_POLICY_OVERRIDE_FIELDS = frozenset(
    {
        "scheduled_sampling_max",
        "scheduled_sampling_start_epoch",
        "scheduled_sampling_ramp_epochs",
    }
)

DEFAULT_CURRICULUM_STAGES = (
    {
        "name": "simple",
        "start_fraction": 0.00,
        "end_fraction": 0.15,
        "weights": {"simple": 1.0},
    },
    {
        "name": "medium",
        "start_fraction": 0.15,
        "end_fraction": 0.40,
        "weights": {"simple": 0.25, "medium": 0.75},
    },
    {
        "name": "complex",
        "start_fraction": 0.40,
        "end_fraction": 1.00,
        "weights": {"simple": 0.10, "medium": 0.20, "complex": 0.70},
    },
)


def curriculum_stage_state(name: str) -> dict[str, object]:
    for stage in DEFAULT_CURRICULUM_STAGES:
        if stage["name"] == name:
            return {
                "name": name,
                "weights": {
                    level: float(dict(stage["weights"]).get(level, 0.0))
                    for level in CURRICULUM_LEVELS
                },
            }
    raise ValueError(f"unknown structural curriculum stage {name!r}")


def competence_curriculum_state(
    current_stage: str | None,
    metrics_by_curriculum: dict[str, dict[str, object]],
    stage_baseline: dict[str, object] | None,
    *,
    epochs_in_stage: int,
    minimum_epochs: int,
    min_delta: float,
    simple_best_score: float = float("-inf"),
    regression_tolerance: float = 0.02,
) -> tuple[dict[str, object], dict[str, object]]:
    """Advance only after baseline-relative competence and regression gates."""

    stage = current_stage or "simple"
    report: dict[str, object] = {
        "current_stage": stage,
        "eligible": False,
        "advanced": False,
        "checks": {},
    }
    if stage == "complex" or stage not in metrics_by_curriculum:
        return curriculum_stage_state(stage), report
    current = balanced_validation_components(metrics_by_curriculum[stage])
    baseline = (
        balanced_validation_components(stage_baseline)
        if stage_baseline is not None
        else current
    )
    checks = {
        "minimum_stage_duration": epochs_in_stage >= minimum_epochs,
        "hard_decode_gates": bool(current["all_hard_gates_pass"]),
        "decode_improved_from_stage_baseline": (
            float(current["decode_score"])
            >= float(baseline["decode_score"]) + min_delta
        ),
        "retrieval_improved_from_stage_baseline": (
            float(current["retrieval_score"])
            >= float(baseline["retrieval_score"]) + min_delta
        ),
    }
    if stage == "medium" and math.isfinite(simple_best_score):
        simple_current = balanced_validation_components(
            metrics_by_curriculum["simple"]
        )
        checks["simple_regression_within_tolerance"] = (
            float(simple_current["balanced_score"])
            >= simple_best_score - regression_tolerance
        )
    report["checks"] = checks
    report["eligible"] = all(checks.values())
    if bool(report["eligible"]):
        next_stage = "medium" if stage == "simple" else "complex"
        report["advanced"] = True
        report["next_stage"] = next_stage
        return curriculum_stage_state(next_stage), report
    return curriculum_stage_state(stage), report


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    group: str = "diagnostic"

    def __post_init__(self) -> None:
        if self.direction not in {"min", "max"}:
            raise ValueError("metric direction must be 'min' or 'max'")


def _metric_specs() -> tuple[MetricSpec, ...]:
    return (
        MetricSpec("loss", "min", 0.0, None, "loss"),
        MetricSpec("tree_reconstruction", "min", 0.0, None, "loss"),
        MetricSpec("trace_to_tree", "min", 0.0, None, "loss"),
        MetricSpec("petri_to_tree", "min", 0.0, None, "loss"),
        MetricSpec("fused_to_tree", "min", 0.0, None, "loss"),
        MetricSpec("mean_behavior_l1", "min", 0.0, 2.0, "decode"),
        MetricSpec("behavior_l1", "min", 0.0, 2.0, "decode"),
        MetricSpec("nearest_neighbor_behavior_l1", "min", 0.0, 2.0, "geometry"),
        MetricSpec("mean_behavior_l1_at_nearest_neighbor", "min", 0.0, 2.0, "geometry"),
        MetricSpec("mean_rank", "min", 1.0, None, "retrieval"),
        MetricSpec("normalized_token_edit_distance", "min", 0.0, 1.0, "decode"),
        MetricSpec("mean_normalized_token_edit_distance", "min", 0.0, 1.0, "decode"),
        MetricSpec("trace_normalized_tree_edit", "min", 0.0, 1.0, "decode"),
        MetricSpec("deployment_duplicate_free_tree_edit", "min", 0.0, 1.0, "decode"),
        MetricSpec("terminated_rate", "max", 0.0, 1.0, "decode"),
        MetricSpec("valid_tree_rate", "max", 0.0, 1.0, "decode"),
        MetricSpec("exact_tree_match_rate", "max", 0.0, 1.0, "decode"),
        MetricSpec("petri_conversion_rate", "max", 0.0, 1.0, "decode"),
        MetricSpec("behavior_eval_success_rate", "max", 0.0, 1.0, "decode"),
        MetricSpec("trace_canonical_exact", "max", 0.0, 1.0, "decode"),
        MetricSpec("deployment_duplicate_free_tree_exact", "max", 0.0, 1.0, "decode"),
        MetricSpec("top1_accuracy", "max", 0.0, 1.0, "retrieval"),
        MetricSpec("mrr", "max", 0.0, 1.0, "retrieval"),
        MetricSpec("recall_at_5", "max", 0.0, 1.0, "retrieval"),
        MetricSpec("partial_order_recall_at_1", "max", 0.0, 1.0, "retrieval"),
        MetricSpec("partial_order_recall_at_5", "max", 0.0, 1.0, "retrieval"),
        MetricSpec("analogy_neighborhood_spearman", "max", -1.0, 1.0, "retrieval"),
        MetricSpec("exact_behavior_recall_at_1", "max", 0.0, 1.0, "retrieval"),
        MetricSpec("within_family_cosine", "max", -1.0, 1.0, "equivalence"),
        MetricSpec("between_family_cosine", "min", -1.0, 1.0, "equivalence"),
        MetricSpec("equivalence_margin", "max", -2.0, 2.0, "equivalence"),
        MetricSpec("behavior_id_retrieval_top1", "max", 0.0, 1.0, "equivalence"),
        MetricSpec("log_resampling_consistency", "max", -1.0, 1.0, "equivalence"),
        MetricSpec("behavior_distance_spearman", "max", -1.0, 1.0, "geometry"),
        MetricSpec("spearman_embedding_distance_vs_behavior_l1", "max", -1.0, 1.0, "geometry"),
        MetricSpec("fitness", "max", 0.0, 1.0, "discovery"),
        MetricSpec("precision", "max", 0.0, 1.0, "discovery"),
        MetricSpec("f1", "max", 0.0, 1.0, "discovery"),
        MetricSpec("mean_fitness", "max", 0.0, 1.0, "discovery"),
        MetricSpec("mean_precision", "max", 0.0, 1.0, "discovery"),
        MetricSpec("mean_f1", "max", 0.0, 1.0, "discovery"),
        MetricSpec("checkpoint_selection_score", "max", 0.0, 1.0, "selection"),
        MetricSpec("checkpoint_selection_decode_score", "max", 0.0, 1.0, "decode"),
        MetricSpec("checkpoint_selection_retrieval_score", "max", 0.0, 1.0, "retrieval"),
        MetricSpec("checkpoint_selection_equivalence_score", "max", 0.0, 1.0, "equivalence"),
        MetricSpec("checkpoint_selection_geometry_score", "max", 0.0, 1.0, "geometry"),
        MetricSpec("checkpoint_selection_discovery_score", "max", 0.0, 1.0, "discovery"),
        MetricSpec("checkpoint_selection_hard_gates_pass", "max", 0.0, 1.0, "selection"),
        MetricSpec("checkpoint_selection_fused_geometry_advantage", "max", None, None, "geometry"),
    )


METRIC_REGISTRY = {spec.name: spec for spec in _metric_specs()}


def metric_spec(name: str) -> MetricSpec:
    """Resolve a registered metric, including a dotted evaluator path."""

    if name in METRIC_REGISTRY:
        return METRIC_REGISTRY[name]
    leaf = name.rsplit(".", 1)[-1]
    if leaf in METRIC_REGISTRY:
        return METRIC_REGISTRY[leaf]
    raise KeyError(f"metric {name!r} is not registered")


def normalized_metric_score(name: str, value: float) -> float:
    spec = metric_spec(name)
    if spec.lower_bound is None or spec.upper_bound is None:
        raise ValueError(f"metric {name!r} has no finite normalization range")
    span = max(spec.upper_bound - spec.lower_bound, 1e-12)
    normalized = min(1.0, max(0.0, (float(value) - spec.lower_bound) / span))
    return normalized if spec.direction == "max" else 1.0 - normalized


def structural_curriculum_for_epoch(
    epoch: int,
    total_epochs: int,
    *,
    minimum_stage: str | None = None,
) -> dict[str, object]:
    if epoch < 1 or total_epochs < 1 or epoch > total_epochs:
        raise ValueError("epoch must be within 1..total_epochs")
    fraction = (epoch - 1) / total_epochs
    selected = DEFAULT_CURRICULUM_STAGES[-1]
    for stage in DEFAULT_CURRICULUM_STAGES:
        if float(stage["start_fraction"]) <= fraction < float(stage["end_fraction"]):
            selected = stage
            break
    if minimum_stage is not None:
        stage_order = {
            str(stage["name"]): index
            for index, stage in enumerate(DEFAULT_CURRICULUM_STAGES)
        }
        if minimum_stage not in stage_order:
            raise ValueError(f"unknown minimum structural stage {minimum_stage!r}")
        if stage_order[str(selected["name"])] < stage_order[minimum_stage]:
            selected = DEFAULT_CURRICULUM_STAGES[stage_order[minimum_stage]]
    return {
        "name": str(selected["name"]),
        "weights": {
            level: float(dict(selected["weights"]).get(level, 0.0))
            for level in CURRICULUM_LEVELS
        },
    }


def curriculum_batch_plan(
    weights: dict[str, float],
    batch_count: int,
    *,
    seed: int,
    epoch: int,
) -> list[str]:
    if batch_count < 1:
        return []
    total = sum(max(0.0, float(weights.get(level, 0.0))) for level in CURRICULUM_LEVELS)
    if total <= 0.0:
        raise ValueError("curriculum sampling weights must contain a positive value")
    raw = {
        level: batch_count * max(0.0, float(weights.get(level, 0.0))) / total
        for level in CURRICULUM_LEVELS
    }
    counts = {level: int(value) for level, value in raw.items()}
    leftovers = batch_count - sum(counts.values())
    order = sorted(CURRICULUM_LEVELS, key=lambda level: (-(raw[level] % 1), level))
    for level in order[:leftovers]:
        counts[level] += 1
    plan = [level for level in CURRICULUM_LEVELS for _ in range(counts[level])]
    random.Random(seed + epoch).shuffle(plan)
    return plan


def deficit_adjusted_curriculum_weights(
    base_weights: dict[str, float],
    metrics_by_curriculum: dict[str, dict[str, object]],
) -> dict[str, float]:
    active = [level for level, weight in base_weights.items() if weight > 0.0]
    if not active or not all(level in metrics_by_curriculum for level in active):
        return dict(base_weights)
    scores = {
        level: float(
            balanced_validation_components(metrics_by_curriculum[level])[
                "balanced_score"
            ]
        )
        for level in active
    }
    best = max(scores.values())
    adjusted = {
        level: float(base_weights[level]) * (1.0 + max(0.0, best - scores[level]))
        for level in active
    }
    total = sum(adjusted.values())
    return {
        level: (adjusted.get(level, 0.0) / total if total else 0.0)
        for level in CURRICULUM_LEVELS
    }


def validation_deficit_loss_weights(
    weights: LossWeights,
    validation_metrics: dict[str, object] | None,
) -> LossWeights:
    if not validation_metrics:
        return weights
    raw = validation_metrics.get(
        "diagnostic_unbounded_decode_quality",
        validation_metrics.get("decode_quality"),
    )
    if not isinstance(raw, dict) or not isinstance(raw.get("methods"), dict):
        return weights
    methods = raw["methods"]
    mapping = {
        "tree_reconstruction": "proc_rosetta_tree_mu",
        "trace_to_tree": "proc_rosetta_trace_mu",
        "petri_to_tree": "proc_rosetta_petri_mu",
        "fused_to_tree": "proc_rosetta_fused_mu",
    }
    scores: dict[str, float] = {}
    for field, method_name in mapping.items():
        values = methods.get(method_name)
        if isinstance(values, dict):
            scores[field] = _decode_method_score(values)[0]
    if len(scores) != len(mapping):
        return weights
    best = max(scores.values())
    factors = {
        field: min(2.0, 1.0 + max(0.0, best - score))
        for field, score in scores.items()
    }
    return replace(
        weights,
        tree_reconstruction=weights.tree_reconstruction * factors["tree_reconstruction"],
        trace_to_tree=weights.trace_to_tree * factors["trace_to_tree"],
        petri_to_tree=weights.petri_to_tree * factors["petri_to_tree"],
        fused_to_tree=weights.fused_to_tree * factors["fused_to_tree"],
    )


class CurriculumMixtureLoader:
    """Deterministically interleave whole homogeneous batches across loaders."""

    def __init__(
        self,
        loaders: dict[str, DataLoader],
        weights: dict[str, float],
        *,
        seed: int,
        epoch: int,
    ) -> None:
        self.loaders = loaders
        active = [level for level, weight in weights.items() if weight > 0.0]
        self.batch_count = max((len(loaders[level]) for level in active), default=0)
        self.plan = curriculum_batch_plan(
            weights,
            self.batch_count,
            seed=seed,
            epoch=epoch,
        )

    def __len__(self) -> int:
        return len(self.plan)

    def __iter__(self):
        iterators = {level: iter(loader) for level, loader in self.loaders.items()}
        for level in self.plan:
            try:
                yield next(iterators[level])
            except StopIteration:
                iterators[level] = iter(self.loaders[level])
                yield next(iterators[level])


@dataclass(frozen=True)
class TrainConfig:
    samples: int = 128
    epochs: int = 100
    batch_size: int = 128
    simple_batch_size: int | None = None
    medium_batch_size: int | None = None
    complex_batch_size: int | None = None
    min_complex_stage_epochs: int = 5
    min_curriculum_stage_epochs: int = 5
    curriculum_regression_tolerance: float = 0.02
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
    scheduled_sampling_max: float = 0.20
    scheduled_sampling_start_epoch: int = 15
    scheduled_sampling_ramp_epochs: int = 20
    gradient_clip_norm: float = 5.0
    tree_reconstruction_weight: float = 1.0
    trace_to_tree_weight: float = 1.0
    petri_to_tree_weight: float = 1.0
    fused_to_tree_weight: float = 1.0
    fused_subset_to_tree_weight: float = 0.25
    deployment_to_tree_weight: float = 0.25
    modality_subset_fusion_probability: float = 0.5
    deployment_policy_probability: float = 0.25
    exact_contrastive_weight: float = 0.30
    within_modality_contrastive_weight: float = 0.25
    semantic_exact_contrastive_weight: float = 0.15
    semantic_memory_contrastive_weight: float = 0.10
    hierarchical_metric_weight: float = 0.15
    observation_view_consistency_weight: float = 0.15
    semantic_memory_bank_size: int = 4096
    soft_behavior_geometry_weight: float = 0.25
    observed_behavior_regression_weight: float = 0.20
    observed_behavior_ranking_weight: float = 0.20
    beam_minimum_risk_weight: float = 0.05
    eos_calibration_weight: float = 0.05
    generated_length_weight: float = 0.05
    unresolved_open_slots_weight: float = 0.05
    completion_feasibility_weight: float = 0.05
    beam_risk_start_epoch: int = 15
    beam_risk_batch_probability: float = 0.10
    beam_risk_size: int = 5
    beam_risk_max_decode_length: int = 128
    variance_weight: float = 0.1
    covariance_weight: float = 0.01
    kl_weight: float = 0.0
    latent_alignment_weight: float = 0.075
    tree_complexity_weight: float = 0.0
    duplicate_activity_weight: float = 0.0
    contrastive_temperature: float = 0.3
    behavior_temperature: float = 0.2
    latent_temperature: float = 0.2
    exact_contrastive_start_epoch: int = 3
    exact_contrastive_ramp_epochs: int = 4
    soft_geometry_start_epoch: int = 5
    soft_geometry_ramp_epochs: int = 6
    structure_regularization_start_epoch: int = 5
    structure_regularization_ramp_epochs: int = 5
    scheduler_monitor: str = "balanced"
    restore_best_weights: bool = True
    use_ema: bool = True
    ema_start_epoch: int = 3
    ema_decay: float = 0.995
    training_stage: str = "full"
    stage_gate_interval: int = 5
    gradient_diagnostics_interval: int = 0
    use_pcgrad: bool = False
    validation_audit_enabled: bool = True
    validation_decode_interval: int = 2
    validation_full_interval: int = 10
    validation_decode_family_count: int = 64
    validation_discovery_family_count: int = 32
    validation_beam_size: int = 5
    validation_max_decode_length: int = 512
    loader_num_workers: int = 0
    loader_pin_memory: bool = True
    loader_persistent_workers: bool = True
    loader_prefetch_factor: int = 2

    def __post_init__(self) -> None:
        if self.kl_weight != 0.0:
            raise ValueError(
                "kl_weight is unsupported: the supervised semantic path is deterministic"
            )
        if self.scheduler_monitor != "balanced":
            raise ValueError("scheduler_monitor must be balanced")
        for name in ("tree_complexity_weight", "duplicate_activity_weight"):
            if not getattr(self, name) >= 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.min_curriculum_stage_epochs < 1:
            raise ValueError("min_curriculum_stage_epochs must be positive")
        if self.curriculum_regression_tolerance < 0.0:
            raise ValueError("curriculum_regression_tolerance must be nonnegative")
        for name in (
            "activity_remap_probability",
            "tree_encoder_dropout",
            "trace_encoder_dropout",
            "petri_encoder_dropout",
            "decoder_dropout",
            "projection_dropout",
            "decoder_input_dropout",
            "scheduled_sampling_max",
            "modality_subset_fusion_probability",
            "deployment_policy_probability",
            "beam_risk_batch_probability",
            "ema_decay",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "validation_decode_interval",
            "validation_full_interval",
            "validation_decode_family_count",
            "validation_discovery_family_count",
            "validation_beam_size",
            "validation_max_decode_length",
            "semantic_memory_bank_size",
            "beam_risk_start_epoch",
            "beam_risk_size",
            "beam_risk_max_decode_length",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")


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
    geometry_stage_scale = (
        1.0 if config.training_stage in {"c", "d", "full"} else 0.0
    )
    variance_weight = 0.0 if config.training_stage == "a" else config.variance_weight
    covariance_weight = 0.0 if config.training_stage == "a" else config.covariance_weight
    semantic_scale = 0.0 if config.training_stage == "a" else 1.0
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
    structure_scale = _objective_ramp(
        epoch,
        start_epoch=config.structure_regularization_start_epoch,
        ramp_epochs=config.structure_regularization_ramp_epochs,
    )
    return LossWeights(
        tree_reconstruction=config.tree_reconstruction_weight,
        trace_to_tree=config.trace_to_tree_weight,
        petri_to_tree=config.petri_to_tree_weight,
        fused_to_tree=config.fused_to_tree_weight,
        fused_subset_to_tree=config.fused_subset_to_tree_weight,
        deployment_to_tree=config.deployment_to_tree_weight,
        exact_contrastive=exact_weight * exact_scale,
        within_modality_contrastive=within_weight * exact_scale,
        semantic_exact_contrastive=(
            config.semantic_exact_contrastive_weight * exact_scale * semantic_scale
        ),
        semantic_memory_contrastive=(
            config.semantic_memory_contrastive_weight * exact_scale * semantic_scale
        ),
        hierarchical_metric=(
            config.hierarchical_metric_weight * exact_scale * semantic_scale
        ),
        observation_view_consistency=(
            config.observation_view_consistency_weight
            * exact_scale
            * semantic_scale
        ),
        soft_behavior_geometry=soft_weight * soft_scale,
        observed_behavior_regression=(
            config.observed_behavior_regression_weight
            * soft_scale
            * geometry_stage_scale
        ),
        observed_behavior_ranking=(
            config.observed_behavior_ranking_weight
            * soft_scale
            * geometry_stage_scale
        ),
        beam_minimum_risk=(
            config.beam_minimum_risk_weight
            * _objective_ramp(
                epoch,
                start_epoch=config.beam_risk_start_epoch,
                ramp_epochs=5,
            )
        ),
        eos_calibration=config.eos_calibration_weight,
        generated_length=config.generated_length_weight,
        unresolved_open_slots=config.unresolved_open_slots_weight,
        completion_feasibility=config.completion_feasibility_weight,
        variance=variance_weight * exact_scale,
        covariance=covariance_weight * exact_scale,
        latent_alignment=config.latent_alignment_weight * exact_scale,
        kl=0.0,
        tree_complexity=config.tree_complexity_weight * structure_scale,
        duplicate_activity=config.duplicate_activity_weight * structure_scale,
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
    if int(checkpoint.get("version", 0)) < 7:
        for name in (
            "fused_to_tree",
            "fused_subset_to_tree",
            "deployment_to_tree",
            "semantic_exact_contrastive",
            "semantic_memory_contrastive",
            "hierarchical_metric",
            "observed_behavior_regression",
            "observed_behavior_ranking",
            "beam_minimum_risk",
            "eos_calibration",
            "generated_length",
            "unresolved_open_slots",
            "completion_feasibility",
            "observation_view_consistency",
        ):
            values[name] = 0.0
    return LossWeights(**values)


def scheduled_sampling_probability(config: TrainConfig, epoch: int | None) -> float:
    if epoch is None or epoch < config.scheduled_sampling_start_epoch:
        return 0.0
    progress = (epoch - config.scheduled_sampling_start_epoch + 1) / max(
        config.scheduled_sampling_ramp_epochs, 1
    )
    return config.scheduled_sampling_max * min(max(progress, 0.0), 1.0)


def adaptive_scheduled_sampling_probability(
    config: TrainConfig,
    epoch: int | None,
    validation_metrics: dict[str, object] | None,
) -> float:
    """Increase exposure training when raw decode lags teacher-forced quality."""

    base = scheduled_sampling_probability(config, epoch)
    if base <= 0.0 or not validation_metrics:
        return base
    raw_decode = validation_metrics.get(
        "diagnostic_unbounded_decode_quality",
        validation_metrics.get("decode_quality"),
    )
    if not isinstance(raw_decode, dict) or not isinstance(
        raw_decode.get("methods"), dict
    ):
        return base
    qualities: list[float] = []
    for values in raw_decode["methods"].values():
        if not isinstance(values, dict):
            continue
        termination = float(values.get("terminated_rate", 0.0))
        edit_quality = 1.0 - min(
            1.0,
            float(values.get("mean_normalized_token_edit_distance", 1.0)),
        )
        qualities.append(0.5 * termination + 0.5 * edit_quality)
    if not qualities:
        return base
    raw_quality = _mean_scores(qualities)
    teacher_quality = math.exp(
        -max(0.0, float(validation_metrics.get("trace_to_tree", 5.0)))
    )
    exposure_gap = max(0.0, teacher_quality - raw_quality)
    failure = max(0.0, 1.0 - raw_quality)
    multiplier = 1.0 + 0.5 * failure + 0.5 * exposure_gap
    return min(config.scheduled_sampling_max, base * multiplier)


def _repeat_latent_row(
    distribution: LatentDistribution,
    row: int,
    count: int,
) -> LatentDistribution:
    def repeat(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return value[row : row + 1].expand(count, *value.shape[1:])

    return LatentDistribution(
        mu=repeat(distribution.mu),
        logvar=repeat(distribution.logvar),
        memory=repeat(distribution.memory),
        pre_normalized=repeat(distribution.pre_normalized),
        activity_mask=repeat(distribution.activity_mask),
        activity_memory=repeat(distribution.activity_memory),
    )


def beam_minimum_risk_loss(
    model: ProcRosettaModel,
    outputs: dict[str, object],
    batch: dict[str, object],
    *,
    beam_size: int,
    max_decode_length: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Optimize expected test-aligned cost under generated beam candidates."""

    dists = outputs["dists"]
    assert isinstance(dists, dict)
    decoder_targets = batch["decoder_targets"]
    source_masks = batch["source_activity_masks"]
    samples = batch.get("samples")
    assert isinstance(decoder_targets, dict)
    assert isinstance(source_masks, dict)
    assert isinstance(samples, list)
    first = dists["fused"].mu
    total = first.sum() * 0.0
    rows = 0
    top1_exact = 0
    oracle_exact = 0
    top1_edits: list[float] = []
    oracle_edits: list[float] = []
    top1_behaviors: list[float] = []
    oracle_behaviors: list[float] = []
    for source_name in ("tree", "trace", "petri", "fused"):
        distribution = dists[source_name]
        assert isinstance(distribution, LatentDistribution)
        allowed = source_masks[source_name]
        assert isinstance(allowed, torch.Tensor)
        for row, sample in enumerate(samples):
            source_row: LatentDistribution | torch.Tensor = (
                distribution.mu[row : row + 1]
                if source_name == "fused"
                else _repeat_latent_row(distribution, row, 1)
            )
            with torch.no_grad():
                candidates = model.tree_decoder.decode_beam_candidates(
                    source_row,
                    max_length=max_decode_length,
                    beam_size=beam_size,
                    length_penalty=0.7,
                    allowed_activity_mask=allowed[row : row + 1],
                    avoid_duplicate_activity_labels=False,
                    completion_policy="bounded",
                )[0]
            if not candidates:
                continue
            candidate_ids = [tokens for tokens, _ in candidates]
            width = max(len(tokens) for tokens in candidate_ids)
            padded = torch.full(
                (len(candidate_ids), width),
                model.tree_tokenizer.pad_id,
                dtype=torch.long,
                device=first.device,
            )
            for candidate_index, token_ids in enumerate(candidate_ids):
                padded[candidate_index, : len(token_ids)] = torch.tensor(
                    token_ids,
                    dtype=torch.long,
                    device=first.device,
                )
            scoring_source: LatentDistribution | torch.Tensor = (
                distribution.mu[row : row + 1].expand(len(candidate_ids), -1)
                if source_name == "fused"
                else _repeat_latent_row(distribution, row, len(candidate_ids))
            )
            logits = model.tree_decoder(
                scoring_source,
                padded[:, :-1],
                allowed_activity_mask=allowed[row : row + 1].expand(
                    len(candidate_ids), -1
                ),
            )
            predicted = padded[:, 1:]
            active = predicted.ne(model.tree_tokenizer.pad_id)
            token_log_prob = F.log_softmax(logits, dim=-1).gather(
                -1,
                predicted.unsqueeze(-1),
            ).squeeze(-1)
            lengths = active.sum(dim=-1).clamp_min(1)
            sequence_log_prob = (
                token_log_prob.masked_fill(~active, 0.0).sum(dim=-1)
                / lengths.to(token_log_prob.dtype).pow(0.7)
            )
            target_ids = _trim_token_ids(
                decoder_targets[source_name][row].detach().cpu().tolist(),
                model.tree_tokenizer,
            )
            target_tree = model.tree_tokenizer.decode_tree(target_ids)
            target_traces = simulate_traces(target_tree, num_traces=16)
            costs: list[float] = []
            edits: list[float] = []
            behaviors: list[float] = []
            exacts: list[bool] = []
            for token_ids in candidate_ids:
                trimmed = _trim_token_ids(token_ids, model.tree_tokenizer)
                edit = _token_edit_distance(target_ids, trimmed) / max(
                    len(target_ids), len(trimmed), 1
                )
                terminated = model.tree_tokenizer.eos_id in token_ids
                valid = False
                convertible = False
                behavior = 2.0
                directly_follows = 2.0
                try:
                    tree = model.tree_tokenizer.decode_tree(token_ids)
                    valid = True
                    tree_to_petri_net(tree)
                    convertible = True
                    decoded_traces = simulate_traces(tree, num_traces=16)
                    reference_traces = (
                        sample.traces if source_name == "trace" else target_traces
                    )
                    distance = behavioral_distance(reference_traces, decoded_traces)
                    behavior = float(distance["mean_l1"])
                    directly_follows = float(distance["directly_follows_l1"])
                except Exception:
                    pass
                exact = trimmed == target_ids
                cost = (
                    0.25 * float(not exact)
                    + 0.25 * edit
                    + 0.10 * float(not terminated)
                    + 0.10 * float(not valid)
                    + 0.10 * float(not convertible)
                    + 0.15 * min(behavior / 2.0, 1.0)
                    + 0.05 * min(directly_follows / 2.0, 1.0)
                )
                costs.append(cost)
                edits.append(edit)
                behaviors.append(behavior)
                exacts.append(exact)
            probabilities = F.softmax(sequence_log_prob, dim=0)
            total = total + (
                probabilities
                * torch.tensor(costs, dtype=probabilities.dtype, device=first.device)
            ).sum()
            rows += 1
            top1_exact += int(exacts[0])
            oracle_exact += int(any(exacts))
            top1_edits.append(edits[0])
            oracle_edits.append(min(edits))
            top1_behaviors.append(behaviors[0])
            oracle_behaviors.append(min(behaviors))
    zero = first.detach().sum() * 0.0
    diagnostics = {
        "beam_top1_exact": zero + top1_exact / max(rows, 1),
        "beam_oracle_exact": zero + oracle_exact / max(rows, 1),
        "beam_top1_edit": zero + _mean_numeric(top1_edits),
        "beam_oracle_edit": zero + _mean_numeric(oracle_edits),
        "beam_top1_behavior_l1": zero + _mean_numeric(top1_behaviors),
        "beam_oracle_behavior_l1": zero + _mean_numeric(oracle_behaviors),
    }
    return total / max(rows, 1), diagnostics


def _mean_numeric(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


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
    collect_curriculum_metrics: bool = False,
    semantic_memory_bank: SemanticMemoryBank | None = None,
    scheduled_sampling_override: float | None = None,
) -> dict[str, object]:
    model.train()
    totals: dict[str, float | torch.Tensor] = {}
    batches = 0
    totals_by_curriculum: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    batches_by_curriculum: dict[str, int] = defaultdict(int)
    iterator = progress_dataloader(
        dataloader,
        desc=f"Epoch {epoch} training" if epoch is not None else "Training",
        enabled=show_progress,
    )
    for batch in iterator:
        batch_level: str | None = None
        if collect_curriculum_metrics:
            levels = {
                str(level)
                for level in batch.get("complexity_levels", [])
                if level is not None
            }
            if len(levels) != 1:
                raise ValueError(
                    "structural curriculum batches must be homogeneous; received "
                    f"{sorted(levels)}"
                )
            batch_level = next(iter(levels))
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            batch,
            deterministic=True,
            input_token_dropout=(
                0.0 if train_config is None else train_config.decoder_input_dropout
            ),
            scheduled_sampling_probability=(
                scheduled_sampling_override
                if scheduled_sampling_override is not None
                else 0.0
                if train_config is None
                else scheduled_sampling_probability(train_config, epoch)
            ),
            modality_subset_fusion_probability=(
                0.0
                if train_config is None
                else train_config.modality_subset_fusion_probability
            ),
            deployment_policy_probability=(
                0.0
                if train_config is None
                else train_config.deployment_policy_probability
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
            exact_positive_mask=batch.get("exact_positive_mask"),
            analogy_mask=batch.get("analogy_mask"),
            family_positive_mask=batch.get("family_positive_mask"),
            negative_mask=batch.get("negative_mask"),
            observation_view_mask=batch.get("observation_view_mask"),
            contrastive_candidate_mask=batch.get("contrastive_candidate_mask"),
            semantic_memory_bank=semantic_memory_bank,
            strong_behavior_ids=batch.get("strong_behavior_ids"),
            equivalence_ids=batch.get("equivalence_ids"),
            behavior_signatures=batch.get("behavior_signatures"),
            observed_behavior_distances=batch.get("observed_behavior_distances"),
            observed_behavior_pair_mask=batch.get("observed_behavior_pair_mask"),
            tokenizer=model.tree_tokenizer,
        )
        active_weights = weights or LossWeights()
        run_beam_risk = (
            train_config is not None
            and epoch is not None
            and epoch >= train_config.beam_risk_start_epoch
            and active_weights.beam_minimum_risk > 0.0
            and bool(
                torch.rand((), device=device)
                < train_config.beam_risk_batch_probability
            )
        )
        if run_beam_risk:
            beam_risk, beam_diagnostics = beam_minimum_risk_loss(
                model,
                outputs,
                batch,
                beam_size=train_config.beam_risk_size,
                max_decode_length=train_config.beam_risk_max_decode_length,
            )
        else:
            beam_risk = losses["loss"].sum() * 0.0
            beam_diagnostics = {
                name: beam_risk.detach()
                for name in (
                    "beam_top1_exact",
                    "beam_oracle_exact",
                    "beam_top1_edit",
                    "beam_oracle_edit",
                    "beam_top1_behavior_l1",
                    "beam_oracle_behavior_l1",
                )
            }
        losses["beam_minimum_risk"] = beam_risk
        losses.update(beam_diagnostics)
        losses["loss"] = (
            losses["loss"] + active_weights.beam_minimum_risk * beam_risk
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
                active_weights,
            )
            totals.update(gradient_metrics)
        if train_config is not None and train_config.use_pcgrad:
            reconstruction_objective = (
                active_weights.tree_reconstruction * losses["tree_reconstruction"]
                + active_weights.trace_to_tree * losses["trace_to_tree"]
                + active_weights.petri_to_tree * losses["petri_to_tree"]
                + active_weights.fused_to_tree * losses["fused_to_tree"]
                + active_weights.fused_subset_to_tree
                * losses["fused_subset_to_tree"]
                + active_weights.deployment_to_tree * losses["deployment_to_tree"]
            )
            metric_objective = (
                active_weights.exact_contrastive * losses["exact_contrastive"]
                + active_weights.within_modality_contrastive
                * losses["within_modality_contrastive"]
                + active_weights.semantic_exact_contrastive
                * losses["semantic_exact_contrastive"]
                + active_weights.semantic_memory_contrastive
                * losses["semantic_memory_contrastive"]
                + active_weights.hierarchical_metric * losses["hierarchical_metric"]
                + active_weights.soft_behavior_geometry
                * losses["soft_behavior_geometry"]
                + active_weights.observed_behavior_regression
                * losses["observed_behavior_regression"]
                + active_weights.observed_behavior_ranking
                * losses["observed_behavior_ranking"]
                + active_weights.observation_view_consistency
                * losses["observation_view_consistency"]
                + active_weights.variance * losses["variance"]
                + active_weights.covariance * losses["covariance"]
                + active_weights.latent_alignment * losses["latent_alignment"]
            )
            pcgrad_metrics = pcgrad_backward(
                model,
                reconstruction_objective,
                metric_objective,
                losses["loss"] - reconstruction_objective - metric_objective,
            )
            for name, value in pcgrad_metrics.items():
                totals[name] = totals.get(name, 0.0) + value
        else:
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
            if batch_level is not None:
                level_totals = totals_by_curriculum[batch_level]
                level_totals[name] = level_totals.get(
                    name, torch.zeros_like(value.detach())
                ) + value.detach()
        if batch_level is not None:
            batches_by_curriculum[batch_level] += 1
        batches += 1
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{float(losses['loss'].detach().cpu()):.4f}")
    result: dict[str, object] = {
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
    if collect_curriculum_metrics:
        result["by_curriculum"] = {
            level: {
                name: float(value.detach().cpu()) / max(batches_by_curriculum[level], 1)
                for name, value in values.items()
            }
            for level, values in totals_by_curriculum.items()
        }
    return result


def gradient_norm_diagnostics(
    model: ProcRosettaModel,
    losses: dict[str, torch.Tensor],
    weights: LossWeights,
) -> dict[str, float]:
    first_loss = next(iter(losses.values()))

    def value(name: str) -> torch.Tensor:
        return losses.get(name, first_loss.sum() * 0.0)

    metric_objective = (
        weights.exact_contrastive * value("exact_contrastive")
        + weights.within_modality_contrastive * value("within_modality_contrastive")
        + weights.semantic_exact_contrastive * value("semantic_exact_contrastive")
        + weights.semantic_memory_contrastive * value("semantic_memory_contrastive")
        + weights.hierarchical_metric * value("hierarchical_metric")
        + weights.soft_behavior_geometry * value("soft_behavior_geometry")
        + weights.observed_behavior_regression * value("observed_behavior_regression")
        + weights.observed_behavior_ranking * value("observed_behavior_ranking")
        + weights.variance * value("variance")
        + weights.covariance * value("covariance")
    )
    semantic_objective = (
        weights.semantic_exact_contrastive * value("semantic_exact_contrastive")
        + weights.semantic_memory_contrastive
        * value("semantic_memory_contrastive")
        + weights.hierarchical_metric * value("hierarchical_metric")
    )
    observed_geometry_objective = (
        weights.observed_behavior_regression
        * value("observed_behavior_regression")
        + weights.observed_behavior_ranking * value("observed_behavior_ranking")
    )
    beam_objective = weights.beam_minimum_risk * value("beam_minimum_risk")
    fused_objective = weights.fused_to_tree * value("fused_to_tree")
    specifications = {
        "tree": (
            model.tree_encoder,
            weights.tree_reconstruction * value("tree_reconstruction")
            + weights.fused_to_tree * value("fused_to_tree"),
        ),
        "trace": (
            model.trace_encoder,
            weights.trace_to_tree * value("trace_to_tree")
            + weights.fused_to_tree * value("fused_to_tree"),
        ),
        "petri": (
            model.petri_encoder,
            weights.petri_to_tree * value("petri_to_tree")
            + weights.fused_to_tree * value("fused_to_tree"),
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
            value("exact_contrastive"),
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        soft_gradients = torch.autograd.grad(
            value("soft_behavior_geometry"),
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        raw_reconstruction_gradients = torch.autograd.grad(
            value(f"{name}_reconstruction" if name == "tree" else f"{name}_to_tree"),
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        fused_gradients = torch.autograd.grad(
            fused_objective,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        semantic_gradients = torch.autograd.grad(
            semantic_objective,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        observed_geometry_gradients = torch.autograd.grad(
            observed_geometry_objective,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        beam_gradients = torch.autograd.grad(
            beam_objective,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        reconstruction_norm = _gradient_norm(reconstruction_gradients)
        metric_norm = _gradient_norm(metric_gradients)
        result[f"reconstruction_gradient_norm_{name}"] = reconstruction_norm
        result[f"metric_gradient_norm_{name}"] = metric_norm
        result[f"source_reconstruction_gradient_norm_{name}"] = _gradient_norm(
            raw_reconstruction_gradients
        )
        result[f"fused_reconstruction_gradient_norm_{name}"] = _gradient_norm(
            fused_gradients
        )
        result[f"semantic_retrieval_gradient_norm_{name}"] = _gradient_norm(
            semantic_gradients
        )
        result[f"observed_geometry_gradient_norm_{name}"] = _gradient_norm(
            observed_geometry_gradients
        )
        result[f"beam_risk_gradient_norm_{name}"] = _gradient_norm(beam_gradients)
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
        result[f"reconstruction_semantic_gradient_cosine_{name}"] = _gradient_cosine(
            raw_reconstruction_gradients,
            semantic_gradients,
        )
        result[f"reconstruction_observed_geometry_gradient_cosine_{name}"] = (
            _gradient_cosine(
                raw_reconstruction_gradients,
                observed_geometry_gradients,
            )
        )
        result[f"semantic_observed_geometry_gradient_cosine_{name}"] = (
            _gradient_cosine(semantic_gradients, observed_geometry_gradients)
        )
        result[f"reconstruction_beam_risk_gradient_cosine_{name}"] = (
            _gradient_cosine(raw_reconstruction_gradients, beam_gradients)
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


def pcgrad_backward(
    model: ProcRosettaModel,
    reconstruction_objective: torch.Tensor,
    metric_objective: torch.Tensor,
    auxiliary_objective: torch.Tensor,
) -> dict[str, float]:
    """Project metric gradients when they conflict with reconstruction."""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
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
    auxiliary_gradients = torch.autograd.grad(
        auxiliary_objective,
        parameters,
        allow_unused=True,
    )
    dot = sum(
        (left * right).sum()
        for left, right in zip(reconstruction_gradients, metric_gradients)
        if left is not None and right is not None
    )
    reconstruction_squared = sum(
        gradient.square().sum()
        for gradient in reconstruction_gradients
        if gradient is not None
    )
    metric_squared = sum(
        gradient.square().sum()
        for gradient in metric_gradients
        if gradient is not None
    )
    dot_value = float(dot.detach().cpu()) if isinstance(dot, torch.Tensor) else 0.0
    denominator = float(
        (reconstruction_squared * metric_squared).sqrt().detach().cpu()
    ) if isinstance(reconstruction_squared, torch.Tensor) and isinstance(metric_squared, torch.Tensor) else 0.0
    conflict = dot_value < 0.0 and isinstance(reconstruction_squared, torch.Tensor)
    coefficient = (
        dot / reconstruction_squared.clamp_min(1e-12)
        if conflict and isinstance(dot, torch.Tensor)
        else None
    )
    for parameter, reconstruction, metric, auxiliary in zip(
        parameters,
        reconstruction_gradients,
        metric_gradients,
        auxiliary_gradients,
    ):
        gradient = None
        for value in (reconstruction, metric, auxiliary):
            if value is not None:
                gradient = value if gradient is None else gradient + value
        if conflict and metric is not None and reconstruction is not None:
            assert coefficient is not None
            projected_metric = metric - coefficient * reconstruction
            gradient = projected_metric
            if reconstruction is not None:
                gradient = gradient + reconstruction
            if auxiliary is not None:
                gradient = gradient + auxiliary
        parameter.grad = None if gradient is None else gradient.detach()
    return {
        "pcgrad_reconstruction_metric_cosine": (
            dot_value / max(denominator, 1e-12)
        ),
        "pcgrad_projection_applied": float(conflict),
    }


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
            exact_positive_mask=batch.get("exact_positive_mask"),
            analogy_mask=batch.get("analogy_mask"),
            family_positive_mask=batch.get("family_positive_mask"),
            negative_mask=batch.get("negative_mask"),
            observation_view_mask=batch.get("observation_view_mask"),
            contrastive_candidate_mask=batch.get("contrastive_candidate_mask"),
            behavior_signatures=batch.get("behavior_signatures"),
            observed_behavior_distances=batch.get("observed_behavior_distances"),
            observed_behavior_pair_mask=batch.get("observed_behavior_pair_mask"),
            tokenizer=model.tree_tokenizer,
        )
        for name, value in losses.items():
            totals[name] = totals.get(name, torch.zeros_like(value.detach())) + value.detach()
        batches += 1
        if compute_discovery_metrics:
            dists = outputs["dists"]
            assert isinstance(dists, dict)
            trace_distribution = dists["trace"]
            maximum_length = min(512, max(3, tree_tokens.shape[1] * 2))
            source_activity_masks = batch.get("source_activity_masks")
            trace_allowed = (
                source_activity_masks.get("trace")
                if isinstance(source_activity_masks, dict)
                else None
            )
            decoded = model.tree_decoder.decode_guaranteed(
                trace_distribution,
                total_token_budget_including_bos_eos=maximum_length,
                allowed_activity_mask=trace_allowed,
                avoid_duplicate_activity_labels=False,
            )
            shuffled_decoded: torch.Tensor | None = None
            zero_decoded: torch.Tensor | None = None
            if source_ablation:
                permutation = _different_behavior_permutation(
                    batch.get(
                        "strong_behavior_ids",
                        batch.get("exact_behavior_ids", []),
                    ),
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
                shuffled_decoded = model.tree_decoder.decode_guaranteed(
                    shuffled_distribution,
                    total_token_budget_including_bos_eos=maximum_length,
                    allowed_activity_mask=trace_allowed,
                    avoid_duplicate_activity_labels=False,
                )
                zero_decoded = model.tree_decoder.decode_guaranteed(
                    zero_distribution,
                    total_token_budget_including_bos_eos=maximum_length,
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
                target_structure = _folded_tree_statistics(
                    target_ids,
                    model.tree_tokenizer,
                )
                decoded_structure = _folded_tree_statistics(
                    prediction_ids,
                    model.tree_tokenizer,
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
                        "target_size": (
                            None if target_structure is None else target_structure[0]
                        ),
                        "target_depth": (
                            None if target_structure is None else target_structure[1]
                        ),
                        "target_duplicate_count": (
                            None if target_structure is None else target_structure[2]
                        ),
                        "decoded_size": (
                            None if decoded_structure is None else decoded_structure[0]
                        ),
                        "decoded_depth": (
                            None if decoded_structure is None else decoded_structure[1]
                        ),
                        "decoded_duplicate_count": (
                            None if decoded_structure is None else decoded_structure[2]
                        ),
                        "decoded_has_duplicates": (
                            False if decoded_structure is None else decoded_structure[3]
                        ),
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
            batch_exact_ids = list(
                batch.get(
                    "strong_behavior_ids",
                    batch.get("exact_behavior_ids", []),
                )
            )
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


def _folded_tree_statistics(
    token_ids: list[int],
    tokenizer: TreeTokenizer,
) -> tuple[int, int, int, bool] | None:
    """Return size, depth, and duplicate statistics for a folded decode."""

    from proc_rosetta.pm4py_bridge import fold_process_tree

    try:
        tree = fold_process_tree(tokenizer.decode_tree(token_ids))
    except (TypeError, ValueError):
        return None
    labels = tree.activity_labels()
    duplicate_count = len(labels) - len(set(labels))
    return (
        tree.size(),
        tree.max_depth(),
        duplicate_count,
        duplicate_count > 0,
    )


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
    structural_rows = [
        row
        for row in rows
        if row.get("decoded_size") is not None
        and row.get("target_size") is not None
    ]
    metrics["trace_decoded_mean_size"] = (
        float(np.mean([float(row["decoded_size"]) for row in structural_rows]))
        if structural_rows
        else 0.0
    )
    metrics["trace_decoded_mean_depth"] = (
        float(np.mean([float(row["decoded_depth"]) for row in structural_rows]))
        if structural_rows
        else 0.0
    )
    metrics["trace_decoded_mean_duplicate_count"] = (
        float(
            np.mean(
                [float(row["decoded_duplicate_count"]) for row in structural_rows]
            )
        )
        if structural_rows
        else 0.0
    )
    metrics["trace_decoded_duplicate_rate"] = (
        sum(bool(row["decoded_has_duplicates"]) for row in structural_rows)
        / len(structural_rows)
        if structural_rows
        else 0.0
    )
    metrics["trace_decoded_mean_size_delta"] = (
        float(
            np.mean(
                [
                    float(row["decoded_size"]) - float(row["target_size"])
                    for row in structural_rows
                ]
            )
        )
        if structural_rows
        else 0.0
    )
    metrics["trace_decoded_mean_depth_delta"] = (
        float(
            np.mean(
                [
                    float(row["decoded_depth"]) - float(row["target_depth"])
                    for row in structural_rows
                ]
            )
        )
        if structural_rows
        else 0.0
    )
    metrics["trace_decoded_mean_duplicate_count_delta"] = (
        float(
            np.mean(
                [
                    float(row["decoded_duplicate_count"])
                    - float(row["target_duplicate_count"])
                    for row in structural_rows
                ]
            )
        )
        if structural_rows
        else 0.0
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
    # Retain legacy component keys for tabular consumers while exposing the
    # unified balanced score used by all training-state decisions.
    behavior_spearman = metrics.get("behavior_distance_spearman", -1.0)
    metrics["checkpoint_selection_primary_exact"] = primary_exact
    metrics["checkpoint_selection_edit_score"] = 1.0 - metrics[
        "trace_normalized_tree_edit"
    ]
    metrics["checkpoint_selection_recall_at_1"] = metrics[
        "exact_behavior_recall_at_1"
    ]
    metrics["checkpoint_selection_spearman"] = behavior_spearman
    components = balanced_validation_components(metrics)
    metrics["checkpoint_selection_score"] = float(components["balanced_score"])
    for name in (
        "decode_score",
        "retrieval_score",
        "equivalence_score",
        "geometry_score",
        "discovery_score",
        "fused_geometry_advantage",
    ):
        metrics[f"checkpoint_selection_{name}"] = float(components[name])
    metrics["checkpoint_selection_hard_gates_pass"] = float(
        bool(components["all_hard_gates_pass"])
    )
    return metrics


def attach_validation_audit(
    model: ProcRosettaModel,
    dataloader: DataLoader,
    metrics: dict[str, float],
    *,
    curriculum: str,
    epoch: int,
    train_config: TrainConfig,
    device: torch.device,
    cache_dir: str | Path,
    stage_transition: bool,
    show_progress: bool,
) -> dict[str, object]:
    """Attach the evaluator-equivalent report used by model selection."""

    if not train_config.validation_audit_enabled:
        return dict(metrics)
    from proc_rosetta.benchmarks import (
        ValidationAuditConfig,
        validation_audit_report,
    )

    samples = getattr(getattr(dataloader, "dataset", None), "samples", None)
    if not isinstance(samples, list):
        raise ValueError("validation audit requires a loader backed by persisted samples")
    audit = validation_audit_report(
        model,
        samples,
        curriculum,
        loss_metrics=metrics,
        epoch=epoch,
        config=ValidationAuditConfig(
            decode_interval=train_config.validation_decode_interval,
            full_interval=train_config.validation_full_interval,
            decode_family_count=train_config.validation_decode_family_count,
            discovery_family_count=train_config.validation_discovery_family_count,
            beam_size=train_config.validation_beam_size,
            max_decode_length=train_config.validation_max_decode_length,
            cache_dir=str(cache_dir),
        ),
        batch_size=train_config.batch_size,
        device=str(device),
        show_progress=show_progress,
        stage_transition=stage_transition,
    )
    combined: dict[str, object] = {**metrics, **audit}
    components = balanced_validation_components(combined)
    combined["checkpoint_selection_score"] = float(components["balanced_score"])
    combined["checkpoint_selection_hard_gates_pass"] = float(
        bool(components["all_hard_gates_pass"])
    )
    for name in (
        "decode_score",
        "retrieval_score",
        "equivalence_score",
        "geometry_score",
        "discovery_score",
        "fused_geometry_advantage",
    ):
        combined[f"checkpoint_selection_{name}"] = float(components[name])
    return combined


def _mean_scores(values: list[float], default: float = 0.0) -> float:
    return float(np.mean(values)) if values else float(default)


def _decode_method_score(values: dict[str, object]) -> tuple[float, bool]:
    scores = [
        float(values.get("exact_tree_match_rate", 0.0)),
        1.0 - min(1.0, float(values.get("mean_normalized_token_edit_distance", 1.0))),
        1.0 - min(1.0, float(values.get("mean_behavior_l1", 2.0)) / 2.0),
    ]
    gate_values = [
        float(values.get(name, 0.0))
        for name in (
            "terminated_rate",
            "valid_tree_rate",
            "petri_conversion_rate",
            "behavior_eval_success_rate",
        )
    ]
    scores.extend(gate_values)
    return _mean_scores(scores), all(value >= 0.95 for value in gate_values)


def balanced_validation_components(
    metrics: dict[str, object],
) -> dict[str, float | bool]:
    """Calculate the test-aligned balanced score for flat or rich metrics."""

    bounded_decode = metrics.get(
        "deployment_decode_quality",
        metrics.get("decode_quality"),
    )
    if isinstance(bounded_decode, dict):
        method_scores: list[float] = []
        hard_gates: list[bool] = []
        methods = bounded_decode.get("methods", {})
        if isinstance(methods, dict):
            for values in methods.values():
                if isinstance(values, dict):
                    score, passed = _decode_method_score(values)
                    method_scores.append(score)
                    hard_gates.append(passed)
        decode_score = _mean_scores(method_scores)
        all_hard_gates_pass = bool(hard_gates) and all(hard_gates)
    else:
        exact = float(
            metrics.get(
                "trace_canonical_exact",
                metrics.get("checkpoint_selection_primary_exact", 0.0),
            )
        )
        edit = float(
            metrics.get(
                "trace_normalized_tree_edit",
                1.0 - float(metrics.get("checkpoint_selection_edit_score", 0.0)),
            )
        )
        bounded_exact = float(
            metrics.get("deployment_duplicate_free_tree_exact", exact)
        )
        bounded_edit = float(
            metrics.get("deployment_duplicate_free_tree_edit", edit)
        )
        decode_score = float(
            metrics.get(
                "checkpoint_selection_decode_score",
                0.60 * _mean_scores([exact, 1.0 - min(edit, 1.0)])
                + 0.40
                * _mean_scores([bounded_exact, 1.0 - min(bounded_edit, 1.0)]),
            )
        )
        available_gates = [
            float(metrics[name])
            for name in (
                "terminated_rate",
                "valid_tree_rate",
                "petri_conversion_rate",
                "behavior_eval_success_rate",
            )
            if isinstance(metrics.get(name), (int, float))
        ]
        stored_gate = metrics.get("checkpoint_selection_hard_gates_pass")
        all_hard_gates_pass = (
            float(stored_gate) >= 1.0
            if isinstance(stored_gate, (int, float))
            else all(value >= 0.95 for value in available_gates)
        )

    retrieval = metrics.get("cross_modal_retrieval")
    retrieval_values: list[float] = []
    if isinstance(retrieval, dict):
        for direction in retrieval.values():
            if not isinstance(direction, dict):
                continue
            for name in (
                "top1_accuracy",
                "mrr",
                "recall_at_5",
                "partial_order_recall_at_1",
                "partial_order_recall_at_5",
            ):
                if isinstance(direction.get(name), (int, float)):
                    retrieval_values.append(float(direction[name]))
            if isinstance(direction.get("analogy_neighborhood_spearman"), (int, float)):
                retrieval_values.append(
                    (float(direction["analogy_neighborhood_spearman"]) + 1.0) / 2.0
                )
    retrieval_score = _mean_scores(
        retrieval_values,
        float(
            metrics.get(
                "checkpoint_selection_retrieval_score",
                metrics.get(
                    "exact_behavior_recall_at_1",
                    metrics.get("checkpoint_selection_recall_at_1", 0.0),
                ),
            )
        ),
    )

    equivalence = metrics.get("equivalence_families")
    equivalence_values: list[float] = []
    if isinstance(equivalence, dict):
        methods = equivalence.get("methods", {})
        if isinstance(methods, dict):
            for values in methods.values():
                if not isinstance(values, dict):
                    continue
                for name in (
                    "within_family_cosine",
                    "log_resampling_consistency",
                ):
                    if isinstance(values.get(name), (int, float)):
                        equivalence_values.append((float(values[name]) + 1.0) / 2.0)
                if isinstance(values.get("equivalence_margin"), (int, float)):
                    equivalence_values.append(
                        min(1.0, max(0.0, (float(values["equivalence_margin"]) + 2.0) / 4.0))
                    )
                if isinstance(values.get("behavior_id_retrieval_top1"), (int, float)):
                    equivalence_values.append(float(values["behavior_id_retrieval_top1"]))
    equivalence_score = _mean_scores(
        equivalence_values,
        float(metrics.get("checkpoint_selection_equivalence_score", retrieval_score)),
    )

    geometry_score = float(
        metrics.get(
            "checkpoint_selection_geometry_score",
            (
                float(
                    metrics.get(
                        "behavior_distance_spearman",
                        metrics.get("checkpoint_selection_spearman", -1.0),
                    )
                )
                + 1.0
            )
            / 2.0,
        )
    )
    fused_geometry_advantage = float(
        metrics.get(
            "fused_geometry_advantage",
            metrics.get("checkpoint_selection_fused_geometry_advantage", 0.0),
        )
    )
    methods = metrics.get("embedding_methods")
    if isinstance(methods, dict):
        fused = methods.get("proc_rosetta_fused_mu", {})
        if isinstance(fused, dict):
            alignment = fused.get("behavior_alignment", {})
            nearest = fused.get("nearest_neighbor_behavior", {})
            geometry_values: list[float] = []
            if isinstance(alignment, dict) and isinstance(
                alignment.get("spearman_embedding_distance_vs_behavior_l1"),
                (int, float),
            ):
                fused_rho = float(
                    alignment["spearman_embedding_distance_vs_behavior_l1"]
                )
                geometry_values.append((fused_rho + 1.0) / 2.0)
            else:
                fused_rho = -1.0
            if isinstance(nearest, dict) and isinstance(
                nearest.get("mean_behavior_l1_at_nearest_neighbor"),
                (int, float),
            ):
                fused_nn = float(nearest["mean_behavior_l1_at_nearest_neighbor"])
                geometry_values.append(1.0 - min(fused_nn / 2.0, 1.0))
            else:
                fused_nn = 2.0
            geometry_score = _mean_scores(geometry_values, geometry_score)
            baseline_rhos: list[float] = []
            baseline_nn: list[float] = []
            for name, values in methods.items():
                if name.startswith("proc_rosetta_") or not isinstance(values, dict):
                    continue
                baseline_alignment = values.get("behavior_alignment", {})
                baseline_nearest = values.get("nearest_neighbor_behavior", {})
                if isinstance(baseline_alignment, dict) and isinstance(
                    baseline_alignment.get("spearman_embedding_distance_vs_behavior_l1"),
                    (int, float),
                ):
                    baseline_rhos.append(
                        float(baseline_alignment["spearman_embedding_distance_vs_behavior_l1"])
                    )
                if isinstance(baseline_nearest, dict) and isinstance(
                    baseline_nearest.get("mean_behavior_l1_at_nearest_neighbor"),
                    (int, float),
                ):
                    baseline_nn.append(
                        float(baseline_nearest["mean_behavior_l1_at_nearest_neighbor"])
                    )
            if baseline_rhos or baseline_nn:
                advantages = []
                if baseline_rhos:
                    advantages.append(fused_rho - max(baseline_rhos))
                if baseline_nn:
                    advantages.append(min(baseline_nn) - fused_nn)
                fused_geometry_advantage = _mean_scores(advantages)

    discovery = metrics.get("discovery_quality")
    discovery_values: list[float] = []
    if isinstance(discovery, dict):
        methods = discovery.get("methods", {})
        proc = methods.get("proc_rosetta_trace_mu", {}) if isinstance(methods, dict) else {}
        if isinstance(proc, dict):
            for name in (
                "mean_fitness",
                "mean_precision",
                "mean_f1",
                "conformance_evaluable_rate",
            ):
                if isinstance(proc.get(name), (int, float)):
                    discovery_values.append(float(proc[name]))
    discovery_score = _mean_scores(
        discovery_values,
        float(metrics.get("checkpoint_selection_discovery_score", decode_score)),
    )

    balanced_score = (
        0.40 * decode_score
        + 0.20 * retrieval_score
        + 0.15 * equivalence_score
        + 0.15 * geometry_score
        + 0.10 * discovery_score
    )
    return {
        "balanced_score": balanced_score,
        "decode_score": decode_score,
        "retrieval_score": retrieval_score,
        "equivalence_score": equivalence_score,
        "geometry_score": geometry_score,
        "discovery_score": discovery_score,
        "all_hard_gates_pass": all_hard_gates_pass,
        "fused_geometry_advantage": fused_geometry_advantage,
    }


def checkpoint_selection_key(metrics: dict[str, object]) -> tuple[float, ...]:
    components = balanced_validation_components(metrics)
    return (
        float(bool(components["all_hard_gates_pass"])),
        float(components["balanced_score"]),
        float(components["fused_geometry_advantage"]),
        -float(metrics.get("loss", float("inf"))),
    )


def final_curriculum_checkpoint_key(
    complex_metrics: dict[str, object],
    macro_metrics: dict[str, object],
    worst_metrics: dict[str, object] | None = None,
    *,
    regression_within_tolerance: bool = True,
) -> tuple[float, ...]:
    """Use robust macro/worst performance before complex and loss tie-breaks."""

    complex_components = balanced_validation_components(complex_metrics)
    macro_components = balanced_validation_components(macro_metrics)
    worst_components = balanced_validation_components(worst_metrics or macro_metrics)
    robust_score = (
        0.70 * float(macro_components["balanced_score"])
        + 0.30 * float(worst_components["balanced_score"])
    )
    return (
        float(
            bool(complex_components["all_hard_gates_pass"])
            and bool(macro_components["all_hard_gates_pass"])
            and bool(worst_components["all_hard_gates_pass"])
            and regression_within_tolerance
        ),
        robust_score,
        float(complex_components["balanced_score"]),
        float(complex_components["fused_geometry_advantage"]),
        -float(complex_metrics.get("loss", float("inf"))),
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
    """Build strong/exact-analogy/family/hard-negative relation bundles."""

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
        self.samples = list(samples)
        self.exact_groups: dict[str, list[int]] = defaultdict(list)
        self.family_groups: dict[str, list[int]] = defaultdict(list)
        self.exact_ids: list[str | None] = []
        self.partial_ids: list[str | None] = []
        self.family_ids: list[str] = []
        for index, sample in enumerate(samples):
            exact_behavior_id = (
                getattr(sample, "strong_behavior_id", None)
                or getattr(sample, "exact_behavior_id", None)
            )
            partial_order_id = getattr(sample, "partial_order_id", None)
            family_id = str(getattr(sample, "equivalence_id", index))
            exact_key = None if exact_behavior_id is None else str(exact_behavior_id)
            partial_key = None if partial_order_id is None else str(partial_order_id)
            self.exact_ids.append(exact_key)
            self.partial_ids.append(partial_key)
            self.family_ids.append(family_id)
            if exact_key is not None:
                self.exact_groups[exact_key].append(index)
            self.family_groups[family_id].append(index)
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
        self.hard_negatives = self._hard_negative_indices()

    def __iter__(self):
        batches = self._planned_batches(self.epoch)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        # Relation bundles add analogies, broader-family positives, and hard
        # negatives.  DataLoader and the curriculum interleaver need the exact
        # expanded batch count or they silently leave valid batches unused.
        return len(self._planned_batches(self.epoch))

    def _planned_batches(self, epoch: int) -> list[list[int]]:
        rng = random.Random(self.seed + epoch)
        groups = [list(group) for group in self.groups]
        if self.shuffle:
            rng.shuffle(groups)
            for group in groups:
                rng.shuffle(group)

        per_group_chunks = [
            [self._relation_bundle(chunk, rng) for chunk in self._positive_chunks(group)]
            for group in groups
        ]
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
        batches: list[list[int]] = []
        batch: list[int] = []
        for chunk in chunks:
            if batch and len(batch) + len(chunk) > self.batch_size:
                batches.append(batch)
                batch = []
            if len(chunk) > self.batch_size:
                for start in range(0, len(chunk), self.batch_size):
                    batches.append(chunk[start : start + self.batch_size])
            else:
                batch.extend(chunk)
        if batch:
            batches.append(batch)
        return batches

    def _positive_chunks(self, group: list[int]) -> list[list[int]]:
        if len(group) <= 1:
            return [group]
        # Balance chunks so a trailing singleton is avoided whenever the class
        # has at least two views.  For three views and a target width of two, a
        # single three-view chunk is preferable to one positive pair plus an
        # orphan row.
        target_width = min(
            max(self.views_per_family, 2),
            max(2, self.batch_size - 3),
        )
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

    def _relation_bundle(self, strong_chunk: list[int], rng) -> list[int]:
        bundle = list(strong_chunk)
        anchor = strong_chunk[0]
        exact_id = self.exact_ids[anchor]
        partial_id = self.partial_ids[anchor]
        if exact_id is not None:
            analogues = [
                index
                for index in self.exact_groups.get(exact_id, [])
                if self.partial_ids[index] != partial_id and index not in bundle
            ]
            if analogues:
                bundle.append(rng.choice(analogues))
        broader_family = [
            index
            for index in self.family_groups.get(self.family_ids[anchor], [])
            if self.exact_ids[index] != exact_id and index not in bundle
        ]
        if broader_family:
            bundle.append(rng.choice(broader_family))
        hard_negative = self.hard_negatives[anchor]
        if hard_negative is not None and hard_negative not in bundle:
            bundle.append(hard_negative)
        return bundle

    def _hard_negative_indices(self) -> list[int | None]:
        signatures = [
            tuple(float(value) for value in getattr(sample, "behavior_signature", ()))
            for sample in self.samples
        ]
        if not signatures or not signatures[0] or len({len(row) for row in signatures}) != 1:
            return self._fallback_negative_indices()

        # A full cosine matrix is quadratic: the default 16,384-row split used
        # to allocate a 2 GiB float64 matrix here, three times at startup, then
        # scan all 268 million entries in Python.  Random-projection neighbors
        # provide a small deterministic candidate set, after which cosine
        # similarity is still evaluated in the original signature space.
        matrix = np.asarray(signatures, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized = matrix / np.maximum(norms, 1e-12)
        projection_count = min(4, normalized.shape[1])
        rng = np.random.default_rng(self.seed)
        directions = rng.standard_normal(
            (normalized.shape[1], projection_count),
            dtype=np.float32,
        )
        directions /= np.maximum(
            np.linalg.norm(directions, axis=0, keepdims=True),
            1e-12,
        )
        projected = normalized @ directions
        orders = np.argsort(projected, axis=0, kind="stable")
        ranks = np.empty_like(orders)
        row_numbers = np.arange(self.sample_count, dtype=orders.dtype)
        for projection in range(projection_count):
            ranks[orders[:, projection], projection] = row_numbers

        neighbor_window = 8
        offsets = np.concatenate(
            (
                np.arange(-neighbor_window, 0),
                np.arange(1, neighbor_window + 1),
            )
        )
        candidates = np.empty(
            (self.sample_count, projection_count * len(offsets)),
            dtype=orders.dtype,
        )
        for projection in range(projection_count):
            positions = np.clip(
                ranks[:, projection, None] + offsets[None, :],
                0,
                self.sample_count - 1,
            )
            start = projection * len(offsets)
            candidates[:, start : start + len(offsets)] = orders[
                positions,
                projection,
            ]

        family_number_by_id = {
            name: number
            for number, name in enumerate(dict.fromkeys(self.family_ids))
        }
        family_numbers = np.fromiter(
            (family_number_by_id[family] for family in self.family_ids),
            dtype=np.int64,
            count=self.sample_count,
        )
        best_indices = np.full(self.sample_count, -1, dtype=np.int64)
        best_similarities = np.full(self.sample_count, -np.inf, dtype=np.float32)
        for column in range(candidates.shape[1]):
            candidate_indices = candidates[:, column]
            valid = family_numbers[candidate_indices] != family_numbers
            similarities = np.einsum(
                "nd,nd->n",
                normalized,
                normalized[candidate_indices],
                optimize=True,
            )
            improved = valid & (similarities > best_similarities)
            best_similarities[improved] = similarities[improved]
            best_indices[improved] = candidate_indices[improved]

        fallback = self._fallback_negative_indices()
        return [
            fallback[index] if candidate < 0 else int(candidate)
            for index, candidate in enumerate(best_indices)
        ]

    def _fallback_negative_indices(self) -> list[int | None]:
        representatives: dict[str, int] = {}
        for index, family_id in enumerate(self.family_ids):
            representatives.setdefault(family_id, index)
        first_two = list(representatives.items())[:2]
        if len(first_two) < 2:
            return [None] * self.sample_count
        (first_family, first_index), (_, second_index) = first_two
        return [
            second_index if family_id == first_family else first_index
            for family_id in self.family_ids
        ]

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
        mode="max",
        factor=train_config.lr_factor,
        patience=train_config.lr_patience,
        min_lr=train_config.min_lr,
        threshold=train_config.min_delta,
        threshold_mode="abs",
    )


def scheduler_monitor_value(
    metrics: dict[str, object],
    monitor: str,
) -> float:
    if monitor != "balanced":
        raise ValueError(f"unsupported scheduler monitor: {monitor}")
    return float(balanced_validation_components(metrics)["balanced_score"])


def select_validation_candidate(
    ordinary_metrics: dict[str, object],
    ema_metrics: dict[str, object] | None,
) -> tuple[dict[str, object], str]:
    """Compare ordinary and EMA weights on the same checkpoint-selection key."""

    if ema_metrics is not None and checkpoint_selection_key(
        ema_metrics
    ) > checkpoint_selection_key(ordinary_metrics):
        return ema_metrics, "ema"
    return ordinary_metrics, "ordinary"


def macro_average_metrics(
    metrics_by_curriculum: dict[str, dict[str, float]],
) -> dict[str, float]:
    names = set.intersection(
        *(set(metrics) for metrics in metrics_by_curriculum.values())
    ) if metrics_by_curriculum else set()
    return {
        name: float(np.mean([metrics[name] for metrics in metrics_by_curriculum.values()]))
        for name in sorted(names)
        if all(isinstance(metrics[name], (int, float)) for metrics in metrics_by_curriculum.values())
    }


def worst_case_metrics(
    metrics_by_curriculum: dict[str, dict[str, float]],
) -> dict[str, float]:
    macro = macro_average_metrics(metrics_by_curriculum)
    result: dict[str, float] = {}
    for name in macro:
        values = [float(metrics[name]) for metrics in metrics_by_curriculum.values()]
        try:
            direction = metric_spec(name).direction
        except KeyError:
            # Unknown diagnostics are deliberately excluded from the robust
            # selection surface instead of guessing their direction by name.
            continue
        result[name] = max(values) if direction == "min" else min(values)
    return result


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
    semantic_memory_bank = SemanticMemoryBank(train_config.semantic_memory_bank_size)
    history = []
    for epoch in range(1, train_config.epochs + 1):
        history.append(
            train_epoch(
                model,
                dataloader,
                optimizer,
                device,
                weights=loss_weights_from_config(train_config, epoch=epoch),
                epoch=epoch,
                train_config=train_config,
                ema=ema,
                semantic_memory_bank=semantic_memory_bank,
            )
        )
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
    curriculum_mode = metadata.get("schema") == "proc-rosetta.structural-curriculum.v1"
    if int(metadata.get("version", 0)) < 5:
        raise ValueError(
            "data uses a legacy schema without semantic folding and per-modality "
            "decoder targets; recreate it with sample.py before training"
        )
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
    if curriculum_mode:
        batch_sizes = {
            "simple": train_config.simple_batch_size or train_config.batch_size,
            "medium": train_config.medium_batch_size
            or max(1, round(train_config.batch_size * 0.75)),
            "complex": train_config.complex_batch_size
            or max(1, round(train_config.batch_size * 0.50)),
        }
        debug("Loading structural-curriculum training splits", enabled=show_progress)
        train_loaders = {
            level: build_jsonl_dataloader(
                split_samples_path(data_dir, "training", level),
                model.tree_tokenizer,
                model.activity_tokenizer,
                batch_size=batch_sizes[level],
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
            for level in CURRICULUM_LEVELS
        }
        debug("Loading structural-curriculum validation splits", enabled=show_progress)
        validation_loaders = {
            level: build_jsonl_dataloader(
                split_samples_path(data_dir, "validation", level),
                model.tree_tokenizer,
                model.activity_tokenizer,
                batch_size=batch_sizes[level],
                shuffle=False,
                show_progress=show_progress,
                num_workers=train_config.loader_num_workers,
                pin_memory=train_config.loader_pin_memory,
                persistent_workers=train_config.loader_persistent_workers,
                prefetch_factor=train_config.loader_prefetch_factor,
            )
            for level in CURRICULUM_LEVELS
        }
        initial_curriculum = structural_curriculum_for_epoch(1, train_config.epochs)
        train_loader = CurriculumMixtureLoader(
            train_loaders,
            dict(initial_curriculum["weights"]),
            seed=train_config.seed,
            epoch=1,
        )
        validation_loader = validation_loaders["simple"]
    else:
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
            split_samples_path(
                data_dir,
                "training",
                "simple" if curriculum_mode else None,
            ),
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
    if curriculum_mode:
        for level in CURRICULUM_LEVELS:
            debug_split(
                f"training/{level}",
                train_loaders[level].dataset.samples,
                len(train_loaders[level]),
                enabled=show_progress,
            )
            debug_split(
                f"validation/{level}",
                validation_loaders[level].dataset.samples,
                len(validation_loaders[level]),
                enabled=show_progress,
            )
    else:
        debug_split("training", train_loader.dataset.samples, len(train_loader), enabled=show_progress)
        debug_split("validation", validation_loader.dataset.samples, len(validation_loader), enabled=show_progress)
    optimizer = build_optimizer(model, train_config)
    scheduler = build_lr_scheduler(optimizer, train_config)
    ema = ModelEMA(train_config.ema_decay) if train_config.use_ema else None
    semantic_memory_bank = SemanticMemoryBank(train_config.semantic_memory_bank_size)
    if ema is not None and resume_checkpoint is not None:
        ema_state = resume_checkpoint.get("ema_state_dict")
        if isinstance(ema_state, dict):
            ema.load_state_dict(ema_state)
    evaluation_weights = loss_weights_from_config(train_config)
    history: list[dict[str, object]] = []
    best_validation_loss = float("inf")
    best_early_stopping_metric = float("-inf")
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
            if int(resume_checkpoint.get("version", 0)) < 7:
                best_early_stopping_metric = max(
                    scheduler_monitor_value(
                        row["validation"],
                        train_config.scheduler_monitor,
                    )
                    for row in history
                    if isinstance(row.get("validation"), dict)
                )
            else:
                best_early_stopping_metric = max(
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
        for name in ("balanced", "decode", "retrieval", "geometry", "discovery")
    }
    structural_checkpoint_paths = {
        level: best_checkpoint_for(checkpoint_path, level)
        for level in CURRICULUM_LEVELS
    }
    best_structural_stage_metrics = {
        level: float("-inf") for level in CURRICULUM_LEVELS
    }
    best_final_curriculum_key = (float("-inf"),) * 5
    for history_row in history:
        level = str(history_row.get("structural_curriculum_stage", ""))
        metric = history_row.get("early_stopping_metric")
        if level in best_structural_stage_metrics and isinstance(metric, (int, float)):
            best_structural_stage_metrics[level] = max(
                best_structural_stage_metrics[level], float(metric)
            )
        stored_final_key = history_row.get("final_checkpoint_selection_key")
        if (
            level == "complex"
            and isinstance(stored_final_key, (list, tuple))
            and len(stored_final_key) == 5
        ):
            best_final_curriculum_key = max(
                best_final_curriculum_key,
                tuple(float(value) for value in stored_final_key),
            )
    best_objectives = best_objectives_from_history(history)
    metrics_csv_path = Path(metrics_csv_path)
    write_metrics_csv(metrics_csv_path, history)
    debug(f"Per-epoch metrics CSV: {metrics_csv_path}", enabled=show_progress)
    previous_structural_stage_value = (
        history[-1].get("structural_curriculum_stage")
        if curriculum_mode and history
        else None
    )
    previous_structural_stage = (
        str(previous_structural_stage_value)
        if previous_structural_stage_value in CURRICULUM_LEVELS
        else None
    )
    previous_validation_by_curriculum = (
        dict(history[-1].get("validation_by_curriculum", {}))
        if curriculum_mode and history
        else {}
    )
    current_stage_epochs = [
        int(row["epoch"])
        for row in history
        if row.get("structural_curriculum_stage") == previous_structural_stage
        and isinstance(row.get("epoch"), (int, float))
    ]
    curriculum_stage_start_epoch = (
        min(current_stage_epochs) if current_stage_epochs else start_epoch
    )
    curriculum_stage_baseline: dict[str, object] | None = None
    if previous_structural_stage is not None:
        for row in history:
            if row.get("structural_curriculum_stage") != previous_structural_stage:
                continue
            by_curriculum = row.get("validation_by_curriculum")
            if isinstance(by_curriculum, dict) and isinstance(
                by_curriculum.get(previous_structural_stage), dict
            ):
                curriculum_stage_baseline = dict(
                    by_curriculum[previous_structural_stage]
                )
                break
    prior_complex_epochs = [
        int(row["epoch"])
        for row in history
        if row.get("structural_curriculum_stage") == "complex"
        and isinstance(row.get("epoch"), (int, float))
    ]
    complex_stage_start_epoch = (
        min(prior_complex_epochs)
        if prior_complex_epochs
        else max(1, math.ceil(0.40 * train_config.epochs) + 1)
    )
    for epoch in range(start_epoch, train_config.epochs + 1):
        epoch_start = perf_counter()
        if curriculum_mode:
            curriculum_state, curriculum_competence = competence_curriculum_state(
                previous_structural_stage,
                previous_validation_by_curriculum,
                curriculum_stage_baseline,
                epochs_in_stage=epoch - curriculum_stage_start_epoch,
                minimum_epochs=train_config.min_curriculum_stage_epochs,
                min_delta=train_config.min_delta,
                simple_best_score=best_structural_stage_metrics["simple"],
                regression_tolerance=train_config.curriculum_regression_tolerance,
            )
        else:
            curriculum_state = {"name": "legacy", "weights": {}}
            curriculum_competence = {
                "current_stage": "legacy",
                "eligible": False,
                "advanced": False,
                "checks": {},
            }
        structural_stage = str(curriculum_state["name"])
        if curriculum_mode:
            curriculum_state["weights"] = deficit_adjusted_curriculum_weights(
                dict(curriculum_state["weights"]),
                previous_validation_by_curriculum,
            )
        stage_transition = (
            curriculum_mode
            and previous_structural_stage is not None
            and structural_stage != previous_structural_stage
        )
        if curriculum_mode:
            if stage_transition:
                curriculum_stage_start_epoch = epoch
                curriculum_stage_baseline = None
            train_loader = CurriculumMixtureLoader(
                train_loaders,
                dict(curriculum_state["weights"]),
                seed=train_config.seed,
                epoch=epoch,
            )
        if stage_transition:
            scheduler = build_lr_scheduler(optimizer, train_config)
            epochs_without_improvement = 0
            best_early_stopping_metric = float("-inf")
            if structural_stage == "complex":
                complex_stage_start_epoch = epoch
                best_validation_loss = float("inf")
                best_validation_score = float("-inf")
                best_validation_key = (float("-inf"),) * 4
                best_objectives = best_objectives_from_history([])
                best_final_curriculum_key = (float("-inf"),) * 5
            debug(
                f"Structural curriculum transition to {structural_stage}; reset "
                "scheduler plateau and early-stopping patience state.",
                enabled=show_progress,
            )
        debug(f"Starting epoch {epoch}/{train_config.epochs}", enabled=show_progress)
        epoch_weights = validation_deficit_loss_weights(
            loss_weights_from_config(train_config, epoch=epoch),
            previous_validation_by_curriculum.get(structural_stage)
            if curriculum_mode
            else (
                history[-1].get("validation")
                if history and isinstance(history[-1].get("validation"), dict)
                else None
            ),
        )
        scheduled_sampling_metrics = (
            previous_validation_by_curriculum.get(structural_stage)
            if curriculum_mode
            else (
                history[-1].get("validation")
                if history and isinstance(history[-1].get("validation"), dict)
                else None
            )
        )
        epoch_scheduled_sampling = adaptive_scheduled_sampling_probability(
            train_config,
            epoch,
            scheduled_sampling_metrics,
        )
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
            collect_curriculum_metrics=curriculum_mode,
            semantic_memory_bank=semantic_memory_bank,
            scheduled_sampling_override=epoch_scheduled_sampling,
        )
        training_by_curriculum = (
            dict(training_metrics.pop("by_curriculum", {}))
            if curriculum_mode
            else {}
        )
        debug(
            f"Epoch {epoch} training complete: {format_metrics(training_metrics)}",
            enabled=show_progress,
        )
        if curriculum_mode:
            ordinary_validation_by_curriculum = {}
            for level in CURRICULUM_LEVELS:
                level_metrics = evaluate_epoch(
                    model,
                    validation_loaders[level],
                    device,
                    weights=evaluation_weights,
                    epoch=epoch,
                    show_progress=show_progress,
                    progress_desc=f"Epoch {epoch} {level} validation",
                    compute_discovery_metrics=(
                        not train_config.validation_audit_enabled
                    ),
                )
                ordinary_validation_by_curriculum[level] = attach_validation_audit(
                    model,
                    validation_loaders[level],
                    level_metrics,
                    curriculum=level,
                    epoch=epoch,
                    train_config=train_config,
                    device=device,
                    cache_dir=Path(checkpoint_path).parent
                    / "validation_cache"
                    / level,
                    stage_transition=stage_transition,
                    show_progress=show_progress,
                )
            ordinary_validation_metrics = ordinary_validation_by_curriculum[
                structural_stage
            ]
        else:
            ordinary_validation_by_curriculum = {}
            ordinary_epoch_metrics = evaluate_epoch(
                model,
                validation_loader,
                device,
                weights=evaluation_weights,
                epoch=epoch,
                show_progress=show_progress,
                compute_discovery_metrics=(
                    not train_config.validation_audit_enabled
                ),
            )
            ordinary_validation_metrics = attach_validation_audit(
                model,
                validation_loader,
                ordinary_epoch_metrics,
                curriculum=str(
                    validation_loader.dataset.samples[0].complexity_level
                    or synthetic_config.complexity_level
                    or "complex"
                ),
                epoch=epoch,
                train_config=train_config,
                device=device,
                cache_dir=Path(checkpoint_path).parent / "validation_cache",
                stage_transition=stage_transition,
                show_progress=show_progress,
            )
        ema_validation_metrics: dict[str, float] | None = None
        validation_weights = "ordinary"
        if ema is not None and ema.initialized:
            with ema.average_parameters(model):
                if curriculum_mode:
                    ema_validation_by_curriculum = {}
                    for level in CURRICULUM_LEVELS:
                        level_metrics = evaluate_epoch(
                            model,
                            validation_loaders[level],
                            device,
                            weights=evaluation_weights,
                            epoch=epoch,
                            show_progress=show_progress,
                            progress_desc=f"Epoch {epoch} EMA {level} validation",
                            compute_discovery_metrics=(
                                not train_config.validation_audit_enabled
                            ),
                        )
                        ema_validation_by_curriculum[level] = attach_validation_audit(
                            model,
                            validation_loaders[level],
                            level_metrics,
                            curriculum=level,
                            epoch=epoch,
                            train_config=train_config,
                            device=device,
                            cache_dir=Path(checkpoint_path).parent
                            / "validation_cache"
                            / level,
                            stage_transition=stage_transition,
                            show_progress=show_progress,
                        )
                    ema_validation_metrics = ema_validation_by_curriculum[
                        structural_stage
                    ]
                else:
                    ema_validation_by_curriculum = {}
                    ema_epoch_metrics = evaluate_epoch(
                        model,
                        validation_loader,
                        device,
                        weights=evaluation_weights,
                        epoch=epoch,
                        show_progress=show_progress,
                        progress_desc=f"Epoch {epoch} EMA validation",
                        compute_discovery_metrics=(
                            not train_config.validation_audit_enabled
                        ),
                    )
                    ema_validation_metrics = attach_validation_audit(
                        model,
                        validation_loader,
                        ema_epoch_metrics,
                        curriculum=str(
                            validation_loader.dataset.samples[0].complexity_level
                            or synthetic_config.complexity_level
                            or "complex"
                        ),
                        epoch=epoch,
                        train_config=train_config,
                        device=device,
                        cache_dir=Path(checkpoint_path).parent / "validation_cache",
                        stage_transition=stage_transition,
                        show_progress=show_progress,
                    )
            validation_metrics, validation_weights = select_validation_candidate(
                ordinary_validation_metrics,
                ema_validation_metrics,
            )
        else:
            ema_validation_by_curriculum = {}
            validation_metrics = ordinary_validation_metrics
        if curriculum_mode:
            validation_by_curriculum = (
                ema_validation_by_curriculum
                if validation_weights == "ema"
                else ordinary_validation_by_curriculum
            )
            validation_macro = macro_average_metrics(validation_by_curriculum)
            validation_worst_case = worst_case_metrics(validation_by_curriculum)
            if curriculum_stage_baseline is None:
                curriculum_stage_baseline = dict(
                    validation_by_curriculum[structural_stage]
                )
        else:
            validation_by_curriculum = {}
            validation_macro = {}
            validation_worst_case = {}
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
        validation_score = float(
            balanced_validation_components(validation_metrics)["balanced_score"]
        )
        monitor_value = scheduler_monitor_value(
            validation_metrics,
            train_config.scheduler_monitor,
        )
        structural_stage_improved = (
            curriculum_mode
            and monitor_value > best_structural_stage_metrics[structural_stage]
        )
        if structural_stage_improved:
            best_structural_stage_metrics[structural_stage] = monitor_value
        curriculum_regression_from_stage_best = (
            {
                level: best_structural_stage_metrics[level]
                - scheduler_monitor_value(
                    validation_by_curriculum[level],
                    train_config.scheduler_monitor,
                )
                for level in ("simple", "medium")
                if curriculum_mode
                and math.isfinite(best_structural_stage_metrics[level])
            }
            if curriculum_mode
            else {}
        )
        scheduler.step(monitor_value)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < current_lr:
            debug(
                f"Validation plateau detected; reducing learning rate {current_lr:.6g} -> {new_lr:.6g}",
                enabled=show_progress,
            )

        validation_key = checkpoint_selection_key(validation_metrics)
        objective_values = checkpoint_objective_values(validation_metrics)
        objective_candidate_weights = {
            name: validation_weights for name in objective_values
        }
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
        improved = objective_improvements["balanced"]
        final_checkpoint_selection_key: tuple[float, ...] | None = None
        final_checkpoint_improved = improved
        if curriculum_mode and structural_stage == "complex":
            final_checkpoint_selection_key = final_curriculum_checkpoint_key(
                validation_by_curriculum["complex"],
                validation_macro,
                validation_worst_case,
                regression_within_tolerance=all(
                    regression
                    <= train_config.curriculum_regression_tolerance
                    for regression in curriculum_regression_from_stage_best.values()
                ),
            )
            final_checkpoint_improved = (
                final_checkpoint_selection_key > best_final_curriculum_key
            )
            if final_checkpoint_improved:
                best_final_curriculum_key = final_checkpoint_selection_key
        best_validation_loss = min(
            best_validation_loss,
            float(validation_metrics["loss"]),
        )
        if validation_key > best_validation_key:
            best_validation_key = validation_key
            best_validation_score = validation_score
        early_stopping_improved = (
            monitor_value > best_early_stopping_metric + train_config.min_delta
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
            "structural_curriculum_stage": structural_stage,
            "curriculum_sampling_weights": dict(curriculum_state["weights"]),
            "curriculum_competence": curriculum_competence,
            "training_by_curriculum": training_by_curriculum,
            "validation_by_curriculum": validation_by_curriculum,
            "validation_macro": validation_macro,
            "validation_worst_case": validation_worst_case,
            "structural_stage_transition": stage_transition,
            "curriculum_regression_from_stage_best": (
                curriculum_regression_from_stage_best
            ),
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
            "final_checkpoint_selection_key": (
                None
                if final_checkpoint_selection_key is None
                else list(final_checkpoint_selection_key)
            ),
            "best_objectives": {
                name: dict(details) for name, details in best_objectives.items()
            },
            "is_best": final_checkpoint_improved,
            "epochs_without_improvement": epochs_without_improvement,
            "stage_acceptance": acceptance,
            "loss_weights": asdict(epoch_weights),
            "scheduled_sampling_probability": epoch_scheduled_sampling,
        }
        if stage_metrics is not None and run_stage_gate:
            row["stage_metrics"] = stage_metrics
        if resume_policy_overrides and epoch == start_epoch:
            row["resume_policy_overrides"] = {
                name: dict(values)
                for name, values in resume_policy_overrides.items()
            }
        history.append(row)
        previous_validation_by_curriculum = validation_by_curriculum
        previous_structural_stage = structural_stage
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
        if structural_stage_improved:
            stage_weight_context = (
                ema.average_parameters(model)
                if validation_weights == "ema" and ema is not None
                else nullcontext()
            )
            with stage_weight_context:
                save_checkpoint(
                    checkpoint_path=structural_checkpoint_paths[structural_stage],
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
                    checkpoint_objective=f"structural_{structural_stage}",
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loader=train_loader,
                    device=device,
                    ema=ema,
                )
        for objective, objective_improved in objective_improvements.items():
            if not objective_improved or (
                objective == "balanced"
                and curriculum_mode
                and structural_stage == "complex"
            ):
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
                if objective == "balanced" and not (
                    curriculum_mode and structural_stage == "complex"
                ):
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
                        checkpoint_objective="balanced",
                        optimizer=optimizer,
                        scheduler=scheduler,
                        train_loader=train_loader,
                        device=device,
                        ema=ema,
                    )
        if (
            curriculum_mode
            and structural_stage == "complex"
            and final_checkpoint_improved
        ):
            final_weight_context = (
                ema.average_parameters(model)
                if validation_weights == "ema" and ema is not None
                else nullcontext()
            )
            with final_weight_context:
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
                    checkpoint_objective="balanced_robust_curriculum",
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loader=train_loader,
                    device=device,
                    ema=ema,
                )
                # Keep the historical .best alias byte-for-byte equivalent in
                # selection semantics to the named balanced checkpoint.
                save_checkpoint(
                    checkpoint_path=objective_checkpoint_paths["balanced"],
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
                    checkpoint_objective="balanced_robust_curriculum",
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loader=train_loader,
                    device=device,
                    ema=ema,
                )
        append_metrics_csv(metrics_csv_path, row)
        epoch_checkpoint_path, epoch_metrics_path = save_epoch_snapshot(
            checkpoint_path=checkpoint_path,
            metrics_csv_path=metrics_csv_path,
            history=history,
            epoch=epoch,
        )
        debug(
            f"Epoch {epoch} checkpoint saved to {checkpoint_path}; "
            f"epoch snapshot: {epoch_checkpoint_path.parent}; "
            f"metrics: {epoch_metrics_path.name}; "
            f"best checkpoint: "
            f"{best_checkpoint_path if final_checkpoint_improved else 'unchanged'} "
            f"({elapsed:.1f}s)",
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
            and (
                not curriculum_mode
                or (
                    structural_stage == "complex"
                    and epoch - complex_stage_start_epoch + 1
                    >= train_config.min_complex_stage_epochs
                )
            )
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
            f"Restored selected comparison weights from {best_checkpoint_path} before return.",
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
        components = balanced_validation_components(metrics)
        retrieval = metrics.get("cross_modal_retrieval", {})
        retrieval_rows = (
            list(retrieval.values()) if isinstance(retrieval, dict) else []
        )
        exact_retrieval = _mean_scores(
            [
                float(row["top1_accuracy"])
                for row in retrieval_rows
                if isinstance(row, dict)
                and isinstance(row.get("top1_accuracy"), (int, float))
            ]
        )
        partial_retrieval = _mean_scores(
            [
                float(row["partial_order_recall_at_1"])
                for row in retrieval_rows
                if isinstance(row, dict)
                and isinstance(row.get("partial_order_recall_at_1"), (int, float))
            ]
        )
        equivalence = metrics.get("equivalence_families", {})
        equivalence_methods = (
            equivalence.get("methods", {})
            if isinstance(equivalence, dict)
            else {}
        )
        fused_equivalence = (
            equivalence_methods.get("proc_rosetta_fused_mu", {})
            if isinstance(equivalence_methods, dict)
            else {}
        )
        family_margin = float(
            fused_equivalence.get("equivalence_margin", -2.0)
        ) if isinstance(fused_equivalence, dict) else -2.0
        family_top1 = float(
            fused_equivalence.get("behavior_id_retrieval_top1", 0.0)
        ) if isinstance(fused_equivalence, dict) else 0.0
        raw_decode = metrics.get(
            "deployment_decode_quality",
            metrics.get("decode_quality", {}),
        )
        raw_methods = raw_decode.get("methods", {}) if isinstance(raw_decode, dict) else {}
        termination = _mean_scores(
            [
                float(row["terminated_rate"])
                for row in raw_methods.values()
                if isinstance(row, dict)
                and isinstance(row.get("terminated_rate"), (int, float))
            ]
        ) if isinstance(raw_methods, dict) else 0.0
        validity = _mean_scores(
            [
                float(row["valid_tree_rate"])
                for row in raw_methods.values()
                if isinstance(row, dict)
                and isinstance(row.get("valid_tree_rate"), (int, float))
            ]
        ) if isinstance(raw_methods, dict) else 0.0
        observation = metrics.get("metrics_by_observation_quality", {})
        observation_scores = [
            float(row.get("canonical_exact_rate", 0.0))
            for row in observation.values()
            if isinstance(row, dict)
        ] if isinstance(observation, dict) else []
        worst_observation_regression = (
            max(observation_scores) - min(observation_scores)
            if observation_scores
            else 1.0
        )
        measurements = {
            "fused_behavior_geometry": float(components["geometry_score"]),
            "fused_geometry_advantage": float(
                components["fused_geometry_advantage"]
            ),
            "exact_retrieval": exact_retrieval,
            "partial_order_retrieval": partial_retrieval,
            "family_margin": family_margin,
            "family_top1": family_top1,
            "raw_termination": termination,
            "raw_validity": validity,
            "worst_observation_regression": worst_observation_regression,
        }
        checks = {
            "fused_behavior_geometry_finite": math.isfinite(
                measurements["fused_behavior_geometry"]
            ),
            "nearest_neighbor_geometry_not_worse_than_all_baselines": (
                measurements["fused_geometry_advantage"] >= 0.0
            ),
            "exact_retrieval_nonzero": exact_retrieval > 0.0,
            "partial_order_retrieval_nonzero": partial_retrieval > 0.0,
            "family_margin_positive": family_margin > 0.0,
            "family_top1_nonzero": family_top1 > 0.0,
            "raw_termination_gate": termination >= 0.95,
            "raw_validity_gate": validity >= 0.95,
            "worst_observation_regression_bounded": (
                worst_observation_regression <= 0.10
            ),
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
    curriculum: str | None = None,
) -> dict[str, float]:
    dataset = JsonlProcessDataset(
        split_samples_path(data_dir, split, curriculum),
        show_progress=show_progress,
    )
    return evaluate_samples_from_checkpoint(
        checkpoint_path=checkpoint_path,
        samples=dataset.samples,
        batch_size=batch_size,
        device=device,
        show_progress=show_progress,
        progress_desc=f"{split.title()} loss",
    )


def evaluate_samples_from_checkpoint(
    checkpoint_path: str | Path,
    samples,
    batch_size: int = 16,
    device: str | None = None,
    show_progress: bool = False,
    progress_desc: str = "Selected samples loss",
) -> dict[str, float]:
    """Evaluate the exact in-memory rows selected by the caller."""

    torch_device = resolve_device(device)
    model, checkpoint = load_checkpoint(checkpoint_path, torch_device)
    checkpoint_train_config = train_config_from_checkpoint(checkpoint, torch_device)
    loader = DataLoader(
        list(samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ProcessBatchCollator(
            model.tree_tokenizer,
            model.activity_tokenizer,
        ),
    )
    return evaluate_epoch(
        model,
        loader,
        torch_device,
        weights=loss_weights_from_checkpoint(checkpoint, checkpoint_train_config),
        show_progress=show_progress,
        progress_desc=progress_desc,
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


def metric_gaps(
    training_metrics: dict[str, object],
    validation_metrics: dict[str, object],
) -> dict[str, float]:
    keys = sorted(set(training_metrics) & set(validation_metrics))
    return {
        key: float(validation_metrics[key]) - float(training_metrics[key])
        for key in keys
        if isinstance(training_metrics[key], (int, float))
        and isinstance(validation_metrics[key], (int, float))
    }


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
    metrics: dict[str, object],
) -> dict[str, float | tuple[float, ...]]:
    components = balanced_validation_components(metrics)
    return {
        "balanced": checkpoint_selection_key(metrics),
        "decode": float(components["decode_score"]),
        "retrieval": float(components["retrieval_score"]),
        "geometry": float(components["geometry_score"]),
        "discovery": float(components["discovery_score"]),
    }


def objective_is_better(
    objective: str,
    value: float | tuple[float, ...],
    best: float | tuple[float, ...],
) -> bool:
    if isinstance(value, tuple) or isinstance(best, tuple):
        return tuple(value) > tuple(best)
    return float(value) > float(best)


def best_objectives_from_history(
    history: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    best: dict[str, dict[str, object]] = {
        "balanced": {"value": (float("-inf"),) * 4, "epoch": 0},
        "decode": {"value": float("-inf"), "epoch": 0},
        "retrieval": {"value": float("-inf"), "epoch": 0},
        "geometry": {"value": float("-inf"), "epoch": 0},
        "discovery": {"value": float("-inf"), "epoch": 0},
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
                if objective == "balanced" and isinstance(value, list):
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
    "fused_to_tree",
    "fused_subset_to_tree",
    "deployment_to_tree",
    "tree_complexity",
    "duplicate_activity",
    "latent_alignment",
    "contrastive",
    "exact_contrastive",
    "within_modality_contrastive",
    "semantic_exact_contrastive",
    "semantic_memory_contrastive",
    "hierarchical_metric",
    "observation_view_consistency",
    "soft_behavior_geometry",
    "observed_behavior_regression",
    "observed_behavior_ranking",
    "beam_minimum_risk",
    "eos_calibration",
    "generated_length",
    "unresolved_open_slots",
    "completion_feasibility",
    "variance",
    "covariance",
    "effective_rank",
    "kl",
    "trace_canonical_exact",
    "ordinary_trace_canonical_exact",
    "trace_normalized_tree_edit",
    "trace_decoded_mean_size",
    "trace_decoded_mean_depth",
    "trace_decoded_mean_duplicate_count",
    "trace_decoded_duplicate_rate",
    "trace_decoded_mean_size_delta",
    "trace_decoded_mean_depth_delta",
    "trace_decoded_mean_duplicate_count_delta",
    "exact_behavior_recall_at_1",
    "behavior_distance_spearman",
    "false_negative_rate",
    "checkpoint_selection_primary_exact",
    "checkpoint_selection_edit_score",
    "checkpoint_selection_recall_at_1",
    "checkpoint_selection_spearman",
    "checkpoint_selection_score",
    "beam_top1_exact",
    "beam_oracle_exact",
    "beam_top1_edit",
    "beam_oracle_edit",
    "beam_top1_behavior_l1",
    "beam_oracle_behavior_l1",
    "pcgrad_reconstruction_metric_cosine",
    "pcgrad_projection_applied",
    *(
        f"{kind}_{modality}"
        for modality in ("tree", "trace", "petri")
        for kind in (
            "reconstruction_gradient_norm",
            "metric_gradient_norm",
            "source_reconstruction_gradient_norm",
            "fused_reconstruction_gradient_norm",
            "semantic_retrieval_gradient_norm",
            "observed_geometry_gradient_norm",
            "beam_risk_gradient_norm",
            "metric_to_reconstruction_gradient_ratio",
            "reconstruction_exact_gradient_cosine",
            "reconstruction_soft_geometry_gradient_cosine",
            "exact_soft_geometry_gradient_cosine",
            "reconstruction_semantic_gradient_cosine",
            "reconstruction_observed_geometry_gradient_cosine",
            "semantic_observed_geometry_gradient_cosine",
            "reconstruction_beam_risk_gradient_cosine",
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
        "scheduled_sampling_probability",
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
        "scheduled_sampling_probability": row.get(
            "scheduled_sampling_probability", ""
        ),
    }
    for name in CSV_METRIC_NAMES:
        flat[f"training_{name}"] = training.get(name, "")
        flat[f"validation_{name}"] = validation.get(name, "")
        flat[f"gap_{name}"] = gap.get(name, "")
    return flat


def epoch_snapshot_directory(
    checkpoint_path: str | Path,
    epoch: int,
) -> Path:
    """Return the stable sibling directory used for one epoch snapshot."""

    if epoch < 1:
        raise ValueError("epoch snapshots require epoch >= 1")
    return Path(checkpoint_path).parent / f"{epoch:05d}"


def save_epoch_snapshot(
    checkpoint_path: str | Path,
    metrics_csv_path: str | Path,
    history: list[dict[str, object]],
    epoch: int,
) -> tuple[Path, Path]:
    """Atomically archive the resumable checkpoint and matching epoch metrics."""

    source_checkpoint = Path(checkpoint_path)
    if not source_checkpoint.is_file():
        raise FileNotFoundError(
            f"latest checkpoint does not exist for epoch snapshot: {source_checkpoint}"
        )
    snapshot_directory = epoch_snapshot_directory(source_checkpoint, epoch)
    snapshot_directory.mkdir(parents=True, exist_ok=True)
    snapshot_checkpoint = snapshot_directory / source_checkpoint.name
    snapshot_metrics = snapshot_directory / Path(metrics_csv_path).name
    temporary_checkpoint = snapshot_checkpoint.with_name(
        f".{snapshot_checkpoint.name}.tmp"
    )

    try:
        shutil.copy2(source_checkpoint, temporary_checkpoint)
        write_metrics_csv(snapshot_metrics, history)
        temporary_checkpoint.replace(snapshot_checkpoint)
    finally:
        if temporary_checkpoint.exists():
            temporary_checkpoint.unlink()
    return snapshot_checkpoint, snapshot_metrics


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
        "structural_curriculum_state": (
            {
                "stage": history[-1].get("structural_curriculum_stage"),
                "sampling_weights": history[-1].get("curriculum_sampling_weights", {}),
            }
            if history
            else None
        ),
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
    if int(checkpoint.get("version", 0)) < 7:
        # Format-v7 unifies scheduling, stopping, EMA comparison, and default
        # checkpoint selection on the balanced validation objective.
        train_config_data["scheduler_monitor"] = "balanced"
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


def capture_training_loader_state(train_loader) -> dict[str, object]:
    if isinstance(train_loader, CurriculumMixtureLoader):
        return {
            "curriculum_loaders": {
                level: capture_training_loader_state(loader)
                for level, loader in train_loader.loaders.items()
            }
        }
    state: dict[str, object] = {}
    batch_sampler = getattr(train_loader, "batch_sampler", None)
    if hasattr(batch_sampler, "epoch"):
        state["batch_sampler_epoch"] = int(batch_sampler.epoch)
    collator_rng = getattr(getattr(train_loader, "collate_fn", None), "rng", None)
    if isinstance(collator_rng, random.Random):
        state["collator_rng"] = collator_rng.getstate()
    return state


def restore_training_loader_state(
    train_loader, state: dict[str, object], completed_epoch: int
) -> bool:
    if isinstance(train_loader, CurriculumMixtureLoader):
        stored = state.get("curriculum_loaders", {})
        if not isinstance(stored, dict):
            return False
        return all(
            isinstance(stored.get(level), dict)
            and restore_training_loader_state(
                loader,
                stored[level],
                completed_epoch,
            )
            for level, loader in train_loader.loaders.items()
        )
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
    scheduler_monitor: str = "balanced",
    *,
    use_stored_metric: bool = True,
) -> None:
    for row in history:
        validation = row.get("validation")
        if isinstance(validation, dict):
            scheduler.step(
                float(
                    (
                        row.get(
                            "lr_scheduler_metric",
                            scheduler_monitor_value(validation, scheduler_monitor),
                        )
                        if use_stored_metric
                        else scheduler_monitor_value(validation, scheduler_monitor)
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
        replay_scheduler_history(
            scheduler,
            history,
            scheduler_monitor,
            use_stored_metric=int(checkpoint.get("version", 0)) >= 7,
        )
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
        replay_scheduler_history(
            scheduler,
            history,
            scheduler_monitor,
            use_stored_metric=int(checkpoint.get("version", 0)) >= 7,
        )
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
