from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from proc_rosetta.benchmarks import (
    CONFORMANCE_METHODS,
    Pm4pyPetriEmbeddingConfig,
    fixed_test_sample_subset,
    format_human_test_report,
    rich_test_report,
)
from proc_rosetta.data import SplitCounts, recreate_data_splits
from proc_rosetta.data import read_samples_jsonl, split_samples_path
from proc_rosetta.devices import default_device
from proc_rosetta.synthetic import (
    CURRICULUM_LEVELS,
    DEFAULT_COMPLEXITY_PROFILE,
    DEFAULT_MAX_ACTIVITIES,
    SyntheticConfig,
)
from proc_rosetta.training import (
    TrainConfig,
    best_checkpoint_for,
    train_from_data_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="proc-rosetta")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="recreate synthetic data/training, validation, and test splits")
    sample.add_argument("--data-dir", default="data")
    sample.add_argument("--count", type=int, default=None, help="legacy alias for --train-count")
    sample.add_argument(
        "--train-count",
        type=int,
        default=None,
        help="Deprecated flattened-row count; prefer --train-families.",
    )
    sample.add_argument(
        "--validation-count",
        type=int,
        default=None,
        help="Deprecated flattened-row count; prefer --validation-families.",
    )
    sample.add_argument(
        "--test-count",
        type=int,
        default=None,
        help="Deprecated flattened-row count; prefer --test-families.",
    )
    sample.add_argument("--train-families", type=int, default=None)
    sample.add_argument("--validation-families", type=int, default=None)
    sample.add_argument("--test-families", type=int, default=None)
    sample.add_argument("--seed", type=int, default=13)
    sample.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_COMPLEXITY_PROFILE.max_depth,
    )
    sample.add_argument(
        "--activity-vocab-size",
        "--max-activities",
        dest="max_activities",
        type=int,
        default=DEFAULT_MAX_ACTIVITIES,
        help="Global tokenizer/model activity capacity shared by all curricula.",
    )
    sample.add_argument(
        "--min-activities",
        type=int,
        default=DEFAULT_COMPLEXITY_PROFILE.min_generated_activities,
    )
    sample.add_argument("--leaf-probability", type=float, default=0.55)
    sample.add_argument(
        "--operator-probabilities",
        default=None,
        help=(
            "Comma-separated operator probabilities for non-root nodes, for example "
            "seq=0.25,xor=0.25,and=0.25,loop=0.25."
        ),
    )
    sample.add_argument(
        "--root-operator-probabilities",
        default=None,
        help=(
            "Comma-separated root-operator probabilities; defaults to "
            "seq=0.7,xor=0.1,and=0.1,loop=0.1."
        ),
    )
    sample.add_argument("--max-arity", type=int, default=3)
    sample.add_argument("--traces-per-sample", type=int, default=128)
    sample.add_argument("--max-trace-length", type=int, default=128)
    sample.add_argument("--curriculum-phase", type=int, default=3)
    sample.add_argument(
        "--generator",
        choices=["behavior_families", "isolated"],
        default="behavior_families",
        help="Generate grouped equivalent representations (default) or legacy isolated triples.",
    )
    sample.add_argument(
        "--generator-config",
        type=Path,
        default=None,
        help="Load nested generator JSON; explicit legacy structure flags still override it.",
    )
    sample.add_argument(
        "--preset",
        choices=[
            "stage_a_tiny_overfit",
            "stage_b_exact_alignment",
            "stage_c_behavior_geometry",
            "stage_d_observation_curriculum",
            "smoke",
            "balanced_train",
            "iid_behavior",
            "equivalence_train",
            "equivalence_test",
            "equivalence_seen",
            "equivalence_unseen",
            "nonblock_ood",
            "scale_ood",
            "sampling_ood",
            "noise_ood",
            "loops_bounded",
        ],
        default=None,
    )
    sample.add_argument("--variants-per-behavior", type=int, default=2)
    sample.add_argument("--log-views-per-behavior", type=int, default=2)
    sample.add_argument(
        "--log-view-modes",
        default="uniform_variants,resampled",
        help=(
            "Comma-separated modes: uniform_variants,resampled,long_tail,sparse,"
            "incomplete,noisy."
        ),
    )
    sample.add_argument(
        "--motif-weights",
        default=None,
        help="Comma-separated KIND=WEIGHT values for behavior-family motifs.",
    )
    sample.add_argument(
        "--min-families-per-motif",
        type=int,
        default=None,
        help="Minimum behavior families for every positive-weight motif in every split.",
    )
    sample.add_argument(
        "--class-coverage-mode",
        choices=["strict", "best_effort"],
        default=None,
        help="Fail on infeasible class quotas (strict) or record deficits (best_effort).",
    )
    sample.add_argument(
        "--multiprocessing",
        action="store_true",
        help="generate samples in parallel with all but one available CPU core",
    )
    sample.add_argument("--quiet", action="store_true", help="disable generation progress bars")

    train = subparsers.add_parser("train", help="train the first-stage multimodal model")
    train.add_argument("--data-dir", default="data")
    train.add_argument(
        "--checkpoint",
        default="checkpoints/proc_rosetta.pt",
        help=(
            "latest checkpoint path; every completed epoch is also archived as "
            "<parent>/00001/<filename>, <parent>/00002/<filename>, and so on"
        ),
    )
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--simple-batch-size", type=int, default=None)
    train.add_argument("--medium-batch-size", type=int, default=None)
    train.add_argument("--complex-batch-size", type=int, default=None)
    train.add_argument(
        "--training-max-traces",
        type=int,
        default=32,
        help="randomly sampled traces per training log and batch",
    )
    train.add_argument("--training-max-trace-length", type=int, default=64)
    train.add_argument("--validation-max-traces", type=int, default=64)
    train.add_argument("--validation-max-trace-length", type=int, default=128)
    train.add_argument("--min-complex-stage-epochs", type=int, default=5)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--latent-dim", type=int, default=96)
    train.add_argument("--hidden-dim", type=int, default=192)
    train.add_argument(
        "--semantic-latent-mode",
        choices=["deterministic"],
        default="deterministic",
        help="Semantic content mode; supervised translation supports deterministic only.",
    )
    train.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Deprecated compatibility override that sets every modality dropout.",
    )
    train.add_argument("--tree-encoder-dropout", type=float, default=0.12)
    train.add_argument("--trace-encoder-dropout", type=float, default=0.20)
    train.add_argument("--petri-encoder-dropout", type=float, default=0.12)
    train.add_argument("--decoder-dropout", type=float, default=0.20)
    train.add_argument("--projection-dropout", type=float, default=0.20)
    train.add_argument("--weight-decay", type=float, default=5e-4)
    train.add_argument("--label-smoothing", type=float, default=0.04)
    train.add_argument("--early-stopping-patience", type=int, default=6)
    train.add_argument("--min-delta", type=float, default=0.005)
    train.add_argument("--lr-patience", type=int, default=1)
    train.add_argument("--lr-factor", type=float, default=0.5)
    train.add_argument("--min-lr", type=float, default=1e-5)
    train.add_argument("--metrics-csv", default="checkpoints/training_metrics.csv")
    train.add_argument(
        "--resume",
        action="store_true",
        help=(
            "continue from --checkpoint; --epochs is the total target epoch count, "
            "not the number of additional epochs"
        ),
    )
    train.add_argument("--stage-gate-interval", type=int, default=5)
    train.add_argument(
        "--gradient-diagnostics-interval",
        type=int,
        default=0,
        help="epochs between expensive gradient diagnostics; disabled by default",
    )
    train.add_argument(
        "--use-pcgrad",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use three-pass PCGrad conflict projection instead of standard backpropagation",
    )
    train.add_argument("--seed", type=int, default=13)
    train.add_argument("--loader-num-workers", type=int, default=0)
    train.add_argument(
        "--loader-pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument(
        "--loader-persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument("--loader-prefetch-factor", type=int, default=2)
    train.add_argument(
        "--device",
        default=default_device(),
        help="Torch device; defaults to cuda or mps when available, otherwise cpu.",
    )
    train.add_argument("--quiet", action="store_true", help="disable stderr debug messages and progress bars")
    train.add_argument("--views-per-family", type=int, default=2)
    train.add_argument(
        "--activity-remap-probability",
        type=float,
        default=0.5,
        help=(
            "Probability of consistently renaming activities within each training "
            "family; preserves behavior while discouraging label memorization."
        ),
    )
    train.add_argument("--memory-tokens", type=int, default=6)
    train.add_argument("--decoder-layers", type=int, default=3)
    train.add_argument("--tree-encoder-layers", type=int, default=3)
    train.add_argument("--trace-event-layers", type=int, default=1)
    train.add_argument("--trace-set-layers", type=int, default=1)
    train.add_argument("--petri-message-passing-steps", type=int, default=5)
    train.add_argument("--decoder-input-dropout", type=float, default=0.15)
    train.add_argument(
        "--scheduled-sampling-max",
        type=float,
        default=0.20,
        help="Maximum scheduled-sampling probability; set to 0 to disable it.",
    )
    train.add_argument("--scheduled-sampling-start-epoch", type=int, default=15)
    train.add_argument("--scheduled-sampling-ramp-epochs", type=int, default=20)
    train.add_argument("--gradient-clip-norm", type=float, default=5.0)
    train.add_argument("--tree-reconstruction-weight", type=float, default=1.0)
    train.add_argument("--trace-to-tree-weight", type=float, default=1.0)
    train.add_argument("--petri-to-tree-weight", type=float, default=1.0)
    train.add_argument("--fused-to-tree-weight", type=float, default=1.0)
    train.add_argument("--fused-subset-to-tree-weight", type=float, default=0.25)
    train.add_argument("--deployment-to-tree-weight", type=float, default=0.25)
    train.add_argument(
        "--modality-subset-fusion-probability",
        type=float,
        default=0.5,
    )
    train.add_argument(
        "--deployment-policy-probability",
        type=float,
        default=0.25,
    )
    train.add_argument("--exact-contrastive-weight", type=float, default=0.30)
    train.add_argument("--within-modality-contrastive-weight", type=float, default=0.25)
    train.add_argument("--semantic-exact-contrastive-weight", type=float, default=0.15)
    train.add_argument("--semantic-memory-contrastive-weight", type=float, default=0.10)
    train.add_argument("--hierarchical-metric-weight", type=float, default=0.15)
    train.add_argument("--observation-view-consistency-weight", type=float, default=0.15)
    train.add_argument("--semantic-memory-bank-size", type=int, default=4096)
    train.add_argument("--soft-behavior-geometry-weight", type=float, default=0.25)
    train.add_argument("--observed-behavior-regression-weight", type=float, default=0.20)
    train.add_argument("--observed-behavior-ranking-weight", type=float, default=0.20)
    train.add_argument("--beam-minimum-risk-weight", type=float, default=0.05)
    train.add_argument("--eos-calibration-weight", type=float, default=0.05)
    train.add_argument("--generated-length-weight", type=float, default=0.05)
    train.add_argument("--unresolved-open-slots-weight", type=float, default=0.05)
    train.add_argument("--completion-feasibility-weight", type=float, default=0.05)
    train.add_argument("--beam-risk-start-epoch", type=int, default=15)
    train.add_argument("--beam-risk-batch-probability", type=float, default=0.10)
    train.add_argument("--beam-risk-size", type=int, default=5)
    train.add_argument("--beam-risk-max-decode-length", type=int, default=128)
    train.add_argument("--variance-weight", type=float, default=0.1)
    train.add_argument("--covariance-weight", type=float, default=0.01)
    train.add_argument("--latent-alignment-weight", type=float, default=0.075)
    train.add_argument("--tree-complexity-weight", type=float, default=0.0)
    train.add_argument("--duplicate-activity-weight", type=float, default=0.0)
    train.add_argument("--contrastive-temperature", type=float, default=0.3)
    train.add_argument("--behavior-temperature", type=float, default=0.2)
    train.add_argument("--latent-temperature", type=float, default=0.2)
    train.add_argument("--exact-contrastive-start-epoch", type=int, default=3)
    train.add_argument("--exact-contrastive-ramp-epochs", type=int, default=4)
    train.add_argument("--soft-geometry-start-epoch", type=int, default=5)
    train.add_argument("--soft-geometry-ramp-epochs", type=int, default=6)
    train.add_argument(
        "--structure-regularization-start-epoch",
        type=int,
        default=5,
    )
    train.add_argument(
        "--structure-regularization-ramp-epochs",
        type=int,
        default=5,
    )
    train.add_argument(
        "--scheduler-monitor",
        choices=["balanced"],
        default="balanced",
        help="Test-aligned balanced validation score used for scheduling and stopping.",
    )
    train.add_argument(
        "--restore-best-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore the selected comparison checkpoint before returning (default: enabled).",
    )
    train.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate an exponential moving average of training weights (default: enabled).",
    )
    train.add_argument("--ema-start-epoch", type=int, default=3)
    train.add_argument("--ema-decay", type=float, default=0.995)
    train.add_argument(
        "--training-stage",
        choices=["a", "b", "c", "d", "full"],
        default="full",
        help="Gate objectives according to the staged remediation sequence.",
    )
    train.add_argument(
        "--no-group-aware-batches",
        action="store_true",
        help="Disable family-grouped batches (multi-positive loss remains ID-aware).",
    )
    train.add_argument("--validation-decode-interval", type=int, default=2)
    train.add_argument("--validation-full-interval", type=int, default=10)
    train.add_argument("--validation-decode-family-count", type=int, default=32)
    train.add_argument("--validation-discovery-family-count", type=int, default=16)
    train.add_argument("--validation-beam-size", type=int, default=5)
    train.add_argument("--validation-max-decode-length", type=int, default=128)

    test = subparsers.add_parser("test", help="load a checkpoint and evaluate the persisted test split")
    test.add_argument("--data-dir", default="data")
    test.add_argument(
        "--curriculum",
        required=True,
        choices=CURRICULUM_LEVELS,
        help="Evaluate exactly one structural curriculum.",
    )
    test.add_argument("--checkpoint", default="checkpoints/proc_rosetta.pt")
    test.add_argument(
        "--checkpoint-selection",
        choices=[
            "best",
            "latest",
            "best_balanced",
            "best_decode",
            "best_retrieval",
            "best_geometry",
            "best_discovery",
        ],
        default="best",
        help="Evaluate the selected comparison checkpoint by default, or the latest epoch.",
    )
    test.add_argument("--batch-size", type=int, default=16)
    test.add_argument("--max-decode-length", type=int, default=512)
    test.add_argument(
        "--max-samples",
        type=int,
        default=64,
        help=(
            "Maximum family-complete test subset size (default: 64); "
            "use 0 for the entire persisted split."
        ),
    )
    test.add_argument(
        "--beam-size",
        type=int,
        default=2,
        help="Decode beam width for routine evaluation (default: 2).",
    )
    test.add_argument(
        "--decode-behavior-traces",
        type=int,
        default=16,
        help="Simulated traces per decode used for behavior metrics (default: 16).",
    )
    test.add_argument(
        "--device",
        default=default_device(),
        help="Torch device; defaults to cuda or mps when available, otherwise cpu.",
    )
    test.add_argument("--skip-pm4py-petri-embedding", action="store_true")
    test.add_argument("--petri-embedding-dim", type=int, default=256)
    test.add_argument("--petri-num-walks", type=int, default=5)
    test.add_argument("--petri-walk-length", type=int, default=20)
    test.add_argument("--petri-window", type=int, default=5)
    test.add_argument("--petri-epochs", type=int, default=5)
    test.add_argument("--petri-seed", type=int, default=42)
    test.add_argument(
        "--conformance-method",
        choices=CONFORMANCE_METHODS,
        default="token_based_replay",
        help=(
            "Discovery fitness/precision method: token replay on converted Petri nets "
            "(default), or footprints computed directly on process trees."
        ),
    )
    test.add_argument("--json", action="store_true", help="print the full machine-readable JSON report")
    test.add_argument(
        "--quiet",
        action="store_true",
        help="disable stderr status messages and progress bars",
    )

    return parser


def synthetic_config_from_args(args: argparse.Namespace) -> SyntheticConfig:
    if args.generator_config is not None:
        with args.generator_config.open("r", encoding="utf-8") as handle:
            config = SyntheticConfig.from_dict(json.load(handle))
    elif args.preset is not None:
        config = SyntheticConfig.preset(args.preset)
    else:
        config = SyntheticConfig()
    configured = args.generator_config is not None or args.preset is not None
    motif_weights = (
        _parse_weights(args.motif_weights)
        if args.motif_weights is not None
        else config.motif_weights
    )
    overrides: dict[str, object] = {"motif_weights": motif_weights}
    if args.operator_probabilities is not None:
        overrides["operator_probabilities"] = _parse_operator_probabilities(
            args.operator_probabilities
        )
    if args.root_operator_probabilities is not None:
        overrides["root_operator_probabilities"] = _parse_operator_probabilities(
            args.root_operator_probabilities
        )
    if args.min_families_per_motif is not None:
        if args.min_families_per_motif < 0:
            raise ValueError("min-families-per-motif must be non-negative")
        overrides["min_families_per_motif"] = {
            split: args.min_families_per_motif
            for split in ("training", "validation", "test")
        }
    if args.class_coverage_mode is not None:
        overrides["class_coverage_mode"] = args.class_coverage_mode
    defaults = {
        "max_depth": DEFAULT_COMPLEXITY_PROFILE.max_depth,
        "max_activities": DEFAULT_MAX_ACTIVITIES,
        "min_activities": DEFAULT_COMPLEXITY_PROFILE.min_generated_activities,
        "leaf_probability": 0.55,
        "max_arity": 3,
        "traces_per_sample": 128,
        "max_trace_length": 128,
        "curriculum_phase": 3,
        "generator": "behavior_families",
        "variants_per_behavior": 2,
        "log_views_per_behavior": 2,
        "log_view_modes": "uniform_variants,resampled",
    }
    for field_name in (
        "max_depth",
        "max_activities",
        "min_activities",
        "leaf_probability",
        "max_arity",
        "traces_per_sample",
        "max_trace_length",
        "curriculum_phase",
        "generator",
        "variants_per_behavior",
        "log_views_per_behavior",
    ):
        value = getattr(args, field_name)
        if not configured or value != defaults[field_name]:
            overrides[field_name] = value
    if "max_activities" in overrides:
        overrides["max_generated_activities"] = min(
            int(overrides["max_activities"]),
            int(config.max_generated_activities or config.max_activities),
        )
        overrides["min_activities"] = min(
            int(overrides.get("min_activities", config.min_activities)),
            int(overrides["max_generated_activities"]),
        )
    if not configured or args.log_view_modes != defaults["log_view_modes"]:
        overrides["log_view_modes"] = tuple(
            value.strip() for value in args.log_view_modes.split(",") if value.strip()
        )
    overrides["variants_per_behavior"] = max(
        1, int(overrides.get("variants_per_behavior", config.variants_per_behavior))
    )
    overrides["log_views_per_behavior"] = max(
        1, int(overrides.get("log_views_per_behavior", config.log_views_per_behavior))
    )
    return replace(config, **overrides)


def run_sample(args: argparse.Namespace) -> int:
    config = synthetic_config_from_args(args)
    if any(
        value is not None
        for value in (args.count, args.train_count, args.validation_count, args.test_count)
    ):
        print(
            "[sample] WARNING: flattened row-count flags are deprecated; use "
            "--train-families/--validation-families/--test-families so behavior "
            "diversity is explicit.",
            file=sys.stderr,
        )
    metadata = recreate_data_splits(
        data_dir=args.data_dir,
        counts=split_counts_from_args(args, config=config),
        config=config,
        seed=args.seed,
        show_progress=not args.quiet,
        use_multiprocessing=args.multiprocessing,
    )
    print(json.dumps(metadata, sort_keys=True))
    return 0


def run_train(args: argparse.Namespace) -> int:
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        simple_batch_size=args.simple_batch_size,
        medium_batch_size=args.medium_batch_size,
        complex_batch_size=args.complex_batch_size,
        min_complex_stage_epochs=max(0, args.min_complex_stage_epochs),
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        semantic_latent_mode=args.semantic_latent_mode,
        dropout=args.dropout,
        tree_encoder_dropout=args.tree_encoder_dropout,
        trace_encoder_dropout=args.trace_encoder_dropout,
        petri_encoder_dropout=args.petri_encoder_dropout,
        decoder_dropout=args.decoder_dropout,
        projection_dropout=args.projection_dropout,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        min_lr=args.min_lr,
        seed=args.seed,
        device=args.device,
        group_aware_batches=not args.no_group_aware_batches,
        views_per_family=max(1, args.views_per_family),
        activity_remap_probability=args.activity_remap_probability,
        memory_tokens=args.memory_tokens,
        decoder_layers=args.decoder_layers,
        tree_encoder_layers=args.tree_encoder_layers,
        trace_event_layers=args.trace_event_layers,
        trace_set_layers=args.trace_set_layers,
        petri_message_passing_steps=args.petri_message_passing_steps,
        decoder_input_dropout=args.decoder_input_dropout,
        scheduled_sampling_max=args.scheduled_sampling_max,
        scheduled_sampling_start_epoch=args.scheduled_sampling_start_epoch,
        scheduled_sampling_ramp_epochs=args.scheduled_sampling_ramp_epochs,
        gradient_clip_norm=args.gradient_clip_norm,
        tree_reconstruction_weight=args.tree_reconstruction_weight,
        trace_to_tree_weight=args.trace_to_tree_weight,
        petri_to_tree_weight=args.petri_to_tree_weight,
        fused_to_tree_weight=args.fused_to_tree_weight,
        fused_subset_to_tree_weight=args.fused_subset_to_tree_weight,
        deployment_to_tree_weight=args.deployment_to_tree_weight,
        modality_subset_fusion_probability=(
            args.modality_subset_fusion_probability
        ),
        deployment_policy_probability=args.deployment_policy_probability,
        exact_contrastive_weight=args.exact_contrastive_weight,
        within_modality_contrastive_weight=args.within_modality_contrastive_weight,
        semantic_exact_contrastive_weight=args.semantic_exact_contrastive_weight,
        semantic_memory_contrastive_weight=args.semantic_memory_contrastive_weight,
        hierarchical_metric_weight=args.hierarchical_metric_weight,
        observation_view_consistency_weight=(
            args.observation_view_consistency_weight
        ),
        semantic_memory_bank_size=args.semantic_memory_bank_size,
        soft_behavior_geometry_weight=args.soft_behavior_geometry_weight,
        observed_behavior_regression_weight=(
            args.observed_behavior_regression_weight
        ),
        observed_behavior_ranking_weight=args.observed_behavior_ranking_weight,
        beam_minimum_risk_weight=args.beam_minimum_risk_weight,
        eos_calibration_weight=args.eos_calibration_weight,
        generated_length_weight=args.generated_length_weight,
        unresolved_open_slots_weight=args.unresolved_open_slots_weight,
        completion_feasibility_weight=args.completion_feasibility_weight,
        beam_risk_start_epoch=args.beam_risk_start_epoch,
        beam_risk_batch_probability=args.beam_risk_batch_probability,
        beam_risk_size=args.beam_risk_size,
        beam_risk_max_decode_length=args.beam_risk_max_decode_length,
        variance_weight=args.variance_weight,
        covariance_weight=args.covariance_weight,
        latent_alignment_weight=args.latent_alignment_weight,
        tree_complexity_weight=args.tree_complexity_weight,
        duplicate_activity_weight=args.duplicate_activity_weight,
        contrastive_temperature=args.contrastive_temperature,
        behavior_temperature=args.behavior_temperature,
        latent_temperature=args.latent_temperature,
        exact_contrastive_start_epoch=args.exact_contrastive_start_epoch,
        exact_contrastive_ramp_epochs=args.exact_contrastive_ramp_epochs,
        soft_geometry_start_epoch=args.soft_geometry_start_epoch,
        soft_geometry_ramp_epochs=args.soft_geometry_ramp_epochs,
        structure_regularization_start_epoch=(
            args.structure_regularization_start_epoch
        ),
        structure_regularization_ramp_epochs=(
            args.structure_regularization_ramp_epochs
        ),
        scheduler_monitor=args.scheduler_monitor,
        restore_best_weights=args.restore_best_weights,
        use_ema=args.use_ema,
        ema_start_epoch=args.ema_start_epoch,
        ema_decay=args.ema_decay,
        training_stage=args.training_stage,
        stage_gate_interval=max(1, args.stage_gate_interval),
        gradient_diagnostics_interval=args.gradient_diagnostics_interval,
        use_pcgrad=args.use_pcgrad,
        validation_decode_interval=args.validation_decode_interval,
        validation_full_interval=args.validation_full_interval,
        validation_decode_family_count=args.validation_decode_family_count,
        validation_discovery_family_count=(
            args.validation_discovery_family_count
        ),
        validation_beam_size=args.validation_beam_size,
        validation_max_decode_length=args.validation_max_decode_length,
        training_max_traces=args.training_max_traces,
        training_max_trace_length=args.training_max_trace_length,
        validation_max_traces=args.validation_max_traces,
        validation_max_trace_length=args.validation_max_trace_length,
        loader_num_workers=max(0, args.loader_num_workers),
        loader_pin_memory=args.loader_pin_memory,
        loader_persistent_workers=args.loader_persistent_workers,
        loader_prefetch_factor=max(1, args.loader_prefetch_factor),
    )
    _, history = train_from_data_dir(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        train_config=train_config,
        show_progress=not args.quiet,
        metrics_csv_path=args.metrics_csv,
        resume=args.resume,
    )
    for row in history:
        print(json.dumps(round_nested_metrics(row), sort_keys=True))
    return 0


def run_test(args: argparse.Namespace) -> int:
    if args.max_decode_length < 3:
        raise ValueError("--max-decode-length must be at least 3")
    if args.max_samples < 0:
        raise ValueError("--max-samples cannot be negative")
    if args.beam_size < 1:
        raise ValueError("--beam-size must be positive")
    if args.decode_behavior_traces < 1:
        raise ValueError("--decode-behavior-traces must be positive")
    show_progress = not args.quiet
    checkpoint_path = checkpoint_for_selection(
        args.checkpoint,
        args.checkpoint_selection,
    )
    sample_path = split_samples_path(args.data_dir, "test", args.curriculum)
    test_debug(f"Loading test samples from {sample_path}", enabled=show_progress)
    samples = read_samples_jsonl(sample_path, show_progress=show_progress)
    mismatches = [
        index
        for index, sample in enumerate(samples)
        if getattr(sample, "complexity_level", args.curriculum) != args.curriculum
    ]
    if mismatches:
        raise ValueError(
            f"test split {sample_path} contains {len(mismatches)} rows outside requested "
            f"curriculum {args.curriculum!r}; first mismatch at row {mismatches[0] + 1}"
        )
    total_sample_count = len(samples)
    samples = fixed_test_sample_subset(samples, args.max_samples)
    sample_count = len(samples)
    sample_label = "sample" if sample_count == 1 else "samples"
    conformance_label = (
        "token-replay" if args.conformance_method == "token_based_replay" else "footprint"
    )
    test_debug(
        f"Plan: {sample_count} {sample_label} selected from {total_sample_count}, "
        f"batch_size={args.batch_size}, device={args.device}, beam_size={args.beam_size}; "
        f"{2 * sample_count} {conformance_label} conformance evaluations, "
        f"{8 * sample_count} matched raw/deployment decodes, "
        f"{sample_count * (sample_count - 1) // 2} behavioral pairs, "
        f"{'no' if args.skip_pm4py_petri_embedding else sample_count} PM4Py Petri embeddings",
        enabled=show_progress,
    )
    report = rich_test_report(
        checkpoint_path=checkpoint_path,
        data_dir=args.data_dir,
        samples=samples,
        batch_size=args.batch_size,
        device=args.device,
        include_pm4py_petri=not args.skip_pm4py_petri_embedding,
        pm4py_petri_config=Pm4pyPetriEmbeddingConfig(
            dimensions=args.petri_embedding_dim,
            num_walks=args.petri_num_walks,
            walk_length=args.petri_walk_length,
            window=args.petri_window,
            epochs=args.petri_epochs,
            seed=args.petri_seed,
        ),
        show_progress=show_progress,
        conformance_method=args.conformance_method,
        max_decode_length=args.max_decode_length,
        beam_size=args.beam_size,
        behavior_traces_per_sample=args.decode_behavior_traces,
        total_test_sample_count=total_sample_count,
        curriculum=args.curriculum,
    )
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(format_human_test_report(report))
    return 0


def test_debug(message: str, enabled: bool = True) -> None:
    if enabled:
        print(f"[test] {message}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "sample":
        return run_sample(args)
    if args.command == "train":
        return run_train(args)
    if args.command == "test":
        return run_test(args)
    parser.error(f"unknown command: {args.command}")
    return 2


def split_counts_from_args(
    args: argparse.Namespace,
    config: SyntheticConfig | None = None,
) -> SplitCounts:
    train_count = args.train_count if args.train_count is not None else args.count
    config = config or SyntheticConfig()
    rows_per_family = config.variants_per_behavior * config.log_views_per_behavior
    explicit_rows = {
        "training": train_count,
        "validation": args.validation_count,
        "test": args.test_count,
    }
    explicit_families = {
        "training": args.train_families,
        "validation": args.validation_families,
        "test": args.test_families,
    }
    for split in explicit_rows:
        if explicit_rows[split] is not None and explicit_families[split] is not None:
            raise ValueError(
                f"choose either {split} row count or family count, not both"
            )
    family_defaults = {"training": 4096, "validation": 512, "test": 512}

    def split_count(split: str) -> int:
        row_count = explicit_rows[split]
        if row_count is not None:
            return _positive(row_count, 1, f"{split}-count")
        family_count = _positive(
            explicit_families[split],
            family_defaults[split],
            f"{split}-families",
        )
        return family_count * rows_per_family

    return SplitCounts(
        training=split_count("training"),
        validation=split_count("validation"),
        test=split_count("test"),
    )


def checkpoint_for_selection(
    checkpoint_path: str | Path,
    selection: str,
) -> Path:
    path = Path(checkpoint_path)
    if selection == "latest":
        return path
    objectives = {
        "best": None,
        "best_balanced": "balanced",
        "best_decode": "decode",
        "best_retrieval": "retrieval",
        "best_geometry": "geometry",
        "best_discovery": "discovery",
    }
    if selection not in objectives:
        raise ValueError(
            "checkpoint selection must be best, latest, best_balanced, "
            "best_decode, best_retrieval, best_geometry, or best_discovery"
        )
    if path.stem.endswith(".best") or ".best_" in path.stem:
        return path
    return best_checkpoint_for(path, objectives[selection])


def _positive(value: int | None, default: int, name: str) -> int:
    value = default if value is None else value
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _parse_weights(value: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        try:
            name, weight = item.split("=", 1)
            result[name.strip()] = float(weight)
        except ValueError as exc:
            raise ValueError("motif weights must use KIND=WEIGHT comma-separated syntax") from exc
    if not result or sum(max(0.0, weight) for weight in result.values()) <= 0:
        raise ValueError("at least one motif weight must be positive")
    return result


def _parse_operator_probabilities(value: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        try:
            name, probability = item.split("=", 1)
            result[name.strip()] = float(probability)
        except ValueError as exc:
            raise ValueError(
                "operator probabilities must use OPERATOR=PROBABILITY "
                "comma-separated syntax"
            ) from exc
    if not result:
        raise ValueError("at least one operator probability must be supplied")
    return result


def round_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {key: _round_nested_value(value) for key, value in metrics.items()}


def _round_nested_value(value: object) -> object:
    if isinstance(value, dict):
        return {key: _round_nested_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_round_nested_value(child) for child in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def round_nested_metrics(row: dict[str, object]) -> dict[str, object]:
    rounded: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, dict):
            rounded[key] = round_metrics(value)
        else:
            rounded[key] = value
    return rounded


if __name__ == "__main__":
    raise SystemExit(main())
