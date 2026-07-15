from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from proc_rosetta.benchmarks import (
    Pm4pyPetriEmbeddingConfig,
    format_human_test_report,
    rich_test_report,
)
from proc_rosetta.data import SplitCounts, recreate_data_splits
from proc_rosetta.data import read_samples_jsonl, split_samples_path
from proc_rosetta.synthetic import DEFAULT_MAX_ACTIVITIES, SyntheticConfig
from proc_rosetta.training import TrainConfig, train_from_data_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="proc-rosetta")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="recreate synthetic data/training, validation, and test splits")
    sample.add_argument("--data-dir", default="data")
    sample.add_argument("--count", type=int, default=None, help="legacy alias for --train-count")
    sample.add_argument("--train-count", type=int, default=None)
    sample.add_argument("--validation-count", type=int, default=None)
    sample.add_argument("--test-count", type=int, default=None)
    sample.add_argument("--train-families", type=int, default=None)
    sample.add_argument("--validation-families", type=int, default=None)
    sample.add_argument("--test-families", type=int, default=None)
    sample.add_argument("--seed", type=int, default=13)
    sample.add_argument("--max-depth", type=int, default=3)
    sample.add_argument("--max-activities", type=int, default=DEFAULT_MAX_ACTIVITIES)
    sample.add_argument("--max-arity", type=int, default=3)
    sample.add_argument("--traces-per-sample", type=int, default=16)
    sample.add_argument("--curriculum-phase", type=int, default=2)
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
    sample.add_argument("--log-views-per-behavior", type=int, default=1)
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

    train = subparsers.add_parser("train", help="train the first-stage multimodal model")
    train.add_argument("--data-dir", default="data")
    train.add_argument("--checkpoint", default="checkpoints/proc_rosetta.pt")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--latent-dim", type=int, default=64)
    train.add_argument("--hidden-dim", type=int, default=128)
    train.add_argument("--dropout", type=float, default=0.15)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--label-smoothing", type=float, default=0.05)
    train.add_argument("--early-stopping-patience", type=int, default=5)
    train.add_argument("--min-delta", type=float, default=0.001)
    train.add_argument("--lr-patience", type=int, default=2)
    train.add_argument("--lr-factor", type=float, default=0.5)
    train.add_argument("--min-lr", type=float, default=1e-5)
    train.add_argument("--metrics-csv", default="checkpoints/training_metrics.csv")
    train.add_argument("--seed", type=int, default=13)
    train.add_argument("--device", default="cpu")
    train.add_argument("--quiet", action="store_true", help="disable stderr debug messages and progress bars")
    train.add_argument("--views-per-family", type=int, default=2)
    train.add_argument(
        "--no-group-aware-batches",
        action="store_true",
        help="Disable family-grouped batches (multi-positive loss remains ID-aware).",
    )

    test = subparsers.add_parser("test", help="load a checkpoint and evaluate the persisted test split")
    test.add_argument("--data-dir", default="data")
    test.add_argument("--checkpoint", default="checkpoints/proc_rosetta.pt")
    test.add_argument("--batch-size", type=int, default=16)
    test.add_argument("--device", default="cpu")
    test.add_argument("--skip-pm4py-petri-embedding", action="store_true")
    test.add_argument("--petri-embedding-dim", type=int, default=64)
    test.add_argument("--petri-num-walks", type=int, default=5)
    test.add_argument("--petri-walk-length", type=int, default=20)
    test.add_argument("--petri-window", type=int, default=5)
    test.add_argument("--petri-epochs", type=int, default=5)
    test.add_argument("--petri-seed", type=int, default=42)
    test.add_argument("--json", action="store_true", help="print the full machine-readable JSON report")

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
    defaults = {
        "max_depth": 3,
        "max_activities": DEFAULT_MAX_ACTIVITIES,
        "max_arity": 3,
        "traces_per_sample": 16,
        "curriculum_phase": 2,
        "generator": "behavior_families",
        "variants_per_behavior": 2,
        "log_views_per_behavior": 1,
        "log_view_modes": "uniform_variants,resampled",
    }
    for field_name in (
        "max_depth",
        "max_activities",
        "max_arity",
        "traces_per_sample",
        "curriculum_phase",
        "generator",
        "variants_per_behavior",
        "log_views_per_behavior",
    ):
        value = getattr(args, field_name)
        if not configured or value != defaults[field_name]:
            overrides[field_name] = value
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
    metadata = recreate_data_splits(
        data_dir=args.data_dir,
        counts=split_counts_from_args(args, config=config),
        config=config,
        seed=args.seed,
    )
    print(json.dumps(metadata, sort_keys=True))
    return 0


def run_train(args: argparse.Namespace) -> int:
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
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
    )
    _, history = train_from_data_dir(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        train_config=train_config,
        show_progress=not args.quiet,
        metrics_csv_path=args.metrics_csv,
    )
    for row in history:
        print(json.dumps(round_nested_metrics(row), sort_keys=True))
    return 0


def run_test(args: argparse.Namespace) -> int:
    samples = read_samples_jsonl(split_samples_path(args.data_dir, "test"))
    report = rich_test_report(
        checkpoint_path=args.checkpoint,
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
    )
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(format_human_test_report(report))
    return 0


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
    return SplitCounts(
        training=(
            _positive(args.train_families, 1, "train-families") * rows_per_family
            if args.train_families is not None
            else _positive(train_count, default=4000, name="train-count")
        ),
        validation=(
            _positive(args.validation_families, 1, "validation-families") * rows_per_family
            if args.validation_families is not None
            else _positive(args.validation_count, default=512, name="validation-count")
        ),
        test=(
            _positive(args.test_families, 1, "test-families") * rows_per_family
            if args.test_families is not None
            else _positive(args.test_count, default=512, name="test-count")
        ),
    )


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


def round_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 6) for key, value in metrics.items()}


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
