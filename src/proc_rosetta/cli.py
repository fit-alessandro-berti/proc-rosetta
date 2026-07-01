from __future__ import annotations

import argparse
import json

from proc_rosetta.synthetic import SyntheticConfig, generate_samples
from proc_rosetta.training import TrainConfig, train_synthetic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="proc-rosetta")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="print synthetic process triples as JSON lines")
    sample.add_argument("--count", type=int, default=3)
    sample.add_argument("--seed", type=int, default=13)
    sample.add_argument("--max-depth", type=int, default=3)
    sample.add_argument("--max-activities", type=int, default=6)
    sample.add_argument("--max-arity", type=int, default=3)
    sample.add_argument("--traces-per-sample", type=int, default=8)
    sample.add_argument("--curriculum-phase", type=int, default=2)

    train = subparsers.add_parser("train", help="train the first-stage multimodal model")
    train.add_argument("--samples", type=int, default=128)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--latent-dim", type=int, default=64)
    train.add_argument("--hidden-dim", type=int, default=128)
    train.add_argument("--seed", type=int, default=13)
    train.add_argument("--device", default="cpu")
    train.add_argument("--max-depth", type=int, default=3)
    train.add_argument("--max-activities", type=int, default=6)
    train.add_argument("--max-arity", type=int, default=3)
    train.add_argument("--traces-per-sample", type=int, default=8)
    train.add_argument("--curriculum-phase", type=int, default=2)

    return parser


def synthetic_config_from_args(args: argparse.Namespace) -> SyntheticConfig:
    return SyntheticConfig(
        max_depth=args.max_depth,
        max_activities=args.max_activities,
        max_arity=args.max_arity,
        traces_per_sample=args.traces_per_sample,
        curriculum_phase=args.curriculum_phase,
    )


def run_sample(args: argparse.Namespace) -> int:
    samples = generate_samples(args.count, config=synthetic_config_from_args(args), seed=args.seed)
    for sample in samples:
        print(json.dumps(sample.to_dict(), sort_keys=True))
    return 0


def run_train(args: argparse.Namespace) -> int:
    synthetic_config = synthetic_config_from_args(args)
    train_config = TrainConfig(
        samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        device=args.device,
    )
    _, history = train_synthetic(train_config=train_config, synthetic_config=synthetic_config)
    for epoch_idx, metrics in enumerate(history, start=1):
        row = {"epoch": epoch_idx, **{key: round(value, 6) for key, value in metrics.items()}}
        print(json.dumps(row, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "sample":
        return run_sample(args)
    if args.command == "train":
        return run_train(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
