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
from proc_rosetta.models import ProcRosettaModel
from proc_rosetta.synthetic import SyntheticConfig
from proc_rosetta.tokenizers import ActivityTokenizer, TreeTokenizer


@dataclass(frozen=True)
class TrainConfig:
    samples: int = 128
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    latent_dim: int = 256
    hidden_dim: int = 96
    seed: int = 13
    device: str = default_device()
    dropout: float = 0.25
    weight_decay: float = 1e-3
    label_smoothing: float = 0.08
    early_stopping_patience: int = 4
    min_delta: float = 0.001
    lr_patience: int = 2
    lr_factor: float = 0.5
    min_lr: float = 1e-5
    group_aware_batches: bool = True
    views_per_family: int = 2
    activity_remap_probability: float = 0.5


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
        outputs = model(batch)
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
        )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
        batches += 1
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{float(losses['loss'].detach().cpu()):.4f}")
    return {name: value / max(batches, 1) for name, value in totals.items()}


@torch.no_grad()
def evaluate_epoch(
    model: ProcRosettaModel,
    dataloader: DataLoader,
    device: torch.device,
    weights: LossWeights | None = None,
    epoch: int | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
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
        )
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
        batches += 1
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{float(losses['loss'].detach().cpu()):.4f}")
    return {name: value / max(batches, 1) for name, value in totals.items()}


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

        chunks: list[list[int]] = []
        for group in groups:
            for start in range(0, len(group), self.views_per_family):
                chunks.append(group[start : start + self.views_per_family])
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
    weights = LossWeights(label_smoothing=train_config.label_smoothing)
    history = [
        train_epoch(model, dataloader, optimizer, device, weights=weights)
        for _ in range(train_config.epochs)
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
    debug_split("training", train_loader.dataset.samples, len(train_loader), enabled=show_progress)
    debug_split("validation", validation_loader.dataset.samples, len(validation_loader), enabled=show_progress)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_config.lr_factor,
        patience=train_config.lr_patience,
        min_lr=train_config.min_lr,
    )
    weights = LossWeights(label_smoothing=train_config.label_smoothing)
    history: list[dict[str, object]] = []
    best_validation_loss = float("inf")
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
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(validation_metrics["loss"])
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < current_lr:
            debug(
                f"Validation plateau detected; reducing learning rate {current_lr:.6g} -> {new_lr:.6g}",
                enabled=show_progress,
            )

        validation_loss = validation_metrics["loss"]
        improved = validation_loss < (best_validation_loss - train_config.min_delta)
        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        gap = validation_metrics["loss"] - training_metrics["loss"]
        debug(
            f"Epoch {epoch} validation complete: {format_metrics(validation_metrics)} "
            f"| gap={gap:+.4f} | best_val={best_validation_loss:.4f} "
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
            "is_best": improved,
            "epochs_without_improvement": epochs_without_improvement,
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
        if (
            train_config.early_stopping_patience > 0
            and epochs_without_improvement >= train_config.early_stopping_patience
        ):
            debug(
                f"Early stopping after {epoch} epochs; validation loss did not improve by "
                f"{train_config.min_delta:g} for {train_config.early_stopping_patience} epochs.",
                enabled=show_progress,
            )
            break
    return model, history


def evaluate_split_from_checkpoint(
    checkpoint_path: str | Path = "checkpoints/proc_rosetta.pt",
    data_dir: str | Path = "data",
    split: str = "test",
    batch_size: int = 16,
    device: str | None = None,
    show_progress: bool = False,
) -> dict[str, float]:
    torch_device = resolve_device(device)
    model, _ = load_checkpoint(checkpoint_path, torch_device)
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
        show_progress=show_progress,
        progress_desc=f"{split.title()} loss",
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
    "kl",
)


def metrics_csv_columns() -> list[str]:
    columns = [
        "epoch",
        "learning_rate",
        "epoch_seconds",
        "best_validation_loss",
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
    is_best: bool = False,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    train_loader: DataLoader | None = None,
    device: torch.device | None = None,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 3,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "train_config": asdict(train_config),
        "synthetic_config": synthetic_config.to_dict(),
        "history": history,
        "best_validation_loss": best_validation_loss,
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
        if isinstance(validation, dict) and "loss" in validation:
            scheduler.step(float(validation["loss"]))


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
