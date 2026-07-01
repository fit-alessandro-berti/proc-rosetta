from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from proc_rosetta.data import (
    BatchConfig,
    JsonlProcessDataset,
    ProcessBatchCollator,
    SyntheticProcessDataset,
    load_data_metadata,
    split_samples_path,
)
from proc_rosetta.losses import LossWeights, multimodal_tree_loss
from proc_rosetta.models import ProcRosettaModel
from proc_rosetta.synthetic import SyntheticConfig
from proc_rosetta.tokenizers import ActivityTokenizer, TreeTokenizer


@dataclass(frozen=True)
class TrainConfig:
    samples: int = 128
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 1e-3
    latent_dim: int = 64
    hidden_dim: int = 128
    seed: int = 13
    device: str = "cpu"


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
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    batches = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        tree_tokens = batch["tree_tokens"]
        assert isinstance(tree_tokens, torch.Tensor)
        losses = multimodal_tree_loss(outputs, tree_tokens, model.tree_tokenizer.pad_id, weights=weights)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
        batches += 1
    return {name: value / max(batches, 1) for name, value in totals.items()}


@torch.no_grad()
def evaluate_epoch(
    model: ProcRosettaModel,
    dataloader: DataLoader,
    device: torch.device,
    weights: LossWeights | None = None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch, deterministic=True)
        tree_tokens = batch["tree_tokens"]
        assert isinstance(tree_tokens, torch.Tensor)
        losses = multimodal_tree_loss(outputs, tree_tokens, model.tree_tokenizer.pad_id, weights=weights)
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
        batches += 1
    return {name: value / max(batches, 1) for name, value in totals.items()}


def build_synthetic_dataloader(
    samples: int,
    synthetic_config: SyntheticConfig,
    tree_tokenizer: TreeTokenizer,
    activity_tokenizer: ActivityTokenizer,
    batch_size: int,
    seed: int,
    batch_config: BatchConfig | None = None,
) -> DataLoader:
    dataset = SyntheticProcessDataset(samples, config=synthetic_config, seed=seed)
    collator = ProcessBatchCollator(tree_tokenizer, activity_tokenizer, config=batch_config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)


def build_jsonl_dataloader(
    sample_path: str | Path,
    tree_tokenizer: TreeTokenizer,
    activity_tokenizer: ActivityTokenizer,
    batch_size: int,
    shuffle: bool = False,
    batch_config: BatchConfig | None = None,
) -> DataLoader:
    dataset = JsonlProcessDataset(sample_path)
    collator = ProcessBatchCollator(tree_tokenizer, activity_tokenizer, config=batch_config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


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
    ).to(device)


def train_synthetic(
    train_config: TrainConfig | None = None,
    synthetic_config: SyntheticConfig | None = None,
) -> tuple[ProcRosettaModel, list[dict[str, float]]]:
    train_config = train_config or TrainConfig()
    synthetic_config = synthetic_config or SyntheticConfig()
    torch.manual_seed(train_config.seed)
    device = torch.device(train_config.device)

    model = build_model(train_config, synthetic_config, device)
    dataloader = build_synthetic_dataloader(
        samples=train_config.samples,
        synthetic_config=synthetic_config,
        tree_tokenizer=model.tree_tokenizer,
        activity_tokenizer=model.activity_tokenizer,
        batch_size=train_config.batch_size,
        seed=train_config.seed,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    history = [
        train_epoch(model, dataloader, optimizer, device)
        for _ in range(train_config.epochs)
    ]
    return model, history


def train_from_data_dir(
    data_dir: str | Path = "data",
    checkpoint_path: str | Path = "checkpoints/proc_rosetta.pt",
    train_config: TrainConfig | None = None,
) -> tuple[ProcRosettaModel, list[dict[str, object]]]:
    train_config = train_config or TrainConfig()
    torch.manual_seed(train_config.seed)
    device = torch.device(train_config.device)
    metadata = load_data_metadata(data_dir)
    synthetic_config = SyntheticConfig.from_dict(metadata.get("synthetic_config", {}))
    model = build_model(train_config, synthetic_config, device)
    train_loader = build_jsonl_dataloader(
        split_samples_path(data_dir, "training"),
        model.tree_tokenizer,
        model.activity_tokenizer,
        batch_size=train_config.batch_size,
        shuffle=True,
    )
    validation_loader = build_jsonl_dataloader(
        split_samples_path(data_dir, "validation"),
        model.tree_tokenizer,
        model.activity_tokenizer,
        batch_size=train_config.batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    history: list[dict[str, object]] = []
    for epoch in range(1, train_config.epochs + 1):
        training_metrics = train_epoch(model, train_loader, optimizer, device)
        validation_metrics = evaluate_epoch(model, validation_loader, device)
        row: dict[str, object] = {
            "epoch": epoch,
            "training": training_metrics,
            "validation": validation_metrics,
        }
        history.append(row)
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            train_config=train_config,
            synthetic_config=synthetic_config,
            history=history,
            epoch=epoch,
        )
    return model, history


def evaluate_split_from_checkpoint(
    checkpoint_path: str | Path = "checkpoints/proc_rosetta.pt",
    data_dir: str | Path = "data",
    split: str = "test",
    batch_size: int = 16,
    device: str = "cpu",
) -> dict[str, float]:
    torch_device = torch.device(device)
    model, _ = load_checkpoint(checkpoint_path, torch_device)
    loader = build_jsonl_dataloader(
        split_samples_path(data_dir, split),
        model.tree_tokenizer,
        model.activity_tokenizer,
        batch_size=batch_size,
        shuffle=False,
    )
    return evaluate_epoch(model, loader, torch_device)


def save_checkpoint(
    checkpoint_path: str | Path,
    model: ProcRosettaModel,
    train_config: TrainConfig,
    synthetic_config: SyntheticConfig,
    history: list[dict[str, object]],
    epoch: int,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "train_config": asdict(train_config),
            "synthetic_config": synthetic_config.to_dict(),
            "history": history,
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[ProcRosettaModel, dict[str, object]]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_config = TrainConfig(**checkpoint["train_config"])
    synthetic_config = SyntheticConfig.from_dict(checkpoint["synthetic_config"])
    train_config = TrainConfig(
        samples=train_config.samples,
        epochs=train_config.epochs,
        batch_size=train_config.batch_size,
        learning_rate=train_config.learning_rate,
        latent_dim=train_config.latent_dim,
        hidden_dim=train_config.hidden_dim,
        seed=train_config.seed,
        device=str(device),
    )
    model = build_model(train_config, synthetic_config, torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
