from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from proc_rosetta.data import BatchConfig, ProcessBatchCollator, SyntheticProcessDataset
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


def train_synthetic(
    train_config: TrainConfig | None = None,
    synthetic_config: SyntheticConfig | None = None,
) -> tuple[ProcRosettaModel, list[dict[str, float]]]:
    train_config = train_config or TrainConfig()
    synthetic_config = synthetic_config or SyntheticConfig()
    torch.manual_seed(train_config.seed)
    device = torch.device(train_config.device)

    tree_tokenizer = TreeTokenizer(
        max_activities=synthetic_config.max_activities,
        max_arity=max(3, synthetic_config.max_arity),
    )
    activity_tokenizer = ActivityTokenizer(max_activities=synthetic_config.max_activities)
    model = ProcRosettaModel(
        tree_tokenizer=tree_tokenizer,
        activity_tokenizer=activity_tokenizer,
        latent_dim=train_config.latent_dim,
        hidden_dim=train_config.hidden_dim,
    ).to(device)
    dataloader = build_synthetic_dataloader(
        samples=train_config.samples,
        synthetic_config=synthetic_config,
        tree_tokenizer=tree_tokenizer,
        activity_tokenizer=activity_tokenizer,
        batch_size=train_config.batch_size,
        seed=train_config.seed,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    history = [
        train_epoch(model, dataloader, optimizer, device)
        for _ in range(train_config.epochs)
    ]
    return model, history
