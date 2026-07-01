from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from proc_rosetta.models import LatentDistribution


@dataclass(frozen=True)
class LossWeights:
    tree_reconstruction: float = 1.0
    trace_to_tree: float = 1.0
    petri_to_tree: float = 1.0
    latent_alignment: float = 0.1
    contrastive: float = 0.1
    kl: float = 0.001


def sequence_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=pad_id,
    )


def latent_alignment_loss(dists: dict[str, LatentDistribution]) -> torch.Tensor:
    names = list(dists)
    total = torch.zeros((), dtype=dists[names[0]].mu.dtype, device=dists[names[0]].mu.device)
    pairs = 0
    for left_idx, left_name in enumerate(names):
        for right_name in names[left_idx + 1 :]:
            total = total + F.mse_loss(dists[left_name].mu, dists[right_name].mu)
            pairs += 1
    return total / max(pairs, 1)


def cross_modal_contrastive_loss(
    dists: dict[str, LatentDistribution],
    temperature: float = 0.2,
) -> torch.Tensor:
    names = list(dists)
    total = torch.zeros((), dtype=dists[names[0]].mu.dtype, device=dists[names[0]].mu.device)
    pairs = 0
    for left_idx, left_name in enumerate(names):
        for right_name in names[left_idx + 1 :]:
            left = F.normalize(dists[left_name].mu, dim=-1)
            right = F.normalize(dists[right_name].mu, dim=-1)
            logits = left @ right.T / temperature
            labels = torch.arange(logits.shape[0], device=logits.device)
            total = total + 0.5 * (
                F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
            )
            pairs += 1
    return total / max(pairs, 1)


def kl_divergence_loss(dists: dict[str, LatentDistribution]) -> torch.Tensor:
    total = torch.zeros((), dtype=next(iter(dists.values())).mu.dtype, device=next(iter(dists.values())).mu.device)
    for dist in dists.values():
        total = total + (-0.5 * (1 + dist.logvar - dist.mu.pow(2) - dist.logvar.exp()).sum(dim=1)).mean()
    return total / len(dists)


def multimodal_tree_loss(
    outputs: dict[str, object],
    tree_tokens: torch.Tensor,
    pad_id: int = 0,
    weights: LossWeights | None = None,
) -> dict[str, torch.Tensor]:
    weights = weights or LossWeights()
    targets = tree_tokens[:, 1:]
    logits = outputs["tree_logits"]
    dists = outputs["dists"]
    assert isinstance(logits, dict)
    assert isinstance(dists, dict)

    rec_tree = sequence_cross_entropy(logits["tree"], targets, pad_id)
    trace_to_tree = sequence_cross_entropy(logits["trace"], targets, pad_id)
    petri_to_tree = sequence_cross_entropy(logits["petri"], targets, pad_id)
    align = latent_alignment_loss(dists)
    contrastive = cross_modal_contrastive_loss(dists)
    kl = kl_divergence_loss(dists)
    total = (
        weights.tree_reconstruction * rec_tree
        + weights.trace_to_tree * trace_to_tree
        + weights.petri_to_tree * petri_to_tree
        + weights.latent_alignment * align
        + weights.contrastive * contrastive
        + weights.kl * kl
    )
    return {
        "loss": total,
        "tree_reconstruction": rec_tree.detach(),
        "trace_to_tree": trace_to_tree.detach(),
        "petri_to_tree": petri_to_tree.detach(),
        "latent_alignment": align.detach(),
        "contrastive": contrastive.detach(),
        "kl": kl.detach(),
    }
