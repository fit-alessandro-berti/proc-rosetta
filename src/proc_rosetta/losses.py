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
    label_smoothing: float = 0.05


def sequence_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int = 0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    if label_smoothing <= 0.0:
        return F.cross_entropy(flat_logits, flat_targets, ignore_index=pad_id)

    valid_rows = flat_targets.ne(pad_id)
    if not valid_rows.any():
        return flat_logits.sum() * 0.0

    flat_logits = flat_logits[valid_rows]
    flat_targets = flat_targets[valid_rows]
    log_probs = F.log_softmax(flat_logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=flat_targets.unsqueeze(-1)).squeeze(-1)

    grammar_mask = flat_logits.gt(-1e8)
    valid_token_count = grammar_mask.sum(dim=-1).clamp(min=1)
    smooth = -(log_probs.masked_fill(~grammar_mask, 0.0).sum(dim=-1) / valid_token_count)
    loss = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    return loss.mean()


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
    positive_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    names = list(dists)
    total = torch.zeros((), dtype=dists[names[0]].mu.dtype, device=dists[names[0]].mu.device)
    pairs = 0
    for left_idx, left_name in enumerate(names):
        for right_name in names[left_idx + 1 :]:
            left = F.normalize(dists[left_name].mu, dim=-1)
            right = F.normalize(dists[right_name].mu, dim=-1)
            logits = left @ right.T / temperature
            if positive_mask is None:
                mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
            else:
                mask = positive_mask.to(device=logits.device, dtype=torch.bool)
                if mask.shape != logits.shape:
                    raise ValueError(
                        f"positive mask shape {tuple(mask.shape)} does not match "
                        f"contrastive logits {tuple(logits.shape)}"
                    )
            total = total + 0.5 * (
                _multi_positive_info_nce(logits, mask)
                + _multi_positive_info_nce(logits.T, mask.T)
            )
            pairs += 1
    return total / max(pairs, 1)


def _multi_positive_info_nce(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    """InfoNCE where every row sharing a behavior ID is a valid positive."""

    valid_rows = positive_mask.any(dim=1)
    if not valid_rows.any():
        return logits.sum() * 0.0
    denominator = torch.logsumexp(logits, dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positive_mask, -torch.inf), dim=1)
    return (denominator[valid_rows] - numerator[valid_rows]).mean()


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
    positive_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    weights = weights or LossWeights()
    targets = tree_tokens[:, 1:]
    logits = outputs["tree_logits"]
    dists = outputs["dists"]
    assert isinstance(logits, dict)
    assert isinstance(dists, dict)

    rec_tree = sequence_cross_entropy(
        logits["tree"], targets, pad_id, label_smoothing=weights.label_smoothing
    )
    trace_to_tree = sequence_cross_entropy(
        logits["trace"], targets, pad_id, label_smoothing=weights.label_smoothing
    )
    petri_to_tree = sequence_cross_entropy(
        logits["petri"], targets, pad_id, label_smoothing=weights.label_smoothing
    )
    align = latent_alignment_loss(dists)
    contrastive = cross_modal_contrastive_loss(dists, positive_mask=positive_mask)
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
