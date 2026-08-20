from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from proc_rosetta.models import LatentDistribution
from proc_rosetta.tokenizers import TreeTokenizer


@dataclass(frozen=True)
class LossWeights:
    tree_reconstruction: float = 0.5
    trace_to_tree: float = 2.0
    petri_to_tree: float = 0.5
    exact_contrastive: float = 0.5
    within_modality_contrastive: float = 0.25
    soft_behavior_geometry: float = 0.25
    variance: float = 0.1
    covariance: float = 0.01
    latent_alignment: float = 0.0
    kl: float = 0.0
    label_smoothing: float = 0.0
    contrastive_temperature: float = 0.2
    behavior_temperature: float = 0.2
    latent_temperature: float = 0.2


def tree_token_weights(tokenizer: TreeTokenizer) -> torch.Tensor:
    weights = torch.ones(tokenizer.vocab_size, dtype=torch.float32)
    weights[tokenizer.token_to_id["TAU"]] = 1.25
    weights[tokenizer.eos_id] = 1.5
    for name in tokenizer.operator_tokens:
        weights[tokenizer.token_to_id[name]] = 2.5
    for name in tokenizer.arity_tokens:
        weights[tokenizer.token_to_id[name]] = 2.0
    weights[tokenizer.pad_id] = 0.0
    return weights


def sequence_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int = 0,
    label_smoothing: float = 0.0,
    token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    valid_rows = flat_targets.ne(pad_id)
    if not valid_rows.any():
        return flat_logits.sum() * 0.0
    flat_logits = flat_logits[valid_rows]
    flat_targets = flat_targets[valid_rows]
    log_probs = F.log_softmax(flat_logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=flat_targets.unsqueeze(-1)).squeeze(-1)

    if label_smoothing > 0:
        grammar_mask = flat_logits.gt(-1e8)
        valid_token_count = grammar_mask.sum(dim=-1).clamp(min=1)
        smooth = -(
            log_probs.masked_fill(~grammar_mask, 0.0).sum(dim=-1)
            / valid_token_count
        )
        nll = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    if token_weights is not None:
        selected = token_weights.to(logits.device)[flat_targets]
        return (nll * selected).sum() / selected.sum().clamp_min(1e-12)
    return nll.mean()


def all_positive_supcon(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Require every labeled positive, rather than accepting the easiest one."""

    positive_boolean = positive_mask.to(device=logits.device, dtype=torch.bool)
    positive_mask = positive_boolean.to(dtype=logits.dtype)
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return logits.sum() * 0.0
    if candidate_mask is not None:
        candidate_mask = candidate_mask.to(device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(~candidate_mask, -torch.inf)
    log_prob = F.log_softmax(logits, dim=1)
    positive_log_prob = (
        log_prob.masked_fill(~positive_boolean, 0.0).sum(dim=1)
        / positive_count.clamp_min(1.0)
    )
    return -positive_log_prob[valid].mean()


def cross_modal_contrastive_loss(
    dists: dict[str, LatentDistribution],
    temperature: float = 0.2,
    positive_mask: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    names = list(dists)
    total = dists[names[0]].mu.sum() * 0.0
    directions = 0
    for left_idx, left_name in enumerate(names):
        for right_name in names[left_idx + 1 :]:
            left = F.normalize(dists[left_name].mu, dim=-1)
            right = F.normalize(dists[right_name].mu, dim=-1)
            logits = left @ right.T / temperature
            mask = (
                torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
                if positive_mask is None
                else positive_mask.to(device=logits.device, dtype=torch.bool)
            )
            if mask.shape != logits.shape:
                raise ValueError(
                    f"positive mask shape {tuple(mask.shape)} does not match "
                    f"contrastive logits {tuple(logits.shape)}"
                )
            candidates = (
                None
                if candidate_mask is None
                else candidate_mask.to(device=logits.device, dtype=torch.bool)
            )
            total = total + all_positive_supcon(logits, mask, candidates)
            total = total + all_positive_supcon(
                logits.T,
                mask.T,
                None if candidates is None else candidates.T,
            )
            directions += 2
    return total / max(directions, 1)


def within_modality_contrastive_loss(
    dists: dict[str, LatentDistribution],
    positive_mask: torch.Tensor | None,
    candidate_mask: torch.Tensor | None = None,
    temperature: float = 0.2,
) -> torch.Tensor:
    first = next(iter(dists.values())).mu
    if positive_mask is None:
        return first.sum() * 0.0
    total = first.sum() * 0.0
    valid_modalities = 0
    for distribution in dists.values():
        embedding = F.normalize(distribution.mu, dim=-1)
        logits = embedding @ embedding.T / temperature
        mask = positive_mask.to(device=logits.device, dtype=torch.bool).clone()
        mask.fill_diagonal_(False)
        candidates = (
            torch.ones_like(mask)
            if candidate_mask is None
            else candidate_mask.to(device=logits.device, dtype=torch.bool).clone()
        )
        candidates.fill_diagonal_(False)
        if mask.any():
            total = total + all_positive_supcon(logits, mask, candidates)
            valid_modalities += 1
    return total / max(valid_modalities, 1)


def soft_behavior_geometry_loss(
    dists: dict[str, LatentDistribution],
    behavior_signatures: torch.Tensor | None,
    behavior_temperature: float = 0.2,
    latent_temperature: float = 0.2,
) -> torch.Tensor:
    first = next(iter(dists.values())).mu
    if behavior_signatures is None or behavior_signatures.shape[0] < 2:
        return first.sum() * 0.0
    signatures = F.normalize(behavior_signatures.to(first.device), dim=-1)
    behavior_distance = 1.0 - signatures @ signatures.T
    target_probability = F.softmax(
        -behavior_distance / behavior_temperature, dim=1
    )
    total = first.sum() * 0.0
    directions = 0
    names = list(dists)
    for left_name in names:
        for right_name in names:
            if left_name == right_name:
                continue
            similarity = (
                F.normalize(dists[left_name].mu, dim=-1)
                @ F.normalize(dists[right_name].mu, dim=-1).T
            )
            latent_log_probability = F.log_softmax(
                similarity / latent_temperature, dim=1
            )
            total = total + F.kl_div(
                latent_log_probability,
                target_probability,
                reduction="batchmean",
            )
            directions += 1
    return total / max(directions, 1)


def variance_covariance_loss(
    dists: dict[str, LatentDistribution],
) -> tuple[torch.Tensor, torch.Tensor]:
    first = next(iter(dists.values())).mu
    variance_total = first.sum() * 0.0
    covariance_total = first.sum() * 0.0
    for distribution in dists.values():
        embedding = (
            distribution.pre_normalized
            if distribution.pre_normalized is not None
            else distribution.mu
        )
        if embedding.shape[0] < 2:
            continue
        std = embedding.std(dim=0, unbiased=False)
        variance_total = variance_total + F.relu(1.0 - std).mean()
        centered = embedding - embedding.mean(dim=0)
        covariance = centered.T @ centered / max(embedding.shape[0] - 1, 1)
        off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
        covariance_total = covariance_total + off_diagonal.pow(2).mean()
    count = max(len(dists), 1)
    return variance_total / count, covariance_total / count


def effective_rank(embedding: torch.Tensor) -> torch.Tensor:
    if embedding.shape[0] < 2:
        return embedding.new_tensor(0.0)
    singular_values = torch.linalg.svdvals(
        embedding - embedding.mean(dim=0, keepdim=True)
    )
    probabilities = singular_values / singular_values.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return entropy.exp()


def multimodal_tree_loss(
    outputs: dict[str, object],
    tree_tokens: torch.Tensor | dict[str, torch.Tensor],
    pad_id: int = 0,
    weights: LossWeights | None = None,
    positive_mask: torch.Tensor | None = None,
    contrastive_candidate_mask: torch.Tensor | None = None,
    behavior_signatures: torch.Tensor | None = None,
    tokenizer: TreeTokenizer | None = None,
) -> dict[str, torch.Tensor]:
    weights = weights or LossWeights()
    decoder_targets = (
        tree_tokens
        if isinstance(tree_tokens, dict)
        else outputs.get("decoder_targets")
    )
    if not isinstance(decoder_targets, dict):
        decoder_targets = {
            name: tree_tokens for name in ("tree", "trace", "petri")
        }
    logits = outputs["tree_logits"]
    dists = outputs["dists"]
    assert isinstance(logits, dict)
    assert isinstance(dists, dict)
    token_weight_tensor = None if tokenizer is None else tree_token_weights(tokenizer)

    def reconstruction(name: str) -> torch.Tensor:
        return sequence_cross_entropy(
            logits[name],
            decoder_targets[name][:, 1:],
            pad_id,
            label_smoothing=weights.label_smoothing,
            token_weights=token_weight_tensor,
        )

    rec_tree = reconstruction("tree")
    trace_to_tree = reconstruction("trace")
    petri_to_tree = reconstruction("petri")
    exact = cross_modal_contrastive_loss(
        dists,
        temperature=weights.contrastive_temperature,
        positive_mask=positive_mask,
        candidate_mask=contrastive_candidate_mask,
    )
    within = within_modality_contrastive_loss(
        dists,
        positive_mask,
        contrastive_candidate_mask,
        temperature=weights.contrastive_temperature,
    )
    soft = soft_behavior_geometry_loss(
        dists,
        behavior_signatures,
        behavior_temperature=weights.behavior_temperature,
        latent_temperature=weights.latent_temperature,
    )
    variance, covariance = variance_covariance_loss(dists)
    total = (
        weights.tree_reconstruction * rec_tree
        + weights.trace_to_tree * trace_to_tree
        + weights.petri_to_tree * petri_to_tree
        + weights.exact_contrastive * exact
        + weights.within_modality_contrastive * within
        + weights.soft_behavior_geometry * soft
        + weights.variance * variance
        + weights.covariance * covariance
    )
    ranks = torch.stack(
        [effective_rank(distribution.mu.detach()) for distribution in dists.values()]
    )
    zero = total.detach() * 0.0
    return {
        "loss": total,
        "tree_reconstruction": rec_tree,
        "trace_to_tree": trace_to_tree,
        "petri_to_tree": petri_to_tree,
        "exact_contrastive": exact,
        "within_modality_contrastive": within,
        "soft_behavior_geometry": soft,
        "variance": variance,
        "covariance": covariance,
        "effective_rank": ranks.mean(),
        # Compatibility report keys are retained with deliberately zeroed
        # objectives for old CSV/UI consumers.
        "contrastive": exact,
        "latent_alignment": zero,
        "kl": zero,
    }
