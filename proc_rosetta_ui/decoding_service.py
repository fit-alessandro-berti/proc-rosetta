from __future__ import annotations

from collections.abc import Callable, Sequence

from proc_rosetta.inference import (
    DecodeResult,
    DecodeStep,
    LoadedCheckpoint,
    combine_encoding_decode_evidence,
    decode_latent,
    fuse_latent_distributions,
    sample_latent_distribution,
)
from proc_rosetta_ui.ui_types import WorkspaceArtifact
from proc_rosetta_ui.cache_service import cache_key, cache_put
from proc_rosetta.pm4py_bridge import TREE_NORMALIZATION_VERSION


def decode_workspace_selection(
    items: Sequence[WorkspaceArtifact],
    checkpoint: LoadedCheckpoint,
    *,
    max_length: int = 512,
    beam_size: int = 5,
    constrain_to_source_activities: bool = True,
    avoid_duplicate_activity_labels: bool = True,
    weights: Sequence[float] | None = None,
    progress_callback: Callable[[DecodeStep], None] | None = None,
    decode_cache: dict[str, object] | None = None,
) -> DecodeResult:
    if not items or any(item.encoding is None or not item.encoding.mu for item in items):
        raise ValueError("all selected artifacts must have embeddings")
    encodings = [item.encoding for item in items if item.encoding is not None]
    latent, _ = fuse_latent_distributions(encodings, weights=weights)
    latent_source = (
        f"{encodings[0].modality.value}_mean"
        if len(encodings) == 1
        else "fused_mean" if weights is None else "weighted_fused_mean_experimental"
    )
    result = decode_workspace_latent(
        items,
        checkpoint,
        latent=latent,
        latent_source=latent_source,
        max_length=max_length,
        beam_size=beam_size,
        constrain_to_source_activities=constrain_to_source_activities,
        avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
        progress_callback=progress_callback,
        decode_cache=decode_cache,
    )
    return result


def decode_workspace_latent(
    items: Sequence[WorkspaceArtifact],
    checkpoint: LoadedCheckpoint,
    *,
    latent: Sequence[float],
    latent_source: str,
    max_length: int = 512,
    beam_size: int = 5,
    constrain_to_source_activities: bool = True,
    avoid_duplicate_activity_labels: bool = True,
    progress_callback: Callable[[DecodeStep], None] | None = None,
    decode_cache: dict[str, object] | None = None,
) -> DecodeResult:
    mapping = _shared_mapping(items)
    allowed_slots, copy_slots, activity_memory = combine_encoding_decode_evidence(
        [item.encoding for item in items if item.encoding is not None]
    )
    key = cache_key(
        checkpoint.metadata.identifier,
        list(latent),
        max_length,
        beam_size,
        mapping,
        allowed_slots,
        constrain_to_source_activities,
        avoid_duplicate_activity_labels,
        TREE_NORMALIZATION_VERSION,
        [item.artifact_id for item in items],
        latent_source,
    )
    cached = decode_cache.get(key) if decode_cache else None
    if cached is None:
        result = decode_latent(
            checkpoint,
            latent,
            source_artifact_ids=[item.artifact_id for item in items],
            source_modalities=[item.parsed.modality for item in items],
            latent_source=latent_source,
            canonical_mapping=mapping,
            max_length=max_length,
            beam_size=beam_size,
            allowed_activity_slots=allowed_slots,
            copy_activity_slots=copy_slots,
            activity_memory=activity_memory,
            constrain_to_source_activities=constrain_to_source_activities,
            avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
            progress_callback=progress_callback,
        )
        if decode_cache is not None:
            cache_put(decode_cache, key, result)
    else:
        result = cached
        if progress_callback is not None:
            for step in result.steps:
                progress_callback(step)
    for item in items:
        item.decodes.append(result)
        item.state = (
            "Petri conversion valid"
            if result.successful
            else "invalid tree" if not result.grammar_valid else "decode limit reached"
        )
        item.touch()
    return result


def sampled_workspace_latents(
    items: Sequence[WorkspaceArtifact],
    *,
    count: int,
    random_seed: int,
    weights: Sequence[float] | None = None,
) -> list[list[float]]:
    if not items or any(item.encoding is None for item in items):
        raise ValueError("all selected artifacts must have embeddings")
    encodings = [item.encoding for item in items if item.encoding is not None]
    mu, logvar = fuse_latent_distributions(encodings, weights=weights)
    return [
        sample_latent_distribution(mu, logvar, random_seed=random_seed + index)
        for index in range(max(1, count))
    ]


def _shared_mapping(items: Sequence[WorkspaceArtifact]) -> dict[str, str] | None:
    mappings = [item.encoding.canonical_mapping for item in items if item.encoding is not None]
    if len(mappings) == 1:
        return mappings[0]
    # Cross-artifact label restoration is safe only when every original-to-
    # canonical mapping is exactly the same.
    return mappings[0] if mappings and all(mapping == mappings[0] for mapping in mappings) else None
