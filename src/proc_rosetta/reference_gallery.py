"""Progressive synthetic reference-gallery construction."""

from __future__ import annotations

from dataclasses import dataclass, replace
import random
from time import perf_counter
from typing import Iterator

import torch
import numpy as np

from proc_rosetta.artifact_io import (
    ArtifactModality,
    event_log_statistics,
    process_tree_statistics,
)
from proc_rosetta.inference import (
    PETRI_LABEL_WARNING,
    ArtifactEncodingResult,
    LoadedCheckpoint,
    petri_graph_to_tensors,
    trace_collection_to_tensors,
)
from proc_rosetta.pm4py_bridge import PetriGraph
from proc_rosetta.synthetic import ProcessSample, SyntheticConfig, generate_sample
from proc_rosetta.tree import ProcessTreeNode


@dataclass
class ReferenceEntry:
    reference_id: str
    process_group: str
    modality: ArtifactModality
    encoding: ArtifactEncodingResult
    tree: ProcessTreeNode
    traces: tuple[tuple[str, ...], ...]
    petri_graph: PetriGraph
    metadata: dict[str, object]


@dataclass(frozen=True)
class ReferenceGalleryUpdate:
    completed: int
    total: int
    entries: tuple[ReferenceEntry, ...]


@torch.no_grad()
def build_reference_gallery_iter(
    checkpoint: LoadedCheckpoint,
    *,
    count: int = 12,
    seed: int = 13,
    traces_per_sample: int = 64,
) -> Iterator[ReferenceGalleryUpdate]:
    if count <= 0:
        raise ValueError("reference count must be positive")
    config = SyntheticConfig.from_dict(checkpoint.metadata.synthetic_configuration)
    config = replace(
        config,
        generator="isolated",
        traces_per_sample=max(1, traces_per_sample),
        max_activities=min(config.max_activities, checkpoint.metadata.maximum_activities),
        max_arity=min(config.max_arity, checkpoint.metadata.maximum_tree_arity),
    )
    rng = random.Random(seed)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        for index in range(1, count + 1):
            sample = generate_sample(
                config=config,
                rng=rng,
                equivalence_id=f"reference-{seed}-{index - 1}",
            )
            entries = tuple(_encode_sample_modalities(checkpoint, sample, index - 1))
            yield ReferenceGalleryUpdate(index, count, entries)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _encode_sample_modalities(
    checkpoint: LoadedCheckpoint,
    sample: ProcessSample,
    sample_index: int,
) -> list[ReferenceEntry]:
    model = checkpoint.model
    device = checkpoint.device
    group = sample.equivalence_id
    entries: list[ReferenceEntry] = []

    start = perf_counter()
    tree_tokens = model.tree_tokenizer.encode_tree(sample.tree, canonicalize=False)
    tree_dist = model.encode_tree(torch.tensor([tree_tokens], dtype=torch.long, device=device))
    entries.append(
        _entry(
            checkpoint,
            sample,
            sample_index,
            ArtifactModality.PROCESS_TREE,
            tree_dist,
            process_tree_statistics(sample.tree),
            perf_counter() - start,
        )
    )

    start = perf_counter()
    trace_tensors = trace_collection_to_tensors(model, sample.traces, device)
    trace_dist, attention = model.trace_encoder.forward_with_attention(
        trace_tensors["tokens"], trace_tensors["lengths"], trace_tensors["mask"]
    )
    entries.append(
        _entry(
            checkpoint,
            sample,
            sample_index,
            ArtifactModality.EVENT_LOG,
            trace_dist,
            event_log_statistics(sample.traces),
            perf_counter() - start,
            attention=[float(value) for value in attention.detach().cpu()[0].tolist()],
        )
    )

    start = perf_counter()
    petri_dist = model.encode_petri(petri_graph_to_tensors(sample.petri_graph, device))
    entries.append(
        _entry(
            checkpoint,
            sample,
            sample_index,
            ArtifactModality.PETRI_NET,
            petri_dist,
            {
                "nodes": sample.petri_graph.num_nodes,
                "arcs": sample.petri_graph.num_edges,
                "visible_labels_used_by_encoder": False,
            },
            perf_counter() - start,
            warnings=[PETRI_LABEL_WARNING],
        )
    )
    return entries


def _entry(
    checkpoint: LoadedCheckpoint,
    sample: ProcessSample,
    sample_index: int,
    modality: ArtifactModality,
    distribution,
    metadata: dict[str, object],
    seconds: float,
    *,
    attention: list[float] | None = None,
    warnings: list[str] | None = None,
) -> ReferenceEntry:
    suffix = {
        ArtifactModality.PROCESS_TREE: "tree",
        ArtifactModality.EVENT_LOG: "log",
        ArtifactModality.PETRI_NET: "petri",
    }[modality]
    artifact_id = f"reference-{sample_index:04d}-{suffix}"
    encoding = ArtifactEncodingResult(
        artifact_id=artifact_id,
        artifact_name=artifact_id,
        modality=modality,
        checkpoint_identifier=checkpoint.metadata.identifier,
        source_metadata=dict(metadata),
        preprocessing_metadata={"reference_gallery": True},
        canonical_mapping={},
        model_input_summary=dict(metadata),
        mu=[float(value) for value in distribution.mu.detach().cpu()[0].tolist()],
        logvar=[float(value) for value in distribution.logvar.detach().cpu()[0].tolist()],
        attention_weights=attention,
        embedding_seconds=seconds,
        process_group=sample.equivalence_id,
        warnings=list(warnings or []),
    )
    return ReferenceEntry(
        reference_id=artifact_id,
        process_group=sample.equivalence_id,
        modality=modality,
        encoding=encoding,
        tree=sample.tree,
        traces=sample.traces,
        petri_graph=sample.petri_graph,
        metadata=dict(sample.metadata),
    )
