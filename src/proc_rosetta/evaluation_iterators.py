"""Progressive evaluation primitives used by the UI and other clients."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable, Iterator

import numpy as np
import torch

from proc_rosetta.artifact_io import ArtifactModality
from proc_rosetta.behavior import behavioral_distance
from proc_rosetta.benchmarks import (
    activity_count_features,
    alignment_f1_score,
    alignment_fitness_precision,
    directly_follows_features,
    evaluate_embedding_method,
    eventually_follows_features,
    pm4py_log_case_features,
    pm4py_petri_method_report,
    Pm4pyPetriEmbeddingConfig,
    retrieval_metrics,
    trace_variant_features,
    traces_to_event_dataframe,
    vectorize_feature_dicts,
)
from proc_rosetta.inference import (
    LoadedCheckpoint,
    combine_encoding_decode_evidence,
    compare_source_and_decoded,
    decode_latent,
    fuse_latent_means,
)
from proc_rosetta.losses import multimodal_tree_loss
from proc_rosetta.pm4py_bridge import tree_to_petri_net
from proc_rosetta.synthetic import ProcessSample
from proc_rosetta.tree import ProcessTreeNode
from proc_rosetta.training import build_jsonl_dataloader, move_batch_to_device


@dataclass(frozen=True)
class EvaluationUpdate:
    completed: int
    total: int
    artifact_id: str
    section: str
    result: dict[str, Any]
    elapsed_seconds: float


def decode_quality_iter(
    items: Iterable[Any],
    checkpoint: LoadedCheckpoint,
    *,
    max_length: int = 512,
    simulated_traces: int = 100,
    exact_conformance: bool = False,
    completion_policy: str = "prefix_only",
) -> Iterator[EvaluationUpdate]:
    selected = [item for item in items if item.encoding is not None and item.encoding.mu]
    tasks = [
        {
            "artifact_id": item.artifact_id,
            "artifact": item.parsed.display_name,
            "modality": item.parsed.modality.label,
            "latent": item.encoding.mu,
            "source_ids": [item.artifact_id],
            "modalities": [item.parsed.modality],
            "mapping": item.encoding.canonical_mapping,
            "comparison_source": item.parsed,
            "behavior_source": item.parsed
            if item.parsed.modality is ArtifactModality.EVENT_LOG
            else None,
            "latent_source": f"{item.parsed.modality.value}_mean",
            "encodings": [item.encoding],
        }
        for item in selected
    ]
    groups: dict[str, list[Any]] = {}
    for item in selected:
        if item.process_group:
            groups.setdefault(item.process_group, []).append(item)
    for group, members in sorted(groups.items()):
        members_by_modality = {}
        for member in members:
            members_by_modality.setdefault(member.parsed.modality, member)
        members = list(members_by_modality.values())
        if len(members) < 2:
            continue
        mappings = [member.encoding.canonical_mapping for member in members]
        mapping = mappings[0] if mappings and all(value == mappings[0] for value in mappings) else None
        comparison_item = next(
            (
                member
                for modality in (
                    ArtifactModality.PROCESS_TREE,
                    ArtifactModality.EVENT_LOG,
                    ArtifactModality.PETRI_NET,
                )
                for member in members
                if member.parsed.modality is modality
            ),
            members[0],
        )
        tasks.append(
            {
                "artifact_id": f"fused::{group}",
                "artifact": f"{group} · fused",
                "modality": "Fused mean",
                "latent": fuse_latent_means([member.encoding for member in members]),
                "source_ids": [member.artifact_id for member in members],
                "modalities": [member.parsed.modality for member in members],
                "mapping": mapping,
                "comparison_source": comparison_item.parsed,
                "behavior_source": next(
                    (
                        member.parsed
                        for member in members
                        if member.parsed.modality is ArtifactModality.EVENT_LOG
                    ),
                    None,
                ),
                "latent_source": "fused_mean",
                "encodings": [member.encoding for member in members],
            }
        )
    start = perf_counter()
    for index, task in enumerate(tasks, 1):
        allowed, copy, activity_memory = combine_encoding_decode_evidence(
            task["encodings"]
        )
        decode = decode_latent(
            checkpoint,
            task["latent"],
            source_artifact_ids=task["source_ids"],
            source_modalities=task["modalities"],
            latent_source=task["latent_source"],
            canonical_mapping=task["mapping"],
            max_length=max_length,
            allowed_activity_slots=allowed,
            copy_activity_slots=copy,
            activity_memory=activity_memory,
            completion_policy=completion_policy,
        )
        comparison: dict[str, Any]
        try:
            comparison = compare_source_and_decoded(
                task["comparison_source"],
                decode,
                simulated_traces=simulated_traces,
            )
        except Exception as exc:
            comparison = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        result = {
            "artifact": task["artifact"],
            "modality": task["modality"],
            "latent_source": task["latent_source"],
            "eos": decode.eos_emitted,
            "valid_tree": decode.grammar_valid,
            "petri_conversion": decode.petri_convertible,
            "decode_length": len(decode.token_ids),
            "decode_seconds": decode.decode_seconds,
            "completion_policy": completion_policy,
            "evaluation_scope": (
                "raw_model_quality"
                if completion_policy == "prefix_only"
                else "deployment_system_quality"
            ),
            "budget_intervention_steps": decode.budget_intervention_steps,
            "argmax_override_steps": decode.argmax_override_steps,
            "raw_unresolved_open_slots": decode.raw_unresolved_open_slots,
            **comparison,
        }
        if (
            task["behavior_source"] is not None
            and task["behavior_source"] is not task["comparison_source"]
        ):
            try:
                behavior_comparison = compare_source_and_decoded(
                    task["behavior_source"],
                    decode,
                    simulated_traces=simulated_traces,
                )
                result.update(
                    {
                        f"behavior_{key}": value
                        for key, value in behavior_comparison.items()
                        if not isinstance(value, (dict, list))
                    }
                )
            except Exception as exc:
                result["behavior_error"] = f"{type(exc).__name__}: {exc}"
        if (
            exact_conformance
            and task["behavior_source"] is not None
            and decode.tree is not None
        ):
            alignment_started = perf_counter()
            try:
                bundle = tree_to_petri_net(decode.restored_tree or decode.tree)
                fitness, precision = alignment_fitness_precision(
                    task["behavior_source"].traces,
                    bundle.net,
                    bundle.initial_marking,
                    bundle.final_marking,
                )
                result.update(
                    alignment_fitness=fitness,
                    alignment_precision=precision,
                    alignment_f1=alignment_f1_score(fitness, precision),
                    alignment_error=None,
                )
            except Exception as exc:
                result["alignment_error"] = f"{type(exc).__name__}: {exc}"
            result["alignment_seconds"] = perf_counter() - alignment_started
        yield EvaluationUpdate(
            completed=index,
            total=len(tasks),
            artifact_id=task["artifact_id"],
            section="decode_quality",
            result=result,
            elapsed_seconds=perf_counter() - start,
        )


@torch.no_grad()
def neural_loss_iter(
    checkpoint: LoadedCheckpoint,
    sample_path: str,
    *,
    batch_size: int = 16,
    max_batches: int | None = None,
    deterministic: bool = True,
    random_seed: int = 13,
    num_workers: int | None = None,
) -> Iterator[EvaluationUpdate]:
    loader = build_jsonl_dataloader(
        sample_path,
        checkpoint.model.tree_tokenizer,
        checkpoint.model.activity_tokenizer,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    total = min(len(loader), max_batches) if max_batches else len(loader)
    cumulative: dict[str, float] = {}
    start = perf_counter()
    torch.manual_seed(random_seed)
    for index, batch in enumerate(loader, 1):
        if max_batches and index > max_batches:
            break
        moved = move_batch_to_device(batch, checkpoint.device)
        outputs = checkpoint.model(moved, deterministic=deterministic)
        tree_tokens = moved["tree_tokens"]
        positive_mask = moved.get("positive_mask")
        losses = multimodal_tree_loss(
            outputs,
            moved.get("decoder_targets", tree_tokens),
            checkpoint.model.tree_tokenizer.pad_id,
            positive_mask=positive_mask,
        )
        batch_values = {name: float(value.detach().cpu()) for name, value in losses.items()}
        for name, value in batch_values.items():
            cumulative[name] = cumulative.get(name, 0.0) + value
        yield EvaluationUpdate(
            completed=index,
            total=total,
            artifact_id=f"batch-{index}",
            section="neural_losses",
            result={
                "batch": index,
                **{f"batch_{name}": value for name, value in batch_values.items()},
                **{f"running_{name}": value / index for name, value in cumulative.items()},
            },
            elapsed_seconds=perf_counter() - start,
        )


def cross_modal_retrieval_iter(items: Iterable[Any]) -> Iterator[EvaluationUpdate]:
    selected = [item for item in items if item.encoding is not None and item.process_group]
    pairs = [
        (ArtifactModality.PROCESS_TREE, ArtifactModality.EVENT_LOG, "tree_to_trace"),
        (ArtifactModality.EVENT_LOG, ArtifactModality.PROCESS_TREE, "trace_to_tree"),
        (ArtifactModality.PROCESS_TREE, ArtifactModality.PETRI_NET, "tree_to_petri"),
        (ArtifactModality.PETRI_NET, ArtifactModality.PROCESS_TREE, "petri_to_tree"),
        (ArtifactModality.EVENT_LOG, ArtifactModality.PETRI_NET, "trace_to_petri"),
        (ArtifactModality.PETRI_NET, ArtifactModality.EVENT_LOG, "petri_to_trace"),
    ]
    start = perf_counter()
    for index, (left, right, name) in enumerate(pairs, 1):
        left_by_group = {
            item.process_group: item.encoding.mu
            for item in selected
            if item.parsed.modality is left
        }
        right_by_group = {
            item.process_group: item.encoding.mu
            for item in selected
            if item.parsed.modality is right
        }
        groups = sorted(set(left_by_group) & set(right_by_group))
        if groups:
            query = np.asarray([left_by_group[group] for group in groups], dtype=float)
            candidates = np.asarray([right_by_group[group] for group in groups], dtype=float)
            metrics = detailed_retrieval_metrics(query, candidates)
        else:
            metrics = {"count": 0, "available": False, "reason": "no paired process groups"}
        yield EvaluationUpdate(
            completed=index,
            total=len(pairs),
            artifact_id=name,
            section="cross_modal_retrieval",
            result={"direction": name, **metrics},
            elapsed_seconds=perf_counter() - start,
        )


def detailed_retrieval_metrics(query: np.ndarray, candidates: np.ndarray) -> dict[str, Any]:
    base = dict(retrieval_metrics(query, candidates))
    left_norm = query / np.where(np.linalg.norm(query, axis=1, keepdims=True) == 0, 1, np.linalg.norm(query, axis=1, keepdims=True))
    right_norm = candidates / np.where(
        np.linalg.norm(candidates, axis=1, keepdims=True) == 0,
        1,
        np.linalg.norm(candidates, axis=1, keepdims=True),
    )
    similarities = left_norm @ right_norm.T
    ranks = []
    for row in range(len(query)):
        order = np.argsort(-similarities[row])
        ranks.append(int(np.where(order == row)[0][0]) + 1)
    ranks_array = np.asarray(ranks)
    base.update(
        available=True,
        top3_accuracy=float(np.mean(ranks_array <= 3)),
        top5_accuracy=float(np.mean(ranks_array <= 5)),
        median_rank=float(np.median(ranks_array)),
        rank_histogram={str(rank): int(np.sum(ranks_array == rank)) for rank in sorted(set(ranks))},
    )
    return base


def discovery_comparison_iter(
    items: Iterable[Any],
    checkpoint: LoadedCheckpoint,
    *,
    max_length: int = 512,
    exact_conformance: bool = False,
) -> Iterator[EvaluationUpdate]:
    logs = [
        item
        for item in items
        if item.parsed.modality is ArtifactModality.EVENT_LOG
        and item.parsed.traces
        and item.encoding is not None
    ]
    total = len(logs) * 2
    completed = 0
    started = perf_counter()
    for item in logs:
        decode_started = perf_counter()
        allowed, copy, activity_memory = combine_encoding_decode_evidence(
            [item.encoding]
        )
        decoded = decode_latent(
            checkpoint,
            item.encoding.mu,
            source_artifact_ids=[item.artifact_id],
            source_modalities=[item.parsed.modality],
            latent_source="event_log_mean",
            canonical_mapping=item.encoding.canonical_mapping,
            max_length=max_length,
            allowed_activity_slots=allowed,
            copy_activity_slots=copy,
            activity_memory=activity_memory,
        )
        proc_row: dict[str, Any] = {
            "artifact": item.parsed.display_name,
            "method": "ProcRosetta",
            "model_discovered": decoded.tree is not None,
            "petri_conversion": decoded.petri_convertible,
            "runtime_seconds": perf_counter() - decode_started,
            "fitness": None,
            "precision": None,
            "f1": None,
            "error": None,
        }
        if decoded.tree is not None:
            try:
                bundle = tree_to_petri_net(decoded.restored_tree or decoded.tree)
                proc_row.update(_bundle_size(bundle))
                if exact_conformance:
                    fitness, precision = alignment_fitness_precision(
                        item.parsed.traces, bundle.net, bundle.initial_marking, bundle.final_marking
                    )
                    proc_row.update(
                        fitness=fitness,
                        precision=precision,
                        f1=alignment_f1_score(fitness, precision),
                    )
            except Exception as exc:
                proc_row["error"] = f"{type(exc).__name__}: {exc}"
        completed += 1
        yield EvaluationUpdate(
            completed,
            total,
            item.artifact_id,
            "discovery_comparison",
            proc_row,
            perf_counter() - started,
        )

        baseline_started = perf_counter()
        baseline_row: dict[str, Any] = {
            "artifact": item.parsed.display_name,
            "method": "Inductive Miner",
            "model_discovered": False,
            "petri_conversion": False,
            "fitness": None,
            "precision": None,
            "f1": None,
            "error": None,
        }
        try:
            import pm4py

            frame = traces_to_event_dataframe(item.parsed.traces)
            tree = pm4py.discover_process_tree_inductive(
                frame,
                activity_key="concept:name",
                timestamp_key="time:timestamp",
                case_id_key="case:concept:name",
            )
            net, initial, final = pm4py.convert_to_petri_net(tree)
            baseline_row["model_discovered"] = True
            baseline_row["petri_conversion"] = True
            baseline_row.update(
                places=len(net.places),
                transitions=len(net.transitions),
                silent_transitions=sum(transition.label is None for transition in net.transitions),
                arcs=len(net.arcs),
            )
            if exact_conformance:
                fitness, precision = alignment_fitness_precision(
                    item.parsed.traces, net, initial, final
                )
                baseline_row.update(
                    fitness=fitness,
                    precision=precision,
                    f1=alignment_f1_score(fitness, precision),
                )
        except Exception as exc:
            baseline_row["error"] = f"{type(exc).__name__}: {exc}"
        baseline_row["runtime_seconds"] = perf_counter() - baseline_started
        completed += 1
        yield EvaluationUpdate(
            completed,
            total,
            item.artifact_id,
            "discovery_comparison",
            baseline_row,
            perf_counter() - started,
        )


def embedding_baseline_iter(
    items: Iterable[Any],
    *,
    include_petri_node2vec: bool = False,
    petri_config: Pm4pyPetriEmbeddingConfig | None = None,
) -> Iterator[EvaluationUpdate]:
    all_items = list(items)
    logs = [
        item
        for item in all_items
        if item.parsed.modality is ArtifactModality.EVENT_LOG
        and item.parsed.traces
        and item.encoding is not None
    ]
    methods = [
        ("Activity counts", activity_count_features),
        ("Variant distributions", trace_variant_features),
        ("Directly follows", directly_follows_features),
        ("Eventually follows", eventually_follows_features),
        ("PM4Py log-case features", pm4py_log_case_features),
    ]
    cross_representation_methods = 4
    method_count = (
        len(methods)
        + 1
        + cross_representation_methods
        + int(include_petri_node2vec)
    )
    started = perf_counter()
    behavior_matrix = _behavior_distance_matrix([item.parsed.traces for item in logs])
    for index, (name, function) in enumerate(methods, 1):
        method_started = perf_counter()
        matrix = vectorize_feature_dicts([function(item.parsed.traces) for item in logs])
        report = evaluate_embedding_method(matrix, behavior_matrix) if len(logs) >= 2 else {"available": False}
        yield EvaluationUpdate(
            index,
            method_count,
            name,
            "embedding_baselines",
            {
                "method": name,
                "dimension": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
                "runtime_seconds": perf_counter() - method_started,
                **report,
            },
            perf_counter() - started,
        )
    neural = np.asarray([item.encoding.mu for item in logs], dtype=float)
    report = evaluate_embedding_method(neural, behavior_matrix) if len(logs) >= 2 else {"available": False}
    yield EvaluationUpdate(
        len(methods) + 1,
        method_count,
        "ProcRosetta event-log means",
        "embedding_baselines",
        {
            "method": "ProcRosetta event-log means",
            "dimension": int(neural.shape[1]) if neural.ndim == 2 and len(neural) else 0,
            **report,
        },
        perf_counter() - started,
    )
    paired_specs = [
        ("Petri structural counts", "petri_structural"),
        ("ProcRosetta process-tree means", ArtifactModality.PROCESS_TREE),
        ("ProcRosetta Petri-net means", ArtifactModality.PETRI_NET),
        ("ProcRosetta fused means", "fused"),
    ]
    for offset, (name, kind) in enumerate(paired_specs, len(methods) + 2):
        method_started = perf_counter()
        matrix, paired_behavior = _group_aligned_embedding_matrix(all_items, kind)
        report = (
            evaluate_embedding_method(matrix, paired_behavior)
            if len(matrix) >= 2
            else {
                "available": False,
                "reason": "at least two process groups with event logs and the requested representation are required",
            }
        )
        yield EvaluationUpdate(
            offset,
            method_count,
            name,
            "embedding_baselines",
            {
                "method": name,
                "dimension": int(matrix.shape[1]) if matrix.ndim == 2 and len(matrix) else 0,
                "runtime_seconds": perf_counter() - method_started,
                **report,
            },
            perf_counter() - started,
        )
    if include_petri_node2vec:
        node_started = perf_counter()
        paired_samples = _paired_petri_samples(all_items)
        if len(paired_samples) < 2:
            node_report = {
                "available": False,
                "reason": "at least two process groups with paired XES and PNML artifacts are required",
            }
        else:
            paired_behavior = _behavior_distance_matrix(
                [sample.traces for sample in paired_samples]
            )
            _, node_report = pm4py_petri_method_report(
                paired_samples,
                paired_behavior,
                petri_config or Pm4pyPetriEmbeddingConfig(),
            )
        yield EvaluationUpdate(
            method_count,
            method_count,
            "PM4Py Petri Node2Vec",
            "embedding_baselines",
            {
                "method": "PM4Py Petri Node2Vec",
                "runtime_seconds": perf_counter() - node_started,
                **node_report,
            },
            perf_counter() - started,
        )


def _behavior_distance_matrix(trace_sets) -> np.ndarray:
    count = len(trace_sets)
    matrix = np.zeros((count, count), dtype=float)
    for left in range(count):
        for right in range(left + 1, count):
            value = behavioral_distance(trace_sets[left], trace_sets[right])["mean_l1"]
            matrix[left, right] = matrix[right, left] = value
    return matrix


def _bundle_size(bundle) -> dict[str, int]:
    return {
        "places": len(bundle.net.places),
        "transitions": len(bundle.net.transitions),
        "silent_transitions": sum(transition.label is None for transition in bundle.net.transitions),
        "arcs": len(bundle.net.arcs),
    }


def _paired_petri_samples(items: list[Any]) -> list[ProcessSample]:
    logs = {
        item.process_group: item
        for item in items
        if item.process_group and item.parsed.modality is ArtifactModality.EVENT_LOG
    }
    petris = {
        item.process_group: item
        for item in items
        if item.process_group and item.parsed.modality is ArtifactModality.PETRI_NET
    }
    trees = {
        item.process_group: item
        for item in items
        if item.process_group and item.parsed.modality is ArtifactModality.PROCESS_TREE
    }
    samples = []
    for group in sorted(set(logs) & set(petris)):
        tree = (
            trees[group].parsed.tree
            if group in trees and trees[group].parsed.tree is not None
            else ProcessTreeNode.activity("A0")
        )
        samples.append(
            ProcessSample(
                tree=tree,
                traces=logs[group].parsed.traces,
                petri_graph=petris[group].parsed.graph,
                equivalence_id=group,
            )
        )
    return samples


def _group_aligned_embedding_matrix(
    items: list[Any],
    kind: str | ArtifactModality,
) -> tuple[np.ndarray, np.ndarray]:
    logs = {
        item.process_group: item
        for item in items
        if item.process_group
        and item.parsed.modality is ArtifactModality.EVENT_LOG
        and item.parsed.traces
    }
    by_group: dict[str, list[Any]] = {}
    for item in items:
        if item.process_group and item.encoding is not None:
            by_group.setdefault(item.process_group, []).append(item)
    groups = []
    vectors = []
    feature_rows = []
    for group, log_item in sorted(logs.items()):
        members = by_group.get(group, [])
        members_by_modality = {}
        for member in members:
            members_by_modality.setdefault(member.parsed.modality, member)
        members = list(members_by_modality.values())
        if kind == "fused":
            if len(members) < 2:
                continue
            vector = fuse_latent_means([member.encoding for member in members])
        elif kind == "petri_structural":
            petri = next(
                (member for member in members if member.parsed.modality is ArtifactModality.PETRI_NET),
                None,
            )
            if petri is None:
                continue
            feature_rows.append(_petri_graph_features(petri.parsed.graph))
            vector = None
        else:
            member = next(
                (member for member in members if member.parsed.modality is kind),
                None,
            )
            if member is None:
                continue
            vector = member.encoding.mu
        groups.append(group)
        if vector is not None:
            vectors.append(vector)
    if kind == "petri_structural":
        matrix = vectorize_feature_dicts(feature_rows)
    else:
        matrix = np.asarray(vectors, dtype=float)
    traces = [logs[group].parsed.traces for group in groups]
    return matrix, _behavior_distance_matrix(traces)


def _petri_graph_features(graph) -> dict[tuple[str, str], float]:
    return {
        ("nodes", "places"): float(sum(value == 0 for value in graph.node_types)),
        ("nodes", "visible"): float(sum(value == 1 for value in graph.node_types)),
        ("nodes", "invisible"): float(sum(value == 2 for value in graph.node_types)),
        ("edges", "arcs"): float(graph.num_edges),
        ("marking", "initial"): float(sum(graph.initial_marking)),
        ("marking", "final"): float(sum(graph.final_marking)),
    }
