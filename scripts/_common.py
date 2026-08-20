from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Keep PM4Py imports usable in scripts that print machine-readable stdout.
os.environ.setdefault("PM4PY_SHOW_PROGRESS_BAR", "False")
os.environ.setdefault("PM4PY_SHOW_INTERNAL_WARNINGS", "False")
sys._pm4py_welcome_shown = True

from proc_rosetta.pm4py_bridge import (  # noqa: E402
    event_log_to_traces,
    fold_process_tree,
    fold_pm4py_tree,
    from_pm4py_tree,
    petri_net_to_graph,
    to_pm4py_tree,
    tree_to_petri_net,
)
from proc_rosetta.artifact_io import ArtifactModality  # noqa: E402
from proc_rosetta.inference import build_decode_result  # noqa: E402
from proc_rosetta.models import DecodeConstraints  # noqa: E402
from proc_rosetta.devices import default_device, resolve_device  # noqa: E402
from proc_rosetta.training import load_checkpoint  # noqa: E402
from proc_rosetta.tree import ProcessTreeNode  # noqa: E402


SUPPORTED_EXTENSIONS = {".xes", ".pnml", ".ptml"}


def load_trained_model(
    checkpoint_path: str | Path,
    device: str | None = None,
) -> tuple[Any, torch.device]:
    torch_device = resolve_device(device)
    model, _ = load_checkpoint(checkpoint_path, torch_device)
    model.eval()
    model.to(torch_device)
    return model, torch_device


def require_suffix(path: str | Path, suffix: str) -> Path:
    path = Path(path)
    if path.suffix.lower() != suffix:
        raise ValueError(f"expected a {suffix} file, got: {path}")
    return path


def require_supported_input(path: str | Path) -> Path:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        expected = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"unsupported input extension {suffix!r}; expected one of {expected}")
    return path


def import_pm4py() -> Any:
    import pm4py

    return pm4py


def read_ptml_tree(path: str | Path) -> ProcessTreeNode:
    path = require_suffix(path, ".ptml")
    pm4py = import_pm4py()
    return from_pm4py_tree(fold_pm4py_tree(pm4py.read_ptml(str(path))))


def read_pnml_graph(
    path: str | Path,
    auto_guess_final_marking: bool = False,
) -> Any:
    path = require_suffix(path, ".pnml")
    pm4py = import_pm4py()
    net, initial_marking, final_marking = pm4py.read_pnml(
        str(path),
        auto_guess_final_marking=auto_guess_final_marking,
    )
    return petri_net_to_graph(net, initial_marking, final_marking)


def read_xes_traces(
    path: str | Path,
    activity_key: str = "concept:name",
    case_id_key: str = "case:concept:name",
) -> tuple[tuple[str, ...], ...]:
    path = require_suffix(path, ".xes")
    pm4py = import_pm4py()
    log = pm4py.read_xes(str(path), return_legacy_log_object=True)
    log = pm4py.convert_to_event_log(log, case_id_key=case_id_key)
    traces = tuple(tuple(trace) for trace in event_log_to_traces(log, activity_key=activity_key))
    if not traces:
        raise ValueError(f"event log has no traces with activity key {activity_key!r}: {path}")
    if not any(traces):
        raise ValueError(f"event log has no events with activity key {activity_key!r}: {path}")
    return traces


def save_ptml_tree(tree: ProcessTreeNode, output_path: str | Path) -> None:
    output_path = require_suffix(output_path, ".ptml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pm4py = import_pm4py()
    pm4py.write_ptml(to_pm4py_tree(fold_process_tree(tree)), str(output_path))


def activity_mapping_from_tree(tree: ProcessTreeNode, max_activities: int) -> dict[str, str]:
    return activity_mapping_from_labels(tree.activity_labels(), max_activities)


def activity_mapping_from_traces(
    traces: Sequence[Sequence[str]],
    max_activities: int,
) -> dict[str, str]:
    return activity_mapping_from_labels(
        (activity for trace in traces for activity in trace),
        max_activities,
    )


def activity_mapping_from_labels(labels: Iterable[Any], max_activities: int) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for label in labels:
        label = str(label)
        if label not in mapping:
            if len(mapping) >= max_activities:
                raise ValueError(
                    f"input has more than {max_activities} distinct activity labels; "
                    "train or load a checkpoint with a larger tokenizer"
                )
            mapping[label] = f"A{len(mapping)}"
    return mapping


def allowed_activity_slots(
    mapping: dict[str, str],
    max_activities: int,
) -> list[bool]:
    canonical = set(mapping.values())
    return [f"A{index}" in canonical for index in range(max_activities)]


def add_decode_constraint_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--avoid-duplicate-transitions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="mask a visible activity label after its first decoded occurrence (default: enabled)",
    )
    parser.add_argument(
        "--constrain-source-activities",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="restrict decoded visible activities to the source alphabet (default: enabled)",
    )


def canonicalize_traces(
    traces: Sequence[Sequence[str]],
    mapping: dict[str, str],
) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(mapping[str(activity)] for activity in trace) for trace in traces)


def relabel_decoded_tree(
    tree: ProcessTreeNode,
    canonical_mapping: dict[str, str] | None,
) -> ProcessTreeNode:
    if not canonical_mapping:
        return tree
    inverse = {canonical: original for original, canonical in canonical_mapping.items()}
    return tree.relabel(inverse)


def tree_tokens_for_model(model: Any, tree: ProcessTreeNode) -> torch.Tensor:
    token_ids = model.tree_tokenizer.encode_tree(tree)
    return torch.tensor([token_ids], dtype=torch.long)


def traces_for_model(
    model: Any,
    traces: Sequence[Sequence[str]],
    max_traces: int,
    max_trace_length: int,
) -> dict[str, torch.Tensor]:
    if max_traces <= 0:
        raise ValueError("--max-traces must be positive")
    if max_trace_length <= 0:
        raise ValueError("--max-trace-length must be positive")

    selected = list(traces[:max_traces])
    if not selected:
        raise ValueError("at least one trace is required")
    width = min(max((len(trace) for trace in selected), default=1), max_trace_length)
    width = max(width, 1)

    tokens = torch.full(
        (1, len(selected), width),
        model.activity_tokenizer.pad_id,
        dtype=torch.long,
    )
    lengths = torch.zeros((1, len(selected)), dtype=torch.long)
    mask = torch.zeros((1, len(selected)), dtype=torch.bool)

    for trace_idx, trace in enumerate(selected):
        clipped = list(trace[:max_trace_length])
        encoded = model.activity_tokenizer.encode_trace(clipped)
        if encoded:
            tokens[0, trace_idx, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
        lengths[0, trace_idx] = len(encoded)
        mask[0, trace_idx] = True

    return {"tokens": tokens, "lengths": lengths, "mask": mask}


def petri_graph_for_model(
    graph: Any,
    max_petri_nodes: int,
    activity_tokenizer: Any | None = None,
) -> dict[str, torch.Tensor]:
    if max_petri_nodes <= 0:
        raise ValueError("--max-petri-nodes must be positive")
    if graph.num_nodes > max_petri_nodes:
        raise ValueError(
            f"Petri net has {graph.num_nodes} nodes, exceeding --max-petri-nodes={max_petri_nodes}"
        )

    node_count = graph.num_nodes
    node_types = torch.zeros((1, node_count), dtype=torch.long)
    node_mask = torch.ones((1, node_count), dtype=torch.bool)
    markings = torch.zeros((1, node_count, 2), dtype=torch.float32)
    transition_label_ids = torch.zeros((1, node_count), dtype=torch.long)
    edge_sources: list[int] = []
    edge_targets: list[int] = []
    edge_types: list[int] = []

    node_types[0, :node_count] = torch.tensor(graph.node_types, dtype=torch.long)
    markings[0, :node_count, 0] = torch.tensor(graph.initial_marking, dtype=torch.float32)
    markings[0, :node_count, 1] = torch.tensor(graph.final_marking, dtype=torch.float32)
    for src, dst, edge_type in graph.edges:
        edge_sources.append(src)
        edge_targets.append(dst)
        edge_types.append(edge_type)
    if activity_tokenizer is not None:
        for index, label in enumerate(graph.transition_labels):
            if label is not None:
                transition_label_ids[0, index] = activity_tokenizer.token_to_id.get(
                    label,
                    activity_tokenizer.unk_id,
                )

    return {
        "node_types": node_types,
        "node_mask": node_mask,
        "markings": markings,
        "transition_label_ids": transition_label_ids,
        "edge_index": torch.tensor([edge_sources, edge_targets], dtype=torch.long),
        "edge_types": torch.tensor(edge_types, dtype=torch.long),
    }


def move_tensors_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tensors_to_device(child, device) for key, child in value.items()}
    return value


@torch.no_grad()
def encode_tree_mu_logvar(
    model: Any,
    tree: ProcessTreeNode,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = move_tensors_to_device(tree_tokens_for_model(model, tree), device)
    dist = model.encode_tree(tokens)
    return dist.mu, dist.logvar


@torch.no_grad()
def encode_traces_mu_logvar(
    model: Any,
    traces: Sequence[Sequence[str]],
    device: torch.device,
    max_traces: int,
    max_trace_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = traces_for_model(model, traces, max_traces, max_trace_length)
    dist = model.encode_traces(move_tensors_to_device(encoded, device))
    return dist.mu, dist.logvar


@torch.no_grad()
def encode_petri_mu_logvar(
    model: Any,
    graph: Any,
    device: torch.device,
    max_petri_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = petri_graph_for_model(
        graph,
        max_petri_nodes,
        activity_tokenizer=model.activity_tokenizer,
    )
    dist = model.encode_petri(move_tensors_to_device(encoded, device))
    return dist.mu, dist.logvar


@torch.no_grad()
def decode_tree_from_latent(
    model: Any,
    latent: torch.Tensor,
    max_decode_length: int,
    require_petri_convertible: bool = True,
    canonical_mapping: dict[str, str] | None = None,
    constrain_source_activities: bool = True,
    avoid_duplicate_transitions: bool = True,
) -> tuple[ProcessTreeNode, list[int]]:
    slots = (
        allowed_activity_slots(canonical_mapping, model.tree_tokenizer.max_activities)
        if canonical_mapping is not None
        else None
    )
    decoded = model.tree_decoder.decode_greedy(
        latent,
        max_length=max_decode_length,
        apply_grammar_mask=True,
        constraints=DecodeConstraints(
            allowed_activity_slots=(
                None if slots is None else torch.tensor([slots], device=latent.device)
            ),
            constrain_to_source_activities=constrain_source_activities,
            avoid_duplicate_activity_labels=avoid_duplicate_transitions,
        ),
    )
    token_ids = [int(token_id) for token_id in decoded[0].detach().cpu().tolist()]
    if model.tree_tokenizer.eos_id not in token_ids:
        raise ValueError(f"decoder did not emit <eos> within {max_decode_length} tokens")

    result = build_decode_result(
        model,
        latent,
        source_artifact_ids=["command-line-source"],
        source_modalities=[ArtifactModality.PROCESS_TREE],
        latent_source="command_line_latent",
        token_ids=token_ids,
        canonical_mapping=canonical_mapping,
        allowed_activity_slots=slots,
        constrain_to_source_activities=constrain_source_activities,
        avoid_duplicate_activity_labels=avoid_duplicate_transitions,
        max_length=max_decode_length,
    )
    if result.tree is None:
        raise ValueError("; ".join(result.errors) or "decoder did not produce a process tree")
    tree = result.tree
    if require_petri_convertible:
        tree_to_petri_net(tree)
    return tree, token_ids


def tensor_row_to_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu()[0].tolist()]


def print_json(data: dict[str, Any], pretty: bool = False) -> None:
    if pretty:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(json.dumps(data, sort_keys=True))


def run_cli(main: Any) -> int:
    try:
        return int(main())
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
