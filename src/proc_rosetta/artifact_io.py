"""Artifact parsing and preprocessing metadata for external inference.

This module deliberately has no Streamlit dependency.  Both command-line
clients and the interactive application can therefore use the same parsing,
canonicalization, sampling, and truncation rules.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable, Sequence
from uuid import uuid4

from proc_rosetta.pm4py_bridge import (
    PetriGraph,
    PetriNetBundle,
    event_log_to_traces,
    fold_pm4py_tree,
    from_pm4py_tree,
    petri_net_to_graph,
    pm4py_tree_prefix_length,
    pm4py_tree_size,
)
from proc_rosetta.tree import NodeKind, ProcessTreeNode


class ArtifactModality(str, Enum):
    EVENT_LOG = "event_log"
    PROCESS_TREE = "process_tree"
    PETRI_NET = "petri_net"

    @property
    def label(self) -> str:
        return {
            self.EVENT_LOG: "Event log",
            self.PROCESS_TREE: "Process tree",
            self.PETRI_NET: "Petri net",
        }[self]


SUFFIX_MODALITIES = {
    ".xes": ArtifactModality.EVENT_LOG,
    ".ptml": ArtifactModality.PROCESS_TREE,
    ".pnml": ArtifactModality.PETRI_NET,
}


class TraceSelectionStrategy(str, Enum):
    FIRST = "first_n"
    SEEDED_RANDOM = "seeded_random"
    FREQUENCY_PRESERVING = "frequency_preserving"
    MOST_FREQUENT_VARIANTS = "most_frequent_variants"
    VARIANT_COVERAGE = "variant_coverage"


@dataclass(frozen=True)
class ArtifactParseSettings:
    activity_key: str = "concept:name"
    case_id_key: str = "case:concept:name"
    lifecycle_key: str = "lifecycle:transition"
    add_lifecycle_to_labels: bool = False
    remove_empty_traces: bool = True
    auto_guess_final_marking: bool = False


@dataclass(frozen=True)
class PreprocessingSettings:
    max_events: int = 1_000_000
    max_traces: int = 128
    max_trace_length: int = 128
    trace_selection_strategy: TraceSelectionStrategy = TraceSelectionStrategy.FIRST
    random_seed: int = 13
    variant_coverage: float = 0.8
    compress_duplicate_variants: bool = False
    max_petri_nodes: int = 512
    max_tree_tokens: int = 512


@dataclass
class ParsedArtifact:
    artifact_id: str
    display_name: str
    modality: ArtifactModality
    content_hash: str
    source_size_bytes: int
    parse_settings: ArtifactParseSettings
    source_metadata: dict[str, Any]
    traces: tuple[tuple[str, ...], ...] | None = None
    tree: ProcessTreeNode | None = None
    petri: PetriNetBundle | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def graph(self) -> PetriGraph | None:
        return None if self.petri is None else self.petri.graph


@dataclass
class PreparedArtifact:
    parsed: ParsedArtifact
    canonical_mapping: dict[str, str]
    canonical_frequencies: dict[str, int]
    model_input: Any | None
    model_input_summary: dict[str, Any]
    preprocessing_metadata: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.model_input is not None and not self.errors


def modality_from_name(filename: str) -> ArtifactModality:
    suffix = Path(filename).suffix.lower()
    try:
        return SUFFIX_MODALITIES[suffix]
    except KeyError as exc:
        expected = ", ".join(sorted(SUFFIX_MODALITIES))
        raise ValueError(f"unsupported artifact extension {suffix!r}; expected {expected}") from exc


def parse_artifact(
    source: str | Path | bytes,
    *,
    filename: str | None = None,
    artifact_id: str | None = None,
    settings: ArtifactParseSettings | None = None,
) -> ParsedArtifact:
    """Parse XES, PTML, or PNML from a path or uploaded bytes."""

    settings = settings or ArtifactParseSettings()
    if isinstance(source, (str, Path)):
        path = Path(source)
        data = path.read_bytes()
        filename = filename or path.name
        return _parse_path(path, data, filename, artifact_id, settings)
    if filename is None:
        raise ValueError("filename is required when parsing artifact bytes")
    suffix = Path(filename).suffix.lower()
    modality_from_name(filename)
    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        handle.write(source)
        handle.flush()
        return _parse_path(Path(handle.name), source, filename, artifact_id, settings)


def _parse_path(
    path: Path,
    data: bytes,
    filename: str,
    artifact_id: str | None,
    settings: ArtifactParseSettings,
) -> ParsedArtifact:
    modality = modality_from_name(filename)
    artifact_id = artifact_id or f"artifact-{uuid4().hex[:10]}"
    base = dict(
        artifact_id=artifact_id,
        display_name=filename,
        modality=modality,
        content_hash=sha256(data).hexdigest(),
        source_size_bytes=len(data),
        parse_settings=settings,
    )
    pm4py = _import_pm4py()

    if modality is ArtifactModality.EVENT_LOG:
        log = pm4py.read_xes(str(path), return_legacy_log_object=True)
        log = pm4py.convert_to_event_log(log, case_id_key=settings.case_id_key)
        traces: list[tuple[str, ...]] = []
        missing_activity_events = 0
        for trace in log:
            labels: list[str] = []
            for event in trace:
                if settings.activity_key not in event:
                    missing_activity_events += 1
                    continue
                label = str(event[settings.activity_key])
                if settings.add_lifecycle_to_labels and settings.lifecycle_key in event:
                    label = f"{label}+{event[settings.lifecycle_key]}"
                labels.append(label)
            if labels or not settings.remove_empty_traces:
                traces.append(tuple(labels))
        if not traces:
            raise ValueError("event log contains no usable traces")
        metadata = event_log_statistics(traces)
        metadata.update(
            activity_key=settings.activity_key,
            case_id_key=settings.case_id_key,
            missing_activity_events=missing_activity_events,
        )
        warnings = []
        if missing_activity_events:
            warnings.append(
                f"{missing_activity_events} events lacked activity attribute "
                f"{settings.activity_key!r} and were excluded."
            )
        return ParsedArtifact(**base, source_metadata=metadata, traces=tuple(traces), warnings=warnings)

    if modality is ArtifactModality.PROCESS_TREE:
        source_pm_tree = pm4py.read_ptml(str(path))
        size_before = pm4py_tree_size(source_pm_tree)
        prefix_before = pm4py_tree_prefix_length(source_pm_tree)
        source_syntax = str(source_pm_tree)
        folded_pm_tree = fold_pm4py_tree(source_pm_tree)
        tree = from_pm4py_tree(folded_pm_tree)
        metadata = process_tree_statistics(tree)
        metadata.update(
            source_tree_size_before_fold=size_before,
            source_tree_size_after_fold=tree.size(),
            source_prefix_length_before_fold=prefix_before,
            source_prefix_length_after_fold=len(tree.to_prefix_tokens()) + 2,
            fold_changed=source_syntax != str(folded_pm_tree),
            fold_version="pm4py-fold-v1",
            normalization_version="pm4py-fold-v1",
        )
        return ParsedArtifact(
            **base,
            source_metadata=metadata,
            tree=tree,
        )

    net, initial, final = pm4py.read_pnml(
        str(path),
        auto_guess_final_marking=settings.auto_guess_final_marking,
    )
    graph = petri_net_to_graph(net, initial, final)
    bundle = PetriNetBundle(net, initial, final, graph)
    metadata = petri_net_statistics(bundle)
    metadata.update(
        final_marking_guessed=settings.auto_guess_final_marking,
        initial_marking_loaded=bool(initial),
        final_marking_loaded=bool(final),
    )
    return ParsedArtifact(
        **base,
        source_metadata=metadata,
        petri=bundle,
    )


def event_log_statistics(traces: Sequence[Sequence[str]]) -> dict[str, Any]:
    variants = Counter(tuple(trace) for trace in traces)
    activities = Counter(activity for trace in traces for activity in trace)
    lengths = [len(trace) for trace in traces]
    starts = Counter(trace[0] for trace in traces if trace)
    ends = Counter(trace[-1] for trace in traces if trace)
    directly_follows = Counter(
        (left, right)
        for trace in traces
        for left, right in zip(trace, trace[1:])
    )
    return {
        "total_traces": len(traces),
        "total_events": sum(lengths),
        "distinct_activities": len(activities),
        "trace_variants": len(variants),
        "mean_trace_length": sum(lengths) / max(len(lengths), 1),
        "maximum_trace_length": max(lengths, default=0),
        "empty_traces": sum(length == 0 for length in lengths),
        "activity_frequencies": dict(activities.most_common()),
        "variant_frequencies": [
            {"variant": " → ".join(variant) or "∅", "frequency": count}
            for variant, count in variants.most_common()
        ],
        "trace_length_frequencies": dict(sorted(Counter(lengths).items())),
        "start_activity_frequencies": dict(starts.most_common()),
        "end_activity_frequencies": dict(ends.most_common()),
        "directly_follows_frequencies": [
            {"source": source, "target": target, "frequency": count}
            for (source, target), count in directly_follows.most_common()
        ],
    }


def process_tree_statistics(tree: ProcessTreeNode) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    max_arity = 0

    def visit(node: ProcessTreeNode) -> None:
        nonlocal max_arity
        counts[node.kind.value] += 1
        max_arity = max(max_arity, len(node.children))
        for child in node.children:
            visit(child)

    visit(tree)
    return {
        "tree_size": tree.size(),
        "maximum_depth": tree.max_depth(),
        "activity_leaves": counts[NodeKind.ACTIVITY.value],
        "tau_leaves": counts[NodeKind.TAU.value],
        "sequence_operators": counts[NodeKind.SEQ.value],
        "exclusive_choice_operators": counts[NodeKind.XOR.value],
        "parallel_operators": counts[NodeKind.AND.value],
        "loop_operators": counts[NodeKind.LOOP.value],
        "maximum_operator_arity": max_arity,
        "prefix_token_length": len(tree.to_prefix_tokens()) + 2,
        "prefix_tokens": ["<bos>", *tree.to_prefix_tokens(), "<eos>"],
        "distinct_activities": len(set(tree.activity_labels())),
    }


def petri_net_statistics(bundle: PetriNetBundle) -> dict[str, Any]:
    graph = bundle.graph
    labels = [label for label in graph.transition_labels if label is not None]
    degree = Counter()
    for source, target, _ in graph.edges:
        degree[source] += 1
        degree[target] += 1
    return {
        "places": sum(node_type == 0 for node_type in graph.node_types),
        "transitions": sum(node_type in {1, 2} for node_type in graph.node_types),
        "visible_transitions": sum(node_type == 1 for node_type in graph.node_types),
        "invisible_transitions": sum(node_type == 2 for node_type in graph.node_types),
        "arcs": graph.num_edges,
        "nodes": graph.num_nodes,
        "initial_tokens": int(sum(graph.initial_marking)),
        "final_tokens": int(sum(graph.final_marking)),
        "duplicate_visible_labels": sum(count - 1 for count in Counter(labels).values() if count > 1),
        "isolated_nodes": sum(degree[index] == 0 for index in range(graph.num_nodes)),
    }


def prepare_artifact_for_model(
    parsed: ParsedArtifact,
    *,
    max_activities: int,
    max_arity: int,
    settings: PreprocessingSettings | None = None,
) -> PreparedArtifact:
    settings = settings or PreprocessingSettings()
    warnings = list(parsed.warnings)
    errors: list[str] = []
    labels = _artifact_labels(parsed)
    frequencies = Counter(labels)
    mapping: dict[str, str] = {}
    for label in labels:
        if label not in mapping and len(mapping) < max_activities:
            mapping[label] = f"A{len(mapping)}"
    unsupported = [label for label in dict.fromkeys(labels) if label not in mapping]
    if unsupported:
        errors.append(
            f"Artifact has {len(set(labels))} activities but the checkpoint supports "
            f"{max_activities}; unsupported labels: {', '.join(unsupported[:8])}"
        )

    if parsed.modality is ArtifactModality.EVENT_LOG:
        assert parsed.traces is not None
        total_events = int(parsed.source_metadata.get("total_events", 0))
        if total_events > settings.max_events:
            errors.append(
                f"Event log has {total_events} events, exceeding the configured workspace limit "
                f"{settings.max_events}; the input is not silently reduced."
            )
        selected = select_traces(parsed.traces, settings)
        clipped = tuple(tuple(trace[: settings.max_trace_length]) for trace in selected)
        clipped_trace_count = sum(len(trace) > settings.max_trace_length for trace in selected)
        discarded_events = sum(max(0, len(trace) - settings.max_trace_length) for trace in selected)
        if len(selected) < len(parsed.traces):
            warnings.append(
                f"Trace selection passed {len(selected)} of {len(parsed.traces)} traces to the encoder."
            )
        if clipped_trace_count:
            warnings.append(
                f"Clipped {clipped_trace_count} traces and discarded {discarded_events} events."
            )
        canonical = tuple(
            tuple(mapping[event] for event in trace if event in mapping)
            for trace in clipped
        )
        original_events = sum(len(trace) for trace in selected)
        metadata = {
            "strategy": settings.trace_selection_strategy.value,
            "random_seed": settings.random_seed,
            "total_traces": len(parsed.traces),
            "total_events": total_events,
            "maximum_workspace_events": settings.max_events,
            "encoder_traces": len(canonical),
            "dropped_traces": len(parsed.traces) - len(canonical),
            "maximum_trace_length": settings.max_trace_length,
            "clipped_traces": clipped_trace_count,
            "discarded_events": discarded_events,
            "discarded_event_percentage": 100.0 * discarded_events / max(original_events, 1),
            "compressed_variants": settings.compress_duplicate_variants,
        }
        return PreparedArtifact(
            parsed=parsed,
            canonical_mapping=mapping,
            canonical_frequencies=dict(frequencies),
            model_input=canonical if not errors else None,
            model_input_summary={
                "shape": [1, len(canonical), min(settings.max_trace_length, max(map(len, canonical), default=1))],
                "selected_traces": [list(trace) for trace in selected],
                "canonical_traces": [list(trace) for trace in canonical],
            },
            preprocessing_metadata=metadata,
            warnings=warnings,
            errors=errors,
        )

    if parsed.modality is ArtifactModality.PROCESS_TREE:
        assert parsed.tree is not None
        canonical_tree = parsed.tree.relabel(mapping)
        unsupported_nodes = [
            f"node-{index}:{node.kind.value}/arity-{len(node.children)}"
            for index, node in enumerate(walk_tree(parsed.tree))
            if node.children and len(node.children) > max_arity
        ]
        if unsupported_nodes:
            canonical_tree = canonical_tree.reassociate_operators(max_arity)
            warnings.append(
                f"Re-associated {len(unsupported_nodes)} operator node(s) to checkpoint maximum "
                f"arity {max_arity}; no activities or branches were dropped."
            )
        prefix_length = len(canonical_tree.to_prefix_tokens()) + 2
        if prefix_length > settings.max_tree_tokens:
            errors.append(
                f"Tree requires {prefix_length} tokens, exceeding configured limit "
                f"{settings.max_tree_tokens}; trees are never silently clipped."
            )
        return PreparedArtifact(
            parsed=parsed,
            canonical_mapping=mapping,
            canonical_frequencies=dict(frequencies),
            model_input=canonical_tree if not errors else None,
            model_input_summary={
                "prefix_tokens": ["<bos>", *canonical_tree.to_prefix_tokens(), "<eos>"],
                "token_count": prefix_length,
                "source_maximum_operator_arity": parsed.source_metadata["maximum_operator_arity"],
                "model_input_maximum_operator_arity": min(
                    parsed.source_metadata["maximum_operator_arity"], max_arity
                ),
            },
            preprocessing_metadata={
                "maximum_tree_tokens": settings.max_tree_tokens,
                "checkpoint_maximum_arity": max_arity,
                "operator_arity_reassociated": bool(unsupported_nodes),
            },
            warnings=warnings,
            errors=errors,
        )

    assert parsed.graph is not None
    graph = parsed.graph.relabel(mapping)
    if graph.num_nodes > settings.max_petri_nodes:
        errors.append(
            f"Petri net has {graph.num_nodes} nodes, exceeding the configured encoder limit "
            f"{settings.max_petri_nodes}; nets are never silently truncated."
        )
    return PreparedArtifact(
        parsed=parsed,
        canonical_mapping=mapping,
        canonical_frequencies=dict(frequencies),
        model_input=graph if not errors else None,
        model_input_summary={
            "node_types": list(graph.node_types),
            "adjacency_edges": graph.num_edges,
            "initial_marking": list(graph.initial_marking),
            "final_marking": list(graph.final_marking),
            "visible_labels_used_by_encoder": True,
        },
        preprocessing_metadata={
            "maximum_petri_nodes": settings.max_petri_nodes,
            "node_count": graph.num_nodes,
            "node_truncation": False,
        },
        warnings=warnings,
        errors=errors,
    )


def select_traces(
    traces: Sequence[Sequence[str]],
    settings: PreprocessingSettings,
) -> tuple[tuple[str, ...], ...]:
    if settings.max_traces <= 0:
        raise ValueError("max_traces must be positive")
    normalized = tuple(tuple(trace) for trace in traces)
    if settings.compress_duplicate_variants:
        normalized = tuple(dict.fromkeys(normalized))
    limit = min(settings.max_traces, len(normalized))
    strategy = TraceSelectionStrategy(settings.trace_selection_strategy)
    if strategy is TraceSelectionStrategy.FIRST:
        return normalized[:limit]
    rng = random.Random(settings.random_seed)
    if strategy is TraceSelectionStrategy.SEEDED_RANDOM:
        indices = sorted(rng.sample(range(len(normalized)), limit))
        return tuple(normalized[index] for index in indices)
    frequencies = Counter(normalized)
    if strategy is TraceSelectionStrategy.MOST_FREQUENT_VARIANTS:
        selected: list[tuple[str, ...]] = []
        for variant, count in frequencies.most_common():
            selected.extend([variant] * min(count, limit - len(selected)))
            if len(selected) == limit:
                break
        return tuple(selected)
    if strategy is TraceSelectionStrategy.VARIANT_COVERAGE:
        selected = []
        target = max(0.0, min(1.0, settings.variant_coverage)) * len(normalized)
        covered = 0
        for variant, count in frequencies.most_common():
            take = min(count, limit - len(selected))
            selected.extend([variant] * take)
            covered += count
            if len(selected) == limit or covered >= target:
                break
        return tuple(selected)

    # Allocate per-variant quotas by largest remainder, then shuffle the
    # resulting cases reproducibly. This preserves variant proportions without
    # giving common variants a second, unintended frequency weighting.
    total = len(normalized)
    quotas = {
        variant: frequencies[variant] * limit / total for variant in frequencies
    }
    allocations = {variant: int(quota) for variant, quota in quotas.items()}
    remaining = limit - sum(allocations.values())
    remainders = sorted(
        frequencies,
        key=lambda variant: (quotas[variant] - allocations[variant], frequencies[variant]),
        reverse=True,
    )
    for variant in remainders[:remaining]:
        allocations[variant] += 1
    selected = [
        variant
        for variant, allocation in allocations.items()
        for _ in range(min(allocation, frequencies[variant]))
    ]
    rng.shuffle(selected)
    return tuple(selected)


def canonical_mapping_rows(prepared: PreparedArtifact) -> list[dict[str, Any]]:
    used_labels = set(prepared.canonical_mapping)
    return [
        {
            "Original label": label,
            "Canonical label": prepared.canonical_mapping.get(label, "—"),
            "Frequency": frequency,
            "Used by encoder": label in used_labels,
        }
        for label, frequency in sorted(
            prepared.canonical_frequencies.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def walk_tree(tree: ProcessTreeNode) -> Iterable[ProcessTreeNode]:
    yield tree
    for child in tree.children:
        yield from walk_tree(child)


def _artifact_labels(parsed: ParsedArtifact) -> list[str]:
    if parsed.traces is not None:
        return [event for trace in parsed.traces for event in trace]
    if parsed.tree is not None:
        return list(parsed.tree.activity_labels())
    if parsed.graph is not None:
        return [label for label in parsed.graph.transition_labels if label is not None]
    return []


def _import_pm4py() -> Any:
    import os
    import sys

    os.environ.setdefault("PM4PY_SHOW_PROGRESS_BAR", "False")
    os.environ.setdefault("PM4PY_SHOW_INTERNAL_WARNINGS", "False")
    sys._pm4py_welcome_shown = True
    import pm4py

    return pm4py
