"""Small, serializable visualization projections for artifacts and latents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from proc_rosetta.artifact_io import ArtifactModality, ParsedArtifact
from proc_rosetta.inference import ArtifactEncodingResult
from proc_rosetta.tree import NodeKind, ProcessTreeNode


MODALITY_COLORS = {
    ArtifactModality.EVENT_LOG: "#30c6b0",
    ArtifactModality.PROCESS_TREE: "#7c6ef6",
    ArtifactModality.PETRI_NET: "#ffad5a",
}


@dataclass(frozen=True)
class ProjectionResult:
    rows: list[dict[str, Any]]
    explained_variance: tuple[float, float] | None
    meaningful: bool


def cosine_similarity_matrix(encodings: Sequence[ArtifactEncodingResult]) -> np.ndarray:
    if not encodings:
        return np.empty((0, 0), dtype=float)
    matrix = np.asarray([encoding.mu for encoding in encodings], dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.where(norms == 0.0, 1.0, norms)
    return normalized @ normalized.T


def euclidean_distance_matrix(encodings: Sequence[ArtifactEncodingResult]) -> np.ndarray:
    if not encodings:
        return np.empty((0, 0), dtype=float)
    matrix = np.asarray([encoding.mu for encoding in encodings], dtype=float)
    return np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=-1)


def project_pca(
    encodings: Sequence[ArtifactEncodingResult],
    groups: dict[str, str] | None = None,
) -> ProjectionResult:
    groups = groups or {}
    if not encodings:
        return ProjectionResult([], None, False)
    matrix = np.asarray([encoding.mu for encoding in encodings], dtype=float)
    if len(encodings) < 3 or np.linalg.matrix_rank(matrix - matrix.mean(axis=0)) < 2:
        rows = [
            {
                "artifact_id": encoding.artifact_id,
                "artifact": encoding.artifact_name,
                "group": groups.get(encoding.artifact_id, ""),
                "modality": encoding.modality.label,
                "pc1": float(index),
                "pc2": 0.0,
            }
            for index, encoding in enumerate(encodings)
        ]
        return ProjectionResult(rows, None, False)
    centered = matrix - matrix.mean(axis=0)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ components[:2].T
    variances = singular**2 / max(len(encodings) - 1, 1)
    ratios = variances / max(float(variances.sum()), np.finfo(float).eps)
    rows = [
        {
            "artifact_id": encoding.artifact_id,
            "artifact": encoding.artifact_name,
            "group": groups.get(encoding.artifact_id, ""),
            "modality": encoding.modality.label,
            "pc1": float(coordinates[index, 0]),
            "pc2": float(coordinates[index, 1]),
        }
        for index, encoding in enumerate(encodings)
    ]
    return ProjectionResult(rows, (float(ratios[0]), float(ratios[1])), True)


def nearest_neighbors(
    selected_id: str,
    encodings: Sequence[ArtifactEncodingResult],
) -> list[dict[str, Any]]:
    index = next(
        (idx for idx, encoding in enumerate(encodings) if encoding.artifact_id == selected_id),
        None,
    )
    if index is None:
        raise KeyError(f"unknown encoded artifact: {selected_id}")
    cosine = cosine_similarity_matrix(encodings)
    euclidean = euclidean_distance_matrix(encodings)
    rows = []
    for other_index, encoding in enumerate(encodings):
        if other_index == index:
            continue
        rows.append(
            {
                "artifact_id": encoding.artifact_id,
                "artifact": encoding.artifact_name,
                "modality": encoding.modality.label,
                "cosine_similarity": float(cosine[index, other_index]),
                "euclidean_distance": float(euclidean[index, other_index]),
            }
        )
    return sorted(rows, key=lambda row: (-row["cosine_similarity"], row["euclidean_distance"]))


def tree_to_dot(
    tree: ProcessTreeNode,
    *,
    title: str = "Process tree",
    maximum_display_depth: int | None = None,
    activity_search: str = "",
) -> str:
    lines = ["digraph process_tree {", "rankdir=TB;", f'label="{_dot_escape(title)}";', "labelloc=t;"]
    counter = 0

    def visit(node: ProcessTreeNode, depth: int = 1) -> str:
        nonlocal counter
        node_id = f"n{counter}"
        counter += 1
        if node.kind is NodeKind.ACTIVITY:
            label = str(node.label)
            shape = "box"
            color = "#30c6b0"
        elif node.kind is NodeKind.TAU:
            label = "τ"
            shape = "box"
            color = "#8892a8"
        else:
            symbols = {
                NodeKind.SEQ: "→ SEQ",
                NodeKind.XOR: "× XOR",
                NodeKind.AND: "∧ AND",
                NodeKind.LOOP: "↻ LOOP",
            }
            label = f"{symbols[node.kind]} · {len(node.children)}"
            shape = "ellipse"
            color = "#7c6ef6"
        if activity_search and activity_search.casefold() in label.casefold():
            color = "#ed6fd1"
        collapsed = bool(maximum_display_depth and depth >= maximum_display_depth and node.children)
        if collapsed:
            label += f" … {node.size() - 1} hidden"
        tooltip = f"node={node_id}, subtree_size={node.size()}, depth={node.max_depth()}"
        lines.append(
            f'{node_id} [label="{_dot_escape(label)}", shape={shape}, '
            f'style="filled", fillcolor="{color}", fontcolor="white", '
            f'tooltip="{_dot_escape(tooltip)}"];'
        )
        if not collapsed:
            for child in node.children:
                child_id = visit(child, depth + 1)
                lines.append(f"{node_id} -> {child_id};")
        return node_id

    visit(tree)
    lines.append("}")
    return "\n".join(lines)


def petri_to_dot(
    artifact: ParsedArtifact | Any,
    *,
    title: str = "Petri net",
    visible_node_indices: set[int] | None = None,
    hide_invisible_transitions: bool = False,
) -> str:
    graph = artifact.graph if isinstance(artifact, ParsedArtifact) else artifact.graph
    lines = ["digraph petri_net {", "rankdir=LR;", f'label="{_dot_escape(title)}";', "labelloc=t;"]
    for index, (node_type, name, label) in enumerate(
        zip(graph.node_types, graph.node_names, graph.transition_labels)
    ):
        if visible_node_indices is not None and index not in visible_node_indices:
            continue
        if hide_invisible_transitions and node_type == 2:
            continue
        initial = graph.initial_marking[index]
        final = graph.final_marking[index]
        if node_type == 0:
            node_label = f"● {int(initial)}" if initial else ("◎" if final else "")
            shape = "circle"
            fill = "#162036"
            tooltip = f"place={name}, initial={initial:g}, final={final:g}"
        elif node_type == 1:
            node_label = str(label)
            shape = "box"
            fill = "#ffad5a"
            tooltip = f"visible transition={name}, label={label}"
        else:
            node_label = "τ"
            shape = "box"
            fill = "#8892a8"
            tooltip = f"invisible transition={name}"
        lines.append(
            f'n{index} [label="{_dot_escape(node_label)}", shape={shape}, '
            f'style="filled", fillcolor="{fill}", fontcolor="white", '
            f'tooltip="{_dot_escape(tooltip)}"];'
        )
    for source, target, _ in graph.edges:
        if visible_node_indices is not None and (
            source not in visible_node_indices or target not in visible_node_indices
        ):
            continue
        if hide_invisible_transitions and (
            graph.node_types[source] == 2 or graph.node_types[target] == 2
        ):
            continue
        lines.append(f"n{source} -> n{target};")
    lines.append("}")
    return "\n".join(lines)


def directly_follows_to_dot(
    artifact: ParsedArtifact,
    *,
    minimum_frequency: int = 1,
    maximum_edges: int = 30,
    activity_search: str = "",
    relative_frequency: bool = False,
) -> str:
    edges = [
        row
        for row in artifact.source_metadata.get("directly_follows_frequencies", [])
        if row["frequency"] >= minimum_frequency
    ]
    if activity_search.strip():
        needle = activity_search.strip().casefold()
        edges = [
            row
            for row in edges
            if needle in row["source"].casefold() or needle in row["target"].casefold()
        ]
    edges = edges[:maximum_edges]
    total_frequency = sum(
        row["frequency"] for row in artifact.source_metadata.get("directly_follows_frequencies", [])
    )
    lines = ["digraph dfg {", "rankdir=LR;"]
    nodes = sorted({row["source"] for row in edges} | {row["target"] for row in edges})
    starts = set(artifact.source_metadata.get("start_activity_frequencies", {}))
    ends = set(artifact.source_metadata.get("end_activity_frequencies", {}))
    for index, label in enumerate(nodes):
        color = "#30c6b0" if label in starts else ("#ffad5a" if label in ends else "#7c6ef6")
        lines.append(
            f'a{index} [label="{_dot_escape(label)}", shape=box, style="filled", '
            f'fillcolor="{color}", fontcolor="white"];'
        )
    node_ids = {label: f"a{index}" for index, label in enumerate(nodes)}
    for row in edges:
        frequency_label = (
            f"{row['frequency'] / max(total_frequency, 1):.1%}"
            if relative_frequency
            else str(row["frequency"])
        )
        lines.append(
            f'{node_ids[row["source"]]} -> {node_ids[row["target"]]} '
            f'[label="{frequency_label}"];'
        )
    lines.append("}")
    return "\n".join(lines)


def _dot_escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
