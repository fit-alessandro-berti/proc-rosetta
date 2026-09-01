#!/usr/bin/env python3
"""Reproduce the two real-log discovery cases reported in the manuscript.

The script encodes each selected log sample once, decodes the same latent with
and without the repeated-visible-label mask, evaluates both trees and an
Inductive Miner baseline with footprint conformance, and writes a compact JSON
record.  It also asks PM4Py for the representative process-tree visualization
and Graphviz layout, then emits the layout as repository-native TikZ; no PDF or
raster figure is created.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
import sys
import textwrap
from typing import Any, Iterable, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("PM4PY_SHOW_PROGRESS_BAR", "False")
os.environ.setdefault("PM4PY_SHOW_INTERNAL_WARNINGS", "False")
sys._pm4py_welcome_shown = True

import pm4py  # noqa: E402

from proc_rosetta.artifact_io import (  # noqa: E402
    PreprocessingSettings,
    TraceSelectionStrategy,
    parse_artifact,
)
from proc_rosetta.benchmarks import (  # noqa: E402
    fitness_precision_f1_score,
    footprint_fitness_precision,
    traces_to_event_dataframe,
)
from proc_rosetta.inference import (  # noqa: E402
    decode_guaranteed,
    encode_artifact,
    load_trusted_checkpoint,
    prepare_artifact_for_model,
)
from proc_rosetta.pm4py_bridge import from_pm4py_tree, to_pm4py_tree  # noqa: E402
from proc_rosetta.tree import ProcessTreeNode  # noqa: E402


CHECKPOINT = ROOT / "checkpoints" / "proc_rosetta.best.pt"
OUTPUT_JSON = ROOT / "paper" / "real_log_evaluation.json"
OUTPUT_FIGURE = ROOT / "paper" / "figures" / "real-log-tree.tex"
LOGS = {
    "receipt": {
        "path": ROOT / "scripts" / "files" / "receipt.xes",
        "title": "Receipt phase of an environmental permit application process",
        "doi": "10.4121/uuid:a07386a5-7be3-4367-9535-70bc9e77dbe6",
    },
    "road_traffic": {
        "path": ROOT / "scripts" / "files" / "roadtraffic100traces.xes",
        "title": "Road Traffic Fine Management Process (supplied 100-case extract)",
        "doi": "10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5",
    },
}


def file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duplicate_count(tree: ProcessTreeNode) -> int:
    return sum(count - 1 for count in Counter(tree.activity_labels()).values())


def tree_record(
    tree: ProcessTreeNode,
    traces: Sequence[Sequence[str]],
    *,
    event_log: Any,
    log_footprints: Any,
) -> dict[str, Any]:
    fitness, precision = footprint_fitness_precision(
        traces,
        to_pm4py_tree(tree),
        event_log=event_log,
        log_footprints=log_footprints,
    )
    labels = tree.activity_labels()
    return {
        "tree_size": tree.size(),
        "tree_depth": tree.max_depth(),
        "visible_leaves": len(labels),
        "distinct_visible_activities": len(set(labels)),
        "repeated_visible_leaves": duplicate_count(tree),
        "footprint_fitness": round(fitness, 6),
        "footprint_precision": round(precision, 6),
        "footprint_f1": fitness_precision_f1_score(fitness, precision),
        "prefix_tokens": tree.to_prefix_tokens(),
    }


def evaluate_log(checkpoint: Any, key: str, specification: dict[str, Any]) -> tuple[dict[str, Any], dict[str, ProcessTreeNode]]:
    path = Path(specification["path"])
    parsed = parse_artifact(path, artifact_id=f"paper-real-log-{key}")
    settings = PreprocessingSettings(
        max_traces=128,
        max_trace_length=128,
        trace_selection_strategy=TraceSelectionStrategy.FREQUENCY_PRESERVING,
        random_seed=13,
    )
    prepared = prepare_artifact_for_model(parsed, checkpoint.model, settings=settings)
    if not prepared.ready:
        raise ValueError(f"could not prepare {path.name}: {prepared.errors}")
    encoding = encode_artifact(prepared, checkpoint)
    if encoding.errors:
        raise ValueError(f"could not encode {path.name}: {encoding.errors}")

    selected_traces = tuple(
        tuple(trace) for trace in prepared.model_input_summary["selected_traces"]
    )
    event_log = traces_to_event_dataframe(selected_traces)
    log_footprints = pm4py.discover_footprints(event_log)
    trees: dict[str, ProcessTreeNode] = {}
    modes = (
        ("duplicates_allowed", False, "allow"),
        ("duplicates_disallowed", True, "disallow"),
    )
    methods: dict[str, Any] = {}
    for mode, avoid_duplicates, duplicate_policy in modes:
        result = decode_guaranteed(
            checkpoint,
            encoding.mu,
            source_artifact_ids=[encoding.artifact_id],
            source_modalities=[encoding.modality],
            latent_source="event_log_mu",
            canonical_mapping=encoding.canonical_mapping,
            total_token_budget_including_bos_eos=512,
            top_k=5,
            beam_size=2,
            allowed_activity_slots=encoding.allowed_activity_slots,
            copy_activity_slots=encoding.copy_activity_slots,
            activity_memory=encoding.activity_memory,
            constrain_to_source_activities=True,
            avoid_duplicate_activity_labels=avoid_duplicates,
            duplicate_policy=duplicate_policy,
        )
        tree = result.restored_tree or result.tree
        if tree is None or not result.successful:
            raise ValueError(f"{path.name} {mode} decode failed: {result.errors}")
        trees[mode] = tree
        methods[mode] = {
            "duplicate_policy": duplicate_policy,
            "grammar_valid": result.grammar_valid,
            "petri_convertible": result.petri_convertible,
            "fallback_used": result.fallback_used,
            "forced_closure_used": result.forced_closure_used,
            **tree_record(
                tree,
                selected_traces,
                event_log=event_log,
                log_footprints=log_footprints,
            ),
        }

    inductive_pm_tree = pm4py.discover_process_tree_inductive(
        event_log,
        activity_key="concept:name",
        timestamp_key="time:timestamp",
        case_id_key="case:concept:name",
    )
    inductive_tree = from_pm4py_tree(inductive_pm_tree)
    trees["inductive_miner"] = inductive_tree
    methods["inductive_miner"] = tree_record(
        inductive_tree,
        selected_traces,
        event_log=event_log,
        log_footprints=log_footprints,
    )

    source = parsed.source_metadata
    record = {
        "title": specification["title"],
        "doi": specification["doi"],
        "source_file": path.relative_to(ROOT).as_posix(),
        "source_sha256": file_digest(path),
        "source_traces": source["total_traces"],
        "source_events": source["total_events"],
        "source_trace_variants": source["trace_variants"],
        "source_distinct_activities": source["distinct_activities"],
        "source_maximum_trace_length": source["maximum_trace_length"],
        "selected_traces": len(selected_traces),
        "selected_events": sum(map(len, selected_traces)),
        "selected_trace_variants": len(set(selected_traces)),
        "selection_strategy": settings.trace_selection_strategy.value,
        "selection_seed": settings.random_seed,
        "duplicate_policy_outputs_identical": (
            trees["duplicates_allowed"].to_prefix_tokens()
            == trees["duplicates_disallowed"].to_prefix_tokens()
        ),
        "methods": methods,
    }
    return record, trees


def wrapped_tree(tree: ProcessTreeNode, width: int = 19) -> ProcessTreeNode:
    mapping = {
        label: r"\n".join(
            textwrap.wrap(
                label,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
        for label in tree.unique_activity_labels()
    }
    return tree.relabel(mapping)


def walk_pm_tree(root: Any) -> Iterable[Any]:
    yield root
    for child in root.children:
        yield from walk_pm_tree(child)


def pm_node_label(node: Any) -> tuple[str, bool]:
    if node.operator is not None:
        name = getattr(node.operator, "name", str(node.operator)).upper()
        return {"SEQUENCE": "SEQ", "PARALLEL": "AND", "XOR": "XOR", "LOOP": "LOOP"}.get(name, name), True
    if node.label is None:
        return r"$\tau$", False
    return str(node.label), False


def tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    lines = []
    for line in value.replace(r"\n", "\n").splitlines() or [value]:
        if line == r"$\tau$":
            lines.append(line)
            continue
        escaped = "".join(replacements.get(character, character) for character in line)
        lines.append(escaped)
    return r"\\".join(lines)


def graphviz_layout(tree: ProcessTreeNode) -> tuple[dict[str, tuple[float, float, float, float]], list[tuple[str, str]], dict[str, tuple[str, bool]], float, float]:
    from pm4py.visualization.process_tree import visualizer
    from pm4py.visualization.process_tree.variants import wo_decoration

    pm_tree = to_pm4py_tree(wrapped_tree(tree))
    labels = {str(id(node)): pm_node_label(node) for node in walk_pm_tree(pm_tree)}
    graph = visualizer.apply(
        pm_tree,
        parameters={
            wo_decoration.Parameters.FORMAT: "dot",
            wo_decoration.Parameters.RANKDIR: "TB",
            wo_decoration.Parameters.FONT_SIZE: 11,
            wo_decoration.Parameters.BGCOLOR: "white",
            wo_decoration.Parameters.ENABLE_DEEPCOPY: False,
        },
        variant=visualizer.Variants.WO_DECORATION,
    )
    nodes: dict[str, tuple[float, float, float, float]] = {}
    edges: list[tuple[str, str]] = []
    graph_width = graph_height = 1.0
    for raw_line in graph.pipe(format="plain").decode("utf-8").splitlines():
        fields = shlex.split(raw_line)
        if not fields:
            continue
        if fields[0] == "graph":
            graph_width, graph_height = float(fields[2]), float(fields[3])
        elif fields[0] == "node":
            nodes[fields[1]] = tuple(map(float, fields[2:6]))  # type: ignore[assignment]
        elif fields[0] == "edge":
            edges.append((fields[1], fields[2]))
    missing = set(nodes) - set(labels)
    if missing:
        raise ValueError(f"PM4Py visualization contained unknown node identifiers: {missing}")
    return nodes, edges, labels, graph_width, graph_height


def render_tikz(tree: ProcessTreeNode, output_path: Path) -> None:
    nodes, edges, labels, graph_width, graph_height = graphviz_layout(tree)
    scale = min(15.2 / graph_width, 11.0 / graph_height)
    lines = [
        r"\begin{figure}[!htbp]",
        r"\centering",
        r"\resizebox{0.98\textwidth}{!}{%",
        r"\begin{tikzpicture}[",
        r"  op/.style={circle,draw=prgray,fill=prlightgold,minimum size=6.5mm,font=\scriptsize\bfseries,inner sep=1pt},",
        r"  leaf/.style={ellipse,draw=prgray!75,fill=prlightblue,align=center,font=\scriptsize,inner sep=2.5pt},",
        r"  branch/.style={draw=prgray!75,line width=0.45pt}",
        r"]",
    ]
    for index, (node_id, (x, y, width, height)) in enumerate(nodes.items()):
        label, is_operator = labels[node_id]
        style = "op" if is_operator else "leaf"
        node_name = f"pt{index}"
        labels[node_id] = (node_name, is_operator)
        minimum_width = max(0.65 if is_operator else 1.1, width * scale)
        minimum_height = max(0.65 if is_operator else 0.55, height * scale)
        lines.append(
            rf"\node[{style},minimum width={minimum_width:.2f}cm,minimum height={minimum_height:.2f}cm] "
            rf"({node_name}) at ({x * scale:.3f},{y * scale:.3f}) {{{tex_escape(label)}}};"
        )
    for source, target in edges:
        lines.append(rf"\draw[branch] ({labels[source][0]}) -- ({labels[target][0]});")
    lines.extend(
        [
            r"\end{tikzpicture}",
            r"}%",
            r"\caption{Road-traffic process tree decoded by \model. PM4Py supplies the process-tree representation and Graphviz layout, which are emitted here as TikZ. Allowing or disallowing repeated visible labels produces this same top-ranked tree.}",
            r"\label{fig:real-log-tree}",
            r"\end{figure}",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-figure", type=Path, default=OUTPUT_FIGURE)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = load_trusted_checkpoint(
        checkpoint_path,
        trusted_directory=checkpoint_path.parent,
        device=args.device,
    )
    checkpoint.model.eval()
    records: dict[str, Any] = {}
    trees: dict[str, dict[str, ProcessTreeNode]] = {}
    for key, specification in LOGS.items():
        records[key], trees[key] = evaluate_log(checkpoint, key, specification)

    output = {
        "description": (
            "Deterministic illustrative real-log comparison. Each selected trace sample is "
            "encoded once; duplicate-policy decodes differ only in the repeated-visible-label mask."
        ),
        "checkpoint": {
            "identifier": checkpoint.metadata.identifier,
            "filename": checkpoint.metadata.filename,
            "epoch": checkpoint.metadata.epoch,
            "sha256": file_digest(checkpoint_path),
        },
        "software": {"pm4py": pm4py.__version__, "torch": torch.__version__},
        "protocol": {
            "trace_selection": "frequency_preserving",
            "selection_seed": 13,
            "maximum_encoder_traces": 128,
            "maximum_trace_length": 128,
            "beam_width": 2,
            "decode_token_budget_including_bos_eos": 512,
            "source_activity_constraint": True,
            "conformance": "PM4Py extensive-log footprint fitness and precision on the exact encoder trace sample",
            "f1": "harmonic mean of footprint fitness and footprint precision",
        },
        "logs": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    render_tikz(trees["road_traffic"]["duplicates_allowed"], args.output_figure)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
