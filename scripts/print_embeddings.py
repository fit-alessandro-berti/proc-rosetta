#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    activity_mapping_from_labels,
    activity_mapping_from_traces,
    activity_mapping_from_tree,
    canonicalize_traces,
    default_device,
    encode_petri_mu_logvar,
    encode_traces_mu_logvar,
    encode_tree_mu_logvar,
    load_trained_model,
    print_json,
    read_pnml_graph,
    read_ptml_tree,
    read_xes_traces,
    require_supported_input,
    tensor_row_to_list,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read a .xes event log, .pnml Petri net, or .ptml process tree and "
            "print the ProcRosetta embedding from a trained checkpoint."
        )
    )
    parser.add_argument("input", help="input .xes, .pnml, or .ptml file")
    parser.add_argument("--checkpoint", default="checkpoints/proc_rosetta.pt")
    parser.add_argument(
        "--device",
        default=default_device(),
        help="Torch device; defaults to cpu (pass cuda or mps explicitly to override).",
    )
    parser.add_argument("--activity-key", default="concept:name")
    parser.add_argument("--case-id-key", default="case:concept:name")
    parser.add_argument("--max-traces", type=int, default=128)
    parser.add_argument("--max-trace-length", type=int, default=128)
    parser.add_argument("--max-petri-nodes", type=int, default=512)
    parser.add_argument("--auto-guess-final-marking", action="store_true")
    parser.add_argument("--include-logvar", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = require_supported_input(args.input)
    model, device = load_trained_model(args.checkpoint, args.device)

    metadata: dict[str, object] = {
        "input": str(Path(input_path)),
        "checkpoint": str(Path(args.checkpoint)),
        "device": str(device),
    }

    if input_path.suffix.lower() == ".ptml":
        tree = read_ptml_tree(input_path)
        mapping = activity_mapping_from_tree(tree, model.tree_tokenizer.max_activities)
        mu, logvar = encode_tree_mu_logvar(model, tree, device)
        modality = "process_tree"
        source = "proc_rosetta_tree_mu"
        metadata["activity_mapping"] = mapping
    elif input_path.suffix.lower() == ".xes":
        traces = read_xes_traces(
            input_path,
            activity_key=args.activity_key,
            case_id_key=args.case_id_key,
        )
        mapping = activity_mapping_from_traces(traces, model.activity_tokenizer.max_activities)
        canonical_traces = canonicalize_traces(traces, mapping)
        mu, logvar = encode_traces_mu_logvar(
            model,
            canonical_traces,
            device,
            max_traces=args.max_traces,
            max_trace_length=args.max_trace_length,
        )
        modality = "event_log"
        source = "proc_rosetta_trace_mu"
        metadata["activity_mapping"] = mapping
        metadata["trace_count"] = len(traces)
        metadata["used_trace_count"] = min(len(traces), args.max_traces)
    else:
        graph = read_pnml_graph(input_path, auto_guess_final_marking=args.auto_guess_final_marking)
        mapping = activity_mapping_from_labels(
            (label for label in graph.transition_labels if label is not None),
            model.activity_tokenizer.max_activities,
        )
        graph = graph.relabel(mapping)
        mu, logvar = encode_petri_mu_logvar(
            model,
            graph,
            device,
            max_petri_nodes=args.max_petri_nodes,
        )
        modality = "petri_net"
        source = "proc_rosetta_petri_mu"
        metadata["petri_nodes"] = graph.num_nodes
        metadata["petri_edges"] = graph.num_edges
        metadata["visible_transition_labels_used_by_encoder"] = True
        metadata["activity_mapping"] = mapping

    metadata["normalization_version"] = "pm4py-fold-v1"
    metadata["source_activity_alphabet"] = list(mapping)

    output: dict[str, object] = {
        "modality": modality,
        "embedding_name": source,
        "embedding": tensor_row_to_list(mu),
        "metadata": metadata,
    }
    if args.include_logvar:
        output["logvar"] = tensor_row_to_list(logvar)
    print_json(output, pretty=args.pretty)
    return 0


if __name__ == "__main__":
    from _common import run_cli

    raise SystemExit(run_cli(main))
