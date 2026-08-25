#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    add_decode_constraint_arguments,
    activity_mapping_from_traces,
    canonicalize_traces,
    default_device,
    decode_tree_from_latent,
    encode_traces_distribution,
    load_trained_model,
    read_xes_traces,
    relabel_decoded_tree,
    run_cli,
    save_ptml_tree,
)
from proc_rosetta.artifact_io import ArtifactModality


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read a .xes event log, decode the model's trace latent into a valid "
            "process tree, and save it as .ptml."
        )
    )
    parser.add_argument("input", help="input .xes file")
    parser.add_argument("output", help="output .ptml file")
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
    parser.add_argument("--max-decode-length", type=int, default=512)
    add_decode_constraint_arguments(parser)
    parser.add_argument(
        "--keep-canonical-labels",
        action="store_true",
        help="save decoded A0/A1/... activity labels instead of restoring input labels",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model, device = load_trained_model(args.checkpoint, args.device)
    traces = read_xes_traces(
        args.input,
        activity_key=args.activity_key,
        case_id_key=args.case_id_key,
    )
    mapping = activity_mapping_from_traces(traces, model.activity_tokenizer.max_activities)
    canonical_traces = canonicalize_traces(traces, mapping)
    distribution = encode_traces_distribution(
        model,
        canonical_traces,
        device,
        max_traces=args.max_traces,
        max_trace_length=args.max_trace_length,
    )
    tree, _ = decode_tree_from_latent(
        model,
        distribution,
        max_decode_length=args.max_decode_length,
        require_petri_convertible=True,
        canonical_mapping=mapping,
        constrain_source_activities=args.constrain_source_activities,
        avoid_duplicate_transitions=args.avoid_duplicate_transitions,
        source_modality=ArtifactModality.EVENT_LOG,
    )
    if not args.keep_canonical_labels:
        tree = relabel_decoded_tree(tree, mapping)
    save_ptml_tree(tree, args.output)
    print(f"saved decoded process tree to {Path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
