#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    activity_mapping_from_labels,
    add_decode_constraint_arguments,
    default_device,
    decode_tree_from_latent,
    encode_petri_mu_logvar,
    load_trained_model,
    read_pnml_graph,
    relabel_decoded_tree,
    run_cli,
    save_ptml_tree,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read a .pnml Petri net, decode the model's Petri latent into a valid "
            "process tree, and save it as .ptml."
        )
    )
    parser.add_argument("input", help="input .pnml file")
    parser.add_argument("output", help="output .ptml file")
    parser.add_argument("--checkpoint", default="checkpoints/proc_rosetta.pt")
    parser.add_argument(
        "--device",
        default=default_device(),
        help="Torch device; defaults to cuda or mps when available, otherwise cpu.",
    )
    parser.add_argument("--max-petri-nodes", type=int, default=512)
    parser.add_argument("--max-decode-length", type=int, default=512)
    add_decode_constraint_arguments(parser)
    parser.add_argument("--auto-guess-final-marking", action="store_true")
    parser.add_argument(
        "--keep-canonical-labels",
        action="store_true",
        help="save decoded A0/A1/... labels instead of restoring PNML transition labels",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model, device = load_trained_model(args.checkpoint, args.device)
    graph = read_pnml_graph(args.input, auto_guess_final_marking=args.auto_guess_final_marking)
    mapping = activity_mapping_from_labels(
        (label for label in graph.transition_labels if label is not None),
        model.activity_tokenizer.max_activities,
    )
    canonical_graph = graph.relabel(mapping)
    mu, _ = encode_petri_mu_logvar(
        model,
        canonical_graph,
        device,
        max_petri_nodes=args.max_petri_nodes,
    )
    tree, _ = decode_tree_from_latent(
        model,
        mu,
        max_decode_length=args.max_decode_length,
        require_petri_convertible=True,
        canonical_mapping=mapping,
        constrain_source_activities=args.constrain_source_activities,
        avoid_duplicate_transitions=args.avoid_duplicate_transitions,
    )
    if not args.keep_canonical_labels:
        tree = relabel_decoded_tree(tree, mapping)
    save_ptml_tree(tree, args.output)
    print(f"saved decoded process tree to {Path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
