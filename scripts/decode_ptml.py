#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    activity_mapping_from_tree,
    decode_tree_from_latent,
    encode_tree_mu_logvar,
    load_trained_model,
    read_ptml_tree,
    relabel_decoded_tree,
    run_cli,
    save_ptml_tree,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read a .ptml process tree, encode it with the trained model, decode a "
            "valid process tree, and save the decoded tree to a .ptml file."
        )
    )
    parser.add_argument("input", help="input .ptml file")
    parser.add_argument("output", help="output .ptml file")
    parser.add_argument("--checkpoint", default="checkpoints/proc_rosetta.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-decode-length", type=int, default=128)
    parser.add_argument(
        "--keep-canonical-labels",
        action="store_true",
        help="save decoded A0/A1/... activity labels instead of restoring input labels",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model, device = load_trained_model(args.checkpoint, args.device)
    tree = read_ptml_tree(args.input)
    mapping = activity_mapping_from_tree(tree, model.tree_tokenizer.max_activities)
    mu, _ = encode_tree_mu_logvar(model, tree, device)
    decoded_tree, _ = decode_tree_from_latent(
        model,
        mu,
        max_decode_length=args.max_decode_length,
        require_petri_convertible=True,
    )
    if not args.keep_canonical_labels:
        decoded_tree = relabel_decoded_tree(decoded_tree, mapping)
    save_ptml_tree(decoded_tree, args.output)
    print(f"saved decoded process tree to {Path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
