#!/usr/bin/env python3
"""Migrate existing ProcRosetta JSONL splits to folded per-modality targets."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from _common import run_cli
from proc_rosetta.data import (
    METADATA_FILENAME,
    SPLIT_NAMES,
    read_samples_jsonl,
    split_samples_path,
    write_samples_jsonl,
)
from proc_rosetta.pm4py_bridge import TREE_NORMALIZATION_VERSION, fold_process_tree
from proc_rosetta.synthetic import decoder_target_trees_for_sample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate every data split to schema v5 semantic folding and decoder targets."
    )
    parser.add_argument("data_dir", help="directory containing metadata.json and split JSONL files")
    return parser


def main(argv: list[str] | None = None) -> int:
    data_dir = Path(build_parser().parse_args(argv).data_dir)
    metadata_path = data_dir / METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    changed = 0
    total = 0
    for split in SPLIT_NAMES:
        path = split_samples_path(data_dir, split)
        samples = read_samples_jsonl(path)
        migrated = []
        for sample in samples:
            semantic_tree = fold_process_tree(sample.tree)
            changed += int(semantic_tree.canonical_key() != sample.tree.canonical_key())
            total += 1
            migrated.append(
                replace(
                    sample,
                    tree=semantic_tree,
                    decoder_target_trees=decoder_target_trees_for_sample(
                        semantic_tree,
                        sample.traces,
                        sample.petri_graph,
                    ),
                    metadata={
                        **sample.metadata,
                        "normalization_version": TREE_NORMALIZATION_VERSION,
                    },
                )
            )
        write_samples_jsonl(path, migrated)
    metadata.update(
        version=5,
        schema="proc-rosetta.behavior-family-splits.v5",
        sample_format="jsonl/process-sample.v4",
        tree_normalization_version=TREE_NORMALIZATION_VERSION,
        migrated_sample_count=total,
        fold_changed_sample_count=changed,
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"migrated {total} samples ({changed} syntax changes) to schema v5")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
