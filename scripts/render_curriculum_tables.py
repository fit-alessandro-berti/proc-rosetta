#!/usr/bin/env python3
"""Render structural-curriculum configuration tables from the data manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FIELDS = (
    ("Maximum recursion depth", "max_depth"),
    ("Minimum final depth", "min_tree_depth"),
    ("Minimum tree size", "min_tree_size"),
    ("Maximum tree size", "max_tree_size"),
    ("Minimum generated activities", "min_generated_activities"),
    ("Maximum generated activities", "max_generated_activities"),
)
LEVELS = ("simple", "medium", "complex")


def render_markdown(manifest: dict[str, object]) -> str:
    curricula = dict(manifest["curricula"])
    rows = ["| Setting | Simple | Medium | Complex |", "| --- | ---: | ---: | ---: |"]
    for label, field in FIELDS:
        values = [dict(dict(curricula[level])["profile"])[field] for level in LEVELS]
        rows.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")
    return "\n".join(rows)


def render_latex(manifest: dict[str, object]) -> str:
    curricula = dict(manifest["curricula"])
    rows = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Setting & Simple & Medium & Complex \\",
        r"\midrule",
    ]
    for label, field in FIELDS:
        values = [dict(dict(curricula[level])["profile"])[field] for level in LEVELS]
        rows.append(f"{label} & {values[0]} & {values[1]} & {values[2]}" + r" \\")
    rows.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, nargs="?", default=Path("data/curriculum_manifest.json"))
    parser.add_argument("--format", choices=("markdown", "latex"), default="markdown")
    args = parser.parse_args()
    with args.manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    print(render_latex(manifest) if args.format == "latex" else render_markdown(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
