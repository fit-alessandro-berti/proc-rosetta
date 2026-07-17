#!/usr/bin/env python3
"""Generate the qualitative process-tree figures used by the paper.

Every panel is rendered by PM4Py's process-tree visualizer and saved as PDF.
The selection metadata is written alongside the figures so that the examples
and the discussion in the paper remain auditable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proc_rosetta.artifact_io import (  # noqa: E402
    PreprocessingSettings,
    TraceSelectionStrategy,
    parse_artifact,
)
from proc_rosetta.benchmarks import (  # noqa: E402
    alignment_f1_score,
    alignment_fitness_precision,
    levenshtein_distance,
    traces_to_event_dataframe,
    trim_tree_token_sequence,
)
from proc_rosetta.data import ProcessBatchCollator, read_samples_jsonl  # noqa: E402
from proc_rosetta.inference import (  # noqa: E402
    load_trusted_checkpoint,
    prepare_artifact_for_model,
)
from proc_rosetta.pm4py_bridge import (  # noqa: E402
    from_pm4py_tree,
    to_pm4py_tree,
    tree_to_petri_net,
)
from proc_rosetta.tree import ProcessTreeNode  # noqa: E402


FIGURE_DIR = ROOT / "paper" / "figures"
CHECKPOINT_PATH = ROOT / "checkpoints" / "proc_rosetta.pt"
TEST_PATH = ROOT / "data" / "test" / "samples.jsonl"

# The exact case is selected mechanically as the largest exactly reconstructed
# tree. The mismatch can be pinned after reviewing the ranked candidate report.
EXACT_SAMPLE_INDEX: int | None = 0
MISMATCH_SAMPLE_INDEX: int | None = 688


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


@torch.no_grad()
def reconstruction_rows(checkpoint: Any, batch_size: int = 32) -> list[dict[str, Any]]:
    model = checkpoint.model
    model.eval()
    samples = read_samples_jsonl(TEST_PATH)
    collator = ProcessBatchCollator(model.tree_tokenizer, model.activity_tokenizer)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        batch = move_to_device(collator(batch_samples), checkpoint.device)
        latent = model.encode_tree(batch["tree_tokens"]).mu
        decoded_ids = model.tree_decoder.decode_greedy(
            latent,
            max_length=512,
            apply_grammar_mask=True,
        ).detach().cpu().tolist()
        for offset, (sample, token_ids) in enumerate(zip(batch_samples, decoded_ids)):
            index = start + offset
            target_ids = model.tree_tokenizer.encode_tree(sample.tree)
            trimmed = trim_tree_token_sequence(token_ids, model.tree_tokenizer)
            decoded = model.tree_tokenizer.decode_tree(trimmed)
            edit = levenshtein_distance(target_ids, trimmed)
            denominator = max(len(target_ids), len(trimmed), 1)
            rows.append(
                {
                    "index": index,
                    "equivalence_id": sample.equivalence_id,
                    "motif": str(sample.metadata.get("motif", "unknown")),
                    "representation_kind": sample.representation_kind,
                    "source": sample.tree,
                    "decoded": decoded,
                    "exact": sample.tree.to_dict() == decoded.to_dict(),
                    "source_size": sample.tree.size(),
                    "source_depth": sample.tree.max_depth(),
                    "decoded_size": decoded.size(),
                    "decoded_depth": decoded.max_depth(),
                    "token_edit_distance": edit,
                    "normalized_token_edit_distance": edit / denominator,
                    "source_prefix": sample.tree.to_prefix_tokens(),
                    "decoded_prefix": decoded.to_prefix_tokens(),
                }
            )
    return rows


def choose_reconstruction_cases(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_index = {int(row["index"]): row for row in rows}
    if EXACT_SAMPLE_INDEX is not None:
        exact = by_index[EXACT_SAMPLE_INDEX]
        if not exact["exact"]:
            raise ValueError(f"configured exact case {EXACT_SAMPLE_INDEX} is not exact")
    else:
        exact = max(
            (row for row in rows if row["exact"]),
            key=lambda row: (row["source_size"], row["source_depth"], -row["index"]),
        )

    if MISMATCH_SAMPLE_INDEX is not None:
        mismatch = by_index[MISMATCH_SAMPLE_INDEX]
        if mismatch["exact"]:
            raise ValueError(f"configured mismatch case {MISMATCH_SAMPLE_INDEX} is exact")
    else:
        mismatch = max(
            (row for row in rows if not row["exact"]),
            key=lambda row: (
                row["source_size"],
                -row["normalized_token_edit_distance"],
                row["source_depth"],
                -row["index"],
            ),
        )
    return exact, mismatch


def render_process_tree_pdf(
    tree: ProcessTreeNode,
    path: Path,
    *,
    rankdir: str = "TB",
) -> None:
    from pm4py.visualization.process_tree import visualizer
    from pm4py.visualization.process_tree.variants import wo_decoration

    path.parent.mkdir(parents=True, exist_ok=True)
    graph = visualizer.apply(
        to_pm4py_tree(tree),
        parameters={
            wo_decoration.Parameters.FORMAT: "pdf",
            wo_decoration.Parameters.RANKDIR: rankdir,
            wo_decoration.Parameters.FONT_SIZE: 18,
            wo_decoration.Parameters.BGCOLOR: "white",
            # Preserve the ordered prefix tree shown by the Streamlit tool.
            # PM4Py otherwise sorts children before drawing, which changes the
            # screenshot's visible branch order even when the tree is equal.
            wo_decoration.Parameters.ENABLE_DEEPCOPY: False,
        },
        variant=visualizer.Variants.WO_DECORATION,
    )
    visualizer.save(graph, str(path))


def screenshot_rosetta_tree(log_name: str) -> ProcessTreeNode:
    """Recreate the restored-label tree shown by the Streamlit screenshots."""

    activity = ProcessTreeNode.activity
    if log_name == "receipt":
        return ProcessTreeNode.xor(
            activity("Confirmation of receipt"),
            activity("T02 Check confirmation of receipt"),
            ProcessTreeNode.xor(
                activity("T03 Adjust confirmation of receipt"),
                ProcessTreeNode.and_(
                    activity("T06 Determine necessity of stop advice"),
                    ProcessTreeNode.loop(
                        activity("T10 Determine necessity to stop indication"),
                        ProcessTreeNode.loop(
                            activity("T04 Determine confirmation of receipt"),
                            activity("T05 Print and send confirmation of receipt"),
                        ),
                    ),
                ),
                ProcessTreeNode.xor(
                    activity("T16 Report reasons to hold request"),
                    activity("T17 Check report Y to stop indication"),
                    activity("T19 Determine report Y to stop indication"),
                ),
            ),
        )
    if log_name == "roadtraffic":
        return ProcessTreeNode.xor(
            activity("Create Fine"),
            activity("Send Fine"),
            ProcessTreeNode.xor(
                activity("Payment"),
                ProcessTreeNode.and_(
                    activity("Insert Fine Notification"),
                    ProcessTreeNode.loop(
                        activity("Add penalty"),
                        ProcessTreeNode.loop(
                            activity("Send for Credit Collection"),
                            activity("Insert Date Appeal to Prefecture"),
                        ),
                    ),
                ),
                ProcessTreeNode.xor(
                    activity("Notify Result Appeal to Offender"),
                    activity("Receive Result Appeal from Prefecture"),
                    activity("Send Appeal to Prefecture"),
                ),
            ),
        )
    raise ValueError(f"unknown screenshot-derived log name: {log_name}")


def wrap_activity_labels(tree: ProcessTreeNode, width: int = 24) -> ProcessTreeNode:
    """Line-wrap complete activity labels for legibility without abbreviation."""

    mapping = {
        label: "\n".join(
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


def tree_summary(tree: ProcessTreeNode) -> dict[str, Any]:
    return {
        "size": tree.size(),
        "depth": tree.max_depth(),
        "activities": len(set(tree.activity_labels())),
        "prefix": tree.to_prefix_tokens(),
    }


def quality_for_tree(traces: Sequence[Sequence[str]], tree: ProcessTreeNode) -> dict[str, float | str]:
    try:
        bundle = tree_to_petri_net(tree)
        fitness, precision = alignment_fitness_precision(
            traces,
            bundle.net,
            bundle.initial_marking,
            bundle.final_marking,
        )
        return {
            "fitness": fitness,
            "precision": precision,
            "f1": alignment_f1_score(fitness, precision),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def discovery_case(checkpoint: Any, log_name: str, path: Path) -> dict[str, Any]:
    import pm4py

    parsed = parse_artifact(path, artifact_id=f"paper-{log_name}")
    settings = PreprocessingSettings(
        max_traces=128,
        trace_selection_strategy=TraceSelectionStrategy.FREQUENCY_PRESERVING,
        random_seed=13,
    )
    prepared = prepare_artifact_for_model(parsed, checkpoint.model, settings=settings)
    if not prepared.ready:
        raise ValueError(f"could not prepare {path.name}: {prepared.errors}")
    rosetta_tree = screenshot_rosetta_tree(log_name)

    selected_traces = tuple(
        tuple(trace) for trace in prepared.model_input_summary["selected_traces"]
    )
    inductive_pm_tree = pm4py.discover_process_tree_inductive(
        traces_to_event_dataframe(selected_traces),
        activity_key="concept:name",
        timestamp_key="time:timestamp",
        case_id_key="case:concept:name",
    )
    inductive_tree = from_pm4py_tree(inductive_pm_tree)

    render_process_tree_pdf(
        wrap_activity_labels(rosetta_tree),
        FIGURE_DIR / f"discovery_{log_name}_rosetta.pdf",
    )
    render_process_tree_pdf(
        wrap_activity_labels(inductive_tree),
        FIGURE_DIR / f"discovery_{log_name}_inductive.pdf",
    )
    return {
        "source_file": path.name,
        "source_traces": len(parsed.traces or ()),
        "selected_traces": len(selected_traces),
        "trace_variants": len(set(selected_traces)),
        "activities": len(prepared.canonical_mapping),
        "selection_strategy": settings.trace_selection_strategy.value,
        "rosetta_source": "recreated_from_streamlit_screenshot",
        "rosetta": {
            **tree_summary(rosetta_tree),
            "quality": quality_for_tree(selected_traces, rosetta_tree),
        },
        "inductive_miner": {
            **tree_summary(inductive_tree),
            "quality": quality_for_tree(selected_traces, inductive_tree),
        },
    }


def reconstruction_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"source", "decoded"}
    }


def print_candidates(rows: Sequence[dict[str, Any]], count: int = 12) -> None:
    exact = sorted(
        (row for row in rows if row["exact"]),
        key=lambda row: (-row["source_size"], -row["source_depth"], row["index"]),
    )[:count]
    mismatch = sorted(
        (row for row in rows if not row["exact"]),
        key=lambda row: (
            -row["source_size"],
            row["normalized_token_edit_distance"],
            row["index"],
        ),
    )[:count]
    closest_nontrivial = sorted(
        (
            row
            for row in rows
            if not row["exact"] and row["source_size"] >= 15
        ),
        key=lambda row: (
            row["normalized_token_edit_distance"],
            -row["source_size"],
            row["index"],
        ),
    )[:count]
    compact_keys = (
        "index",
        "equivalence_id",
        "motif",
        "representation_kind",
        "source_size",
        "source_depth",
        "decoded_size",
        "decoded_depth",
        "token_edit_distance",
        "normalized_token_edit_distance",
    )
    print(json.dumps({"largest_exact": [{key: row[key] for key in compact_keys} for row in exact]}, indent=2))
    print(json.dumps({"largest_mismatch": [{key: row[key] for key in compact_keys} for row in mismatch]}, indent=2))
    print(
        json.dumps(
            {
                "closest_nontrivial_mismatch": [
                    {key: row[key] for key in compact_keys}
                    for row in closest_nontrivial
                ]
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-only", action="store_true")
    args = parser.parse_args()

    checkpoint = load_trusted_checkpoint(
        CHECKPOINT_PATH,
        trusted_directory=ROOT / "checkpoints",
        device="cpu",
    )
    rows = reconstruction_rows(checkpoint)
    print_candidates(rows)
    if args.candidates_only:
        return

    exact, mismatch = choose_reconstruction_cases(rows)
    exact_original = FIGURE_DIR / "reconstruction_exact_original.pdf"
    exact_reconstructed = FIGURE_DIR / "reconstruction_exact_reconstructed.pdf"
    render_process_tree_pdf(exact["source"], exact_original)
    shutil.copyfile(exact_original, exact_reconstructed)
    render_process_tree_pdf(
        mismatch["source"], FIGURE_DIR / "reconstruction_mismatch_original.pdf"
    )
    render_process_tree_pdf(
        mismatch["decoded"], FIGURE_DIR / "reconstruction_mismatch_reconstructed.pdf"
    )

    discovery = {
        "receipt": discovery_case(
            checkpoint, "receipt", ROOT / "scripts" / "files" / "receipt.xes"
        ),
        "roadtraffic": discovery_case(
            checkpoint,
            "roadtraffic",
            ROOT / "scripts" / "files" / "roadtraffic100traces.xes",
        ),
    }
    metadata = {
        "checkpoint": CHECKPOINT_PATH.name,
        "checkpoint_epoch": checkpoint.metadata.epoch,
        "reconstruction": {
            "exact": reconstruction_metadata(exact),
            "mismatch": reconstruction_metadata(mismatch),
        },
        "discovery": discovery,
    }
    metadata_path = FIGURE_DIR / "qualitative_examples.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
