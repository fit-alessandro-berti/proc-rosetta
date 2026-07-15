from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from proc_rosetta_ui.common import checkpoint_sidebar, configure_page, page_header
from proc_rosetta.evaluation_iterators import (
    cross_modal_retrieval_iter,
    decode_quality_iter,
    discovery_comparison_iter,
    embedding_baseline_iter,
    neural_loss_iter,
)
from proc_rosetta.benchmarks import Pm4pyPetriEmbeddingConfig
from proc_rosetta_ui.app_state import workspace
from proc_rosetta_ui.cache_service import cache_key, cache_put, stage_cache
from proc_rosetta_ui.visualizations.evaluation_charts import render_decode_quality


configure_page("Quality evaluation", "◎")
checkpoint = checkpoint_sidebar()
page_header(
    "View 04 / quality evaluation",
    "Results arrive while the run is still alive.",
    "Evaluate each workspace artifact independently. Completed rows remain inspectable as later "
    "decodes, conversions, simulations, and comparisons continue.",
)
items = [
    item
    for item in workspace(st.session_state).values()
    if item.encoding is not None
    and item.encoding.mu
    and item.encoding.checkpoint_identifier == checkpoint.metadata.identifier
]
if not items:
    st.info("Encode one or more artifacts with the active checkpoint before evaluation.")
    st.stop()

scope, controls = st.columns([2, 1])
with scope:
    selected_ids = st.multiselect(
        "Evaluation artifacts",
        [item.artifact_id for item in items],
        default=[item.artifact_id for item in items],
        format_func=lambda artifact_id: next(item.parsed.display_name for item in items if item.artifact_id == artifact_id),
    )
with controls:
    max_length = st.slider("Decode token limit", 16, 1024, 256, 16)
    simulation_count = st.number_input("Quick simulation traces", 10, 10_000, 100, 10)

tabs = st.tabs(
    [
        "Decode & behavior",
        "Neural losses",
        "Cross-modal retrieval",
        "Discovery comparison",
        "Embedding baselines",
    ]
)
with tabs[0]:
    st.caption(
        "Quick distribution comparison runs automatically. Expensive alignment fitness and precision "
        "are deliberately kept as a separate manual stage."
    )
    exact_behavior = st.checkbox("Also run expensive alignment fitness and precision", False)
    if st.button("Run progressive evaluation", type="primary", disabled=not selected_ids):
        chosen = [item for item in items if item.artifact_id in selected_ids]
        evaluation_cache = stage_cache(st.session_state, "evaluation")
        key = cache_key(
            "decode_quality",
            checkpoint.metadata.identifier,
            selected_ids,
            max_length,
            simulation_count,
            exact_behavior,
        )
        rows = evaluation_cache.get(key)
        if rows is None:
            rows = []
            progress = st.progress(0.0, text="Starting evaluation…")
            table = st.empty()
            for update in decode_quality_iter(
                chosen,
                checkpoint,
                max_length=max_length,
                simulated_traces=int(simulation_count),
                exact_conformance=exact_behavior,
            ):
                rows.append(update.result)
                table.dataframe(rows, use_container_width=True, hide_index=True)
                progress.progress(
                    update.completed / max(update.total, 1),
                    text=f"Completed {update.completed}/{update.total} · {update.elapsed_seconds:.1f}s",
                )
            cache_put(evaluation_cache, key, rows)
        else:
            st.info("Loaded decode-quality results from the evaluation cache.")
        st.session_state["evaluation_results"] = rows
    render_decode_quality(st.session_state.get("evaluation_results", []))
    completed_rows = st.session_state.get("evaluation_results", [])
    if completed_rows:
        failed_names = [
            row["artifact"]
            for row in completed_rows
            if not row.get("valid_tree") or not row.get("petri_conversion")
        ]
        failed_items = [
            item
            for item in items
            if item.parsed.display_name in failed_names
        ]
        if failed_items:
            failed_id = st.selectbox(
                "Failed example",
                [item.artifact_id for item in failed_items],
                format_func=lambda artifact_id: next(
                    item.parsed.display_name
                    for item in failed_items
                    if item.artifact_id == artifact_id
                ),
            )
            if st.button("Open failed example in Translation studio"):
                st.session_state["translation_prefill_ids"] = [failed_id]
                st.switch_page("pages/02_translation.py")
        exports = st.columns(2)
        exports[0].download_button(
            "Download evaluation JSON",
            json.dumps(completed_rows, indent=2, default=str),
            "proc-rosetta-evaluation.json",
            "application/json",
            use_container_width=True,
        )
        exports[1].download_button(
            "Download evaluation CSV",
            pd.DataFrame(completed_rows).to_csv(index=False),
            "proc-rosetta-evaluation.csv",
            "text/csv",
            use_container_width=True,
        )
with tabs[1]:
    data_root = Path(os.environ.get("PROC_ROSETTA_DATA_DIR", "data")).expanduser()
    split = st.selectbox("Server-side split", ["test", "validation", "training"])
    sample_path = data_root / split / "samples.jsonl"
    neural_controls = st.columns(2)
    neural_batch_size = int(neural_controls[0].number_input("Batch size", 1, 512, 16))
    neural_max_batches = int(neural_controls[1].number_input("Maximum batches", 1, 10_000, 20))
    deterministic_neural = st.checkbox("Use deterministic latent means", True)
    neural_seed = int(st.number_input("Neural evaluation seed", value=13))
    st.caption(f"Configured sample file: `{sample_path}`")
    if not sample_path.exists():
        st.info(
            "No server-side synthetic split is installed at this path. Set `PROC_ROSETTA_DATA_DIR` "
            "to enable progressive neural-loss evaluation."
        )
    elif st.button("Run neural-loss evaluation"):
        evaluation_cache = stage_cache(st.session_state, "evaluation")
        key = cache_key(
            "neural",
            checkpoint.metadata.identifier,
            str(sample_path),
            neural_batch_size,
            neural_max_batches,
            deterministic_neural,
            neural_seed,
        )
        rows = evaluation_cache.get(key)
        if rows is None:
            rows = []
            progress = st.progress(0.0)
            chart = st.empty()
            for update in neural_loss_iter(
                checkpoint,
                str(sample_path),
                batch_size=neural_batch_size,
                max_batches=neural_max_batches,
                deterministic=deterministic_neural,
                random_seed=neural_seed,
            ):
                rows.append(update.result)
                progress.progress(update.completed / update.total, text=f"Batch {update.completed}/{update.total}")
                running = [
                    {name: value for name, value in row.items() if name == "batch" or name.startswith("running_")}
                    for row in rows
                ]
                chart.line_chart(pd.DataFrame(running).set_index("batch"))
            cache_put(evaluation_cache, key, rows)
        else:
            st.info("Loaded neural-loss results from the evaluation cache.")
        st.session_state["neural_evaluation_results"] = rows
    if st.session_state.get("neural_evaluation_results"):
        st.dataframe(st.session_state["neural_evaluation_results"], use_container_width=True, hide_index=True)
with tabs[2]:
    st.markdown(
        "Evaluate all six directions on process groups that contain aligned modalities. Top-k, MRR, "
        "mean/median rank, and rank histograms update one direction at a time."
    )
    if st.button("Run cross-modal retrieval"):
        evaluation_cache = stage_cache(st.session_state, "evaluation")
        key = cache_key("retrieval", checkpoint.metadata.identifier, [(item.artifact_id, item.process_group) for item in items])
        rows = evaluation_cache.get(key)
        if rows is None:
            rows = []
            progress = st.progress(0.0)
            table = st.empty()
            for update in cross_modal_retrieval_iter(items):
                rows.append(update.result)
                progress.progress(update.completed / update.total, text=update.artifact_id)
                table.dataframe(rows, use_container_width=True, hide_index=True)
            cache_put(evaluation_cache, key, rows)
        else:
            st.info("Loaded retrieval results from the evaluation cache.")
        st.session_state["retrieval_evaluation_results"] = rows
    if st.session_state.get("retrieval_evaluation_results"):
        retrieval_rows = st.session_state["retrieval_evaluation_results"]
        st.dataframe(retrieval_rows, use_container_width=True, hide_index=True)
        retrieval_frame = pd.DataFrame(retrieval_rows).set_index("direction")
        retrieval_columns = [
            name
            for name in ("top1_accuracy", "top3_accuracy", "top5_accuracy", "mrr")
            if name in retrieval_frame
        ]
        if retrieval_columns:
            st.bar_chart(retrieval_frame[retrieval_columns])
with tabs[3]:
    st.warning(
        "ProcRosetta and Inductive Miner solve different optimization problems. Exact alignment fitness "
        "and precision can be expensive and run only when explicitly enabled."
    )
    exact = st.checkbox("Run alignment fitness and precision", False)
    if st.button("Compare with Inductive Miner"):
        evaluation_cache = stage_cache(st.session_state, "evaluation")
        key = cache_key("discovery", checkpoint.metadata.identifier, selected_ids, max_length, exact)
        rows = evaluation_cache.get(key)
        if rows is None:
            rows = []
            progress = st.progress(0.0)
            table = st.empty()
            for update in discovery_comparison_iter(
                [item for item in items if item.artifact_id in selected_ids],
                checkpoint,
                max_length=max_length,
                exact_conformance=exact,
            ):
                rows.append(update.result)
                progress.progress(update.completed / max(update.total, 1), text=f"{update.result['artifact']} · {update.result['method']}")
                table.dataframe(rows, use_container_width=True, hide_index=True)
            cache_put(evaluation_cache, key, rows)
        else:
            st.info("Loaded discovery results from the evaluation cache.")
        st.session_state["discovery_evaluation_results"] = rows
    if st.session_state.get("discovery_evaluation_results"):
        discovery_rows = st.session_state["discovery_evaluation_results"]
        st.dataframe(discovery_rows, use_container_width=True, hide_index=True)
        discovery_frame = pd.DataFrame(discovery_rows)
        if "f1" in discovery_frame and discovery_frame["f1"].notna().any():
            st.bar_chart(
                discovery_frame.pivot_table(
                    index="artifact",
                    columns="method",
                    values="f1",
                    aggfunc="mean",
                )
            )
with tabs[4]:
    st.markdown(
        "Compare ProcRosetta event-log means with activity-count, variant, directly-follows, and "
        "eventually-follows vectors. At least two event logs are required; larger sets give stronger evidence."
    )
    include_node2vec = st.checkbox("Include optional PM4Py Petri Node2Vec", False)
    if include_node2vec:
        node_controls = st.columns(3)
        node_dimensions = int(node_controls[0].number_input("Node2Vec dimensions", 4, 512, 64))
        node_walks = int(node_controls[1].number_input("Walks per node", 1, 100, 5))
        node_epochs = int(node_controls[2].number_input("Node2Vec epochs", 1, 100, 5))
    else:
        node_dimensions, node_walks, node_epochs = 64, 5, 5
    if st.button("Run embedding baselines"):
        evaluation_cache = stage_cache(st.session_state, "evaluation")
        key = cache_key("baselines", checkpoint.metadata.identifier, selected_ids, include_node2vec, node_dimensions, node_walks, node_epochs)
        rows = evaluation_cache.get(key)
        if rows is None:
            rows = []
            progress = st.progress(0.0)
            details = st.empty()
            for update in embedding_baseline_iter(
                [item for item in items if item.artifact_id in selected_ids],
                include_petri_node2vec=include_node2vec,
                petri_config=Pm4pyPetriEmbeddingConfig(
                    dimensions=node_dimensions,
                    num_walks=node_walks,
                    epochs=node_epochs,
                ),
            ):
                rows.append(update.result)
                progress.progress(update.completed / update.total, text=update.artifact_id)
                details.json(rows)
            cache_put(evaluation_cache, key, rows)
        else:
            st.info("Loaded embedding-baseline results from the evaluation cache.")
        st.session_state["embedding_baseline_results"] = rows
    if st.session_state.get("embedding_baseline_results"):
        baseline_results = st.session_state["embedding_baseline_results"]
        st.json(baseline_results)
        summary_rows = []
        for result in baseline_results:
            alignment = result.get("behavior_alignment", {})
            neighbor = result.get("nearest_neighbor_behavior", {})
            agreement = result.get("neighbor_agreement", {})
            summary_rows.append(
                {
                    "Method": result.get("method"),
                    "Dimension": result.get("dimension"),
                    "Behavior Spearman": alignment.get(
                        "spearman_embedding_distance_vs_behavior_l1"
                    ),
                    "Behavior Pearson": alignment.get(
                        "pearson_embedding_distance_vs_behavior_l1"
                    ),
                    "Nearest-neighbor behavior": neighbor.get(
                        "mean_behavior_l1_at_nearest_neighbor"
                    ),
                    "Improvement over random": neighbor.get("improvement_over_random"),
                    "Top-1 neighbor agreement": agreement.get("top1_agreement"),
                    "Top-3 neighbor agreement": agreement.get("top3_agreement"),
                    "Runtime": result.get("runtime_seconds"),
                }
            )
        ranked = sorted(
            summary_rows,
            key=lambda row: (
                row["Behavior Spearman"] is not None,
                row["Behavior Spearman"] if row["Behavior Spearman"] is not None else float("-inf"),
            ),
            reverse=True,
        )
        for rank, row in enumerate(ranked, 1):
            row["Rank"] = rank
        summary_rows = ranked
        st.dataframe(summary_rows, use_container_width=True, hide_index=True)
        chart_frame = pd.DataFrame(summary_rows).set_index("Method")
        chart_columns = [
            column
            for column in ("Behavior Spearman", "Nearest-neighbor behavior")
            if chart_frame[column].notna().any()
        ]
        if chart_columns:
            st.bar_chart(chart_frame[chart_columns])

st.markdown("### Complete evaluation report")
report = {
    "checkpoint": checkpoint.metadata.identifier,
    "configuration": {
        "selected_artifact_ids": selected_ids,
        "maximum_decode_length": max_length,
        "simulation_trace_count": int(simulation_count),
        "exact_behavioral_conformance": exact_behavior,
    },
    "decode_and_behavior": st.session_state.get("evaluation_results", []),
    "neural_losses": st.session_state.get("neural_evaluation_results", []),
    "cross_modal_retrieval": st.session_state.get("retrieval_evaluation_results", []),
    "discovery_comparison": st.session_state.get("discovery_evaluation_results", []),
    "embedding_baselines": st.session_state.get("embedding_baseline_results", []),
}
st.download_button(
    "Download complete JSON report",
    json.dumps(report, indent=2, default=str),
    "proc-rosetta-complete-evaluation.json",
    "application/json",
)
