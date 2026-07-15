from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json

import pandas as pd
import streamlit as st

from proc_rosetta_ui.common import (
    PETRI_WARNING,
    checkpoint_metadata_dict,
    checkpoint_sidebar,
    configure_page,
    page_header,
)
from proc_rosetta.artifact_io import (
    ArtifactModality,
    ArtifactParseSettings,
    PreprocessingSettings,
    TraceSelectionStrategy,
    canonical_mapping_rows,
)
from proc_rosetta.inference import prepare_artifact_for_model
from proc_rosetta_ui.app_state import workspace
from proc_rosetta_ui.cache_service import stage_cache
from proc_rosetta_ui.encoding_service import encode_workspace_items
from proc_rosetta_ui.export_service import complete_workspace_zip
from proc_rosetta_ui.visualizations.event_log import render_event_log
from proc_rosetta_ui.visualizations.petri_net import render_petri_net
from proc_rosetta_ui.visualizations.process_tree import render_process_tree
from proc_rosetta_ui.workspace_service import (
    add_uploaded_artifact,
    bundled_artifact_paths,
    set_process_group,
    workspace_rows,
)


configure_page("Artifact workspace", "◫")
checkpoint = checkpoint_sidebar()
page_header(
    "View 01 / artifact workspace",
    "Import once. Explore immediately.",
    "Upload XES, PTML, and PNML artifacts or choose a bundled example. Every imported artifact "
    "is parsed, canonicalized, and embedded automatically with the active checkpoint.",
)
items = workspace(st.session_state)

notice = st.session_state.pop("workspace_import_notice", None)
if notice:
    if notice["errors"]:
        st.warning(
            f"Embedded {notice['embedded']} of {notice['requested']} selected artifacts. "
            + " · ".join(notice["errors"])
        )
    else:
        st.success(
            f"{notice['embedded']} artifact{'s' if notice['embedded'] != 1 else ''} imported "
            "and embedded successfully."
        )

bundled_paths = bundled_artifact_paths()
all_bundled = f"All {len(bundled_paths)} artifacts"
with st.expander("Import artifacts", expanded=not items):
    upload_column, bundled_column = st.columns(2)
    with upload_column:
        upload_generation = int(st.session_state.get("artifact_upload_generation", 0))
        uploaded = st.file_uploader(
            "Upload your artifacts",
            type=["xes", "ptml", "pnml"],
            accept_multiple_files=True,
            key=f"artifact-uploads-{upload_generation}",
            help="Files are parsed and embedded as soon as they are selected. Checkpoints cannot be uploaded.",
        )
        st.caption("Embedding starts automatically after file selection.")
    with bundled_column:
        bundled_options = ["Choose an example…", all_bundled, *[path.name for path in bundled_paths]]
        bundled_choice = st.selectbox(
            "Bundled artifacts",
            bundled_options,
            help="Import any one of the nine repository artifacts, or all of them together.",
        )
        import_bundled = st.button(
            "Import bundled selection",
            type="primary",
            disabled=bundled_choice == bundled_options[0],
            use_container_width=True,
        )

with st.expander("Advanced preprocessing", expanded=False):
    parse_left, parse_right = st.columns(2)
    with parse_left:
        activity_key = st.text_input("Activity attribute", "concept:name")
        case_id_key = st.text_input("Case identifier attribute", "case:concept:name")
        lifecycle = st.checkbox("Append lifecycle transition to activity labels", False)
        remove_empty = st.checkbox("Remove empty traces", True)
        auto_final = st.checkbox("Guess missing PNML final marking", False)
    with parse_right:
        max_traces = st.number_input("Maximum encoder traces", 1, 100_000, 128)
        max_trace_length = st.number_input("Maximum trace length", 1, 10_000, 128)
        max_events = st.number_input("Maximum workspace events per log", 1, 10_000_000, 1_000_000)
        selection = st.selectbox(
            "Trace selection",
            options=list(TraceSelectionStrategy),
            format_func=lambda value: value.value.replace("_", " ").title(),
        )
        compress_variants = st.checkbox("Compress duplicate traces to variants before selection", False)
        coverage = (
            st.slider("Variant coverage target", 0.1, 1.0, 0.8, 0.05)
            if selection is TraceSelectionStrategy.VARIANT_COVERAGE
            else 0.8
        )
        seed = st.number_input("Selection seed", value=13)
        max_nodes = st.number_input("Maximum Petri-net nodes", 1, 100_000, 512)
        max_tree_tokens = st.number_input("Maximum process-tree tokens", 2, 100_000, 512)

parse_settings = ArtifactParseSettings(
    activity_key=activity_key,
    case_id_key=case_id_key,
    add_lifecycle_to_labels=lifecycle,
    remove_empty_traces=remove_empty,
    auto_guess_final_marking=auto_final,
)
preprocessing = PreprocessingSettings(
    max_events=int(max_events),
    max_traces=int(max_traces),
    max_trace_length=int(max_trace_length),
    trace_selection_strategy=selection,
    random_seed=int(seed),
    variant_coverage=float(coverage),
    compress_duplicate_variants=compress_variants,
    max_petri_nodes=int(max_nodes),
    max_tree_tokens=int(max_tree_tokens),
)

import_sources = [
    (file.name, file.getvalue(), "")
    for file in uploaded or []
]
if import_bundled:
    selected_paths = (
        bundled_paths
        if bundled_choice == all_bundled
        else tuple(path for path in bundled_paths if path.name == bundled_choice)
    )
    import_sources.extend((path.name, path.read_bytes(), path.stem) for path in selected_paths)

if import_sources:
    progress = st.progress(0.0, text="Importing and embedding artifacts…")
    errors = []
    embedded = 0
    last_artifact_id = None
    for index, (filename, data, suggested_group) in enumerate(import_sources, 1):
        try:
            content_hash = sha256(data).hexdigest()
            already_present = any(
                existing.parsed.content_hash == content_hash
                and existing.parsed.display_name == filename
                for existing in items.values()
            )
            if not already_present and len(items) >= 25:
                raise ValueError("workspace limit reached: at most 25 artifacts")
            if len(data) > 50 * 1024 * 1024:
                raise ValueError("upload exceeds the configured 50 MiB artifact limit")
            imported_item = add_uploaded_artifact(
                items,
                data,
                filename,
                parse_settings,
                parsed_cache=stage_cache(st.session_state, "parsed_artifact"),
            )
            if suggested_group and not imported_item.process_group:
                set_process_group(imported_item, suggested_group)
            completed = next(
                encode_workspace_items(
                    [imported_item],
                    checkpoint,
                    preprocessing,
                    preprocessed_cache=stage_cache(st.session_state, "preprocessed_input"),
                    embedding_cache=stage_cache(st.session_state, "embedding"),
                )
            )
            last_artifact_id = completed.artifact_id
            if completed.encoding is not None and completed.encoding.mu and not completed.errors:
                embedded += 1
            else:
                errors.append(f"{filename}: {completed.state}")
        except Exception as exc:
            errors.append(f"{filename}: {type(exc).__name__}: {exc}")
        progress.progress(
            index / len(import_sources),
            text=f"Processed {index} of {len(import_sources)} artifacts",
        )
    if uploaded:
        st.session_state["artifact_upload_generation"] = upload_generation + 1
    if last_artifact_id is not None:
        st.session_state["workspace_selected_artifact"] = last_artifact_id
    st.session_state["workspace_import_notice"] = {
        "requested": len(import_sources),
        "embedded": embedded,
        "errors": errors,
    }
    st.rerun()

st.markdown("### Workspace register")
if not items:
    st.info("No artifacts yet. Add one or more `.xes`, `.ptml`, or `.pnml` files above.")
    st.stop()

st.dataframe(workspace_rows(items.values()), use_container_width=True, hide_index=True)

manage_left, manage_right = st.columns([2, 1])
with manage_left:
    selected_id = st.selectbox(
        "Inspect artifact",
        list(items),
        format_func=lambda artifact_id: items[artifact_id].parsed.display_name,
        key="workspace_selected_artifact",
    )
with manage_right:
    group = st.text_input(
        "Process group",
        value=items[selected_id].process_group,
        placeholder="e.g. Order process",
        key=f"group-{selected_id}",
    )
    if group != items[selected_id].process_group:
        set_process_group(items[selected_id], group)

item = items[selected_id]
if item.parsed.modality is ArtifactModality.PETRI_NET:
    st.warning(PETRI_WARNING, icon="⚠️")

preview_tab, model_tab, diagnostics_tab = st.tabs(
    ["Source preview", "Model-facing input", "Warnings & state"]
)
with preview_tab:
    if item.parsed.modality is ArtifactModality.EVENT_LOG:
        render_event_log(item.parsed)
    elif item.parsed.modality is ArtifactModality.PROCESS_TREE:
        assert item.parsed.tree is not None
        tree_label_view = st.radio(
            "Tree labels",
            ["Original labels", "Canonical labels"],
            horizontal=True,
        )
        preview_tree = (
            item.parsed.tree.relabel(item.prepared.canonical_mapping)
            if tree_label_view == "Canonical labels" and item.prepared is not None
            else item.parsed.tree
        )
        render_process_tree(preview_tree, title=item.parsed.display_name)
        st.dataframe([item.parsed.source_metadata], use_container_width=True, hide_index=True)
    else:
        render_petri_net(item.parsed, title=item.parsed.display_name)
        st.dataframe([item.parsed.source_metadata], use_container_width=True, hide_index=True)
with model_tab:
    if item.prepared is None:
        item.prepared = prepare_artifact_for_model(item.parsed, checkpoint.model, preprocessing)
    st.markdown("#### Canonical activity map")
    mapping_rows = canonical_mapping_rows(item.prepared)
    mapping_metrics = st.columns(4)
    mapping_metrics[0].metric("Original labels", len(item.prepared.canonical_frequencies))
    mapping_metrics[1].metric("Canonical labels", len(item.prepared.canonical_mapping))
    mapping_metrics[2].metric("Checkpoint limit", checkpoint.metadata.maximum_activities)
    mapping_metrics[3].metric(
        "Reversible",
        "yes"
        if len(item.prepared.canonical_frequencies) == len(item.prepared.canonical_mapping)
        and item.parsed.modality is not ArtifactModality.PETRI_NET
        else "no",
    )
    if mapping_rows:
        st.dataframe(mapping_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No label mapping is sent by the structural PNML encoding path.")
    st.markdown("#### Preprocessing ledger")
    st.json(item.prepared.preprocessing_metadata)
    st.markdown("#### Model input summary")
    st.json(item.prepared.model_input_summary)
    if item.parsed.modality is ArtifactModality.EVENT_LOG:
        selected_traces = item.prepared.model_input_summary.get("selected_traces", [])
        st.dataframe(
            [
                {
                    "Trace": index,
                    "Events": len(trace),
                    "Selected sequence": " → ".join(trace),
                    "Clipped": len(trace) > int(max_trace_length),
                }
                for index, trace in enumerate(selected_traces)
            ],
            use_container_width=True,
            hide_index=True,
        )
with diagnostics_tab:
    st.write("Current state:", item.state)
    for warning in item.warnings:
        st.warning(warning)
    for error in item.errors:
        st.error(error)
    if not item.warnings and not item.errors:
        st.success("No parsing or preprocessing limitations detected.")

if item.encoding is not None and item.encoding.mu:
    st.success(
        f"Embedding ready · {item.encoding.dimension} dimensions · "
        f"{item.encoding.embedding_seconds:.3f}s"
    )
    if item.encoding.attention_weights is not None:
        st.markdown("#### Event-log trace attention")
        traces = item.encoding.model_input_summary.get("selected_traces", [])
        variant_counts = Counter(tuple(trace) for trace in traces)
        attention_max_length = int(
            item.encoding.preprocessing_metadata.get("maximum_trace_length", 0)
        )
        attention_rows = [
                {
                    "Trace": index,
                    "Sequence": " → ".join(trace),
                    "Length": len(trace),
                    "Variant frequency": variant_counts[tuple(trace)],
                    "Clipped": len(trace) > attention_max_length,
                    "Attention weight": item.encoding.attention_weights[index],
                }
                for index, trace in enumerate(traces)
            ]
        attention_rows.sort(key=lambda row: row["Attention weight"], reverse=True)
        if st.checkbox("Aggregate attention by trace variant", False):
            aggregated = {}
            for row in attention_rows:
                variant = row["Sequence"]
                target = aggregated.setdefault(
                    variant,
                    {
                        "Variant": variant,
                        "Frequency": 0,
                        "Length": row["Length"],
                        "Attention weight": 0.0,
                    },
                )
                target["Frequency"] += 1
                target["Attention weight"] += row["Attention weight"]
            attention_rows = sorted(
                aggregated.values(),
                key=lambda row: row["Attention weight"],
                reverse=True,
            )
        st.dataframe(
            attention_rows,
            use_container_width=True,
            hide_index=True,
        )
        if attention_rows and "Length" in attention_rows[0]:
            size_column = (
                "Variant frequency"
                if "Variant frequency" in attention_rows[0]
                else "Frequency"
            )
            st.scatter_chart(
                pd.DataFrame(attention_rows),
                x="Length",
                y="Attention weight",
                size=size_column,
            )
        st.caption(
            "The model assigns attention across trace representations. This is not event-level "
            "attention and is not a complete explanation of individual events."
        )

st.divider()
actions = st.columns([1, 1, 3])
with actions[0]:
    if st.button("Prepare workspace export", use_container_width=True):
        st.session_state["workspace_export_package"] = complete_workspace_zip(
            items.values(),
            checkpoint_metadata=checkpoint_metadata_dict(checkpoint),
            application_configuration={
                "preprocessing": json.loads(json.dumps(preprocessing.__dict__, default=str))
            },
            evaluation_results={
                "decode_and_behavior": st.session_state.get("evaluation_results", []),
                "neural_losses": st.session_state.get("neural_evaluation_results", []),
                "cross_modal_retrieval": st.session_state.get("retrieval_evaluation_results", []),
                "discovery_comparison": st.session_state.get("discovery_evaluation_results", []),
                "embedding_baselines": st.session_state.get("embedding_baseline_results", []),
            },
        )
    if st.session_state.get("workspace_export_package"):
        st.download_button(
            "Download workspace ZIP",
            st.session_state["workspace_export_package"],
            "proc-rosetta-workspace.zip",
            "application/zip",
            use_container_width=True,
        )
with actions[1]:
    if st.button("Remove selected", use_container_width=True):
        del items[selected_id]
        st.rerun()
