from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from types import SimpleNamespace

from proc_rosetta_ui.common import checkpoint_sidebar, configure_page, page_header
from proc_rosetta.artifact_io import ArtifactModality, event_log_statistics
from proc_rosetta.inference import (
    ArtifactEncodingResult,
    combine_encoding_decode_evidence,
    decode_latent,
    fuse_latent_distributions,
    interpolate_latents,
)
from proc_rosetta.reference_gallery import build_reference_gallery_iter
from proc_rosetta.behavior import behavioral_distance
from proc_rosetta.visualization_data import (
    cosine_similarity_matrix,
    euclidean_distance_matrix,
    nearest_neighbors,
    project_pca,
)
from proc_rosetta_ui.app_state import workspace
from proc_rosetta_ui.export_service import embedding_csv, embedding_json, embedding_numpy
from proc_rosetta_ui.visualizations.latent_space import render_projection
from proc_rosetta_ui.visualizations.petri_net import render_petri_net
from proc_rosetta_ui.visualizations.process_tree import render_process_tree
from proc_rosetta_ui.visualizations.similarity_heatmap import render_similarity_heatmap
from proc_rosetta_ui.workspace_service import completed_encodings


configure_page("Latent-space explorer", "⌁")
checkpoint = checkpoint_sidebar()
page_header(
    "View 03 / latent-space explorer",
    "Make multimodal agreement visible.",
    "Compare every latent mean, connect artifacts that claim the same process identity, and inspect "
    "whether proximity in the learned space agrees with structural or behavioral intuition.",
)
items = workspace(st.session_state)
eligible_items = [
    item
    for item in items.values()
    if item.encoding is not None
    and item.encoding.mu
    and not item.encoding.errors
    and item.encoding.checkpoint_identifier == checkpoint.metadata.identifier
]
encodings = completed_encodings(eligible_items)
if not encodings:
    st.info("Encode artifacts with the active checkpoint to populate the latent explorer.")
    st.stop()
if len(completed_encodings(items.values())) != len(encodings):
    st.warning("Embeddings produced by other checkpoints are excluded; latent coordinate systems are not aligned.")

similarity = cosine_similarity_matrix(encodings)
distance = euclidean_distance_matrix(encodings)
groups = {item.artifact_id: item.process_group for item in eligible_items}
if st.session_state.get("reference_gallery_checkpoint") != checkpoint.metadata.identifier:
    st.session_state["reference_gallery"] = []
reference_entries = st.session_state.get("reference_gallery", [])


def fused_group_encodings(source_encodings, source_groups):
    fused = []
    for group in sorted({value for value in source_groups.values() if value}):
        candidates = [
            encoding
            for encoding in source_encodings
            if source_groups.get(encoding.artifact_id) == group
        ]
        members_by_modality = {}
        for encoding in candidates:
            members_by_modality.setdefault(encoding.modality, encoding)
        members = list(members_by_modality.values())
        if len(members) < 2:
            continue
        mu, logvar = fuse_latent_distributions(members)
        allowed, copy, activity_memory = combine_encoding_decode_evidence(members)
        fused.append(
            ArtifactEncodingResult(
                artifact_id=f"fused::{group}",
                artifact_name=f"{group} · fused mean",
                modality=ArtifactModality.PROCESS_TREE,
                checkpoint_identifier=checkpoint.metadata.identifier,
                source_metadata={"process_group": group, "fused": True},
                preprocessing_metadata={"weights": "equal"},
                canonical_mapping={},
                model_input_summary={},
                mu=mu,
                logvar=logvar,
                attention_weights=None,
                embedding_seconds=0.0,
                source_activity_labels=sorted(
                    {label for member in members for label in member.source_activity_labels}
                ),
                source_canonical_activity_labels=sorted(
                    {
                        label
                        for member in members
                        for label in member.source_canonical_activity_labels
                    }
                ),
                allowed_activity_slots=allowed or [],
                copy_activity_slots=copy or [],
                activity_memory=activity_memory,
                process_group=group,
            )
        )
    return fused
table_rows = []
for index, encoding in enumerate(encodings):
    candidates = [(similarity[index, j], j) for j in range(len(encodings)) if j != index]
    nearest = max(candidates)[1] if candidates else None
    table_rows.append(
        {
            "Artifact": encoding.artifact_name,
            "Group": groups.get(encoding.artifact_id) or "—",
            "Modality": encoding.modality.label,
            "Dimension": encoding.dimension,
            "Norm": float(np.linalg.norm(encoding.mu)),
            "Latent spread": encoding.latent_spread,
            "Nearest artifact": "—" if nearest is None else encodings[nearest].artifact_name,
            "Similarity": None if nearest is None else float(similarity[index, nearest]),
            "Encoded at": encoding.encoded_at,
            "Checkpoint": encoding.checkpoint_identifier,
        }
    )
st.markdown("### Embedding register")
st.dataframe(table_rows, use_container_width=True, hide_index=True)

projection_tab, heatmap_tab, agreement_tab, neighbor_tab, dimensions_tab, interpolation_tab, reference_tab = st.tabs(
    [
        "PCA projection",
        "Similarity heatmap",
        "Group agreement",
        "Nearest neighbors",
        "Dimensions",
        "Interpolation",
        "Reference gallery",
    ]
)
with projection_tab:
    show_fused = st.checkbox("Show equal-weight fused group means", True)
    show_reference = st.checkbox("Include cached reference gallery", bool(reference_entries))
    projection_encodings = list(encodings)
    projection_groups = dict(groups)
    if show_fused:
        fused = fused_group_encodings(encodings, groups)
        projection_encodings.extend(fused)
        projection_groups.update({encoding.artifact_id: encoding.source_metadata["process_group"] for encoding in fused})
    if show_reference:
        projection_encodings.extend(entry.encoding for entry in reference_entries)
        projection_groups.update({entry.reference_id: entry.process_group for entry in reference_entries})
    projection = project_pca(projection_encodings, projection_groups)
    for row in projection.rows:
        if str(row["artifact_id"]).startswith("fused::"):
            row["modality"] = "Fused mean"
        elif str(row["artifact_id"]).startswith("reference-"):
            row["artifact"] = f"Reference · {row['artifact']}"
    render_projection(projection)
with heatmap_tab:
    ordering = st.selectbox(
        "Ordering",
        ["Upload order", "Process group", "Modality", "Similarity to selected", "Hierarchical clustering"],
    )
    order = list(range(len(encodings)))
    if ordering == "Process group":
        order.sort(key=lambda index: (groups.get(encodings[index].artifact_id, ""), encodings[index].artifact_name))
    elif ordering == "Modality":
        order.sort(key=lambda index: (encodings[index].modality.value, encodings[index].artifact_name))
    elif ordering == "Similarity to selected":
        anchor = st.selectbox(
            "Similarity anchor",
            order,
            format_func=lambda index: encodings[index].artifact_name,
        )
        order.sort(key=lambda index: -similarity[anchor, index])
    elif ordering == "Hierarchical clustering" and len(encodings) > 2:
        try:
            from scipy.cluster.hierarchy import leaves_list, linkage
            from scipy.spatial.distance import squareform

            order = leaves_list(linkage(squareform(np.maximum(0.0, 1.0 - similarity), checks=False))).tolist()
        except Exception as exc:
            st.caption(f"Clustering unavailable; using upload order ({type(exc).__name__}).")
    ordered_names = [encodings[index].artifact_name for index in order]
    ordered_similarity = similarity[np.ix_(order, order)]
    ordered_distance = distance[np.ix_(order, order)]
    metric = st.radio(
        "Pairwise measure",
        ["Cosine similarity", "Euclidean distance", "Normalized Euclidean distance"],
        horizontal=True,
    )
    if metric == "Cosine similarity":
        render_similarity_heatmap(ordered_names, ordered_similarity)
    else:
        display_distance = (
            ordered_distance / max(np.sqrt(encodings[0].dimension), 1.0)
            if metric.startswith("Normalized")
            else ordered_distance
        )
        frame = pd.DataFrame(display_distance, index=ordered_names, columns=ordered_names)
        st.dataframe(frame, use_container_width=True)
with agreement_tab:
    rows = []
    for group in sorted({value for value in groups.values() if value}):
        indices = [i for i, encoding in enumerate(encodings) if groups.get(encoding.artifact_id) == group]
        indices = list(
            {
                encodings[index].modality: index
                for index in reversed(indices)
            }.values()
        )
        if len(indices) < 2:
            continue
        vectors = np.asarray([encodings[i].mu for i in indices])
        fused = vectors.mean(axis=0)
        pair_distances = [distance[left, right] for x, left in enumerate(indices) for right in indices[x + 1 :]]
        row = {
            "Process group": group,
            "Artifacts": len(indices),
            "Mean within-group distance": float(np.mean(pair_distances)),
            "Maximum within-group distance": float(np.max(pair_distances)),
        }
        by_modality = {encodings[i].modality: i for i in indices}
        modality_pairs = [
            (ArtifactModality.PROCESS_TREE, ArtifactModality.EVENT_LOG, "Tree ↔ Log cosine"),
            (ArtifactModality.PROCESS_TREE, ArtifactModality.PETRI_NET, "Tree ↔ Petri cosine"),
            (ArtifactModality.EVENT_LOG, ArtifactModality.PETRI_NET, "Log ↔ Petri cosine"),
        ]
        for left_modality, right_modality, label in modality_pairs:
            if left_modality in by_modality and right_modality in by_modality:
                row[label] = float(
                    similarity[by_modality[left_modality], by_modality[right_modality]]
                )
        for i in indices:
            row[f"{encodings[i].modality.label} distance from fused"] = float(
                np.linalg.norm(np.asarray(encodings[i].mu) - fused)
            )
        rows.append(row)
    if rows:
        ranking = st.selectbox(
            "Rank groups by",
            ["Best agreement", "Worst agreement", "Largest Petri discrepancy", "Largest event-log discrepancy"],
        )
        if ranking == "Best agreement":
            rows.sort(key=lambda row: row["Mean within-group distance"])
        elif ranking == "Worst agreement":
            rows.sort(key=lambda row: row["Mean within-group distance"], reverse=True)
        elif ranking == "Largest Petri discrepancy":
            rows.sort(
                key=lambda row: row.get("Petri net distance from fused", float("-inf")),
                reverse=True,
            )
        else:
            rows.sort(
                key=lambda row: row.get("Event log distance from fused", float("-inf")),
                reverse=True,
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Assign at least two encoded artifacts to the same process group.")
with neighbor_tab:
    neighbor_scope = st.radio(
        "Neighbor pool",
        ["Workspace", "Workspace + reference gallery"],
        horizontal=True,
        disabled=not reference_entries,
    )
    neighbor_pool = list(encodings)
    neighbor_groups = dict(groups)
    if neighbor_scope.endswith("reference gallery"):
        neighbor_pool.extend(entry.encoding for entry in reference_entries)
        neighbor_groups.update({entry.reference_id: entry.process_group for entry in reference_entries})
    selected_id = st.selectbox(
        "Selected artifact",
        [encoding.artifact_id for encoding in encodings],
        format_func=lambda artifact_id: next(e.artifact_name for e in encodings if e.artifact_id == artifact_id),
    )
    neighbor_filter = st.selectbox(
        "Modality filter",
        ["Any modality", "Other modalities only", "Same modality only"],
    )
    neighbor_rows = nearest_neighbors(selected_id, neighbor_pool)
    selected_modality = next(
        encoding.modality.label for encoding in neighbor_pool if encoding.artifact_id == selected_id
    )
    if neighbor_filter == "Other modalities only":
        neighbor_rows = [row for row in neighbor_rows if row["modality"] != selected_modality]
    elif neighbor_filter == "Same modality only":
        neighbor_rows = [row for row in neighbor_rows if row["modality"] == selected_modality]
    selected_item = items.get(selected_id)
    selected_traces = selected_item.parsed.traces if selected_item is not None else None
    if selected_traces is None and selected_item is not None and selected_item.process_group:
        selected_traces = next(
            (
                item.parsed.traces
                for item in items.values()
                if item.process_group == selected_item.process_group and item.parsed.traces
            ),
            None,
        )
    for row in neighbor_rows:
        row["same_process_group"] = bool(
            neighbor_groups.get(selected_id)
            and neighbor_groups.get(selected_id) == neighbor_groups.get(row["artifact_id"])
        )
        row["source"] = "reference" if str(row["artifact_id"]).startswith("reference-") else "workspace"
        neighbor_traces = None
        if row["source"] == "reference":
            neighbor_traces = next(
                entry.traces
                for entry in reference_entries
                if entry.reference_id == row["artifact_id"]
            )
        elif row["artifact_id"] in items:
            neighbor_item = items[row["artifact_id"]]
            neighbor_traces = neighbor_item.parsed.traces
            if neighbor_traces is None and neighbor_item.process_group:
                neighbor_traces = next(
                    (
                        item.parsed.traces
                        for item in items.values()
                        if item.process_group == neighbor_item.process_group and item.parsed.traces
                    ),
                    None,
                )
        if selected_traces is not None and neighbor_traces is not None:
            row["behavioral_distance"] = behavioral_distance(
                selected_traces,
                neighbor_traces,
            )["mean_l1"]
    st.dataframe(neighbor_rows, use_container_width=True, hide_index=True)
    if neighbor_rows:
        inspect_neighbor = st.selectbox(
            "Inspect neighbor source",
            [row["artifact_id"] for row in neighbor_rows[:20]],
            format_func=lambda value: next(
                row["artifact"] for row in neighbor_rows if row["artifact_id"] == value
            ),
        )
        if str(inspect_neighbor).startswith("reference-"):
            entry = next(
                entry for entry in reference_entries if entry.reference_id == inspect_neighbor
            )
            render_process_tree(entry.tree, title=f"Neighbor reference · {entry.process_group}")
        elif inspect_neighbor in items:
            neighbor_item = items[inspect_neighbor]
            if neighbor_item.parsed.tree is not None:
                render_process_tree(neighbor_item.parsed.tree, title="Neighbor source tree")
            elif neighbor_item.parsed.graph is not None:
                render_petri_net(neighbor_item.parsed, title="Neighbor source Petri net")
            else:
                st.dataframe(
                    neighbor_item.parsed.source_metadata.get("variant_frequencies", [])[:20],
                    use_container_width=True,
                    hide_index=True,
                )
            if neighbor_item.decodes and neighbor_item.decodes[-1].tree is not None:
                render_process_tree(
                    neighbor_item.decodes[-1].tree,
                    title="Neighbor decoded tree",
                )
with dimensions_tab:
    choices = st.multiselect(
        "Compare artifacts",
        [encoding.artifact_id for encoding in encodings],
        default=[encoding.artifact_id for encoding in encodings[:3]],
        format_func=lambda artifact_id: next(e.artifact_name for e in encodings if e.artifact_id == artifact_id),
    )
    chart = {
        encoding.artifact_name: encoding.mu
        for encoding in encodings
        if encoding.artifact_id in choices
    }
    if chart:
        st.caption("Latent mean by dimension")
        st.line_chart(pd.DataFrame(chart))
        spread_chart = {
            encoding.artifact_name: np.exp(0.5 * np.asarray(encoding.logvar))
            for encoding in encodings
            if encoding.artifact_id in choices
        }
        st.caption("Latent standard deviation by dimension")
        st.line_chart(pd.DataFrame(spread_chart))
        if len(choices) > 1:
            selected_vectors = np.asarray(
                [encoding.mu for encoding in encodings if encoding.artifact_id in choices]
            )
            fused_vector = selected_vectors.mean(axis=0)
            difference_chart = {
                encoding.artifact_name: np.asarray(encoding.mu) - fused_vector
                for encoding in encodings
                if encoding.artifact_id in choices
            }
            st.caption("Difference from selected fused mean")
            st.line_chart(pd.DataFrame(difference_chart))
    st.caption("Latent dimensions are intentionally unnamed; no single dimension is assigned a process meaning.")
with interpolation_tab:
    st.warning(
        "Exploratory: linear movement in latent coordinates does not guarantee a semantically monotonic process change."
    )
    if len(encodings) < 2:
        st.info("Encode at least two artifacts for interpolation.")
    else:
        endpoints = st.columns(2)
        left_id = endpoints[0].selectbox(
            "Left endpoint",
            [encoding.artifact_id for encoding in encodings],
            format_func=lambda value: next(e.artifact_name for e in encodings if e.artifact_id == value),
            key="interpolation-left",
        )
        right_options = [encoding.artifact_id for encoding in encodings if encoding.artifact_id != left_id]
        right_id = endpoints[1].selectbox(
            "Right endpoint",
            right_options,
            format_func=lambda value: next(e.artifact_name for e in encodings if e.artifact_id == value),
            key="interpolation-right",
        )
        alpha = st.slider("alpha", 0.0, 1.0, 0.5, 0.05)
        if st.button("Decode interpolation point"):
            left_encoding = next(e for e in encodings if e.artifact_id == left_id)
            right_encoding = next(e for e in encodings if e.artifact_id == right_id)
            latent = interpolate_latents(left_encoding.mu, right_encoding.mu, alpha)
            allowed, copy, activity_memory = combine_encoding_decode_evidence(
                [left_encoding, right_encoding]
            )
            mapping = (
                left_encoding.canonical_mapping
                if left_encoding.canonical_mapping == right_encoding.canonical_mapping
                else None
            )
            result = decode_latent(
                checkpoint,
                latent,
                source_artifact_ids=[left_id, right_id],
                source_modalities=[left_encoding.modality, right_encoding.modality],
                latent_source=f"linear_interpolation_alpha_{alpha:.2f}",
                canonical_mapping=mapping,
                max_length=256,
                allowed_activity_slots=allowed,
                copy_activity_slots=copy,
                activity_memory=activity_memory,
            )
            st.session_state["interpolation_result"] = result
        interpolation_result = st.session_state.get("interpolation_result")
        if interpolation_result is not None:
            st.code(" ".join(interpolation_result.token_names), language=None)
            metrics = st.columns(3)
            metrics[0].metric("Tree size", interpolation_result.tree.size() if interpolation_result.tree else "—")
            metrics[1].metric("Decode tokens", len(interpolation_result.token_ids))
            metrics[2].metric("Petri conversion", "valid" if interpolation_result.petri_convertible else "failed")
            if interpolation_result.tree:
                render_process_tree(interpolation_result.tree, title="Interpolated decoded tree")
            if interpolation_result.petri_net:
                render_petri_net(interpolation_result.petri_net, title="Interpolated derived Petri net")
with reference_tab:
    st.markdown(
        "Build a checkpoint-specific synthetic gallery so uploaded artifacts can be interpreted even when the workspace is small."
    )
    reference_controls = st.columns(3)
    reference_count = int(reference_controls[0].number_input("Reference processes", 3, 100, 12))
    reference_seed = int(reference_controls[1].number_input("Reference seed", value=13))
    reference_traces = int(reference_controls[2].number_input("Traces per process", 8, 512, 64))
    if st.button("Build reference gallery", type="primary"):
        collected = []
        progress = st.progress(0.0, text="Generating reference processes…")
        for update in build_reference_gallery_iter(
            checkpoint,
            count=reference_count,
            seed=reference_seed,
            traces_per_sample=reference_traces,
        ):
            collected.extend(update.entries)
            progress.progress(update.completed / update.total, text=f"Encoded {update.completed}/{update.total} processes")
        st.session_state["reference_gallery"] = collected
        st.session_state["reference_gallery_checkpoint"] = checkpoint.metadata.identifier
        st.rerun()
    if reference_entries:
        st.success(f"{len(reference_entries) // 3} processes · {len(reference_entries)} modality embeddings cached")
        reference_rows = [
            {
                "Reference": entry.reference_id,
                "Group": entry.process_group,
                "Modality": entry.modality.label,
                "Tree size": entry.tree.size(),
                "Traces": len(entry.traces),
                "Petri nodes": entry.petri_graph.num_nodes,
            }
            for entry in reference_entries
        ]
        st.dataframe(reference_rows, use_container_width=True, hide_index=True)
        preview_id = st.selectbox(
            "Preview reference process",
            list(dict.fromkeys(entry.process_group for entry in reference_entries)),
        )
        preview = next(entry for entry in reference_entries if entry.process_group == preview_id)
        reference_tree, reference_log, reference_petri = st.tabs(
            ["Process tree", "Event-log behavior", "Petri structure"]
        )
        with reference_tree:
            render_process_tree(preview.tree, title=f"Reference · {preview_id}")
        with reference_log:
            reference_stats = event_log_statistics(preview.traces)
            st.dataframe(
                reference_stats["variant_frequencies"][:30],
                use_container_width=True,
                hide_index=True,
            )
        with reference_petri:
            render_petri_net(
                SimpleNamespace(graph=preview.petri_graph),
                title=f"Reference Petri · {preview_id}",
            )

st.markdown("### Inspect and export one vector")
export_id = st.selectbox(
    "Embedding",
    [encoding.artifact_id for encoding in encodings],
    format_func=lambda artifact_id: next(e.artifact_name for e in encodings if e.artifact_id == artifact_id),
    key="embedding_export_id",
)
encoding = next(e for e in encodings if e.artifact_id == export_id)
with st.expander("Raw latent mean and log variance"):
    st.dataframe(
        {"Dimension": range(encoding.dimension), "mu": encoding.mu, "logvar": encoding.logvar},
        use_container_width=True,
        hide_index=True,
    )
downloads = st.columns(3)
downloads[0].download_button("JSON", embedding_json(encoding), f"{export_id}.json", "application/json", use_container_width=True)
downloads[1].download_button("CSV", embedding_csv(encoding), f"{export_id}.csv", "text/csv", use_container_width=True)
downloads[2].download_button("NumPy", embedding_numpy(encoding), f"{export_id}.npz", "application/octet-stream", use_container_width=True)
