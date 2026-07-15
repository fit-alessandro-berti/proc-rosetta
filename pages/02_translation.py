from __future__ import annotations

import pandas as pd
import streamlit as st

from proc_rosetta_ui.common import checkpoint_sidebar, configure_page, page_header, status_pill
from proc_rosetta.artifact_io import ArtifactModality
from proc_rosetta.inference import PETRI_LABEL_WARNING, compare_source_and_decoded, validate_decoded_tree
from proc_rosetta_ui.app_state import workspace
from proc_rosetta_ui.cache_service import cache_key, cache_put, stage_cache
from proc_rosetta_ui.decoding_service import (
    decode_workspace_latent,
    decode_workspace_selection,
    sampled_workspace_latents,
)
from proc_rosetta_ui.export_service import (
    petri_net_pnml,
    process_tree_ptml,
    process_tree_report,
    simulated_log_xes,
)
from proc_rosetta_ui.visualizations.decoder_steps import render_decoder_steps
from proc_rosetta_ui.visualizations.petri_net import render_petri_net
from proc_rosetta_ui.visualizations.process_tree import render_process_tree


configure_page("Translation studio", "⇢")
checkpoint = checkpoint_sidebar()
page_header(
    "View 02 / translation studio",
    "From latent point to validated model.",
    "Decode one modality mean or an equal-weight fused representation. Watch grammar masking at "
    "work, then keep process-tree validation separate from deterministic Petri conversion.",
)
items = workspace(st.session_state)
encoded_items = {
    artifact_id: item
    for artifact_id, item in items.items()
    if item.encoding is not None
    and item.encoding.mu
    and not item.encoding.errors
    and item.encoding.checkpoint_identifier == checkpoint.metadata.identifier
}
if not encoded_items:
    st.info("Encode at least one artifact with the active checkpoint in the Artifact workspace.")
    st.caption("Open **Artifact workspace** from the page navigation to continue.")
    st.stop()

controls, explanation = st.columns([2, 1])
with controls:
    prefilled_ids = [
        artifact_id
        for artifact_id in st.session_state.get("translation_prefill_ids", [])
        if artifact_id in encoded_items
    ]
    selected_ids = st.multiselect(
        "Latent sources",
        list(encoded_items),
        default=prefilled_ids or [next(iter(encoded_items))],
        format_func=lambda artifact_id: (
            f"{encoded_items[artifact_id].parsed.display_name} · "
            f"{encoded_items[artifact_id].parsed.modality.label}"
        ),
    )
    max_length = st.slider("Maximum decode length", 16, 1024, 256, 16)
    latent_mode = st.radio(
        "Latent mode",
        ["Deterministic mean", "Stochastic latent samples · experimental"],
        horizontal=True,
    )
    custom_weights = False
    weights = None
    if len(selected_ids) > 1:
        custom_weights = st.checkbox("Use custom fusion weights · experimental", False)
        if custom_weights:
            raw_weights = []
            weight_columns = st.columns(len(selected_ids))
            for index, artifact_id in enumerate(selected_ids):
                raw_weights.append(
                    weight_columns[index].number_input(
                        encoded_items[artifact_id].parsed.display_name,
                        min_value=0.0,
                        value=1.0,
                        step=0.1,
                        key=f"fusion-weight-{artifact_id}",
                    )
                )
            weights = raw_weights
    sample_seed = 13
    sample_count = 1
    collapse_duplicates = True
    if latent_mode.startswith("Stochastic"):
        sample_controls = st.columns(3)
        sample_seed = int(sample_controls[0].number_input("Sampling seed", value=13))
        sample_count = int(sample_controls[1].number_input("Alternatives", 1, 12, 4))
        collapse_duplicates = sample_controls[2].checkbox("Collapse duplicate trees", True)
        st.caption(
            "Sampling uses VAE latent spread (`mu + exp(0.5 × logvar) × epsilon`). "
            "This is sampling variability, not calibrated predictive uncertainty."
        )
with explanation:
    st.markdown(
        '<div class="pr-card"><div class="pr-kicker">Latent source</div>'
        '<div style="color:#e9eefb;margin-top:.4rem">One artifact uses its deterministic μ. '
        'Multiple artifacts use their equal-weight arithmetic mean.</div>'
        '<div style="color:#9ba8bf;margin-top:.5rem">Custom weighted fusion and stochastic sampling '
        'remain explicitly experimental.</div></div>',
        unsafe_allow_html=True,
    )

selected = [encoded_items[artifact_id] for artifact_id in selected_ids]
contains_pnml = any(item.parsed.modality is ArtifactModality.PETRI_NET for item in selected)
if contains_pnml:
    st.warning(PETRI_LABEL_WARNING, icon="⚠️")

if st.button("Decode process tree", type="primary", disabled=not selected):
    prefix = st.empty()
    progress = st.progress(0.0, text="Starting grammar-constrained decoder…")

    def update(step):
        if step.step_index % 4 == 0 or step.eos_emitted:
            prefix.code("  ".join(step.current_prefix), language=None)
            progress.progress(
                min(step.step_index / max_length, 1.0),
                text=f"Step {step.step_index} · {step.grammar_state} · {len(step.valid_next_tokens)} valid choices",
            )

    try:
        if latent_mode.startswith("Deterministic"):
            results = [
                decode_workspace_selection(
                    selected,
                    checkpoint,
                    max_length=max_length,
                    weights=weights,
                    progress_callback=update,
                    decode_cache=stage_cache(st.session_state, "decode"),
                )
            ]
        else:
            results = []
            latents = sampled_workspace_latents(
                selected,
                count=sample_count,
                random_seed=sample_seed,
                weights=weights,
            )
            for index, latent in enumerate(latents):
                progress.progress(index / len(latents), text=f"Decoding alternative {index + 1}/{len(latents)}")
                candidate = decode_workspace_latent(
                    selected,
                    checkpoint,
                    latent=latent,
                    latent_source=f"stochastic_latent_sample_seed_{sample_seed + index}",
                    max_length=max_length,
                    progress_callback=update if index == 0 else None,
                    decode_cache=stage_cache(st.session_state, "decode"),
                )
                if not collapse_duplicates or all(
                    existing.token_names != candidate.token_names for existing in results
                ):
                    results.append(candidate)
        result = results[0]
        st.session_state["latest_decode_result"] = result
        st.session_state["latest_decode_alternatives"] = results
        st.session_state["latest_decode_source_ids"] = selected_ids
        progress.progress(1.0, text="Decode complete")
    except Exception as exc:
        st.error(f"Decode failed: {type(exc).__name__}: {exc}")

result = st.session_state.get("latest_decode_result")
alternatives = st.session_state.get("latest_decode_alternatives", [])
source_ids = st.session_state.get("latest_decode_source_ids", [])
if result is None or not set(source_ids).issubset(encoded_items):
    st.stop()

if len(alternatives) > 1:
    st.markdown("### Stochastic decode gallery")
    gallery = st.columns(min(4, len(alternatives)))
    for index, candidate in enumerate(alternatives):
        with gallery[index % len(gallery)]:
            st.caption(f"Alternative {index + 1} · {candidate.latent_source}")
            st.code(" ".join(candidate.token_names), language=None)
            st.write(
                "✓ valid" if candidate.successful else "incomplete",
                f"· {len(candidate.token_ids)} tokens",
            )
    selected_alternative = st.selectbox(
        "Inspect alternative",
        range(len(alternatives)),
        format_func=lambda index: f"Alternative {index + 1}",
    )
    result = alternatives[selected_alternative]

st.markdown("### Validation gates")
validation = validate_decoded_tree(result)
st.markdown(
    " ".join(
        [
            status_pill("EOS emitted", validation["eos_emitted"]),
            status_pill("Grammar valid", validation["grammar_valid"]),
            status_pill("Arity valid", validation["arity_valid"]),
            status_pill("Vocabulary valid", validation["vocabulary_valid"]),
            status_pill("Petri convertible", validation["petri_convertible"]),
            status_pill("Length limit", not validation["length_limit_reached"]),
        ]
    ),
    unsafe_allow_html=True,
)
for warning in result.warnings:
    st.warning(warning)
for error in result.errors:
    st.error(error)

tokens_tab, tree_tab, petri_tab, comparison_tab = st.tabs(
    ["Decoder progress", "Decoded process tree", "Derived Petri net", "Source comparison"]
)
with tokens_tab:
    render_decoder_steps(result)
with tree_tab:
    st.info("The process tree is the direct decoded model representation.")
    if result.tree is not None:
        canonical, restored = st.tabs(["Canonical labels", "Original labels"])
        with canonical:
            render_process_tree(result.tree, title="Decoded process tree · canonical")
        with restored:
            if contains_pnml:
                st.warning("Original-label restoration is unavailable for PNML-derived output.")
                render_process_tree(result.tree, title="Decoded process tree · canonical")
            else:
                render_process_tree(result.restored_tree or result.tree, title="Decoded process tree · restored")
    else:
        st.error("No complete process tree could be parsed from the generated prefix.")
with petri_tab:
    st.info(
        "This Petri net is derived deterministically from the decoded process tree. It is not "
        "independently generated by the neural decoder."
    )
    if result.petri_net is not None:
        render_petri_net(result.petri_net, title="Derived from decoded process tree")
        graph = result.petri_net.graph
        stats = st.columns(4)
        stats[0].metric("Places", sum(value == 0 for value in graph.node_types))
        stats[1].metric("Transitions", sum(value in {1, 2} for value in graph.node_types))
        stats[2].metric("Silent", sum(value == 2 for value in graph.node_types))
        stats[3].metric("Arcs", graph.num_edges)
with comparison_tab:
    comparisons = []
    for artifact_id in source_ids:
        try:
            row = compare_source_and_decoded(encoded_items[artifact_id].parsed, result)
            row["artifact"] = encoded_items[artifact_id].parsed.display_name
            comparisons.append(row)
        except Exception as exc:
            comparisons.append(
                {"artifact": encoded_items[artifact_id].parsed.display_name, "error": str(exc)}
            )
    scalar_comparisons = [
        {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        for row in comparisons
    ]
    st.dataframe(scalar_comparisons, use_container_width=True, hide_index=True)
    for artifact_id, comparison in zip(source_ids, comparisons):
        source_item = encoded_items[artifact_id]
        st.markdown(f"#### {source_item.parsed.display_name}")
        if source_item.parsed.modality is ArtifactModality.PROCESS_TREE and result.tree is not None:
            source_column, decoded_column = st.columns(2)
            with source_column:
                render_process_tree(
                    source_item.parsed.tree,
                    title=f"Source · {source_item.parsed.display_name}",
                )
            with decoded_column:
                render_process_tree(
                    result.restored_tree or result.tree,
                    title=f"Decoded · {source_item.parsed.display_name}",
                )
        elif source_item.parsed.modality is ArtifactModality.EVENT_LOG and comparison.get("available"):
            chart_left, chart_right = st.columns(2)
            activities = sorted(
                set(comparison["source_activity_frequencies"])
                | set(comparison["simulated_activity_frequencies"])
            )
            chart_left.bar_chart(
                pd.DataFrame(
                    {
                        "Source": [comparison["source_activity_frequencies"].get(key, 0) for key in activities],
                        "Simulated": [comparison["simulated_activity_frequencies"].get(key, 0) for key in activities],
                    },
                    index=activities,
                )
            )
            lengths = sorted(
                set(comparison["source_trace_length_frequencies"])
                | set(comparison["simulated_trace_length_frequencies"])
            )
            chart_right.bar_chart(
                pd.DataFrame(
                    {
                        "Source": [comparison["source_trace_length_frequencies"].get(key, 0) for key in lengths],
                        "Simulated": [comparison["simulated_trace_length_frequencies"].get(key, 0) for key in lengths],
                    },
                    index=lengths,
                )
            )
            variant_columns = st.columns(2)
            variant_columns[0].dataframe(
                comparison["source_variant_frequencies"],
                use_container_width=True,
                hide_index=True,
            )
            variant_columns[1].dataframe(
                comparison["simulated_variant_frequencies"],
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Top directly-follows relations · source vs simulated")
            dfg_columns = st.columns(2)
            dfg_columns[0].dataframe(
                comparison["source_directly_follows_frequencies"],
                use_container_width=True,
                hide_index=True,
            )
            dfg_columns[1].dataframe(
                comparison["simulated_directly_follows_frequencies"],
                use_container_width=True,
                hide_index=True,
            )
        elif source_item.parsed.modality is ArtifactModality.PETRI_NET and result.petri_net:
            source_column, decoded_column = st.columns(2)
            with source_column:
                render_petri_net(
                    source_item.parsed,
                    title=f"Source · {source_item.parsed.display_name}",
                )
            with decoded_column:
                render_petri_net(
                    result.petri_net,
                    title=f"Decoded-derived · {source_item.parsed.display_name}",
                )
    if contains_pnml:
        st.caption(
            "Structural or behavioral similarity may be evaluated, but visible activity-name "
            "preservation cannot be attributed to the current external Petri encoder path."
        )

st.markdown("### Export decoded model")
exports = st.columns(4)
if result.grammar_valid:
    restore = False if contains_pnml else st.checkbox("Restore original labels in PTML", True)
    exports[0].download_button(
        "Download PTML",
        process_tree_ptml(result, restore_labels=restore),
        "decoded-process-tree.ptml",
        "application/xml",
        use_container_width=True,
    )
    exports[1].download_button(
        "Download validation JSON",
        process_tree_report(result),
        "decoded-process-tree.json",
        "application/json",
        use_container_width=True,
    )
if result.petri_convertible:
    exports[2].download_button(
        "Download derived PNML",
        petri_net_pnml(result),
        "decoded-tree-derived.pnml",
        "application/xml",
        use_container_width=True,
    )
if result.grammar_valid:
    simulation_cache = stage_cache(st.session_state, "simulation")
    simulation_key = cache_key(
        checkpoint.metadata.identifier,
        result.token_ids,
        result.restored_label_mapping,
        100,
        13,
    )
    simulated_xes = simulation_cache.get(simulation_key)
    if simulated_xes is None:
        simulated_xes = simulated_log_xes(result, num_traces=100)
        cache_put(simulation_cache, simulation_key, simulated_xes)
    exports[3].download_button(
        "Download simulated XES",
        simulated_xes,
        "decoded-model-simulation.xes",
        "application/xml",
        use_container_width=True,
    )
