from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from proc_rosetta_ui.common import (
    checkpoint_sidebar,
    configure_page,
    page_header,
    trusted_checkpoint_directory,
)
from proc_rosetta_ui.cache_service import CACHE_STAGES, clear_inference_caches
from proc_rosetta.inference import checkpoint_metadata, list_trusted_checkpoints
from proc_rosetta.training import load_checkpoint


configure_page("Checkpoint dashboard", "▥")
checkpoint = checkpoint_sidebar()
page_header(
    "View 04 / checkpoint & training dashboard",
    "Inspect the run behind the latent space.",
    "Review immutable checkpoint facts, training history, component losses, the generalization gap, "
    "and the best epoch. This dashboard does not launch training inside Streamlit.",
)
metadata = checkpoint.metadata
best_epoch = next(
    (row.get("epoch") for row in reversed(metadata.history) if row.get("is_best")),
    None,
)
early_stopped = bool(
    metadata.history
    and metadata.training_configuration.get("early_stopping_patience", 0)
    and metadata.history[-1].get("epochs_without_improvement", 0)
    >= metadata.training_configuration.get("early_stopping_patience", 0)
)
metrics = st.columns(6)
metrics[0].metric("Epoch", metadata.epoch or "—")
metrics[1].metric("Latent dimension", metadata.latent_dimension)
metrics[2].metric("Hidden dimension", metadata.hidden_dimension)
metrics[3].metric("Activity limit", metadata.maximum_activities)
metrics[4].metric(
    "Best validation loss",
    "—" if metadata.best_validation_loss is None else f"{metadata.best_validation_loss:.4f}",
)
metrics[5].metric("Best epoch", best_epoch or "—", delta="early stopped" if early_stopped else None)
st.dataframe(
    [
        {
            "Tree arity": metadata.maximum_tree_arity,
            "Tree token limit": metadata.maximum_tree_token_length,
            "Petri node limit": metadata.maximum_petri_nodes,
            "Training traces/sample": metadata.maximum_traces,
            "Trace length limit": metadata.maximum_trace_length,
            "Checkpoint type": metadata.checkpoint_type,
            "Training/file timestamp": metadata.training_timestamp,
        }
    ],
    use_container_width=True,
    hide_index=True,
)

history_tab, config_tab, compare_tab = st.tabs(["Training history", "Configuration", "Compare checkpoints"])
with history_tab:
    if not metadata.history:
        metrics_csv = Path(metadata.path).with_name("training_metrics.csv")
        if metrics_csv.exists():
            csv_frame = pd.read_csv(metrics_csv)
            st.caption(f"Loaded server-side metrics CSV: `{metrics_csv}`")
            if {"training_loss", "validation_loss"}.issubset(csv_frame):
                st.line_chart(csv_frame.set_index("epoch")[["training_loss", "validation_loss"]])
            st.dataframe(csv_frame, use_container_width=True, hide_index=True)
        else:
            st.info("This checkpoint contains no embedded training history or adjacent metrics CSV.")
    else:
        rows = []
        for epoch in metadata.history:
            row = {
                "epoch": epoch.get("epoch"),
                "learning_rate": epoch.get("learning_rate"),
                "epoch_seconds": epoch.get("epoch_seconds"),
                "is_best": epoch.get("is_best"),
                "best_validation_loss": epoch.get("best_validation_loss"),
            }
            for split in ("training", "validation"):
                for name, value in epoch.get(split, {}).items():
                    row[f"{split}_{name}"] = value
            for name, value in epoch.get("generalization_gap", {}).items():
                row[f"gap_{name}"] = value
            rows.append(row)
        frame = pd.DataFrame(rows).set_index("epoch")
        total_columns = [name for name in ("training_loss", "validation_loss") if name in frame]
        if total_columns:
            st.line_chart(frame[total_columns])
        components = [
            name
            for name in frame.columns
            if name.startswith("validation_") and name not in {"validation_loss"}
        ]
        if components:
            st.markdown("#### Validation component losses")
            st.line_chart(frame[components])
        if "gap_loss" in frame:
            st.markdown("#### Generalization gap")
            st.line_chart(frame[["gap_loss"]])
        st.dataframe(frame.reset_index(), use_container_width=True, hide_index=True)
with config_tab:
    left, right = st.columns(2)
    with left:
        st.markdown("#### Training configuration")
        st.json(metadata.training_configuration)
    with right:
        st.markdown("#### Synthetic-data configuration")
        st.json(metadata.synthetic_configuration)
    st.markdown("#### Complete checkpoint metadata")
    st.json(asdict(metadata))
    st.markdown("#### Stage caches")
    cache_rows = [
        {
            "Stage": stage.replace("_", " ").title(),
            "Entries": len(st.session_state["stage_caches"].get(stage, {})),
            "Invalidated by": {
                "parsed_artifact": "artifact hash + parse settings",
                "preprocessed_input": "artifact + checkpoint + preprocessing",
                "embedding": "preprocessed input + checkpoint + latent mode",
                "decode": "latent + checkpoint + decoder configuration",
                "simulation": "tree + trace count + seed",
                "evaluation": "source + model + evaluation settings",
            }[stage],
        }
        for stage in CACHE_STAGES
    ]
    st.dataframe(cache_rows, use_container_width=True, hide_index=True)
    if st.button("Clear inference and evaluation caches"):
        clear_inference_caches(st.session_state)
        st.rerun()
with compare_tab:
    paths = list_trusted_checkpoints(trusted_checkpoint_directory())
    comparison = []
    for path in paths:
        try:
            model, raw = load_checkpoint(path, "cpu")
            meta = checkpoint_metadata(path, model, raw)
            comparison.append(
                {
                    "Checkpoint": meta.filename,
                    "Type": meta.checkpoint_type,
                    "Epoch": meta.epoch,
                    "Latent": meta.latent_dimension,
                    "Hidden": meta.hidden_dimension,
                    "Activities": meta.maximum_activities,
                    "Arity": meta.maximum_tree_arity,
                    "Best validation loss": meta.best_validation_loss,
                }
            )
        except Exception as exc:
            comparison.append({"Checkpoint": path.name, "Error": f"{type(exc).__name__}: {exc}"})
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.caption(
        "Embeddings from different checkpoints are never combined automatically, even when dimensions match."
    )
