from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from proc_rosetta.visualization_data import ProjectionResult


def render_projection(projection: ProjectionResult) -> None:
    if not projection.rows:
        st.info("Encode artifacts to populate the projection.")
        return
    frame = pd.DataFrame(projection.rows)
    if not projection.meaningful:
        st.info("Fewer than three independent points: pairwise distances are more informative than PCA.")
        return
    lines = (
        alt.Chart(frame[frame["group"] != ""])
        .mark_line(opacity=0.35, strokeWidth=1.5)
        .encode(x="pc1:Q", y="pc2:Q", detail="group:N", color=alt.value("#78849a"))
    )
    points = (
        alt.Chart(frame)
        .mark_point(filled=True, size=150, stroke="#f4f6ff", strokeWidth=0.7)
        .encode(
            x=alt.X("pc1:Q", title="Principal component 1"),
            y=alt.Y("pc2:Q", title="Principal component 2"),
            color=alt.Color(
                "modality:N",
                scale=alt.Scale(
                    domain=["Event log", "Process tree", "Petri net", "Fused mean"],
                    range=["#30c6b0", "#8d7bff", "#ffad5a", "#f3f6ff"],
                ),
            ),
            shape=alt.Shape("modality:N"),
            tooltip=["artifact:N", "group:N", "modality:N", "pc1:Q", "pc2:Q"],
        )
    )
    st.altair_chart((lines + points).properties(height=520).interactive(), use_container_width=True)
    if projection.explained_variance:
        st.caption(
            f"Explained variance: PC1 {projection.explained_variance[0]:.1%} · "
            f"PC2 {projection.explained_variance[1]:.1%}"
        )
