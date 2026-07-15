from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st


def render_similarity_heatmap(names: list[str], matrix) -> None:
    rows = [
        {"left": left, "right": right, "similarity": float(matrix[i, j])}
        for i, left in enumerate(names)
        for j, right in enumerate(names)
    ]
    frame = pd.DataFrame(rows)
    heatmap = (
        alt.Chart(frame)
        .mark_rect()
        .encode(
            x=alt.X("left:N", title=None, sort=names),
            y=alt.Y("right:N", title=None, sort=names),
            color=alt.Color(
                "similarity:Q",
                scale=alt.Scale(domain=[-1, 0, 1], range=["#ee6b76", "#162036", "#30c6b0"]),
            ),
            tooltip=["left:N", "right:N", alt.Tooltip("similarity:Q", format=".4f")],
        )
        .properties(height=max(320, 42 * len(names)))
    )
    st.altair_chart(heatmap, use_container_width=True)

