from __future__ import annotations

import pandas as pd
import streamlit as st


def render_decode_quality(rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    available = [column for column in ("eos", "valid_tree", "petri_conversion") if column in frame]
    if available:
        rates = (
            frame.groupby("completion_policy")[available].mean().T
            if "completion_policy" in frame
            else frame[available].astype(float).mean().rename("Rate").to_frame()
        )
        st.bar_chart(rates)
