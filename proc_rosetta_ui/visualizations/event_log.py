from __future__ import annotations

import pandas as pd
import streamlit as st

from proc_rosetta.artifact_io import ParsedArtifact
from proc_rosetta.visualization_data import directly_follows_to_dot


def render_event_log(artifact: ParsedArtifact) -> None:
    metadata = artifact.source_metadata
    metrics = st.columns(5)
    metrics[0].metric("Traces", metadata["total_traces"])
    metrics[1].metric("Events", metadata["total_events"])
    metrics[2].metric("Activities", metadata["distinct_activities"])
    metrics[3].metric("Variants", metadata["trace_variants"])
    metrics[4].metric("Mean length", f"{metadata['mean_trace_length']:.1f}")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Activity frequency")
        frequencies = pd.DataFrame(
            list(metadata["activity_frequencies"].items()), columns=["Activity", "Events"]
        ).set_index("Activity")
        st.bar_chart(frequencies)
    with right:
        st.markdown("#### Trace length")
        lengths = pd.DataFrame(
            list(metadata["trace_length_frequencies"].items()), columns=["Length", "Traces"]
        ).set_index("Length")
        st.bar_chart(lengths)
    boundary_left, boundary_right = st.columns(2)
    with boundary_left:
        st.markdown("#### Start activities")
        starts = pd.DataFrame(
            list(metadata["start_activity_frequencies"].items()),
            columns=["Activity", "Traces"],
        ).set_index("Activity")
        st.bar_chart(starts)
    with boundary_right:
        st.markdown("#### End activities")
        ends = pd.DataFrame(
            list(metadata["end_activity_frequencies"].items()),
            columns=["Activity", "Traces"],
        ).set_index("Activity")
        st.bar_chart(ends)
    st.markdown("#### Directly-follows graph")
    controls = st.columns(4)
    minimum = controls[0].number_input("Minimum edge frequency", 1, value=1)
    maximum = controls[1].slider("Maximum displayed edges", 5, 100, 30)
    search = controls[2].text_input("Activity search", placeholder="Find an activity")
    frequency_mode = controls[3].selectbox("Frequency", ["Absolute", "Relative"])
    st.graphviz_chart(
        directly_follows_to_dot(
            artifact,
            minimum_frequency=minimum,
            maximum_edges=maximum,
            activity_search=search,
            relative_frequency=frequency_mode == "Relative",
        ),
        use_container_width=True,
    )
    st.markdown("#### Most frequent variants")
    st.dataframe(metadata["variant_frequencies"][:100], use_container_width=True, hide_index=True)
