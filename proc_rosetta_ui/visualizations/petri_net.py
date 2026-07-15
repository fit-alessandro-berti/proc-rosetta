from __future__ import annotations

import streamlit as st

from proc_rosetta.visualization_data import petri_to_dot


def render_petri_net(value, *, title: str = "Petri net") -> None:
    graph = value.graph
    controls = st.columns(3)
    view = controls[0].selectbox(
        "Graph view",
        ["Full graph", "Local neighborhood", "Structural summary only"],
        key=f"petri-view-{title}",
    )
    hide_invisible = controls[1].checkbox(
        "Hide invisible transitions",
        key=f"petri-hide-tau-{title}",
    )
    largest_only = controls[2].checkbox(
        "Largest component only",
        key=f"petri-largest-{title}",
    )
    visible = set(range(graph.num_nodes))
    if largest_only:
        visible = _largest_component(graph)
    if view == "Local neighborhood":
        center = st.selectbox(
            "Neighborhood center",
            sorted(visible),
            format_func=lambda index: graph.node_names[index],
            key=f"petri-center-{title}",
        )
        radius = st.slider("Radius", 1, 4, 1, key=f"petri-radius-{title}")
        visible &= _neighborhood(graph, center, radius)
    metrics = st.columns(4)
    metrics[0].metric("Places", sum(graph.node_types[index] == 0 for index in visible))
    metrics[1].metric("Visible", sum(graph.node_types[index] == 1 for index in visible))
    metrics[2].metric("Invisible", sum(graph.node_types[index] == 2 for index in visible))
    metrics[3].metric(
        "Arcs",
        sum(source in visible and target in visible for source, target, _ in graph.edges),
    )
    if view != "Structural summary only":
        st.graphviz_chart(
            petri_to_dot(
                value,
                title=title,
                visible_node_indices=visible,
                hide_invisible_transitions=hide_invisible,
            ),
            use_container_width=True,
        )


def _neighborhood(graph, center: int, radius: int) -> set[int]:
    adjacency = {index: set() for index in range(graph.num_nodes)}
    for source, target, _ in graph.edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    visible = {center}
    frontier = {center}
    for _ in range(radius):
        frontier = {neighbor for node in frontier for neighbor in adjacency[node]} - visible
        visible |= frontier
    return visible


def _largest_component(graph) -> set[int]:
    remaining = set(range(graph.num_nodes))
    components = []
    while remaining:
        start = next(iter(remaining))
        component = _neighborhood(graph, start, graph.num_nodes)
        components.append(component)
        remaining -= component
    return max(components, key=len) if components else set()
