from __future__ import annotations

import streamlit as st

from proc_rosetta.tree import ProcessTreeNode
from proc_rosetta.visualization_data import tree_to_dot


def render_process_tree(tree: ProcessTreeNode, *, title: str = "Process tree") -> None:
    indexed = []

    def index_nodes(node, path=()):
        indexed.append((path, node))
        for child_index, child in enumerate(node.children):
            index_nodes(child, (*path, child_index))

    index_nodes(tree)
    focus_path = st.selectbox(
        "Focus subtree",
        [path for path, _ in indexed],
        format_func=lambda path: (
            "Complete tree"
            if not path
            else f"node {'/'.join(map(str, path))} · "
            + str(next(node for candidate, node in indexed if candidate == path))
        ),
        key=f"tree-focus-{title}",
    )
    focused_tree = next(node for path, node in indexed if path == focus_path)
    controls = st.columns(2)
    if focused_tree.max_depth() > 1:
        depth = controls[0].slider(
            "Displayed depth",
            1,
            focused_tree.max_depth(),
            focused_tree.max_depth(),
            key=f"tree-depth-{title}-{focus_path}",
        )
    else:
        depth = 1
        controls[0].caption("Single-node tree")
    search = controls[1].text_input(
        "Highlight activity",
        key=f"tree-search-{title}",
        placeholder="Activity label",
    )
    st.graphviz_chart(
        tree_to_dot(
            focused_tree,
            title=title,
            maximum_display_depth=depth,
            activity_search=search,
        ),
        use_container_width=True,
    )
    with st.expander("Prefix tokens", expanded=False):
        st.code("  ".join(["<bos>", *tree.to_prefix_tokens(), "<eos>"]), language=None)
