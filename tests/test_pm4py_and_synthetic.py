from proc_rosetta.pm4py_bridge import simulate_traces, tree_to_petri_net
from proc_rosetta.synthetic import SyntheticConfig, generate_sample
from proc_rosetta.tree import ProcessTreeNode


def test_tree_to_petri_net_and_trace_simulation():
    tree = ProcessTreeNode.seq(ProcessTreeNode.activity("A0"), ProcessTreeNode.activity("A1"))

    petri = tree_to_petri_net(tree)
    traces = simulate_traces(tree, num_traces=3)

    assert petri.graph.num_nodes > 0
    assert petri.graph.num_edges > 0
    assert traces == [["A0", "A1"], ["A0", "A1"], ["A0", "A1"]]


def test_generate_sample_contains_all_modalities():
    sample = generate_sample(SyntheticConfig(traces_per_sample=4, max_depth=2), equivalence_id="x")

    assert sample.equivalence_id == "x"
    assert sample.tree.size() >= 2
    assert len(sample.traces) == 4
    assert sample.petri_graph.num_nodes > 0
