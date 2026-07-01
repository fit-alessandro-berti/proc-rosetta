from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from proc_rosetta.tree import NodeKind, ProcessTreeNode


@dataclass(frozen=True)
class PetriGraph:
    """Typed Petri-net graph extracted from pm4py objects.

    Node types:
      0 = place
      1 = visible transition
      2 = invisible transition

    Edge types:
      0 = place-to-transition
      1 = transition-to-place
    """

    node_types: tuple[int, ...]
    node_names: tuple[str, ...]
    transition_labels: tuple[str | None, ...]
    edges: tuple[tuple[int, int, int], ...]
    initial_marking: tuple[float, ...]
    final_marking: tuple[float, ...]

    @property
    def num_nodes(self) -> int:
        return len(self.node_types)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_types": list(self.node_types),
            "node_names": list(self.node_names),
            "transition_labels": list(self.transition_labels),
            "edges": [list(edge) for edge in self.edges],
            "initial_marking": list(self.initial_marking),
            "final_marking": list(self.final_marking),
        }


@dataclass(frozen=True)
class PetriNetBundle:
    net: Any
    initial_marking: Any
    final_marking: Any
    graph: PetriGraph


def to_pm4py_tree(node: ProcessTreeNode) -> Any:
    from pm4py.objects.process_tree.obj import Operator, ProcessTree

    if node.kind is NodeKind.ACTIVITY:
        return ProcessTree(label=node.label)
    if node.kind is NodeKind.TAU:
        return ProcessTree(label=None)

    operator_map = {
        NodeKind.SEQ: Operator.SEQUENCE,
        NodeKind.XOR: Operator.XOR,
        NodeKind.AND: Operator.PARALLEL,
        NodeKind.LOOP: Operator.LOOP,
    }
    root = ProcessTree(operator=operator_map[node.kind])
    children = [to_pm4py_tree(child) for child in node.children]
    for child in children:
        child.parent = root
    root.children = children
    return root


def from_pm4py_tree(tree: Any) -> ProcessTreeNode:
    from pm4py.objects.process_tree.obj import Operator

    if tree.operator is None:
        if tree.label is None:
            return ProcessTreeNode.tau()
        return ProcessTreeNode.activity(str(tree.label))

    operator_map = {
        Operator.SEQUENCE: NodeKind.SEQ,
        Operator.XOR: NodeKind.XOR,
        Operator.PARALLEL: NodeKind.AND,
        Operator.LOOP: NodeKind.LOOP,
    }
    if tree.operator not in operator_map:
        raise ValueError(f"unsupported pm4py process-tree operator: {tree.operator}")

    return ProcessTreeNode(
        operator_map[tree.operator],
        children=tuple(from_pm4py_tree(child) for child in tree.children),
    )


def tree_to_petri_net(node: ProcessTreeNode) -> PetriNetBundle:
    from pm4py.objects.conversion.process_tree import converter

    pm_tree = to_pm4py_tree(node)
    net, initial_marking, final_marking = converter.apply(pm_tree)
    graph = petri_net_to_graph(net, initial_marking, final_marking)
    return PetriNetBundle(net, initial_marking, final_marking, graph)


def simulate_traces(node: ProcessTreeNode, num_traces: int = 100, variant: str = "topbottom") -> list[list[str]]:
    from pm4py.algo.simulation.playout.process_tree import algorithm as pt_playout
    from pm4py.algo.simulation.playout.process_tree.variants import basic_playout, topbottom

    pm_tree = to_pm4py_tree(node)
    if variant == "basic":
        log = pt_playout.apply(
            pm_tree,
            variant=pt_playout.Variants.BASIC_PLAYOUT,
            parameters={basic_playout.Parameters.NO_TRACES: num_traces},
        )
    elif variant == "topbottom":
        log = pt_playout.apply(
            pm_tree,
            variant=pt_playout.Variants.TOPBOTTOM,
            parameters={topbottom.Parameters.NO_TRACES: num_traces},
        )
    else:
        raise ValueError("variant must be 'topbottom' or 'basic'")
    return event_log_to_traces(log)


def event_log_to_traces(log: Any, activity_key: str = "concept:name") -> list[list[str]]:
    return [[str(event[activity_key]) for event in trace if activity_key in event] for trace in log]


def petri_net_to_graph(net: Any, initial_marking: Any, final_marking: Any) -> PetriGraph:
    places = sorted(net.places, key=lambda place: str(place.name))
    transitions = sorted(net.transitions, key=lambda transition: str(transition.name))
    nodes = [*places, *transitions]
    index = {node: idx for idx, node in enumerate(nodes)}

    node_types: list[int] = []
    node_names: list[str] = []
    transition_labels: list[str | None] = []
    for place in places:
        node_types.append(0)
        node_names.append(str(place.name))
        transition_labels.append(None)
    for transition in transitions:
        node_types.append(1 if transition.label is not None else 2)
        node_names.append(str(transition.name))
        transition_labels.append(None if transition.label is None else str(transition.label))

    edges: list[tuple[int, int, int]] = []
    for arc in sorted(net.arcs, key=lambda item: (str(item.source.name), str(item.target.name))):
        src = index[arc.source]
        dst = index[arc.target]
        if node_types[src] == 0 and node_types[dst] in {1, 2}:
            edge_type = 0
        elif node_types[src] in {1, 2} and node_types[dst] == 0:
            edge_type = 1
        else:
            raise ValueError("Petri-net graph is not bipartite")
        edges.append((src, dst, edge_type))

    initial = [0.0 for _ in nodes]
    final = [0.0 for _ in nodes]
    for place, tokens in initial_marking.items():
        initial[index[place]] = float(tokens)
    for place, tokens in final_marking.items():
        final[index[place]] = float(tokens)

    return PetriGraph(
        node_types=tuple(node_types),
        node_names=tuple(node_names),
        transition_labels=tuple(transition_labels),
        edges=tuple(edges),
        initial_marking=tuple(initial),
        final_marking=tuple(final),
    )
