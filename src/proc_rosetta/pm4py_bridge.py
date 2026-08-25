from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from proc_rosetta.tree import NodeKind, ProcessTreeNode


TREE_NORMALIZATION_VERSION = "pm4py-fold-v1"


@dataclass(frozen=True)
class TreeNormalizationResult:
    """Semantic and tokenizer-bounded views of one process tree."""

    semantic_tree: ProcessTreeNode
    model_tree: ProcessTreeNode
    fold_changed: bool
    normalization_version: str = TREE_NORMALIZATION_VERSION


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

    def relabel(self, mapping: dict[str, str]) -> "PetriGraph":
        """Return a copy with visible transition labels renamed through ``mapping``."""

        if not mapping:
            return self
        return PetriGraph(
            node_types=self.node_types,
            node_names=self.node_names,
            transition_labels=tuple(
                None if label is None else mapping.get(label, label)
                for label in self.transition_labels
            ),
            edges=self.edges,
            initial_marking=self.initial_marking,
            final_marking=self.final_marking,
        )

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "PetriGraph":
        return PetriGraph(
            node_types=tuple(int(value) for value in data["node_types"]),
            node_names=tuple(str(value) for value in data["node_names"]),
            transition_labels=tuple(
                None if value is None else str(value) for value in data["transition_labels"]
            ),
            edges=tuple(tuple(int(value) for value in edge) for edge in data["edges"]),
            initial_marking=tuple(float(value) for value in data["initial_marking"]),
            final_marking=tuple(float(value) for value in data["final_marking"]),
        )


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


def fold_pm4py_tree(tree: Any) -> Any:
    """Return PM4Py's deep-copied semantic fold of ``tree``."""

    from pm4py.objects.process_tree.utils.generic import fold

    return fold(tree)


def fold_process_tree(node: ProcessTreeNode) -> ProcessTreeNode:
    """Fold a validated local tree through the centrally supported PM4Py API."""

    return from_pm4py_tree(fold_pm4py_tree(to_pm4py_tree(node)))


def prepare_tree_for_model(
    node: ProcessTreeNode,
    maximum_arity: int,
) -> TreeNormalizationResult:
    """Produce the semantic folded tree and its bounded model representation."""

    semantic_tree = fold_process_tree(node)
    model_tree = semantic_tree.normalize(
        maximum_arity,
        canonicalize_activity_labels=False,
    )
    return TreeNormalizationResult(
        semantic_tree=semantic_tree,
        model_tree=model_tree,
        fold_changed=semantic_tree.canonical_key() != node.canonical_key(),
    )


def pm4py_tree_size(tree: Any) -> int:
    return 1 + sum(pm4py_tree_size(child) for child in tree.children)


def pm4py_tree_prefix_length(tree: Any) -> int:
    """Return tokenizer-style prefix length, including BOS and EOS."""

    if tree.operator is None:
        return 3
    return 4 + sum(pm4py_tree_prefix_length(child) - 2 for child in tree.children)


def tree_to_petri_net(node: ProcessTreeNode) -> PetriNetBundle:
    from pm4py.objects.conversion.process_tree import converter

    pm_tree = to_pm4py_tree(node)
    net, initial_marking, final_marking = converter.apply(pm_tree)
    graph = petri_net_to_graph(net, initial_marking, final_marking)
    return PetriNetBundle(net, initial_marking, final_marking, graph)


class _PlayoutLimitExceeded(RuntimeError):
    pass


def _minimum_visible_length(node: ProcessTreeNode) -> int:
    if node.kind is NodeKind.ACTIVITY:
        return 1
    if node.kind is NodeKind.TAU:
        return 0
    if node.kind is NodeKind.XOR:
        return min(_minimum_visible_length(child) for child in node.children)
    if node.kind is NodeKind.LOOP:
        return _minimum_visible_length(node.children[0]) + (
            _minimum_visible_length(node.children[2])
            if len(node.children) == 3
            else 0
        )
    return sum(_minimum_visible_length(child) for child in node.children)


def _bounded_topbottom_trace(
    node: ProcessTreeNode,
    rng: Any,
    *,
    max_trace_length: int,
    max_loop_iterations: int,
) -> list[str]:
    """Sample one complete trace without PM4Py's unbounded loop playout."""

    minimum_lengths: dict[int, int] = {}

    def record_minimum_lengths(current: ProcessTreeNode) -> None:
        minimum_lengths[id(current)] = _minimum_visible_length(current)
        for child in current.children:
            record_minimum_lengths(child)

    record_minimum_lengths(node)
    # This is a second bound for silent/nested loops, where visible length does
    # not necessarily increase. Ordinary finite trees use far fewer visits.
    maximum_node_visits = max(256, max_trace_length * 8, node.size() * 8)
    node_visits = 0

    def visit(current: ProcessTreeNode) -> list[str]:
        nonlocal node_visits
        node_visits += 1
        if node_visits > maximum_node_visits:
            raise _PlayoutLimitExceeded("playout exceeded its node-visit budget")

        if current.kind is NodeKind.ACTIVITY:
            assert current.label is not None
            return [current.label]
        if current.kind is NodeKind.TAU:
            return []
        if current.kind is NodeKind.XOR:
            result = visit(rng.choice(current.children))
        elif current.kind is NodeKind.SEQ:
            result = []
            for child in current.children:
                result.extend(visit(child))
                if len(result) > max_trace_length:
                    raise _PlayoutLimitExceeded("sampled trace is too long")
        elif current.kind is NodeKind.AND:
            child_traces = [visit(child) for child in current.children]
            if sum(map(len, child_traces)) > max_trace_length:
                raise _PlayoutLimitExceeded("sampled trace is too long")
            choices = [
                child_index
                for child_index, trace in enumerate(child_traces)
                for _ in trace
            ]
            rng.shuffle(choices)
            offsets = [0] * len(child_traces)
            result = []
            for child_index in choices:
                result.append(child_traces[child_index][offsets[child_index]])
                offsets[child_index] += 1
        elif current.kind is NodeKind.LOOP:
            body, redo = current.children[:2]
            exit_node = current.children[2] if len(current.children) == 3 else None
            result = visit(body)
            minimum_additional_length = minimum_lengths[id(redo)] + minimum_lengths[id(body)]
            minimum_exit_length = (
                minimum_lengths[id(exit_node)] if exit_node is not None else 0
            )
            for _ in range(max_loop_iterations - 1):
                if rng.random() > 0.5:
                    break
                # Ending a process-tree loop after its current body execution is
                # always valid. Stop here when another redo/body pair cannot fit.
                if (
                    len(result) + minimum_additional_length + minimum_exit_length
                    > max_trace_length
                ):
                    break
                result.extend(visit(redo))
                result.extend(visit(body))
                if len(result) > max_trace_length:
                    raise _PlayoutLimitExceeded("sampled trace is too long")
            if exit_node is not None:
                result.extend(visit(exit_node))
        else:  # pragma: no cover - ProcessTreeNode validation prevents this
            raise ValueError(f"unsupported process-tree kind: {current.kind}")

        if len(result) > max_trace_length:
            raise _PlayoutLimitExceeded("sampled trace is too long")
        return result

    return visit(node)


def simulate_traces(
    node: ProcessTreeNode,
    num_traces: int = 100,
    variant: str = "topbottom",
    *,
    max_trace_length: int = 128,
    rng: random.Random | None = None,
    max_loop_iterations: int = 32,
    max_attempts_per_trace: int = 100,
) -> list[list[str]]:
    """Simulate bounded traces from a process tree.

    PM4Py's top-bottom playout continues each loop with a coin flip inside an
    unbounded ``while``. Nested loops consequently have a heavy-tailed runtime.
    The local top-bottom implementation retains the same operator semantics but
    bounds visible length, loop iterations, retries, and silent-node work.
    """

    if num_traces < 0:
        raise ValueError("num_traces must be non-negative")
    if max_trace_length < 1:
        raise ValueError("max_trace_length must be positive")
    if max_loop_iterations < 1:
        raise ValueError("max_loop_iterations must be positive")
    if max_attempts_per_trace < 1:
        raise ValueError("max_attempts_per_trace must be positive")
    if variant not in {"topbottom", "basic"}:
        raise ValueError("variant must be 'topbottom' or 'basic'")

    if variant == "basic":
        # Retain the explicitly requested PM4Py token-game variant. Data
        # generation uses the bounded top-bottom path above.
        from pm4py.algo.simulation.playout.process_tree import algorithm as pt_playout
        from pm4py.algo.simulation.playout.process_tree.variants import basic_playout

        log = pt_playout.apply(
            to_pm4py_tree(node),
            variant=pt_playout.Variants.BASIC_PLAYOUT,
            parameters={basic_playout.Parameters.NO_TRACES: num_traces},
        )
        traces = event_log_to_traces(log)
        if any(len(trace) > max_trace_length for trace in traces):
            raise RuntimeError(
                f"basic playout produced a trace longer than {max_trace_length} events"
            )
        return traces

    # Keep the historical global-random behavior for callers that deliberately
    # bracket this function with random.seed()/setstate(). Data generation
    # passes a local Random instance so worker scheduling cannot affect it.
    random_source = rng if rng is not None else random
    traces: list[list[str]] = []
    for trace_index in range(num_traces):
        for _ in range(max_attempts_per_trace):
            try:
                traces.append(
                    _bounded_topbottom_trace(
                        node,
                        random_source,
                        max_trace_length=max_trace_length,
                        max_loop_iterations=max_loop_iterations,
                    )
                )
                break
            except _PlayoutLimitExceeded:
                continue
        else:
            raise RuntimeError(
                f"could not sample trace {trace_index + 1}/{num_traces} within "
                f"max_trace_length={max_trace_length} after "
                f"{max_attempts_per_trace} attempts"
            )
    return traces


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


def petri_graph_to_net(graph: PetriGraph, name: str = "petri_graph") -> PetriNetBundle:
    """Reconstruct a PM4Py accepting net from the serialized typed graph."""

    from pm4py.objects.petri_net.obj import Marking, PetriNet
    from pm4py.objects.petri_net.utils import petri_utils

    net = PetriNet(name)
    nodes: list[Any] = []
    for node_type, node_name, label in zip(
        graph.node_types, graph.node_names, graph.transition_labels
    ):
        if node_type == 0:
            node = PetriNet.Place(node_name)
            net.places.add(node)
        elif node_type in {1, 2}:
            node = PetriNet.Transition(node_name, label if node_type == 1 else None)
            net.transitions.add(node)
        else:
            raise ValueError(f"unknown Petri graph node type: {node_type}")
        nodes.append(node)
    for source, target, _ in graph.edges:
        petri_utils.add_arc_from_to(nodes[source], nodes[target], net)
    initial = Marking(
        {
            nodes[index]: int(tokens)
            for index, tokens in enumerate(graph.initial_marking)
            if graph.node_types[index] == 0 and tokens
        }
    )
    final = Marking(
        {
            nodes[index]: int(tokens)
            for index, tokens in enumerate(graph.final_marking)
            if graph.node_types[index] == 0 and tokens
        }
    )
    return PetriNetBundle(net, initial, final, graph)
