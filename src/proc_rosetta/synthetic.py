from __future__ import annotations

import random
from dataclasses import dataclass

from proc_rosetta.pm4py_bridge import PetriGraph, simulate_traces, tree_to_petri_net
from proc_rosetta.tree import NodeKind, ProcessTreeNode


@dataclass(frozen=True)
class SyntheticConfig:
    max_depth: int = 3
    max_activities: int = 6
    max_arity: int = 3
    traces_per_sample: int = 16
    curriculum_phase: int = 2
    reuse_activity_probability: float = 0.15
    leaf_probability: float = 0.35


@dataclass(frozen=True)
class ProcessSample:
    tree: ProcessTreeNode
    traces: tuple[tuple[str, ...], ...]
    petri_graph: PetriGraph
    equivalence_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "tree": self.tree.to_dict(),
            "traces": [list(trace) for trace in self.traces],
            "petri_graph": self.petri_graph.to_dict(),
            "equivalence_id": self.equivalence_id,
        }


def generate_process_tree(config: SyntheticConfig, rng: random.Random | None = None) -> ProcessTreeNode:
    rng = rng or random.Random()
    activity_count = rng.randint(2, config.max_activities)
    activity_pool = [f"a{i}" for i in range(activity_count)]
    next_activity = 0

    def make_leaf() -> ProcessTreeNode:
        nonlocal next_activity
        if next_activity >= activity_count or (
            next_activity > 0 and rng.random() < config.reuse_activity_probability
        ):
            label = rng.choice(activity_pool[: max(next_activity, 1)])
        else:
            label = activity_pool[next_activity]
            next_activity += 1
        return ProcessTreeNode.activity(label)

    def make_node(depth: int) -> ProcessTreeNode:
        if depth >= config.max_depth or rng.random() < config.leaf_probability:
            return make_leaf()

        operators = [NodeKind.SEQ, NodeKind.XOR]
        if config.curriculum_phase >= 2:
            operators.append(NodeKind.AND)
        if config.curriculum_phase >= 3:
            operators.append(NodeKind.LOOP)

        op = rng.choice(operators)
        if op is NodeKind.LOOP:
            body = make_node(depth + 1)
            redo = make_node(depth + 1)
            exit_ = ProcessTreeNode.tau() if rng.random() < 0.5 else make_node(depth + 1)
            return ProcessTreeNode.loop(body, redo, exit_)

        arity = rng.randint(2, max(2, config.max_arity))
        children = tuple(make_node(depth + 1) for _ in range(arity))
        return ProcessTreeNode(op, children=children)

    tree = make_node(0)
    if len(tree.unique_activity_labels()) < 2:
        tree = ProcessTreeNode.seq(tree, make_leaf())
    return tree.canonicalize_activity_labels()


def generate_sample(
    config: SyntheticConfig | None = None,
    rng: random.Random | None = None,
    equivalence_id: str | None = None,
) -> ProcessSample:
    config = config or SyntheticConfig()
    rng = rng or random.Random()
    tree = generate_process_tree(config, rng)
    petri = tree_to_petri_net(tree)
    traces = tuple(tuple(trace) for trace in simulate_traces(tree, num_traces=config.traces_per_sample))
    return ProcessSample(
        tree=tree,
        traces=traces,
        petri_graph=petri.graph,
        equivalence_id=equivalence_id or f"synthetic-{rng.getrandbits(64):016x}",
    )


def generate_samples(
    count: int,
    config: SyntheticConfig | None = None,
    seed: int | None = None,
) -> list[ProcessSample]:
    rng = random.Random(seed)
    config = config or SyntheticConfig()
    return [generate_sample(config=config, rng=rng, equivalence_id=f"synthetic-{idx}") for idx in range(count)]
