from __future__ import annotations

import random
from dataclasses import dataclass, field

from proc_rosetta.pm4py_bridge import PetriGraph, simulate_traces, tree_to_petri_net
from proc_rosetta.tree import NodeKind, ProcessTreeNode


DEFAULT_MAX_ACTIVITIES = 30


@dataclass(frozen=True)
class SyntheticConfig:
    max_depth: int = 8
    max_activities: int = DEFAULT_MAX_ACTIVITIES
    max_arity: int = 3
    traces_per_sample: int = 128
    curriculum_phase: int = 3
    reuse_activity_probability: float = 0.15
    leaf_probability: float = 0.65
    motif_context_size: int = 4
    min_tree_depth: int = 4
    min_tree_size: int = 20
    generator: str = "behavior_families"
    variants_per_behavior: int = 2
    exact_equivalence_only_for_training: bool = False
    log_views_per_behavior: int = 1
    log_view_modes: tuple[str, ...] = ("uniform_variants", "resampled")
    motif_weights: dict[str, float] = field(
        default_factory=lambda: {
            "ordinary_tree": 0.25,
            "duplicate_vs_silent": 0.25,
            "concurrent_vs_interleaved": 0.25,
            "m_nonfreechoice": 0.25,
        }
    )
    min_families_per_motif: dict[str, int] = field(
        default_factory=lambda: {"training": 8, "validation": 4, "test": 4}
    )
    class_coverage_mode: str = "strict"
    exact_language_max_states: int = 5000
    exact_language_max_traces: int = 10000
    bounded_visible_length: int = 32
    noise_clean_fraction: float = 0.2
    noise_edit_count_weights: dict[int, float] = field(
        default_factory=lambda: {1: 0.5, 2: 0.3, 3: 0.2}
    )
    noise_operation_weights: dict[str, float] = field(
        default_factory=lambda: {
            "delete": 0.15,
            "insert": 0.15,
            "substitute": 0.15,
            "swap": 0.15,
            "repeat": 0.15,
            "prefix_truncate": 0.1,
            "suffix_truncate": 0.1,
            "outside_insert": 0.05,
        }
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "max_depth": self.max_depth,
            "max_activities": self.max_activities,
            "max_arity": self.max_arity,
            "traces_per_sample": self.traces_per_sample,
            "curriculum_phase": self.curriculum_phase,
            "reuse_activity_probability": self.reuse_activity_probability,
            "leaf_probability": self.leaf_probability,
            "motif_context_size": self.motif_context_size,
            "min_tree_depth": self.min_tree_depth,
            "min_tree_size": self.min_tree_size,
            "generator": self.generator,
            "representations": {
                "variants_per_behavior": self.variants_per_behavior,
                "exact_equivalence_only_for_training": (
                    self.exact_equivalence_only_for_training
                ),
            },
            "logs": {
                "log_views_per_behavior": self.log_views_per_behavior,
                "sampling_modes": list(self.log_view_modes),
                "traces_per_log": self.traces_per_sample,
            },
            "motifs": dict(self.motif_weights),
            "class_coverage": {
                "mode": self.class_coverage_mode,
                "min_families_per_motif": dict(self.min_families_per_motif),
            },
            "validation": {
                "exact_language_max_states": self.exact_language_max_states,
                "exact_language_max_traces": self.exact_language_max_traces,
                "bounded_visible_length": self.bounded_visible_length,
            },
            "noise": {
                "clean_fraction": self.noise_clean_fraction,
                "edit_count_weights": {
                    str(key): value for key, value in self.noise_edit_count_weights.items()
                },
                "operation_weights": dict(self.noise_operation_weights),
            },
        }

    @staticmethod
    def from_dict(data: dict[str, object]) -> "SyntheticConfig":
        representations = data.get("representations", {})
        logs = data.get("logs", {})
        validation = data.get("validation", {})
        noise = data.get("noise", {})
        coverage = data.get("class_coverage", {})
        representations = representations if isinstance(representations, dict) else {}
        logs = logs if isinstance(logs, dict) else {}
        validation = validation if isinstance(validation, dict) else {}
        noise = noise if isinstance(noise, dict) else {}
        coverage = coverage if isinstance(coverage, dict) else {}
        motif_weights = data.get("motifs", {})
        motif_weights = motif_weights if isinstance(motif_weights, dict) else {}
        return SyntheticConfig(
            max_depth=int(data.get("max_depth", 8)),
            max_activities=int(data.get("max_activities", DEFAULT_MAX_ACTIVITIES)),
            max_arity=int(data.get("max_arity", 3)),
            traces_per_sample=int(logs.get("traces_per_log", data.get("traces_per_sample", 128))),
            curriculum_phase=int(data.get("curriculum_phase", 3)),
            reuse_activity_probability=float(data.get("reuse_activity_probability", 0.15)),
            leaf_probability=float(data.get("leaf_probability", 0.65)),
            motif_context_size=int(data.get("motif_context_size", 4)),
            min_tree_depth=int(data.get("min_tree_depth", 4)),
            min_tree_size=int(data.get("min_tree_size", 20)),
            generator=str(data.get("generator", "behavior_families")),
            variants_per_behavior=int(representations.get("variants_per_behavior", 2)),
            exact_equivalence_only_for_training=bool(
                representations.get("exact_equivalence_only_for_training", False)
            ),
            log_views_per_behavior=int(logs.get("log_views_per_behavior", 1)),
            log_view_modes=tuple(
                str(value)
                for value in logs.get("sampling_modes", ("uniform_variants", "resampled"))
            ),
            motif_weights={
                str(key): float(value) for key, value in motif_weights.items()
            }
            or {
                "ordinary_tree": 0.25,
                "duplicate_vs_silent": 0.25,
                "concurrent_vs_interleaved": 0.25,
                "m_nonfreechoice": 0.25,
            },
            min_families_per_motif={
                str(key): int(value)
                for key, value in (
                    coverage.get(
                        "min_families_per_motif",
                        {"training": 8, "validation": 4, "test": 4},
                    )
                    if isinstance(coverage.get("min_families_per_motif", {}), dict)
                    else {"training": 8, "validation": 4, "test": 4}
                ).items()
            },
            class_coverage_mode=str(coverage.get("mode", "strict")),
            exact_language_max_states=int(
                validation.get("exact_language_max_states", 5000)
            ),
            exact_language_max_traces=int(
                validation.get("exact_language_max_traces", 10000)
            ),
            bounded_visible_length=int(validation.get("bounded_visible_length", 32)),
            noise_clean_fraction=float(noise.get("clean_fraction", 0.2)),
            noise_edit_count_weights={
                int(key): float(value)
                for key, value in (
                    noise.get("edit_count_weights", {"1": 0.5, "2": 0.3, "3": 0.2})
                    if isinstance(noise.get("edit_count_weights", {}), dict)
                    else {"1": 0.5, "2": 0.3, "3": 0.2}
                ).items()
            },
            noise_operation_weights={
                str(key): float(value)
                for key, value in (
                    noise.get("operation_weights", SyntheticConfig().noise_operation_weights)
                    if isinstance(noise.get("operation_weights", {}), dict)
                    else SyntheticConfig().noise_operation_weights
                ).items()
            },
        )

    @staticmethod
    def preset(name: str) -> "SyntheticConfig":
        presets: dict[str, dict[str, object]] = {
            "smoke": {
                "max_depth": 2,
                "max_activities": 8,
                "traces_per_sample": 4,
                "motifs": {
                    "ordinary_tree": 0.0,
                    "duplicate_vs_silent": 1.0,
                    "concurrent_vs_interleaved": 1.0,
                    "m_nonfreechoice": 1.0,
                },
                "class_coverage": {"mode": "best_effort"},
            },
            "balanced_train": {},
            "iid_behavior": {},
            "equivalence_train": {
                "motifs": {
                    "ordinary_tree": 0.1,
                    "duplicate_vs_silent": 0.3,
                    "concurrent_vs_interleaved": 0.3,
                    "m_nonfreechoice": 0.3,
                },
                "logs": {"log_views_per_behavior": 2},
            },
            "equivalence_test": {
                "motifs": {
                    "ordinary_tree": 0.0,
                    "duplicate_vs_silent": 1.0,
                    "concurrent_vs_interleaved": 1.0,
                    "m_nonfreechoice": 1.0,
                }
            },
            "equivalence_seen": {
                "motifs": {
                    "ordinary_tree": 0.0,
                    "duplicate_vs_silent": 1.0,
                    "concurrent_vs_interleaved": 1.0,
                    "m_nonfreechoice": 1.0,
                }
            },
            "equivalence_unseen": {
                "motifs": {
                    "ordinary_tree": 0.0,
                    "duplicate_vs_silent": 1.0,
                    "concurrent_vs_interleaved": 1.0,
                    "m_nonfreechoice": 1.0,
                },
                "representations": {"variants_per_behavior": 4},
            },
            "nonblock_ood": {
                "motifs": {"m_nonfreechoice": 1.0},
            },
            "scale_ood": {
                "max_depth": 10,
                "max_activities": DEFAULT_MAX_ACTIVITIES,
                "traces_per_sample": 128,
            },
            "sampling_ood": {
                "logs": {
                    "log_views_per_behavior": 2,
                    "sampling_modes": ["sparse", "long_tail"],
                }
            },
            "noise_ood": {
                "logs": {
                    "log_views_per_behavior": 2,
                    "sampling_modes": ["uniform_variants", "noisy"],
                },
                "noise": {
                    "clean_fraction": 0.0,
                    "edit_count_weights": {"2": 0.5, "3": 0.5},
                },
            },
            "loops_bounded": {
                "curriculum_phase": 3,
                "motifs": {"ordinary_tree": 1.0},
                "representations": {
                    "variants_per_behavior": 2,
                    "exact_equivalence_only_for_training": False,
                },
            },
        }
        if name not in presets:
            raise ValueError(f"unknown generator preset: {name}")
        return SyntheticConfig.from_dict(presets[name])


@dataclass(frozen=True)
class ProcessSample:
    tree: ProcessTreeNode
    traces: tuple[tuple[str, ...], ...]
    petri_graph: PetriGraph
    equivalence_id: str
    model_variant_id: str | None = None
    log_view_id: str | None = None
    representation_kind: str = "canonical_block_pm4py"
    equivalence_level: str = "sampled"
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "tree": self.tree.to_dict(),
            "traces": [list(trace) for trace in self.traces],
            "petri_graph": self.petri_graph.to_dict(),
            "equivalence_id": self.equivalence_id,
            "model_variant_id": self.model_variant_id,
            "log_view_id": self.log_view_id,
            "representation_kind": self.representation_kind,
            "equivalence_level": self.equivalence_level,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: dict[str, object]) -> "ProcessSample":
        from proc_rosetta.pm4py_bridge import PetriGraph

        return ProcessSample(
            tree=ProcessTreeNode.from_dict(data["tree"]),
            traces=tuple(tuple(str(event) for event in trace) for trace in data["traces"]),
            petri_graph=PetriGraph.from_dict(data["petri_graph"]),
            equivalence_id=str(data["equivalence_id"]),
            model_variant_id=(
                None if data.get("model_variant_id") is None else str(data["model_variant_id"])
            ),
            log_view_id=None if data.get("log_view_id") is None else str(data["log_view_id"]),
            representation_kind=str(data.get("representation_kind", "canonical_block_pm4py")),
            equivalence_level=str(data.get("equivalence_level", "sampled")),
            metadata=dict(data.get("metadata", {})),
        )


def generate_process_tree(config: SyntheticConfig, rng: random.Random | None = None) -> ProcessTreeNode:
    rng = rng or random.Random()
    require_loop = config.curriculum_phase >= 3
    min_depth = min(
        max(1, int(config.min_tree_depth)),
        max(1, int(config.max_depth) + 1),
    )
    min_size = max(1, int(config.min_tree_size))

    def contains_loop(node: ProcessTreeNode) -> bool:
        return node.kind is NodeKind.LOOP or any(contains_loop(child) for child in node.children)

    def build_once() -> ProcessTreeNode:
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
                return ProcessTreeNode.loop(body, redo)

            arity = rng.randint(2, max(2, config.max_arity))
            children = tuple(make_node(depth + 1) for _ in range(arity))
            return ProcessTreeNode(op, children=children)

        tree = make_node(0)
        if len(tree.unique_activity_labels()) < 2:
            tree = ProcessTreeNode.seq(tree, make_leaf())
        return tree.canonicalize_activity_labels()

    best_tree: ProcessTreeNode | None = None
    best_score: tuple[bool, int, int] | None = None
    for _ in range(100):
        tree = build_once()
        has_required_loop = (not require_loop) or contains_loop(tree)
        score = (has_required_loop, tree.max_depth(), tree.size())
        if best_score is None or score > best_score:
            best_tree = tree
            best_score = score
        if has_required_loop and tree.max_depth() >= min_depth and tree.size() >= min_size:
            return tree

    assert best_tree is not None
    return best_tree


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
    config = config or SyntheticConfig()
    if config.generator == "isolated":
        rng = random.Random(seed)
        return [
            generate_sample(config=config, rng=rng, equivalence_id=f"synthetic-{idx}")
            for idx in range(count)
        ]
    from proc_rosetta.families import generate_family_samples

    return generate_family_samples(count, config, seed or 0, split="synthetic")
