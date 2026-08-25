from __future__ import annotations

from collections.abc import Mapping
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import repeat
import math
from typing import Callable, Sequence

from proc_rosetta.pm4py_bridge import (
    PetriGraph,
    prepare_tree_for_model,
    simulate_traces,
    tree_to_petri_net,
)
from proc_rosetta.tree import NodeKind, ProcessTreeNode


DEFAULT_MAX_ACTIVITIES = 30
OPERATOR_KINDS = (
    NodeKind.SEQ,
    NodeKind.XOR,
    NodeKind.AND,
    NodeKind.LOOP,
)


def _default_operator_probabilities() -> dict[str, float]:
    return {kind.value: 0.25 for kind in OPERATOR_KINDS}


def _default_root_operator_probabilities() -> dict[str, float]:
    return {
        NodeKind.SEQ.value: 0.7,
        NodeKind.XOR.value: 0.1,
        NodeKind.AND.value: 0.1,
        NodeKind.LOOP.value: 0.1,
    }


def _normalize_operator_probabilities(
    probabilities: Mapping[object, object],
    *,
    field_name: str,
) -> dict[str, float]:
    aliases = {"sequence": NodeKind.SEQ.value}
    normalized = {kind.value: 0.0 for kind in OPERATOR_KINDS}
    seen: set[str] = set()
    for raw_kind, raw_probability in probabilities.items():
        name = raw_kind.value if isinstance(raw_kind, NodeKind) else str(raw_kind)
        name = aliases.get(name.strip().lower(), name.strip().lower())
        if name not in normalized:
            allowed = ", ".join(kind.value for kind in OPERATOR_KINDS)
            raise ValueError(f"{field_name} contains unknown operator {raw_kind!r}; use {allowed}")
        if name in seen:
            raise ValueError(f"{field_name} specifies {name!r} more than once")
        probability = float(raw_probability)
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError(f"{field_name} probabilities must be finite and non-negative")
        normalized[name] = probability
        seen.add(name)

    total = sum(normalized.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{field_name} probabilities must sum to 1.0 (received {total:g})")
    return {name: probability / total for name, probability in normalized.items()}


@dataclass(frozen=True)
class SyntheticConfig:
    max_depth: int = 8
    max_activities: int = DEFAULT_MAX_ACTIVITIES
    min_activities: int = 8
    max_arity: int = 3
    traces_per_sample: int = 128
    max_trace_length: int = 128
    curriculum_phase: int = 3
    reuse_activity_probability: float = 0.15
    leaf_probability: float = 0.55
    operator_probabilities: dict[str, float] = field(
        default_factory=_default_operator_probabilities
    )
    root_operator_probabilities: dict[str, float] = field(
        default_factory=_default_root_operator_probabilities
    )
    motif_context_size: int = 6
    motif_context_min_nodes: int = 4
    motif_context_max_nodes: int = 12
    min_tree_depth: int = 4
    min_tree_size: int = 20
    generator: str = "behavior_families"
    variants_per_behavior: int = 2
    exact_equivalence_only_for_training: bool = True
    log_views_per_behavior: int = 2
    log_view_modes: tuple[str, ...] = (
        "uniform_variants",
        "resampled",
    )
    motif_weights: dict[str, float] = field(
        default_factory=lambda: {
            "ordinary_tree": 0.75,
            "duplicate_vs_silent": 1.0 / 12.0,
            "concurrent_vs_interleaved": 1.0 / 12.0,
            "m_nonfreechoice": 1.0 / 12.0,
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

    def __post_init__(self) -> None:
        for field_name in ("operator_probabilities", "root_operator_probabilities"):
            raw = getattr(self, field_name)
            if not isinstance(raw, Mapping):
                raise ValueError(f"{field_name} must map operators to probabilities")
            probabilities = _normalize_operator_probabilities(raw, field_name=field_name)
            object.__setattr__(self, field_name, probabilities)

            enabled = {NodeKind.SEQ.value, NodeKind.XOR.value}
            if self.curriculum_phase >= 2:
                enabled.add(NodeKind.AND.value)
            if self.curriculum_phase >= 3:
                enabled.add(NodeKind.LOOP.value)
            if sum(probabilities[kind] for kind in enabled) <= 0.0:
                raise ValueError(
                    f"{field_name} must assign positive probability to an operator "
                    f"enabled in curriculum phase {self.curriculum_phase}"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "max_depth": self.max_depth,
            "max_activities": self.max_activities,
            "min_activities": self.min_activities,
            "max_arity": self.max_arity,
            "traces_per_sample": self.traces_per_sample,
            "curriculum_phase": self.curriculum_phase,
            "reuse_activity_probability": self.reuse_activity_probability,
            "leaf_probability": self.leaf_probability,
            "operator_probabilities": dict(self.operator_probabilities),
            "root_operator_probabilities": dict(self.root_operator_probabilities),
            "motif_context_size": self.motif_context_size,
            "motif_context_min_nodes": self.motif_context_min_nodes,
            "motif_context_max_nodes": self.motif_context_max_nodes,
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
                "max_trace_length": self.max_trace_length,
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
            min_activities=int(data.get("min_activities", 8)),
            max_arity=int(data.get("max_arity", 3)),
            traces_per_sample=int(logs.get("traces_per_log", data.get("traces_per_sample", 128))),
            max_trace_length=int(logs.get("max_trace_length", 128)),
            curriculum_phase=int(data.get("curriculum_phase", 3)),
            reuse_activity_probability=float(data.get("reuse_activity_probability", 0.15)),
            leaf_probability=float(data.get("leaf_probability", 0.55)),
            operator_probabilities=(
                {
                    str(key): float(value)
                    for key, value in data["operator_probabilities"].items()
                }
                if isinstance(data.get("operator_probabilities"), dict)
                else _default_operator_probabilities()
            ),
            root_operator_probabilities=(
                {
                    str(key): float(value)
                    for key, value in data["root_operator_probabilities"].items()
                }
                if isinstance(data.get("root_operator_probabilities"), dict)
                else _default_root_operator_probabilities()
            ),
            motif_context_size=int(data.get("motif_context_size", 6)),
            motif_context_min_nodes=int(data.get("motif_context_min_nodes", 4)),
            motif_context_max_nodes=int(data.get("motif_context_max_nodes", 12)),
            min_tree_depth=int(data.get("min_tree_depth", 4)),
            min_tree_size=int(data.get("min_tree_size", 20)),
            generator=str(data.get("generator", "behavior_families")),
            variants_per_behavior=int(representations.get("variants_per_behavior", 2)),
            exact_equivalence_only_for_training=bool(
                representations.get("exact_equivalence_only_for_training", True)
            ),
            log_views_per_behavior=int(logs.get("log_views_per_behavior", 2)),
            log_view_modes=tuple(
                str(value)
                for value in logs.get(
                    "sampling_modes",
                    ("uniform_variants", "resampled"),
                )
            ),
            motif_weights={
                str(key): float(value) for key, value in motif_weights.items()
            }
            or {
                "ordinary_tree": 0.75,
                "duplicate_vs_silent": 1.0 / 12.0,
                "concurrent_vs_interleaved": 1.0 / 12.0,
                "m_nonfreechoice": 1.0 / 12.0,
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
            "stage_a_tiny_overfit": {
                "curriculum_phase": 2,
                "max_depth": 3,
                "max_activities": 6,
                "min_activities": 3,
                "min_tree_depth": 2,
                "min_tree_size": 5,
                "motifs": {"ordinary_tree": 1.0},
                "representations": {
                    "variants_per_behavior": 1,
                    "exact_equivalence_only_for_training": True,
                },
                "logs": {
                    "log_views_per_behavior": 2,
                    "sampling_modes": ["uniform_variants", "resampled"],
                    "traces_per_log": 32,
                },
                "class_coverage": {"mode": "best_effort"},
            },
            "stage_b_exact_alignment": {
                "curriculum_phase": 2,
                "representations": {
                    "variants_per_behavior": 2,
                    "exact_equivalence_only_for_training": True,
                },
                "logs": {
                    "log_views_per_behavior": 4,
                    "sampling_modes": [
                        "uniform_variants",
                        "resampled",
                        "sparse",
                        "long_tail",
                    ],
                },
            },
            "stage_c_behavior_geometry": {},
            "stage_d_observation_curriculum": {
                "logs": {
                    "log_views_per_behavior": 6,
                    "sampling_modes": [
                        "uniform_variants",
                        "resampled",
                        "sparse",
                        "incomplete",
                        "long_tail",
                        "noisy",
                    ],
                },
            },
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
    exact_behavior_id: str | None = None
    behavior_signature: tuple[float, ...] = ()
    exact_trace_language_id: str | None = None
    partial_order_id: str | None = None
    structural_motif_id: str | None = None
    model_variant_id: str | None = None
    log_view_id: str | None = None
    representation_kind: str = "canonical_block_pm4py"
    equivalence_level: str = "sampled"
    decoder_target_trees: dict[str, ProcessTreeNode] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "tree": self.tree.to_dict(),
            "traces": [list(trace) for trace in self.traces],
            "petri_graph": self.petri_graph.to_dict(),
            "equivalence_id": self.equivalence_id,
            "exact_behavior_id": self.exact_behavior_id,
            "behavior_signature": list(self.behavior_signature),
            "exact_trace_language_id": self.exact_trace_language_id,
            "partial_order_id": self.partial_order_id,
            "structural_motif_id": self.structural_motif_id,
            "model_variant_id": self.model_variant_id,
            "log_view_id": self.log_view_id,
            "representation_kind": self.representation_kind,
            "equivalence_level": self.equivalence_level,
            "decoder_target_trees": {
                name: tree.to_dict() for name, tree in self.decoder_target_trees.items()
            },
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
            exact_behavior_id=(
                None
                if data.get("exact_behavior_id") is None
                else str(data["exact_behavior_id"])
            ),
            behavior_signature=tuple(
                float(value) for value in data.get("behavior_signature", ())
            ),
            exact_trace_language_id=(
                None
                if data.get("exact_trace_language_id") is None
                else str(data["exact_trace_language_id"])
            ),
            partial_order_id=(
                None if data.get("partial_order_id") is None else str(data["partial_order_id"])
            ),
            structural_motif_id=(
                None
                if data.get("structural_motif_id") is None
                else str(data["structural_motif_id"])
            ),
            model_variant_id=(
                None if data.get("model_variant_id") is None else str(data["model_variant_id"])
            ),
            log_view_id=None if data.get("log_view_id") is None else str(data["log_view_id"]),
            representation_kind=str(data.get("representation_kind", "canonical_block_pm4py")),
            equivalence_level=str(data.get("equivalence_level", "sampled")),
            decoder_target_trees={
                str(name): ProcessTreeNode.from_dict(tree)
                for name, tree in dict(data.get("decoder_target_trees", {})).items()
            },
            metadata=dict(data.get("metadata", {})),
        )


def decoder_target_trees_for_sample(
    tree: ProcessTreeNode,
    traces: Sequence[Sequence[str]],
    petri_graph: PetriGraph,
) -> dict[str, ProcessTreeNode]:
    """Build semantic folded targets legal for each source modality."""

    from proc_rosetta.pm4py_bridge import fold_process_tree
    from proc_rosetta.tree import sanitize_activity_labels

    alphabets = {
        "tree": set(tree.activity_labels()),
        "trace": {label for trace in traces for label in trace},
        "petri": {
            label for label in petri_graph.transition_labels if label is not None
        },
    }
    return {
        name: fold_process_tree(
            sanitize_activity_labels(tree, allowed_labels=alphabet).tree
        )
        for name, alphabet in alphabets.items()
    }


def generate_process_tree(config: SyntheticConfig, rng: random.Random | None = None) -> ProcessTreeNode:
    rng = rng or random.Random()
    require_loop = config.curriculum_phase >= 3
    min_depth = min(
        max(1, int(config.min_tree_depth)),
        max(1, int(config.max_depth) + 1),
    )
    min_size = max(1, int(config.min_tree_size))
    min_activities = min(
        max(2, int(getattr(config, "min_activities", 2))),
        max(2, int(config.max_activities)),
    )

    def contains_loop(node: ProcessTreeNode) -> bool:
        return node.kind is NodeKind.LOOP or any(contains_loop(child) for child in node.children)

    def build_once() -> ProcessTreeNode:
        activity_count = rng.randint(min_activities, config.max_activities)
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

            probabilities = (
                config.root_operator_probabilities
                if depth == 0
                else config.operator_probabilities
            )
            op = rng.choices(
                operators,
                weights=[probabilities[kind.value] for kind in operators],
                k=1,
            )[0]
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
    best_score: tuple[bool, int, int, int] | None = None
    for _ in range(100):
        tree = build_once()
        has_required_loop = (not require_loop) or contains_loop(tree)
        distinct = len(tree.unique_activity_labels())
        score = (has_required_loop, min(distinct, min_activities), tree.max_depth(), tree.size())
        if best_score is None or score > best_score:
            best_tree = tree
            best_score = score
        if (
            has_required_loop
            and distinct >= min_activities
            and tree.max_depth() >= min_depth
            and tree.size() >= min_size
        ):
            return tree

    assert best_tree is not None
    return _pad_tree_to_min_activities(best_tree, min_activities, rng)


def _pad_tree_to_min_activities(
    tree: ProcessTreeNode,
    min_activities: int,
    rng: random.Random,
) -> ProcessTreeNode:
    """Guarantee the alphabet floor by weaving unused activities into the tree.

    Retried sampling occasionally falls short of ``min_activities``; rather than
    rejecting the whole family, extend the best candidate with a short visible
    sequence of fresh labels so downstream consumers can rely on the floor.
    """

    used = set(tree.unique_activity_labels())
    if len(used) >= min_activities:
        return tree
    fresh: list[str] = []
    index = 0
    while len(used) + len(fresh) < min_activities:
        label = f"A{index}"
        index += 1
        if label in used:
            continue
        fresh.append(label)
    prepend = rng.random() < 0.5
    for label in fresh:
        # Fold one label at a time so operator arity stays within tokenizer limits.
        if prepend:
            tree = ProcessTreeNode.seq(ProcessTreeNode.activity(label), tree)
        else:
            tree = ProcessTreeNode.seq(tree, ProcessTreeNode.activity(label))
    return tree


def generate_sample(
    config: SyntheticConfig | None = None,
    rng: random.Random | None = None,
    equivalence_id: str | None = None,
) -> ProcessSample:
    from proc_rosetta.families import first_seen_activity_mapping

    config = config or SyntheticConfig()
    rng = rng or random.Random()
    raw_tree = generate_process_tree(config, rng)
    normalized = prepare_tree_for_model(raw_tree, config.max_arity)
    semantic_tree = normalized.semantic_tree
    traces = tuple(
        tuple(trace)
        for trace in simulate_traces(
            semantic_tree,
            num_traces=config.traces_per_sample,
        )
    )
    # Store every modality under the labeling external logs receive at
    # inference: A0, A1, ... in first-seen trace order.
    mapping = first_seen_activity_mapping(
        traces,
        tuple(dict.fromkeys(semantic_tree.activity_labels())),
        config.max_activities,
    )
    semantic_tree = semantic_tree.relabel(mapping)
    traces = tuple(tuple(mapping.get(label, label) for label in trace) for trace in traces)
    petri = tree_to_petri_net(semantic_tree)
    decoder_targets = decoder_target_trees_for_sample(semantic_tree, traces, petri.graph)
    return ProcessSample(
        tree=semantic_tree,
        traces=traces,
        petri_graph=petri.graph,
        equivalence_id=equivalence_id or f"synthetic-{rng.getrandbits(64):016x}",
        decoder_target_trees=decoder_targets,
        metadata={
            "label_scheme": "first_seen_per_log_view",
            "normalization_version": normalized.normalization_version,
            "fold_changed": normalized.fold_changed,
            "semantic_tree": semantic_tree.to_dict(),
            "model_tree": semantic_tree.normalize(
                config.max_arity,
                canonicalize_activity_labels=False,
            ).to_dict(),
        },
    )


def generate_samples(
    count: int,
    config: SyntheticConfig | None = None,
    seed: int | None = None,
    progress_update: Callable[[int], None] | None = None,
    num_workers: int | None = None,
) -> list[ProcessSample]:
    config = config or SyntheticConfig()
    if config.generator == "isolated":
        if num_workers is not None:
            if num_workers < 1:
                raise ValueError("num_workers must be positive")
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                samples = list(
                    executor.map(
                        _generate_isolated_sample,
                        repeat(config),
                        repeat(seed or 0),
                        range(count),
                    )
                )
            if progress_update is not None:
                progress_update(len(samples))
            return samples
        rng = random.Random(seed)
        samples: list[ProcessSample] = []
        for idx in range(count):
            samples.append(
                generate_sample(config=config, rng=rng, equivalence_id=f"synthetic-{idx}")
            )
            if progress_update is not None:
                progress_update(1)
        return samples
    from proc_rosetta.families import generate_family_samples

    return generate_family_samples(
        count,
        config,
        seed or 0,
        split="synthetic",
        progress_update=progress_update,
        num_workers=num_workers,
    )


def _generate_isolated_sample(
    config: SyntheticConfig,
    seed: int,
    index: int,
) -> ProcessSample:
    rng = random.Random(f"{seed}:isolated:{index}")
    return generate_sample(
        config=config,
        rng=rng,
        equivalence_id=f"synthetic-{index}",
    )
