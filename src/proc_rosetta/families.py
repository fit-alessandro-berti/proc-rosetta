from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from hashlib import blake2b, sha256
from itertools import repeat
import math
import random
from typing import Any, Callable, Sequence

from proc_rosetta.pm4py_bridge import (
    PetriGraph,
    fold_process_tree,
    petri_net_to_graph,
    simulate_traces,
    tree_to_petri_net,
)
from proc_rosetta.tree import NodeKind, ProcessTreeNode


EQUIVALENCE_SEMANTICS = "visible_complete_trace_language"
MOTIF_KINDS = (
    "ordinary_tree",
    "duplicate_vs_silent",
    "concurrent_vs_interleaved",
    "m_nonfreechoice",
)


@dataclass(frozen=True)
class EquivalenceCertificate:
    status: str
    semantics: str
    reference_language_size: int
    checked_variants: tuple[str, ...]
    max_visible_length: int

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "semantics": self.semantics,
            "reference_language_size": self.reference_language_size,
            "checked_variants": list(self.checked_variants),
            "max_visible_length": self.max_visible_length,
        }


@dataclass(frozen=True)
class ModelVariant:
    variant_id: str
    representation_kind: str
    petri_graph: PetriGraph
    transformation_sequence: tuple[str, ...]
    structural_statistics: dict[str, object]
    equivalence_level: str
    partial_order_equivalent: bool | None = None


@dataclass(frozen=True)
class LogView:
    log_view_id: str
    sampling_mode: str
    traces: tuple[tuple[str, ...], ...]
    trace_edits: tuple[tuple["TraceEdit", ...], ...] = ()


@dataclass(frozen=True)
class TraceEdit:
    kind: str
    position: int
    old_label: str | None
    new_label: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "position": self.position,
            "old_label": self.old_label,
            "new_label": self.new_label,
        }


class TraceCorruptor:
    """Apply an exact number of visible-label edits and retain provenance."""

    def __init__(self, operation_weights: dict[str, float]) -> None:
        self.operation_weights = operation_weights

    def corrupt(
        self,
        labels: Sequence[str],
        alphabet: Sequence[str],
        rng: random.Random,
        edit_count: int,
    ) -> tuple[tuple[str, ...], tuple[TraceEdit, ...]]:
        result = list(labels)
        edits: list[TraceEdit] = []
        operations = list(self.operation_weights)
        weights = [max(0.0, self.operation_weights[name]) for name in operations]
        for _ in range(edit_count):
            kind = rng.choices(operations, weights=weights, k=1)[0]
            if kind == "delete" and result:
                position = rng.randrange(len(result))
                old = result.pop(position)
                edits.append(TraceEdit(kind, position, old, None))
            elif kind == "insert":
                position = rng.randrange(len(result) + 1)
                new = rng.choice(list(alphabet))
                result.insert(position, new)
                edits.append(TraceEdit(kind, position, None, new))
            elif kind == "substitute" and result:
                position = rng.randrange(len(result))
                old = result[position]
                choices = [label for label in alphabet if label != old] or ["__OOD__"]
                new = rng.choice(choices)
                result[position] = new
                edits.append(TraceEdit(kind, position, old, new))
            elif kind == "swap" and len(result) >= 2:
                position = rng.randrange(len(result) - 1)
                old, new = result[position], result[position + 1]
                result[position], result[position + 1] = new, old
                edits.append(TraceEdit(kind, position, old, new))
            elif kind == "repeat" and result:
                position = rng.randrange(len(result))
                new = result[position]
                result.insert(position, new)
                edits.append(TraceEdit(kind, position, None, new))
            elif kind == "prefix_truncate" and result:
                amount = rng.randint(1, len(result))
                old = result[amount - 1]
                del result[:amount]
                edits.append(TraceEdit(kind, 0, old, None))
            elif kind == "suffix_truncate" and result:
                position = rng.randrange(len(result))
                old = result[position]
                del result[position:]
                edits.append(TraceEdit(kind, position, old, None))
            else:
                position = rng.randrange(len(result) + 1)
                result.insert(position, "__OOD__")
                edits.append(TraceEdit("outside_insert", position, None, "__OOD__"))
        return tuple(result), tuple(edits)


@dataclass(frozen=True)
class BehaviorFamily:
    behavior_id: str
    exact_behavior_id: str | None
    exact_trace_language_id: str | None
    behavior_signature: tuple[float, ...]
    canonical_tree: ProcessTreeNode
    model_variants: tuple[ModelVariant, ...]
    clean_trace_pool: tuple[tuple[str, ...], ...]
    log_views: tuple[LogView, ...]
    equivalence_certificate: EquivalenceCertificate
    metadata: dict[str, object]


@dataclass(frozen=True)
class FamilyGenerationPlan:
    motif: str
    forced_root_operator: NodeKind | None = None


@dataclass(frozen=True)
class _RuntimeVariant:
    representation_kind: str
    net: Any
    initial_marking: Any
    final_marking: Any
    transformations: tuple[str, ...]
    nonblock_reason: str | None = None
    partial_order_equivalent: bool | None = None


def stable_seed(global_seed: int, *parts: object) -> int:
    payload = "|".join([str(global_seed), *(str(part) for part in parts)])
    digest = blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def active_motifs(weights: dict[str, float]) -> tuple[str, ...]:
    motifs = tuple(name for name in MOTIF_KINDS if float(weights.get(name, 0.0)) > 0)
    unknown = sorted(
        name
        for name, weight in weights.items()
        if float(weight) > 0 and name not in MOTIF_KINDS
    )
    if unknown:
        raise ValueError(f"unknown positive-weight motifs: {', '.join(unknown)}")
    if not motifs:
        raise ValueError("at least one known motif must have positive weight")
    return motifs


def allocate_motif_quotas(
    family_count: int,
    weights: dict[str, float],
    minimum_per_motif: int,
    *,
    strict: bool,
) -> dict[str, int]:
    """Allocate deterministic weighted quotas while reserving class minima."""

    if family_count < 0:
        raise ValueError("family_count must be non-negative")
    if minimum_per_motif < 0:
        raise ValueError("minimum_per_motif must be non-negative")
    motifs = active_motifs(weights)
    required = minimum_per_motif * len(motifs)
    if strict and family_count < required:
        raise ValueError(
            f"{family_count} families cannot provide at least {minimum_per_motif} "
            f"families for each of {len(motifs)} active motifs; need at least {required}"
        )

    effective_minimum = min(minimum_per_motif, family_count // len(motifs))
    quotas = {motif: effective_minimum for motif in motifs}
    remaining = family_count - effective_minimum * len(motifs)
    total_weight = sum(max(0.0, float(weights[motif])) for motif in motifs)
    raw = {
        motif: remaining * max(0.0, float(weights[motif])) / total_weight
        for motif in motifs
    }
    for motif in motifs:
        quotas[motif] += int(raw[motif])
    leftovers = family_count - sum(quotas.values())
    order = sorted(motifs, key=lambda motif: (-(raw[motif] % 1), motif))
    for motif in order[:leftovers]:
        quotas[motif] += 1
    return quotas


def motif_quota_plan(
    family_count: int,
    config: Any,
    seed: int,
    split: str,
) -> tuple[list[str], dict[str, int], int, bool]:
    minimums = getattr(config, "min_families_per_motif", {})
    minimum = int(minimums.get(split, 0)) if isinstance(minimums, dict) else 0
    mode = str(getattr(config, "class_coverage_mode", "strict"))
    if mode not in {"strict", "best_effort"}:
        raise ValueError("class_coverage_mode must be 'strict' or 'best_effort'")
    quotas = allocate_motif_quotas(
        family_count,
        _motif_weights(config),
        minimum,
        strict=mode == "strict",
    )
    plan = [motif for motif in MOTIF_KINDS for _ in range(quotas.get(motif, 0))]
    random.Random(stable_seed(seed, split, "motif-quota-plan")).shuffle(plan)
    meets_minimum = all(value >= minimum for value in quotas.values())
    return plan, quotas, minimum, meets_minimum


def allocate_operator_quotas(
    count: int,
    probabilities: dict[str, float],
) -> dict[str, int]:
    """Largest-remainder allocation used for auditable root frequencies."""

    raw = {kind.value: count * float(probabilities[kind.value]) for kind in _operator_kinds()}
    quotas = {name: int(value) for name, value in raw.items()}
    remaining = count - sum(quotas.values())
    order = sorted(raw, key=lambda name: (-(raw[name] - quotas[name]), name))
    for name in order[:remaining]:
        quotas[name] += 1
    return quotas


def family_generation_plan(
    family_count: int,
    config: Any,
    seed: int,
    split: str,
) -> tuple[list[FamilyGenerationPlan], dict[str, int]]:
    motif_plan, _, _, _ = motif_quota_plan(family_count, config, seed, split)
    ordinary_count = sum(motif == "ordinary_tree" for motif in motif_plan)
    if int(getattr(config, "curriculum_phase", 3)) < 3:
        return [FamilyGenerationPlan(motif=motif) for motif in motif_plan], {}
    root_quotas = allocate_operator_quotas(
        ordinary_count,
        dict(getattr(config, "root_operator_probabilities")),
    )
    roots = [
        NodeKind(name)
        for name in (kind.value for kind in _operator_kinds())
        for _ in range(root_quotas[name])
    ]
    random.Random(stable_seed(seed, split, "root-quota-plan")).shuffle(roots)
    root_iterator = iter(roots)
    return [
        FamilyGenerationPlan(
            motif=motif,
            forced_root_operator=(next(root_iterator) if motif == "ordinary_tree" else None),
        )
        for motif in motif_plan
    ], root_quotas


def _operator_kinds() -> tuple[NodeKind, ...]:
    return (NodeKind.SEQ, NodeKind.XOR, NodeKind.AND, NodeKind.LOOP)


def generate_behavior_family(
    config: Any,
    family_index: int,
    seed: int,
    split: str = "training",
    motif: str | None = None,
    forced_root_operator: NodeKind | None = None,
) -> BehaviorFamily:
    family_seed = stable_seed(seed, split, family_index, "family")
    seed_bundle = {
        "model": stable_seed(family_seed, "model"),
        "representation": stable_seed(family_seed, "representation"),
        "playout": stable_seed(family_seed, "playout"),
        "log_view": stable_seed(family_seed, "log-view"),
        "noise": stable_seed(family_seed, "noise"),
    }
    rng = random.Random(seed_bundle["model"])
    configured_operator_probabilities = dict(
        getattr(config, "operator_probabilities")
    )
    configured_root_operator_probabilities = dict(
        getattr(config, "root_operator_probabilities")
    )
    if motif is None:
        motif = _weighted_choice(rng, _motif_weights(config))
    elif motif not in MOTIF_KINDS:
        raise ValueError(f"unknown motif: {motif}")
    behavior_id = f"{split}-{_stable_id(seed, split, family_index)}"

    if motif == "duplicate_vs_silent":
        tree, runtime_variants = _duplicate_vs_silent_family(behavior_id)
    elif motif == "concurrent_vs_interleaved":
        tree, runtime_variants = _concurrent_vs_interleaved_family(behavior_id)
    elif motif == "m_nonfreechoice":
        tree, runtime_variants = _m_nonfreechoice_family(behavior_id)
    else:
        tree, runtime_variants, generation_provenance = _ordinary_tree_family(
            config,
            rng,
            behavior_id,
            forced_root_operator=forced_root_operator,
        )
    if motif != "ordinary_tree":
        generation_provenance = {}
    ordinary_loop_tree = motif == "ordinary_tree" and _tree_contains_kind(
        tree, NodeKind.LOOP
    )

    context_metadata: dict[str, object] = {}
    if motif != "ordinary_tree":
        tree, runtime_variants, context_metadata = _apply_random_structural_context(
            tree,
            runtime_variants,
            rng,
            behavior_id,
            max_activities=int(getattr(config, "max_activities", 30)),
            min_nodes=int(getattr(config, "motif_context_min_nodes", 4)),
            max_nodes=int(getattr(config, "motif_context_max_nodes", 12)),
        )
    tree = fold_process_tree(tree)
    from proc_rosetta.synthetic import tree_complexity

    complexity_level = str(getattr(config, "complexity_level", "complex"))
    complexity_role = "ordinary_tree" if motif == "ordinary_tree" else "anchor_motif"
    folded_complexity = tree_complexity(tree)
    if motif != "ordinary_tree":
        generation_provenance = {
            "raw_complexity": folded_complexity,
            "folded_complexity": folded_complexity,
            "operator_draws": {"root": None, "non_root": {}},
        }
    canonical_tree_hash = sha256(
        repr(tree.canonicalize_activity_labels().canonical_key()).encode("utf-8")
    ).hexdigest()
    family_label_mapping = tree.activity_label_mapping()
    tree = tree.relabel(family_label_mapping)
    runtime_variants = tuple(
        _relabel_runtime_variant(variant, family_label_mapping)
        for variant in runtime_variants
    )

    runtime_variants = list(runtime_variants)
    requested_variants = max(1, int(getattr(config, "variants_per_behavior", 2)))
    max_states = int(getattr(config, "exact_language_max_states", 5000))
    max_traces = int(getattr(config, "exact_language_max_traces", 10000))
    max_visible_length = int(getattr(config, "bounded_visible_length", 32))
    if requested_variants >= 3 and not ordinary_loop_tree:
        finite_language, finite_complete = enumerate_visible_language(
            runtime_variants[0].net,
            runtime_variants[0].initial_marking,
            runtime_variants[0].final_marking,
            max_states=max_states,
            max_traces=max_traces,
            max_visible_length=max_visible_length,
        )
        if finite_complete:
            trie = _make_prefix_trie_net(
                finite_language, f"{behavior_id}-prefix-trie"
            )
            runtime_variants.append(
                _RuntimeVariant(
                    "prefix_trie",
                    trie[0],
                    trie[1],
                    trie[2],
                    ("exact_language_enumeration", "prefix_trie_compilation"),
                )
            )
    if requested_variants >= 4:
        base = runtime_variants[0]
        refined = _tau_prefix_refinement(
            base.net,
            base.initial_marking,
            base.final_marking,
            f"{behavior_id}-tau-refined",
        )
        runtime_variants.append(
            _RuntimeVariant(
                "tau_refinement",
                refined[0],
                refined[1],
                refined[2],
                (*base.transformations, "invisible_prefix_refinement"),
                nonblock_reason=base.nonblock_reason,
                partial_order_equivalent=base.partial_order_equivalent,
            )
        )
    base_variants = tuple(runtime_variants)
    while len(runtime_variants) < requested_variants:
        base = base_variants[(len(runtime_variants) - len(base_variants)) % len(base_variants)]
        clone = _clone_isomorphic(
            base.net,
            base.initial_marking,
            base.final_marking,
            f"{behavior_id}-{base.representation_kind}-iso-{len(runtime_variants)}",
        )
        runtime_variants.append(
            _RuntimeVariant(
                representation_kind=(
                    f"{base.representation_kind}_isomorphic_{len(runtime_variants)}"
                ),
                net=clone[0],
                initial_marking=clone[1],
                final_marking=clone[2],
                transformations=(*base.transformations, "isomorphic_renaming"),
                nonblock_reason=base.nonblock_reason,
                partial_order_equivalent=base.partial_order_equivalent,
            )
        )

    if ordinary_loop_tree:
        equivalence_level = "isomorphic"
        variants = tuple(
            ModelVariant(
                variant_id=f"{behavior_id}:{variant.representation_kind}",
                representation_kind=variant.representation_kind,
                petri_graph=petri_net_to_graph(
                    variant.net, variant.initial_marking, variant.final_marking
                ),
                transformation_sequence=variant.transformations,
                structural_statistics=structural_statistics(
                    variant.net,
                    variant.initial_marking,
                    variant.final_marking,
                    nonblock_reason=variant.nonblock_reason,
                ),
                equivalence_level=equivalence_level,
                partial_order_equivalent=variant.partial_order_equivalent,
            )
            for variant in runtime_variants[:requested_variants]
        )
        if any(
            not variant.structural_statistics["final_marking_reachable"]
            or int(variant.structural_statistics["dead_transition_count"]) > 0
            for variant in variants
        ):
            raise ValueError(f"family {behavior_id} contains an unsound structural variant")

        playout_count = max(256, int(getattr(config, "traces_per_sample", 128)) * 4)
        max_trace_length = max(1, int(getattr(config, "max_trace_length", 128)))
        trace_pool = tuple(
            sorted(
                {
                    tuple(trace)
                    for trace in simulate_traces(
                        tree,
                        num_traces=playout_count,
                        max_trace_length=max_trace_length,
                        rng=random.Random(seed_bundle["playout"]),
                    )
                }
            )
        )
        if not trace_pool:
            raise ValueError(f"family {behavior_id} has no sampled visible traces")
        certificate = EquivalenceCertificate(
            status=equivalence_level,
            semantics=EQUIVALENCE_SEMANTICS,
            reference_language_size=len(trace_pool),
            checked_variants=tuple(
                variant.representation_kind for variant in runtime_variants[:requested_variants]
            ),
            max_visible_length=max(len(trace) for trace in trace_pool),
        )
        log_views = _make_log_views(config, trace_pool, behavior_id, seed_bundle["log_view"])
        signature = bounded_behavior_signature(trace_pool, tree)
        return BehaviorFamily(
            behavior_id=behavior_id,
            exact_behavior_id=None,
            exact_trace_language_id=None,
            behavior_signature=signature,
            canonical_tree=tree,
            model_variants=variants,
            clean_trace_pool=trace_pool,
            log_views=log_views,
            equivalence_certificate=certificate,
            metadata={
                "motif": motif,
                "family_index": family_index,
                "family_seed": family_seed,
                "seed_bundle": seed_bundle,
                "equivalence_semantics": EQUIVALENCE_SEMANTICS,
                "trace_language_equivalent": True,
                "partial_order_equivalent": None,
                "structural_context": context_metadata,
                "trace_pool_source": "process_tree_playout",
                "playout_trace_count": playout_count,
                "complexity_level": complexity_level,
                "complexity_role": complexity_role,
                "canonical_tree_hash": canonical_tree_hash,
                "expected_operator_probabilities": configured_operator_probabilities,
                "expected_root_operator_probabilities": (
                    configured_root_operator_probabilities
                ),
                **generation_provenance,
            },
        )

    languages: list[set[tuple[str, ...]]] = []
    exact = True
    for variant in runtime_variants:
        language, completed = enumerate_visible_language(
            variant.net,
            variant.initial_marking,
            variant.final_marking,
            max_states=max_states,
            max_traces=max_traces,
            max_visible_length=max_visible_length,
        )
        languages.append(language)
        exact = exact and completed

    reference_language = languages[0]
    for variant, language in zip(runtime_variants[1:], languages[1:]):
        if language != reference_language:
            raise ValueError(
                f"equivalence validation failed for {behavior_id}: "
                f"{runtime_variants[0].representation_kind} != {variant.representation_kind}"
            )
    if not reference_language:
        raise ValueError(f"family {behavior_id} has no complete visible traces")

    equivalence_level = "exact" if exact else "bounded"
    certificate = EquivalenceCertificate(
        status=equivalence_level,
        semantics=EQUIVALENCE_SEMANTICS,
        reference_language_size=len(reference_language),
        checked_variants=tuple(variant.representation_kind for variant in runtime_variants),
        max_visible_length=max_visible_length,
    )
    variants = tuple(
        ModelVariant(
            variant_id=f"{behavior_id}:{variant.representation_kind}",
            representation_kind=variant.representation_kind,
            petri_graph=petri_net_to_graph(
                variant.net, variant.initial_marking, variant.final_marking
            ),
            transformation_sequence=variant.transformations,
            structural_statistics=structural_statistics(
                variant.net,
                variant.initial_marking,
                variant.final_marking,
                nonblock_reason=variant.nonblock_reason,
            ),
            equivalence_level=equivalence_level,
            partial_order_equivalent=variant.partial_order_equivalent,
        )
        for variant in runtime_variants[:requested_variants]
    )
    if any(
        not variant.structural_statistics["final_marking_reachable"]
        or int(variant.structural_statistics["dead_transition_count"]) > 0
        for variant in variants
    ):
        raise ValueError(f"family {behavior_id} contains an unsound structural variant")

    trace_pool = tuple(sorted(reference_language))
    exact_behavior_id = (
        exact_language_behavior_id(reference_language, tree) if exact else None
    )
    signature = bounded_behavior_signature(reference_language, tree)
    variants = tuple(
        replace(
            variant,
            variant_id=f"{behavior_id}:{variant.representation_kind}",
        )
        for variant in variants
    )
    # The family identifier remains split/index scoped. The language-derived
    # exact ID is stored separately so independently sampled equal behaviors
    # are recognized as strong positives without collapsing distinct families.
    log_views = _make_log_views(config, trace_pool, behavior_id, seed_bundle["log_view"])
    return BehaviorFamily(
        behavior_id=behavior_id,
        exact_behavior_id=exact_behavior_id,
        exact_trace_language_id=exact_behavior_id,
        behavior_signature=signature,
        canonical_tree=tree,
        model_variants=variants,
        clean_trace_pool=trace_pool,
        log_views=log_views,
        equivalence_certificate=certificate,
        metadata={
            "motif": motif,
            "family_index": family_index,
            "family_seed": family_seed,
            "seed_bundle": seed_bundle,
            "equivalence_semantics": EQUIVALENCE_SEMANTICS,
            "trace_language_equivalent": True,
            "partial_order_equivalent": (
                False if motif == "concurrent_vs_interleaved" else None
            ),
            "structural_context": context_metadata,
            "complexity_level": complexity_level,
            "complexity_role": complexity_role,
            "canonical_tree_hash": canonical_tree_hash,
            "expected_operator_probabilities": configured_operator_probabilities,
            "expected_root_operator_probabilities": (
                configured_root_operator_probabilities
            ),
            **generation_provenance,
        },
    )


def first_seen_activity_mapping(
    traces: Sequence[Sequence[str]],
    alphabet: Sequence[str],
    max_activities: int,
) -> dict[str, str]:
    """Assign ``A0, A1, ...`` in first-occurrence order over ``traces``.

    This mirrors exactly how external event logs are canonicalized at inference
    time, so training examples carry the same labeling scheme the encoder will
    see for real logs. Labels from ``alphabet`` (the family's model labels) that
    never occur in the traces are appended afterwards in stable order. Labels
    outside ``alphabet`` (noise insertions such as ``__OOD__``) only consume an
    index when capacity remains after every alphabet label got one, so a noisy
    view can never push a model label out of the tokenizer range.
    """

    alphabet_set = set(alphabet)
    mapping: dict[str, str] = {}
    unassigned_alphabet = len(alphabet_set)
    for trace in traces:
        for label in trace:
            if label in mapping:
                continue
            if label in alphabet_set:
                mapping[label] = f"A{len(mapping)}"
                unassigned_alphabet -= 1
            elif len(mapping) < max_activities - unassigned_alphabet:
                mapping[label] = f"A{len(mapping)}"
    for label in alphabet:
        if label not in mapping and len(mapping) < max_activities:
            mapping[label] = f"A{len(mapping)}"
    return mapping


def _relabel_trace_edits(
    trace_edits: tuple[tuple[TraceEdit, ...], ...],
    mapping: dict[str, str],
) -> tuple[tuple[TraceEdit, ...], ...]:
    return tuple(
        tuple(
            TraceEdit(
                kind=edit.kind,
                position=edit.position,
                old_label=None if edit.old_label is None else mapping.get(edit.old_label, edit.old_label),
                new_label=None if edit.new_label is None else mapping.get(edit.new_label, edit.new_label),
            )
            for edit in edits
        )
        for edits in trace_edits
    )


def flatten_behavior_family(
    family: BehaviorFamily,
    max_activities: int | None = None,
) -> list[Any]:
    # Import lazily to avoid the synthetic -> families -> synthetic cycle.
    from proc_rosetta.synthetic import (
        DEFAULT_MAX_ACTIVITIES,
        ProcessSample,
        decoder_target_trees_for_sample,
    )

    if max_activities is None:
        max_activities = DEFAULT_MAX_ACTIVITIES
    # DFS order fixes a stable extension order for tree labels missing from a view.
    tree_alphabet = tuple(dict.fromkeys(family.canonical_tree.activity_labels()))
    pool_alphabet = tuple(
        sorted({label for trace in family.clean_trace_pool for label in trace})
    )
    alphabet = tuple(
        dict.fromkeys((*tree_alphabet, *pool_alphabet))
    )

    rows: list[ProcessSample] = []
    strong_behavior_id = (
        family.exact_behavior_id
        if family.exact_behavior_id is not None
        else family.behavior_id
        if family.equivalence_certificate.status == "isomorphic"
        else None
    )
    for log_view in family.log_views:
        # Relabel every modality of this view with the labels an external log
        # would receive: A0, A1, ... in first-seen trace order.
        mapping = first_seen_activity_mapping(log_view.traces, alphabet, max_activities)
        view_tree = family.canonical_tree.relabel(mapping)
        view_traces = tuple(
            tuple(mapping.get(label, label) for label in trace) for trace in log_view.traces
        )
        view_edits = _relabel_trace_edits(log_view.trace_edits, mapping)
        for representation_slot, variant in enumerate(family.model_variants):
            view_graph = variant.petri_graph.relabel(mapping)
            decoder_targets = decoder_target_trees_for_sample(
                view_tree,
                view_traces,
                view_graph,
            )
            if strong_behavior_id is None:
                partial_order_id = None
                partial_order_kind = None
            elif variant.partial_order_equivalent is False:
                partial_order_id = f"{family.exact_behavior_id}:partial-order:interleaved"
                partial_order_kind = "interleaved"
            elif variant.partial_order_equivalent is True:
                partial_order_id = f"{family.exact_behavior_id}:partial-order:concurrent"
                partial_order_kind = "concurrent"
            else:
                partial_order_id = f"{strong_behavior_id}:partial-order:shared"
                partial_order_kind = None
            row_signature = behavior_signature_with_partial_order(
                family.behavior_signature,
                partial_order_kind,
            )
            rows.append(
                ProcessSample(
                    tree=view_tree,
                    traces=view_traces,
                    petri_graph=view_graph,
                    equivalence_id=family.behavior_id,
                    exact_behavior_id=family.exact_behavior_id,
                    strong_behavior_id=strong_behavior_id,
                    complexity_level=str(
                        family.metadata.get("complexity_level", "complex")
                    ),
                    complexity_role=str(
                        family.metadata.get("complexity_role", "ordinary_tree")
                    ),
                    behavior_signature=row_signature,
                    exact_trace_language_id=family.exact_trace_language_id,
                    partial_order_id=partial_order_id,
                    structural_motif_id=str(family.metadata.get("motif", "unknown")),
                    model_variant_id=variant.variant_id,
                    log_view_id=log_view.log_view_id,
                    representation_kind=variant.representation_kind,
                    equivalence_level=variant.equivalence_level,
                    decoder_target_trees=decoder_targets,
                    metadata={
                        **family.metadata,
                        "representation_slot": representation_slot,
                        "sampling_mode": log_view.sampling_mode,
                        "label_scheme": "first_seen_per_log_view",
                        "trace_edits": [
                            [edit.to_dict() for edit in trace_edits]
                            for trace_edits in view_edits
                        ],
                        "original_trace_lengths": [
                            len(trace) for trace in view_traces
                        ],
                        "was_truncated": False,
                        "observation_quality": (
                            "noisy" if any(log_view.trace_edits) else log_view.sampling_mode
                        ),
                        "transformation_sequence": list(variant.transformation_sequence),
                        "structural_statistics": variant.structural_statistics,
                        "equivalence_certificate": family.equivalence_certificate.to_dict(),
                        "exact_behavior_id": family.exact_behavior_id,
                        "strong_behavior_id": strong_behavior_id,
                        "exact_trace_language_id": family.exact_trace_language_id,
                        "partial_order_id": partial_order_id,
                        "structural_motif_id": str(
                            family.metadata.get("motif", "unknown")
                        ),
                    },
                )
            )
    return rows


def generate_family_samples(
    count: int,
    config: Any,
    seed: int,
    split: str,
    progress_update: Callable[[int], None] | None = None,
    num_workers: int | None = None,
) -> list[Any]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return []
    rows_per_family = max(1, int(getattr(config, "variants_per_behavior", 2))) * max(
        1, int(getattr(config, "log_views_per_behavior", 2))
    )
    mode = str(getattr(config, "class_coverage_mode", "strict"))
    if mode == "strict" and split in getattr(config, "min_families_per_motif", {}):
        if count % rows_per_family:
            raise ValueError(
                f"strict class coverage requires the {split} sample count ({count}) to "
                f"be divisible by rows_per_family ({rows_per_family}); use family counts "
                "or class_coverage mode 'best_effort'"
            )
    family_count = (count + rows_per_family - 1) // rows_per_family
    generation_plan, _ = family_generation_plan(family_count, config, seed, split)
    if num_workers is not None:
        return _generate_family_samples_parallel(
            count=count,
            config=config,
            seed=seed,
            split=split,
            generation_plan=generation_plan,
            rows_per_family=rows_per_family,
            num_workers=num_workers,
            progress_update=progress_update,
        )
    rows: list[Any] = []
    family_index = 0
    attempts = 0
    max_attempts = max(100, family_count * 20)
    for planned in generation_plan:
        accepted = False
        while not accepted and attempts < max_attempts:
            attempts += 1
            try:
                family = generate_behavior_family(
                    config,
                    family_index,
                    seed,
                    split=split,
                    motif=planned.motif,
                    forced_root_operator=planned.forced_root_operator,
                )
            except (RuntimeError, ValueError):
                family_index += 1
                continue
            family_index += 1
            if (
                split == "training"
                and bool(getattr(config, "exact_equivalence_only_for_training", True))
                and family.equivalence_certificate.status not in {"exact", "isomorphic"}
            ):
                continue
            family_rows = flatten_behavior_family(
                family,
                max_activities=int(getattr(config, "max_activities", 30)),
            )
            accepted_rows = family_rows[: count - len(rows)]
            rows.extend(accepted_rows)
            if progress_update is not None:
                progress_update(len(accepted_rows))
            accepted = True
        if not accepted:
            break
    if len(rows) != count:
        raise RuntimeError(f"generated {len(rows)} of {count} requested family samples")
    return rows


def _generate_family_samples_parallel(
    *,
    count: int,
    config: Any,
    seed: int,
    split: str,
    generation_plan: Sequence[FamilyGenerationPlan],
    rows_per_family: int,
    num_workers: int,
    progress_update: Callable[[int], None] | None,
) -> list[Any]:
    if num_workers < 1:
        raise ValueError("num_workers must be positive")

    accepted_rows: list[list[Any] | None] = [None] * len(generation_plan)
    unresolved = list(range(len(generation_plan)))
    next_family_index = 0
    attempts = 0
    max_attempts = max(100, len(generation_plan) * 20)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        while unresolved and attempts < max_attempts:
            batch_size = min(len(unresolved), max_attempts - attempts)
            slots = unresolved[:batch_size]
            family_indices = range(next_family_index, next_family_index + batch_size)
            candidates = executor.map(
                _generate_behavior_family_candidate,
                repeat(config),
                family_indices,
                repeat(seed),
                repeat(split),
                (generation_plan[slot] for slot in slots),
            )
            attempts += batch_size
            next_family_index += batch_size
            retry_slots = unresolved[batch_size:]

            for slot, family in zip(slots, candidates):
                if family is None:
                    retry_slots.append(slot)
                    continue
                if (
                    split == "training"
                    and bool(getattr(config, "exact_equivalence_only_for_training", True))
                    and family.equivalence_certificate.status not in {"exact", "isomorphic"}
                ):
                    retry_slots.append(slot)
                    continue
                family_rows = flatten_behavior_family(
                    family,
                    max_activities=int(getattr(config, "max_activities", 30)),
                )
                row_offset = slot * rows_per_family
                rows_for_slot = family_rows[: max(0, min(rows_per_family, count - row_offset))]
                accepted_rows[slot] = rows_for_slot
                if progress_update is not None:
                    progress_update(len(rows_for_slot))
            unresolved = retry_slots

    rows = [row for family_rows in accepted_rows if family_rows for row in family_rows]
    if len(rows) != count:
        raise RuntimeError(f"generated {len(rows)} of {count} requested family samples")
    return rows


def _generate_behavior_family_candidate(
    config: Any,
    family_index: int,
    seed: int,
    split: str,
    plan: FamilyGenerationPlan,
) -> BehaviorFamily | None:
    try:
        return generate_behavior_family(
            config,
            family_index,
            seed,
            split=split,
            motif=plan.motif,
            forced_root_operator=plan.forced_root_operator,
        )
    except (RuntimeError, ValueError):
        return None


def enumerate_visible_language(
    net: Any,
    initial_marking: Any,
    final_marking: Any,
    *,
    max_states: int = 5000,
    max_traces: int = 10000,
    max_visible_length: int = 20,
) -> tuple[set[tuple[str, ...]], bool]:
    """Enumerate complete visible traces with explicit, honest resource bounds."""

    places = sorted(net.places, key=lambda place: str(place.name))
    transitions = sorted(net.transitions, key=lambda transition: str(transition.name))
    initial = Counter({place: int(tokens) for place, tokens in initial_marking.items()})
    final = Counter({place: int(tokens) for place, tokens in final_marking.items()})
    stack: list[tuple[Counter, tuple[str, ...], int]] = [(initial, (), 0)]
    seen: set[tuple[tuple[int, ...], tuple[str, ...], int]] = set()
    language: set[tuple[str, ...]] = set()
    completed = True

    while stack:
        marking, trace, firings = stack.pop()
        state_key = (
            tuple(int(marking.get(place, 0)) for place in places),
            trace,
            firings,
        )
        if state_key in seen:
            continue
        seen.add(state_key)
        if len(seen) > max_states or len(language) > max_traces:
            completed = False
            break
        if _marking_equal(marking, final):
            language.add(trace)
            continue

        enabled = [transition for transition in transitions if _enabled(transition, marking)]
        for transition in reversed(enabled):
            next_trace = trace
            if transition.label is not None:
                if len(trace) >= max_visible_length:
                    completed = False
                    continue
                next_trace = (*trace, str(transition.label))
            if firings >= max_visible_length * 4 + len(transitions):
                completed = False
                continue
            stack.append((_fire(transition, marking), next_trace, firings + 1))
    return language, completed


def structural_statistics(
    net: Any,
    initial_marking: Any,
    final_marking: Any,
    *,
    nonblock_reason: str | None = None,
) -> dict[str, object]:
    labels = [str(t.label) for t in net.transitions if t.label is not None]
    counts = Counter(labels)
    graph = petri_net_to_graph(net, initial_marking, final_marking)
    reachability = _reachability_statistics(net, initial_marking, final_marking)
    return {
        "num_places": len(net.places),
        "num_transitions": len(net.transitions),
        "num_arcs": len(net.arcs),
        "num_visible_transitions": len(labels),
        "num_invisible_transitions": sum(1 for t in net.transitions if t.label is None),
        "invisible_transition_ratio": (
            sum(1 for t in net.transitions if t.label is None) / max(1, len(net.transitions))
        ),
        "duplicate_label_count": sum(value for value in counts.values() if value > 1),
        "free_choice_violation_count": _free_choice_violation_count(net),
        "has_cycle": _has_directed_cycle(graph),
        "cyclic_scc_count": _cyclic_scc_count(graph),
        "initial_token_count": sum(int(value) for value in initial_marking.values()),
        "final_token_count": sum(int(value) for value in final_marking.values()),
        **reachability,
        "nonblock_reason": nonblock_reason,
    }


def normalized_visible_language(
    language: Sequence[Sequence[str]],
    tree: ProcessTreeNode,
) -> tuple[tuple[str, ...], ...]:
    """Canonicalize activity identity before hashing behavioral evidence."""

    normalized_tree = tree.normalize(canonicalize_activity_labels=False)
    mapping = normalized_tree.activity_label_mapping()
    normalized = {
        tuple(mapping.get(str(label), str(label)) for label in trace)
        for trace in language
    }
    return tuple(sorted(normalized))


def exact_language_behavior_id(
    language: Sequence[Sequence[str]],
    tree: ProcessTreeNode,
) -> str:
    payload = "\n".join(
        "|".join(trace) for trace in normalized_visible_language(language, tree)
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def bounded_behavior_signature(
    language: Sequence[Sequence[str]],
    tree: ProcessTreeNode,
    dimensions: int = 128,
) -> tuple[float, ...]:
    """Build a compact feature-hashed language/footprint signature."""

    if dimensions <= 0:
        raise ValueError("behavior signature dimensions must be positive")
    traces = normalized_visible_language(language, tree)
    features: Counter[str] = Counter()
    for trace in traces:
        features[f"length:{len(trace)}"] += 1
        if trace:
            features[f"start:{trace[0]}"] += 1
            features[f"end:{trace[-1]}"] += 1
        features.update(f"activity:{label}" for label in trace)
        features.update(
            f"direct:{left}>{right}" for left, right in zip(trace, trace[1:])
        )
        features.update(
            f"eventual:{trace[left]}>{trace[right]}"
            for left in range(len(trace))
            for right in range(left + 1, len(trace))
        )
        features[f"variant:{'|'.join(trace)}"] += 1

    vector = [0.0] * dimensions
    for feature, count in features.items():
        digest = blake2b(feature.encode("utf-8"), digest_size=9).digest()
        index = int.from_bytes(digest[:8], "big") % dimensions
        sign = 1.0 if digest[8] & 1 else -1.0
        vector[index] += sign * float(count)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return tuple(round(value, 8) for value in vector)


def behavior_signature_with_partial_order(
    signature: Sequence[float],
    partial_order_kind: str | None,
    strength: float = 0.25,
) -> tuple[float, ...]:
    """Add a small partial-order component without turning analogues into negatives."""

    values = [float(value) for value in signature]
    if partial_order_kind is None or not values:
        return tuple(values)
    direction = [0.0] * len(values)
    for index in range(len(values)):
        digest = blake2b(
            f"partial-order:{partial_order_kind}:{index}".encode("utf-8"),
            digest_size=1,
        ).digest()[0]
        direction[index] = 1.0 if digest & 1 else -1.0
    direction_norm = math.sqrt(sum(value * value for value in direction))
    combined = [
        value + strength * component / max(direction_norm, 1e-12)
        for value, component in zip(values, direction)
    ]
    norm = math.sqrt(sum(value * value for value in combined))
    if norm:
        combined = [value / norm for value in combined]
    return tuple(round(value, 8) for value in combined)


_CONTEXT_PLACEHOLDER = "__PROC_ROSETTA_MOTIF__"


def _apply_random_structural_context(
    motif_tree: ProcessTreeNode,
    variants: Sequence[_RuntimeVariant],
    rng: random.Random,
    behavior_id: str,
    *,
    max_activities: int,
    min_nodes: int,
    max_nodes: int,
) -> tuple[ProcessTreeNode, tuple[_RuntimeVariant, ...], dict[str, object]]:
    """Insert a motif into one randomized structural context for every view."""

    used = set(motif_tree.unique_activity_labels())
    available = max_activities - len(used)
    if available < 2:
        raise ValueError("controlled motifs require room for at least two context activities")
    # The template also contains the placeholder leaf, so a binary tree with
    # ``n`` context activities has 2*n+1 nodes.  Cap at four context leaves to
    # keep an all-AND complete language exactly enumerable under default gates.
    min_leaves = max(2, math.ceil((min_nodes - 1) / 2))
    max_leaves = min(available, 4, max(2, (max_nodes - 1) // 2))
    if min_leaves > max_leaves:
        raise ValueError(
            "structural context bounds are infeasible for the activity and language limits"
        )
    leaf_count = rng.randint(min_leaves, max_leaves)
    labels: list[str] = []
    candidate = 0
    while len(labels) < leaf_count:
        label = f"CTX{candidate}"
        candidate += 1
        if label not in used:
            labels.append(label)

    nodes = [ProcessTreeNode.activity(_CONTEXT_PLACEHOLDER)]
    nodes.extend(ProcessTreeNode.activity(label) for label in labels)
    rng.shuffle(nodes)
    operators: list[str] = []
    while len(nodes) > 1:
        left_index, right_index = sorted(rng.sample(range(len(nodes)), 2), reverse=True)
        left = nodes.pop(left_index)
        right = nodes.pop(right_index)
        kind = rng.choice((NodeKind.SEQ, NodeKind.XOR, NodeKind.AND))
        operators.append(kind.value)
        if kind is NodeKind.SEQ and rng.random() < 0.5:
            left, right = right, left
        nodes.append(ProcessTreeNode(kind, children=(left, right)))
    template = nodes[0]
    completed_tree = _replace_context_placeholder(template, motif_tree)
    completed_variants = tuple(
        _compile_context_variant(template, variant, f"{behavior_id}-context-{index}")
        for index, variant in enumerate(variants)
    )
    return completed_tree, completed_variants, {
        "template": template.to_dict(),
        "node_count": template.size(),
        "activity_count": leaf_count,
        "operators": operators,
    }


def _replace_context_placeholder(
    template: ProcessTreeNode,
    motif_tree: ProcessTreeNode,
) -> ProcessTreeNode:
    if template.kind is NodeKind.ACTIVITY and template.label == _CONTEXT_PLACEHOLDER:
        return motif_tree
    if not template.children:
        return template
    return ProcessTreeNode(
        template.kind,
        children=tuple(
            _replace_context_placeholder(child, motif_tree) for child in template.children
        ),
    )


def _contains_context_placeholder(node: ProcessTreeNode) -> bool:
    return (
        node.kind is NodeKind.ACTIVITY
        and node.label == _CONTEXT_PLACEHOLDER
    ) or any(_contains_context_placeholder(child) for child in node.children)


def _compile_context_variant(
    template: ProcessTreeNode,
    motif: _RuntimeVariant,
    name: str,
) -> _RuntimeVariant:
    if template.kind is NodeKind.ACTIVITY and template.label == _CONTEXT_PLACEHOLDER:
        clone = _clone_isomorphic(
            motif.net, motif.initial_marking, motif.final_marking, f"{name}-motif"
        )
        return _RuntimeVariant(
            motif.representation_kind,
            clone[0],
            clone[1],
            clone[2],
            motif.transformations,
            nonblock_reason=motif.nonblock_reason,
            partial_order_equivalent=motif.partial_order_equivalent,
        )
    if not _contains_context_placeholder(template):
        bundle = tree_to_petri_net(template)
        return _RuntimeVariant(
            "structural_context",
            bundle.net,
            bundle.initial_marking,
            bundle.final_marking,
            ("context_tree_to_petri",),
        )
    if template.kind not in {NodeKind.SEQ, NodeKind.XOR, NodeKind.AND}:
        raise ValueError("random structural context supports SEQ, XOR, and AND")
    children = tuple(
        _compile_context_variant(child, motif, f"{name}-{index}")
        for index, child in enumerate(template.children)
    )
    composed = _compose_runtime_variants(template.kind, children, name)
    return _RuntimeVariant(
        motif.representation_kind,
        composed[0],
        composed[1],
        composed[2],
        (*motif.transformations, f"shared_{template.kind.value}_context"),
        nonblock_reason=motif.nonblock_reason,
        partial_order_equivalent=motif.partial_order_equivalent,
    )


def _compose_runtime_variants(
    kind: NodeKind,
    components: Sequence[_RuntimeVariant],
    name: str,
):
    from pm4py.objects.petri_net.obj import Marking, PetriNet
    from pm4py.objects.petri_net.utils import petri_utils

    net = PetriNet(name)
    markings: list[tuple[Any, Any]] = []
    for component_index, component in enumerate(components):
        places = sorted(component.net.places, key=lambda place: str(place.name))
        transitions = sorted(
            component.net.transitions, key=lambda transition: str(transition.name)
        )
        place_map = {
            place: PetriNet.Place(f"c{component_index}_p{index}_{place.name}")
            for index, place in enumerate(places)
        }
        transition_map = {
            transition: PetriNet.Transition(
                f"c{component_index}_t{index}_{transition.name}", transition.label
            )
            for index, transition in enumerate(transitions)
        }
        net.places.update(place_map.values())
        net.transitions.update(transition_map.values())
        node_map = {**place_map, **transition_map}
        for arc in component.net.arcs:
            petri_utils.add_arc_from_to(
                node_map[arc.source], node_map[arc.target], net, int(arc.weight)
            )
        markings.append(
            (
                Marking({place_map[p]: int(tokens) for p, tokens in component.initial_marking.items()}),
                Marking({place_map[p]: int(tokens) for p, tokens in component.final_marking.items()}),
            )
        )

    def connect(marking: Any, transition: Any, outgoing: bool) -> None:
        for place, tokens in marking.items():
            if outgoing:
                petri_utils.add_arc_from_to(transition, place, net, int(tokens))
            else:
                petri_utils.add_arc_from_to(place, transition, net, int(tokens))

    if kind is NodeKind.SEQ:
        for index in range(len(markings) - 1):
            bridge = PetriNet.Transition(f"context_seq_{index}", None)
            net.transitions.add(bridge)
            connect(markings[index][1], bridge, False)
            connect(markings[index + 1][0], bridge, True)
        return net, markings[0][0], markings[-1][1]

    source = PetriNet.Place("context_source")
    sink = PetriNet.Place("context_sink")
    net.places.update({source, sink})
    if kind is NodeKind.XOR:
        for index, (initial, final) in enumerate(markings):
            enter = PetriNet.Transition(f"context_xor_enter_{index}", None)
            leave = PetriNet.Transition(f"context_xor_leave_{index}", None)
            net.transitions.update({enter, leave})
            petri_utils.add_arc_from_to(source, enter, net)
            connect(initial, enter, True)
            connect(final, leave, False)
            petri_utils.add_arc_from_to(leave, sink, net)
    elif kind is NodeKind.AND:
        split = PetriNet.Transition("context_and_split", None)
        join = PetriNet.Transition("context_and_join", None)
        net.transitions.update({split, join})
        petri_utils.add_arc_from_to(source, split, net)
        for initial, final in markings:
            connect(initial, split, True)
            connect(final, join, False)
        petri_utils.add_arc_from_to(join, sink, net)
    else:
        raise ValueError(f"unsupported context operator: {kind.value}")
    return net, Marking({source: 1}), Marking({sink: 1})


def _ordinary_tree_family(
    config: Any,
    rng: random.Random,
    behavior_id: str,
    *,
    forced_root_operator: NodeKind | None = None,
):
    # Local import avoids coupling the neutral family structures to the legacy API.
    from proc_rosetta.synthetic import generate_process_tree_with_provenance

    tree, provenance = generate_process_tree_with_provenance(
        config,
        rng,
        forced_root_operator=forced_root_operator,
    )
    bundle = tree_to_petri_net(tree)
    renamed = _clone_isomorphic(
        bundle.net, bundle.initial_marking, bundle.final_marking, f"{behavior_id}-renamed"
    )
    return tree, (
        _RuntimeVariant(
            "canonical_block_pm4py",
            bundle.net,
            bundle.initial_marking,
            bundle.final_marking,
            ("process_tree_to_petri",),
        ),
        _RuntimeVariant(
            "isomorphic_renaming",
            renamed[0],
            renamed[1],
            renamed[2],
            ("process_tree_to_petri", "isomorphic_renaming"),
        ),
    ), provenance


def _relabel_runtime_variant(
    variant: _RuntimeVariant,
    mapping: dict[str, str],
) -> _RuntimeVariant:
    for transition in variant.net.transitions:
        if transition.label is not None:
            transition.label = mapping.get(str(transition.label), str(transition.label))
    return variant


def _duplicate_vs_silent_family(behavior_id: str):
    tree = ProcessTreeNode.xor(
        ProcessTreeNode.seq(ProcessTreeNode.activity("A0"), ProcessTreeNode.activity("A1")),
        ProcessTreeNode.seq(ProcessTreeNode.activity("A0"), ProcessTreeNode.activity("A2")),
    )
    duplicate = _make_duplicate_prefix_net(f"{behavior_id}-duplicate")
    silent = _make_silent_routing_net(f"{behavior_id}-silent")
    return tree, (
        _RuntimeVariant("duplicate_prefix", *duplicate, ("duplicate_visible_prefix",)),
        _RuntimeVariant("silent_routing", *silent, ("shared_activity_tau_routing",)),
    )


def _concurrent_vs_interleaved_family(behavior_id: str):
    tree = ProcessTreeNode.and_(
        ProcessTreeNode.activity("A0"), ProcessTreeNode.activity("A1")
    )
    bundle = tree_to_petri_net(tree)
    interleaved = _make_interleaving_net(f"{behavior_id}-interleaved")
    return tree, (
        _RuntimeVariant(
            "parallel",
            bundle.net,
            bundle.initial_marking,
            bundle.final_marking,
            ("true_token_concurrency",),
            partial_order_equivalent=True,
        ),
        _RuntimeVariant(
            "explicit_interleaving",
            *interleaved,
            ("progress_state_interleaving",),
            partial_order_equivalent=False,
        ),
    )


def _m_nonfreechoice_family(behavior_id: str):
    tree = ProcessTreeNode.xor(
        ProcessTreeNode.activity("A0"),
        ProcessTreeNode.and_(
            ProcessTreeNode.activity("A1"), ProcessTreeNode.activity("A2")
        ),
    )
    bundle = tree_to_petri_net(tree)
    nonfree = _make_m_nonfreechoice_net(f"{behavior_id}-m")
    return tree, (
        _RuntimeVariant(
            "canonical_block_pm4py",
            bundle.net,
            bundle.initial_marking,
            bundle.final_marking,
            ("process_tree_to_petri",),
        ),
        _RuntimeVariant(
            "m_nonfreechoice",
            *nonfree,
            ("non_free_choice_m_pattern",),
            nonblock_reason="non_free_choice_m_pattern",
        ),
    )


def _make_duplicate_prefix_net(name: str):
    from pm4py.objects.petri_net.obj import Marking, PetriNet
    from pm4py.objects.petri_net.utils import petri_utils

    net = PetriNet(name)
    p0, pl, pr, pf = (PetriNet.Place(value) for value in ("p0", "pl", "pr", "pf"))
    net.places.update({p0, pl, pr, pf})
    transitions = (
        PetriNet.Transition("t_A0_left", "A0"),
        PetriNet.Transition("t_A0_right", "A0"),
        PetriNet.Transition("t_A1", "A1"),
        PetriNet.Transition("t_A2", "A2"),
    )
    net.transitions.update(transitions)
    for source, transition, target in (
        (p0, transitions[0], pl),
        (p0, transitions[1], pr),
        (pl, transitions[2], pf),
        (pr, transitions[3], pf),
    ):
        petri_utils.add_arc_from_to(source, transition, net)
        petri_utils.add_arc_from_to(transition, target, net)
    return net, Marking({p0: 1}), Marking({pf: 1})


def _make_silent_routing_net(name: str):
    from pm4py.objects.petri_net.obj import Marking, PetriNet
    from pm4py.objects.petri_net.utils import petri_utils

    net = PetriNet(name)
    p0, pc, pl, pr, pf = (
        PetriNet.Place(value) for value in ("p0", "pc", "pl", "pr", "pf")
    )
    net.places.update({p0, pc, pl, pr, pf})
    a = PetriNet.Transition("t_A0", "A0")
    tau_l = PetriNet.Transition("tau_left", None)
    tau_r = PetriNet.Transition("tau_right", None)
    b = PetriNet.Transition("t_A1", "A1")
    c = PetriNet.Transition("t_A2", "A2")
    net.transitions.update({a, tau_l, tau_r, b, c})
    for source, transition, target in (
        (p0, a, pc),
        (pc, tau_l, pl),
        (pc, tau_r, pr),
        (pl, b, pf),
        (pr, c, pf),
    ):
        petri_utils.add_arc_from_to(source, transition, net)
        petri_utils.add_arc_from_to(transition, target, net)
    return net, Marking({p0: 1}), Marking({pf: 1})


def _make_interleaving_net(name: str):
    from pm4py.objects.petri_net.obj import Marking, PetriNet
    from pm4py.objects.petri_net.utils import petri_utils

    net = PetriNet(name)
    p0, pa, pb, pf = (PetriNet.Place(value) for value in ("p0", "pa", "pb", "pf"))
    net.places.update({p0, pa, pb, pf})
    transitions = (
        PetriNet.Transition("t_A0_first", "A0"),
        PetriNet.Transition("t_A1_after_A0", "A1"),
        PetriNet.Transition("t_A1_first", "A1"),
        PetriNet.Transition("t_A0_after_A1", "A0"),
    )
    net.transitions.update(transitions)
    for source, transition, target in (
        (p0, transitions[0], pa),
        (pa, transitions[1], pf),
        (p0, transitions[2], pb),
        (pb, transitions[3], pf),
    ):
        petri_utils.add_arc_from_to(source, transition, net)
        petri_utils.add_arc_from_to(transition, target, net)
    return net, Marking({p0: 1}), Marking({pf: 1})


def _make_m_nonfreechoice_net(name: str):
    from pm4py.objects.petri_net.obj import Marking, PetriNet
    from pm4py.objects.petri_net.utils import petri_utils

    net = PetriNet(name)
    p0, p1, p2, q1, q2, pf = (
        PetriNet.Place(value) for value in ("p0", "p1", "p2", "q1", "q2", "pf")
    )
    net.places.update({p0, p1, p2, q1, q2, pf})
    split = PetriNet.Transition("tau_split", None)
    a = PetriNet.Transition("t_A1", "A1")
    b = PetriNet.Transition("t_A0", "A0")
    c = PetriNet.Transition("t_A2", "A2")
    join = PetriNet.Transition("tau_join", None)
    net.transitions.update({split, a, b, c, join})
    petri_utils.add_arc_from_to(p0, split, net)
    petri_utils.add_arc_from_to(split, p1, net)
    petri_utils.add_arc_from_to(split, p2, net)
    petri_utils.add_arc_from_to(p1, a, net)
    petri_utils.add_arc_from_to(a, q1, net)
    petri_utils.add_arc_from_to(p2, c, net)
    petri_utils.add_arc_from_to(c, q2, net)
    petri_utils.add_arc_from_to(p1, b, net)
    petri_utils.add_arc_from_to(p2, b, net)
    petri_utils.add_arc_from_to(b, pf, net)
    petri_utils.add_arc_from_to(q1, join, net)
    petri_utils.add_arc_from_to(q2, join, net)
    petri_utils.add_arc_from_to(join, pf, net)
    return net, Marking({p0: 1}), Marking({pf: 1})


def _make_prefix_trie_net(language: set[tuple[str, ...]], name: str):
    from pm4py.objects.petri_net.obj import Marking, PetriNet
    from pm4py.objects.petri_net.utils import petri_utils

    net = PetriNet(name)
    source = PetriNet.Place("prefix_root")
    sink = PetriNet.Place("prefix_sink")
    net.places.update({source, sink})
    places: dict[tuple[str, ...], Any] = {(): source}
    edges: dict[tuple[tuple[str, ...], str], Any] = {}
    terminals: set[tuple[str, ...]] = set()
    for trace in sorted(language):
        prefix: tuple[str, ...] = ()
        for label in trace:
            next_prefix = (*prefix, label)
            if next_prefix not in places:
                place = PetriNet.Place(f"prefix_{len(places):05d}")
                places[next_prefix] = place
                net.places.add(place)
            edge_key = (prefix, label)
            if edge_key not in edges:
                transition = PetriNet.Transition(f"edge_{len(edges):05d}_{label}", label)
                edges[edge_key] = transition
                net.transitions.add(transition)
                petri_utils.add_arc_from_to(places[prefix], transition, net)
                petri_utils.add_arc_from_to(transition, places[next_prefix], net)
            prefix = next_prefix
        terminals.add(prefix)
    for index, prefix in enumerate(sorted(terminals)):
        transition = PetriNet.Transition(f"tau_accept_{index:05d}", None)
        net.transitions.add(transition)
        petri_utils.add_arc_from_to(places[prefix], transition, net)
        petri_utils.add_arc_from_to(transition, sink, net)
    return net, Marking({source: 1}), Marking({sink: 1})


def _clone_isomorphic(net: Any, initial_marking: Any, final_marking: Any, name: str):
    from pm4py.objects.petri_net.obj import Marking, PetriNet
    from pm4py.objects.petri_net.utils import petri_utils

    clone = PetriNet(name)
    places = sorted(net.places, key=lambda place: str(place.name))
    transitions = sorted(net.transitions, key=lambda transition: str(transition.name))
    place_map = {
        place: PetriNet.Place(f"renamed_p_{len(places) - index:04d}")
        for index, place in enumerate(places)
    }
    transition_map = {
        transition: PetriNet.Transition(
            f"renamed_t_{len(transitions) - index:04d}", transition.label
        )
        for index, transition in enumerate(transitions)
    }
    clone.places.update(place_map.values())
    clone.transitions.update(transition_map.values())
    node_map = {**place_map, **transition_map}
    for arc in net.arcs:
        petri_utils.add_arc_from_to(node_map[arc.source], node_map[arc.target], clone, arc.weight)
    initial = Marking({place_map[p]: tokens for p, tokens in initial_marking.items()})
    final = Marking({place_map[p]: tokens for p, tokens in final_marking.items()})
    return clone, initial, final


def _append_visible_suffix(
    variant: _RuntimeVariant,
    labels: Sequence[str],
    behavior_id: str,
) -> _RuntimeVariant:
    from pm4py.objects.petri_net.obj import Marking, PetriNet
    from pm4py.objects.petri_net.utils import petri_utils

    net = variant.net
    current_marking = variant.final_marking
    for index, label in enumerate(labels):
        target = PetriNet.Place(f"context_sink_{index:03d}")
        transition = PetriNet.Transition(f"context_{index:03d}_{label}", label)
        net.places.add(target)
        net.transitions.add(transition)
        for place, tokens in current_marking.items():
            petri_utils.add_arc_from_to(place, transition, net, int(tokens))
        petri_utils.add_arc_from_to(transition, target, net)
        current_marking = Marking({target: 1})
    return _RuntimeVariant(
        variant.representation_kind,
        net,
        variant.initial_marking,
        current_marking,
        (*variant.transformations, "shared_sequence_context"),
        nonblock_reason=variant.nonblock_reason,
        partial_order_equivalent=variant.partial_order_equivalent,
    )


def _tau_prefix_refinement(
    net: Any, initial_marking: Any, final_marking: Any, name: str
):
    from pm4py.objects.petri_net.obj import Marking, PetriNet
    from pm4py.objects.petri_net.utils import petri_utils

    clone, clone_initial, clone_final = _clone_isomorphic(
        net, initial_marking, final_marking, name
    )
    source = PetriNet.Place("tau_refinement_source")
    tau = PetriNet.Transition("tau_refinement_enter", None)
    clone.places.add(source)
    clone.transitions.add(tau)
    petri_utils.add_arc_from_to(source, tau, clone)
    for place, tokens in clone_initial.items():
        petri_utils.add_arc_from_to(tau, place, clone, int(tokens))
    return clone, Marking({source: 1}), clone_final


def _make_log_views(
    config: Any,
    trace_pool: tuple[tuple[str, ...], ...],
    behavior_id: str,
    seed: int,
) -> tuple[LogView, ...]:
    count = max(1, int(getattr(config, "log_views_per_behavior", 2)))
    traces_per_view = max(1, int(getattr(config, "traces_per_sample", 128)))
    modes = tuple(getattr(config, "log_view_modes", ("uniform_variants", "resampled")))
    maximum_length = max(1, int(getattr(config, "max_trace_length", 128)))
    trace_pool = tuple(trace for trace in trace_pool if len(trace) <= maximum_length)
    if not trace_pool:
        raise ValueError(
            f"behavior {behavior_id} has no traces within max_trace_length={maximum_length}"
        )
    alphabet = tuple(sorted({label for trace in trace_pool for label in trace}))
    views: list[LogView] = []
    for index in range(count):
        mode = modes[index % len(modes)] if modes else "uniform_variants"
        rng = random.Random(stable_seed(seed, "log-view", index))
        if mode == "uniform_variants":
            traces = [trace_pool[position % len(trace_pool)] for position in range(traces_per_view)]
            rng.shuffle(traces)
        elif mode == "long_tail" and len(trace_pool) > 1:
            common = trace_pool[0]
            traces = [common for _ in range(max(1, traces_per_view - len(trace_pool) + 1))]
            traces.extend(trace_pool[1:])
            traces = traces[:traces_per_view]
            rng.shuffle(traces)
        elif mode == "sparse":
            traces = list(trace_pool[: min(len(trace_pool), traces_per_view)])
        elif mode == "incomplete" and len(trace_pool) > 1:
            incomplete_pool = trace_pool[:-1]
            traces = [
                incomplete_pool[position % len(incomplete_pool)]
                for position in range(traces_per_view)
            ]
        else:
            traces = [rng.choice(trace_pool) for _ in range(traces_per_view)]
        trace_edits: list[tuple[TraceEdit, ...]] = [() for _ in traces]
        if mode == "noisy":
            corruptor = TraceCorruptor(dict(getattr(config, "noise_operation_weights", {})))
            noisy_traces: list[tuple[str, ...]] = []
            trace_edits = []
            clean_fraction = float(getattr(config, "noise_clean_fraction", 0.2))
            edit_weights = dict(getattr(config, "noise_edit_count_weights", {1: 1.0}))
            for trace in traces:
                if rng.random() < clean_fraction:
                    noisy_traces.append(trace)
                    trace_edits.append(())
                    continue
                counts = list(edit_weights)
                weights = [max(0.0, edit_weights[count]) for count in counts]
                edit_count = rng.choices(counts, weights=weights, k=1)[0]
                noisy, edits = corruptor.corrupt(trace, alphabet, rng, int(edit_count))
                noisy_traces.append(noisy)
                trace_edits.append(edits)
            traces = noisy_traces
            for trace_index, trace in enumerate(traces):
                if len(trace) > maximum_length:
                    traces[trace_index] = rng.choice(trace_pool)
                    trace_edits[trace_index] = ()
        views.append(
            LogView(
                log_view_id=f"{behavior_id}:log-{index:02d}",
                sampling_mode=mode,
                traces=tuple(traces),
                trace_edits=tuple(trace_edits),
            )
        )
    return tuple(views)


def _motif_weights(config: Any) -> dict[str, float]:
    raw = dict(getattr(config, "motif_weights", {}) or {})
    weights = {kind: float(raw.get(kind, 0.0)) for kind in MOTIF_KINDS}
    if sum(max(0.0, weight) for weight in weights.values()) <= 0:
        return {
            "ordinary_tree": 0.25,
            "duplicate_vs_silent": 0.25,
            "concurrent_vs_interleaved": 0.25,
            "m_nonfreechoice": 0.25,
        }
    return weights


def _tree_contains_kind(node: ProcessTreeNode, kind: NodeKind) -> bool:
    return node.kind is kind or any(_tree_contains_kind(child, kind) for child in node.children)


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights)
    values = [max(0.0, weights[key]) for key in keys]
    return rng.choices(keys, weights=values, k=1)[0]


def _stable_id(seed: int, *parts: object) -> str:
    return f"{stable_seed(seed, *parts):016x}"


def _enabled(transition: Any, marking: Counter) -> bool:
    return all(marking.get(arc.source, 0) >= int(arc.weight) for arc in transition.in_arcs)


def _fire(transition: Any, marking: Counter) -> Counter:
    result = Counter(marking)
    for arc in transition.in_arcs:
        result[arc.source] -= int(arc.weight)
        if result[arc.source] == 0:
            del result[arc.source]
    for arc in transition.out_arcs:
        result[arc.target] += int(arc.weight)
    return result


def _marking_equal(left: Counter, right: Counter) -> bool:
    return +left == +right


def _free_choice_violation_count(net: Any) -> int:
    violations: set[tuple[str, str]] = set()
    for place in net.places:
        outgoing = [arc.target for arc in place.out_arcs]
        for left_index, left in enumerate(outgoing):
            left_preset = {arc.source for arc in left.in_arcs}
            for right in outgoing[left_index + 1 :]:
                right_preset = {arc.source for arc in right.in_arcs}
                if left_preset != right_preset:
                    violations.add(tuple(sorted((str(left.name), str(right.name)))))
    return len(violations)


def _has_directed_cycle(graph: PetriGraph) -> bool:
    adjacency: dict[int, list[int]] = {index: [] for index in range(graph.num_nodes)}
    for source, target, _ in graph.edges:
        adjacency[source].append(target)
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(node: int) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(child) for child in adjacency[node]):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in adjacency)


def _cyclic_scc_count(graph: PetriGraph) -> int:
    adjacency: dict[int, list[int]] = {index: [] for index in range(graph.num_nodes)}
    for source, target, _ in graph.edges:
        adjacency[source].append(target)
    index = 0
    indices: dict[int, int] = {}
    lowlinks: dict[int, int] = {}
    stack: list[int] = []
    on_stack: set[int] = set()
    cyclic = 0

    def connect(node: int) -> None:
        nonlocal index, cyclic
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for child in adjacency[node]:
            if child not in indices:
                connect(child)
                lowlinks[node] = min(lowlinks[node], lowlinks[child])
            elif child in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[child])
        if lowlinks[node] != indices[node]:
            return
        component: list[int] = []
        while stack:
            child = stack.pop()
            on_stack.remove(child)
            component.append(child)
            if child == node:
                break
        if len(component) > 1 or any(child == node for child in adjacency[node]):
            cyclic += 1

    for node in adjacency:
        if node not in indices:
            connect(node)
    return cyclic


def _reachability_statistics(
    net: Any, initial_marking: Any, final_marking: Any, max_states: int = 5000
) -> dict[str, object]:
    places = sorted(net.places, key=lambda place: str(place.name))
    transitions = sorted(net.transitions, key=lambda transition: str(transition.name))
    stack = [Counter({place: int(tokens) for place, tokens in initial_marking.items()})]
    final = Counter({place: int(tokens) for place, tokens in final_marking.items()})
    seen: set[tuple[int, ...]] = set()
    fired: set[Any] = set()
    final_reachable = False
    max_tokens = 0
    truncated = False
    while stack:
        marking = stack.pop()
        key = tuple(marking.get(place, 0) for place in places)
        if key in seen:
            continue
        seen.add(key)
        max_tokens = max(max_tokens, sum(key))
        final_reachable = final_reachable or (+marking == +final)
        if len(seen) >= max_states:
            truncated = bool(stack)
            break
        for transition in transitions:
            if _enabled(transition, marking):
                fired.add(transition)
                stack.append(_fire(transition, marking))
    return {
        "reachable_state_count": len(seen),
        "reachability_truncated": truncated,
        "final_marking_reachable": final_reachable,
        "dead_transition_count": len(net.transitions) - len(fired),
        "max_reachable_token_count": max_tokens,
    }
