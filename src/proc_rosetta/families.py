from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import blake2b
import random
from typing import Any, Callable, Sequence

from proc_rosetta.pm4py_bridge import (
    PetriGraph,
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
    canonical_tree: ProcessTreeNode
    model_variants: tuple[ModelVariant, ...]
    clean_trace_pool: tuple[tuple[str, ...], ...]
    log_views: tuple[LogView, ...]
    equivalence_certificate: EquivalenceCertificate
    metadata: dict[str, object]


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


def generate_behavior_family(
    config: Any,
    family_index: int,
    seed: int,
    split: str = "training",
    motif: str | None = None,
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
        tree, runtime_variants = _ordinary_tree_family(config, rng, behavior_id)
    ordinary_loop_tree = motif == "ordinary_tree" and _tree_contains_kind(
        tree, NodeKind.LOOP
    )

    context_labels: tuple[str, ...] = ()
    if motif != "ordinary_tree":
        used_labels = set(tree.unique_activity_labels())
        context_limit = max(0, int(getattr(config, "max_activities", 30)) - len(used_labels))
        context_size = min(
            max(0, int(getattr(config, "motif_context_size", 2))), context_limit
        )
        next_index = 0
        labels: list[str] = []
        while len(labels) < context_size:
            label = f"A{next_index}"
            next_index += 1
            if label not in used_labels:
                labels.append(label)
        context_labels = tuple(labels)
        if context_labels:
            for label in context_labels:
                tree = ProcessTreeNode.seq(tree, ProcessTreeNode.activity(label))
            runtime_variants = tuple(
                _append_visible_suffix(variant, context_labels, behavior_id)
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
        trace_pool = tuple(
            sorted({tuple(trace) for trace in simulate_traces(tree, num_traces=playout_count)})
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
        return BehaviorFamily(
            behavior_id=behavior_id,
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
                "context_suffix": list(context_labels),
                "trace_pool_source": "process_tree_playout",
                "playout_trace_count": playout_count,
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
    log_views = _make_log_views(config, trace_pool, behavior_id, seed_bundle["log_view"])
    return BehaviorFamily(
        behavior_id=behavior_id,
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
            "context_suffix": list(context_labels),
        },
    )


def flatten_behavior_family(family: BehaviorFamily) -> list[Any]:
    # Import lazily to avoid the synthetic -> families -> synthetic cycle.
    from proc_rosetta.synthetic import ProcessSample

    rows: list[ProcessSample] = []
    for log_view in family.log_views:
        for representation_slot, variant in enumerate(family.model_variants):
            rows.append(
                ProcessSample(
                    tree=family.canonical_tree,
                    traces=log_view.traces,
                    petri_graph=variant.petri_graph,
                    equivalence_id=family.behavior_id,
                    model_variant_id=variant.variant_id,
                    log_view_id=log_view.log_view_id,
                    representation_kind=variant.representation_kind,
                    equivalence_level=variant.equivalence_level,
                    metadata={
                        **family.metadata,
                        "representation_slot": representation_slot,
                        "sampling_mode": log_view.sampling_mode,
                        "trace_edits": [
                            [edit.to_dict() for edit in trace_edits]
                            for trace_edits in log_view.trace_edits
                        ],
                        "observation_quality": (
                            "noisy" if any(log_view.trace_edits) else log_view.sampling_mode
                        ),
                        "transformation_sequence": list(variant.transformation_sequence),
                        "structural_statistics": variant.structural_statistics,
                        "equivalence_certificate": family.equivalence_certificate.to_dict(),
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
) -> list[Any]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return []
    rows_per_family = max(1, int(getattr(config, "variants_per_behavior", 2))) * max(
        1, int(getattr(config, "log_views_per_behavior", 1))
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
    motif_plan, _, _, _ = motif_quota_plan(family_count, config, seed, split)
    rows: list[Any] = []
    family_index = 0
    attempts = 0
    max_attempts = max(100, count * 20)
    for planned_motif in motif_plan:
        accepted = False
        while not accepted and attempts < max_attempts:
            attempts += 1
            try:
                family = generate_behavior_family(
                    config, family_index, seed, split=split, motif=planned_motif
                )
            except (RuntimeError, ValueError):
                family_index += 1
                continue
            family_index += 1
            if (
                split == "training"
                and bool(getattr(config, "exact_equivalence_only_for_training", True))
                and family.equivalence_certificate.status != "exact"
            ):
                continue
            family_rows = flatten_behavior_family(family)
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


def _ordinary_tree_family(config: Any, rng: random.Random, behavior_id: str):
    # Local import avoids coupling the neutral family structures to the legacy API.
    from proc_rosetta.synthetic import generate_process_tree

    tree = generate_process_tree(config, rng)
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
    )


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
    count = max(1, int(getattr(config, "log_views_per_behavior", 1)))
    traces_per_view = max(1, int(getattr(config, "traces_per_sample", 128)))
    modes = tuple(getattr(config, "log_view_modes", ("uniform_variants", "resampled")))
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
