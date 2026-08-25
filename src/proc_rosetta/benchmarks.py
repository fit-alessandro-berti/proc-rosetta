from __future__ import annotations

import math
import sys
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Hashable, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from proc_rosetta.behavior import behavioral_distance
from proc_rosetta.data import (
    ProcessBatchCollator,
    progress_bar,
    progress_iterator,
    sample_statistics,
)
from proc_rosetta.devices import resolve_device
from proc_rosetta.pm4py_bridge import (
    fold_process_tree,
    petri_graph_to_net,
    simulate_traces,
    to_pm4py_tree,
    tree_to_petri_net,
)
from proc_rosetta.synthetic import ProcessSample, decoder_target_trees_for_sample
from proc_rosetta.tree import ProcessTreeNode, sanitize_activity_labels
from proc_rosetta.training import evaluate_samples_from_checkpoint, load_checkpoint


Trace = Sequence[str]
FeatureDict = dict[Hashable, float]
CONFORMANCE_METHODS = ("token_based_replay", "footprints")


@dataclass(frozen=True)
class ValidationAuditConfig:
    decode_interval: int = 2
    full_interval: int = 10
    decode_family_count: int = 64
    discovery_family_count: int = 32
    beam_size: int = 5
    max_decode_length: int = 512
    cache_dir: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "decode_interval",
            "full_interval",
            "decode_family_count",
            "discovery_family_count",
            "beam_size",
            "max_decode_length",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")


_VALIDATION_BASELINE_CACHE: dict[str, dict[str, np.ndarray]] = {}
_VALIDATION_INDUCTIVE_CACHE: dict[
    tuple[str, str], list[dict[str, object]]
] = {}


@dataclass(frozen=True)
class Pm4pyPetriEmbeddingConfig:
    dimensions: int = 256
    num_walks: int = 5
    walk_length: int = 20
    window: int = 5
    epochs: int = 5
    seed: int = 42

    def to_dict(self) -> dict[str, int]:
        return {
            "dimensions": self.dimensions,
            "num_walks": self.num_walks,
            "walk_length": self.walk_length,
            "window": self.window,
            "epochs": self.epochs,
            "seed": self.seed,
        }


def rich_test_report(
    checkpoint_path: str,
    data_dir: str,
    samples: Sequence[ProcessSample],
    curriculum: str,
    batch_size: int = 16,
    device: str | None = None,
    include_pm4py_petri: bool = True,
    pm4py_petri_config: Pm4pyPetriEmbeddingConfig | None = None,
    show_progress: bool = False,
    conformance_method: str = "token_based_replay",
    max_decode_length: int = 512,
) -> dict[str, object]:
    validate_conformance_method(conformance_method)
    mismatches = [
        sample.complexity_level
        for sample in samples
        if sample.complexity_level != curriculum
    ]
    if mismatches:
        raise ValueError(
            f"selected samples contain complexity levels other than {curriculum!r}"
        )
    started = perf_counter()
    torch_device = resolve_device(device)
    device_name = str(torch_device)
    pm4py_petri_config = pm4py_petri_config or Pm4pyPetriEmbeddingConfig()
    stage_count = 8
    test_debug(
        f"[1/{stage_count}] Evaluating neural loss over "
        f"{math.ceil(len(samples) / max(batch_size, 1))} batches",
        enabled=show_progress,
    )
    loss_metrics = evaluate_samples_from_checkpoint(
        checkpoint_path=checkpoint_path,
        samples=samples,
        batch_size=batch_size,
        device=device_name,
        show_progress=show_progress,
        progress_desc=f"{curriculum.title()} test loss",
    )
    test_debug(
        f"[2/{stage_count}] Loading checkpoint and computing neural embeddings",
        enabled=show_progress,
    )
    model, _ = load_checkpoint(checkpoint_path, torch_device)
    neural_embeddings = proc_rosetta_embeddings(
        model,
        samples,
        batch_size=batch_size,
        device=device_name,
        show_progress=show_progress,
    )
    test_debug(
        f"[3/{stage_count}] Evaluating {8 * len(samples)} matched raw/deployment decodes",
        enabled=show_progress,
    )
    diagnostic_unbounded_decode_quality = decode_quality_report(
        model,
        samples,
        batch_size=batch_size,
        device=device_name,
        max_decode_length=max_decode_length,
        show_progress=show_progress,
        completion_policy="prefix_only",
    )
    deployment_decode_quality = decode_quality_report(
        model,
        samples,
        batch_size=batch_size,
        device=device_name,
        max_decode_length=max_decode_length,
        show_progress=show_progress,
        completion_policy="bounded",
    )
    test_debug(
        f"[4/{stage_count}] Running {2 * len(samples)} "
        f"{conformance_method_label(conformance_method)} evaluations "
        "(fitness + precision each)",
        enabled=show_progress,
    )
    discovery_quality = discovery_quality_report(
        model,
        samples,
        batch_size=batch_size,
        device=device_name,
        max_decode_length=max_decode_length,
        show_progress=show_progress,
        conformance_method=conformance_method,
    )

    pair_count = len(samples) * (len(samples) - 1) // 2
    test_debug(
        f"[5/{stage_count}] Computing {pair_count} pairwise behavioral distances",
        enabled=show_progress,
    )
    behavior = behavior_matrices(samples, show_progress=show_progress)
    methods: dict[str, dict[str, object]] = {}
    method_embeddings: dict[str, np.ndarray] = {}
    for name, matrix in neural_embeddings.items():
        method_embeddings[name] = matrix
        methods[name] = evaluate_embedding_method(matrix, behavior["mean_l1"])
        methods[name]["kind"] = "learned_proc_rosetta_latent"

    test_debug(
        f"[6/{stage_count}] Extracting {6 * len(samples)} deterministic baseline feature sets",
        enabled=show_progress,
    )
    baselines = deterministic_baseline_embeddings(samples, show_progress=show_progress)
    for name, matrix in baselines.items():
        method_embeddings[name] = matrix
        methods[name] = evaluate_embedding_method(matrix, behavior["mean_l1"])
        methods[name]["kind"] = "deterministic_baseline"

    if include_pm4py_petri:
        test_debug(
            f"[7/{stage_count}] Computing {len(samples)} PM4Py Petri Node2Vec embeddings",
            enabled=show_progress,
        )
        pm4py_embeddings, pm4py_report = pm4py_petri_method_report(
            samples,
            behavior["mean_l1"],
            pm4py_petri_config,
            show_progress=show_progress,
        )
        methods["pm4py_colonna_petri_node2vec"] = pm4py_report
        if pm4py_embeddings is not None:
            method_embeddings["pm4py_colonna_petri_node2vec"] = pm4py_embeddings
    else:
        test_debug(
            f"[7/{stage_count}] Skipping PM4Py Petri Node2Vec embeddings",
            enabled=show_progress,
        )

    test_debug(
        f"[8/{stage_count}] Aggregating rankings, retrieval, and equivalence metrics",
        enabled=show_progress,
    )
    report = {
        "split": "test",
        "curriculum": curriculum,
        "sample_count": len(samples),
        "behavior_family_count": len({sample.equivalence_id for sample in samples}),
        "dataset_statistics": sample_statistics(samples),
        "loss_metrics": round_float_dict(loss_metrics),
        "behavioral_distance_summary": summarize_distance_matrix(behavior["mean_l1"]),
        "behavioral_component_summaries": {
            key: summarize_distance_matrix(value)
            for key, value in behavior.items()
            if key != "mean_l1"
        },
        "diagnostic_unbounded_decode_quality": diagnostic_unbounded_decode_quality,
        "deployment_decode_quality": deployment_decode_quality,
        # Legacy consumers receive deployment results, never unbounded diagnostics.
        "decode_quality": deployment_decode_quality,
        "discovery_quality": discovery_quality,
        "cross_modal_retrieval": cross_modal_retrieval(
            neural_embeddings,
            exact_behavior_ids=[
                sample.strong_behavior_id or sample.exact_behavior_id
                for sample in samples
            ],
            partial_order_ids=[sample.partial_order_id for sample in samples],
            behavior_signatures=[sample.behavior_signature for sample in samples],
        ),
        "equivalence_families": equivalence_family_embedding_report(
            samples,
            neural_embeddings,
            show_progress=show_progress,
        ),
        "embedding_methods": methods,
        "method_ranking": rank_embedding_methods(methods),
        "method_comparisons_against_proc_rosetta_fused_mu": compare_methods_against_reference(
            method_embeddings,
            methods,
            reference_name="proc_rosetta_fused_mu",
        ),
        "references": {
            "pm4py_colonna_petri_node2vec": (
                "Colonna et al., Process mining embeddings: Learning vector "
                "representations for Petri nets, Intelligent Systems with "
                "Applications 23 (2024): 200423; pm4py.objects.petri_net.utils."
                "embeddings_similarity"
            ),
            "pm4py_log_case_features_mean_std": "pm4py.extract_features_dataframe aggregated per log",
        },
    }
    trace_decode = deployment_decode_quality["methods"]["proc_rosetta_trace_mu"]
    assert isinstance(trace_decode, dict)
    strata = trace_decode.get("strata", {})
    assert isinstance(strata, dict)
    report["metrics_by_motif"] = _strata_with_prefix(strata, "motif:")
    report["metrics_by_observation_quality"] = observation_quality_report(
        audit_samples,
        neural_embeddings,
        _strata_with_prefix(strata, "observation:"),
    )
    report["metrics_by_tree_size_quantile"] = _strata_with_prefix(
        strata, "tree_size_bin:"
    )
    report["metrics_by_loop_presence"] = _strata_with_prefix(strata, "loop:")
    strong_ids = [
        sample.strong_behavior_id or sample.exact_behavior_id
        for sample in audit_samples
    ]
    retrieval_rows = cross_modal_top1_rows(
        neural_embeddings["proc_rosetta_trace_mu"],
        neural_embeddings["proc_rosetta_tree_mu"],
        strong_ids,
        [sample.equivalence_id for sample in audit_samples],
    )
    report["family_bootstrap_95_intervals"] = {
        "exact_tree_match_rate": trace_decode["family_bootstrap_95_intervals"][
            "exact_tree_match_rate"
        ],
        "normalized_edit_distance": trace_decode["family_bootstrap_95_intervals"][
            "normalized_tree_edit_distance"
        ],
        "discovery_f1": discovery_quality["methods"]["proc_rosetta_trace_mu"][
            "family_bootstrap_95_intervals"
        ]["discovery_f1"],
        "cross_modal_recall_at_1": family_bootstrap_interval(
            retrieval_rows, "hit", boolean=True
        ),
        "behavior_distance_spearman": family_bootstrap_spearman_interval(
            neural_embeddings["proc_rosetta_fused_mu"],
            behavior["mean_l1"],
            [sample.equivalence_id for sample in audit_samples],
        ),
    }
    test_debug(
        f"Evaluation complete in {perf_counter() - started:.1f}s; formatting final report",
        enabled=show_progress,
    )
    return report


def validation_audit_level(
    epoch: int,
    config: ValidationAuditConfig,
    *,
    stage_transition: bool = False,
) -> str:
    if stage_transition or epoch % config.full_interval == 0:
        return "full"
    if epoch % config.decode_interval == 0:
        return "decode"
    return "semantic"


def fixed_validation_family_subset(
    samples: Sequence[ProcessSample],
    family_count: int,
) -> list[ProcessSample]:
    """Choose a deterministic family-complete, stratum-interleaved subset."""

    grouped: dict[str, list[ProcessSample]] = {}
    strata: dict[str, str] = {}
    for sample in samples:
        family_id = str(sample.equivalence_id)
        grouped.setdefault(family_id, []).append(sample)
        strata.setdefault(
            family_id,
            "|".join(
                (
                    str(sample.complexity_level),
                    str(sample.metadata.get("motif", "ordinary_tree")),
                    str(sample.metadata.get("observation_quality", "unknown")),
                )
            ),
        )
    by_stratum: dict[str, list[str]] = {}
    for family_id, stratum in strata.items():
        by_stratum.setdefault(stratum, []).append(family_id)
    for family_ids in by_stratum.values():
        family_ids.sort(
            key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
        )
    selected: list[str] = []
    stratum_names = sorted(by_stratum)
    while len(selected) < min(family_count, len(grouped)):
        progressed = False
        for stratum in stratum_names:
            family_ids = by_stratum[stratum]
            if family_ids:
                selected.append(family_ids.pop(0))
                progressed = True
                if len(selected) >= min(family_count, len(grouped)):
                    break
        if not progressed:
            break
    selected_set = set(selected)
    return [sample for sample in samples if str(sample.equivalence_id) in selected_set]


def _cached_validation_baselines(
    samples: Sequence[ProcessSample],
    *,
    show_progress: bool,
) -> dict[str, np.ndarray]:
    key = validation_split_hash(samples)
    if key not in _VALIDATION_BASELINE_CACHE:
        _VALIDATION_BASELINE_CACHE[key] = deterministic_baseline_embeddings(
            samples,
            show_progress=show_progress,
        )
    return {
        name: values.copy()
        for name, values in _VALIDATION_BASELINE_CACHE[key].items()
    }


def _cached_inductive_rows(
    samples: Sequence[ProcessSample],
    conformance_method: str,
) -> list[dict[str, object]]:
    key = (validation_split_hash(samples), conformance_method)
    if key not in _VALIDATION_INDUCTIVE_CACHE:
        _VALIDATION_INDUCTIVE_CACHE[key] = [
            evaluate_inductive_miner_discovery(
                sample,
                conformance_method=conformance_method,
            )
            for sample in samples
        ]
    return [dict(row) for row in _VALIDATION_INDUCTIVE_CACHE[key]]


def validation_audit_report(
    model,
    samples: Sequence[ProcessSample],
    curriculum: str,
    *,
    loss_metrics: dict[str, float],
    epoch: int,
    config: ValidationAuditConfig | None = None,
    batch_size: int = 16,
    device: str | None = None,
    show_progress: bool = False,
    stage_transition: bool = False,
    conformance_method: str = "token_based_replay",
) -> dict[str, object]:
    """Run test-equivalent learned metrics without resolving a checkpoint/path."""

    config = config or ValidationAuditConfig()
    level = validation_audit_level(
        epoch,
        config,
        stage_transition=stage_transition,
    )
    mismatches = [
        sample.complexity_level
        for sample in samples
        if sample.complexity_level not in {None, curriculum}
    ]
    if mismatches:
        raise ValueError(
            f"validation samples contain complexity levels other than {curriculum!r}"
        )
    if not samples:
        raise ValueError("validation audit requires at least one sample")
    decode_family_count = (
        config.decode_family_count
        if level in {"decode", "full"}
        else max(1, config.decode_family_count // 4)
    )
    audit_family_count = max(
        config.decode_family_count,
        config.discovery_family_count,
    )
    audit_samples = fixed_validation_family_subset(samples, audit_family_count)
    decode_samples = fixed_validation_family_subset(
        audit_samples,
        decode_family_count,
    )
    discovery_samples = fixed_validation_family_subset(
        audit_samples,
        config.discovery_family_count,
    )
    device_name = str(resolve_device(device))
    neural_embeddings = proc_rosetta_embeddings(
        model,
        audit_samples,
        batch_size=batch_size,
        device=device_name,
        show_progress=show_progress,
    )
    diagnostic_unbounded_decode_quality = decode_quality_report(
        model,
        decode_samples,
        batch_size=batch_size,
        device=device_name,
        max_decode_length=config.max_decode_length,
        show_progress=show_progress,
        completion_policy="prefix_only",
        beam_size=config.beam_size,
    )
    deployment_decode_quality = decode_quality_report(
        model,
        decode_samples,
        batch_size=batch_size,
        device=device_name,
        max_decode_length=config.max_decode_length,
        show_progress=show_progress,
        completion_policy="bounded",
        beam_size=config.beam_size,
    )
    if level == "full":
        discovery_quality = discovery_quality_report(
            model,
            discovery_samples,
            batch_size=batch_size,
            device=device_name,
            max_decode_length=config.max_decode_length,
            show_progress=show_progress,
            conformance_method=conformance_method,
            inductive_rows=_cached_inductive_rows(
                discovery_samples,
                conformance_method,
            ),
        )
    else:
        discovery_quality = {
            "description": (
                f"Periodic full validation discovery audit is not due at epoch {epoch}."
            ),
            "conformance_method": conformance_method,
            "max_decode_length": int(config.max_decode_length),
            "methods": {
                name: summarize_discovery_quality(
                    [],
                    conformance_method=conformance_method,
                )
                for name in ("proc_rosetta_trace_mu", "inductive_miner")
            },
        }
    behavior = behavior_matrices(
        audit_samples,
        show_progress=show_progress,
        cache_dir=config.cache_dir,
    )
    methods: dict[str, dict[str, object]] = {}
    method_embeddings: dict[str, np.ndarray] = {}
    for name, matrix in neural_embeddings.items():
        method_embeddings[name] = matrix
        methods[name] = evaluate_embedding_method(matrix, behavior["mean_l1"])
        methods[name]["kind"] = "learned_proc_rosetta_latent"
    baselines = _cached_validation_baselines(
        audit_samples,
        show_progress=show_progress,
    )
    for name, matrix in baselines.items():
        method_embeddings[name] = matrix
        methods[name] = evaluate_embedding_method(matrix, behavior["mean_l1"])
        methods[name]["kind"] = "deterministic_baseline"
    retrieval = cross_modal_retrieval(
        neural_embeddings,
        exact_behavior_ids=[
            sample.strong_behavior_id or sample.exact_behavior_id
            for sample in audit_samples
        ],
        partial_order_ids=[sample.partial_order_id for sample in audit_samples],
        behavior_signatures=[sample.behavior_signature for sample in audit_samples],
    )
    equivalence = equivalence_family_embedding_report(
        audit_samples,
        neural_embeddings,
        show_progress=show_progress,
    )
    report: dict[str, object] = {
        "split": "validation",
        "curriculum": curriculum,
        "sample_count": len(audit_samples),
        "total_validation_sample_count": len(samples),
        "behavior_family_count": len(
            {sample.equivalence_id for sample in audit_samples}
        ),
        "dataset_statistics": sample_statistics(audit_samples),
        "loss_metrics": round_float_dict(loss_metrics),
        "behavioral_distance_summary": summarize_distance_matrix(behavior["mean_l1"]),
        "behavioral_component_summaries": {
            key: summarize_distance_matrix(value)
            for key, value in behavior.items()
            if key != "mean_l1"
        },
        "diagnostic_unbounded_decode_quality": diagnostic_unbounded_decode_quality,
        "deployment_decode_quality": deployment_decode_quality,
        # Legacy consumers receive deployment results, never unbounded diagnostics.
        "decode_quality": deployment_decode_quality,
        "discovery_quality": discovery_quality,
        "cross_modal_retrieval": retrieval,
        "equivalence_families": equivalence,
        "embedding_methods": methods,
        "method_ranking": rank_embedding_methods(methods),
        "method_comparisons_against_proc_rosetta_fused_mu": compare_methods_against_reference(
            method_embeddings,
            methods,
            reference_name="proc_rosetta_fused_mu",
        ),
        "references": {
            "pm4py_colonna_petri_node2vec": (
                "Not rerun during validation checkpoint selection."
            ),
            "pm4py_log_case_features_mean_std": (
                "pm4py.extract_features_dataframe aggregated per log"
            ),
        },
    }
    trace_decode = deployment_decode_quality["methods"]["proc_rosetta_trace_mu"]
    assert isinstance(trace_decode, dict)
    strata = trace_decode.get("strata", {})
    assert isinstance(strata, dict)
    report["metrics_by_motif"] = _strata_with_prefix(strata, "motif:")
    report["metrics_by_observation_quality"] = observation_quality_report(
        samples,
        neural_embeddings,
        _strata_with_prefix(strata, "observation:"),
    )
    report["metrics_by_tree_size_quantile"] = _strata_with_prefix(
        strata, "tree_size_bin:"
    )
    report["metrics_by_loop_presence"] = _strata_with_prefix(strata, "loop:")
    strong_ids = [
        sample.strong_behavior_id or sample.exact_behavior_id
        for sample in samples
    ]
    retrieval_rows = cross_modal_top1_rows(
        neural_embeddings["proc_rosetta_trace_mu"],
        neural_embeddings["proc_rosetta_tree_mu"],
        strong_ids,
        [sample.equivalence_id for sample in samples],
    )
    report["family_bootstrap_95_intervals"] = {
        "exact_tree_match_rate": trace_decode["family_bootstrap_95_intervals"][
            "exact_tree_match_rate"
        ],
        "normalized_edit_distance": trace_decode["family_bootstrap_95_intervals"][
            "normalized_tree_edit_distance"
        ],
        "discovery_f1": (
            discovery_quality["methods"]["proc_rosetta_trace_mu"][
                "family_bootstrap_95_intervals"
            ]["discovery_f1"]
            if level == "full"
            else {"family_count": 0, "estimate": 0.0, "lower": 0.0, "upper": 0.0}
        ),
        "cross_modal_recall_at_1": family_bootstrap_interval(
            retrieval_rows, "hit", boolean=True
        ),
        "behavior_distance_spearman": family_bootstrap_spearman_interval(
            neural_embeddings["proc_rosetta_fused_mu"],
            behavior["mean_l1"],
            [sample.equivalence_id for sample in samples],
        ),
    }
    return report


def _strata_with_prefix(
    strata: dict[str, object],
    prefix: str,
) -> dict[str, object]:
    return {
        name.removeprefix(prefix): values
        for name, values in strata.items()
        if name.startswith(prefix)
    }


def observation_quality_report(
    samples: Sequence[ProcessSample],
    embeddings: dict[str, np.ndarray],
    decode_strata: dict[str, object],
) -> dict[str, object]:
    qualities = [
        str(
            sample.metadata.get(
                "observation_quality",
                sample.metadata.get("sampling_mode", "unknown"),
            )
        )
        for sample in samples
    ]
    exact_ids = [
        sample.strong_behavior_id or sample.exact_behavior_id for sample in samples
    ]
    family_ids = [
        str(getattr(sample, "equivalence_id", exact_ids[index]))
        for index, sample in enumerate(samples)
    ]
    trace = np.asarray(embeddings["proc_rosetta_trace_mu"], dtype=float)
    tree = np.asarray(embeddings["proc_rosetta_tree_mu"], dtype=float)
    trace_normalized = trace / np.maximum(
        np.linalg.norm(trace, axis=1, keepdims=True),
        1e-12,
    )
    clean_indices = [
        index
        for index, quality in enumerate(qualities)
        if quality.lower() in {"clean", "full", "uniform", "resampled"}
    ]
    if not clean_indices:
        clean_indices = list(range(len(samples)))
    report: dict[str, object] = {}
    for quality in sorted(set(qualities)):
        indices = [index for index, value in enumerate(qualities) if value == quality]
        retrieval = retrieval_metrics(
            trace[indices],
            tree,
            query_labels=[exact_ids[index] for index in indices],
            candidate_labels=exact_ids,
        )
        values = dict(decode_strata.get(quality, {}))
        clean_cosines: list[float] = []
        family_margins: list[float] = []
        for index in indices:
            clean_positives = [
                candidate
                for candidate in clean_indices
                if candidate != index and family_ids[candidate] == family_ids[index]
            ]
            if clean_positives:
                clean_cosines.append(
                    max(
                        float(trace_normalized[index] @ trace_normalized[candidate])
                        for candidate in clean_positives
                    )
                )
            positives = [
                candidate
                for candidate in range(len(samples))
                if candidate != index and family_ids[candidate] == family_ids[index]
            ]
            negatives = [
                candidate
                for candidate in range(len(samples))
                if family_ids[candidate] != family_ids[index]
            ]
            if positives and negatives:
                family_margins.append(
                    max(
                        float(trace_normalized[index] @ trace_normalized[candidate])
                        for candidate in positives
                    )
                    - max(
                        float(trace_normalized[index] @ trace_normalized[candidate])
                        for candidate in negatives
                    )
                )
        values.update(
            exact_retrieval_recall_at_1=retrieval["top1_accuracy"],
            exact_retrieval_recall_at_5=retrieval["recall_at_5"],
            exact_retrieval_mrr=retrieval["mrr"],
            clean_view_semantic_cosine=round_float(mean(clean_cosines)),
            worst_view_family_margin=round_float(mean(family_margins)),
        )
        report[quality] = values
    return report


def test_debug(message: str, enabled: bool = True) -> None:
    if enabled:
        print(f"[test] {message}", file=sys.stderr, flush=True)


def equivalence_family_embedding_report(
    samples: Sequence[ProcessSample],
    embeddings: dict[str, np.ndarray],
    show_progress: bool = False,
) -> dict[str, object]:
    """Measure representation invariance using explicit behavior-family IDs."""

    family_ids = [sample.equivalence_id for sample in samples]
    representation_kinds = [sample.representation_kind for sample in samples]
    family_sizes = Counter(family_ids)
    paired_indices = [index for index, value in enumerate(family_ids) if family_sizes[value] > 1]
    methods: dict[str, object] = {}
    iterator = progress_iterator(
        embeddings.items(),
        enabled=show_progress,
        total=len(embeddings),
        desc="Equivalence metrics",
        unit="method",
    )
    for method, matrix in iterator:
        normalized = np.asarray(matrix, dtype=float)
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        normalized = normalized / np.maximum(norms, 1e-12)
        similarities = normalized @ normalized.T
        within: list[float] = []
        between: list[float] = []
        by_kind: dict[str, list[float]] = {}
        log_resampling: list[float] = []
        margins: list[float] = []
        retrieval_hits = 0
        retrieval_count = 0
        for left in range(len(samples)):
            positive = [
                right
                for right in range(len(samples))
                if right != left and family_ids[right] == family_ids[left]
            ]
            negative = [
                right
                for right in range(len(samples))
                if family_ids[right] != family_ids[left]
            ]
            if positive and negative:
                margins.append(
                    max(float(similarities[left, right]) for right in positive)
                    - max(float(similarities[left, right]) for right in negative)
                )
                candidates = [right for right in range(len(samples)) if right != left]
                nearest = max(candidates, key=lambda right: similarities[left, right])
                retrieval_hits += int(family_ids[nearest] == family_ids[left])
                retrieval_count += 1
            for right in range(left + 1, len(samples)):
                similarity = float(similarities[left, right])
                if family_ids[left] == family_ids[right]:
                    within.append(similarity)
                    if (
                        samples[left].model_variant_id == samples[right].model_variant_id
                        and samples[left].log_view_id != samples[right].log_view_id
                    ):
                        log_resampling.append(similarity)
                    pair = "__vs__".join(
                        sorted((representation_kinds[left], representation_kinds[right]))
                    )
                    by_kind.setdefault(pair, []).append(1.0 - similarity)
                else:
                    between.append(similarity)
        methods[method] = {
            "within_family_cosine": round_float(mean(within)),
            "between_family_cosine": round_float(mean(between)),
            "equivalence_margin": round_float(mean(margins)),
            "behavior_id_retrieval_top1": round_float(
                retrieval_hits / retrieval_count if retrieval_count else 0.0
            ),
            "log_resampling_consistency": round_float(mean(log_resampling)),
            "representation_distance_by_kind": {
                key: round_float(mean(values)) for key, values in sorted(by_kind.items())
            },
        }
    return {
        "semantics": "visible_complete_trace_language",
        "behavior_count": len(family_sizes),
        "paired_behavior_count": sum(1 for size in family_sizes.values() if size > 1),
        "paired_sample_count": len(paired_indices),
        "methods": methods,
    }


@torch.no_grad()
def discovery_quality_report(
    model,
    samples: Sequence[ProcessSample],
    batch_size: int = 16,
    device: str | None = None,
    max_decode_length: int = 512,
    show_progress: bool = False,
    conformance_method: str = "token_based_replay",
    inductive_rows: Sequence[dict[str, object]] | None = None,
) -> dict[str, object]:
    validate_conformance_method(conformance_method)
    if inductive_rows is not None and len(inductive_rows) != len(samples):
        raise ValueError("cached Inductive Miner rows must match the selected samples")
    model.eval()
    torch_device = resolve_device(device)
    model.to(torch_device)
    collator = ProcessBatchCollator(model.tree_tokenizer, model.activity_tokenizer)
    rows: dict[str, list[dict[str, object]]] = {
        "proc_rosetta_trace_mu": [],
        "inductive_miner": [],
    }

    with progress_bar(
        total=2 * len(samples),
        enabled=show_progress,
        desc=(
            "Discovery replays"
            if conformance_method == "token_based_replay"
            else "Footprint conformance"
        ),
        unit="replay" if conformance_method == "token_based_replay" else "model",
    ) as progress:
        conformance_successes = 0
        completed_models = 0
        total_models = 2 * len(samples)
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            batch = _move_batch_to_device(collator(batch_samples), torch_device)
            trace_dist = model.encode_traces(batch["traces"])
            source_masks = batch.get("source_activity_masks", {})
            decoded_rows, beam_candidate_rows = (
                decode_trace_candidates_with_conformance_reranking(
                    model,
                    trace_dist,
                    batch_samples,
                    max_length=max_decode_length,
                    allowed_activity_mask=source_masks.get("trace"),
                )
            )
            for offset, (sample, token_ids, beam_candidates) in enumerate(
                zip(batch_samples, decoded_rows, beam_candidate_rows)
            ):
                proc_rosetta_row = evaluate_proc_rosetta_discovery(
                    model,
                    sample,
                    token_ids,
                    conformance_method=conformance_method,
                )
                proc_rosetta_row["equivalence_id"] = str(
                    getattr(sample, "equivalence_id", start)
                )
                metadata = getattr(sample, "metadata", {})
                proc_rosetta_row["motif"] = str(metadata.get("motif", "unknown"))
                proc_rosetta_row["observation_quality"] = str(
                    metadata.get("observation_quality", "unknown")
                )
                candidate_quality = [proc_rosetta_row]
                selected_key = tuple(int(token_id) for token_id in token_ids)
                seen = {selected_key}
                for candidate_ids in beam_candidates:
                    candidate_key = tuple(int(token_id) for token_id in candidate_ids)
                    if candidate_key in seen:
                        continue
                    seen.add(candidate_key)
                    candidate_quality.append(
                        evaluate_proc_rosetta_discovery(
                            model,
                            sample,
                            candidate_ids,
                            conformance_method=conformance_method,
                        )
                    )
                oracle = max(
                    candidate_quality,
                    key=lambda value: (
                        float(value["f1"])
                        if isinstance(value.get("f1"), (int, float))
                        else -1.0
                    ),
                )
                reranked_f1 = proc_rosetta_row.get("f1")
                oracle_f1 = oracle.get("f1")
                proc_rosetta_row.update(
                    beam_candidate_count=len(seen),
                    beam_reranked_f1=reranked_f1,
                    beam_oracle_fitness=oracle.get("fitness"),
                    beam_oracle_precision=oracle.get("precision"),
                    beam_oracle_f1=oracle_f1,
                    beam_oracle_f1_gap=(
                        round_float(float(oracle_f1) - float(reranked_f1))
                        if isinstance(oracle_f1, (int, float))
                        and isinstance(reranked_f1, (int, float))
                        else None
                    ),
                )
                rows["proc_rosetta_trace_mu"].append(proc_rosetta_row)
                conformance_successes += int(proc_rosetta_row["conformance_evaluable"])
                completed_models += 1
                progress.update()
                set_progress_postfix(
                    progress,
                    method="ProcRosetta",
                    ok=conformance_successes,
                    remaining=total_models - completed_models,
                )

                cached_index = start + offset
                inductive_miner_row = (
                    dict(inductive_rows[cached_index])
                    if inductive_rows is not None
                    else evaluate_inductive_miner_discovery(
                        sample,
                        conformance_method=conformance_method,
                    )
                )
                inductive_miner_row["equivalence_id"] = str(
                    getattr(sample, "equivalence_id", start)
                )
                rows["inductive_miner"].append(inductive_miner_row)
                conformance_successes += int(inductive_miner_row["conformance_evaluable"])
                completed_models += 1
                progress.update()
                set_progress_postfix(
                    progress,
                    method="Inductive Miner",
                    ok=conformance_successes,
                    remaining=total_models - completed_models,
                )

    if conformance_method == "footprints":
        description = (
            "Footprint-based discovery quality on each test log. ProcRosetta uses "
            "the trace encoder and grammar-masked process-tree decoder; the baseline "
            "uses PM4Py Inductive Miner. Footprints are computed directly on each log "
            "and discovered process tree (not on a Petri net), then scored by footprint "
            "fitness, footprint precision, and their harmonic-mean F1."
        )
    else:
        description = (
            "Token-based-replay discovery quality on each test log. ProcRosetta uses "
            "the trace encoder and grammar-masked process-tree decoder; the baseline "
            "uses PM4Py Inductive Miner. Each discovered process tree is converted to "
            "a Petri net and scored by token-based replay fitness, token-based replay "
            "precision, and their harmonic-mean F1."
        )
    return {
        "description": description,
        "conformance_method": conformance_method,
        "max_decode_length": int(max_decode_length),
        "methods": {
            name: summarize_discovery_quality(values, conformance_method=conformance_method)
            for name, values in rows.items()
        },
    }


def evaluate_proc_rosetta_discovery(
    model,
    sample: ProcessSample,
    token_ids: Sequence[int],
    conformance_method: str = "token_based_replay",
) -> dict[str, object]:
    validate_conformance_method(conformance_method)
    row = discovery_quality_row()
    try:
        if model.tree_tokenizer.eos_id not in [int(token_id) for token_id in token_ids]:
            raise ValueError("decoder did not emit <eos>")
        tree = model.tree_tokenizer.decode_tree(token_ids)
    except Exception as exc:
        row["error"] = f"decode:{type(exc).__name__}: {exc}"
        return row

    if conformance_method == "footprints":
        try:
            pm4py_tree = to_pm4py_tree(tree)
        except Exception as exc:
            row["error"] = f"process_tree:{type(exc).__name__}: {exc}"
            return row
        row["model_discovered"] = True
        return score_discovered_process_tree(sample, pm4py_tree, row=row)

    try:
        bundle = tree_to_petri_net(tree)
    except Exception as exc:
        row["error"] = f"petri:{type(exc).__name__}: {exc}"
        return row
    row["model_discovered"] = True
    return score_discovered_petri_net(
        sample,
        bundle.net,
        bundle.initial_marking,
        bundle.final_marking,
        row=row,
    )


def evaluate_inductive_miner_discovery(
    sample: ProcessSample,
    conformance_method: str = "token_based_replay",
) -> dict[str, object]:
    validate_conformance_method(conformance_method)
    row = discovery_quality_row()
    try:
        import pm4py

        log = traces_to_event_dataframe(sample.traces)
        tree = pm4py.discover_process_tree_inductive(
            log,
            activity_key="concept:name",
            timestamp_key="time:timestamp",
            case_id_key="case:concept:name",
        )
    except Exception as exc:
        row["error"] = f"discover:{type(exc).__name__}: {exc}"
        return row

    if conformance_method == "footprints":
        row["model_discovered"] = True
        return score_discovered_process_tree(sample, tree, row=row)

    try:
        net, initial_marking, final_marking = pm4py.convert_to_petri_net(tree)
    except Exception as exc:
        row["error"] = f"petri:{type(exc).__name__}: {exc}"
        return row
    row["model_discovered"] = True
    return score_discovered_petri_net(
        sample,
        net,
        initial_marking,
        final_marking,
        row=row,
    )


def discovery_quality_row() -> dict[str, object]:
    return {
        "model_discovered": False,
        "conformance_evaluable": False,
        "fitness": None,
        "precision": None,
        "f1": None,
        "error": None,
    }


def score_discovered_petri_net(
    sample: ProcessSample,
    net,
    initial_marking,
    final_marking,
    row: dict[str, object] | None = None,
) -> dict[str, object]:
    row = row or discovery_quality_row()
    try:
        fitness, precision = token_based_replay_fitness_precision(
            sample.traces,
            net,
            initial_marking,
            final_marking,
        )
        row["fitness"] = round_float(fitness)
        row["precision"] = round_float(precision)
        row["f1"] = fitness_precision_f1_score(fitness, precision)
        row["conformance_evaluable"] = True
    except Exception as exc:
        row["error"] = f"token_replay:{type(exc).__name__}: {exc}"
    return row


def score_discovered_process_tree(
    sample: ProcessSample,
    tree,
    row: dict[str, object] | None = None,
) -> dict[str, object]:
    row = row or discovery_quality_row()
    try:
        fitness, precision = footprint_fitness_precision(sample.traces, tree)
        row["fitness"] = round_float(fitness)
        row["precision"] = round_float(precision)
        row["f1"] = fitness_precision_f1_score(fitness, precision)
        row["conformance_evaluable"] = True
    except Exception as exc:
        row["error"] = f"footprints:{type(exc).__name__}: {exc}"
    return row


def token_based_replay_fitness_precision(
    traces: Sequence[Trace],
    net,
    initial_marking,
    final_marking,
) -> tuple[float, float]:
    import pm4py

    log = traces_to_event_dataframe(traces)
    fitness = extract_token_based_replay_fitness(
        pm4py.fitness_token_based_replay(
            log,
            net,
            initial_marking,
            final_marking,
            activity_key="concept:name",
            timestamp_key="time:timestamp",
            case_id_key="case:concept:name",
        )
    )
    precision = float(
        pm4py.precision_token_based_replay(
            log,
            net,
            initial_marking,
            final_marking,
            activity_key="concept:name",
            timestamp_key="time:timestamp",
            case_id_key="case:concept:name",
        )
    )
    return fitness, precision


def footprint_fitness_precision(
    traces: Sequence[Trace],
    process_tree,
) -> tuple[float, float]:
    import pm4py
    from pm4py.algo.conformance.footprints import algorithm as footprint_conformance
    from pm4py.algo.conformance.footprints.util import evaluation as footprint_evaluation

    log = traces_to_event_dataframe(traces)
    log_footprints = pm4py.discover_footprints(log)
    tree_footprints = pm4py.discover_footprints(process_tree)
    conformance = footprint_conformance.apply(
        log_footprints,
        tree_footprints,
        variant=footprint_conformance.Variants.LOG_EXTENSIVE,
    )
    fitness = float(
        footprint_evaluation.fp_fitness(log_footprints, tree_footprints, conformance)
    )
    precision = float(footprint_evaluation.fp_precision(log_footprints, tree_footprints))
    return fitness, precision


def alignment_fitness_precision(
    traces: Sequence[Trace],
    net,
    initial_marking,
    final_marking,
) -> tuple[float, float]:
    import pm4py

    log = traces_to_event_dataframe(traces)
    fitness = extract_alignment_fitness(
        pm4py.fitness_alignments(
            log,
            net,
            initial_marking,
            final_marking,
            multi_processing=False,
            activity_key="concept:name",
            timestamp_key="time:timestamp",
            case_id_key="case:concept:name",
        )
    )
    precision = float(
        pm4py.precision_alignments(
            log,
            net,
            initial_marking,
            final_marking,
            multi_processing=False,
            activity_key="concept:name",
            timestamp_key="time:timestamp",
            case_id_key="case:concept:name",
        )
    )
    return fitness, precision


def traces_to_event_dataframe(traces: Sequence[Trace]) -> pd.DataFrame:
    rows = []
    for case_idx, trace in enumerate(traces):
        for event_idx, activity in enumerate(trace):
            rows.append(
                {
                    "case:concept:name": f"case-{case_idx}",
                    "concept:name": str(activity),
                    "time:timestamp": pd.Timestamp("2024-01-01")
                    + pd.Timedelta(seconds=case_idx * 1000 + event_idx),
                }
            )
    if not rows:
        raise ValueError("conformance evaluation requires at least one event")
    return pd.DataFrame(rows)


def extract_token_based_replay_fitness(value: object) -> float:
    if isinstance(value, dict):
        for key in ("log_fitness", "average_trace_fitness"):
            if key in value:
                return float(value[key])
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"unsupported token-based replay fitness result: {value!r}")


def extract_alignment_fitness(value: object) -> float:
    if isinstance(value, dict):
        for key in ("log_fitness", "averageFitness", "average_trace_fitness"):
            if key in value:
                return float(value[key])
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"unsupported alignment fitness result: {value!r}")


def fitness_precision_f1_score(fitness: float, precision: float) -> float:
    denominator = fitness + precision
    if denominator <= 0.0:
        return 0.0
    return round_float(2.0 * fitness * precision / denominator)


def alignment_f1_score(fitness: float, precision: float) -> float:
    """Backward-compatible name for the generic fitness/precision F1 calculation."""

    return fitness_precision_f1_score(fitness, precision)


def summarize_discovery_quality(
    rows: Sequence[dict[str, object]],
    conformance_method: str = "token_based_replay",
) -> dict[str, object]:
    validate_conformance_method(conformance_method)
    fitness_values = numeric_row_values(rows, "fitness")
    precision_values = numeric_row_values(rows, "precision")
    f1_values = numeric_row_values(rows, "f1")
    oracle_f1_values = numeric_row_values(rows, "beam_oracle_f1")
    oracle_f1_gaps = numeric_row_values(rows, "beam_oracle_f1_gap")
    first_error = next((str(row["error"]) for row in rows if row.get("error")), None)
    evaluable_count = sum(1 for row in rows if conformance_row_evaluable(row))
    evaluable_rate = round_float(evaluable_count / len(rows)) if rows else 0.0
    error_count = len(rows) - evaluable_count
    summary = {
        "count": int(len(rows)),
        "model_discovered_rate": rate(rows, "model_discovered"),
        "conformance_evaluable_rate": evaluable_rate,
        "mean_fitness": round_float(mean(fitness_values)),
        "mean_precision": round_float(mean(precision_values)),
        "mean_f1": round_float(mean(f1_values)),
        "median_f1": round_float(float(np.median(f1_values))) if f1_values else 0.0,
        "mean_beam_oracle_f1": round_float(mean(oracle_f1_values)),
        "mean_beam_oracle_f1_gap": round_float(mean(oracle_f1_gaps)),
        "conformance_error_count": error_count,
        "first_error": first_error,
    }
    if conformance_method == "token_based_replay":
        summary["token_replay_evaluable_rate"] = evaluable_rate
        summary["token_replay_error_count"] = error_count
    else:
        summary["footprint_evaluable_rate"] = evaluable_rate
        summary["footprint_error_count"] = error_count
    summary["family_bootstrap_95_intervals"] = {
        "discovery_f1": family_bootstrap_interval(rows, "f1"),
    }
    return summary


def conformance_row_evaluable(row: dict[str, object]) -> bool:
    if "conformance_evaluable" in row:
        return bool(row["conformance_evaluable"])
    return bool(row.get("token_replay_evaluable", False))


def validate_conformance_method(conformance_method: str) -> None:
    if conformance_method not in CONFORMANCE_METHODS:
        raise ValueError(
            f"unknown conformance method {conformance_method!r}; "
            f"expected one of {', '.join(CONFORMANCE_METHODS)}"
        )


def conformance_method_label(conformance_method: str) -> str:
    validate_conformance_method(conformance_method)
    if conformance_method == "footprints":
        return "footprint-conformance"
    return "token-replay"


def numeric_row_values(rows: Sequence[dict[str, object]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]


@torch.no_grad()
def proc_rosetta_embeddings(
    model,
    samples: Sequence[ProcessSample],
    batch_size: int = 16,
    device: str | None = None,
    show_progress: bool = False,
) -> dict[str, np.ndarray]:
    model.eval()
    torch_device = resolve_device(device)
    model.to(torch_device)
    collator = ProcessBatchCollator(model.tree_tokenizer, model.activity_tokenizer)
    chunks: dict[str, list[np.ndarray]] = {"tree": [], "trace": [], "petri": []}
    starts = range(0, len(samples), batch_size)
    iterator = progress_iterator(
        starts,
        enabled=show_progress,
        total=len(starts),
        desc="Neural embeddings",
        unit="batch",
    )
    for start in iterator:
        batch_samples = samples[start : start + batch_size]
        batch = collator(batch_samples)
        batch = _move_batch_to_device(batch, torch_device)
        tree_dist = model.encode_tree(batch["tree_tokens"])
        trace_dist = model.encode_traces(batch["traces"])
        petri_dist = model.encode_petri(batch["petri"])
        chunks["tree"].append(tree_dist.mu.detach().cpu().numpy())
        chunks["trace"].append(trace_dist.mu.detach().cpu().numpy())
        chunks["petri"].append(petri_dist.mu.detach().cpu().numpy())

    embeddings = {
        "proc_rosetta_tree_mu": np.vstack(chunks["tree"]),
        "proc_rosetta_trace_mu": np.vstack(chunks["trace"]),
        "proc_rosetta_petri_mu": np.vstack(chunks["petri"]),
    }
    embeddings["proc_rosetta_fused_mu"] = np.mean(
        [
            embeddings["proc_rosetta_tree_mu"],
            embeddings["proc_rosetta_trace_mu"],
            embeddings["proc_rosetta_petri_mu"],
        ],
        axis=0,
    )
    return embeddings


@torch.no_grad()
def decode_quality_report(
    model,
    samples: Sequence[ProcessSample],
    batch_size: int = 16,
    device: str | None = None,
    max_decode_length: int = 512,
    behavior_traces_per_sample: int = 128,
    show_progress: bool = False,
    completion_policy: str = "bounded",
    beam_size: int = 5,
    include_beam_oracle: bool = True,
) -> dict[str, object]:
    model.eval()
    torch_device = resolve_device(device)
    model.to(torch_device)
    collator = ProcessBatchCollator(model.tree_tokenizer, model.activity_tokenizer)
    rows: dict[str, list[dict[str, object]]] = {
        "proc_rosetta_tree_mu": [],
        "proc_rosetta_trace_mu": [],
        "proc_rosetta_petri_mu": [],
        "proc_rosetta_fused_mu": [],
    }

    with progress_bar(
        total=4 * len(samples),
        enabled=show_progress,
        desc="Decode quality",
        unit="decode",
    ) as progress:
        successful_decodes = 0
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            batch = _move_batch_to_device(collator(batch_samples), torch_device)
            tree_dist = model.encode_tree(batch["tree_tokens"])
            trace_dist = model.encode_traces(batch["traces"])
            petri_dist = model.encode_petri(batch["petri"])
            latents = {
                "proc_rosetta_tree_mu": tree_dist,
                "proc_rosetta_trace_mu": trace_dist,
                "proc_rosetta_petri_mu": petri_dist,
                "proc_rosetta_fused_mu": torch.stack(
                    (tree_dist.mu, trace_dist.mu, petri_dist.mu), dim=0
                ).mean(dim=0),
            }
            source_masks = batch["source_activity_masks"]
            allowed_masks = {
                "proc_rosetta_tree_mu": source_masks["tree"],
                "proc_rosetta_trace_mu": source_masks["trace"],
                "proc_rosetta_petri_mu": source_masks["petri"],
                "proc_rosetta_fused_mu": (
                    source_masks["tree"]
                    | source_masks["trace"]
                    | source_masks["petri"]
                ),
            }
            source_names = {
                "proc_rosetta_tree_mu": "tree",
                "proc_rosetta_trace_mu": "trace",
                "proc_rosetta_petri_mu": "petri",
                "proc_rosetta_fused_mu": "fused",
            }
            duplicate_free = completion_policy != "prefix_only"

            for name, latent in latents.items():
                if include_beam_oracle and hasattr(
                    model.tree_decoder, "decode_beam_candidates"
                ):
                    candidate_rows = model.tree_decoder.decode_beam_candidates(
                        latent,
                        max_length=max_decode_length,
                        beam_size=beam_size,
                        length_penalty=0.7,
                        allowed_activity_mask=allowed_masks[name],
                        completion_policy=completion_policy,
                        avoid_duplicate_activity_labels=duplicate_free,
                    )
                else:
                    decoded = decode_with_beam(
                        model.tree_decoder,
                        latent,
                        max_length=max_decode_length,
                        beam_size=beam_size,
                        allowed_activity_mask=allowed_masks[name],
                        completion_policy=completion_policy,
                        avoid_duplicate_activity_labels=duplicate_free,
                    )
                    candidate_rows = [
                        [(token_ids, 0.0)]
                        for token_ids in decoded.detach().cpu().tolist()
                    ]
                for sample, candidates in zip(batch_samples, candidate_rows):
                    token_ids = candidates[0][0]
                    source_name = source_names[name]
                    target_tree = decode_target_tree(
                        sample,
                        source_name,
                        avoid_duplicates=duplicate_free,
                    )
                    row = evaluate_single_decode(
                        model,
                        sample,
                        token_ids,
                        source_name=source_name,
                        target_tree=target_tree,
                        behavior_traces_per_sample=behavior_traces_per_sample,
                        completion_policy=completion_policy,
                    )
                    candidate_evaluations = [row]
                    for candidate_ids, _ in candidates[1:]:
                        candidate_evaluations.append(
                            evaluate_single_decode(
                                model,
                                sample,
                                candidate_ids,
                                source_name=source_name,
                                target_tree=target_tree,
                                behavior_traces_per_sample=(
                                    behavior_traces_per_sample
                                ),
                                completion_policy=completion_policy,
                            )
                        )
                    behavior_values = [
                        float(value["behavior_l1"])
                        for value in candidate_evaluations
                        if isinstance(value.get("behavior_l1"), (int, float))
                    ]
                    row.update(
                        beam_any_exact=any(
                            bool(value["exact_tree_match"])
                            for value in candidate_evaluations
                        ),
                        beam_best_normalized_token_edit_distance=min(
                            float(value["normalized_token_edit_distance"])
                            for value in candidate_evaluations
                        ),
                        beam_best_behavior_l1=(
                            min(behavior_values) if behavior_values else None
                        ),
                    )
                    rows[name].append(row)
                    successful_decodes += int(row["behavior_evaluable"])
                    progress.update()
                    set_progress_postfix(
                        progress,
                        source=name.removeprefix("proc_rosetta_").removesuffix("_mu"),
                        ok=successful_decodes,
                    )

    return {
        "description": (
            "Length-normalized beam decodes from each ProcRosetta source into the grammar-masked "
            f"process-tree decoder under {completion_policy!r} completion. Petri validity is "
            "measured by converting the decoded process tree to a Petri net."
        ),
        "evaluation_scope": (
            "raw_semantic_decoding"
            if completion_policy == "prefix_only"
            else "deployment_duplicate_free_decoding"
        ),
        "max_decode_length": int(max_decode_length),
        "beam_size": int(beam_size),
        "beam_oracle_diagnostics": bool(include_beam_oracle),
        "completion_policy": completion_policy,
        "behavior_traces_per_sample": int(behavior_traces_per_sample),
        "methods": {name: summarize_decode_quality(values) for name, values in rows.items()},
    }


def decode_with_beam(
    decoder,
    source,
    *,
    max_length: int,
    beam_size: int = 5,
    allowed_activity_mask: torch.Tensor | None = None,
    completion_policy: str = "bounded",
    avoid_duplicate_activity_labels: bool = False,
):
    if hasattr(decoder, "decode_beam"):
        return decoder.decode_beam(
            source,
            max_length=max_length,
            beam_size=beam_size,
            length_penalty=0.7,
            allowed_activity_mask=allowed_activity_mask,
            completion_policy=completion_policy,
            avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
        )
    return decoder.decode_greedy(
        source.mu if hasattr(source, "mu") else source,
        max_length=max_length,
        apply_grammar_mask=True,
        allowed_activity_mask=allowed_activity_mask,
        completion_policy=completion_policy,
        avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
    )


def decode_target_tree(
    sample: ProcessSample,
    source_name: str,
    *,
    avoid_duplicates: bool,
) -> ProcessTreeNode:
    """Return the source- and decoding-policy-specific evaluation target."""

    if source_name not in {"tree", "trace", "petri", "fused"}:
        raise ValueError(f"unknown decode source {source_name!r}")
    if source_name == "fused":
        # Construct this explicitly from every fused source rather than relying
        # on the current fact that the tree alphabet subsumes the other two.
        union_alphabet = set(sample.tree.activity_labels())
        union_alphabet.update(label for trace in sample.traces for label in trace)
        union_alphabet.update(
            label
            for label in sample.petri_graph.transition_labels
            if label is not None
        )
        target = fold_process_tree(
            sanitize_activity_labels(
                sample.tree,
                allowed_labels=union_alphabet,
            ).tree
        )
    else:
        targets = sample.decoder_target_trees
        if set(targets) != {"tree", "trace", "petri"}:
            targets = decoder_target_trees_for_sample(
                sample.tree,
                sample.traces,
                sample.petri_graph,
            )
        target = targets[source_name]
    if avoid_duplicates:
        target = fold_process_tree(
            sanitize_activity_labels(target, avoid_duplicates=True).tree
        )
    return target


def decode_trace_candidates_with_conformance_reranking(
    model,
    source,
    samples: Sequence[ProcessSample],
    *,
    max_length: int,
    beam_size: int = 5,
    fitness_weight: float = 0.25,
    footprint_precision_weight: float = 0.25,
    allowed_activity_mask: torch.Tensor | None = None,
    completion_policy: str = "bounded",
) -> tuple[list[list[int]], list[list[list[int]]]]:
    decoder = model.tree_decoder
    if not hasattr(decoder, "decode_beam_candidates"):
        decoded = decode_with_beam(
            decoder,
            source,
            max_length=max_length,
            beam_size=beam_size,
            allowed_activity_mask=allowed_activity_mask,
            completion_policy=completion_policy,
            avoid_duplicate_activity_labels=True,
        )
        selected = decoded.detach().cpu().tolist()
        return selected, [[row] for row in selected]
    candidate_rows = decoder.decode_beam_candidates(
        source,
        max_length=max_length,
        beam_size=beam_size,
        length_penalty=0.7,
        allowed_activity_mask=allowed_activity_mask,
        completion_policy=completion_policy,
        avoid_duplicate_activity_labels=True,
    )
    selected: list[list[int]] = []
    all_candidates: list[list[list[int]]] = []
    for sample, candidates in zip(samples, candidate_rows):
        scored: list[tuple[float, list[int]]] = []
        for token_ids, log_probability in candidates:
            normalized_log_probability = log_probability / max(len(token_ids), 1) ** 0.7
            score = normalized_log_probability
            try:
                tree = model.tree_tokenizer.decode_tree(token_ids)
                simulated = simulate_traces(
                    tree,
                    num_traces=max(1, min(len(sample.traces), 32)),
                )
                distance = behavioral_distance(sample.traces, simulated)
                fitness = 1.0 - min(1.0, float(distance["mean_l1"]) / 2.0)
                footprint_precision = 1.0 - min(
                    1.0, float(distance["directly_follows_l1"]) / 2.0
                )
                score += fitness_weight * fitness
                score += footprint_precision_weight * footprint_precision
            except Exception:
                score -= 1.0
            scored.append((score, token_ids))
        selected.append(max(scored, key=lambda item: item[0])[1])
        all_candidates.append([token_ids for token_ids, _ in candidates])
    return selected, all_candidates


def decode_trace_with_conformance_reranking(
    model,
    source,
    samples: Sequence[ProcessSample],
    *,
    max_length: int,
    beam_size: int = 5,
    fitness_weight: float = 0.25,
    footprint_precision_weight: float = 0.25,
    allowed_activity_mask: torch.Tensor | None = None,
    completion_policy: str = "bounded",
) -> list[list[int]]:
    """Return reranked beam winners while retaining the historical public API."""

    selected, _ = decode_trace_candidates_with_conformance_reranking(
        model,
        source,
        samples,
        max_length=max_length,
        beam_size=beam_size,
        fitness_weight=fitness_weight,
        footprint_precision_weight=footprint_precision_weight,
        allowed_activity_mask=allowed_activity_mask,
        completion_policy=completion_policy,
    )
    return selected


def evaluate_single_decode(
    model,
    sample: ProcessSample,
    token_ids: Sequence[int],
    *,
    source_name: str,
    target_tree: ProcessTreeNode,
    behavior_traces_per_sample: int = 128,
    completion_policy: str = "bounded",
    include_original_tree_metrics: bool = False,
) -> dict[str, object]:
    tokenizer = model.tree_tokenizer
    decoded_tokens = trim_tree_token_sequence(token_ids, tokenizer)
    # Match the collator: targets keep the stored first-seen labels.
    target_tokens = tokenizer.encode_tree(target_tree, canonicalize=False)
    normalized_denominator = max(len(target_tokens), len(decoded_tokens), 1)
    token_edit_distance = levenshtein_distance(target_tokens, decoded_tokens)
    decoded_names = [
        tokenizer.tokens[int(token_id)]
        for token_id in decoded_tokens
        if 0 <= int(token_id) < tokenizer.vocab_size
    ]
    operator_count = sum(name in tokenizer.operator_tokens for name in decoded_names)
    leaf_count = sum(
        name == "TAU" or name in tokenizer.activity_tokens
        for name in decoded_names
    )
    unresolved_open_slots: int | None = None
    try:
        grammar_state, pending_operator, open_nodes = tokenizer._grammar_state(decoded_tokens)
        if grammar_state.value != "invalid":
            unresolved_open_slots = int(open_nodes)
            if pending_operator is not None:
                unresolved_open_slots += tokenizer.minimum_legal_arity(pending_operator)
    except Exception:
        pass
    row: dict[str, object] = {
        "source_name": source_name,
        "target_policy": (
            "semantic"
            if completion_policy == "prefix_only"
            else "deployment_duplicate_free"
        ),
        "equivalence_id": sample.equivalence_id,
        "motif": str(sample.metadata.get("motif", "ordinary_tree")),
        "contains_loop": tree_contains_loop(target_tree),
        "tree_size": target_tree.size(),
        "tree_depth": target_tree.max_depth(),
        "activity_count": len(target_tree.unique_activity_labels()),
        "observation_quality": str(
            sample.metadata.get(
                "observation_quality",
                sample.metadata.get("sampling_mode", "unknown"),
            )
        ),
        "terminated": tokenizer.eos_id in [int(token_id) for token_id in token_ids],
        "completion_policy": completion_policy,
        "generated_length": len(decoded_tokens),
        "operator_count": operator_count,
        "leaf_count": leaf_count,
        "operator_to_leaf_ratio": operator_count / max(leaf_count, 1),
        "unresolved_open_slots": unresolved_open_slots,
        "valid_tree": False,
        "hard_structural_success": False,
        "neural_decode_without_fallback": False,
        "exact_tree_match": False,
        "petri_convertible": False,
        "behavior_evaluable": False,
        "token_edit_distance": token_edit_distance,
        "normalized_token_edit_distance": token_edit_distance / normalized_denominator,
        "behavior_l1": None,
        "error": None,
    }

    if include_original_tree_metrics:
        original_tokens = tokenizer.encode_tree(sample.tree, canonicalize=False)
        original_edit = levenshtein_distance(original_tokens, decoded_tokens)
        row.update(
            original_tree_token_edit_distance=original_edit,
            original_tree_normalized_token_edit_distance=(
                original_edit / max(len(original_tokens), len(decoded_tokens), 1)
            ),
            original_tree_exact_match=False,
            original_tree_behavior_l1=None,
        )

    try:
        decoded_tree = (
            tokenizer.validate_complete_tree_sequence(
                token_ids,
                token_budget=len(token_ids),
            )
            if completion_policy == "bounded"
            else tokenizer.decode_tree(token_ids)
        )
        row["valid_tree"] = True
        row["hard_structural_success"] = True
        row["neural_decode_without_fallback"] = True
    except Exception as exc:
        row["error"] = f"decode:{type(exc).__name__}: {exc}"
        return row

    target_tree = tokenizer.decode_tree(target_tokens)
    row["exact_tree_match"] = decoded_tree.to_dict() == target_tree.to_dict()
    if include_original_tree_metrics:
        row["original_tree_exact_match"] = (
            decoded_tree.to_dict() == sample.tree.to_dict()
        )

    try:
        tree_to_petri_net(decoded_tree)
        row["petri_convertible"] = True
    except Exception as exc:
        row["error"] = f"petri:{type(exc).__name__}: {exc}"

    try:
        trace_count = max(1, min(len(sample.traces), behavior_traces_per_sample))
        target_traces = simulate_traces(target_tree, num_traces=trace_count)
        decoded_traces = simulate_traces(decoded_tree, num_traces=trace_count)
        row["behavior_l1"] = behavioral_distance(target_traces, decoded_traces)["mean_l1"]
        row["behavior_evaluable"] = True
        if include_original_tree_metrics:
            row["original_tree_behavior_l1"] = behavioral_distance(
                sample.traces,
                decoded_traces,
            )["mean_l1"]
    except Exception as exc:
        if row["error"] is None:
            row["error"] = f"behavior:{type(exc).__name__}: {exc}"

    return row


def summarize_decode_quality(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    count = len(rows)
    behavior_values = [
        float(row["behavior_l1"]) for row in rows if isinstance(row.get("behavior_l1"), (int, float))
    ]
    first_error = next((str(row["error"]) for row in rows if row.get("error")), None)
    summary = {
        "count": int(count),
        "terminated_rate": rate(rows, "terminated"),
        "valid_tree_rate": rate(rows, "valid_tree"),
        "hard_structural_success_rate": rate(rows, "hard_structural_success"),
        "neural_decode_without_fallback_rate": rate(
            rows, "neural_decode_without_fallback"
        ),
        "exact_tree_match_rate": rate(rows, "exact_tree_match"),
        "petri_conversion_rate": rate(rows, "petri_convertible"),
        "behavior_eval_success_rate": rate(rows, "behavior_evaluable"),
        "mean_token_edit_distance": round_float(mean(row["token_edit_distance"] for row in rows)),
        "mean_normalized_token_edit_distance": round_float(
            mean(row["normalized_token_edit_distance"] for row in rows)
        ),
        "mean_behavior_l1": round_float(mean(behavior_values)),
        "mean_generated_length": round_float(mean(row["generated_length"] for row in rows)),
        "mean_operator_to_leaf_ratio": round_float(
            mean(row["operator_to_leaf_ratio"] for row in rows)
        ),
        "mean_unresolved_open_slots": round_float(
            mean(
                row["unresolved_open_slots"]
                for row in rows
                if isinstance(row.get("unresolved_open_slots"), int)
            )
        ),
        "median_behavior_l1": round_float(float(np.median(behavior_values))) if behavior_values else 0.0,
        "invalid_decode_count": count_false(rows, "valid_tree"),
        "petri_conversion_error_count": count_false(rows, "petri_convertible"),
        "behavior_error_count": count_false(rows, "behavior_evaluable"),
        "first_error": first_error,
    }
    oracle_behavior = [
        float(row["beam_best_behavior_l1"])
        for row in rows
        if isinstance(row.get("beam_best_behavior_l1"), (int, float))
    ]
    summary.update(
        beam_oracle_exact_rate=rate(rows, "beam_any_exact"),
        beam_oracle_mean_normalized_token_edit_distance=round_float(
            mean(
                row.get(
                    "beam_best_normalized_token_edit_distance",
                    row["normalized_token_edit_distance"],
                )
                for row in rows
            )
        ),
        beam_oracle_mean_behavior_l1=round_float(mean(oracle_behavior)),
        beam_exact_ranking_gap=round_float(
            rate(rows, "beam_any_exact") - rate(rows, "exact_tree_match")
        ),
        beam_edit_ranking_gap=round_float(
            mean(row["normalized_token_edit_distance"] for row in rows)
            - mean(
                row.get(
                    "beam_best_normalized_token_edit_distance",
                    row["normalized_token_edit_distance"],
                )
                for row in rows
            )
        ),
    )
    summary["strata"] = decode_quality_strata(rows)
    summary["family_bootstrap_95_intervals"] = {
        "exact_tree_match_rate": family_bootstrap_interval(
            rows, "exact_tree_match", boolean=True
        ),
        "normalized_tree_edit_distance": family_bootstrap_interval(
            rows, "normalized_token_edit_distance"
        ),
    }
    return summary


def decode_quality_strata(
    rows: Sequence[dict[str, object]],
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        motif = str(row.get("motif", "unknown"))
        groups.setdefault(f"motif:{motif}", []).append(row)
        groups.setdefault(
            "loop:loop" if bool(row.get("contains_loop")) else "loop:non_loop", []
        ).append(row)
        size = int(row.get("tree_size", 0))
        size_bin = "1_10" if size <= 10 else "11_19" if size <= 19 else "20_plus"
        groups.setdefault(f"tree_size_bin:{size_bin}", []).append(row)
        depth = int(row.get("tree_depth", 0))
        depth_bin = "shallow_1_4" if depth <= 4 else "medium_5_7" if depth <= 7 else "deep_8_plus"
        groups.setdefault(f"tree_depth:{depth_bin}", []).append(row)
        activities = int(row.get("activity_count", 0))
        activity_bin = (
            "few_1_8"
            if activities <= 8
            else "medium_9_16"
            if activities <= 16
            else "many_17_plus"
        )
        groups.setdefault(f"activities:{activity_bin}", []).append(row)
        quality = str(row.get("observation_quality", "unknown"))
        groups.setdefault(f"observation:{quality}", []).append(row)
    return {
        name: {
            "count": len(selected),
            "canonical_exact_rate": round_float(
                sum(bool(row.get("exact_tree_match")) for row in selected)
                / max(len(selected), 1)
            ),
            "mean_normalized_tree_edit": round_float(
                mean(float(row["normalized_token_edit_distance"]) for row in selected)
            ),
            "mean_behavior_l1": round_float(
                mean(
                    row["behavior_l1"]
                    for row in selected
                    if isinstance(row.get("behavior_l1"), (int, float))
                )
            ),
            "behavior_eval_success_rate": round_float(
                sum(bool(row.get("behavior_evaluable")) for row in selected)
                / max(len(selected), 1)
            ),
        }
        for name, selected in sorted(groups.items())
    }


def tree_contains_loop(tree) -> bool:
    from proc_rosetta.tree import NodeKind

    return tree.kind is NodeKind.LOOP or any(tree_contains_loop(child) for child in tree.children)


def trim_tree_token_sequence(token_ids: Sequence[int], tokenizer) -> list[int]:
    trimmed = [int(token_id) for token_id in token_ids if int(token_id) != tokenizer.pad_id]
    if tokenizer.eos_id in trimmed:
        return trimmed[: trimmed.index(tokenizer.eos_id) + 1]
    return trimmed


def levenshtein_distance(left: Sequence[int], right: Sequence[int]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_idx, left_value in enumerate(left, start=1):
        current = [left_idx]
        for right_idx, right_value in enumerate(right, start=1):
            insert_cost = current[right_idx - 1] + 1
            delete_cost = previous[right_idx] + 1
            replace_cost = previous[right_idx - 1] + (left_value != right_value)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def family_bootstrap_interval(
    rows: Sequence[dict[str, object]],
    key: str,
    *,
    boolean: bool = False,
    replicates: int = 500,
    seed: int = 1729,
) -> dict[str, float | int]:
    """Bootstrap a row metric by resampling behavior families as clusters."""

    grouped: dict[str, list[float]] = {}
    for index, row in enumerate(rows):
        value = row.get(key)
        if boolean:
            numeric = float(bool(value))
        elif isinstance(value, (int, float)):
            numeric = float(value)
        else:
            continue
        grouped.setdefault(str(row.get("equivalence_id", index)), []).append(numeric)
    family_values = np.asarray(
        [float(np.mean(values)) for values in grouped.values()],
        dtype=float,
    )
    if family_values.size == 0:
        return {"family_count": 0, "estimate": 0.0, "lower": 0.0, "upper": 0.0}
    rng = np.random.default_rng(seed)
    draws = family_values[
        rng.integers(0, family_values.size, size=(max(1, replicates), family_values.size))
    ].mean(axis=1)
    return {
        "family_count": int(family_values.size),
        "estimate": round_float(float(family_values.mean())),
        "lower": round_float(float(np.quantile(draws, 0.025))),
        "upper": round_float(float(np.quantile(draws, 0.975))),
    }


def cross_modal_top1_rows(
    query: np.ndarray,
    candidates: np.ndarray,
    labels: Sequence[str | None],
    family_ids: Sequence[str],
) -> list[dict[str, object]]:
    query = np.asarray(query, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    candidates /= np.maximum(np.linalg.norm(candidates, axis=1, keepdims=True), 1e-12)
    similarities = query @ candidates.T
    rows: list[dict[str, object]] = []
    for index in range(len(query)):
        nearest = int(np.argmax(similarities[index]))
        label = labels[index]
        hit = nearest == index if label is None else labels[nearest] == label
        rows.append({"equivalence_id": family_ids[index], "hit": hit})
    return rows


def family_bootstrap_spearman_interval(
    embeddings: np.ndarray,
    behavior_distance: np.ndarray,
    family_ids: Sequence[str],
    *,
    replicates: int = 200,
    seed: int = 1733,
) -> dict[str, float | int]:
    groups: dict[str, list[int]] = {}
    for index, family_id in enumerate(family_ids):
        groups.setdefault(str(family_id), []).append(index)
    names = list(groups)
    if len(names) < 2:
        return {
            "family_count": len(names),
            "estimate": 0.0,
            "lower": 0.0,
            "upper": 0.0,
        }
    embedding_distance = cosine_distance_matrix(np.asarray(embeddings, dtype=float))
    behavior_distance = np.asarray(behavior_distance, dtype=float)
    estimate = spearman_upper(embedding_distance, behavior_distance)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(max(1, replicates)):
        selected_names = [names[index] for index in rng.integers(0, len(names), len(names))]
        indices = [index for name in selected_names for index in groups[name]]
        if len(indices) < 2:
            continue
        values.append(
            spearman_upper(
                embedding_distance[np.ix_(indices, indices)],
                behavior_distance[np.ix_(indices, indices)],
            )
        )
    return {
        "family_count": len(names),
        "estimate": round_float(float(estimate)),
        "lower": round_float(float(np.quantile(values, 0.025))) if values else 0.0,
        "upper": round_float(float(np.quantile(values, 0.975))) if values else 0.0,
    }


def rate(rows: Sequence[dict[str, object]], key: str) -> float:
    if not rows:
        return 0.0
    return round_float(sum(1 for row in rows if row.get(key)) / len(rows))


def count_false(rows: Sequence[dict[str, object]], key: str) -> int:
    return sum(1 for row in rows if not row.get(key))


def mean(values: Iterable[object]) -> float:
    numeric = [float(value) for value in values]
    if not numeric:
        return 0.0
    return float(np.mean(numeric))


def validation_split_hash(samples: Sequence[ProcessSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(
            json.dumps(
                sample.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def behavior_matrices(
    samples: Sequence[ProcessSample],
    show_progress: bool = False,
    *,
    cache_dir: str | Path | None = None,
) -> dict[str, np.ndarray]:
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"behavior-{validation_split_hash(samples)}.npz"
        if cache_path.exists():
            with np.load(cache_path) as cached:
                return {name: cached[name].copy() for name in cached.files}
    n = len(samples)
    matrices = {
        "mean_l1": np.zeros((n, n), dtype=float),
        "variant_l1": np.zeros((n, n), dtype=float),
        "directly_follows_l1": np.zeros((n, n), dtype=float),
        "length_l1": np.zeros((n, n), dtype=float),
    }
    with progress_bar(
        total=n * (n - 1) // 2,
        enabled=show_progress,
        desc="Behavior distances",
        unit="pair",
    ) as progress:
        for i in range(n):
            for j in range(i + 1, n):
                distance = behavioral_distance(samples[i].traces, samples[j].traces)
                for key, matrix in matrices.items():
                    matrix[i, j] = matrix[j, i] = float(distance[key])
                progress.update()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **matrices)
        temporary.replace(cache_path)
    return matrices


def deterministic_baseline_embeddings(
    samples: Sequence[ProcessSample],
    show_progress: bool = False,
) -> dict[str, np.ndarray]:
    extractors = {
        "trace_activity_counts": lambda sample: activity_count_features(sample.traces),
        "trace_variant_distribution": lambda sample: trace_variant_features(sample.traces),
        "trace_directly_follows": lambda sample: directly_follows_features(sample.traces),
        "trace_eventually_follows": lambda sample: eventually_follows_features(sample.traces),
        "pm4py_log_case_features_mean_std": lambda sample: pm4py_log_case_features(sample.traces),
        "petri_structural_counts": petri_structural_features,
    }
    baselines: dict[str, np.ndarray] = {}
    with progress_bar(
        total=len(extractors) * len(samples),
        enabled=show_progress,
        desc="Baseline features",
        unit="sample-method",
    ) as progress:
        for name, extractor in extractors.items():
            features = []
            for sample in samples:
                features.append(extractor(sample))
                progress.update()
            baselines[name] = vectorize_feature_dicts(features)
            set_progress_postfix(progress, method=name)
    return baselines


def evaluate_embedding_method(embeddings: np.ndarray, behavior_distance: np.ndarray) -> dict[str, object]:
    embeddings = np.asarray(embeddings, dtype=float)
    distance = cosine_distance_matrix(embeddings)
    return {
        "available": True,
        "vector_statistics": vector_statistics(embeddings),
        "pairwise_statistics": pairwise_statistics(distance),
        "behavior_alignment": {
            "spearman_embedding_distance_vs_behavior_l1": spearman_upper(distance, behavior_distance),
            "pearson_embedding_distance_vs_behavior_l1": pearson_upper(distance, behavior_distance),
        },
        "nearest_neighbor_behavior": nearest_neighbor_behavior(distance, behavior_distance),
        "neighbor_agreement": neighbor_agreement(distance, behavior_distance),
    }


def neighbor_agreement(
    embedding_distance: np.ndarray,
    behavior_distance: np.ndarray,
) -> dict[str, float | int]:
    count = int(embedding_distance.shape[0])
    if count < 2:
        return {"count": count, "top1_agreement": 0.0, "top3_agreement": 0.0}
    top1_hits = 0
    top3_hits = 0
    for index in range(count):
        behavior_row = behavior_distance[index].copy()
        embedding_row = embedding_distance[index].copy()
        behavior_row[index] = math.inf
        embedding_row[index] = math.inf
        behavior_neighbor = int(np.argmin(behavior_row))
        embedding_order = np.argsort(embedding_row)
        top1_hits += int(int(embedding_order[0]) == behavior_neighbor)
        top3_hits += int(behavior_neighbor in embedding_order[: min(3, count - 1)])
    return {
        "count": count,
        "top1_agreement": round_float(top1_hits / count),
        "top3_agreement": round_float(top3_hits / count),
    }


def pm4py_petri_method_report(
    samples: Sequence[ProcessSample],
    behavior_distance: np.ndarray,
    config: Pm4pyPetriEmbeddingConfig,
    show_progress: bool = False,
) -> tuple[np.ndarray | None, dict[str, object]]:
    try:
        from pm4py.objects.petri_net.utils import embeddings_similarity
    except Exception as exc:
        return (
            None,
            {
                "available": False,
                "kind": "pm4py_petri_net_embedding",
                "reason": f"pm4py embedding helper unavailable: {type(exc).__name__}: {exc}",
                "config": config.to_dict(),
            },
        )

    vectors: list[np.ndarray] = []
    try:
        iterator = progress_iterator(
            samples,
            enabled=show_progress,
            total=len(samples),
            desc="PM4Py Petri embeddings",
            unit="net",
        )
        for sample in iterator:
            bundle = petri_graph_to_net(
                sample.petri_graph,
                name=sample.model_variant_id or sample.equivalence_id,
            )
            vectors.append(
                embeddings_similarity.petri_net_embedding(
                    bundle.net,
                    dimensions=config.dimensions,
                    num_walks=config.num_walks,
                    walk_length=config.walk_length,
                    window=config.window,
                    epochs=config.epochs,
                    seed=config.seed,
                )
            )
    except Exception as exc:
        return (
            None,
            {
                "available": False,
                "kind": "pm4py_petri_net_embedding",
                "reason": f"{type(exc).__name__}: {exc}",
                "config": config.to_dict(),
            },
        )

    matrix = np.vstack(vectors)
    report = evaluate_embedding_method(matrix, behavior_distance)
    report["kind"] = "pm4py_petri_net_embedding"
    report["config"] = config.to_dict()
    return matrix, report


def set_progress_postfix(progress, **values: object) -> None:
    if hasattr(progress, "set_postfix"):
        progress.set_postfix(values, refresh=False)


def rank_embedding_methods(methods: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, method in methods.items():
        if not method.get("available", False):
            continue
        behavior_alignment = method["behavior_alignment"]
        nearest = method["nearest_neighbor_behavior"]
        assert isinstance(behavior_alignment, dict)
        assert isinstance(nearest, dict)
        rows.append(
            {
                "method": name,
                "behavior_spearman": behavior_alignment[
                    "spearman_embedding_distance_vs_behavior_l1"
                ],
                "nearest_neighbor_behavior_l1": nearest[
                    "mean_behavior_l1_at_nearest_neighbor"
                ],
                "improvement_over_random": nearest["improvement_over_random"],
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -float(row["behavior_spearman"]),
            float(row["nearest_neighbor_behavior_l1"]),
        ),
    )


def compare_methods_against_reference(
    embeddings: dict[str, np.ndarray],
    methods: dict[str, dict[str, object]],
    reference_name: str,
) -> dict[str, object]:
    if reference_name not in embeddings:
        return {"reference_method": reference_name, "available": False, "comparisons": {}}

    reference_distance = cosine_distance_matrix(embeddings[reference_name])
    reference_method = methods.get(reference_name, {})
    comparisons: dict[str, dict[str, object]] = {}
    for name, matrix in embeddings.items():
        if name == reference_name:
            continue
        method = methods.get(name, {})
        if not method.get("available", False):
            continue
        distance = cosine_distance_matrix(matrix)
        comparisons[name] = {
            "pairwise_distance_spearman_agreement": spearman_upper(reference_distance, distance),
            "pairwise_distance_pearson_agreement": pearson_upper(reference_distance, distance),
            "top1_neighbor_overlap": nearest_neighbor_overlap(reference_distance, distance, k=1),
            "top3_neighbor_overlap": nearest_neighbor_overlap(reference_distance, distance, k=3),
            "behavior_spearman_delta_vs_reference": metric_delta(
                method,
                reference_method,
                "behavior_alignment",
                "spearman_embedding_distance_vs_behavior_l1",
            ),
            "nearest_neighbor_behavior_l1_delta_vs_reference": metric_delta(
                method,
                reference_method,
                "nearest_neighbor_behavior",
                "mean_behavior_l1_at_nearest_neighbor",
            ),
        }
    return {
        "reference_method": reference_name,
        "available": True,
        "comparisons": comparisons,
    }


def metric_delta(
    method: dict[str, object],
    reference: dict[str, object],
    group: str,
    metric: str,
) -> float:
    left_group = method.get(group, {})
    right_group = reference.get(group, {})
    if not isinstance(left_group, dict) or not isinstance(right_group, dict):
        return 0.0
    return round_float(float(left_group.get(metric, 0.0)) - float(right_group.get(metric, 0.0)))


def nearest_neighbor_overlap(left_distance: np.ndarray, right_distance: np.ndarray, k: int = 1) -> float:
    n = left_distance.shape[0]
    if n < 2:
        return 0.0
    k = max(1, min(k, n - 1))
    overlaps = []
    for idx in range(n):
        left_row = left_distance[idx].copy()
        right_row = right_distance[idx].copy()
        left_row[idx] = math.inf
        right_row[idx] = math.inf
        left_neighbors = set(np.argsort(left_row)[:k].tolist())
        right_neighbors = set(np.argsort(right_row)[:k].tolist())
        overlaps.append(len(left_neighbors & right_neighbors) / k)
    return round_float(float(np.mean(overlaps)))


def format_human_test_report(report: dict[str, object]) -> str:
    lines: list[str] = []
    discovery_f1_method = "token-based replay"
    lines.append("ProcRosetta Test Report")
    lines.append("=======================")
    lines.append("")
    lines.append(
        f"Split: {report['split']}  |  curriculum: {report.get('curriculum', 'legacy')}"
        f"  |  samples: {report['sample_count']}"
    )
    lines.append("")

    loss_metrics = report["loss_metrics"]
    assert isinstance(loss_metrics, dict)
    lines.append("Neural test losses")
    lines.append("------------------")
    lines.append(
        "loss={loss:.4f}  tree={tree_reconstruction:.4f}  trace->tree={trace_to_tree:.4f}  "
        "petri->tree={petri_to_tree:.4f}  contrastive={contrastive:.4f}  kl={kl:.4f}".format(
            **loss_metrics
        )
    )
    lines.append("")

    decode_quality = report.get("deployment_decode_quality", {})
    if isinstance(decode_quality, dict):
        decode_methods = decode_quality.get("methods", {})
        if isinstance(decode_methods, dict):
            lines.append("Decode quality (deployment)")
            lines.append("---------------------------")
            lines.append(
                "Greedy decodes use the process-tree decoder; Petri ok means the decoded tree "
                "converted to a Petri net."
            )
            lines.append(
                format_table(
                    [
                        "source latent",
                        "ended",
                        "valid tree",
                        "exact tree",
                        "Petri ok",
                        "behavior L1",
                        "norm edit",
                    ],
                    decode_quality_rows(decode_methods),
                )
            )
            lines.append("")

    discovery_quality = report.get("discovery_quality", {})
    if isinstance(discovery_quality, dict):
        discovery_methods = discovery_quality.get("methods", {})
        if isinstance(discovery_methods, dict):
            conformance_method = discovery_quality.get(
                "conformance_method", "token_based_replay"
            )
            lines.append("Process discovery quality")
            lines.append("-------------------------")
            if conformance_method == "footprints":
                discovery_f1_method = "footprint"
                lines.append(
                    "Footprint conformance compares each test log with footprints computed "
                    "directly on the discovered process trees (not Petri nets)."
                )
                conformance_ok_label = "footprint ok"
            else:
                lines.append(
                    "Token-based replay compares the trace-decoded ProcRosetta model "
                    "with PM4Py Inductive Miner on each test log."
                )
                conformance_ok_label = "replay ok"
            lines.append(
                format_table(
                    [
                        "method",
                        "model ok",
                        conformance_ok_label,
                        "fitness",
                        "precision",
                        "F1",
                    ],
                    discovery_quality_rows(discovery_methods),
                )
            )
            lines.append("")

    behavior = report["behavioral_distance_summary"]
    assert isinstance(behavior, dict)
    lines.append("Behavioral distance scale")
    lines.append("-------------------------")
    lines.append(
        "Mean pairwise behavior L1 among test logs is {mean:.4f} "
        "(min={min:.4f}, max={max:.4f}, pairs={pair_count}).".format(**behavior)
    )
    lines.append(
        "For nearest-neighbor behavior L1 below, lower is better; improvement over random is higher better."
    )
    lines.append("")

    ranking = report.get("method_ranking", [])
    assert isinstance(ranking, list)
    methods = report["embedding_methods"]
    assert isinstance(methods, dict)
    lines.append("Embedding quality ranking")
    lines.append("-------------------------")
    lines.append(
        format_table(
            ["method", "behavior rho", "NN behavior", "impr. vs random", "dim"],
            [
                [
                    human_method_name(row["method"]),
                    format_metric(row["behavior_spearman"]),
                    format_metric(row["nearest_neighbor_behavior_l1"]),
                    format_metric(row["improvement_over_random"]),
                    str(method_dimension(methods, str(row["method"]))),
                ]
                for row in ranking
            ],
        )
    )
    lines.append("")

    comparisons = report.get("method_comparisons_against_proc_rosetta_fused_mu", {})
    assert isinstance(comparisons, dict)
    lines.append("Agreement against ProcRosetta fused encoding")
    lines.append("-------------------------------------------")
    lines.append(
        "Pairwise agreement compares the whole geometry of each method against our fused latent "
        "(1.0 means the methods order all test-pair distances the same way)."
    )
    comparison_rows = method_comparison_rows(comparisons)
    if comparison_rows:
        lines.append(
            format_table(
                [
                    "method",
                    "pairwise rho",
                    "top1 NN overlap",
                    "behavior rho delta",
                    "NN behavior delta",
                ],
                comparison_rows,
            )
        )
    else:
        lines.append("No comparable embedding methods were available.")
    pm4py_summary = pm4py_vs_ours_sentence(comparisons)
    if pm4py_summary:
        lines.append("")
        lines.append(pm4py_summary)
    elif "pm4py_colonna_petri_node2vec" in methods:
        pm4py_method = methods["pm4py_colonna_petri_node2vec"]
        if isinstance(pm4py_method, dict) and not pm4py_method.get("available", False):
            lines.append("")
            lines.append(
                "pm4py Petri Node2Vec vs ProcRosetta fused: unavailable ("
                f"{pm4py_method.get('reason', 'dependency or runtime error')})."
            )
    lines.append("")

    family_report = report.get("equivalence_families", {})
    if isinstance(family_report, dict):
        family_methods = family_report.get("methods", {})
        if isinstance(family_methods, dict):
            lines.append("Behavior-family equivalence")
            lines.append("---------------------------")
            lines.append(
                f"paired behaviors: {family_report.get('paired_behavior_count', 0)}; "
                "within-family cosine and retrieval use alternate exact-equivalent representations."
            )
            lines.append(
                format_table(
                    ["embedding", "within cosine", "between cosine", "margin", "family top1"],
                    [
                        [
                            human_method_name(name),
                            format_metric(values.get("within_family_cosine")),
                            format_metric(values.get("between_family_cosine")),
                            format_metric(values.get("equivalence_margin")),
                            format_metric(values.get("behavior_id_retrieval_top1")),
                        ]
                        for name, values in sorted(family_methods.items())
                        if isinstance(values, dict)
                    ],
                )
            )
            lines.append("")

    retrieval = report["cross_modal_retrieval"]
    assert isinstance(retrieval, dict)
    lines.append("ProcRosetta cross-modal retrieval")
    lines.append("---------------------------------")
    lines.append(
        format_table(
            ["query -> target", "top1", "MRR", "mean rank"],
            [
                [
                    name.replace("_to_", " -> "),
                    format_metric(values["top1_accuracy"]),
                    format_metric(values["mrr"]),
                    format_metric(values["mean_rank"]),
                ]
                for name, values in sorted(retrieval.items())
                if isinstance(values, dict)
            ],
        )
    )
    lines.append("")

    unavailable = [
        (name, method.get("reason", "not available"))
        for name, method in sorted(methods.items())
        if isinstance(method, dict) and not method.get("available", False)
    ]
    if unavailable:
        lines.append("Unavailable methods")
        lines.append("-------------------")
        for name, reason in unavailable:
            lines.append(f"- {human_method_name(name)}: {reason}")
        lines.append("")

    lines.append("Legend")
    lines.append("------")
    lines.append("- behavior rho: Spearman correlation between embedding distance and behavior distance.")
    lines.append("- NN behavior: behavior L1 distance to the nearest neighbor under that embedding.")
    lines.append("- NN behavior delta: method NN behavior minus ProcRosetta fused NN behavior; negative is better.")
    lines.append("- decode behavior L1: original traces vs traces simulated from the decoded tree; lower is better.")
    lines.append(
        f"- discovery F1: harmonic mean of {discovery_f1_method} fitness and precision; "
        "higher is better."
    )
    return "\n".join(lines)


def decode_quality_rows(methods: dict[str, object]) -> list[list[str]]:
    rows: list[list[str]] = []
    order = [
        "proc_rosetta_tree_mu",
        "proc_rosetta_trace_mu",
        "proc_rosetta_petri_mu",
        "proc_rosetta_fused_mu",
    ]
    for name in order:
        method = methods.get(name)
        if not isinstance(method, dict):
            continue
        rows.append(
            [
                human_method_name(name),
                format_rate(method.get("terminated_rate", 0.0)),
                format_rate(method.get("valid_tree_rate", 0.0)),
                format_rate(method.get("exact_tree_match_rate", 0.0)),
                format_rate(method.get("petri_conversion_rate", 0.0)),
                format_metric(method.get("mean_behavior_l1", 0.0)),
                format_metric(method.get("mean_normalized_token_edit_distance", 0.0)),
            ]
        )
    return rows


def discovery_quality_rows(methods: dict[str, object]) -> list[list[str]]:
    rows: list[list[str]] = []
    for name in ("proc_rosetta_trace_mu", "inductive_miner"):
        method = methods.get(name)
        if not isinstance(method, dict):
            continue
        rows.append(
            [
                human_method_name(name),
                format_rate(method.get("model_discovered_rate", 0.0)),
                format_rate(
                    method.get(
                        "conformance_evaluable_rate",
                        method.get(
                            "token_replay_evaluable_rate",
                            method.get("footprint_evaluable_rate", 0.0),
                        ),
                    )
                ),
                format_metric(method.get("mean_fitness", 0.0)),
                format_metric(method.get("mean_precision", 0.0)),
                format_metric(method.get("mean_f1", 0.0)),
            ]
        )
    return rows


def method_comparison_rows(comparisons: dict[str, object]) -> list[list[str]]:
    rows: list[list[str]] = []
    raw = comparisons.get("comparisons", {})
    if not isinstance(raw, dict):
        return rows
    ordered = sorted(
        raw.items(),
        key=lambda item: (
            item[0] != "pm4py_colonna_petri_node2vec",
            -float(item[1].get("pairwise_distance_spearman_agreement", 0.0)),
        ),
    )
    for name, comparison in ordered:
        if not isinstance(comparison, dict):
            continue
        rows.append(
            [
                human_method_name(name),
                format_metric(comparison.get("pairwise_distance_spearman_agreement", 0.0)),
                format_metric(comparison.get("top1_neighbor_overlap", 0.0)),
                signed_metric(comparison.get("behavior_spearman_delta_vs_reference", 0.0)),
                signed_metric(
                    comparison.get("nearest_neighbor_behavior_l1_delta_vs_reference", 0.0)
                ),
            ]
        )
    return rows


def pm4py_vs_ours_sentence(comparisons: dict[str, object]) -> str | None:
    raw = comparisons.get("comparisons", {})
    if not isinstance(raw, dict):
        return None
    comparison = raw.get("pm4py_colonna_petri_node2vec")
    if not isinstance(comparison, dict):
        return None
    return (
        "pm4py Petri Node2Vec vs ProcRosetta fused: pairwise geometry rho={rho}, "
        "top1 nearest-neighbor overlap={overlap}, behavior-rho delta={rho_delta}, "
        "NN-behavior delta={nn_delta}.".format(
            rho=format_metric(comparison.get("pairwise_distance_spearman_agreement", 0.0)),
            overlap=format_metric(comparison.get("top1_neighbor_overlap", 0.0)),
            rho_delta=signed_metric(comparison.get("behavior_spearman_delta_vs_reference", 0.0)),
            nn_delta=signed_metric(
                comparison.get("nearest_neighbor_behavior_l1_delta_vs_reference", 0.0)
            ),
        )
    )


def method_dimension(methods: dict[str, object], name: str) -> int:
    method = methods.get(name, {})
    if not isinstance(method, dict):
        return 0
    stats = method.get("vector_statistics", {})
    if not isinstance(stats, dict):
        return 0
    return int(stats.get("dimension", 0))


def human_method_name(name: object) -> str:
    labels = {
        "proc_rosetta_fused_mu": "ProcRosetta fused",
        "proc_rosetta_tree_mu": "ProcRosetta tree",
        "proc_rosetta_trace_mu": "ProcRosetta trace",
        "proc_rosetta_petri_mu": "ProcRosetta Petri",
        "pm4py_colonna_petri_node2vec": "pm4py Petri Node2Vec",
        "pm4py_log_case_features_mean_std": "pm4py log features",
        "trace_activity_counts": "trace activity counts",
        "trace_variant_distribution": "trace variants",
        "trace_directly_follows": "directly-follows",
        "trace_eventually_follows": "eventually-follows",
        "petri_structural_counts": "Petri structural counts",
        "inductive_miner": "Inductive Miner",
    }
    return labels.get(str(name), str(name))


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "(no rows)"
    widths = [
        max(len(str(header)), *(len(str(row[idx])) for row in rows))
        for idx, header in enumerate(headers)
    ]
    header_line = "  ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers))
    rule = "  ".join("-" * width for width in widths)
    row_lines = [
        "  ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row))
        for row in rows
    ]
    return "\n".join([header_line, rule, *row_lines])


def format_metric(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def signed_metric(value: object) -> str:
    try:
        return f"{float(value):+.3f}"
    except (TypeError, ValueError):
        return str(value)


def format_rate(value: object) -> str:
    try:
        return f"{100.0 * float(value):.1f}%"
    except (TypeError, ValueError):
        return str(value)


def cross_modal_retrieval(
    embeddings: dict[str, np.ndarray],
    exact_behavior_ids: Sequence[str | None] | None = None,
    partial_order_ids: Sequence[str | None] | None = None,
    behavior_signatures: Sequence[Sequence[float]] | None = None,
) -> dict[str, dict[str, float | int]]:
    pairs = {
        "tree_to_trace": ("proc_rosetta_tree_mu", "proc_rosetta_trace_mu"),
        "trace_to_tree": ("proc_rosetta_trace_mu", "proc_rosetta_tree_mu"),
        "tree_to_petri": ("proc_rosetta_tree_mu", "proc_rosetta_petri_mu"),
        "petri_to_tree": ("proc_rosetta_petri_mu", "proc_rosetta_tree_mu"),
        "trace_to_petri": ("proc_rosetta_trace_mu", "proc_rosetta_petri_mu"),
        "petri_to_trace": ("proc_rosetta_petri_mu", "proc_rosetta_trace_mu"),
    }
    report: dict[str, dict[str, float | int]] = {}
    for name, (left, right) in pairs.items():
        metrics = retrieval_metrics(
            embeddings[left],
            embeddings[right],
            query_labels=exact_behavior_ids,
            candidate_labels=exact_behavior_ids,
        )
        if partial_order_ids is not None:
            partial = retrieval_metrics(
                embeddings[left],
                embeddings[right],
                query_labels=partial_order_ids,
                candidate_labels=partial_order_ids,
            )
            metrics.update(
                {
                    "partial_order_recall_at_1": partial["top1_accuracy"],
                    "partial_order_recall_at_5": partial["recall_at_5"],
                }
            )
        if behavior_signatures is not None and behavior_signatures:
            signature_matrix = np.asarray(behavior_signatures, dtype=float)
            if signature_matrix.ndim == 2 and signature_matrix.shape[1] > 0:
                metrics["analogy_neighborhood_spearman"] = round_float(
                    neighborhood_distance_correlation(
                        embeddings[left], signature_matrix
                    )
                )
        report[name] = metrics
    return report


def retrieval_metrics(
    query: np.ndarray,
    candidates: np.ndarray,
    query_labels: Sequence[str | None] | None = None,
    candidate_labels: Sequence[str | None] | None = None,
) -> dict[str, float | int]:
    similarity = cosine_similarity_matrix(query, candidates)
    ranks = []
    recall_at_5 = []
    for idx in range(similarity.shape[0]):
        order = np.argsort(-similarity[idx])
        if query_labels is None or candidate_labels is None:
            relevant = np.asarray([idx], dtype=int)
        else:
            label = query_labels[idx]
            if label is None:
                continue
            relevant = np.asarray(
                [
                    candidate_index
                    for candidate_index, candidate_label in enumerate(candidate_labels)
                    if candidate_label == label
                ],
                dtype=int,
            )
            if not len(relevant):
                continue
        positions = [int(np.where(order == candidate)[0][0]) + 1 for candidate in relevant]
        rank = min(positions)
        ranks.append(rank)
        recall_at_5.append(float(rank <= 5))
    ranks_array = np.asarray(ranks, dtype=float)
    return {
        "count": int(len(ranks)),
        "top1_accuracy": round_float(float(np.mean(ranks_array == 1.0))) if len(ranks) else 0.0,
        "mean_rank": round_float(float(np.mean(ranks_array))) if len(ranks) else 0.0,
        "mrr": round_float(float(np.mean(1.0 / ranks_array))) if len(ranks) else 0.0,
        "recall_at_5": round_float(float(np.mean(recall_at_5))) if recall_at_5 else 0.0,
    }


def neighborhood_distance_correlation(
    embeddings: np.ndarray,
    behavior_signatures: np.ndarray,
) -> float:
    if len(embeddings) < 3:
        return 0.0
    latent_distance = 1.0 - cosine_similarity_matrix(embeddings, embeddings)
    behavior_distance = 1.0 - cosine_similarity_matrix(
        behavior_signatures, behavior_signatures
    )
    upper = np.triu_indices(len(embeddings), k=1)
    latent = latent_distance[upper]
    behavior = behavior_distance[upper]
    latent_rank = np.argsort(np.argsort(latent)).astype(float)
    behavior_rank = np.argsort(np.argsort(behavior)).astype(float)
    if np.std(latent_rank) == 0 or np.std(behavior_rank) == 0:
        return 0.0
    return float(np.corrcoef(latent_rank, behavior_rank)[0, 1])


def activity_count_features(traces: Iterable[Trace]) -> FeatureDict:
    counts: Counter[Hashable] = Counter()
    total = 0
    for trace in traces:
        counts.update(("activity", event) for event in trace)
        total += len(trace)
    return normalize_counts(counts, total)


def trace_variant_features(traces: Iterable[Trace]) -> FeatureDict:
    traces = list(traces)
    return normalize_counts(Counter(("variant", tuple(trace)) for trace in traces), len(traces))


def directly_follows_features(traces: Iterable[Trace]) -> FeatureDict:
    counts: Counter[Hashable] = Counter()
    total = 0
    for trace in traces:
        events = ["<start>", *trace, "<end>"]
        pairs = list(zip(events, events[1:]))
        counts.update(("dfg", left, right) for left, right in pairs)
        total += len(pairs)
    return normalize_counts(counts, total)


def eventually_follows_features(traces: Iterable[Trace]) -> FeatureDict:
    counts: Counter[Hashable] = Counter()
    total = 0
    for trace in traces:
        events = list(trace)
        for left_idx, left in enumerate(events):
            for right in events[left_idx + 1 :]:
                counts[("efg", left, right)] += 1
                total += 1
    return normalize_counts(counts, total)


def pm4py_log_case_features(traces: Iterable[Trace]) -> FeatureDict:
    import pm4py

    rows = []
    for case_idx, trace in enumerate(traces):
        for event_idx, activity in enumerate(trace):
            rows.append(
                {
                    "case:concept:name": f"case-{case_idx}",
                    "concept:name": activity,
                    "time:timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=event_idx),
                }
            )
    if not rows:
        return {}
    dataframe = pd.DataFrame(rows)
    features = pm4py.extract_features_dataframe(
        dataframe,
        activity_key="concept:name",
        case_id_key="case:concept:name",
        timestamp_key="time:timestamp",
        count_occurrences=True,
    )
    numeric = features.select_dtypes(include=["number"]).fillna(0.0)
    result: FeatureDict = {}
    for column in numeric.columns:
        result[("pm4py_case_feature_mean", column)] = float(numeric[column].mean())
        result[("pm4py_case_feature_std", column)] = float(numeric[column].std(ddof=0))
    return result


def petri_structural_features(sample: ProcessSample) -> FeatureDict:
    graph = sample.petri_graph
    node_counts = Counter(graph.node_types)
    edge_counts = Counter(edge_type for _, _, edge_type in graph.edges)
    transition_labels = [label for label in graph.transition_labels if label is not None]
    return {
        ("nodes", "places"): float(node_counts.get(0, 0)),
        ("nodes", "visible_transitions"): float(node_counts.get(1, 0)),
        ("nodes", "invisible_transitions"): float(node_counts.get(2, 0)),
        ("edges", "place_to_transition"): float(edge_counts.get(0, 0)),
        ("edges", "transition_to_place"): float(edge_counts.get(1, 0)),
        ("marking", "initial_tokens"): float(sum(graph.initial_marking)),
        ("marking", "final_tokens"): float(sum(graph.final_marking)),
        ("labels", "unique_transition_labels"): float(len(set(transition_labels))),
        ("labels", "duplicate_visible_transitions"): float(
            max(0, len(transition_labels) - len(set(transition_labels)))
        ),
    }


def vectorize_feature_dicts(rows: Sequence[FeatureDict]) -> np.ndarray:
    vocabulary = sorted({key for row in rows for key in row}, key=repr)
    matrix = np.zeros((len(rows), len(vocabulary)), dtype=float)
    for row_idx, row in enumerate(rows):
        for col_idx, key in enumerate(vocabulary):
            matrix[row_idx, col_idx] = float(row.get(key, 0.0))
    return matrix


def normalize_counts(counts: Counter[Hashable], total: int) -> FeatureDict:
    if total <= 0:
        return {}
    return {key: value / total for key, value in counts.items()}


def cosine_similarity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    left_norm[left_norm == 0.0] = 1.0
    right_norm[right_norm == 0.0] = 1.0
    return (left / left_norm) @ (right / right_norm).T


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    return 1.0 - cosine_similarity_matrix(embeddings, embeddings)


def vector_statistics(embeddings: np.ndarray) -> dict[str, float | int]:
    embeddings = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(embeddings, axis=1) if embeddings.size else np.asarray([])
    return {
        "count": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "l2_norm_mean": round_float(float(np.mean(norms))) if norms.size else 0.0,
        "l2_norm_std": round_float(float(np.std(norms))) if norms.size else 0.0,
        "feature_variance_mean": round_float(float(np.mean(np.var(embeddings, axis=0))))
        if embeddings.size
        else 0.0,
    }


def pairwise_statistics(distance: np.ndarray) -> dict[str, float | int]:
    values = upper_triangle_values(distance)
    if not values.size:
        return {"pair_count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "pair_count": int(values.size),
        "mean": round_float(float(values.mean())),
        "std": round_float(float(values.std())),
        "min": round_float(float(values.min())),
        "max": round_float(float(values.max())),
    }


def summarize_distance_matrix(distance: np.ndarray) -> dict[str, float | int]:
    return pairwise_statistics(distance)


def nearest_neighbor_behavior(
    embedding_distance: np.ndarray,
    behavior_distance: np.ndarray,
) -> dict[str, float | int]:
    n = embedding_distance.shape[0]
    if n < 2:
        return {
            "count": int(n),
            "mean_behavior_l1_at_nearest_neighbor": 0.0,
            "random_pair_behavior_l1_mean": 0.0,
            "improvement_over_random": 0.0,
        }
    nearest_values = []
    for idx in range(n):
        row = embedding_distance[idx].copy()
        row[idx] = math.inf
        nearest = int(np.argmin(row))
        nearest_values.append(float(behavior_distance[idx, nearest]))
    random_mean = float(np.mean(upper_triangle_values(behavior_distance)))
    nearest_mean = float(np.mean(nearest_values))
    return {
        "count": int(n),
        "mean_behavior_l1_at_nearest_neighbor": round_float(nearest_mean),
        "random_pair_behavior_l1_mean": round_float(random_mean),
        "improvement_over_random": round_float(random_mean - nearest_mean),
    }


def spearman_upper(left: np.ndarray, right: np.ndarray) -> float:
    left_values = upper_triangle_values(left)
    right_values = upper_triangle_values(right)
    if left_values.size < 2:
        return 0.0
    return round_float(pearson(rankdata(left_values), rankdata(right_values)))


def pearson_upper(left: np.ndarray, right: np.ndarray) -> float:
    left_values = upper_triangle_values(left)
    right_values = upper_triangle_values(right)
    if left_values.size < 2:
        return 0.0
    return round_float(pearson(left_values, right_values))


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] < 2:
        return np.asarray([], dtype=float)
    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices]


def round_float(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(float(value), 6)


def round_float_dict(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round_float(value) for key, value in metrics.items()}


def _move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = {
                child_key: child_value.to(device) if isinstance(child_value, torch.Tensor) else child_value
                for child_key, child_value in value.items()
            }
        else:
            moved[key] = value
    return moved
