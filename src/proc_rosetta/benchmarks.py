from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from proc_rosetta.behavior import behavioral_distance
from proc_rosetta.data import ProcessBatchCollator
from proc_rosetta.devices import resolve_device
from proc_rosetta.pm4py_bridge import petri_graph_to_net, simulate_traces, tree_to_petri_net
from proc_rosetta.synthetic import ProcessSample
from proc_rosetta.training import evaluate_split_from_checkpoint, load_checkpoint


Trace = Sequence[str]
FeatureDict = dict[Hashable, float]


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
    batch_size: int = 16,
    device: str | None = None,
    include_pm4py_petri: bool = True,
    pm4py_petri_config: Pm4pyPetriEmbeddingConfig | None = None,
) -> dict[str, object]:
    torch_device = resolve_device(device)
    device_name = str(torch_device)
    pm4py_petri_config = pm4py_petri_config or Pm4pyPetriEmbeddingConfig()
    loss_metrics = evaluate_split_from_checkpoint(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        split="test",
        batch_size=batch_size,
        device=device_name,
    )
    model, _ = load_checkpoint(checkpoint_path, torch_device)
    neural_embeddings = proc_rosetta_embeddings(
        model, samples, batch_size=batch_size, device=device_name
    )
    decode_quality = decode_quality_report(
        model,
        samples,
        batch_size=batch_size,
        device=device_name,
    )
    discovery_quality = discovery_quality_report(
        model,
        samples,
        batch_size=batch_size,
        device=device_name,
    )

    behavior = behavior_matrices(samples)
    methods: dict[str, dict[str, object]] = {}
    method_embeddings: dict[str, np.ndarray] = {}
    for name, matrix in neural_embeddings.items():
        method_embeddings[name] = matrix
        methods[name] = evaluate_embedding_method(matrix, behavior["mean_l1"])
        methods[name]["kind"] = "learned_proc_rosetta_latent"

    baselines = {
        "trace_activity_counts": vectorize_feature_dicts(
            [activity_count_features(sample.traces) for sample in samples]
        ),
        "trace_variant_distribution": vectorize_feature_dicts(
            [trace_variant_features(sample.traces) for sample in samples]
        ),
        "trace_directly_follows": vectorize_feature_dicts(
            [directly_follows_features(sample.traces) for sample in samples]
        ),
        "trace_eventually_follows": vectorize_feature_dicts(
            [eventually_follows_features(sample.traces) for sample in samples]
        ),
        "pm4py_log_case_features_mean_std": vectorize_feature_dicts(
            [pm4py_log_case_features(sample.traces) for sample in samples]
        ),
        "petri_structural_counts": vectorize_feature_dicts(
            [petri_structural_features(sample) for sample in samples]
        ),
    }
    for name, matrix in baselines.items():
        method_embeddings[name] = matrix
        methods[name] = evaluate_embedding_method(matrix, behavior["mean_l1"])
        methods[name]["kind"] = "deterministic_baseline"

    if include_pm4py_petri:
        pm4py_embeddings, pm4py_report = pm4py_petri_method_report(
            samples,
            behavior["mean_l1"],
            pm4py_petri_config,
        )
        methods["pm4py_colonna_petri_node2vec"] = pm4py_report
        if pm4py_embeddings is not None:
            method_embeddings["pm4py_colonna_petri_node2vec"] = pm4py_embeddings

    report = {
        "split": "test",
        "sample_count": len(samples),
        "loss_metrics": round_float_dict(loss_metrics),
        "behavioral_distance_summary": summarize_distance_matrix(behavior["mean_l1"]),
        "behavioral_component_summaries": {
            key: summarize_distance_matrix(value)
            for key, value in behavior.items()
            if key != "mean_l1"
        },
        "decode_quality": decode_quality,
        "discovery_quality": discovery_quality,
        "cross_modal_retrieval": cross_modal_retrieval(neural_embeddings),
        "equivalence_families": equivalence_family_embedding_report(
            samples, neural_embeddings
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
    return report


def equivalence_family_embedding_report(
    samples: Sequence[ProcessSample],
    embeddings: dict[str, np.ndarray],
) -> dict[str, object]:
    """Measure representation invariance using explicit behavior-family IDs."""

    family_ids = [sample.equivalence_id for sample in samples]
    representation_kinds = [sample.representation_kind for sample in samples]
    family_sizes = Counter(family_ids)
    paired_indices = [index for index, value in enumerate(family_ids) if family_sizes[value] > 1]
    methods: dict[str, object] = {}
    for method, matrix in embeddings.items():
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
) -> dict[str, object]:
    model.eval()
    torch_device = resolve_device(device)
    model.to(torch_device)
    collator = ProcessBatchCollator(model.tree_tokenizer, model.activity_tokenizer)
    rows: dict[str, list[dict[str, object]]] = {
        "proc_rosetta_trace_mu": [],
        "inductive_miner": [],
    }

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        batch = _move_batch_to_device(collator(batch_samples), torch_device)
        trace_dist = model.encode_traces(batch["traces"])
        decoded = model.tree_decoder.decode_greedy(
            trace_dist.mu,
            max_length=max_decode_length,
            apply_grammar_mask=True,
        )
        for sample, token_ids in zip(batch_samples, decoded.detach().cpu().tolist()):
            rows["proc_rosetta_trace_mu"].append(
                evaluate_proc_rosetta_discovery(
                    model,
                    sample,
                    token_ids,
                )
            )
            rows["inductive_miner"].append(evaluate_inductive_miner_discovery(sample))

    return {
        "description": (
            "Alignment-based discovery quality on each test log. ProcRosetta uses "
            "the trace encoder and grammar-masked process-tree decoder; the "
            "baseline uses PM4Py Inductive Miner. Each discovered process tree is "
            "converted to a Petri net and scored by alignment fitness, alignment "
            "precision, and their harmonic-mean F1."
        ),
        "max_decode_length": int(max_decode_length),
        "methods": {name: summarize_discovery_quality(values) for name, values in rows.items()},
    }


def evaluate_proc_rosetta_discovery(
    model,
    sample: ProcessSample,
    token_ids: Sequence[int],
) -> dict[str, object]:
    row = discovery_quality_row()
    try:
        if model.tree_tokenizer.eos_id not in [int(token_id) for token_id in token_ids]:
            raise ValueError("decoder did not emit <eos>")
        tree = model.tree_tokenizer.decode_tree(token_ids)
        bundle = tree_to_petri_net(tree)
        row["model_discovered"] = True
    except Exception as exc:
        row["error"] = f"decode:{type(exc).__name__}: {exc}"
        return row

    return score_discovered_petri_net(
        sample,
        bundle.net,
        bundle.initial_marking,
        bundle.final_marking,
        row=row,
    )


def evaluate_inductive_miner_discovery(sample: ProcessSample) -> dict[str, object]:
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
        net, initial_marking, final_marking = pm4py.convert_to_petri_net(tree)
        row["model_discovered"] = True
    except Exception as exc:
        row["error"] = f"discover:{type(exc).__name__}: {exc}"
        return row

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
        "alignment_evaluable": False,
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
        fitness, precision = alignment_fitness_precision(
            sample.traces,
            net,
            initial_marking,
            final_marking,
        )
        row["fitness"] = round_float(fitness)
        row["precision"] = round_float(precision)
        row["f1"] = alignment_f1_score(fitness, precision)
        row["alignment_evaluable"] = True
    except Exception as exc:
        row["error"] = f"alignment:{type(exc).__name__}: {exc}"
    return row


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
        raise ValueError("alignment-based discovery quality requires at least one event")
    return pd.DataFrame(rows)


def extract_alignment_fitness(value: object) -> float:
    if isinstance(value, dict):
        for key in ("log_fitness", "averageFitness", "average_trace_fitness"):
            if key in value:
                return float(value[key])
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"unsupported alignment fitness result: {value!r}")


def alignment_f1_score(fitness: float, precision: float) -> float:
    denominator = fitness + precision
    if denominator <= 0.0:
        return 0.0
    return round_float(2.0 * fitness * precision / denominator)


def summarize_discovery_quality(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    fitness_values = numeric_row_values(rows, "fitness")
    precision_values = numeric_row_values(rows, "precision")
    f1_values = numeric_row_values(rows, "f1")
    first_error = next((str(row["error"]) for row in rows if row.get("error")), None)
    return {
        "count": int(len(rows)),
        "model_discovered_rate": rate(rows, "model_discovered"),
        "alignment_evaluable_rate": rate(rows, "alignment_evaluable"),
        "mean_fitness": round_float(mean(fitness_values)),
        "mean_precision": round_float(mean(precision_values)),
        "mean_f1": round_float(mean(f1_values)),
        "median_f1": round_float(float(np.median(f1_values))) if f1_values else 0.0,
        "alignment_error_count": count_false(rows, "alignment_evaluable"),
        "first_error": first_error,
    }


def numeric_row_values(rows: Sequence[dict[str, object]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]


@torch.no_grad()
def proc_rosetta_embeddings(
    model,
    samples: Sequence[ProcessSample],
    batch_size: int = 16,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    torch_device = resolve_device(device)
    model.to(torch_device)
    collator = ProcessBatchCollator(model.tree_tokenizer, model.activity_tokenizer)
    chunks: dict[str, list[np.ndarray]] = {"tree": [], "trace": [], "petri": []}
    for start in range(0, len(samples), batch_size):
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

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        batch = _move_batch_to_device(collator(batch_samples), torch_device)
        tree_dist = model.encode_tree(batch["tree_tokens"])
        trace_dist = model.encode_traces(batch["traces"])
        petri_dist = model.encode_petri(batch["petri"])
        latents = {
            "proc_rosetta_tree_mu": tree_dist.mu,
            "proc_rosetta_trace_mu": trace_dist.mu,
            "proc_rosetta_petri_mu": petri_dist.mu,
        }
        latents["proc_rosetta_fused_mu"] = torch.stack(tuple(latents.values()), dim=0).mean(dim=0)

        for name, latent in latents.items():
            decoded = model.tree_decoder.decode_greedy(
                latent,
                max_length=max_decode_length,
                apply_grammar_mask=True,
            )
            for sample, token_ids in zip(batch_samples, decoded.detach().cpu().tolist()):
                rows[name].append(
                    evaluate_single_decode(
                        model,
                        sample,
                        token_ids,
                        behavior_traces_per_sample=behavior_traces_per_sample,
                    )
                )

    return {
        "description": (
            "Greedy decodes from each ProcRosetta latent into the grammar-masked "
            "process-tree decoder. Petri validity is measured by converting the "
            "decoded process tree to a Petri net."
        ),
        "max_decode_length": int(max_decode_length),
        "behavior_traces_per_sample": int(behavior_traces_per_sample),
        "methods": {name: summarize_decode_quality(values) for name, values in rows.items()},
    }


def evaluate_single_decode(
    model,
    sample: ProcessSample,
    token_ids: Sequence[int],
    behavior_traces_per_sample: int = 128,
) -> dict[str, object]:
    tokenizer = model.tree_tokenizer
    decoded_tokens = trim_tree_token_sequence(token_ids, tokenizer)
    target_tokens = tokenizer.encode_tree(sample.tree)
    normalized_denominator = max(len(target_tokens), len(decoded_tokens), 1)
    token_edit_distance = levenshtein_distance(target_tokens, decoded_tokens)
    row: dict[str, object] = {
        "terminated": tokenizer.eos_id in [int(token_id) for token_id in token_ids],
        "valid_tree": False,
        "exact_tree_match": False,
        "petri_convertible": False,
        "behavior_evaluable": False,
        "token_edit_distance": token_edit_distance,
        "normalized_token_edit_distance": token_edit_distance / normalized_denominator,
        "behavior_l1": None,
        "error": None,
    }

    try:
        decoded_tree = tokenizer.decode_tree(token_ids)
        row["valid_tree"] = True
    except Exception as exc:
        row["error"] = f"decode:{type(exc).__name__}: {exc}"
        return row

    target_tree = tokenizer.decode_tree(target_tokens)
    row["exact_tree_match"] = decoded_tree.to_dict() == target_tree.to_dict()

    try:
        tree_to_petri_net(decoded_tree)
        row["petri_convertible"] = True
    except Exception as exc:
        row["error"] = f"petri:{type(exc).__name__}: {exc}"

    try:
        trace_count = max(1, min(len(sample.traces), behavior_traces_per_sample))
        decoded_traces = simulate_traces(decoded_tree, num_traces=trace_count)
        row["behavior_l1"] = behavioral_distance(sample.traces, decoded_traces)["mean_l1"]
        row["behavior_evaluable"] = True
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
    return {
        "count": int(count),
        "terminated_rate": rate(rows, "terminated"),
        "valid_tree_rate": rate(rows, "valid_tree"),
        "exact_tree_match_rate": rate(rows, "exact_tree_match"),
        "petri_conversion_rate": rate(rows, "petri_convertible"),
        "behavior_eval_success_rate": rate(rows, "behavior_evaluable"),
        "mean_token_edit_distance": round_float(mean(row["token_edit_distance"] for row in rows)),
        "mean_normalized_token_edit_distance": round_float(
            mean(row["normalized_token_edit_distance"] for row in rows)
        ),
        "mean_behavior_l1": round_float(mean(behavior_values)),
        "median_behavior_l1": round_float(float(np.median(behavior_values))) if behavior_values else 0.0,
        "invalid_decode_count": count_false(rows, "valid_tree"),
        "petri_conversion_error_count": count_false(rows, "petri_convertible"),
        "behavior_error_count": count_false(rows, "behavior_evaluable"),
        "first_error": first_error,
    }


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


def behavior_matrices(samples: Sequence[ProcessSample]) -> dict[str, np.ndarray]:
    n = len(samples)
    matrices = {
        "mean_l1": np.zeros((n, n), dtype=float),
        "variant_l1": np.zeros((n, n), dtype=float),
        "directly_follows_l1": np.zeros((n, n), dtype=float),
        "length_l1": np.zeros((n, n), dtype=float),
    }
    for i in range(n):
        for j in range(i + 1, n):
            distance = behavioral_distance(samples[i].traces, samples[j].traces)
            for key, matrix in matrices.items():
                matrix[i, j] = matrix[j, i] = float(distance[key])
    return matrices


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
        for sample in samples:
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
    lines.append("ProcRosetta Test Report")
    lines.append("=======================")
    lines.append("")
    lines.append(f"Split: {report['split']}  |  samples: {report['sample_count']}")
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

    decode_quality = report.get("decode_quality", {})
    if isinstance(decode_quality, dict):
        decode_methods = decode_quality.get("methods", {})
        if isinstance(decode_methods, dict):
            lines.append("Decode quality")
            lines.append("--------------")
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
            lines.append("Process discovery quality")
            lines.append("-------------------------")
            lines.append(
                "Alignment-based quality compares the trace-decoded ProcRosetta model "
                "with PM4Py Inductive Miner on each test log."
            )
            lines.append(
                format_table(
                    ["method", "model ok", "align ok", "fitness", "precision", "F1"],
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
        "- discovery F1: harmonic mean of alignment fitness and alignment precision; "
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
                format_rate(method.get("alignment_evaluable_rate", 0.0)),
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


def cross_modal_retrieval(embeddings: dict[str, np.ndarray]) -> dict[str, dict[str, float | int]]:
    pairs = {
        "tree_to_trace": ("proc_rosetta_tree_mu", "proc_rosetta_trace_mu"),
        "trace_to_tree": ("proc_rosetta_trace_mu", "proc_rosetta_tree_mu"),
        "tree_to_petri": ("proc_rosetta_tree_mu", "proc_rosetta_petri_mu"),
        "petri_to_tree": ("proc_rosetta_petri_mu", "proc_rosetta_tree_mu"),
        "trace_to_petri": ("proc_rosetta_trace_mu", "proc_rosetta_petri_mu"),
        "petri_to_trace": ("proc_rosetta_petri_mu", "proc_rosetta_trace_mu"),
    }
    return {
        name: retrieval_metrics(embeddings[left], embeddings[right])
        for name, (left, right) in pairs.items()
    }


def retrieval_metrics(query: np.ndarray, candidates: np.ndarray) -> dict[str, float | int]:
    similarity = cosine_similarity_matrix(query, candidates)
    ranks = []
    for idx in range(similarity.shape[0]):
        order = np.argsort(-similarity[idx])
        rank = int(np.where(order == idx)[0][0]) + 1
        ranks.append(rank)
    ranks_array = np.asarray(ranks, dtype=float)
    return {
        "count": int(len(ranks)),
        "top1_accuracy": round_float(float(np.mean(ranks_array == 1.0))) if len(ranks) else 0.0,
        "mean_rank": round_float(float(np.mean(ranks_array))) if len(ranks) else 0.0,
        "mrr": round_float(float(np.mean(1.0 / ranks_array))) if len(ranks) else 0.0,
    }


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
