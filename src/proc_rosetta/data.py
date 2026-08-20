from __future__ import annotations

from contextlib import nullcontext
import json
import random
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from proc_rosetta.pm4py_bridge import PetriGraph
from proc_rosetta.synthetic import (
    ProcessSample,
    SyntheticConfig,
    decoder_target_trees_for_sample,
    generate_samples,
)
from proc_rosetta.tokenizers import ActivityTokenizer, TreeTokenizer

SPLIT_NAMES = ("training", "validation", "test")
SAMPLES_FILENAME = "samples.jsonl"
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class BatchConfig:
    max_tree_tokens: int = 512
    max_traces: int = 128
    max_trace_length: int = 128
    max_petri_nodes: int = 512
    strict_trace_lengths: bool = True
    strict_tree_tokens: bool = True
    strict_petri_nodes: bool = True


@dataclass(frozen=True)
class SplitCounts:
    training: int = 16384
    validation: int = 2048
    test: int = 2048

    def items(self) -> tuple[tuple[str, int], ...]:
        return (
            ("training", self.training),
            ("validation", self.validation),
            ("test", self.test),
        )


class SyntheticProcessDataset(Dataset[ProcessSample]):
    def __init__(
        self,
        count: int,
        config: SyntheticConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.samples = generate_samples(count, config=config, seed=seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ProcessSample:
        return self.samples[index]


class JsonlProcessDataset(Dataset[ProcessSample]):
    def __init__(self, path: str | Path, show_progress: bool = False) -> None:
        path = Path(path)
        if path.is_dir():
            path = path / SAMPLES_FILENAME
        self.path = path
        self.samples = read_samples_jsonl(path, show_progress=show_progress)
        # Legacy rows are migrated once at load time, never in the hot collator.
        self.samples = [
            sample
            if set(sample.decoder_target_trees) == {"tree", "trace", "petri"}
            else replace(
                sample,
                decoder_target_trees=decoder_target_trees_for_sample(
                    sample.tree,
                    sample.traces,
                    sample.petri_graph,
                ),
            )
            for sample in self.samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ProcessSample:
        return self.samples[index]


class ProcessBatchCollator:
    def __init__(
        self,
        tree_tokenizer: TreeTokenizer,
        activity_tokenizer: ActivityTokenizer,
        config: BatchConfig | None = None,
        activity_remap_probability: float = 0.0,
        seed: int = 13,
    ) -> None:
        if not 0.0 <= activity_remap_probability <= 1.0:
            raise ValueError("activity_remap_probability must be in [0, 1]")
        self.tree_tokenizer = tree_tokenizer
        self.activity_tokenizer = activity_tokenizer
        self.config = config or BatchConfig()
        self.activity_remap_probability = activity_remap_probability
        self.rng = random.Random(seed)

    def __call__(self, samples: Sequence[ProcessSample]) -> dict[str, Any]:
        equivalence_ids = [sample.equivalence_id for sample in samples]
        exact_behavior_ids = [sample.exact_behavior_id for sample in samples]
        partial_order_ids = [sample.partial_order_id for sample in samples]
        signatures = self._behavior_signatures(samples)
        exact_ids = self._categorical_ids(exact_behavior_ids)
        partial_ids = self._categorical_ids(partial_order_ids)
        exact_valid = exact_ids.ge(0)
        partial_valid = partial_ids.ge(0)
        same_exact = exact_ids[:, None].eq(exact_ids[None, :])
        same_partial = partial_ids[:, None].eq(partial_ids[None, :])
        valid_exact_pair = exact_valid[:, None] & exact_valid[None, :]
        positive_mask = (
            valid_exact_pair
            & same_exact
            & partial_valid[:, None]
            & same_partial
        )
        contrastive_candidate_mask = valid_exact_pair & (~same_exact | same_partial)
        tree_tokens = self._tree_tokens(samples)
        tree_depths, tree_parents = self.tree_tokenizer.structure_features(tree_tokens)
        batch = {
            "tree_tokens": tree_tokens,
            "tree_encoder_tokens": tree_tokens,
            "tree_structure": {
                "depths": tree_depths,
                "parents": tree_parents,
            },
            "decoder_targets": self._decoder_targets(samples),
            "source_activity_masks": self._source_activity_masks(samples),
            "traces": self._trace_tokens(samples),
            "petri": self._petri_graphs([sample.petri_graph for sample in samples]),
            "equivalence_ids": equivalence_ids,
            "exact_behavior_ids": exact_behavior_ids,
            "exact_trace_language_ids": [
                sample.exact_trace_language_id for sample in samples
            ],
            "partial_order_ids": partial_order_ids,
            "structural_motif_ids": [sample.structural_motif_id for sample in samples],
            "behavior_signatures": signatures,
            "positive_mask": positive_mask,
            "contrastive_candidate_mask": contrastive_candidate_mask,
            "analogy_mask": valid_exact_pair & same_exact & ~same_partial,
            "samples": list(samples),
        }
        self._remap_activity_ids(batch, equivalence_ids)
        return batch

    @staticmethod
    def _behavior_signatures(samples: Sequence[ProcessSample]) -> torch.Tensor | None:
        dimensions = {len(sample.behavior_signature) for sample in samples}
        if dimensions == {0}:
            return None
        if len(dimensions) != 1 or 0 in dimensions:
            raise ValueError("behavior signatures in one batch must have one non-zero size")
        return torch.tensor(
            [sample.behavior_signature for sample in samples], dtype=torch.float32
        )

    @staticmethod
    def _categorical_ids(values: Sequence[str | None]) -> torch.Tensor:
        mapping: dict[str, int] = {}
        encoded: list[int] = []
        for value in values:
            if value is None:
                encoded.append(-1)
            else:
                mapping.setdefault(value, len(mapping))
                encoded.append(mapping[value])
        return torch.tensor(encoded, dtype=torch.long)

    def _remap_activity_ids(
        self,
        batch: dict[str, Any],
        equivalence_ids: Sequence[str],
    ) -> None:
        if self.activity_remap_probability <= 0:
            return
        tree_tokens = batch["tree_tokens"]
        decoder_targets = batch["decoder_targets"]
        source_activity_masks = batch["source_activity_masks"]
        traces = batch["traces"]
        petri = batch["petri"]
        assert isinstance(tree_tokens, torch.Tensor)
        assert isinstance(decoder_targets, dict)
        assert isinstance(source_activity_masks, dict)
        assert isinstance(traces, dict)
        assert isinstance(petri, dict)
        trace_tokens = traces["tokens"]
        transition_label_ids = petri["transition_label_ids"]
        assert isinstance(trace_tokens, torch.Tensor)
        assert isinstance(transition_label_ids, torch.Tensor)

        family_permutations: dict[str, list[int] | None] = {}
        activity_count = min(
            self.tree_tokenizer.max_activities,
            self.activity_tokenizer.max_activities,
        )
        for row, equivalence_id in enumerate(equivalence_ids):
            if equivalence_id not in family_permutations:
                family_permutations[equivalence_id] = (
                    self.rng.sample(range(activity_count), activity_count)
                    if self.rng.random() < self.activity_remap_probability
                    else None
                )
            permutation = family_permutations[equivalence_id]
            if permutation is None:
                continue

            original_tree = tree_tokens[row].clone()
            original_targets = {
                name: value[row].clone() for name, value in decoder_targets.items()
            }
            original_source_masks = {
                name: value[row].clone() for name, value in source_activity_masks.items()
            }
            original_traces = trace_tokens[row].clone()
            original_petri = transition_label_ids[row].clone()
            for source_index, target_index in enumerate(permutation):
                source_name = f"A{source_index}"
                target_name = f"A{target_index}"
                source_tree_id = self.tree_tokenizer.token_to_id[source_name]
                target_tree_id = self.tree_tokenizer.token_to_id[target_name]
                source_activity_id = self.activity_tokenizer.token_to_id[source_name]
                target_activity_id = self.activity_tokenizer.token_to_id[target_name]
                tree_tokens[row][original_tree == source_tree_id] = target_tree_id
                for name, target_tokens in decoder_targets.items():
                    target_tokens[row][
                        original_targets[name] == source_tree_id
                    ] = target_tree_id
                for name, source_mask in source_activity_masks.items():
                    source_mask[row, target_index] = original_source_masks[name][source_index]
                trace_tokens[row][original_traces == source_activity_id] = target_activity_id
                transition_label_ids[row][
                    original_petri == source_activity_id
                ] = target_activity_id

    def _tree_tokens(self, samples: Sequence[ProcessSample]) -> torch.Tensor:
        # Keep the stored first-seen labels: canonicalizing here would restore
        # tree-DFS label order and desynchronize the tree tokens from the trace
        # and Petri labels of the same sample.
        encoded = [
            self.tree_tokenizer.encode_tree(sample.tree, canonicalize=False)
            for sample in samples
        ]
        longest = max(len(row) for row in encoded)
        if self.config.strict_tree_tokens and longest > self.config.max_tree_tokens:
            raise ValueError(
                "tree target exceeds the configured token maximum in strict mode: "
                f"observed={longest}, maximum={self.config.max_tree_tokens}"
            )
        max_len = min(longest, self.config.max_tree_tokens)
        out = torch.full((len(samples), max_len), self.tree_tokenizer.pad_id, dtype=torch.long)
        for idx, row in enumerate(encoded):
            row = row[:max_len]
            if row[-1] != self.tree_tokenizer.eos_id:
                row[-1] = self.tree_tokenizer.eos_id
            out[idx, : len(row)] = torch.tensor(row, dtype=torch.long)
        return out

    def _decoder_targets(
        self,
        samples: Sequence[ProcessSample],
    ) -> dict[str, torch.Tensor]:
        target_rows: dict[str, list[list[int]]] = {name: [] for name in ("tree", "trace", "petri")}
        for sample in samples:
            targets = sample.decoder_target_trees
            if set(targets) != set(target_rows):
                targets = decoder_target_trees_for_sample(
                    sample.tree,
                    sample.traces,
                    sample.petri_graph,
                )
            for name in target_rows:
                target_rows[name].append(
                    self.tree_tokenizer.encode_tree(targets[name], canonicalize=False)
                )
        longest = max(len(row) for rows in target_rows.values() for row in rows)
        if self.config.strict_tree_tokens and longest > self.config.max_tree_tokens:
            raise ValueError(
                "decoder target exceeds the configured token maximum in strict mode: "
                f"observed={longest}, maximum={self.config.max_tree_tokens}"
            )
        width = min(longest, self.config.max_tree_tokens)
        result: dict[str, torch.Tensor] = {}
        for name, rows in target_rows.items():
            tensor = torch.full(
                (len(samples), width),
                self.tree_tokenizer.pad_id,
                dtype=torch.long,
            )
            for index, encoded in enumerate(rows):
                encoded = encoded[:width]
                if encoded[-1] != self.tree_tokenizer.eos_id:
                    encoded[-1] = self.tree_tokenizer.eos_id
                tensor[index, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
            result[name] = tensor
        return result

    def _source_activity_masks(
        self,
        samples: Sequence[ProcessSample],
    ) -> dict[str, torch.Tensor]:
        masks = {
            name: torch.zeros(
                (len(samples), self.tree_tokenizer.max_activities),
                dtype=torch.bool,
            )
            for name in ("tree", "trace", "petri")
        }
        for row, sample in enumerate(samples):
            labels = {
                "tree": set(sample.tree.activity_labels()),
                "trace": {label for trace in sample.traces for label in trace},
                "petri": {
                    label for label in sample.petri_graph.transition_labels if label is not None
                },
            }
            for name, alphabet in labels.items():
                for label in alphabet:
                    if label.startswith("A") and label[1:].isdigit():
                        index = int(label[1:])
                        if 0 <= index < self.tree_tokenizer.max_activities:
                            masks[name][row, index] = True
        return masks

    def _trace_tokens(self, samples: Sequence[ProcessSample]) -> dict[str, torch.Tensor]:
        original_trace_counts = torch.tensor(
            [len(sample.traces) for sample in samples], dtype=torch.long
        )
        largest_trace_count = int(original_trace_counts.max().item())
        if self.config.strict_trace_lengths and largest_trace_count > self.config.max_traces:
            raise ValueError(
                "trace-set size exceeds the configured maximum in strict mode: "
                f"observed={largest_trace_count}, maximum={self.config.max_traces}"
            )
        max_traces = min(
            largest_trace_count,
            self.config.max_traces,
        )
        original_max_trace_length = max(
            (len(trace) for sample in samples for trace in sample.traces), default=1
        )
        if self.config.strict_trace_lengths and original_max_trace_length > self.config.max_trace_length:
            raise ValueError(
                "trace length exceeds the configured maximum in strict mode: "
                f"observed={original_max_trace_length}, maximum={self.config.max_trace_length}"
            )
        max_trace_length = min(original_max_trace_length, self.config.max_trace_length)
        tokens = torch.full(
            (len(samples), max_traces, max_trace_length),
            self.activity_tokenizer.pad_id,
            dtype=torch.long,
        )
        lengths = torch.zeros((len(samples), max_traces), dtype=torch.long)
        mask = torch.zeros((len(samples), max_traces), dtype=torch.bool)
        original_lengths = torch.zeros((len(samples), max_traces), dtype=torch.long)
        was_truncated = torch.zeros((len(samples), max_traces), dtype=torch.bool)
        for batch_idx, sample in enumerate(samples):
            for trace_idx, trace in enumerate(sample.traces[:max_traces]):
                encoded = self.activity_tokenizer.encode_trace(trace[:max_trace_length])
                if encoded:
                    tokens[batch_idx, trace_idx, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
                lengths[batch_idx, trace_idx] = len(encoded)
                original_lengths[batch_idx, trace_idx] = len(trace)
                was_truncated[batch_idx, trace_idx] = len(trace) > max_trace_length
                mask[batch_idx, trace_idx] = True
        return {
            "tokens": tokens,
            "lengths": lengths,
            "original_lengths": original_lengths,
            "was_truncated": was_truncated,
            "original_trace_counts": original_trace_counts,
            "trace_sets_were_truncated": original_trace_counts.gt(max_traces),
            "mask": mask,
        }

    def _petri_graphs(self, graphs: Sequence[PetriGraph]) -> dict[str, torch.Tensor]:
        largest_graph = max(graph.num_nodes for graph in graphs)
        if self.config.strict_petri_nodes and largest_graph > self.config.max_petri_nodes:
            raise ValueError(
                "Petri graph exceeds the configured node maximum in strict mode: "
                f"observed={largest_graph}, maximum={self.config.max_petri_nodes}"
            )
        max_nodes = min(largest_graph, self.config.max_petri_nodes)
        batch_size = len(graphs)
        node_types = torch.zeros((batch_size, max_nodes), dtype=torch.long)
        node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
        transition_label_ids = torch.zeros((batch_size, max_nodes), dtype=torch.long)
        markings = torch.zeros((batch_size, max_nodes, 2), dtype=torch.float32)
        edge_sources: list[int] = []
        edge_targets: list[int] = []
        edge_types: list[int] = []

        for batch_idx, graph in enumerate(graphs):
            node_count = min(graph.num_nodes, max_nodes)
            node_types[batch_idx, :node_count] = torch.tensor(graph.node_types[:node_count], dtype=torch.long)
            node_mask[batch_idx, :node_count] = True
            for node_idx, label in enumerate(graph.transition_labels[:node_count]):
                if label is None:
                    continue
                token_id = self.activity_tokenizer.token_to_id.get(label)
                if token_id is None:
                    token_id = self.activity_tokenizer.unk_id
                transition_label_ids[batch_idx, node_idx] = token_id
            markings[batch_idx, :node_count, 0] = torch.tensor(
                graph.initial_marking[:node_count], dtype=torch.float32
            )
            markings[batch_idx, :node_count, 1] = torch.tensor(
                graph.final_marking[:node_count], dtype=torch.float32
            )
            for src, dst, edge_type in graph.edges:
                if src < max_nodes and dst < max_nodes:
                    edge_sources.append(batch_idx * max_nodes + src)
                    edge_targets.append(batch_idx * max_nodes + dst)
                    edge_types.append(edge_type)

        return {
            "node_types": node_types,
            "node_mask": node_mask,
            "transition_label_ids": transition_label_ids,
            "markings": markings,
            "edge_index": torch.tensor(
                [edge_sources, edge_targets],
                dtype=torch.long,
            ),
            "edge_types": torch.tensor(edge_types, dtype=torch.long),
        }


def split_samples_path(data_dir: str | Path, split: str) -> Path:
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}; expected one of {', '.join(SPLIT_NAMES)}")
    return Path(data_dir) / split / SAMPLES_FILENAME


def write_samples_jsonl(path: str | Path, samples: Sequence[ProcessSample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), sort_keys=True))
            handle.write("\n")


def read_samples_jsonl(path: str | Path, show_progress: bool = False) -> list[ProcessSample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"sample file does not exist: {path}")
    samples: list[ProcessSample] = []
    total = count_lines(path) if show_progress else None
    with path.open("r", encoding="utf-8") as handle:
        iterator = progress_iterator(
            handle,
            total=total,
            desc=f"Loading {path.parent.name}",
            enabled=show_progress,
            unit="samples",
        )
        for line_number, line in enumerate(iterator, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(ProcessSample.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid sample JSON at {path}:{line_number}") from exc
    if not samples:
        raise ValueError(f"sample file is empty: {path}")
    return samples


def sample_statistics(samples: Sequence[ProcessSample]) -> dict[str, Any]:
    trace_lengths = [len(trace) for sample in samples for trace in sample.traces]
    family_motifs: dict[str, str] = {}
    for sample in samples:
        motif = str(sample.metadata.get("motif", "unknown"))
        existing = family_motifs.setdefault(sample.equivalence_id, motif)
        if existing != motif:
            raise ValueError(
                f"behavior {sample.equivalence_id} has inconsistent motif labels"
            )
    motif_representation_counts: dict[str, dict[str, int]] = {}
    motif_representation_slot_counts: dict[str, dict[str, int]] = {}
    for sample in samples:
        motif = str(sample.metadata.get("motif", "unknown"))
        counts = motif_representation_counts.setdefault(motif, {})
        counts[sample.representation_kind] = counts.get(sample.representation_kind, 0) + 1
        slot = str(sample.metadata.get("representation_slot", "unknown"))
        slot_counts = motif_representation_slot_counts.setdefault(motif, {})
        slot_counts[slot] = slot_counts.get(slot, 0) + 1
    result: dict[str, Any] = {
        "count": len(samples),
        "avg_tree_size": _mean(sample.tree.size() for sample in samples),
        "avg_tree_depth": _mean(sample.tree.max_depth() for sample in samples),
        "avg_trace_count": _mean(len(sample.traces) for sample in samples),
        "avg_trace_length": _mean(trace_lengths),
        "max_petri_nodes": max((sample.petri_graph.num_nodes for sample in samples), default=0),
        "max_petri_edges": max((sample.petri_graph.num_edges for sample in samples), default=0),
        "max_trace_length": max(trace_lengths, default=0),
        "unexpected_truncation_count": sum(
            int(bool(sample.metadata.get("was_truncated", False))) for sample in samples
        ),
        "behavior_count": len({sample.equivalence_id for sample in samples}),
        "exact_behavior_count": len(
            {
                sample.exact_behavior_id
                for sample in samples
                if sample.exact_behavior_id is not None
            }
        ),
        "bounded_behavior_count": len(
            {
                sample.equivalence_id
                for sample in samples
                if sample.exact_behavior_id is None
            }
        ),
        "family_counts_by_motif": dict(
            sorted(_counts(family_motifs.values()).items())
        ),
        "sample_counts_by_motif": dict(
            sorted(_counts(str(sample.metadata.get("motif", "unknown")) for sample in samples).items())
        ),
        "representation_counts": dict(
            sorted(
                _counts(sample.representation_kind for sample in samples).items()
            )
        ),
        "motif_representation_counts": {
            motif: dict(sorted(counts.items()))
            for motif, counts in sorted(motif_representation_counts.items())
        },
        "motif_representation_slot_counts": {
            motif: dict(sorted(counts.items()))
            for motif, counts in sorted(motif_representation_slot_counts.items())
        },
    }
    return result


def class_coverage_report(
    samples: Sequence[ProcessSample],
    config: SyntheticConfig,
    seed: int,
    split: str,
) -> dict[str, Any]:
    from proc_rosetta.families import active_motifs, motif_quota_plan

    statistics = sample_statistics(samples)
    family_count = int(statistics["behavior_count"])
    _, planned, minimum, planned_meets_minimum = motif_quota_plan(
        family_count, config, seed, split
    )
    actual = {
        str(key): int(value)
        for key, value in dict(statistics["family_counts_by_motif"]).items()
    }
    active = active_motifs(config.motif_weights)
    deficits = {
        motif: max(0, minimum - actual.get(motif, 0))
        for motif in active
        if actual.get(motif, 0) < minimum
    }
    minimum_rows_per_slot = minimum * config.log_views_per_behavior
    slot_counts = dict(statistics["motif_representation_slot_counts"])
    representation_slot_deficits: dict[str, dict[str, int]] = {}
    for motif in active:
        per_motif = dict(slot_counts.get(motif, {}))
        missing = {
            str(slot): max(0, minimum_rows_per_slot - int(per_motif.get(str(slot), 0)))
            for slot in range(config.variants_per_behavior)
            if int(per_motif.get(str(slot), 0)) < minimum_rows_per_slot
        }
        if missing:
            representation_slot_deficits[motif] = missing
    exact_quota_match = all(actual.get(motif, 0) == planned[motif] for motif in active)
    meets_minimum = not deficits and not representation_slot_deficits
    report: dict[str, Any] = {
        "mode": config.class_coverage_mode,
        "unit": "behavior_family",
        "active_motifs": list(active),
        "minimum_families_per_motif": minimum,
        "planned_family_counts_by_motif": dict(sorted(planned.items())),
        "actual_family_counts_by_motif": dict(sorted(actual.items())),
        "sample_counts_by_motif": statistics["sample_counts_by_motif"],
        "motif_representation_counts": statistics["motif_representation_counts"],
        "motif_representation_slot_counts": slot_counts,
        "minimum_rows_per_representation_slot": minimum_rows_per_slot,
        "representation_slot_deficits_by_motif": representation_slot_deficits,
        "deficits_by_motif": dict(sorted(deficits.items())),
        "planned_meets_minimum": planned_meets_minimum,
        "exact_quota_match": exact_quota_match,
        "meets_minimum": meets_minimum,
    }
    if config.class_coverage_mode == "strict" and (not meets_minimum or not exact_quota_match):
        raise RuntimeError(f"strict class coverage failed for {split}: {report}")
    return report


def recreate_data_splits(
    data_dir: str | Path,
    counts: SplitCounts | None = None,
    config: SyntheticConfig | None = None,
    seed: int = 13,
    show_progress: bool = False,
) -> dict[str, object]:
    data_dir = Path(data_dir)
    counts = counts or SplitCounts()
    config = config or SyntheticConfig()
    if config.generator == "behavior_families":
        from proc_rosetta.families import motif_quota_plan

        if config.variants_per_behavior <= 0 or config.log_views_per_behavior <= 0:
            raise ValueError(
                "variants_per_behavior and log_views_per_behavior must be positive"
            )
        rows_per_family = config.variants_per_behavior * config.log_views_per_behavior
        for split, count in counts.items():
            if config.class_coverage_mode == "strict" and count % rows_per_family:
                raise ValueError(
                    f"strict class coverage requires the {split} sample count ({count}) "
                    f"to be divisible by rows_per_family ({rows_per_family})"
                )
            family_count = (count + rows_per_family - 1) // rows_per_family
            motif_quota_plan(family_count, config, seed, split)
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)

    split_metadata: dict[str, object] = {}
    exact_signatures_by_split: dict[str, set[str]] = {}
    reserved_exact_behavior_ids: set[str] = set()
    for split, count in counts.items():
        with progress_bar(
            total=count,
            enabled=show_progress,
            desc=f"Generating {split}",
            unit="triplet",
        ) as progress:
            if config.generator == "isolated":
                samples = generate_samples(
                    count,
                    config=config,
                    seed=seed + SPLIT_NAMES.index(split),
                    progress_update=progress.update,
                )
                samples = [
                    ProcessSample(
                        tree=sample.tree,
                        traces=sample.traces,
                        petri_graph=sample.petri_graph,
                        equivalence_id=f"{split}-{idx}",
                        model_variant_id=sample.model_variant_id,
                        log_view_id=sample.log_view_id,
                        representation_kind=sample.representation_kind,
                        equivalence_level=sample.equivalence_level,
                        decoder_target_trees=sample.decoder_target_trees,
                        metadata=sample.metadata,
                    )
                    for idx, sample in enumerate(samples)
                ]
            else:
                from proc_rosetta.families import generate_family_samples

                samples = generate_family_samples(
                    count,
                    config,
                    seed,
                    split=split,
                    progress_update=progress.update,
                    excluded_exact_behavior_ids=reserved_exact_behavior_ids,
                )
        split_exact_ids = {
            sample.exact_behavior_id
            for sample in samples
            if sample.exact_behavior_id is not None
        }
        for prior_split, prior_ids in exact_signatures_by_split.items():
            overlap = prior_ids & split_exact_ids
            if overlap:
                raise RuntimeError(
                    f"exact behavior signatures overlap between {prior_split} and {split}: "
                    f"{sorted(overlap)[:3]}"
                )
        exact_signatures_by_split[split] = split_exact_ids
        reserved_exact_behavior_ids.update(split_exact_ids)
        path = split_samples_path(data_dir, split)
        write_samples_jsonl(path, samples)
        statistics = sample_statistics(samples)
        split_metadata[split] = {
            "path": str(path.relative_to(data_dir)),
            "statistics": statistics,
            "class_coverage": (
                class_coverage_report(samples, config, seed, split)
                if config.generator == "behavior_families"
                else {"mode": "not_applicable", "reason": "isolated_generator"}
            ),
        }

    metadata = {
        "version": 5,
        "schema": "proc-rosetta.behavior-family-splits.v5",
        "sample_format": "jsonl/process-sample.v4",
        "tree_normalization_version": "pm4py-fold-v1",
        "seed": seed,
        "synthetic_config": config.to_dict(),
        "exact_behavior_signatures_disjoint": True,
        "exact_behavior_signature_counts": {
            split: len(values) for split, values in exact_signatures_by_split.items()
        },
        "splits": split_metadata,
    }
    with (data_dir / METADATA_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata


def load_data_metadata(data_dir: str | Path) -> dict[str, object]:
    metadata_path = Path(data_dir) / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata file does not exist: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def progress_iterator(iterable: Any, enabled: bool, **kwargs: Any) -> Any:
    if not enabled:
        return iterable
    from tqdm.auto import tqdm

    return tqdm(iterable, leave=False, **kwargs)


def progress_bar(total: int, enabled: bool, **kwargs: Any) -> Any:
    if not enabled:
        return nullcontext(_NoOpProgress())
    from tqdm.auto import tqdm

    return tqdm(total=total, leave=False, **kwargs)


class _NoOpProgress:
    def update(self, _: int = 1) -> None:
        return


def _mean(values: Sequence[float] | Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts
