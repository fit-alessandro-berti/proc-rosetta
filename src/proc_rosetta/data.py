from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from proc_rosetta.pm4py_bridge import PetriGraph
from proc_rosetta.synthetic import ProcessSample, SyntheticConfig, generate_samples
from proc_rosetta.tokenizers import ActivityTokenizer, TreeTokenizer

SPLIT_NAMES = ("training", "validation", "test")
SAMPLES_FILENAME = "samples.jsonl"
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class BatchConfig:
    max_tree_tokens: int = 128
    max_traces: int = 32
    max_trace_length: int = 64
    max_petri_nodes: int = 128


@dataclass(frozen=True)
class SplitCounts:
    training: int = 8192
    validation: int = 1024
    test: int = 1024

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
        batch = {
            "tree_tokens": self._tree_tokens(samples),
            "traces": self._trace_tokens(samples),
            "petri": self._petri_graphs([sample.petri_graph for sample in samples]),
            "equivalence_ids": equivalence_ids,
            "positive_mask": torch.tensor(
                [
                    [left == right for right in equivalence_ids]
                    for left in equivalence_ids
                ],
                dtype=torch.bool,
            ),
            "samples": list(samples),
        }
        self._remap_activity_ids(batch, equivalence_ids)
        return batch

    def _remap_activity_ids(
        self,
        batch: dict[str, Any],
        equivalence_ids: Sequence[str],
    ) -> None:
        if self.activity_remap_probability <= 0:
            return
        tree_tokens = batch["tree_tokens"]
        traces = batch["traces"]
        petri = batch["petri"]
        assert isinstance(tree_tokens, torch.Tensor)
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
                trace_tokens[row][original_traces == source_activity_id] = target_activity_id
                transition_label_ids[row][
                    original_petri == source_activity_id
                ] = target_activity_id

    def _tree_tokens(self, samples: Sequence[ProcessSample]) -> torch.Tensor:
        encoded = [self.tree_tokenizer.encode_tree(sample.tree) for sample in samples]
        max_len = min(max(len(row) for row in encoded), self.config.max_tree_tokens)
        out = torch.full((len(samples), max_len), self.tree_tokenizer.pad_id, dtype=torch.long)
        for idx, row in enumerate(encoded):
            row = row[:max_len]
            if row[-1] != self.tree_tokenizer.eos_id:
                row[-1] = self.tree_tokenizer.eos_id
            out[idx, : len(row)] = torch.tensor(row, dtype=torch.long)
        return out

    def _trace_tokens(self, samples: Sequence[ProcessSample]) -> dict[str, torch.Tensor]:
        max_traces = min(
            max(len(sample.traces) for sample in samples),
            self.config.max_traces,
        )
        max_trace_length = min(
            max((len(trace) for sample in samples for trace in sample.traces), default=1),
            self.config.max_trace_length,
        )
        tokens = torch.full(
            (len(samples), max_traces, max_trace_length),
            self.activity_tokenizer.pad_id,
            dtype=torch.long,
        )
        lengths = torch.zeros((len(samples), max_traces), dtype=torch.long)
        mask = torch.zeros((len(samples), max_traces), dtype=torch.bool)
        for batch_idx, sample in enumerate(samples):
            for trace_idx, trace in enumerate(sample.traces[:max_traces]):
                encoded = self.activity_tokenizer.encode_trace(trace[:max_trace_length])
                if encoded:
                    tokens[batch_idx, trace_idx, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
                lengths[batch_idx, trace_idx] = len(encoded)
                mask[batch_idx, trace_idx] = True
        return {"tokens": tokens, "lengths": lengths, "mask": mask}

    def _petri_graphs(self, graphs: Sequence[PetriGraph]) -> dict[str, torch.Tensor]:
        max_nodes = min(max(graph.num_nodes for graph in graphs), self.config.max_petri_nodes)
        batch_size = len(graphs)
        node_types = torch.zeros((batch_size, max_nodes), dtype=torch.long)
        node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
        transition_label_ids = torch.zeros((batch_size, max_nodes), dtype=torch.long)
        markings = torch.zeros((batch_size, max_nodes, 2), dtype=torch.float32)
        adjacency = torch.zeros((batch_size, 2, max_nodes, max_nodes), dtype=torch.float32)

        for batch_idx, graph in enumerate(graphs):
            node_count = min(graph.num_nodes, max_nodes)
            node_types[batch_idx, :node_count] = torch.tensor(graph.node_types[:node_count], dtype=torch.long)
            node_mask[batch_idx, :node_count] = True
            fallback_labels: dict[str, int] = {}
            for node_idx, label in enumerate(graph.transition_labels[:node_count]):
                if label is None:
                    continue
                token_id = self.activity_tokenizer.token_to_id.get(label)
                if token_id is None:
                    if label not in fallback_labels:
                        fallback_labels[label] = min(
                            len(fallback_labels) + 1,
                            self.activity_tokenizer.vocab_size - 1,
                        )
                    token_id = fallback_labels[label]
                transition_label_ids[batch_idx, node_idx] = token_id
            markings[batch_idx, :node_count, 0] = torch.tensor(
                graph.initial_marking[:node_count], dtype=torch.float32
            )
            markings[batch_idx, :node_count, 1] = torch.tensor(
                graph.final_marking[:node_count], dtype=torch.float32
            )
            for src, dst, edge_type in graph.edges:
                if src < max_nodes and dst < max_nodes:
                    adjacency[batch_idx, edge_type, src, dst] = 1.0

        return {
            "node_types": node_types,
            "node_mask": node_mask,
            "transition_label_ids": transition_label_ids,
            "markings": markings,
            "adjacency": adjacency,
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
        "behavior_count": len({sample.equivalence_id for sample in samples}),
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
    for split, count in counts.items():
        if config.generator == "isolated":
            samples = generate_samples(
                count,
                config=config,
                seed=seed + SPLIT_NAMES.index(split),
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
                    metadata=sample.metadata,
                )
                for idx, sample in enumerate(samples)
            ]
        else:
            from proc_rosetta.families import generate_family_samples

            samples = generate_family_samples(count, config, seed, split=split)
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
        "version": 3,
        "schema": "proc-rosetta.behavior-family-splits.v3",
        "sample_format": "jsonl/process-sample.v2",
        "seed": seed,
        "synthetic_config": config.to_dict(),
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
