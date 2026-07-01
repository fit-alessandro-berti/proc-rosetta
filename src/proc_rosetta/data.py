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
from proc_rosetta.synthetic import ProcessSample, SyntheticConfig, generate_sample, generate_samples
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
    training: int = 128
    validation: int = 32
    test: int = 32

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
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if path.is_dir():
            path = path / SAMPLES_FILENAME
        self.path = path
        self.samples = read_samples_jsonl(path)

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
    ) -> None:
        self.tree_tokenizer = tree_tokenizer
        self.activity_tokenizer = activity_tokenizer
        self.config = config or BatchConfig()

    def __call__(self, samples: Sequence[ProcessSample]) -> dict[str, Any]:
        return {
            "tree_tokens": self._tree_tokens(samples),
            "traces": self._trace_tokens(samples),
            "petri": self._petri_graphs([sample.petri_graph for sample in samples]),
            "samples": list(samples),
        }

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
        markings = torch.zeros((batch_size, max_nodes, 2), dtype=torch.float32)
        adjacency = torch.zeros((batch_size, 2, max_nodes, max_nodes), dtype=torch.float32)

        for batch_idx, graph in enumerate(graphs):
            node_count = min(graph.num_nodes, max_nodes)
            node_types[batch_idx, :node_count] = torch.tensor(graph.node_types[:node_count], dtype=torch.long)
            node_mask[batch_idx, :node_count] = True
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


def read_samples_jsonl(path: str | Path) -> list[ProcessSample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"sample file does not exist: {path}")
    samples: list[ProcessSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
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


def sample_statistics(samples: Sequence[ProcessSample]) -> dict[str, float | int]:
    trace_lengths = [len(trace) for sample in samples for trace in sample.traces]
    return {
        "count": len(samples),
        "avg_tree_size": _mean(sample.tree.size() for sample in samples),
        "avg_tree_depth": _mean(sample.tree.max_depth() for sample in samples),
        "avg_trace_count": _mean(len(sample.traces) for sample in samples),
        "avg_trace_length": _mean(trace_lengths),
        "max_petri_nodes": max((sample.petri_graph.num_nodes for sample in samples), default=0),
        "max_petri_edges": max((sample.petri_graph.num_edges for sample in samples), default=0),
    }


def recreate_data_splits(
    data_dir: str | Path,
    counts: SplitCounts | None = None,
    config: SyntheticConfig | None = None,
    seed: int = 13,
) -> dict[str, object]:
    data_dir = Path(data_dir)
    counts = counts or SplitCounts()
    config = config or SyntheticConfig()
    rng = random.Random(seed)

    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)

    split_metadata: dict[str, object] = {}
    for split, count in counts.items():
        samples = [
            generate_sample(config=config, rng=rng, equivalence_id=f"{split}-{idx}")
            for idx in range(count)
        ]
        path = split_samples_path(data_dir, split)
        write_samples_jsonl(path, samples)
        split_metadata[split] = {
            "path": str(path.relative_to(data_dir)),
            "statistics": sample_statistics(samples),
        }

    metadata = {
        "version": 1,
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


def _mean(values: Sequence[float] | Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))
