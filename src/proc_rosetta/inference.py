"""Public external-inference API for ProcRosetta.

The functions here are intentionally UI-agnostic.  Streamlit pages and thin
command-line wrappers share the same trusted-checkpoint, preprocessing,
encoding, progressive decoding, validation, and comparison behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch

from proc_rosetta.artifact_io import (
    ArtifactModality,
    ParsedArtifact,
    PreparedArtifact,
    PreprocessingSettings,
    event_log_statistics,
    parse_artifact,
    prepare_artifact_for_model as prepare_parsed_artifact,
)
from proc_rosetta.behavior import behavioral_distance
from proc_rosetta.devices import resolve_device
from proc_rosetta.models import CompletionPolicy, DecodeConstraints, LatentDistribution
from proc_rosetta.pm4py_bridge import (
    TREE_NORMALIZATION_VERSION,
    PetriGraph,
    PetriNetBundle,
    fold_process_tree,
    simulate_traces,
    tree_to_petri_net,
)
from proc_rosetta.training import load_checkpoint
from proc_rosetta.tree import NodeKind, ProcessTreeNode, sanitize_activity_labels


PETRI_LABEL_WARNING = (
    "Legacy checkpoints may not have learned Petri transition-label embeddings; "
    "retrain with the deterministic label-aware architecture before interpreting "
    "activity-copy quality."
)


@dataclass(frozen=True)
class CheckpointMetadata:
    identifier: str
    path: str
    filename: str
    checkpoint_type: str
    epoch: int | None
    latent_dimension: int
    hidden_dimension: int
    maximum_activities: int
    maximum_tree_arity: int
    maximum_tree_token_length: int | None
    maximum_petri_nodes: int | None
    maximum_traces: int | None
    maximum_trace_length: int | None
    best_validation_loss: float | None
    is_best: bool
    training_timestamp: str
    training_configuration: dict[str, Any]
    synthetic_configuration: dict[str, Any]
    history: list[dict[str, Any]]
    petri_label_embeddings_trained: bool


@dataclass
class LoadedCheckpoint:
    model: Any
    device: torch.device
    metadata: CheckpointMetadata
    raw: dict[str, Any]


@dataclass
class ArtifactEncodingResult:
    artifact_id: str
    artifact_name: str
    modality: ArtifactModality
    checkpoint_identifier: str
    source_metadata: dict[str, Any]
    preprocessing_metadata: dict[str, Any]
    canonical_mapping: dict[str, str]
    model_input_summary: dict[str, Any]
    mu: list[float]
    logvar: list[float]
    attention_weights: list[float] | None
    embedding_seconds: float
    source_activity_labels: list[str] = field(default_factory=list)
    source_canonical_activity_labels: list[str] = field(default_factory=list)
    allowed_activity_slots: list[bool] = field(default_factory=list)
    copy_activity_slots: list[bool] = field(default_factory=list)
    activity_memory: list[list[float]] | None = None
    process_group: str = ""
    encoded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def dimension(self) -> int:
        return len(self.mu)

    @property
    def latent_spread(self) -> float:
        if not self.logvar:
            return 0.0
        return float(np.exp(0.5 * np.asarray(self.logvar, dtype=float)).mean())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["modality"] = self.modality.value
        return data


@dataclass(frozen=True)
class DecodeStep:
    step_index: int
    grammar_state: str
    valid_next_tokens: tuple[str, ...]
    chosen_token_id: int
    chosen_token: str
    chosen_token_score: float
    top_valid_token_scores: tuple[tuple[str, float], ...]
    invalid_high_scoring_tokens: tuple[tuple[str, float], ...]
    current_prefix: tuple[str, ...]
    open_child_slots: int
    subtree_position: str
    eos_emitted: bool
    budget_mask_active: bool = False
    argmax_overridden: bool = False
    operators_removed: int = 0
    arities_removed: int = 0
    pre_budget_top_token: str = ""
    pre_budget_top_probability: float = 0.0
    selected_pre_budget_probability: float = 0.0
    selected_conditional_probability: float = 0.0
    completion_slack: int | None = None


@dataclass
class DecodeResult:
    source_artifact_ids: list[str]
    source_modalities: list[ArtifactModality]
    latent_source: str
    latent_vector: list[float]
    raw_token_ids: list[int]
    raw_token_names: list[str]
    raw_tree: ProcessTreeNode | None
    model_normalized_token_ids: list[int]
    model_normalized_token_names: list[str]
    steps: list[DecodeStep]
    eos_emitted: bool
    grammar_valid: bool
    arity_valid: bool
    vocabulary_valid: bool
    length_limit_reached: bool
    tree: ProcessTreeNode | None
    restored_tree: ProcessTreeNode | None
    petri_convertible: bool
    petri_net: PetriNetBundle | None
    restored_label_mapping: dict[str, str]
    unmapped_labels: list[str]
    decode_seconds: float
    out_of_source_activities_replaced: int = 0
    duplicate_activities_replaced: int = 0
    source_alphabet_respected: bool = True
    duplicate_free: bool = True
    output_fold_changed: bool = False
    output_fold_idempotent: bool = False
    budget_intervention_steps: int = 0
    argmax_override_steps: int = 0
    first_budget_intervention_step: int | None = None
    minimum_completion_slack: int | None = None
    operators_removed: int = 0
    arities_removed: int = 0
    raw_unresolved_open_slots: int | None = None
    decoder_configuration: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def successful(self) -> bool:
        return self.eos_emitted and self.grammar_valid and self.petri_convertible

    @property
    def token_ids(self) -> list[int]:
        """Compatibility alias for the raw decoder token stream."""

        return self.raw_token_ids

    @property
    def token_names(self) -> list[str]:
        """Compatibility alias for the raw decoder token names."""

        return self.raw_token_names


def list_trusted_checkpoints(directory: str | Path) -> list[Path]:
    root = Path(directory).expanduser().resolve()
    if not root.exists():
        return []
    return sorted(
        (path for path in root.rglob("*.pt") if path.is_file()),
        key=lambda path: (
            "best" not in path.stem.lower(),
            len(path.relative_to(root).parts) > 1,
            path.relative_to(root).as_posix().lower(),
        ),
    )


def load_trusted_checkpoint(
    checkpoint_path: str | Path,
    *,
    trusted_directory: str | Path,
    device: str | torch.device | None = None,
) -> LoadedCheckpoint:
    """Load a checkpoint only after proving it is inside a trusted directory."""

    root = Path(trusted_directory).expanduser().resolve()
    path = Path(checkpoint_path).expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"checkpoint is outside trusted directory {root}: {path}") from exc
    if path.suffix.lower() != ".pt":
        raise ValueError("trusted checkpoint must use the .pt extension")
    torch_device = resolve_device(device)
    model, raw = load_checkpoint(path, torch_device)
    return LoadedCheckpoint(
        model=model,
        device=torch_device,
        metadata=checkpoint_metadata(path, model, raw),
        raw=raw,
    )


def checkpoint_metadata(path: str | Path, model: Any, raw: dict[str, Any]) -> CheckpointMetadata:
    path = Path(path)
    train = dict(raw.get("train_config", {}))
    synthetic = dict(raw.get("synthetic_config", {}))
    identifier = f"{path.name}:{_file_digest(path)[:12]}"
    state = raw.get("model_state_dict", {})
    petri_label_weight = (
        state.get("petri_encoder.transition_label_embedding.weight")
        if isinstance(state, dict)
        else None
    )
    expected_petri_label_shape = tuple(
        model.petri_encoder.transition_label_embedding.weight.shape
    )
    latent_dimension = int(model.tree_encoder.projection.projection[-1].out_features)
    hidden_dimension = int(model.tree_encoder.embedding.embedding_dim)
    return CheckpointMetadata(
        identifier=identifier,
        path=str(path.resolve()),
        filename=path.name,
        checkpoint_type="best" if raw.get("is_best") or "best" in path.stem.lower() else "latest",
        epoch=_optional_int(raw.get("epoch")),
        latent_dimension=latent_dimension,
        hidden_dimension=hidden_dimension,
        maximum_activities=int(model.tree_tokenizer.max_activities),
        maximum_tree_arity=int(model.tree_tokenizer.max_arity),
        maximum_tree_token_length=_first_int(synthetic, "max_tree_tokens", "max_tree_length") or 512,
        maximum_petri_nodes=_first_int(synthetic, "max_petri_nodes") or 512,
        maximum_traces=_first_int(synthetic, "traces_per_sample", "max_traces"),
        maximum_trace_length=_first_int(synthetic, "max_trace_length") or 128,
        best_validation_loss=_optional_float(raw.get("best_validation_loss")),
        is_best=bool(raw.get("is_best", False)),
        training_timestamp=datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        training_configuration=train,
        synthetic_configuration=synthetic,
        history=list(raw.get("history", [])),
        petri_label_embeddings_trained=(
            isinstance(petri_label_weight, torch.Tensor)
            and tuple(petri_label_weight.shape) == expected_petri_label_shape
        ),
    )


def prepare_artifact_for_model(
    artifact: ParsedArtifact,
    model: Any,
    settings: PreprocessingSettings | None = None,
) -> PreparedArtifact:
    return prepare_parsed_artifact(
        artifact,
        max_activities=int(model.tree_tokenizer.max_activities),
        max_arity=int(model.tree_tokenizer.max_arity),
        settings=settings,
    )


@torch.no_grad()
def encode_artifact(
    prepared: PreparedArtifact,
    checkpoint: LoadedCheckpoint,
) -> ArtifactEncodingResult:
    start = perf_counter()
    warnings = list(prepared.warnings)
    if (
        prepared.parsed.modality is ArtifactModality.PETRI_NET
        and not checkpoint.metadata.petri_label_embeddings_trained
    ):
        warnings.append(PETRI_LABEL_WARNING)
    errors = list(prepared.errors)
    mu: list[float] = []
    logvar: list[float] = []
    attention: list[float] | None = None
    copy_activity_slots: list[bool] = []
    activity_memory: list[list[float]] | None = None
    if prepared.ready:
        try:
            if prepared.parsed.modality is ArtifactModality.PROCESS_TREE:
                tokens = checkpoint.model.tree_tokenizer.encode_tree(
                    prepared.model_input, canonicalize=False
                )
                tensor = torch.tensor([tokens], dtype=torch.long, device=checkpoint.device)
                distribution = checkpoint.model.encode_tree(tensor)
            elif prepared.parsed.modality is ArtifactModality.EVENT_LOG:
                trace_tensors = trace_collection_to_tensors(
                    checkpoint.model,
                    prepared.model_input,
                    checkpoint.device,
                )
                distribution, weights = checkpoint.model.trace_encoder.forward_with_attention(
                    trace_tensors["tokens"],
                    trace_tensors["lengths"],
                    trace_tensors["mask"],
                )
                attention = _tensor_row(weights)
            else:
                graph_tensors = petri_graph_to_tensors(
                    prepared.model_input,
                    checkpoint.device,
                    checkpoint.model.activity_tokenizer,
                )
                distribution = checkpoint.model.encode_petri(graph_tensors)
            mu = _tensor_row(distribution.mu)
            logvar = _tensor_row(distribution.logvar)
            if distribution.activity_mask is not None:
                copy_activity_slots = [
                    bool(value) for value in distribution.activity_mask.detach().cpu()[0].tolist()
                ]
            if distribution.activity_memory is not None:
                activity_memory = [
                    [float(value) for value in row]
                    for row in distribution.activity_memory.detach().cpu()[0].tolist()
                ]
        except Exception as exc:  # retain a complete result for workspace diagnostics
            errors.append(f"{type(exc).__name__}: {exc}")
    return ArtifactEncodingResult(
        artifact_id=prepared.parsed.artifact_id,
        artifact_name=prepared.parsed.display_name,
        modality=prepared.parsed.modality,
        checkpoint_identifier=checkpoint.metadata.identifier,
        source_metadata=dict(prepared.parsed.source_metadata),
        preprocessing_metadata=dict(prepared.preprocessing_metadata),
        canonical_mapping=dict(prepared.canonical_mapping),
        model_input_summary=dict(prepared.model_input_summary),
        mu=mu,
        logvar=logvar,
        attention_weights=attention,
        embedding_seconds=perf_counter() - start,
        source_activity_labels=list(prepared.canonical_mapping),
        source_canonical_activity_labels=list(prepared.canonical_mapping.values()),
        allowed_activity_slots=[
            f"A{index}" in set(prepared.canonical_mapping.values())
            for index in range(checkpoint.model.tree_tokenizer.max_activities)
        ],
        copy_activity_slots=copy_activity_slots,
        activity_memory=activity_memory,
        warnings=warnings,
        errors=errors,
    )


def trace_collection_to_tensors(
    model: Any,
    traces: Sequence[Sequence[str]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not traces:
        raise ValueError("at least one selected trace is required")
    width = max(1, max((len(trace) for trace in traces), default=1))
    tokens = torch.full(
        (1, len(traces), width),
        model.activity_tokenizer.pad_id,
        dtype=torch.long,
        device=device,
    )
    lengths = torch.zeros((1, len(traces)), dtype=torch.long)
    mask = torch.ones((1, len(traces)), dtype=torch.bool, device=device)
    for index, trace in enumerate(traces):
        encoded = model.activity_tokenizer.encode_trace(trace)
        if encoded:
            tokens[0, index, : len(encoded)] = torch.tensor(encoded, dtype=torch.long, device=device)
        lengths[0, index] = len(encoded)
    return {"tokens": tokens, "lengths": lengths, "mask": mask}


def petri_graph_to_tensors(
    graph: PetriGraph,
    device: torch.device,
    activity_tokenizer: Any | None = None,
) -> dict[str, torch.Tensor]:
    node_count = graph.num_nodes
    node_types = torch.tensor([graph.node_types], dtype=torch.long, device=device)
    node_mask = torch.ones((1, node_count), dtype=torch.bool, device=device)
    markings = torch.zeros((1, node_count, 2), dtype=torch.float32, device=device)
    markings[0, :, 0] = torch.tensor(graph.initial_marking, dtype=torch.float32, device=device)
    markings[0, :, 1] = torch.tensor(graph.final_marking, dtype=torch.float32, device=device)
    transition_label_ids = torch.zeros((1, node_count), dtype=torch.long, device=device)
    edge_sources: list[int] = []
    edge_targets: list[int] = []
    edge_types: list[int] = []
    if activity_tokenizer is not None:
        for index, label in enumerate(graph.transition_labels):
            if label is not None:
                transition_label_ids[0, index] = activity_tokenizer.token_to_id.get(
                    label, activity_tokenizer.unk_id
                )
    for source, target, edge_type in graph.edges:
        edge_sources.append(source)
        edge_targets.append(target)
        edge_types.append(edge_type)
    return {
        "node_types": node_types,
        "node_mask": node_mask,
        "markings": markings,
        "transition_label_ids": transition_label_ids,
        "edge_index": torch.tensor(
            [edge_sources, edge_targets],
            dtype=torch.long,
            device=device,
        ),
        "edge_types": torch.tensor(edge_types, dtype=torch.long, device=device),
    }


def _activity_slot_tensor(
    values: Sequence[bool] | torch.Tensor | None,
    *,
    max_activities: int,
    device: torch.device,
) -> torch.Tensor | None:
    if values is None:
        return None
    result = torch.as_tensor(values, dtype=torch.bool, device=device)
    if result.ndim == 1:
        result = result.unsqueeze(0)
    if result.ndim != 2 or result.shape[-1] != max_activities:
        raise ValueError(
            f"activity-slot mask must have shape [batch, {max_activities}]"
        )
    return result


def _decoder_source(
    latent: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    copy_activity_slots: Sequence[bool] | torch.Tensor | None = None,
    activity_memory: Sequence[Sequence[float]] | torch.Tensor | None = None,
    max_activities: int,
) -> LatentDistribution:
    mu = _latent_tensor(latent, device)
    copy_mask = _activity_slot_tensor(
        copy_activity_slots,
        max_activities=max_activities,
        device=device,
    )
    memory_tensor = None
    if activity_memory is not None:
        memory_tensor = torch.as_tensor(activity_memory, dtype=mu.dtype, device=device)
        if memory_tensor.ndim == 2:
            memory_tensor = memory_tensor.unsqueeze(0)
        if memory_tensor.ndim != 3 or memory_tensor.shape[1] != max_activities:
            raise ValueError("activity memory has incompatible shape")
    return LatentDistribution(
        mu=mu,
        logvar=torch.zeros_like(mu),
        activity_mask=copy_mask,
        activity_memory=memory_tensor,
    )


@torch.no_grad()
def decode_latent_iter(
    model: Any,
    latent: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    max_length: int = 512,
    top_k: int = 5,
    allowed_activity_slots: Sequence[bool] | torch.Tensor | None = None,
    copy_activity_slots: Sequence[bool] | torch.Tensor | None = None,
    activity_memory: Sequence[Sequence[float]] | torch.Tensor | None = None,
    constrain_to_source_activities: bool = True,
    avoid_duplicate_activity_labels: bool = True,
    duplicate_policy: str = "disallow",
    completion_policy: CompletionPolicy = "bounded",
    chosen_token_ids: Sequence[int] | None = None,
) -> Iterator[DecodeStep]:
    """Yield shared-step diagnostics for greedy or supplied decoder decisions."""

    if completion_policy not in {"prefix_only", "bounded"}:
        raise ValueError("completion_policy must be 'prefix_only' or 'bounded'")
    minimum_length = 3 if completion_policy == "bounded" else 2
    if max_length < minimum_length:
        raise ValueError(f"{completion_policy} completion requires max_length >= {minimum_length}")
    device = next(model.parameters()).device
    tokenizer = model.tree_tokenizer
    source = _decoder_source(
        latent,
        device=device,
        copy_activity_slots=copy_activity_slots,
        activity_memory=activity_memory,
        max_activities=tokenizer.max_activities,
    )
    allowed = _activity_slot_tensor(
        allowed_activity_slots,
        max_activities=tokenizer.max_activities,
        device=device,
    )
    if not constrain_to_source_activities:
        allowed = None
    prefix_ids = [tokenizer.bos_id]
    prefix_names = [tokenizer.tokens[tokenizer.bos_id]]
    used_activity_mask = torch.zeros(
        (1, tokenizer.max_activities),
        dtype=torch.bool,
        device=device,
    )
    memory, copy_mask = model.tree_decoder.source_memory(source)
    caches: list[torch.Tensor | None] = [None] * len(model.tree_decoder.decoder.layers)
    open_nodes = torch.ones(1, dtype=torch.long, device=device)
    pending_operator = torch.zeros_like(open_nodes)
    current = torch.tensor([tokenizer.bos_id], dtype=torch.long, device=device)

    for step_index in range(1, max_length):
        if chosen_token_ids is not None and step_index > len(chosen_token_ids):
            break
        remaining_tokens = max_length - step_index
        scores, caches = model.tree_decoder._incremental_next_token_scores(
            current,
            step_index - 1,
            memory,
            caches,
            open_nodes=open_nodes,
            pending_operator=pending_operator,
            activity_mask=copy_mask,
            activity_memory=source.activity_memory,
            allowed_activity_mask=allowed,
            avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
            duplicate_policy=duplicate_policy,
            used_activity_mask=used_activity_mask,
            completion_policy=completion_policy,
            remaining_tokens=remaining_tokens,
        )
        mask = scores.effective_mask[0]
        valid_ids = torch.where(mask)[0]
        base_probabilities = scores.base_log_probs[0].exp()
        conditional_probabilities = torch.softmax(scores.logits[0, valid_ids], dim=0)
        chosen_id = (
            int(scores.search_scores[0].argmax().item())
            if chosen_token_ids is None
            else int(chosen_token_ids[step_index - 1])
        )
        if not 0 <= chosen_id < tokenizer.vocab_size or not bool(mask[chosen_id]):
            raise ValueError(
                f"supplied token {chosen_id} is infeasible at decode step {step_index}"
            )
        selected_index = (valid_ids == chosen_id).nonzero(as_tuple=True)[0][0]
        chosen_probability = float(base_probabilities[chosen_id].item())
        chosen_conditional_probability = float(
            conditional_probabilities[selected_index].item()
        )
        valid_pairs = sorted(
            (
                (tokenizer.tokens[int(token_id)], float(probability))
                for token_id, probability in zip(
                    valid_ids.tolist(), base_probabilities[valid_ids].tolist()
                )
            ),
            key=lambda item: item[1],
            reverse=True,
        )[: max(1, top_k)]
        invalid_ids = torch.where(~mask)[0]
        invalid_pairs = sorted(
            (
                (tokenizer.tokens[int(token_id)], float(base_probabilities[token_id]))
                for token_id in invalid_ids.tolist()
            ),
            key=lambda item: item[1],
            reverse=True,
        )[: max(1, top_k)]
        grammar_state = (
            "need_arity"
            if int(pending_operator[0])
            else "complete" if int(open_nodes[0]) == 0 else "need_node"
        )
        open_slots = int(open_nodes[0])
        removed = scores.prefix_grammar_mask[0] & ~scores.completion_mask[0]
        budget_mask_active = bool(removed.any())
        pre_budget_top_id = int(scores.base_log_probs[0].argmax().item())
        argmax_overridden = bool(
            scores.prefix_grammar_mask[0, pre_budget_top_id]
            and not scores.completion_mask[0, pre_budget_top_id]
        )
        operators_removed = sum(
            bool(removed[tokenizer.token_to_id[token]])
            for token in tokenizer.operator_tokens
        )
        arities_removed = sum(
            bool(removed[tokenizer.token_to_id[token]])
            for token in tokenizer.arity_tokens
        )
        chosen_name = tokenizer.tokens[chosen_id]
        prefix_ids.append(chosen_id)
        prefix_names.append(chosen_name)
        chosen_slot = torch.where(model.tree_decoder.activity_token_ids.eq(chosen_id))[0]
        if chosen_slot.numel():
            used_activity_mask[0, int(chosen_slot[0])] = True
        chosen_tensor = torch.tensor([chosen_id], dtype=torch.long, device=device)
        model.tree_decoder._advance_incremental_grammar(
            chosen_tensor,
            open_nodes,
            pending_operator,
            torch.ones(1, dtype=torch.bool, device=device),
        )
        completion_slack = remaining_tokens - 1
        if chosen_id != tokenizer.eos_id:
            completion_slack -= int(
                tokenizer.minimum_tokens_to_finish(
                    open_nodes,
                    pending_operator,
                )[0]
            )
        yield DecodeStep(
            step_index=step_index,
            grammar_state=grammar_state,
            valid_next_tokens=tuple(tokenizer.tokens[int(token_id)] for token_id in valid_ids),
            chosen_token_id=chosen_id,
            chosen_token=chosen_name,
            chosen_token_score=chosen_probability,
            top_valid_token_scores=tuple(valid_pairs),
            invalid_high_scoring_tokens=tuple(invalid_pairs),
            current_prefix=tuple(prefix_names),
            open_child_slots=int(open_slots),
            subtree_position=_next_subtree_position(prefix_names[:-1]),
            eos_emitted=chosen_id == tokenizer.eos_id,
            budget_mask_active=budget_mask_active,
            argmax_overridden=argmax_overridden,
            operators_removed=operators_removed,
            arities_removed=arities_removed,
            pre_budget_top_token=tokenizer.tokens[pre_budget_top_id],
            pre_budget_top_probability=float(base_probabilities[pre_budget_top_id]),
            selected_pre_budget_probability=chosen_probability,
            selected_conditional_probability=chosen_conditional_probability,
            completion_slack=completion_slack,
        )
        if chosen_id == tokenizer.eos_id:
            break
        current = chosen_tensor


def decode_latent(
    checkpoint: LoadedCheckpoint,
    latent: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    source_artifact_ids: Sequence[str],
    source_modalities: Sequence[ArtifactModality],
    latent_source: str,
    canonical_mapping: dict[str, str] | None = None,
    max_length: int = 512,
    top_k: int = 5,
    beam_size: int = 5,
    allowed_activity_slots: Sequence[bool] | torch.Tensor | None = None,
    copy_activity_slots: Sequence[bool] | torch.Tensor | None = None,
    activity_memory: Sequence[Sequence[float]] | torch.Tensor | None = None,
    constrain_to_source_activities: bool = True,
    avoid_duplicate_activity_labels: bool = True,
    duplicate_policy: str = "disallow",
    completion_policy: CompletionPolicy = "bounded",
    progress_callback: Callable[[DecodeStep], None] | None = None,
) -> DecodeResult:
    start = perf_counter()
    steps: list[DecodeStep] = []
    if allowed_activity_slots is None and canonical_mapping is not None:
        canonical_labels = set(canonical_mapping.values())
        allowed_activity_slots = [
            f"A{index}" in canonical_labels
            for index in range(checkpoint.model.tree_tokenizer.max_activities)
        ]
    allowed = _activity_slot_tensor(
        allowed_activity_slots,
        max_activities=checkpoint.model.tree_tokenizer.max_activities,
        device=checkpoint.device,
    )
    constraints = DecodeConstraints(
        allowed_activity_slots=allowed,
        constrain_to_source_activities=constrain_to_source_activities,
        avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
        duplicate_policy=duplicate_policy,
        completion_policy=completion_policy,
    )
    if beam_size > 1 and progress_callback is None:
        source = _decoder_source(
            latent,
            device=checkpoint.device,
            copy_activity_slots=copy_activity_slots,
            activity_memory=activity_memory,
            max_activities=checkpoint.model.tree_tokenizer.max_activities,
        )
        decoded = checkpoint.model.tree_decoder.decode_beam(
            source,
            max_length=max_length,
            beam_size=beam_size,
            length_penalty=0.7,
            constraints=constraints,
        )
        token_ids = decoded[0].detach().cpu().tolist()
        chosen_ids: list[int] = []
        for token_id in token_ids[1:]:
            if token_id == checkpoint.model.tree_tokenizer.pad_id:
                break
            chosen_ids.append(token_id)
            if token_id == checkpoint.model.tree_tokenizer.eos_id:
                break
        steps = list(
            decode_latent_iter(
                checkpoint.model,
                latent,
                max_length=max_length,
                top_k=top_k,
                allowed_activity_slots=allowed,
                copy_activity_slots=copy_activity_slots,
                activity_memory=activity_memory,
                constrain_to_source_activities=constrain_to_source_activities,
                avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
                duplicate_policy=duplicate_policy,
                completion_policy=completion_policy,
                chosen_token_ids=chosen_ids,
            )
        )
    else:
        for step in decode_latent_iter(
            checkpoint.model,
            latent,
            max_length=max_length,
            top_k=top_k,
            allowed_activity_slots=allowed,
            copy_activity_slots=copy_activity_slots,
            activity_memory=activity_memory,
            constrain_to_source_activities=constrain_to_source_activities,
            avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
            duplicate_policy=duplicate_policy,
            completion_policy=completion_policy,
        ):
            steps.append(step)
            if progress_callback is not None:
                progress_callback(step)
        token_ids = [checkpoint.model.tree_tokenizer.bos_id]
        token_ids.extend(step.chosen_token_id for step in steps)
    return build_decode_result(
        checkpoint.model,
        latent,
        source_artifact_ids=source_artifact_ids,
        source_modalities=source_modalities,
        latent_source=latent_source,
        token_ids=token_ids,
        steps=steps,
        canonical_mapping=canonical_mapping,
        allowed_activity_slots=allowed,
        constrain_to_source_activities=constrain_to_source_activities,
        avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
        duplicate_policy=duplicate_policy,
        completion_policy=completion_policy,
        token_strategy="beam" if beam_size > 1 and progress_callback is None else "greedy",
        beam_size=beam_size,
        max_length=max_length,
        decode_seconds=perf_counter() - start,
    )


def build_decode_result(
    model: Any,
    latent: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    source_artifact_ids: Sequence[str],
    source_modalities: Sequence[ArtifactModality],
    latent_source: str,
    token_ids: Sequence[int],
    steps: Sequence[DecodeStep] = (),
    canonical_mapping: dict[str, str] | None = None,
    allowed_activity_slots: Sequence[bool] | torch.Tensor | None = None,
    constrain_to_source_activities: bool = True,
    avoid_duplicate_activity_labels: bool = True,
    duplicate_policy: str = "disallow",
    completion_policy: CompletionPolicy = "prefix_only",
    token_strategy: str = "greedy",
    beam_size: int = 1,
    max_length: int = 512,
    decode_seconds: float = 0.0,
) -> DecodeResult:
    tokenizer = model.tree_tokenizer
    ids = [int(token_id) for token_id in token_ids]
    names = [
        tokenizer.tokens[token_id]
        if 0 <= token_id < tokenizer.vocab_size
        else f"<invalid:{token_id}>"
        for token_id in ids
    ]
    eos = tokenizer.eos_id in ids
    errors: list[str] = []
    warnings: list[str] = []
    raw_tree: ProcessTreeNode | None = None
    tree: ProcessTreeNode | None = None
    bundle: PetriNetBundle | None = None
    grammar_valid = False
    arity_valid = False
    vocabulary_valid = all(0 <= token_id < tokenizer.vocab_size for token_id in ids)
    if vocabulary_valid:
        try:
            raw_tree = tokenizer.decode_tree(ids)
            grammar_valid = True
            arity_valid = _arity_valid(raw_tree, tokenizer.max_arity)
        except Exception as exc:
            errors.append(f"Process-tree parse failed: {type(exc).__name__}: {exc}")
    else:
        errors.append("Process-tree token stream contains IDs outside the vocabulary.")

    allowed_mask: list[bool] | None
    if allowed_activity_slots is not None:
        allowed_mask = [
            bool(value)
            for value in torch.as_tensor(allowed_activity_slots).reshape(-1).tolist()
        ]
        if len(allowed_mask) != tokenizer.max_activities:
            raise ValueError("allowed activity slots have incompatible length")
    elif canonical_mapping is not None:
        canonical_labels = set(canonical_mapping.values())
        allowed_mask = [
            f"A{index}" in canonical_labels
            for index in range(tokenizer.max_activities)
        ]
    else:
        allowed_mask = None
    allowed_labels = (
        {
            f"A{index}"
            for index, is_allowed in enumerate(allowed_mask)
            if is_allowed
        }
        if constrain_to_source_activities and allowed_mask is not None
        else None
    )
    out_of_source_replaced = 0
    duplicates_replaced = 0
    output_fold_changed = False
    output_fold_idempotent = False
    model_normalized_ids: list[int] = []
    model_normalized_names: list[str] = []
    if raw_tree is not None and grammar_valid and arity_valid:
        sanitized = sanitize_activity_labels(
            raw_tree,
            allowed_labels=allowed_labels,
            avoid_duplicates=(
                avoid_duplicate_activity_labels and duplicate_policy == "disallow"
            ),
        )
        out_of_source_replaced = sanitized.out_of_source_activities_replaced
        duplicates_replaced = sanitized.duplicate_activities_replaced
        try:
            tree = fold_process_tree(sanitized.tree)
            output_fold_changed = (
                tree.canonical_key() != sanitized.tree.canonical_key()
            )
            refolded = fold_process_tree(tree)
            output_fold_idempotent = refolded.canonical_key() == tree.canonical_key()
            model_normalized_ids = tokenizer.encode_tree(tree, canonicalize=False)
            model_normalized_names = [tokenizer.tokens[token_id] for token_id in model_normalized_ids]
        except Exception as exc:
            errors.append(f"Output normalization failed: {type(exc).__name__}: {exc}")
    if out_of_source_replaced:
        warnings.append(
            f"Replaced {out_of_source_replaced} out-of-source visible activit"
            f"{'y' if out_of_source_replaced == 1 else 'ies'} with tau."
        )
    if duplicates_replaced:
        warnings.append(
            f"Replaced {duplicates_replaced} repeated visible activity "
            f"occurrence{'s' if duplicates_replaced != 1 else ''} with tau."
        )
    length_limit = not eos and len(ids) >= max_length
    if not eos:
        warnings.append("Decoder did not emit <eos>; this is not a fully successful decode.")
    modalities = [ArtifactModality(item) for item in source_modalities]
    inverse = {canonical: original for original, canonical in (canonical_mapping or {}).items()}
    restored_tree = tree.relabel(inverse) if tree is not None and inverse else tree
    generated_labels = set(tree.activity_labels()) if tree is not None else set()
    unmapped = (
        sorted(label for label in generated_labels if label not in inverse)
        if inverse
        else []
    )
    if inverse and unmapped:
        warnings.append(
            "Generated canonical labels without source mappings were kept unchanged: "
            + ", ".join(unmapped)
        )
    if restored_tree is not None:
        try:
            bundle = tree_to_petri_net(restored_tree)
        except Exception as exc:
            errors.append(f"Petri-net conversion failed: {type(exc).__name__}: {exc}")
    step_rows = list(steps)
    intervention_rows = [step for step in step_rows if step.budget_mask_active]
    slack_values = [
        step.completion_slack
        for step in step_rows
        if step.completion_slack is not None
    ]
    unresolved_open_slots: int | None = None
    if vocabulary_valid:
        try:
            grammar_state, pending_operator, open_nodes = tokenizer._grammar_state(ids)
            if grammar_state.value != "invalid":
                unresolved_open_slots = int(open_nodes)
                if pending_operator is not None:
                    unresolved_open_slots += tokenizer.minimum_legal_arity(pending_operator)
        except Exception:
            unresolved_open_slots = None
    return DecodeResult(
        source_artifact_ids=list(source_artifact_ids),
        source_modalities=modalities,
        latent_source=latent_source,
        latent_vector=_latent_list(latent),
        raw_token_ids=ids,
        raw_token_names=names,
        raw_tree=raw_tree,
        model_normalized_token_ids=model_normalized_ids,
        model_normalized_token_names=model_normalized_names,
        steps=step_rows,
        eos_emitted=eos,
        grammar_valid=grammar_valid,
        arity_valid=arity_valid,
        vocabulary_valid=vocabulary_valid,
        length_limit_reached=length_limit,
        tree=tree,
        restored_tree=restored_tree,
        petri_convertible=bundle is not None,
        petri_net=bundle,
        restored_label_mapping=inverse,
        unmapped_labels=unmapped,
        decode_seconds=decode_seconds,
        out_of_source_activities_replaced=out_of_source_replaced,
        duplicate_activities_replaced=duplicates_replaced,
        source_alphabet_respected=(
            allowed_labels is None or generated_labels.issubset(allowed_labels)
        ),
        duplicate_free=(
            tree is None
            or len(tree.activity_labels()) == len(set(tree.activity_labels()))
        ),
        output_fold_changed=output_fold_changed,
        output_fold_idempotent=output_fold_idempotent,
        budget_intervention_steps=len(intervention_rows),
        argmax_override_steps=sum(step.argmax_overridden for step in step_rows),
        first_budget_intervention_step=(
            intervention_rows[0].step_index if intervention_rows else None
        ),
        minimum_completion_slack=(min(slack_values) if slack_values else None),
        operators_removed=sum(step.operators_removed for step in step_rows),
        arities_removed=sum(step.arities_removed for step in step_rows),
        raw_unresolved_open_slots=unresolved_open_slots,
        decoder_configuration={
            "maximum_token_length": max_length,
            "token_strategy": token_strategy,
            "beam_size": beam_size,
            "grammar_mask": True,
            "completion_policy": completion_policy,
            "constrain_to_source_activities": constrain_to_source_activities,
            "avoid_duplicate_activity_labels": avoid_duplicate_activity_labels,
            "duplicate_policy": duplicate_policy,
            "allowed_activity_slots": allowed_mask,
            "normalization_version": TREE_NORMALIZATION_VERSION,
            "latent_source": latent_source,
        },
        warnings=warnings,
        errors=errors,
    )


def validate_decoded_tree(result: DecodeResult) -> dict[str, bool]:
    return {
        "eos_emitted": result.eos_emitted,
        "grammar_valid": result.grammar_valid,
        "arity_valid": result.arity_valid,
        "vocabulary_valid": result.vocabulary_valid,
        "petri_convertible": result.petri_convertible,
        "label_restoration_complete": not result.unmapped_labels,
        "length_limit_reached": result.length_limit_reached,
        "source_alphabet_respected": result.source_alphabet_respected,
        "duplicate_free": result.duplicate_free,
        "output_fold_idempotent": result.output_fold_idempotent,
    }


def combine_encoding_decode_evidence(
    encodings: Sequence[ArtifactEncodingResult],
) -> tuple[list[bool] | None, list[bool] | None, list[list[float]] | None]:
    """Union source alphabets and average contextual copy evidence by slot."""

    slot_count = max((len(item.allowed_activity_slots) for item in encodings), default=0)
    if slot_count == 0:
        return None, None, None
    allowed = [False] * slot_count
    copy = [False] * slot_count
    memory_rows: list[list[np.ndarray]] = [[] for _ in range(slot_count)]
    for encoding in encodings:
        for index, value in enumerate(encoding.allowed_activity_slots):
            allowed[index] |= bool(value)
        for index, value in enumerate(encoding.copy_activity_slots[:slot_count]):
            copy[index] |= bool(value)
        if encoding.activity_memory is not None:
            for index, row in enumerate(encoding.activity_memory[:slot_count]):
                if index < len(encoding.copy_activity_slots) and encoding.copy_activity_slots[index]:
                    memory_rows[index].append(np.asarray(row, dtype=np.float32))
    if not any(copy) or not any(memory_rows):
        return allowed, copy, None
    hidden_dim = next(rows[0].size for rows in memory_rows if rows)
    memory = [
        np.stack(rows).mean(axis=0).astype(float).tolist()
        if rows
        else [0.0] * hidden_dim
        for rows in memory_rows
    ]
    return allowed, copy, memory


def convert_tree_to_petri(tree: ProcessTreeNode) -> PetriNetBundle:
    return tree_to_petri_net(tree)


def simulate_decoded_behavior(
    result: DecodeResult,
    *,
    num_traces: int = 100,
    random_seed: int = 13,
) -> tuple[tuple[str, ...], ...]:
    if result.tree is None or not result.grammar_valid:
        raise ValueError("a grammar-valid decoded tree is required for simulation")
    import random

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(random_seed)
        np.random.seed(random_seed)
        return tuple(
            tuple(trace)
            for trace in simulate_traces(result.restored_tree or result.tree, num_traces)
        )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def compare_source_and_decoded(
    source: ParsedArtifact,
    result: DecodeResult,
    *,
    simulated_traces: int = 100,
) -> dict[str, Any]:
    if result.tree is None:
        return {"available": False, "reason": "decoded tree is invalid"}
    if source.modality is ArtifactModality.PROCESS_TREE:
        assert source.tree is not None
        source_tokens = source.tree.canonicalize_activity_labels().to_prefix_tokens()
        decoded_tokens = result.tree.to_prefix_tokens()
        distance = levenshtein_distance(source_tokens, decoded_tokens)
        denominator = max(len(source_tokens), len(decoded_tokens), 1)
        source_operators = [token for token in source_tokens if token in {"SEQ", "XOR", "AND", "LOOP"}]
        decoded_operators = [token for token in decoded_tokens if token in {"SEQ", "XOR", "AND", "LOOP"}]
        source_activities = [token for token in source_tokens if token.startswith("A") and token[1:].isdigit()]
        decoded_activities = [token for token in decoded_tokens if token.startswith("A") and token[1:].isdigit()]
        return {
            "available": True,
            "kind": "tree",
            "exact_tree_match": source.tree.canonical_key() == (result.restored_tree or result.tree).canonical_key(),
            "exact_prefix_match": source_tokens == decoded_tokens,
            "token_edit_distance": distance,
            "normalized_token_edit_distance": distance / denominator,
            "source_size": source.tree.size(),
            "decoded_size": result.tree.size(),
            "source_depth": source.tree.max_depth(),
            "decoded_depth": result.tree.max_depth(),
            "operator_accuracy": _multiset_overlap_accuracy(source_operators, decoded_operators),
            "activity_leaf_accuracy": _multiset_overlap_accuracy(source_activities, decoded_activities),
            "operator_count_difference": len(decoded_operators) - len(source_operators),
            "activity_leaf_count_difference": len(decoded_activities) - len(source_activities),
            "source_tokens": source_tokens,
            "decoded_tokens": decoded_tokens,
        }
    if source.modality is ArtifactModality.EVENT_LOG:
        assert source.traces is not None
        simulated = simulate_decoded_behavior(result, num_traces=simulated_traces)
        source_stats = event_log_statistics(source.traces)
        simulated_stats = event_log_statistics(simulated)
        return {
            "available": True,
            "kind": "behavior",
            **behavioral_distance(source.traces, simulated),
            "simulated_trace_count": len(simulated),
            "source_activity_frequencies": source_stats["activity_frequencies"],
            "simulated_activity_frequencies": simulated_stats["activity_frequencies"],
            "source_trace_length_frequencies": source_stats["trace_length_frequencies"],
            "simulated_trace_length_frequencies": simulated_stats[
                "trace_length_frequencies"
            ],
            "source_variant_frequencies": source_stats["variant_frequencies"][:20],
            "simulated_variant_frequencies": simulated_stats["variant_frequencies"][:20],
            "source_directly_follows_frequencies": source_stats[
                "directly_follows_frequencies"
            ][:30],
            "simulated_directly_follows_frequencies": simulated_stats[
                "directly_follows_frequencies"
            ][:30],
        }
    assert source.graph is not None
    decoded_graph = None if result.petri_net is None else result.petri_net.graph
    if decoded_graph is None:
        return {"available": False, "reason": "decoded tree could not be converted to a Petri net"}
    source_visible_labels = {
        label for label in source.graph.transition_labels if label is not None
    }
    restored_visible_labels = set(
        (result.restored_tree or result.tree).activity_labels()
    )
    return {
        "available": True,
        "kind": "petri_structure",
        "source_places": sum(value == 0 for value in source.graph.node_types),
        "decoded_places": sum(value == 0 for value in decoded_graph.node_types),
        "source_transitions": sum(value in {1, 2} for value in source.graph.node_types),
        "decoded_transitions": sum(value in {1, 2} for value in decoded_graph.node_types),
        "source_visible_transitions": sum(value == 1 for value in source.graph.node_types),
        "decoded_visible_transitions": sum(value == 1 for value in decoded_graph.node_types),
        "source_invisible_transitions": sum(value == 2 for value in source.graph.node_types),
        "decoded_invisible_transitions": sum(value == 2 for value in decoded_graph.node_types),
        "source_arcs": source.graph.num_edges,
        "decoded_arcs": decoded_graph.num_edges,
        "source_initial_tokens": int(sum(source.graph.initial_marking)),
        "decoded_initial_tokens": int(sum(decoded_graph.initial_marking)),
        "source_final_tokens": int(sum(source.graph.final_marking)),
        "decoded_final_tokens": int(sum(decoded_graph.final_marking)),
        "source_duplicate_visible_labels": _duplicate_label_count(source.graph.transition_labels),
        "decoded_duplicate_visible_labels": _duplicate_label_count(decoded_graph.transition_labels),
        "label_preservation_evaluated": True,
        "source_visible_labels": sorted(source_visible_labels),
        "decoded_visible_labels": sorted(restored_visible_labels),
        "visible_label_subset_respected": restored_visible_labels.issubset(
            source_visible_labels
        ),
    }


def fuse_latent_means(
    encodings: Sequence[ArtifactEncodingResult],
    weights: Sequence[float] | None = None,
) -> list[float]:
    if not encodings:
        raise ValueError("at least one encoding is required")
    checkpoints = {encoding.checkpoint_identifier for encoding in encodings}
    if len(checkpoints) != 1:
        raise ValueError("embeddings from different checkpoints cannot be fused")
    matrix = np.asarray([encoding.mu for encoding in encodings], dtype=float)
    if len({len(row) for row in matrix}) != 1:
        raise ValueError("embedding dimensions do not match")
    normalized = np.ones(len(encodings), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if normalized.shape != (len(encodings),):
        raise ValueError("one fusion weight is required per encoding")
    if np.any(normalized < 0) or float(normalized.sum()) <= 0:
        raise ValueError("fusion weights must be non-negative and sum to more than zero")
    normalized = normalized / normalized.sum()
    return np.average(matrix, axis=0, weights=normalized).tolist()


def fuse_latent_distributions(
    encodings: Sequence[ArtifactEncodingResult],
    weights: Sequence[float] | None = None,
) -> tuple[list[float], list[float]]:
    """Fuse modality distributions with normalized arithmetic weights.

    This mirrors the model's standard equal-weight mean for ``mu``. Averaging
    ``logvar`` is provided only to drive explicitly experimental latent
    sampling; it must not be presented as calibrated predictive uncertainty.
    """

    mu = fuse_latent_means(encodings, weights)
    normalized = np.ones(len(encodings), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if normalized.shape != (len(encodings),) or np.any(normalized < 0) or normalized.sum() <= 0:
        raise ValueError("fusion weights must be non-negative with one value per encoding")
    normalized = normalized / normalized.sum()
    logvar = np.average(
        np.asarray([encoding.logvar for encoding in encodings], dtype=float),
        axis=0,
        weights=normalized,
    )
    return mu, logvar.tolist()


def sample_latent_distribution(
    mu: Sequence[float],
    logvar: Sequence[float],
    *,
    random_seed: int = 13,
) -> list[float]:
    mu_array = np.asarray(mu, dtype=float)
    logvar_array = np.asarray(logvar, dtype=float)
    if mu_array.shape != logvar_array.shape or mu_array.ndim != 1:
        raise ValueError("mu and logvar must be one-dimensional vectors with matching shapes")
    rng = np.random.default_rng(random_seed)
    epsilon = rng.standard_normal(mu_array.shape)
    return (mu_array + np.exp(0.5 * logvar_array) * epsilon).tolist()


def interpolate_latents(
    left: Sequence[float],
    right: Sequence[float],
    alpha: float,
) -> list[float]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        raise ValueError("latent vectors must have matching shapes")
    return ((1.0 - alpha) * left_array + alpha * right_array).tolist()


def levenshtein_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, 1):
        current = [left_index]
        for right_index, right_value in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _multiset_overlap_accuracy(left: Sequence[str], right: Sequence[str]) -> float:
    from collections import Counter

    left_counts = Counter(left)
    right_counts = Counter(right)
    overlap = sum((left_counts & right_counts).values())
    return overlap / max(len(left), len(right), 1)


def _duplicate_label_count(labels: Sequence[str | None]) -> int:
    from collections import Counter

    counts = Counter(label for label in labels if label is not None)
    return sum(count - 1 for count in counts.values() if count > 1)


def _arity_valid(tree: ProcessTreeNode, maximum: int) -> bool:
    if tree.children:
        if len(tree.children) > maximum:
            return False
        if tree.kind is NodeKind.LOOP and len(tree.children) not in {2, 3}:
            return False
        if tree.kind in {NodeKind.SEQ, NodeKind.XOR, NodeKind.AND} and len(tree.children) < 2:
            return False
    return all(_arity_valid(child, maximum) for child in tree.children)


def _next_subtree_position(prefix_names: Sequence[str]) -> str:
    open_paths: list[tuple[int, ...]] = [()]
    pending_path: tuple[int, ...] | None = None
    for token in prefix_names:
        if token in {"<bos>", "<pad>", "<eos>"}:
            continue
        if token in {"SEQ", "XOR", "AND", "LOOP"}:
            if open_paths:
                pending_path = open_paths.pop(0)
            continue
        if token.startswith("ARITY_") and pending_path is not None:
            arity = int(token.split("_", 1)[1])
            children = [(*pending_path, index) for index in range(arity)]
            open_paths = children + open_paths
            pending_path = None
            continue
        if (token == "TAU" or (token.startswith("A") and token[1:].isdigit())) and open_paths:
            open_paths.pop(0)
    path = pending_path if pending_path is not None else (open_paths[0] if open_paths else None)
    return "complete" if path is None else "root" + "".join(f"/{index}" for index in path)


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_int(values: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in values and values[key] is not None:
            return int(values[key])
    return None


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _tensor_row(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu()[0].tolist()]


def _latent_tensor(
    latent: Sequence[float] | np.ndarray | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(latent, torch.Tensor):
        tensor = latent.detach().to(device=device, dtype=torch.float32)
    else:
        tensor = torch.tensor(latent, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[0] != 1:
        raise ValueError("latent vector must have shape [dimension] or [1, dimension]")
    return tensor


def _latent_list(latent: Sequence[float] | np.ndarray | torch.Tensor) -> list[float]:
    if isinstance(latent, torch.Tensor):
        values = latent.detach().cpu().reshape(-1).tolist()
    else:
        values = np.asarray(latent, dtype=float).reshape(-1).tolist()
    return [float(value) for value in values]
