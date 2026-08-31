from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from proc_rosetta.tokenizers import ActivityTokenizer, TreeTokenizer


CompletionPolicy = Literal["prefix_only", "bounded"]


class DecoderInvariantError(RuntimeError):
    """Raised when a bounded prefix cannot finish inside its token budget."""


def _validate_completion_policy(policy: str) -> None:
    if policy not in {"prefix_only", "bounded"}:
        raise ValueError("completion_policy must be 'prefix_only' or 'bounded'")


def _attention_heads(hidden_dim: int) -> int:
    for heads in (8, 4, 2):
        if hidden_dim % heads == 0:
            return heads
    return 1


@dataclass
class LatentDistribution:
    """Compatibility container for a deterministic semantic encoding."""

    mu: torch.Tensor
    logvar: torch.Tensor
    memory: torch.Tensor | None = None
    pre_normalized: torch.Tensor | None = None
    activity_mask: torch.Tensor | None = None
    activity_memory: torch.Tensor | None = None

    def sample(self, deterministic: bool = True) -> torch.Tensor:
        return self.mu


@dataclass(frozen=True)
class DecodeConstraints:
    """Hard legality policy shared by every autoregressive decode path.

    ``allowed_activity_slots`` is indexed by canonical activity slot (A0,
    A1, ...), not by full tree-vocabulary ID.  ``activity_mask`` on
    :class:`LatentDistribution` remains copy evidence and is intentionally a
    separate concept. ``completion_policy`` is likewise independent of the
    ordinary prefix grammar and defaults to the deployment-safe bounded mode.
    """

    allowed_activity_slots: torch.Tensor | None = None
    constrain_to_source_activities: bool = True
    avoid_duplicate_activity_labels: bool = True
    duplicate_policy: str = "disallow"
    completion_policy: CompletionPolicy = "bounded"

    def __post_init__(self) -> None:
        if self.duplicate_policy not in {"disallow", "penalize", "allow"}:
            raise ValueError("duplicate_policy must be 'disallow', 'penalize', or 'allow'")
        _validate_completion_policy(self.completion_policy)


@dataclass(frozen=True)
class NextTokenScores:
    """Next-token scores with ordinary legality and budget feasibility separate."""

    logits: torch.Tensor
    base_log_probs: torch.Tensor
    search_scores: torch.Tensor
    prefix_grammar_mask: torch.Tensor
    completion_mask: torch.Tensor
    source_constraint_mask: torch.Tensor
    duplicate_constraint_mask: torch.Tensor
    effective_mask: torch.Tensor


class SemanticProjection(nn.Module):
    def __init__(
        self,
        input_dim: int,
        semantic_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, semantic_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        *,
        memory: torch.Tensor,
        activity_mask: torch.Tensor | None,
        activity_memory: torch.Tensor | None = None,
    ) -> LatentDistribution:
        raw = self.projection(features)
        metric = F.normalize(raw, dim=-1)
        return LatentDistribution(
            mu=metric,
            logvar=torch.zeros_like(metric),
            memory=memory,
            pre_normalized=raw,
            activity_mask=activity_mask,
            activity_memory=activity_memory,
        )


def _pool_activity_occurrences(
    features: torch.Tensor,
    token_ids: torch.Tensor,
    activity_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate contextual source states for every visible activity label."""

    occurrence_mask = token_ids.unsqueeze(-1).eq(
        activity_token_ids.to(token_ids.device).view(1, 1, -1)
    )
    counts = occurrence_mask.sum(dim=1)
    evidence = torch.einsum(
        "bnh,bna->bah", features, occurrence_mask.to(features.dtype)
    )
    evidence = evidence / counts.clamp_min(1).unsqueeze(-1).to(features.dtype)
    return evidence, counts.gt(0)


class SeedMemoryPool(nn.Module):
    def __init__(self, hidden_dim: int, memory_tokens: int) -> None:
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(memory_tokens, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            _attention_heads(hidden_dim),
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        source: torch.Tensor,
        source_padding_mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        seeds = self.seeds.unsqueeze(0).expand(source.shape[0], -1, -1)
        memory, weights = self.attention(
            seeds,
            source,
            source,
            key_padding_mask=source_padding_mask,
            need_weights=return_attention,
            average_attn_weights=True,
        )
        return self.norm(memory + seeds), weights if return_attention else None


class TreeEncoder(nn.Module):
    def __init__(
        self,
        tokenizer: TreeTokenizer,
        semantic_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        layers: int = 4,
        memory_tokens: int = 8,
        max_sequence_length: int = 1024,
        projection_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding = nn.Embedding(
            tokenizer.vocab_size, hidden_dim, padding_idx=tokenizer.pad_id
        )
        self.position_embedding = nn.Embedding(max_sequence_length, hidden_dim)
        self.depth_embedding = nn.Embedding(64, hidden_dim)
        self.parent_embedding = nn.Embedding(max_sequence_length, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=_attention_heads(hidden_dim),
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.memory_pool = SeedMemoryPool(hidden_dim, memory_tokens)
        self.projection = SemanticProjection(
            hidden_dim,
            semantic_dim,
            dropout=projection_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.max_sequence_length = max_sequence_length
        activity_ids = [
            tokenizer.token_to_id[name] for name in tokenizer.activity_tokens
        ]
        self.register_buffer(
            "activity_token_ids", torch.tensor(activity_ids, dtype=torch.long)
        )

    def _structure_features(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tokenizer.structure_features(
            tokens.detach().cpu(),
            max_sequence_length=self.max_sequence_length,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        depths: torch.Tensor | None = None,
        parents: torch.Tensor | None = None,
    ) -> LatentDistribution:
        if tokens.shape[1] > self.max_sequence_length:
            raise ValueError("tree token sequence exceeds encoder position capacity")
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        if depths is None or parents is None:
            depths, parents = self._structure_features(tokens)
            depths = depths.to(tokens.device)
            parents = parents.to(tokens.device)
        x = self.dropout(
            self.embedding(tokens)
            + self.position_embedding(positions)
            + self.depth_embedding(depths)
            + self.parent_embedding(parents)
        )
        padding = tokens.eq(self.tokenizer.pad_id)
        encoded = self.encoder(x, src_key_padding_mask=padding)
        memory, _ = self.memory_pool(encoded, padding)
        pooled = memory.mean(dim=1)
        activity_memory, activity_mask = _pool_activity_occurrences(
            encoded,
            tokens,
            self.activity_token_ids,
        )
        return self.projection(
            pooled,
            memory=memory,
            activity_mask=activity_mask,
            activity_memory=activity_memory,
        )


class TraceEncoder(nn.Module):
    def __init__(
        self,
        tokenizer: ActivityTokenizer,
        semantic_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        event_layers: int = 2,
        set_layers: int = 2,
        memory_tokens: int = 8,
        projection_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding = nn.Embedding(
            tokenizer.vocab_size, hidden_dim, padding_idx=tokenizer.pad_id
        )
        direction_dim = max(1, hidden_dim // 2)
        self.event_gru = nn.GRU(
            hidden_dim,
            direction_dim,
            num_layers=event_layers,
            dropout=dropout if event_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        event_dim = 2 * direction_dim
        self.event_projection = (
            nn.Identity() if event_dim == hidden_dim else nn.Linear(event_dim, hidden_dim)
        )
        self.event_attention = nn.Linear(hidden_dim, 1)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=_attention_heads(hidden_dim),
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            layer, num_layers=set_layers, enable_nested_tensor=False
        )
        self.memory_pool = SeedMemoryPool(hidden_dim, memory_tokens)
        self.projection = SemanticProjection(
            hidden_dim,
            semantic_dim,
            dropout=projection_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "activity_token_ids",
            torch.tensor(
                [
                    tokenizer.token_to_id[f"A{index}"]
                    for index in range(tokenizer.max_activities)
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        trace_mask: torch.Tensor,
    ) -> LatentDistribution:
        distribution, _ = self.forward_with_attention(tokens, lengths, trace_mask)
        return distribution

    def forward_with_attention(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        trace_mask: torch.Tensor,
    ) -> tuple[LatentDistribution, torch.Tensor]:
        batch_size, trace_count, trace_length = tokens.shape
        flat_tokens = tokens.reshape(batch_size * trace_count, trace_length)
        flat_lengths = lengths.reshape(-1).clamp(min=1)
        embedded = self.dropout(self.embedding(flat_tokens))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            flat_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _ = self.event_gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=trace_length,
        )
        outputs = self.event_projection(outputs)
        event_positions = torch.arange(trace_length, device=tokens.device).unsqueeze(0)
        event_mask = event_positions < flat_lengths.to(tokens.device).unsqueeze(1)
        event_scores = self.event_attention(outputs).squeeze(-1).masked_fill(
            ~event_mask, -1e9
        )
        event_weights = torch.softmax(event_scores, dim=1)
        trace_vectors = (outputs * event_weights.unsqueeze(-1)).sum(dim=1)
        trace_vectors = trace_vectors.reshape(batch_size, trace_count, -1)
        trace_vectors = trace_vectors * trace_mask.unsqueeze(-1).to(trace_vectors.dtype)
        set_encoded = self.set_encoder(
            trace_vectors,
            src_key_padding_mask=~trace_mask,
        )
        memory, attention = self.memory_pool(
            set_encoded,
            ~trace_mask,
            return_attention=True,
        )
        pooled = memory.mean(dim=1)
        activity_memory, activity_mask = _pool_activity_occurrences(
            outputs.reshape(batch_size, trace_count * trace_length, -1),
            tokens.reshape(batch_size, trace_count * trace_length),
            self.activity_token_ids,
        )
        distribution = self.projection(
            pooled,
            memory=memory,
            activity_mask=activity_mask,
            activity_memory=activity_memory,
        )
        assert attention is not None
        return distribution, attention.mean(dim=1)


class PetriGraphEncoder(nn.Module):
    def __init__(
        self,
        tokenizer: ActivityTokenizer,
        semantic_dim: int,
        hidden_dim: int = 256,
        node_type_count: int = 3,
        message_passing_steps: int = 5,
        dropout: float = 0.1,
        memory_tokens: int = 8,
        projection_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.node_embedding = nn.Embedding(node_type_count, hidden_dim)
        self.transition_label_embedding = nn.Embedding(
            tokenizer.vocab_size, hidden_dim, padding_idx=tokenizer.pad_id
        )
        self.marking_projection = nn.Linear(2, hidden_dim)
        self.self_layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(message_passing_steps)
        )
        self.edge_layers = nn.ModuleList(
            nn.ModuleList(
                nn.Linear(hidden_dim, hidden_dim) for _ in range(4)
            )
            for _ in range(message_passing_steps)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(message_passing_steps)
        )
        self.jumping_projection = nn.Linear(
            hidden_dim * (message_passing_steps + 1), hidden_dim
        )
        self.memory_pool = SeedMemoryPool(hidden_dim, memory_tokens)
        self.projection = SemanticProjection(
            hidden_dim,
            semantic_dim,
            dropout=projection_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "activity_token_ids",
            torch.tensor(
                [
                    tokenizer.token_to_id[f"A{index}"]
                    for index in range(tokenizer.max_activities)
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )

    def forward(
        self,
        node_types: torch.Tensor,
        markings: torch.Tensor,
        adjacency: torch.Tensor | None,
        node_mask: torch.Tensor,
        transition_label_ids: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
    ) -> LatentDistribution:
        if transition_label_ids is None:
            transition_label_ids = torch.zeros_like(node_types)
        h = self.dropout(
            self.node_embedding(node_types)
            + self.marking_projection(markings)
            + self.transition_label_embedding(transition_label_ids)
        )
        valid = node_mask.unsqueeze(-1).to(h.dtype)
        h = h * valid
        states = [h]
        for self_layer, edge_layers, norm in zip(
            self.self_layers, self.edge_layers, self.norms
        ):
            messages = self_layer(h)
            if edge_index is not None and edge_types is not None:
                flat_h = h.reshape(-1, h.shape[-1])
                # Linear layers run in the autocast dtype while embeddings may
                # keep ``h`` in float32.  Match the message dtype so the
                # in-place sparse accumulation remains AMP-compatible.
                flat_aggregation = torch.zeros_like(
                    flat_h,
                    dtype=messages.dtype,
                )
                for edge_type in range(len(edge_layers) // 2):
                    selected = edge_types.eq(edge_type)
                    sources = edge_index[0, selected]
                    targets = edge_index[1, selected]
                    if sources.numel():
                        flat_aggregation.index_add_(
                            0,
                            sources,
                            edge_layers[2 * edge_type](flat_h[targets]),
                        )
                        flat_aggregation.index_add_(
                            0,
                            targets,
                            edge_layers[2 * edge_type + 1](flat_h[sources]),
                        )
                messages = messages + flat_aggregation.view_as(messages)
            else:
                if adjacency is None:
                    raise ValueError("dense Petri path requires adjacency")
                for edge_type in range(adjacency.shape[1]):
                    channel = adjacency[:, edge_type]
                    messages = messages + torch.bmm(channel, edge_layers[2 * edge_type](h))
                    messages = messages + torch.bmm(
                        channel.transpose(1, 2), edge_layers[2 * edge_type + 1](h)
                    )
            h = norm(h + self.dropout(F.gelu(messages))) * valid
            states.append(h)
        h = self.jumping_projection(torch.cat(states, dim=-1)) * valid
        memory, _ = self.memory_pool(h, ~node_mask)
        pooled = memory.mean(dim=1)
        activity_memory, activity_mask = _pool_activity_occurrences(
            h,
            transition_label_ids,
            self.activity_token_ids,
        )
        return self.projection(
            pooled,
            memory=memory,
            activity_mask=activity_mask,
            activity_memory=activity_memory,
        )


class GrammarTreeDecoder(nn.Module):
    def __init__(
        self,
        tokenizer: TreeTokenizer,
        semantic_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        layers: int = 4,
        memory_tokens: int = 8,
        max_sequence_length: int = 1024,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.memory_tokens = memory_tokens
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(
            tokenizer.vocab_size, hidden_dim, padding_idx=tokenizer.pad_id
        )
        self.position_embedding = nn.Embedding(max_sequence_length, hidden_dim)
        self.latent_to_memory = nn.Linear(
            semantic_dim, memory_tokens * hidden_dim
        )
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=_attention_heads(hidden_dim),
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        structural_names = [
            "<pad>",
            "<bos>",
            "<eos>",
            "TAU",
            *tokenizer.operator_tokens,
        ]
        self.structural_output = nn.Linear(hidden_dim, len(structural_names))
        self.arity_output = nn.Linear(hidden_dim, len(tokenizer.arity_tokens))
        self.activity_output = nn.Linear(hidden_dim, len(tokenizer.activity_tokens))
        self.copy_gate = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.max_sequence_length = max_sequence_length
        activity_ids = [
            tokenizer.token_to_id[name] for name in tokenizer.activity_tokens
        ]
        self.register_buffer(
            "activity_token_ids", torch.tensor(activity_ids, dtype=torch.long)
        )
        self.register_buffer(
            "structural_token_ids",
            torch.tensor(
                [tokenizer.token_to_id[name] for name in structural_names],
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "arity_token_ids",
            torch.tensor(
                [tokenizer.token_to_id[name] for name in tokenizer.arity_tokens],
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(max_sequence_length, max_sequence_length, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )

    @property
    def maximum_supported_decode_length(self) -> int:
        """Maximum total output length, including BOS and EOS."""

        # The final output token is selected from the last supported decoder
        # position and does not itself need another position embedding.
        return self.max_sequence_length + 1

    def validate_token_budget(self, token_budget: int) -> int:
        """Validate the deployment budget, which includes BOS and EOS."""

        budget = int(token_budget)
        if budget < 3:
            raise ValueError(
                "No process tree can be encoded with fewer than 3 tokens: "
                "<bos> TAU <eos>; bounded completion requires max_length >= 3."
            )
        if budget > self.maximum_supported_decode_length:
            raise ValueError(
                "Token budget exceeds decoder positional capacity "
                f"({budget} > {self.maximum_supported_decode_length})."
            )
        return budget

    def _forced_completion_token(
        self,
        open_nodes: torch.Tensor,
        pending_operator: torch.Tensor,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        """Return the next deterministic closure token for selected rows."""

        chosen = torch.full_like(open_nodes, self.tokenizer.pad_id)
        for row in torch.where(rows)[0].tolist():
            completion = self.tokenizer.shortest_completion(
                int(open_nodes[row].item()),
                int(pending_operator[row].item()),
            )
            if not completion:
                raise DecoderInvariantError("Shortest completion unexpectedly returned no token.")
            chosen[row] = completion[0]
        return chosen

    def source_memory(
        self, source: torch.Tensor | LatentDistribution
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(source, LatentDistribution):
            source_tensor = source.mu
            activity_mask = source.activity_mask
        else:
            source_tensor = source
            activity_mask = None
        if source_tensor.ndim == 3:
            return source_tensor, activity_mask
        if source_tensor.ndim != 2:
            raise ValueError("decoder source must have shape [batch, dim] or [batch, memory, dim]")
        memory = self.latent_to_memory(source_tensor).reshape(
            source_tensor.shape[0], self.memory_tokens, self.hidden_dim
        )
        return memory, activity_mask

    def forward(
        self,
        source: torch.Tensor | LatentDistribution,
        input_tokens: torch.Tensor,
        apply_grammar_mask: bool = True,
        activity_mask: torch.Tensor | None = None,
        activity_memory: torch.Tensor | None = None,
        allowed_activity_mask: torch.Tensor | None = None,
        avoid_duplicate_activity_labels: bool = False,
        duplicate_policy: str = "disallow",
        used_activity_mask: torch.Tensor | None = None,
        input_token_dropout: float = 0.0,
        completion_policy: CompletionPolicy = "prefix_only",
        remaining_tokens: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_completion_policy(completion_policy)
        if completion_policy == "bounded" and not apply_grammar_mask:
            raise ValueError("bounded completion requires grammar masking")
        if completion_policy == "bounded" and remaining_tokens is None:
            raise ValueError("bounded completion requires a remaining token budget")
        if input_tokens.shape[1] > self.max_sequence_length:
            raise ValueError("decoder input exceeds position capacity")
        memory, source_activity_mask = self.source_memory(source)
        if activity_mask is None:
            activity_mask = source_activity_mask
        if activity_memory is None and isinstance(source, LatentDistribution):
            activity_memory = source.activity_memory
        positions = torch.arange(input_tokens.shape[1], device=input_tokens.device).unsqueeze(0)
        embedded = self.embedding(input_tokens) + self.position_embedding(positions)
        if self.training and input_token_dropout > 0:
            drop = torch.rand(input_tokens.shape, device=input_tokens.device) < input_token_dropout
            drop &= input_tokens.ne(self.tokenizer.bos_id)
            drop &= input_tokens.ne(self.tokenizer.pad_id)
            embedded = embedded.masked_fill(drop.unsqueeze(-1), 0.0)
        causal = self.causal_mask[: input_tokens.shape[1], : input_tokens.shape[1]]
        hidden = self.decoder(
            self.dropout(embedded),
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=input_tokens.eq(self.tokenizer.pad_id),
        )
        projected = self.dropout(hidden)
        logits = torch.full(
            (*hidden.shape[:2], self.tokenizer.vocab_size),
            -1e9,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        logits[..., self.structural_token_ids] = self.structural_output(projected)
        logits[..., self.arity_token_ids] = self.arity_output(projected)
        logits[..., self.activity_token_ids] = self.activity_output(projected)
        source_constraint = self.decode_constraint_mask(
            input_tokens,
            allowed_activity_mask=allowed_activity_mask,
        )
        logits = logits.masked_fill(~source_constraint, -torch.inf)
        if activity_mask is not None:
            logits = self._mix_activity_copy(
                logits,
                hidden,
                activity_mask,
                activity_memory,
                allowed_activity_mask,
            )
        grammar = (
            self.tokenizer.valid_next_token_masks(
                input_tokens,
                remaining_tokens=(
                    remaining_tokens if completion_policy == "bounded" else None
                ),
            )
            if apply_grammar_mask
            else None
        )
        hard_constraint = self.decode_constraint_mask(
            input_tokens,
            grammar_mask=grammar,
            allowed_activity_mask=allowed_activity_mask,
            avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
            duplicate_policy=duplicate_policy,
            used_activity_mask=used_activity_mask,
        )
        logits = logits.masked_fill(~hard_constraint, -torch.inf)
        return logits

    def decode_constraint_mask(
        self,
        input_tokens: torch.Tensor,
        *,
        grammar_mask: torch.Tensor | None = None,
        allowed_activity_mask: torch.Tensor | None = None,
        avoid_duplicate_activity_labels: bool = False,
        duplicate_policy: str = "disallow",
        used_activity_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the shared hard mask for training and every decode strategy."""

        if duplicate_policy not in {"disallow", "penalize", "allow"}:
            raise ValueError("invalid duplicate policy")
        if grammar_mask is None:
            mask = torch.ones(
                (*input_tokens.shape, self.tokenizer.vocab_size),
                dtype=torch.bool,
                device=input_tokens.device,
            )
        else:
            mask = grammar_mask.clone()
        activity_legal = torch.ones(
            (input_tokens.shape[0], input_tokens.shape[1], self.tokenizer.max_activities),
            dtype=torch.bool,
            device=input_tokens.device,
        )
        if allowed_activity_mask is not None:
            allowed = allowed_activity_mask.to(device=input_tokens.device, dtype=torch.bool)
            if allowed.ndim == 1:
                allowed = allowed.unsqueeze(0)
            if allowed.shape[-1] != self.tokenizer.max_activities:
                raise ValueError("allowed activity mask has incompatible slot count")
            if allowed.shape[0] == 1 and input_tokens.shape[0] != 1:
                allowed = allowed.expand(input_tokens.shape[0], -1)
            if allowed.shape[0] != input_tokens.shape[0]:
                raise ValueError("allowed activity mask has incompatible batch size")
            activity_legal &= allowed.unsqueeze(1)
        if (
            avoid_duplicate_activity_labels
            and duplicate_policy == "disallow"
        ):
            if used_activity_mask is None:
                occurrences = input_tokens.unsqueeze(-1).eq(
                    self.activity_token_ids.view(1, 1, -1)
                )
                used = occurrences.cumsum(dim=1).gt(0)
            else:
                used = used_activity_mask.to(device=input_tokens.device, dtype=torch.bool)
                if used.ndim == 1:
                    used = used.unsqueeze(0)
                if used.shape != (input_tokens.shape[0], self.tokenizer.max_activities):
                    raise ValueError("used activity mask has incompatible shape")
                used = used.unsqueeze(1).expand(-1, input_tokens.shape[1], -1)
            activity_legal &= ~used
        mask[..., self.activity_token_ids] = (
            mask[..., self.activity_token_ids] & activity_legal
        )
        return mask

    def _mix_activity_copy(
        self,
        logits: torch.Tensor,
        hidden: torch.Tensor,
        activity_mask: torch.Tensor,
        activity_memory: torch.Tensor | None,
        allowed_activity_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        copy_keys = (
            self.embedding(self.activity_token_ids).unsqueeze(0).expand(
                hidden.shape[0], -1, -1
            )
            if activity_memory is None
            else activity_memory
        )
        copy_logits = torch.einsum("bth,bah->bta", hidden, copy_keys)
        copy_logits = copy_logits / math.sqrt(self.hidden_dim)
        copy_mask = activity_mask.to(dtype=torch.bool)
        if allowed_activity_mask is not None:
            allowed = allowed_activity_mask.to(device=hidden.device, dtype=torch.bool)
            if allowed.ndim == 1:
                allowed = allowed.unsqueeze(0)
            if allowed.shape[0] == 1 and copy_mask.shape[0] != 1:
                allowed = allowed.expand(copy_mask.shape[0], -1)
            copy_mask = copy_mask & allowed
        has_activity = copy_mask.any(dim=1, keepdim=True)
        safe_mask = copy_mask | ~has_activity
        copy_logits = copy_logits.masked_fill(~safe_mask.unsqueeze(1), -1e9)
        copy_probabilities = torch.softmax(copy_logits, dim=-1)
        copy_full = torch.zeros_like(logits)
        copy_full[..., self.activity_token_ids] = copy_probabilities
        vocabulary_probabilities = torch.softmax(logits, dim=-1)
        generate = torch.sigmoid(self.copy_gate(hidden))
        generate = torch.where(
            has_activity.unsqueeze(1), generate, torch.ones_like(generate)
        )
        probabilities = (
            generate * vocabulary_probabilities + (1.0 - generate) * copy_full
        )
        return probabilities.clamp_min(1e-12).log()

    def _project_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        projected = self.dropout(hidden)
        logits = torch.full(
            (*hidden.shape[:2], self.tokenizer.vocab_size),
            -1e9,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        logits[..., self.structural_token_ids] = self.structural_output(projected)
        logits[..., self.arity_token_ids] = self.arity_output(projected)
        logits[..., self.activity_token_ids] = self.activity_output(projected)
        return logits

    def _incremental_hidden(
        self,
        input_token: torch.Tensor,
        position: int,
        memory: torch.Tensor,
        layer_caches: list[torch.Tensor | None],
        input_token_dropout: float = 0.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Evaluate one decoder position while caching causal self-attention keys."""

        x = self.embedding(input_token).unsqueeze(1)
        x = x + self.position_embedding(
            torch.full_like(input_token, position)
        ).unsqueeze(1)
        if self.training and input_token_dropout > 0:
            drop = torch.rand(input_token.shape, device=input_token.device)
            drop = drop < input_token_dropout
            drop &= input_token.ne(self.tokenizer.bos_id)
            drop &= input_token.ne(self.tokenizer.pad_id)
            x = x.masked_fill(drop.view(-1, 1, 1), 0.0)
        x = self.dropout(x)
        next_caches: list[torch.Tensor] = []
        for layer, cached in zip(self.decoder.layers, layer_caches):
            if not layer.norm_first:
                raise RuntimeError("incremental decoder requires norm_first layers")
            normalized = layer.norm1(x)
            keys = normalized if cached is None else torch.cat([cached, normalized], dim=1)
            attention = layer.self_attn(
                normalized,
                keys,
                keys,
                need_weights=False,
            )[0]
            x = x + layer.dropout1(attention)
            cross_input = layer.norm2(x)
            cross_attention = layer.multihead_attn(
                cross_input,
                memory,
                memory,
                need_weights=False,
            )[0]
            x = x + layer.dropout2(cross_attention)
            feedforward_input = layer.norm3(x)
            feedforward = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(feedforward_input)))
            )
            x = x + layer.dropout3(feedforward)
            next_caches.append(keys)
        if self.decoder.norm is not None:
            x = self.decoder.norm(x)
        return x, next_caches

    def _incremental_step(
        self,
        input_token: torch.Tensor,
        position: int,
        memory: torch.Tensor,
        layer_caches: list[torch.Tensor | None],
        *,
        open_nodes: torch.Tensor,
        pending_operator: torch.Tensor,
        activity_mask: torch.Tensor | None = None,
        activity_memory: torch.Tensor | None = None,
        allowed_activity_mask: torch.Tensor | None = None,
        avoid_duplicate_activity_labels: bool = False,
        duplicate_policy: str = "disallow",
        used_activity_mask: torch.Tensor | None = None,
        input_token_dropout: float = 0.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Decode one constrained position using reusable incremental state."""

        hidden, next_caches = self._incremental_hidden(
            input_token,
            position,
            memory,
            layer_caches,
            input_token_dropout=input_token_dropout,
        )
        logits = self._project_hidden(hidden)
        one_token = input_token.unsqueeze(1)
        source_constraint = self.decode_constraint_mask(
            one_token,
            allowed_activity_mask=allowed_activity_mask,
        )
        logits = logits.masked_fill(~source_constraint, -torch.inf)
        if activity_mask is not None:
            logits = self._mix_activity_copy(
                logits,
                hidden,
                activity_mask,
                activity_memory,
                allowed_activity_mask,
            )
        grammar = self._incremental_grammar_mask(
            open_nodes,
            pending_operator,
        ).unsqueeze(1)
        hard_constraint = self.decode_constraint_mask(
            one_token,
            grammar_mask=grammar,
            allowed_activity_mask=allowed_activity_mask,
            avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
            duplicate_policy=duplicate_policy,
            used_activity_mask=used_activity_mask,
        )
        logits = logits.masked_fill(~hard_constraint, -torch.inf)
        return logits[:, 0], next_caches

    def _finalize_next_token_scores(
        self,
        base_logits: torch.Tensor,
        input_tokens: torch.Tensor,
        *,
        prefix_grammar_mask: torch.Tensor,
        completion_mask: torch.Tensor,
        allowed_activity_mask: torch.Tensor | None,
        avoid_duplicate_activity_labels: bool,
        duplicate_policy: str,
        used_activity_mask: torch.Tensor | None,
    ) -> NextTokenScores:
        source_constraint = self.decode_constraint_mask(
            input_tokens,
            allowed_activity_mask=allowed_activity_mask,
        )[:, -1]
        duplicate_constraint = self.decode_constraint_mask(
            input_tokens,
            avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
            duplicate_policy=duplicate_policy,
            used_activity_mask=used_activity_mask,
        )[:, -1]
        held_constraint = (
            prefix_grammar_mask & source_constraint & duplicate_constraint
        )
        base_logits = base_logits.masked_fill(~held_constraint, -torch.inf)
        if not held_constraint.any(dim=-1).all():
            raise RuntimeError("ordinary decode constraints produced an empty candidate set")
        base_log_probs = F.log_softmax(base_logits, dim=-1)
        effective = held_constraint & completion_mask
        if not effective.any(dim=-1).all():
            raise RuntimeError("bounded completion produced an empty candidate set")
        search_scores = base_log_probs.masked_fill(~effective, -torch.inf)
        if avoid_duplicate_activity_labels and duplicate_policy == "penalize":
            if used_activity_mask is None:
                used = input_tokens.unsqueeze(-1).eq(
                    self.activity_token_ids.view(1, 1, -1)
                ).any(dim=1)
            else:
                used = used_activity_mask.to(
                    device=base_logits.device,
                    dtype=torch.bool,
                )
                if used.ndim == 1:
                    used = used.unsqueeze(0)
                if used.shape != (
                    base_logits.shape[0],
                    self.tokenizer.max_activities,
                ):
                    raise ValueError("used activity mask has incompatible shape")
            search_scores = search_scores.clone()
            search_scores[:, self.activity_token_ids] -= (
                0.75 * used.to(search_scores.dtype)
            )
        return NextTokenScores(
            logits=base_logits.masked_fill(~effective, -torch.inf),
            base_log_probs=base_log_probs,
            search_scores=search_scores,
            prefix_grammar_mask=prefix_grammar_mask,
            completion_mask=completion_mask,
            source_constraint_mask=source_constraint,
            duplicate_constraint_mask=duplicate_constraint,
            effective_mask=effective,
        )

    def next_token_scores(
        self,
        source: torch.Tensor | LatentDistribution,
        input_tokens: torch.Tensor,
        *,
        activity_mask: torch.Tensor | None = None,
        activity_memory: torch.Tensor | None = None,
        allowed_activity_mask: torch.Tensor | None = None,
        avoid_duplicate_activity_labels: bool = False,
        duplicate_policy: str = "disallow",
        used_activity_mask: torch.Tensor | None = None,
        completion_policy: CompletionPolicy = "bounded",
        remaining_tokens: int | torch.Tensor | None = None,
    ) -> NextTokenScores:
        """Score a full prefix while preserving pre-budget model probability."""

        _validate_completion_policy(completion_policy)
        if completion_policy == "bounded" and remaining_tokens is None:
            raise ValueError("bounded completion requires a remaining token budget")
        base_logits = self.forward(
            source,
            input_tokens,
            apply_grammar_mask=True,
            activity_mask=activity_mask,
            activity_memory=activity_memory,
            allowed_activity_mask=allowed_activity_mask,
            avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
            duplicate_policy=duplicate_policy,
            used_activity_mask=used_activity_mask,
            completion_policy="prefix_only",
        )[:, -1]
        prefix_grammar = self.tokenizer.valid_next_token_masks(input_tokens)[:, -1]
        if completion_policy == "bounded":
            bounded = self.tokenizer.valid_next_token_masks(
                input_tokens,
                remaining_tokens=remaining_tokens,
            )[:, -1]
            completion = ~prefix_grammar | bounded
        else:
            completion = torch.ones_like(prefix_grammar)
        return self._finalize_next_token_scores(
            base_logits,
            input_tokens,
            prefix_grammar_mask=prefix_grammar,
            completion_mask=completion,
            allowed_activity_mask=allowed_activity_mask,
            avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
            duplicate_policy=duplicate_policy,
            used_activity_mask=used_activity_mask,
        )

    def _incremental_next_token_scores(
        self,
        input_token: torch.Tensor,
        position: int,
        memory: torch.Tensor,
        layer_caches: list[torch.Tensor | None],
        *,
        open_nodes: torch.Tensor,
        pending_operator: torch.Tensor,
        activity_mask: torch.Tensor | None = None,
        activity_memory: torch.Tensor | None = None,
        allowed_activity_mask: torch.Tensor | None = None,
        avoid_duplicate_activity_labels: bool = False,
        duplicate_policy: str = "disallow",
        used_activity_mask: torch.Tensor | None = None,
        completion_policy: CompletionPolicy = "bounded",
        remaining_tokens: int | torch.Tensor | None = None,
    ) -> tuple[NextTokenScores, list[torch.Tensor]]:
        """Shared cached step used by greedy and progressive decoding."""

        _validate_completion_policy(completion_policy)
        if completion_policy == "bounded" and remaining_tokens is None:
            raise ValueError("bounded completion requires a remaining token budget")
        base_logits, next_caches = self._incremental_step(
            input_token,
            position,
            memory,
            layer_caches,
            open_nodes=open_nodes,
            pending_operator=pending_operator,
            activity_mask=activity_mask,
            activity_memory=activity_memory,
            allowed_activity_mask=allowed_activity_mask,
            avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
            duplicate_policy=duplicate_policy,
            used_activity_mask=used_activity_mask,
        )
        prefix_grammar = self.tokenizer.prefix_grammar_mask(
            open_nodes,
            pending_operator,
        )
        completion = (
            self.tokenizer.completion_feasibility_mask(
                open_nodes,
                pending_operator,
                remaining_tokens,
            )
            if completion_policy == "bounded"
            else torch.ones_like(prefix_grammar)
        )
        scores = self._finalize_next_token_scores(
            base_logits,
            input_token.unsqueeze(1),
            prefix_grammar_mask=prefix_grammar,
            completion_mask=completion,
            allowed_activity_mask=allowed_activity_mask,
            avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
            duplicate_policy=duplicate_policy,
            used_activity_mask=used_activity_mask,
        )
        return scores, next_caches

    def _incremental_grammar_mask(
        self,
        open_nodes: torch.Tensor,
        pending_operator: torch.Tensor,
    ) -> torch.Tensor:
        return self.tokenizer.prefix_grammar_mask(open_nodes, pending_operator)

    def _advance_incremental_grammar(
        self,
        chosen: torch.Tensor,
        open_nodes: torch.Tensor,
        pending_operator: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        self.tokenizer.advance_grammar_state(
            chosen,
            open_nodes,
            pending_operator,
            active,
        )

    @torch.no_grad()
    def decode_guaranteed(
        self,
        source: torch.Tensor | LatentDistribution,
        *,
        total_token_budget_including_bos_eos: int = 128,
        beam_size: int = 1,
        length_penalty: float = 0.7,
        activity_mask: torch.Tensor | None = None,
        allowed_activity_mask: torch.Tensor | None = None,
        constrain_to_source_activities: bool = True,
        avoid_duplicate_activity_labels: bool = True,
        duplicate_policy: str = "disallow",
    ) -> torch.Tensor:
        """Decode with an enforced structural and total-token-budget contract.

        This deployment API intentionally exposes no completion-policy or
        grammar-mask switch. A successful return has passed strict completed
        sequence validation for every batch row.
        """

        budget = self.validate_token_budget(total_token_budget_including_bos_eos)
        if beam_size > 1:
            decoded = self.decode_beam(
                source,
                max_length=budget,
                beam_size=beam_size,
                length_penalty=length_penalty,
                allowed_activity_mask=allowed_activity_mask,
                constrain_to_source_activities=constrain_to_source_activities,
                avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
                duplicate_policy=duplicate_policy,
                completion_policy="bounded",
            )
        else:
            decoded = self.decode_greedy(
                source,
                max_length=budget,
                apply_grammar_mask=True,
                activity_mask=activity_mask,
                allowed_activity_mask=allowed_activity_mask,
                constrain_to_source_activities=constrain_to_source_activities,
                avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
                duplicate_policy=duplicate_policy,
                completion_policy="bounded",
            )
        for token_ids in decoded.detach().cpu().tolist():
            self.tokenizer.validate_complete_tree_sequence(
                token_ids,
                token_budget=budget,
            )
        return decoded

    @torch.no_grad()
    def decode_unbounded_for_diagnostics(
        self,
        source: torch.Tensor | LatentDistribution,
        *,
        maximum_tokens_including_bos: int = 128,
        beam_size: int = 1,
        length_penalty: float = 0.7,
        activity_mask: torch.Tensor | None = None,
        allowed_activity_mask: torch.Tensor | None = None,
        constrain_to_source_activities: bool = True,
        avoid_duplicate_activity_labels: bool = False,
        duplicate_policy: str = "allow",
    ) -> torch.Tensor:
        """Run historical prefix-only decoding for explicitly named diagnostics."""

        maximum = int(maximum_tokens_including_bos)
        if maximum < 2:
            raise ValueError("diagnostic decoding requires at least BOS plus one token")
        if maximum > self.maximum_supported_decode_length:
            raise ValueError(
                "Diagnostic decode length exceeds decoder positional capacity."
            )
        if beam_size > 1:
            return self.decode_beam(
                source,
                max_length=maximum,
                beam_size=beam_size,
                length_penalty=length_penalty,
                allowed_activity_mask=allowed_activity_mask,
                constrain_to_source_activities=constrain_to_source_activities,
                avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
                duplicate_policy=duplicate_policy,
                completion_policy="prefix_only",
            )
        return self.decode_greedy(
            source,
            max_length=maximum,
            apply_grammar_mask=True,
            activity_mask=activity_mask,
            allowed_activity_mask=allowed_activity_mask,
            constrain_to_source_activities=constrain_to_source_activities,
            avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
            duplicate_policy=duplicate_policy,
            completion_policy="prefix_only",
        )

    @torch.no_grad()
    def decode_greedy(
        self,
        source: torch.Tensor | LatentDistribution,
        max_length: int = 128,
        apply_grammar_mask: bool = True,
        activity_mask: torch.Tensor | None = None,
        allowed_activity_mask: torch.Tensor | None = None,
        constrain_to_source_activities: bool = True,
        avoid_duplicate_activity_labels: bool = True,
        duplicate_policy: str = "disallow",
        completion_policy: CompletionPolicy = "bounded",
        constraints: DecodeConstraints | None = None,
    ) -> torch.Tensor:
        """Low-level greedy decode; ``max_length`` includes BOS and EOS."""

        if constraints is not None:
            allowed_activity_mask = constraints.allowed_activity_slots
            constrain_to_source_activities = constraints.constrain_to_source_activities
            avoid_duplicate_activity_labels = constraints.avoid_duplicate_activity_labels
            duplicate_policy = constraints.duplicate_policy
            completion_policy = constraints.completion_policy
        _validate_completion_policy(completion_policy)
        if completion_policy == "bounded" and not apply_grammar_mask:
            raise ValueError("bounded completion requires grammar masking")
        if completion_policy == "bounded":
            self.validate_token_budget(max_length)
        elif max_length > self.maximum_supported_decode_length:
            raise ValueError("decode length exceeds decoder positional capacity")
        if not constrain_to_source_activities:
            allowed_activity_mask = None
        activity_memory = (
            source.activity_memory
            if isinstance(source, LatentDistribution)
            else None
        )
        memory, inferred_mask = self.source_memory(source)
        activity_mask = inferred_mask if activity_mask is None else activity_mask
        batch_size = memory.shape[0]
        if activity_mask is not None and activity_mask.shape[0] == 1 and batch_size > 1:
            activity_mask = activity_mask.expand(batch_size, -1)
        if activity_memory is not None and activity_memory.shape[0] == 1 and batch_size > 1:
            activity_memory = activity_memory.expand(batch_size, -1, -1)
        generated = torch.full(
            (batch_size, 1),
            self.tokenizer.bos_id,
            dtype=torch.long,
            device=memory.device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=memory.device)
        used_activity_mask = torch.zeros(
            (batch_size, self.tokenizer.max_activities),
            dtype=torch.bool,
            device=memory.device,
        )
        if not self.training and apply_grammar_mask:
            return self._decode_greedy_incremental(
                memory,
                generated,
                max_length=max_length,
                activity_mask=activity_mask,
                activity_memory=activity_memory,
                allowed_activity_mask=allowed_activity_mask,
                avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
                duplicate_policy=duplicate_policy,
                completion_policy=completion_policy,
            )
        open_nodes = torch.ones(batch_size, dtype=torch.long, device=memory.device)
        pending_operator = torch.zeros_like(open_nodes)
        for _ in range(max_length - 1):
            active = ~finished
            active_rows = torch.where(active)[0]
            if active_rows.numel() == 0:
                break
            forced = torch.zeros_like(active)
            if completion_policy == "bounded":
                remaining = max_length - generated.shape[1]
                minimum = self.tokenizer.minimum_tokens_to_finish(
                    open_nodes[active_rows],
                    pending_operator[active_rows],
                )
                if (minimum > remaining).any():
                    raise DecoderInvariantError(
                        "Current prefix cannot be completed inside the token budget."
                    )
                forced[active_rows] = minimum.eq(remaining)
            neural_rows = torch.where(active & ~forced)[0]
            next_token = self._forced_completion_token(
                open_nodes,
                pending_operator,
                forced,
            )
            active_rows = neural_rows
            active_logits = None
            active_activity_mask = (
                None if activity_mask is None else activity_mask[active_rows]
            )
            active_activity_memory = (
                None if activity_memory is None else activity_memory[active_rows]
            )
            if allowed_activity_mask is None or allowed_activity_mask.ndim == 1:
                active_allowed = allowed_activity_mask
            elif allowed_activity_mask.shape[0] == 1:
                active_allowed = allowed_activity_mask
            else:
                active_allowed = allowed_activity_mask[active_rows]
            if active_rows.numel() and apply_grammar_mask:
                active_logits = self.next_token_scores(
                    memory[active_rows],
                    generated[active_rows],
                    activity_mask=active_activity_mask,
                    activity_memory=active_activity_memory,
                    allowed_activity_mask=active_allowed,
                    avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
                    duplicate_policy=duplicate_policy,
                    used_activity_mask=used_activity_mask[active_rows],
                    completion_policy=completion_policy,
                    remaining_tokens=max_length - generated.shape[1],
                ).search_scores
            elif active_rows.numel():
                active_logits = self.forward(
                    memory[active_rows],
                    generated[active_rows],
                    apply_grammar_mask=False,
                    activity_mask=active_activity_mask,
                    activity_memory=active_activity_memory,
                    allowed_activity_mask=active_allowed,
                    avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
                    duplicate_policy=duplicate_policy,
                    used_activity_mask=used_activity_mask[active_rows],
                )[:, -1]
            if active_logits is not None:
                next_token[active_rows] = active_logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            self._advance_incremental_grammar(
                next_token,
                open_nodes,
                pending_operator,
                active,
            )
            chosen_activity = next_token.unsqueeze(-1).eq(
                self.activity_token_ids.view(1, -1)
            )
            used_activity_mask |= chosen_activity
            finished |= next_token.eq(self.tokenizer.eos_id)
            if finished.all():
                break
        return generated

    def _decode_greedy_incremental(
        self,
        memory: torch.Tensor,
        generated: torch.Tensor,
        *,
        max_length: int,
        activity_mask: torch.Tensor | None,
        activity_memory: torch.Tensor | None,
        allowed_activity_mask: torch.Tensor | None,
        avoid_duplicate_activity_labels: bool,
        duplicate_policy: str,
        completion_policy: CompletionPolicy,
    ) -> torch.Tensor:
        batch_size = memory.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=memory.device)
        used = torch.zeros(
            (batch_size, self.tokenizer.max_activities),
            dtype=torch.bool,
            device=memory.device,
        )
        open_nodes = torch.ones(batch_size, dtype=torch.long, device=memory.device)
        pending_operator = torch.zeros_like(open_nodes)
        caches: list[torch.Tensor | None] = [None] * len(self.decoder.layers)
        current = generated[:, 0]

        def selected_rows(
            value: torch.Tensor | None,
            rows: torch.Tensor,
            *,
            allow_unbatched: bool = False,
        ) -> torch.Tensor | None:
            if value is None:
                return None
            if allow_unbatched and value.ndim == 1:
                return value
            if value.shape[0] == 1 and batch_size > 1:
                value = value.expand(batch_size, *value.shape[1:])
            return value[rows]

        for position in range(max_length - 1):
            active = ~finished
            remaining = max_length - position - 1
            forced = torch.zeros_like(active)
            if completion_policy == "bounded":
                minimum = self.tokenizer.minimum_tokens_to_finish(
                    open_nodes[active],
                    pending_operator[active],
                )
                if (minimum > remaining).any():
                    raise DecoderInvariantError(
                        "Current prefix cannot be completed inside the token budget."
                    )
                forced[active] = minimum.eq(remaining)
            next_token = self._forced_completion_token(
                open_nodes,
                pending_operator,
                forced,
            )
            neural = active & ~forced
            if neural.any():
                active_caches = [
                    None if cache is None else cache[neural]
                    for cache in caches
                ]
                scores, next_active_caches = self._incremental_next_token_scores(
                    current[neural],
                    position,
                    memory[neural],
                    active_caches,
                    open_nodes=open_nodes[neural],
                    pending_operator=pending_operator[neural],
                    activity_mask=selected_rows(activity_mask, neural),
                    activity_memory=selected_rows(activity_memory, neural),
                    allowed_activity_mask=selected_rows(
                        allowed_activity_mask,
                        neural,
                        allow_unbatched=True,
                    ),
                    avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
                    duplicate_policy=duplicate_policy,
                    used_activity_mask=used[neural],
                    completion_policy=completion_policy,
                    remaining_tokens=remaining,
                )
                next_token[neural] = scores.search_scores.argmax(dim=-1)
                next_caches: list[torch.Tensor] = []
                for next_cache in next_active_caches:
                    full_cache = torch.zeros(
                        (batch_size, *next_cache.shape[1:]),
                        dtype=next_cache.dtype,
                        device=next_cache.device,
                    )
                    full_cache[neural] = next_cache
                    next_caches.append(full_cache)
                caches = next_caches
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            self._advance_incremental_grammar(
                next_token,
                open_nodes,
                pending_operator,
                active,
            )
            used |= next_token.unsqueeze(-1).eq(
                self.activity_token_ids.view(1, -1)
            )
            finished |= next_token.eq(self.tokenizer.eos_id)
            if finished.all():
                break
            current = next_token
        return generated

    @torch.no_grad()
    def decode_beam(
        self,
        source: torch.Tensor | LatentDistribution,
        max_length: int = 128,
        beam_size: int = 5,
        length_penalty: float = 0.7,
        allowed_activity_mask: torch.Tensor | None = None,
        constrain_to_source_activities: bool = True,
        avoid_duplicate_activity_labels: bool = True,
        duplicate_policy: str = "disallow",
        completion_policy: CompletionPolicy = "bounded",
        constraints: DecodeConstraints | None = None,
    ) -> torch.Tensor:
        """Low-level beam decode; ``max_length`` includes BOS and EOS."""

        if beam_size <= 1:
            return self.decode_greedy(
                source,
                max_length=max_length,
                allowed_activity_mask=allowed_activity_mask,
                constrain_to_source_activities=constrain_to_source_activities,
                avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
                duplicate_policy=duplicate_policy,
                completion_policy=completion_policy,
                constraints=constraints,
            )
        candidates = self.decode_beam_candidates(
            source,
            max_length=max_length,
            beam_size=beam_size,
            length_penalty=length_penalty,
            allowed_activity_mask=allowed_activity_mask,
            constrain_to_source_activities=constrain_to_source_activities,
            avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
            duplicate_policy=duplicate_policy,
            completion_policy=completion_policy,
            constraints=constraints,
        )
        memory, _ = self.source_memory(source)
        rows = [
            torch.tensor(row[0][0], device=memory.device) for row in candidates
        ]
        width = max(row.numel() for row in rows)
        output = torch.full(
            (len(rows), width), self.tokenizer.pad_id, dtype=torch.long, device=memory.device
        )
        for index, row in enumerate(rows):
            output[index, : row.numel()] = row
        return output

    @torch.no_grad()
    def decode_beam_candidates(
        self,
        source: torch.Tensor | LatentDistribution,
        max_length: int = 128,
        beam_size: int = 5,
        length_penalty: float = 0.7,
        allowed_activity_mask: torch.Tensor | None = None,
        constrain_to_source_activities: bool = True,
        avoid_duplicate_activity_labels: bool = True,
        duplicate_policy: str = "disallow",
        completion_policy: CompletionPolicy = "bounded",
        constraints: DecodeConstraints | None = None,
    ) -> list[list[tuple[list[int], float]]]:
        """Return beam candidates under a total budget including BOS and EOS."""

        if constraints is not None:
            allowed_activity_mask = constraints.allowed_activity_slots
            constrain_to_source_activities = constraints.constrain_to_source_activities
            avoid_duplicate_activity_labels = constraints.avoid_duplicate_activity_labels
            duplicate_policy = constraints.duplicate_policy
            completion_policy = constraints.completion_policy
        _validate_completion_policy(completion_policy)
        if completion_policy == "bounded":
            self.validate_token_budget(max_length)
        elif max_length > self.maximum_supported_decode_length:
            raise ValueError("decode length exceeds decoder positional capacity")
        if not constrain_to_source_activities:
            allowed_activity_mask = None
        memory, activity_mask = self.source_memory(source)
        activity_memory = (
            source.activity_memory
            if isinstance(source, LatentDistribution)
            else None
        )
        batch_size = memory.shape[0]
        if batch_size == 0:
            return []

        # Search every sample and beam together and retain each decoder layer's
        # causal cache. This keeps one-token decoding linear in prefix length
        # and gives accelerators a useful batch instead of serial one-row work.
        device = memory.device
        vocabulary_size = self.tokenizer.vocab_size
        sequences = torch.full(
            (batch_size, beam_size, max_length),
            self.tokenizer.pad_id,
            dtype=torch.long,
            device=device,
        )
        sequences[:, 0, 0] = self.tokenizer.bos_id
        scores = torch.full(
            (batch_size, beam_size),
            -torch.inf,
            dtype=memory.dtype,
            device=device,
        )
        scores[:, 0] = 0.0
        lengths = torch.ones(
            (batch_size, beam_size), dtype=torch.long, device=device
        )
        finished = torch.zeros(
            (batch_size, beam_size), dtype=torch.bool, device=device
        )
        used_activity = torch.zeros(
            (batch_size, beam_size, self.tokenizer.max_activities),
            dtype=torch.bool,
            device=device,
        )
        open_nodes = torch.ones(
            (batch_size, beam_size), dtype=torch.long, device=device
        )
        pending_operator = torch.zeros_like(open_nodes)
        caches: list[torch.Tensor | None] = [None] * len(self.decoder.layers)

        def active_source_rows(
            value: torch.Tensor | None,
            active: torch.Tensor,
            *,
            allow_unbatched: bool = False,
        ) -> torch.Tensor | None:
            if value is None:
                return None
            value = value.to(device)
            if allow_unbatched and value.ndim == 1:
                value = value.unsqueeze(0)
            if value.shape[0] == 1 and batch_size > 1:
                value = value.expand(batch_size, *value.shape[1:])
            if value.shape[0] != batch_size:
                raise ValueError("beam-search source has incompatible batch size")
            expanded = value.unsqueeze(1).expand(
                batch_size, beam_size, *value.shape[1:]
            )
            return expanded[active]

        for position in range(max_length - 1):
            valid = torch.isfinite(scores)
            active = valid & ~finished
            if not active.any():
                break

            remaining = max_length - position - 1
            forced = torch.zeros_like(active)
            if completion_policy == "bounded":
                minimum = self.tokenizer.minimum_tokens_to_finish(
                    open_nodes[active],
                    pending_operator[active],
                )
                if (minimum > remaining).any():
                    raise DecoderInvariantError(
                        "Current prefix cannot be completed inside the token budget."
                    )
                forced[active] = minimum.eq(remaining)
            neural_active = active & ~forced

            current = sequences[:, :, position][neural_active]
            active_caches = [
                None if cache is None else cache[neural_active]
                for cache in caches
            ]
            step_scores = torch.full(
                (batch_size, beam_size, vocabulary_size),
                -torch.inf,
                dtype=scores.dtype,
                device=device,
            )
            next_active_caches: list[torch.Tensor] = []
            if neural_active.any():
                next_token_scores, next_active_caches = self._incremental_next_token_scores(
                    current,
                    position,
                    active_source_rows(memory, neural_active),
                    active_caches,
                    open_nodes=open_nodes[neural_active],
                    pending_operator=pending_operator[neural_active],
                    activity_mask=active_source_rows(activity_mask, neural_active),
                    activity_memory=active_source_rows(activity_memory, neural_active),
                    allowed_activity_mask=active_source_rows(
                        allowed_activity_mask,
                        neural_active,
                        allow_unbatched=True,
                    ),
                    avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
                    duplicate_policy=duplicate_policy,
                    used_activity_mask=used_activity[neural_active],
                    completion_policy=completion_policy,
                    remaining_tokens=remaining,
                )
                step_scores[neural_active] = next_token_scores.search_scores.to(scores.dtype)
            if forced.any():
                forced_tokens = self._forced_completion_token(
                    open_nodes[forced],
                    pending_operator[forced],
                    torch.ones(int(forced.sum().item()), dtype=torch.bool, device=device),
                )
                forced_batch, forced_beam = torch.where(forced)
                step_scores[forced_batch, forced_beam, forced_tokens] = 0.0
            expanded_scores = scores.unsqueeze(-1) + step_scores
            expanded_ranks = expanded_scores / float(
                (position + 2) ** length_penalty
            )

            carry_scores = scores.masked_fill(~(valid & finished), -torch.inf)
            carry_ranks = carry_scores / lengths.to(scores.dtype).pow(length_penalty)
            all_scores = torch.cat(
                (expanded_scores.flatten(1), carry_scores), dim=1
            )
            all_ranks = torch.cat(
                (expanded_ranks.flatten(1), carry_ranks), dim=1
            )
            if beam_size == 1:
                # Match greedy decoding's deterministic lowest-token tie break.
                selected = all_ranks.argmax(dim=1, keepdim=True)
                selected_ranks = all_ranks.gather(1, selected)
            else:
                selected_ranks, selected = torch.topk(all_ranks, beam_size, dim=1)
            selected_valid = torch.isfinite(selected_ranks)
            selected_scores = all_scores.gather(1, selected)
            selected_scores = selected_scores.masked_fill(~selected_valid, -torch.inf)

            expansion_count = beam_size * vocabulary_size
            selected_carry = selected.ge(expansion_count)
            expansion_parent = selected.div(vocabulary_size, rounding_mode="floor")
            carry_parent = selected - expansion_count
            selected_parent = torch.where(
                selected_carry, carry_parent, expansion_parent
            )
            selected_token = selected.remainder(vocabulary_size)

            sequence_index = selected_parent.unsqueeze(-1).expand(
                -1, -1, max_length
            )
            sequences = sequences.gather(1, sequence_index)
            parent_lengths = lengths.gather(1, selected_parent)
            selected_expansion = selected_valid & ~selected_carry
            next_lengths = parent_lengths + selected_expansion.to(torch.long)
            batch_indices, beam_indices = torch.where(selected_expansion)
            sequences[
                batch_indices,
                beam_indices,
                next_lengths[selected_expansion] - 1,
            ] = selected_token[selected_expansion]

            state_index = selected_parent.unsqueeze(-1).expand(
                -1, -1, self.tokenizer.max_activities
            )
            used_activity = used_activity.gather(1, state_index)
            chosen_activity = selected_token.unsqueeze(-1).eq(
                self.activity_token_ids.view(1, 1, -1)
            )
            used_activity |= chosen_activity & selected_expansion.unsqueeze(-1)
            open_nodes = open_nodes.gather(1, selected_parent)
            pending_operator = pending_operator.gather(1, selected_parent)
            self._advance_incremental_grammar(
                selected_token,
                open_nodes,
                pending_operator,
                selected_expansion,
            )

            if next_active_caches:
                selected_caches: list[torch.Tensor] = []
                for next_cache in next_active_caches:
                    full_cache = torch.zeros(
                        (batch_size, beam_size, *next_cache.shape[1:]),
                        dtype=next_cache.dtype,
                        device=device,
                    )
                    full_cache[neural_active] = next_cache
                    cache_index = selected_parent.view(
                        batch_size,
                        beam_size,
                        *([1] * (full_cache.ndim - 2)),
                    ).expand(-1, -1, *full_cache.shape[2:])
                    selected_caches.append(full_cache.gather(1, cache_index))
                caches = selected_caches
            else:
                caches = [None] * len(self.decoder.layers)
            scores = selected_scores
            lengths = next_lengths
            finished = selected_valid & (
                selected_carry | selected_token.eq(self.tokenizer.eos_id)
            )

        candidate_rows: list[list[tuple[list[int], float]]] = []
        for row in range(batch_size):
            row_candidates: list[tuple[list[int], float]] = []
            for beam in range(beam_size):
                if not torch.isfinite(scores[row, beam]):
                    continue
                length = int(lengths[row, beam].item())
                row_candidates.append(
                    (
                        sequences[row, beam, :length].tolist(),
                        float(scores[row, beam].item()),
                    )
                )
            candidate_rows.append(row_candidates)
        return candidate_rows


class ProcRosettaModel(nn.Module):
    def __init__(
        self,
        tree_tokenizer: TreeTokenizer | None = None,
        activity_tokenizer: ActivityTokenizer | None = None,
        latent_dim: int = 96,
        hidden_dim: int = 192,
        dropout: float | None = None,
        memory_tokens: int = 6,
        decoder_layers: int = 3,
        tree_encoder_dropout: float = 0.12,
        trace_encoder_dropout: float = 0.20,
        petri_encoder_dropout: float = 0.12,
        decoder_dropout: float = 0.20,
        projection_dropout: float = 0.20,
        tree_encoder_layers: int = 3,
        trace_event_layers: int = 1,
        trace_set_layers: int = 1,
        petri_message_passing_steps: int = 5,
    ) -> None:
        super().__init__()
        # ``dropout`` remains a source-compatible alias for callers created
        # before modality-specific regularization was introduced.
        if dropout is not None:
            tree_encoder_dropout = dropout
            trace_encoder_dropout = dropout
            petri_encoder_dropout = dropout
            decoder_dropout = dropout
        self.tree_tokenizer = tree_tokenizer or TreeTokenizer()
        self.activity_tokenizer = activity_tokenizer or ActivityTokenizer(
            max_activities=self.tree_tokenizer.max_activities
        )
        self.tree_encoder = TreeEncoder(
            self.tree_tokenizer,
            latent_dim,
            hidden_dim,
            tree_encoder_dropout,
            projection_dropout=projection_dropout,
            layers=tree_encoder_layers,
            memory_tokens=memory_tokens,
        )
        self.trace_encoder = TraceEncoder(
            self.activity_tokenizer,
            latent_dim,
            hidden_dim,
            trace_encoder_dropout,
            projection_dropout=projection_dropout,
            event_layers=trace_event_layers,
            set_layers=trace_set_layers,
            memory_tokens=memory_tokens,
        )
        self.petri_encoder = PetriGraphEncoder(
            self.activity_tokenizer,
            latent_dim,
            hidden_dim,
            message_passing_steps=petri_message_passing_steps,
            dropout=petri_encoder_dropout,
            projection_dropout=projection_dropout,
            memory_tokens=memory_tokens,
        )
        self.tree_decoder = GrammarTreeDecoder(
            self.tree_tokenizer,
            latent_dim,
            hidden_dim,
            decoder_dropout,
            layers=decoder_layers,
            memory_tokens=memory_tokens,
        )
        # The semantic latent remains the representation used by decoding,
        # retrieval, and geometry.  Instance discrimination gets a disposable
        # head so family identity does not have to dominate that representation.
        self.contrastive_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(projection_dropout),
            nn.Linear(latent_dim, latent_dim),
        )

    def encode_tree(
        self,
        tree_tokens: torch.Tensor,
        structure: dict[str, torch.Tensor] | None = None,
    ) -> LatentDistribution:
        return self.tree_encoder(
            tree_tokens,
            None if structure is None else structure.get("depths"),
            None if structure is None else structure.get("parents"),
        )

    def encode_traces(self, traces: dict[str, torch.Tensor]) -> LatentDistribution:
        return self.trace_encoder(traces["tokens"], traces["lengths"], traces["mask"])

    def encode_petri(self, petri: dict[str, torch.Tensor]) -> LatentDistribution:
        return self.petri_encoder(
            petri["node_types"],
            petri["markings"],
            petri.get("adjacency"),
            petri["node_mask"],
            petri.get("transition_label_ids"),
            petri.get("edge_index"),
            petri.get("edge_types"),
        )

    def forward(
        self,
        batch: dict[str, object],
        deterministic: bool = True,
        input_token_dropout: float = 0.0,
        scheduled_sampling_probability: float = 0.0,
        modality_subset_fusion_probability: float = 0.0,
        deployment_policy_probability: float = 0.0,
    ) -> dict[str, object]:
        tree_tokens = batch.get("tree_encoder_tokens", batch["tree_tokens"])
        decoder_targets = batch.get("decoder_targets")
        source_activity_masks = batch.get("source_activity_masks")
        tree_structure = batch.get("tree_structure")
        traces = batch["traces"]
        petri = batch["petri"]
        assert isinstance(tree_tokens, torch.Tensor)
        assert isinstance(traces, dict)
        assert isinstance(petri, dict)
        if decoder_targets is None:
            decoder_targets = {
                name: tree_tokens for name in ("tree", "trace", "petri")
            }
        if source_activity_masks is None:
            source_activity_masks = {
                name: None for name in ("tree", "trace", "petri")
            }
        assert isinstance(decoder_targets, dict)
        assert isinstance(source_activity_masks, dict)
        decoder_targets = dict(decoder_targets)
        for fusion_name in ("tree_trace", "tree_petri", "trace_petri", "fused"):
            decoder_targets.setdefault(fusion_name, decoder_targets["tree"])
        for name in (
            "tree",
            "trace",
            "petri",
            "tree_trace",
            "tree_petri",
            "trace_petri",
            "fused",
        ):
            decoder_targets.setdefault(f"deployment_{name}", decoder_targets[name])
        source_activity_masks = dict(source_activity_masks)
        fusion_modalities = {
            "tree_trace": ("tree", "trace"),
            "tree_petri": ("tree", "petri"),
            "trace_petri": ("trace", "petri"),
            "fused": ("tree", "trace", "petri"),
        }
        for fusion_name, modalities in fusion_modalities.items():
            if fusion_name in source_activity_masks:
                continue
            values = [source_activity_masks.get(name) for name in modalities]
            source_activity_masks[fusion_name] = (
                None
                if any(value is None for value in values)
                else torch.stack(values, dim=0).any(dim=0)
            )
        dists = {
            "tree": self.encode_tree(
                tree_tokens,
                tree_structure if isinstance(tree_structure, dict) else None,
            ),
            "trace": self.encode_traces(traces),
            "petri": self.encode_petri(petri),
        }
        names = ("tree", "trace", "petri")
        fused_mu = torch.stack(
            [dists[name].mu for name in names],
            dim=0,
        ).mean(dim=0)
        dists["fused"] = LatentDistribution(
            mu=fused_mu,
            logvar=torch.zeros_like(fused_mu),
        )
        batch_size = tree_tokens.shape[0]
        stacked_distribution = _stack_latent_distributions([dists[name] for name in names])
        stacked_input = torch.cat(
            [decoder_targets[name][:, :-1] for name in names],
            dim=0,
        )
        allowed_rows = [source_activity_masks[name] for name in names]
        stacked_allowed = (
            None
            if any(value is None for value in allowed_rows)
            else torch.cat(allowed_rows, dim=0)
        )
        decoder_loss_targets = dict(decoder_targets)
        decoder_budget_targets = dict(decoder_targets)
        if self.training and scheduled_sampling_probability > 0:
            stacked_targets = torch.cat(
                [decoder_targets[name] for name in names],
                dim=0,
            )
            stacked_input, stacked_loss_targets = self._scheduled_sampling_inputs(
                stacked_distribution,
                stacked_input,
                stacked_allowed,
                stacked_targets=stacked_targets,
                probability=scheduled_sampling_probability,
                input_token_dropout=input_token_dropout,
            )
            split_loss_targets = stacked_loss_targets.split(batch_size, dim=0)
            decoder_loss_targets = dict(zip(names, split_loss_targets))
        stacked_logits = self.tree_decoder(
            stacked_distribution,
            stacked_input,
            allowed_activity_mask=stacked_allowed,
            input_token_dropout=input_token_dropout,
        )
        split_logits = stacked_logits.split(batch_size, dim=0)
        logits = dict(zip(names, split_logits))
        split_inputs = stacked_input.split(batch_size, dim=0)
        decoder_inputs = dict(zip(names, split_inputs))
        fused_input = decoder_targets["fused"][:, :-1]
        fused_loss_target = decoder_targets["fused"]
        fused_allowed = source_activity_masks.get("fused")
        if self.training and scheduled_sampling_probability > 0:
            fused_input, fused_loss_target = self._scheduled_sampling_inputs(
                fused_mu,
                fused_input,
                fused_allowed,
                stacked_targets=decoder_targets["fused"],
                probability=scheduled_sampling_probability,
                input_token_dropout=input_token_dropout,
            )
        # Deliberately pass the tensor, not a LatentDistribution: evaluation
        # supplies exactly this raw mean and has no modality/copy memory.
        logits["fused"] = self.tree_decoder(
            fused_mu,
            fused_input,
            allowed_activity_mask=fused_allowed,
            input_token_dropout=input_token_dropout,
        )
        decoder_inputs["fused"] = fused_input
        decoder_loss_targets["fused"] = fused_loss_target

        fusion_subset_name: str | None = None
        if (
            self.training
            and modality_subset_fusion_probability > 0.0
            and bool(
                torch.rand((), device=fused_mu.device)
                < modality_subset_fusion_probability
            )
        ):
            subset_names = (
                ("tree_trace", ("tree", "trace")),
                ("tree_petri", ("tree", "petri")),
                ("trace_petri", ("trace", "petri")),
            )
            subset_index = int(
                torch.randint(len(subset_names), (), device=fused_mu.device).item()
            )
            fusion_subset_name, modalities = subset_names[subset_index]
            subset_mu = torch.stack(
                [dists[name].mu for name in modalities],
                dim=0,
            ).mean(dim=0)
            subset_input = decoder_targets[fusion_subset_name][:, :-1]
            subset_loss_target = decoder_targets[fusion_subset_name]
            subset_allowed = source_activity_masks.get(fusion_subset_name)
            if scheduled_sampling_probability > 0:
                subset_input, subset_loss_target = self._scheduled_sampling_inputs(
                    subset_mu,
                    subset_input,
                    subset_allowed,
                    stacked_targets=decoder_targets[fusion_subset_name],
                    probability=scheduled_sampling_probability,
                    input_token_dropout=input_token_dropout,
                )
            logits["fused_subset"] = self.tree_decoder(
                subset_mu,
                subset_input,
                allowed_activity_mask=subset_allowed,
                input_token_dropout=input_token_dropout,
            )
            decoder_inputs["fused_subset"] = subset_input
            decoder_loss_targets["fused_subset"] = subset_loss_target
            decoder_budget_targets["fused_subset"] = decoder_targets[
                fusion_subset_name
            ]
        deployment_source_name: str | None = None
        if (
            self.training
            and deployment_policy_probability > 0.0
            and bool(
                torch.rand((), device=fused_mu.device) < deployment_policy_probability
            )
        ):
            deployment_sources = ("tree", "trace", "petri", "fused")
            deployment_source_name = deployment_sources[
                int(
                    torch.randint(
                        len(deployment_sources),
                        (),
                        device=fused_mu.device,
                    ).item()
                )
            ]
            deployment_source = (
                fused_mu
                if deployment_source_name == "fused"
                else dists[deployment_source_name]
            )
            deployment_target_name = f"deployment_{deployment_source_name}"
            deployment_input = decoder_targets[deployment_target_name][:, :-1]
            remaining_tokens = torch.arange(
                deployment_input.shape[1],
                0,
                -1,
                device=deployment_input.device,
            ).unsqueeze(0).expand(deployment_input.shape[0], -1)
            logits["deployment"] = self.tree_decoder(
                deployment_source,
                deployment_input,
                allowed_activity_mask=source_activity_masks.get(
                    deployment_source_name
                ),
                avoid_duplicate_activity_labels=True,
                completion_policy="bounded",
                remaining_tokens=remaining_tokens,
                input_token_dropout=input_token_dropout,
            )
            decoder_inputs["deployment"] = deployment_input
            decoder_loss_targets["deployment"] = decoder_targets[
                deployment_target_name
            ]
            decoder_budget_targets["deployment"] = decoder_targets[
                deployment_target_name
            ]
        contrastive_embeddings = {
            name: F.normalize(self.contrastive_head(dists[name].mu), dim=-1)
            for name in (*names, "fused")
        }
        return {
            "dists": dists,
            "z": {name: distribution.mu for name, distribution in dists.items()},
            "contrastive_embeddings": contrastive_embeddings,
            "tree_logits": logits,
            "decoder_targets": decoder_targets,
            "decoder_loss_targets": decoder_loss_targets,
            "decoder_budget_targets": decoder_budget_targets,
            "decoder_inputs": decoder_inputs,
            "fused_decoder_source": fused_mu,
            "fusion_subset_name": fusion_subset_name,
            "deployment_source_name": deployment_source_name,
        }

    @torch.no_grad()
    def _scheduled_sampling_inputs(
        self,
        source: LatentDistribution | torch.Tensor,
        teacher_input: torch.Tensor,
        allowed_activity_mask: torch.Tensor | None,
        *,
        stacked_targets: torch.Tensor,
        probability: float,
        input_token_dropout: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build sampled prefixes with cached, graph-free incremental decoding.

        Every predicted replacement is produced under the same grammar and
        source-activity constraints as normal decoding.  A teacher token that
        became illegal under a sampled prefix is replaced unconditionally and
        excluded from the reconstruction loss. Rows are compacted after
        EOS/PAD or the end of their teacher target, so completed sequences do
        not consume later incremental decoder steps.
        """

        sampled_input = torch.full_like(teacher_input, self.tree_tokenizer.pad_id)
        sampled_input[:, 0] = teacher_input[:, 0]
        loss_targets = torch.full_like(stacked_targets, self.tree_tokenizer.pad_id)
        loss_targets[:, 0] = stacked_targets[:, 0]

        memory, activity_mask = self.tree_decoder.source_memory(source)
        activity_memory = (
            source.activity_memory
            if isinstance(source, LatentDistribution)
            else None
        )
        batch_size = teacher_input.shape[0]
        row_ids = torch.arange(batch_size, device=teacher_input.device)
        row_ids = row_ids[
            teacher_input[:, 0].ne(self.tree_tokenizer.eos_id)
            & teacher_input[:, 0].ne(self.tree_tokenizer.pad_id)
            & stacked_targets[:, 1].ne(self.tree_tokenizer.pad_id)
        ]
        memory = memory[row_ids]
        if activity_mask is not None:
            activity_mask = activity_mask[row_ids]
        if activity_memory is not None:
            activity_memory = activity_memory[row_ids]
        if allowed_activity_mask is not None:
            allowed_activity_mask = allowed_activity_mask.to(teacher_input.device)
            if allowed_activity_mask.ndim == 1:
                allowed_activity_mask = allowed_activity_mask.unsqueeze(0)
            if allowed_activity_mask.shape[0] == 1 and batch_size > 1:
                allowed_activity_mask = allowed_activity_mask.expand(batch_size, -1)
            allowed_activity_mask = allowed_activity_mask[row_ids]

        open_nodes = torch.ones(
            row_ids.numel(), dtype=torch.long, device=teacher_input.device
        )
        pending_operator = torch.zeros_like(open_nodes)
        caches: list[torch.Tensor | None] = [None] * len(
            self.tree_decoder.decoder.layers
        )
        current = teacher_input[row_ids, 0]
        for position in range(teacher_input.shape[1]):
            if row_ids.numel() == 0:
                break
            current_logits, caches = self.tree_decoder._incremental_step(
                current,
                position,
                memory,
                caches,
                open_nodes=open_nodes,
                pending_operator=pending_operator,
                activity_mask=activity_mask,
                activity_memory=activity_memory,
                allowed_activity_mask=allowed_activity_mask,
                input_token_dropout=input_token_dropout,
            )
            teacher_next = stacked_targets[row_ids, position + 1]
            teacher_is_legal = torch.isfinite(
                current_logits.gather(1, teacher_next.unsqueeze(1)).squeeze(1)
            )
            legal_target = teacher_next.ne(self.tree_tokenizer.pad_id)
            legal_target &= teacher_is_legal
            loss_targets[row_ids, position + 1] = torch.where(
                legal_target,
                teacher_next,
                torch.full_like(teacher_next, self.tree_tokenizer.pad_id),
            )
            if position + 1 >= teacher_input.shape[1]:
                continue

            prediction = current_logits.argmax(dim=-1)
            replace = torch.rand(teacher_next.shape, device=teacher_next.device)
            replace = (replace < probability) | ~teacher_is_legal
            next_token = torch.where(replace, prediction, teacher_next)
            sampled_input[row_ids, position + 1] = next_token
            active = next_token.ne(self.tree_tokenizer.eos_id)
            active &= next_token.ne(self.tree_tokenizer.pad_id)
            active &= stacked_targets[row_ids, position + 2].ne(
                self.tree_tokenizer.pad_id
            )
            self.tree_decoder._advance_incremental_grammar(
                next_token,
                open_nodes,
                pending_operator,
                torch.ones_like(active),
            )

            row_ids = row_ids[active]
            current = next_token[active]
            memory = memory[active]
            open_nodes = open_nodes[active]
            pending_operator = pending_operator[active]
            caches = [cache[active] for cache in caches]
            if activity_mask is not None:
                activity_mask = activity_mask[active]
            if activity_memory is not None:
                activity_memory = activity_memory[active]
            if allowed_activity_mask is not None:
                allowed_activity_mask = allowed_activity_mask[active]
        return sampled_input, loss_targets


def _stack_latent_distributions(
    distributions: list[LatentDistribution],
) -> LatentDistribution:
    def concatenate(attribute: str) -> torch.Tensor | None:
        values = [getattr(distribution, attribute) for distribution in distributions]
        return None if any(value is None for value in values) else torch.cat(values, dim=0)

    return LatentDistribution(
        mu=torch.cat([distribution.mu for distribution in distributions], dim=0),
        logvar=torch.cat([distribution.logvar for distribution in distributions], dim=0),
        memory=concatenate("memory"),
        pre_normalized=concatenate("pre_normalized"),
        activity_mask=concatenate("activity_mask"),
        activity_memory=concatenate("activity_memory"),
    )
