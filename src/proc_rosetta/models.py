from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from proc_rosetta.tokenizers import ActivityTokenizer, TreeTokenizer


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
    separate concept.
    """

    allowed_activity_slots: torch.Tensor | None = None
    constrain_to_source_activities: bool = True
    avoid_duplicate_activity_labels: bool = True
    duplicate_policy: str = "disallow"

    def __post_init__(self) -> None:
        if self.duplicate_policy not in {"disallow", "penalize", "allow"}:
            raise ValueError("duplicate_policy must be 'disallow', 'penalize', or 'allow'")


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
                flat_aggregation = torch.zeros_like(flat_h)
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
    ) -> torch.Tensor:
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
            self.tokenizer.valid_next_token_masks(input_tokens)
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
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Evaluate one decoder position while caching causal self-attention keys."""

        x = self.embedding(input_token).unsqueeze(1)
        x = x + self.position_embedding(
            torch.full_like(input_token, position)
        ).unsqueeze(1)
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

    def _incremental_grammar_mask(
        self,
        open_nodes: torch.Tensor,
        pending_operator: torch.Tensor,
    ) -> torch.Tensor:
        mask = torch.zeros(
            (open_nodes.shape[0], self.tokenizer.vocab_size),
            dtype=torch.bool,
            device=open_nodes.device,
        )
        awaiting = pending_operator.ne(0)
        ordinary = awaiting & pending_operator.eq(1)
        loop = awaiting & pending_operator.eq(2)
        for arity in range(2, self.tokenizer.max_arity + 1):
            token_id = self.tokenizer.token_to_id[f"ARITY_{arity}"]
            mask[ordinary, token_id] = True
            if arity in {2, 3}:
                mask[loop, token_id] = True
        complete = ~awaiting & open_nodes.eq(0)
        mask[complete, self.tokenizer.eos_id] = True
        need_node = ~awaiting & open_nodes.gt(0)
        mask[need_node, self.tokenizer.token_to_id["TAU"]] = True
        rows = torch.where(need_node)[0]
        if rows.numel():
            mask[rows.unsqueeze(1), self.activity_token_ids.unsqueeze(0)] = True
            operator_ids = self.structural_token_ids[4:]
            mask[rows.unsqueeze(1), operator_ids.unsqueeze(0)] = True
        return mask

    def _advance_incremental_grammar(
        self,
        chosen: torch.Tensor,
        open_nodes: torch.Tensor,
        pending_operator: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        awaiting = active & pending_operator.ne(0)
        for arity in range(2, self.tokenizer.max_arity + 1):
            selected = awaiting & chosen.eq(
                self.tokenizer.token_to_id[f"ARITY_{arity}"]
            )
            open_nodes[selected] += arity
        pending_operator[awaiting] = 0
        need_node = active & ~awaiting & open_nodes.gt(0)
        leaf = chosen.eq(self.tokenizer.token_to_id["TAU"])
        leaf |= chosen.unsqueeze(-1).eq(self.activity_token_ids.view(1, -1)).any(dim=-1)
        operator_ids = {
            self.tokenizer.token_to_id[name] for name in self.tokenizer.operator_tokens
        }
        operator = torch.zeros_like(active)
        for token_id in operator_ids:
            operator |= chosen.eq(token_id)
        consumed = need_node & (leaf | operator)
        open_nodes[consumed] -= 1
        pending_operator[need_node & operator] = 1
        pending_operator[
            need_node & chosen.eq(self.tokenizer.token_to_id["LOOP"])
        ] = 2

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
        constraints: DecodeConstraints | None = None,
    ) -> torch.Tensor:
        if constraints is not None:
            allowed_activity_mask = constraints.allowed_activity_slots
            constrain_to_source_activities = constraints.constrain_to_source_activities
            avoid_duplicate_activity_labels = constraints.avoid_duplicate_activity_labels
            duplicate_policy = constraints.duplicate_policy
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
            )
        for _ in range(max_length - 1):
            active_rows = torch.where(~finished)[0]
            if active_rows.numel() == 0:
                break
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
            active_logits = self.forward(
                memory[active_rows],
                generated[active_rows],
                apply_grammar_mask=apply_grammar_mask,
                activity_mask=active_activity_mask,
                activity_memory=active_activity_memory,
                allowed_activity_mask=active_allowed,
                avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
                duplicate_policy=duplicate_policy,
                used_activity_mask=used_activity_mask[active_rows],
            )[:, -1]
            next_token = torch.full(
                (batch_size,),
                self.tokenizer.pad_id,
                dtype=torch.long,
                device=memory.device,
            )
            next_token[active_rows] = active_logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
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
        for position in range(max_length - 1):
            hidden, caches = self._incremental_hidden(
                current,
                position,
                memory,
                caches,
            )
            logits = self._project_hidden(hidden)
            one_token = current.unsqueeze(1)
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
                used_activity_mask=used,
            )
            logits = logits.masked_fill(~hard_constraint, -torch.inf)
            active = ~finished
            next_token = logits[:, 0].argmax(dim=-1)
            next_token = torch.where(
                active,
                next_token,
                torch.full_like(next_token, self.tokenizer.pad_id),
            )
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
        constraints: DecodeConstraints | None = None,
    ) -> torch.Tensor:
        if beam_size <= 1:
            return self.decode_greedy(
                source,
                max_length=max_length,
                allowed_activity_mask=allowed_activity_mask,
                constrain_to_source_activities=constrain_to_source_activities,
                avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
                duplicate_policy=duplicate_policy,
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
        constraints: DecodeConstraints | None = None,
    ) -> list[list[tuple[list[int], float]]]:
        if constraints is not None:
            allowed_activity_mask = constraints.allowed_activity_slots
            constrain_to_source_activities = constraints.constrain_to_source_activities
            avoid_duplicate_activity_labels = constraints.avoid_duplicate_activity_labels
            duplicate_policy = constraints.duplicate_policy
        if not constrain_to_source_activities:
            allowed_activity_mask = None
        memory, activity_mask = self.source_memory(source)
        activity_memory = (
            source.activity_memory
            if isinstance(source, LatentDistribution)
            else None
        )
        candidate_rows: list[list[tuple[list[int], float]]] = []
        for row in range(memory.shape[0]):
            row_memory = memory[row : row + 1]
            row_mask = None if activity_mask is None else activity_mask[row : row + 1]
            if allowed_activity_mask is None or allowed_activity_mask.ndim == 1:
                row_allowed = allowed_activity_mask
            elif allowed_activity_mask.shape[0] > 1:
                row_allowed = allowed_activity_mask[row : row + 1]
            else:
                row_allowed = allowed_activity_mask
            row_activity_memory = (
                None
                if activity_memory is None
                else activity_memory[row : row + 1]
            )
            beams: list[tuple[list[int], float, torch.Tensor]] = [
                (
                    [self.tokenizer.bos_id],
                    0.0,
                    torch.zeros(
                        self.tokenizer.max_activities,
                        dtype=torch.bool,
                        device=memory.device,
                    ),
                )
            ]
            for _ in range(max_length - 1):
                candidates: list[tuple[list[int], float, torch.Tensor]] = []
                for prefix, score, used_activity_mask in beams:
                    if prefix[-1] == self.tokenizer.eos_id:
                        candidates.append((prefix, score, used_activity_mask))
                        continue
                    tokens = torch.tensor([prefix], device=memory.device)
                    log_prob = F.log_softmax(
                        self.forward(
                            row_memory,
                            tokens,
                            activity_mask=row_mask,
                            activity_memory=row_activity_memory,
                            allowed_activity_mask=row_allowed,
                            avoid_duplicate_activity_labels=avoid_duplicate_activity_labels,
                            duplicate_policy=duplicate_policy,
                            used_activity_mask=used_activity_mask.unsqueeze(0),
                        )[0, -1],
                        dim=-1,
                    )
                    legal_ids = torch.where(torch.isfinite(log_prob))[0]
                    legal_count = int(legal_ids.numel())
                    if legal_count == 0:
                        continue
                    k = min(beam_size, legal_count)
                    values, legal_indices = torch.topk(log_prob[legal_ids], k)
                    indices = legal_ids[legal_indices]
                    for value, token in zip(values.tolist(), indices.tolist()):
                        next_used = used_activity_mask.clone()
                        activity_slot = torch.where(self.activity_token_ids.eq(int(token)))[0]
                        if activity_slot.numel():
                            next_used[int(activity_slot[0])] = True
                        candidates.append(
                            (prefix + [int(token)], score + float(value), next_used)
                        )
                if not candidates:
                    break
                beams = sorted(
                    candidates,
                    key=lambda item: item[1] / (len(item[0]) ** length_penalty),
                    reverse=True,
                )[:beam_size]
                if all(prefix[-1] == self.tokenizer.eos_id for prefix, _, _ in beams):
                    break
            candidate_rows.append([(prefix, score) for prefix, score, _ in beams])
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
        dists = {
            "tree": self.encode_tree(
                tree_tokens,
                tree_structure if isinstance(tree_structure, dict) else None,
            ),
            "trace": self.encode_traces(traces),
            "petri": self.encode_petri(petri),
        }
        names = ("tree", "trace", "petri")
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
        teacher_logits = self.tree_decoder(
            stacked_distribution,
            stacked_input,
            allowed_activity_mask=stacked_allowed,
            input_token_dropout=input_token_dropout,
        )
        if self.training and scheduled_sampling_probability > 0:
            teacher_logits = self._scheduled_sampling_logits(
                stacked_distribution,
                stacked_input,
                stacked_allowed,
                probability=scheduled_sampling_probability,
                input_token_dropout=input_token_dropout,
            )
        split_logits = teacher_logits.split(batch_size, dim=0)
        logits = dict(zip(names, split_logits))
        contrastive_embeddings = {
            name: F.normalize(self.contrastive_head(dists[name].mu), dim=-1)
            for name in names
        }
        return {
            "dists": dists,
            "z": {name: distribution.mu for name, distribution in dists.items()},
            "contrastive_embeddings": contrastive_embeddings,
            "tree_logits": logits,
            "decoder_targets": decoder_targets,
        }

    def _scheduled_sampling_logits(
        self,
        source: LatentDistribution,
        teacher_input: torch.Tensor,
        allowed_activity_mask: torch.Tensor | None,
        *,
        probability: float,
        input_token_dropout: float,
    ) -> torch.Tensor:
        """Sample a legal prefix one token at a time.

        Every predicted replacement is produced under the same grammar and
        source-activity constraints as normal decoding.  A teacher token that
        became illegal under a sampled prefix is replaced unconditionally, and
        rows are padded after EOS/PAD instead of sampling beyond termination.
        """

        sampled_input = torch.full_like(teacher_input, self.tree_tokenizer.pad_id)
        sampled_input[:, 0] = teacher_input[:, 0]
        active = sampled_input[:, 0].ne(self.tree_tokenizer.eos_id)
        step_logits: list[torch.Tensor] = []
        for position in range(teacher_input.shape[1]):
            prefix = sampled_input[:, : position + 1]
            logits = self.tree_decoder(
                source,
                prefix,
                allowed_activity_mask=allowed_activity_mask,
                input_token_dropout=input_token_dropout,
            )
            current = logits[:, -1]
            step_logits.append(current)
            if position + 1 >= teacher_input.shape[1]:
                continue

            teacher_next = teacher_input[:, position + 1]
            prediction = current.detach().argmax(dim=-1)
            teacher_is_legal = torch.isfinite(
                current.detach().gather(1, teacher_next.unsqueeze(1)).squeeze(1)
            )
            replace = (
                torch.rand(teacher_next.shape, device=teacher_next.device) < probability
            )
            replace = active & (replace | ~teacher_is_legal)
            next_token = torch.where(replace, prediction, teacher_next)
            next_token = torch.where(
                active,
                next_token,
                torch.full_like(next_token, self.tree_tokenizer.pad_id),
            )
            sampled_input[:, position + 1] = next_token
            active = active & next_token.ne(self.tree_tokenizer.eos_id)
            active = active & next_token.ne(self.tree_tokenizer.pad_id)
        return torch.stack(step_logits, dim=1)


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
