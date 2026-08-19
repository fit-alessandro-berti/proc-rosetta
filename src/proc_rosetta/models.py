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


class SemanticProjection(nn.Module):
    def __init__(self, input_dim: int, semantic_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
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
        self.projection = SemanticProjection(hidden_dim, semantic_dim)
        self.dropout = nn.Dropout(dropout)
        self.max_sequence_length = max_sequence_length
        activity_ids = [
            tokenizer.token_to_id[name] for name in tokenizer.activity_tokens
        ]
        self.register_buffer(
            "activity_token_ids", torch.tensor(activity_ids, dtype=torch.long)
        )

    def _structure_features(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        depths = torch.zeros_like(tokens)
        parents = torch.zeros_like(tokens)
        operator_ids = {
            self.tokenizer.token_to_id[name] for name in self.tokenizer.operator_tokens
        }
        arity_by_id = {
            self.tokenizer.token_to_id[name]: int(name.split("_", 1)[1])
            for name in self.tokenizer.arity_tokens
        }
        rows = tokens.detach().cpu().tolist()
        for row_index, row in enumerate(rows):
            frames: list[list[int]] = [[1, 0, 0]]
            pending_operator: tuple[int, int] | None = None
            for position, token_id in enumerate(row):
                if token_id == self.tokenizer.pad_id:
                    break
                while frames and frames[-1][0] == 0:
                    frames.pop()
                depth = max(0, len(frames) - 1)
                parent = frames[-1][1] if frames else 0
                if pending_operator is not None and token_id in arity_by_id:
                    operator_position, operator_depth = pending_operator
                    depths[row_index, position] = operator_depth
                    parents[row_index, position] = operator_position
                    frames.append([arity_by_id[token_id], operator_position, operator_depth + 1])
                    pending_operator = None
                    continue
                depths[row_index, position] = min(depth, 63)
                parents[row_index, position] = min(parent, self.max_sequence_length - 1)
                if position == 0:
                    continue
                if frames:
                    frames[-1][0] -= 1
                if token_id in operator_ids:
                    pending_operator = (position, depth)
        return depths.to(tokens.device), parents.to(tokens.device)

    def forward(self, tokens: torch.Tensor) -> LatentDistribution:
        if tokens.shape[1] > self.max_sequence_length:
            raise ValueError("tree token sequence exceeds encoder position capacity")
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        depths, parents = self._structure_features(tokens)
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
        self.projection = SemanticProjection(hidden_dim, semantic_dim)
        self.dropout = nn.Dropout(dropout)

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
        activity_token_ids = torch.tensor(
            [
                self.tokenizer.token_to_id[f"A{index}"]
                for index in range(self.tokenizer.max_activities)
            ],
            dtype=torch.long,
            device=tokens.device,
        )
        activity_memory, activity_mask = _pool_activity_occurrences(
            outputs.reshape(batch_size, trace_count * trace_length, -1),
            tokens.reshape(batch_size, trace_count * trace_length),
            activity_token_ids,
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
        self.projection = SemanticProjection(hidden_dim, semantic_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_types: torch.Tensor,
        markings: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
        transition_label_ids: torch.Tensor | None = None,
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
        activity_token_ids = torch.tensor(
            [
                self.tokenizer.token_to_id[f"A{index}"]
                for index in range(self.tokenizer.max_activities)
            ],
            dtype=torch.long,
            device=node_types.device,
        )
        activity_memory, activity_mask = _pool_activity_occurrences(
            h,
            transition_label_ids,
            activity_token_ids,
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

    def source_memory(
        self, source: torch.Tensor | LatentDistribution
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(source, LatentDistribution):
            if source.memory is not None:
                return source.memory, source.activity_mask
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
        causal = torch.triu(
            torch.ones(
                input_tokens.shape[1],
                input_tokens.shape[1],
                dtype=torch.bool,
                device=input_tokens.device,
            ),
            diagonal=1,
        )
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
        if apply_grammar_mask:
            grammar = self.tokenizer.valid_next_token_masks(input_tokens)
            logits = logits.masked_fill(~grammar, -1e9)
        if activity_mask is not None:
            logits = self._mix_activity_copy(
                logits,
                hidden,
                activity_mask,
                activity_memory,
            )
            if apply_grammar_mask:
                logits = logits.masked_fill(~grammar, -1e9)
        return logits

    def _mix_activity_copy(
        self,
        logits: torch.Tensor,
        hidden: torch.Tensor,
        activity_mask: torch.Tensor,
        activity_memory: torch.Tensor | None,
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
        has_activity = activity_mask.any(dim=1, keepdim=True)
        safe_mask = activity_mask | ~has_activity
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

    @torch.no_grad()
    def decode_greedy(
        self,
        source: torch.Tensor | LatentDistribution,
        max_length: int = 128,
        apply_grammar_mask: bool = True,
        activity_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        activity_memory = (
            source.activity_memory
            if isinstance(source, LatentDistribution)
            else None
        )
        memory, inferred_mask = self.source_memory(source)
        activity_mask = inferred_mask if activity_mask is None else activity_mask
        batch_size = memory.shape[0]
        generated = torch.full(
            (batch_size, 1),
            self.tokenizer.bos_id,
            dtype=torch.long,
            device=memory.device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=memory.device)
        for _ in range(max_length - 1):
            logits = self.forward(
                memory,
                generated,
                apply_grammar_mask=apply_grammar_mask,
                activity_mask=activity_mask,
                activity_memory=activity_memory,
            )[:, -1]
            next_token = logits.argmax(dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.tokenizer.pad_id),
                next_token,
            )
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished |= next_token.eq(self.tokenizer.eos_id)
            if finished.all():
                break
        return generated

    @torch.no_grad()
    def decode_beam(
        self,
        source: torch.Tensor | LatentDistribution,
        max_length: int = 128,
        beam_size: int = 5,
        length_penalty: float = 0.7,
    ) -> torch.Tensor:
        if beam_size <= 1:
            return self.decode_greedy(source, max_length=max_length)
        candidates = self.decode_beam_candidates(
            source,
            max_length=max_length,
            beam_size=beam_size,
            length_penalty=length_penalty,
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
    ) -> list[list[tuple[list[int], float]]]:
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
            row_activity_memory = (
                None
                if activity_memory is None
                else activity_memory[row : row + 1]
            )
            beams: list[tuple[list[int], float]] = [([self.tokenizer.bos_id], 0.0)]
            for _ in range(max_length - 1):
                candidates: list[tuple[list[int], float]] = []
                for prefix, score in beams:
                    if prefix[-1] == self.tokenizer.eos_id:
                        candidates.append((prefix, score))
                        continue
                    tokens = torch.tensor([prefix], device=memory.device)
                    log_prob = F.log_softmax(
                        self.forward(
                            row_memory,
                            tokens,
                            activity_mask=row_mask,
                            activity_memory=row_activity_memory,
                        )[0, -1],
                        dim=-1,
                    )
                    values, indices = torch.topk(log_prob, beam_size)
                    candidates.extend(
                        (prefix + [int(token)], score + float(value))
                        for value, token in zip(values.tolist(), indices.tolist())
                    )
                beams = sorted(
                    candidates,
                    key=lambda item: item[1] / (len(item[0]) ** length_penalty),
                    reverse=True,
                )[:beam_size]
                if all(prefix[-1] == self.tokenizer.eos_id for prefix, _ in beams):
                    break
            candidate_rows.append(beams)
        return candidate_rows


class ProcRosettaModel(nn.Module):
    def __init__(
        self,
        tree_tokenizer: TreeTokenizer | None = None,
        activity_tokenizer: ActivityTokenizer | None = None,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        memory_tokens: int = 8,
        decoder_layers: int = 4,
    ) -> None:
        super().__init__()
        self.tree_tokenizer = tree_tokenizer or TreeTokenizer()
        self.activity_tokenizer = activity_tokenizer or ActivityTokenizer(
            max_activities=self.tree_tokenizer.max_activities
        )
        self.tree_encoder = TreeEncoder(
            self.tree_tokenizer,
            latent_dim,
            hidden_dim,
            dropout,
            layers=4,
            memory_tokens=memory_tokens,
        )
        self.trace_encoder = TraceEncoder(
            self.activity_tokenizer,
            latent_dim,
            hidden_dim,
            dropout,
            event_layers=2,
            set_layers=2,
            memory_tokens=memory_tokens,
        )
        self.petri_encoder = PetriGraphEncoder(
            self.activity_tokenizer,
            latent_dim,
            hidden_dim,
            message_passing_steps=5,
            dropout=dropout,
            memory_tokens=memory_tokens,
        )
        self.tree_decoder = GrammarTreeDecoder(
            self.tree_tokenizer,
            latent_dim,
            hidden_dim,
            dropout,
            layers=decoder_layers,
            memory_tokens=memory_tokens,
        )

    def encode_tree(self, tree_tokens: torch.Tensor) -> LatentDistribution:
        return self.tree_encoder(tree_tokens)

    def encode_traces(self, traces: dict[str, torch.Tensor]) -> LatentDistribution:
        return self.trace_encoder(traces["tokens"], traces["lengths"], traces["mask"])

    def encode_petri(self, petri: dict[str, torch.Tensor]) -> LatentDistribution:
        return self.petri_encoder(
            petri["node_types"],
            petri["markings"],
            petri["adjacency"],
            petri["node_mask"],
            petri.get("transition_label_ids"),
        )

    def forward(
        self,
        batch: dict[str, object],
        deterministic: bool = True,
        input_token_dropout: float = 0.0,
        scheduled_sampling_probability: float = 0.0,
    ) -> dict[str, object]:
        tree_tokens = batch["tree_tokens"]
        traces = batch["traces"]
        petri = batch["petri"]
        assert isinstance(tree_tokens, torch.Tensor)
        assert isinstance(traces, dict)
        assert isinstance(petri, dict)
        dists = {
            "tree": self.encode_tree(tree_tokens),
            "trace": self.encode_traces(traces),
            "petri": self.encode_petri(petri),
        }
        decoder_input = tree_tokens[:, :-1]
        logits: dict[str, torch.Tensor] = {}
        for name, distribution in dists.items():
            teacher_logits = self.tree_decoder(
                distribution,
                decoder_input,
                input_token_dropout=input_token_dropout,
            )
            if self.training and scheduled_sampling_probability > 0:
                sampled_input = decoder_input.clone()
                predictions = teacher_logits.detach().argmax(dim=-1)
                replacement = (
                    torch.rand(
                        sampled_input[:, 1:].shape,
                        device=sampled_input.device,
                    )
                    < scheduled_sampling_probability
                )
                replacement &= sampled_input[:, 1:].ne(self.tree_tokenizer.pad_id)
                sampled_input[:, 1:] = torch.where(
                    replacement,
                    predictions[:, :-1],
                    sampled_input[:, 1:],
                )
                teacher_logits = self.tree_decoder(
                    distribution,
                    sampled_input,
                    apply_grammar_mask=False,
                    input_token_dropout=input_token_dropout,
                )
            logits[name] = teacher_logits
        return {
            "dists": dists,
            "z": {name: distribution.mu for name, distribution in dists.items()},
            "tree_logits": logits,
        }
