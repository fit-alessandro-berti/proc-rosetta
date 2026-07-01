from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from proc_rosetta.tokenizers import ActivityTokenizer, TreeTokenizer


@dataclass
class LatentDistribution:
    mu: torch.Tensor
    logvar: torch.Tensor

    def sample(self, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return self.mu
        std = torch.exp(0.5 * self.logvar)
        return self.mu + torch.randn_like(std) * std


class LatentProjection(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.mu = nn.Linear(input_dim, latent_dim)
        self.logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, features: torch.Tensor) -> LatentDistribution:
        return LatentDistribution(self.mu(features), self.logvar(features).clamp(min=-8.0, max=8.0))


class TreeEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        latent_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.projection = LatentProjection(hidden_dim, latent_dim)

    def forward(self, tokens: torch.Tensor) -> LatentDistribution:
        lengths = tokens.ne(0).sum(dim=1).clamp(min=1)
        embedded = self.dropout(self.embedding(tokens))
        outputs, _ = self.gru(embedded)
        final = outputs[torch.arange(tokens.shape[0], device=tokens.device), lengths - 1]
        final = self.dropout(final)
        return self.projection(final)


class TraceEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        latent_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.projection = LatentProjection(hidden_dim, latent_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        trace_mask: torch.Tensor,
    ) -> LatentDistribution:
        batch_size, trace_count, trace_length = tokens.shape
        flat_tokens = tokens.reshape(batch_size * trace_count, trace_length)
        flat_lengths = lengths.reshape(batch_size * trace_count).clamp(min=1)
        embedded = self.dropout(self.embedding(flat_tokens))
        outputs, _ = self.gru(embedded)
        final = outputs[
            torch.arange(flat_tokens.shape[0], device=tokens.device),
            flat_lengths.to(tokens.device) - 1,
        ]
        final = final.reshape(batch_size, trace_count, -1)
        final = final * trace_mask.unsqueeze(-1).to(final.dtype)

        scores = self.attention(final).squeeze(-1).masked_fill(~trace_mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = self.dropout((final * weights).sum(dim=1))
        return self.projection(pooled)


class PetriGraphEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        node_type_count: int = 3,
        message_passing_steps: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.node_embedding = nn.Embedding(node_type_count, hidden_dim)
        self.marking_projection = nn.Linear(2, hidden_dim)
        self.self_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(message_passing_steps))
        self.in_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(message_passing_steps))
        self.out_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(message_passing_steps))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(message_passing_steps))
        self.dropout = nn.Dropout(dropout)
        self.projection = LatentProjection(hidden_dim, latent_dim)

    def forward(
        self,
        node_types: torch.Tensor,
        markings: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> LatentDistribution:
        h = self.dropout(self.node_embedding(node_types) + self.marking_projection(markings))
        h = h * node_mask.unsqueeze(-1).to(h.dtype)
        adjacency_any = adjacency.sum(dim=1).clamp(max=1.0)

        for self_layer, in_layer, out_layer, norm in zip(
            self.self_layers,
            self.in_layers,
            self.out_layers,
            self.norms,
        ):
            incoming = torch.bmm(adjacency_any.transpose(1, 2), in_layer(h))
            outgoing = torch.bmm(adjacency_any, out_layer(h))
            updated = self_layer(h) + incoming + outgoing
            h = self.dropout(norm(F.relu(updated))) * node_mask.unsqueeze(-1).to(h.dtype)

        denominator = node_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        pooled = self.dropout(h.sum(dim=1) / denominator)
        return self.projection(pooled)


class GrammarTreeDecoder(nn.Module):
    def __init__(
        self,
        tokenizer: TreeTokenizer,
        latent_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding = nn.Embedding(tokenizer.vocab_size, hidden_dim, padding_idx=tokenizer.pad_id)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, tokenizer.vocab_size)

    def forward(
        self,
        z: torch.Tensor,
        input_tokens: torch.Tensor,
        apply_grammar_mask: bool = True,
    ) -> torch.Tensor:
        hidden = torch.tanh(self.latent_to_hidden(z)).unsqueeze(0)
        embedded = self.dropout(self.embedding(input_tokens))
        outputs, _ = self.gru(embedded, hidden)
        logits = self.output(self.dropout(outputs))
        if apply_grammar_mask:
            masks = self.tokenizer.valid_next_token_masks(input_tokens)
            logits = logits.masked_fill(~masks, -1e9)
        return logits

    @torch.no_grad()
    def decode_greedy(
        self,
        z: torch.Tensor,
        max_length: int = 128,
        apply_grammar_mask: bool = True,
    ) -> torch.Tensor:
        batch_size = z.shape[0]
        hidden = torch.tanh(self.latent_to_hidden(z)).unsqueeze(0)
        current = torch.full((batch_size, 1), self.tokenizer.bos_id, dtype=torch.long, device=z.device)
        generated = [current]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=z.device)

        for _ in range(max_length - 1):
            embedded = self.embedding(current)
            output, hidden = self.gru(embedded, hidden)
            logits = self.output(output[:, -1])
            if apply_grammar_mask:
                masks = torch.stack(
                    [
                        self.tokenizer.next_token_mask(
                            torch.cat(generated, dim=1)[row_idx].tolist(),
                            device=z.device,
                        )
                        for row_idx in range(batch_size)
                    ]
                )
                logits = logits.masked_fill(~masks, -1e9)
            next_token = logits.argmax(dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.tokenizer.pad_id),
                next_token,
            )
            current = next_token.unsqueeze(1)
            generated.append(current)
            finished = finished | next_token.eq(self.tokenizer.eos_id)
            if finished.all():
                break
        return torch.cat(generated, dim=1)


class ProcRosettaModel(nn.Module):
    def __init__(
        self,
        tree_tokenizer: TreeTokenizer | None = None,
        activity_tokenizer: ActivityTokenizer | None = None,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.tree_tokenizer = tree_tokenizer or TreeTokenizer()
        self.activity_tokenizer = activity_tokenizer or ActivityTokenizer(
            max_activities=self.tree_tokenizer.max_activities
        )
        self.tree_encoder = TreeEncoder(self.tree_tokenizer.vocab_size, latent_dim, hidden_dim, dropout)
        self.trace_encoder = TraceEncoder(self.activity_tokenizer.vocab_size, latent_dim, hidden_dim, dropout)
        self.petri_encoder = PetriGraphEncoder(latent_dim, hidden_dim, dropout=dropout)
        self.tree_decoder = GrammarTreeDecoder(self.tree_tokenizer, latent_dim, hidden_dim, dropout)

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
        )

    def forward(self, batch: dict[str, object], deterministic: bool = False) -> dict[str, object]:
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
        z = {name: dist.sample(deterministic=deterministic) for name, dist in dists.items()}
        decoder_input = tree_tokens[:, :-1]
        logits = {name: self.tree_decoder(latent, decoder_input) for name, latent in z.items()}
        return {"dists": dists, "z": z, "tree_logits": logits}
