from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import torch

from proc_rosetta.synthetic import DEFAULT_MAX_ACTIVITIES
from proc_rosetta.tree import NodeKind, ProcessTreeNode


class GrammarState(str, Enum):
    NEED_NODE = "need_node"
    NEED_ARITY = "need_arity"
    DONE = "done"
    INVALID = "invalid"


@dataclass(frozen=True)
class TreeTokenizer:
    max_activities: int = DEFAULT_MAX_ACTIVITIES
    max_arity: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_arity", max(3, self.max_arity))
        tokens = ["<pad>", "<bos>", "<eos>", "TAU"]
        tokens.extend(kind.name for kind in (NodeKind.SEQ, NodeKind.XOR, NodeKind.AND, NodeKind.LOOP))
        tokens.extend(f"ARITY_{arity}" for arity in range(2, self.max_arity + 1))
        tokens.extend(f"A{idx}" for idx in range(self.max_activities))
        object.__setattr__(self, "tokens", tuple(tokens))
        object.__setattr__(self, "token_to_id", {token: idx for idx, token in enumerate(tokens)})

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def activity_tokens(self) -> tuple[str, ...]:
        return tuple(f"A{idx}" for idx in range(self.max_activities))

    @property
    def operator_tokens(self) -> tuple[str, ...]:
        return tuple(kind.name for kind in (NodeKind.SEQ, NodeKind.XOR, NodeKind.AND, NodeKind.LOOP))

    @property
    def arity_tokens(self) -> tuple[str, ...]:
        return tuple(f"ARITY_{arity}" for arity in range(2, self.max_arity + 1))

    def encode_tree(self, tree: ProcessTreeNode, canonicalize: bool = True) -> list[int]:
        tree = tree.normalize(
            self.max_arity,
            canonicalize_activity_labels=canonicalize,
        )
        tokens = ["<bos>", *tree.to_prefix_tokens(), "<eos>"]
        try:
            return [self.token_to_id[token] for token in tokens]
        except KeyError as exc:
            raise ValueError(f"tree token outside tokenizer vocabulary: {exc.args[0]}") from exc

    def decode_tree(self, token_ids: Sequence[int]) -> ProcessTreeNode:
        tokens = [self.tokens[int(token_id)] for token_id in token_ids]
        if tokens and tokens[0] == "<bos>":
            tokens = tokens[1:]
        if "<eos>" in tokens:
            tokens = tokens[: tokens.index("<eos>")]
        tokens = [token for token in tokens if token != "<pad>"]

        def parse_at(index: int) -> tuple[ProcessTreeNode, int]:
            if index >= len(tokens):
                raise ValueError("unexpected end of tree token stream")
            token = tokens[index]
            if self._is_activity_token(token):
                return ProcessTreeNode.activity(token), index + 1
            if token == "TAU":
                return ProcessTreeNode.tau(), index + 1
            if token not in self.operator_tokens:
                raise ValueError(f"expected tree node token, got {token}")
            if index + 1 >= len(tokens) or not tokens[index + 1].startswith("ARITY_"):
                raise ValueError(f"operator {token} missing arity token")
            arity = int(tokens[index + 1].split("_", 1)[1])
            children = []
            offset = index + 2
            for _ in range(arity):
                child, offset = parse_at(offset)
                children.append(child)
            return ProcessTreeNode(NodeKind[token], children=tuple(children)), offset

        tree, next_index = parse_at(0)
        if next_index != len(tokens):
            raise ValueError("extra tokens after valid process tree")
        return tree

    def next_token_mask(self, prefix_ids: Sequence[int], device: torch.device | None = None) -> torch.Tensor:
        valid = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
        state, pending_operator, open_nodes = self._grammar_state(prefix_ids)

        if state is GrammarState.INVALID:
            valid[self.eos_id] = True
            return valid
        if state is GrammarState.DONE:
            valid[self.pad_id] = True
            return valid
        if state is GrammarState.NEED_ARITY:
            assert pending_operator is not None
            if pending_operator == "LOOP":
                for arity in (2, 3):
                    if arity <= self.max_arity:
                        valid[self.token_to_id[f"ARITY_{arity}"]] = True
            else:
                for token in self.arity_tokens:
                    valid[self.token_to_id[token]] = True
            return valid

        if open_nodes == 0:
            valid[self.eos_id] = True
            return valid

        valid[self.token_to_id["TAU"]] = True
        for token in self.activity_tokens:
            valid[self.token_to_id[token]] = True
        for token in self.operator_tokens:
            valid[self.token_to_id[token]] = True
        return valid

    def _grammar_state(self, prefix_ids: Sequence[int]) -> tuple[GrammarState, str | None, int]:
        tokens = [self.tokens[int(token_id)] for token_id in prefix_ids if int(token_id) != self.pad_id]
        if not tokens:
            return GrammarState.INVALID, None, 0
        if tokens[0] != "<bos>":
            return GrammarState.INVALID, None, 0

        open_nodes = 1
        pending_operator: str | None = None
        ended = False
        for token in tokens[1:]:
            if ended:
                return GrammarState.DONE if token == "<pad>" else (GrammarState.INVALID, None, 0)[0], None, 0
            if pending_operator is not None:
                if not token.startswith("ARITY_"):
                    return GrammarState.INVALID, None, 0
                arity = int(token.split("_", 1)[1])
                if arity < 2 or arity > self.max_arity:
                    return GrammarState.INVALID, None, 0
                if pending_operator == "LOOP" and arity not in {2, 3}:
                    return GrammarState.INVALID, None, 0
                open_nodes += arity
                pending_operator = None
                continue

            if open_nodes == 0:
                if token == "<eos>":
                    ended = True
                    continue
                return GrammarState.INVALID, None, 0

            if self._is_activity_token(token) or token == "TAU":
                open_nodes -= 1
            elif token in self.operator_tokens:
                open_nodes -= 1
                pending_operator = token
            else:
                return GrammarState.INVALID, None, 0

        if ended:
            return GrammarState.DONE, None, 0
        if pending_operator is not None:
            return GrammarState.NEED_ARITY, pending_operator, open_nodes
        return GrammarState.NEED_NODE, None, open_nodes

    @staticmethod
    def _is_activity_token(token: str) -> bool:
        return len(token) > 1 and token[0] == "A" and token[1:].isdigit()

    def valid_next_token_masks(self, prefixes: torch.Tensor) -> torch.Tensor:
        """Compute every prefix mask with one vectorized recurrence over time."""

        masks = torch.zeros(
            (*prefixes.shape, self.vocab_size),
            dtype=torch.bool,
            device=prefixes.device,
        )
        flat_prefixes = prefixes.reshape(-1, prefixes.shape[-1])
        flat_masks = masks.reshape(-1, prefixes.shape[-1], self.vocab_size)
        batch = flat_prefixes.shape[0]
        started = torch.zeros(batch, dtype=torch.bool, device=prefixes.device)
        invalid = torch.zeros_like(started)
        ended = torch.zeros_like(started)
        open_nodes = torch.zeros(batch, dtype=torch.long, device=prefixes.device)
        # 0 = none, 1 = ordinary associative operator, 2 = loop.
        pending = torch.zeros(batch, dtype=torch.long, device=prefixes.device)
        activity_ids = torch.tensor(
            [self.token_to_id[token] for token in self.activity_tokens],
            device=prefixes.device,
        )
        operator_ids = torch.tensor(
            [self.token_to_id[token] for token in self.operator_tokens],
            device=prefixes.device,
        )
        loop_id = self.token_to_id["LOOP"]

        for position in range(prefixes.shape[-1]):
            token = flat_prefixes[:, position]
            non_padding = token.ne(self.pad_id)
            active = non_padding & ~invalid

            invalid |= active & ended
            active &= ~ended & ~invalid

            was_started = started.clone()
            starting = active & ~was_started
            invalid |= starting & token.ne(self.bos_id)
            valid_start = starting & token.eq(self.bos_id)
            started |= valid_start
            open_nodes = torch.where(valid_start, torch.ones_like(open_nodes), open_nodes)

            consuming = active & was_started & ~invalid
            awaiting_arity = consuming & pending.ne(0)
            arity = torch.zeros_like(open_nodes)
            is_arity = torch.zeros_like(started)
            for value in range(2, self.max_arity + 1):
                matches = token.eq(self.token_to_id[f"ARITY_{value}"])
                arity = torch.where(matches, torch.full_like(arity, value), arity)
                is_arity |= matches
            valid_arity = awaiting_arity & is_arity
            valid_arity &= (pending.ne(2) | arity.eq(2) | arity.eq(3))
            invalid |= awaiting_arity & ~valid_arity
            open_nodes = torch.where(valid_arity, open_nodes + arity, open_nodes)
            pending = torch.where(valid_arity, torch.zeros_like(pending), pending)

            consuming_node = consuming & ~awaiting_arity & ~invalid
            at_completed_root = consuming_node & open_nodes.eq(0)
            valid_end = at_completed_root & token.eq(self.eos_id)
            invalid |= at_completed_root & ~valid_end
            ended |= valid_end

            need_node = consuming_node & open_nodes.gt(0)
            is_activity = token.unsqueeze(-1).eq(activity_ids.view(1, -1)).any(dim=-1)
            is_leaf = is_activity | token.eq(self.token_to_id["TAU"])
            is_operator = token.unsqueeze(-1).eq(operator_ids.view(1, -1)).any(dim=-1)
            valid_node = need_node & (is_leaf | is_operator)
            invalid |= need_node & ~valid_node
            open_nodes = torch.where(valid_node, open_nodes - 1, open_nodes)
            pending = torch.where(
                need_node & is_operator,
                torch.where(
                    token.eq(loop_id),
                    torch.full_like(pending, 2),
                    torch.ones_like(pending),
                ),
                pending,
            )

            invalid_state = invalid | ~started
            flat_masks[invalid_state, position, self.eos_id] = True
            flat_masks[ended & ~invalid, position, self.pad_id] = True
            need_arity = started & ~invalid & ~ended & pending.ne(0)
            ordinary_arity = need_arity & pending.eq(1)
            loop_arity = need_arity & pending.eq(2)
            for value in range(2, self.max_arity + 1):
                token_id = self.token_to_id[f"ARITY_{value}"]
                flat_masks[ordinary_arity, position, token_id] = True
                if value in {2, 3}:
                    flat_masks[loop_arity, position, token_id] = True
            completed = started & ~invalid & ~ended & pending.eq(0) & open_nodes.eq(0)
            flat_masks[completed, position, self.eos_id] = True
            node_rows = started & ~invalid & ~ended & pending.eq(0) & open_nodes.gt(0)
            flat_masks[node_rows, position, self.token_to_id["TAU"]] = True
            node_indices = torch.where(node_rows)[0]
            if node_indices.numel():
                flat_masks[
                    node_indices.unsqueeze(1),
                    position,
                    activity_ids.unsqueeze(0),
                ] = True
                flat_masks[
                    node_indices.unsqueeze(1),
                    position,
                    operator_ids.unsqueeze(0),
                ] = True
        return masks

    def valid_next_token_masks_reference(self, prefixes: torch.Tensor) -> torch.Tensor:
        """Quadratic reference implementation retained for equivalence tests."""

        masks = torch.zeros(
            (*prefixes.shape, self.vocab_size),
            dtype=torch.bool,
            device=prefixes.device,
        )
        flat_prefixes = prefixes.reshape(-1, prefixes.shape[-1])
        flat_masks = masks.reshape(-1, prefixes.shape[-1], self.vocab_size)
        for row_idx, row in enumerate(flat_prefixes):
            for pos in range(prefixes.shape[-1]):
                flat_masks[row_idx, pos] = self.next_token_mask(
                    row[: pos + 1].tolist(),
                    device=prefixes.device,
                )
        return masks

    def structure_features(
        self,
        tokens: torch.Tensor,
        *,
        max_sequence_length: int = 1024,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute depth and parent positions on CPU during collation."""

        depths = torch.zeros_like(tokens)
        parents = torch.zeros_like(tokens)
        operator_ids = {self.token_to_id[name] for name in self.operator_tokens}
        arity_by_id = {
            self.token_to_id[name]: int(name.split("_", 1)[1])
            for name in self.arity_tokens
        }
        for row_index, row in enumerate(tokens.tolist()):
            frames: list[list[int]] = [[1, 0, 0]]
            pending_operator: tuple[int, int] | None = None
            for position, token_id in enumerate(row):
                if token_id == self.pad_id:
                    break
                while frames and frames[-1][0] == 0:
                    frames.pop()
                depth = max(0, len(frames) - 1)
                parent = frames[-1][1] if frames else 0
                if pending_operator is not None and token_id in arity_by_id:
                    operator_position, operator_depth = pending_operator
                    depths[row_index, position] = operator_depth
                    parents[row_index, position] = operator_position
                    frames.append(
                        [arity_by_id[token_id], operator_position, operator_depth + 1]
                    )
                    pending_operator = None
                    continue
                depths[row_index, position] = min(depth, 63)
                parents[row_index, position] = min(parent, max_sequence_length - 1)
                if position == 0:
                    continue
                if frames:
                    frames[-1][0] -= 1
                if token_id in operator_ids:
                    pending_operator = (position, depth)
        return depths, parents


@dataclass(frozen=True)
class ActivityTokenizer:
    max_activities: int = DEFAULT_MAX_ACTIVITIES

    def __post_init__(self) -> None:
        tokens = ["<pad>", "<unk>", *[f"A{idx}" for idx in range(self.max_activities)]]
        object.__setattr__(self, "tokens", tuple(tokens))
        object.__setattr__(self, "token_to_id", {token: idx for idx, token in enumerate(tokens)})

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode_trace(self, trace: Sequence[str]) -> list[int]:
        return [self.token_to_id.get(label, self.unk_id) for label in trace]

    def encode_traces(self, traces: Sequence[Sequence[str]]) -> list[list[int]]:
        return [self.encode_trace(trace) for trace in traces]
