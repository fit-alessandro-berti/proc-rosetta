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
        if canonicalize:
            tree = tree.canonicalize_activity_labels()
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
                token = "ARITY_3" if self.max_arity >= 3 else "ARITY_2"
                valid[self.token_to_id[token]] = True
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
        masks = torch.zeros(
            (*prefixes.shape, self.vocab_size),
            dtype=torch.bool,
            device=prefixes.device,
        )
        flat_prefixes = prefixes.reshape(-1, prefixes.shape[-1])
        flat_masks = masks.reshape(-1, prefixes.shape[-1], self.vocab_size)
        for row_idx, row in enumerate(flat_prefixes):
            for pos in range(prefixes.shape[-1]):
                flat_masks[row_idx, pos] = self.next_token_mask(row[: pos + 1].tolist(), device=prefixes.device)
        return masks


@dataclass(frozen=True)
class ActivityTokenizer:
    max_activities: int = DEFAULT_MAX_ACTIVITIES

    def __post_init__(self) -> None:
        tokens = ["<pad>", *[f"A{idx}" for idx in range(self.max_activities)]]
        object.__setattr__(self, "tokens", tuple(tokens))
        object.__setattr__(self, "token_to_id", {token: idx for idx, token in enumerate(tokens)})

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode_trace(self, trace: Sequence[str]) -> list[int]:
        return [self.token_to_id[label] for label in trace]

    def encode_traces(self, traces: Sequence[Sequence[str]]) -> list[list[int]]:
        return [self.encode_trace(trace) for trace in traces]
