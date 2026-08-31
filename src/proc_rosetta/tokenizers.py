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

    def legal_arities(self, operator: str | int) -> tuple[int, ...]:
        """Return the grammar's legal arities for one operator token."""

        name = self.tokens[int(operator)] if isinstance(operator, int) else operator
        if name not in self.operator_tokens:
            raise ValueError(f"not an operator token: {name}")
        if name == "LOOP":
            return tuple(arity for arity in (2, 3) if arity <= self.max_arity)
        return tuple(range(2, self.max_arity + 1))

    def minimum_legal_arity(self, operator: str | int) -> int:
        legal = self.legal_arities(operator)
        if not legal:
            raise ValueError(f"operator has no legal arity: {operator}")
        return min(legal)

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

    def next_token_mask(
        self,
        prefix_ids: Sequence[int],
        device: torch.device | None = None,
        *,
        remaining_tokens: int | None = None,
    ) -> torch.Tensor:
        """Return the next-token grammar mask, optionally bounded by a budget.

        ``remaining_tokens`` includes the token about to be selected. Omitting
        it preserves the historical prefix-only grammar behavior.
        """

        state, pending_operator, open_nodes = self._grammar_state(prefix_ids)

        if state is GrammarState.INVALID:
            if remaining_tokens is not None:
                raise ValueError("bounded completion requires a valid grammar prefix")
            valid = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
            valid[self.eos_id] = True
            return valid
        if state is GrammarState.DONE:
            valid = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
            valid[self.pad_id] = True
            return valid
        pending_id = 0 if pending_operator is None else self.token_to_id[pending_operator]
        open_tensor = torch.tensor([open_nodes], dtype=torch.long, device=device)
        pending_tensor = torch.tensor([pending_id], dtype=torch.long, device=device)
        grammar = self.prefix_grammar_mask(open_tensor, pending_tensor)[0]
        if remaining_tokens is None:
            return grammar
        completion = self.completion_feasibility_mask(
            open_tensor,
            pending_tensor,
            remaining_tokens,
        )[0]
        bounded = grammar & completion
        if not bounded.any():
            raise ValueError("no grammar-legal completion fits the remaining token budget")
        return bounded

    def prefix_grammar_mask(
        self,
        open_nodes: torch.Tensor,
        pending_operator: torch.Tensor,
    ) -> torch.Tensor:
        """Build prefix-legal masks from the shared incremental grammar state.

        A zero ``pending_operator`` means that no arity token is pending;
        otherwise it is the vocabulary ID of the operator awaiting its arity.
        """

        if open_nodes.shape != pending_operator.shape:
            raise ValueError("grammar state tensors must have identical shapes")
        if (open_nodes < 0).any():
            raise ValueError("grammar state cannot contain negative open slots")
        flat_open = open_nodes.reshape(-1)
        flat_pending = pending_operator.reshape(-1)
        mask = torch.zeros(
            (flat_open.shape[0], self.vocab_size),
            dtype=torch.bool,
            device=open_nodes.device,
        )
        awaiting = flat_pending.ne(0)
        known_pending = torch.zeros_like(awaiting)
        for operator in self.operator_tokens:
            operator_id = self.token_to_id[operator]
            rows = awaiting & flat_pending.eq(operator_id)
            known_pending |= rows
            for arity in self.legal_arities(operator):
                mask[rows, self.token_to_id[f"ARITY_{arity}"]] = True
        if (awaiting & ~known_pending).any():
            raise ValueError("grammar state contains an unknown pending operator")

        complete = ~awaiting & flat_open.eq(0)
        mask[complete, self.eos_id] = True
        need_node = ~awaiting & flat_open.gt(0)
        mask[need_node, self.token_to_id["TAU"]] = True
        rows = torch.where(need_node)[0]
        if rows.numel():
            activity_ids = torch.tensor(
                [self.token_to_id[token] for token in self.activity_tokens],
                dtype=torch.long,
                device=open_nodes.device,
            )
            operator_ids = torch.tensor(
                [self.token_to_id[token] for token in self.operator_tokens],
                dtype=torch.long,
                device=open_nodes.device,
            )
            mask[rows.unsqueeze(1), activity_ids.unsqueeze(0)] = True
            mask[rows.unsqueeze(1), operator_ids.unsqueeze(0)] = True
        return mask.reshape((*open_nodes.shape, self.vocab_size))

    def minimum_tokens_to_finish(
        self,
        open_nodes: torch.Tensor,
        pending_operator: torch.Tensor,
    ) -> torch.Tensor:
        """Minimum continuation length, including a still-unemitted EOS."""

        if open_nodes.shape != pending_operator.shape:
            raise ValueError("grammar state tensors must have identical shapes")
        minimum = open_nodes + 1
        for operator in self.operator_tokens:
            rows = pending_operator.eq(self.token_to_id[operator])
            minimum = torch.where(
                rows,
                open_nodes + self.minimum_legal_arity(operator) + 2,
                minimum,
            )
        unknown = pending_operator.ne(0)
        for operator in self.operator_tokens:
            unknown &= pending_operator.ne(self.token_to_id[operator])
        if unknown.any():
            raise ValueError("grammar state contains an unknown pending operator")
        return minimum

    def shortest_completion(
        self,
        open_nodes: int,
        pending_operator: int | str | None,
    ) -> list[int]:
        """Return the deterministic shortest legal continuation through EOS.

        ``open_nodes`` and ``pending_operator`` describe the grammar state after
        the current prefix.  The returned continuation never includes BOS.
        Choosing TAU for every remaining child makes this independent of the
        source alphabet and duplicate-activity constraints.
        """

        open_count = int(open_nodes)
        if open_count < 0:
            raise ValueError("grammar state cannot contain negative open slots")
        pending: str | None
        if pending_operator is None or pending_operator == 0:
            pending = None
        elif isinstance(pending_operator, str):
            pending = pending_operator
        else:
            pending_id = int(pending_operator)
            if not 0 <= pending_id < self.vocab_size:
                raise ValueError("grammar state contains an unknown pending operator")
            pending = self.tokens[pending_id]
        completion: list[int] = []
        if pending is not None:
            arity = self.minimum_legal_arity(pending)
            completion.append(self.token_to_id[f"ARITY_{arity}"])
            open_count += arity
        completion.extend([self.token_to_id["TAU"]] * open_count)
        completion.append(self.eos_id)
        return completion

    def validate_complete_tree_sequence(
        self,
        token_ids: Sequence[int],
        *,
        token_budget: int,
    ) -> ProcessTreeNode:
        """Strictly validate a deployment token stream and decode its tree.

        The budget includes BOS and EOS. Padding is permitted only after the
        first EOS, which allows a padded batch row to satisfy the same strict
        postcondition as an unpadded result.
        """

        ids = [int(token_id) for token_id in token_ids]
        if not ids:
            raise ValueError("empty token sequence")
        if ids[0] != self.bos_id:
            raise ValueError("missing BOS")
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in ids):
            raise ValueError("token sequence contains an ID outside the vocabulary")
        if self.eos_id not in ids:
            raise ValueError("missing EOS")
        eos_position = ids.index(self.eos_id)
        consumed = ids[: eos_position + 1]
        suffix = ids[eos_position + 1 :]
        if len(consumed) > int(token_budget):
            raise ValueError("tree exceeds token budget")
        if any(token_id != self.pad_id for token_id in suffix):
            raise ValueError("non-padding token after EOS")
        if self.pad_id in consumed:
            raise ValueError("padding token before EOS")
        return self.decode_tree(consumed)

    def completion_feasibility_mask(
        self,
        open_nodes: torch.Tensor,
        pending_operator: torch.Tensor,
        remaining_tokens: int | torch.Tensor,
    ) -> torch.Tensor:
        """Keep candidates that leave a complete tree and EOS within budget."""

        remaining = torch.as_tensor(
            remaining_tokens,
            dtype=torch.long,
            device=open_nodes.device,
        )
        try:
            remaining = torch.broadcast_to(remaining, open_nodes.shape).reshape(-1)
        except RuntimeError as exc:
            raise ValueError("remaining token budget has incompatible shape") from exc
        if (remaining < 1).any():
            raise ValueError("remaining token budget must include the next token")

        flat_open = open_nodes.reshape(-1)
        flat_pending = pending_operator.reshape(-1)
        feasible = torch.ones(
            (flat_open.shape[0], self.vocab_size),
            dtype=torch.bool,
            device=open_nodes.device,
        )
        awaiting = flat_pending.ne(0)
        leaf_feasible = remaining >= flat_open + 1
        leaf_ids = [self.token_to_id["TAU"]]
        leaf_ids.extend(self.token_to_id[token] for token in self.activity_tokens)
        feasible[:, leaf_ids] = leaf_feasible.unsqueeze(1)

        for operator in self.operator_tokens:
            operator_id = self.token_to_id[operator]
            feasible[:, operator_id] = (
                remaining
                >= flat_open + self.minimum_legal_arity(operator) + 2
            )

        for arity in range(2, self.max_arity + 1):
            arity_id = self.token_to_id[f"ARITY_{arity}"]
            feasible[:, arity_id] = remaining >= flat_open + arity + 2

        feasible[:, self.eos_id] = (~awaiting & flat_open.eq(0) & remaining.ge(1))
        return feasible.reshape((*open_nodes.shape, self.vocab_size))

    def advance_grammar_state(
        self,
        chosen: torch.Tensor,
        open_nodes: torch.Tensor,
        pending_operator: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        """Advance shared grammar state in place after a legal selection."""

        if not (chosen.shape == open_nodes.shape == pending_operator.shape == active.shape):
            raise ValueError("chosen token and grammar state tensors must have identical shapes")
        awaiting = active & pending_operator.ne(0)
        for arity in range(2, self.max_arity + 1):
            selected = awaiting & chosen.eq(self.token_to_id[f"ARITY_{arity}"])
            open_nodes[selected] += arity
        pending_operator[awaiting] = 0

        need_node = active & ~awaiting & open_nodes.gt(0)
        leaf = chosen.eq(self.token_to_id["TAU"])
        for token in self.activity_tokens:
            leaf |= chosen.eq(self.token_to_id[token])
        operator = torch.zeros_like(active)
        for token in self.operator_tokens:
            selected = need_node & chosen.eq(self.token_to_id[token])
            operator |= selected
            pending_operator[selected] = self.token_to_id[token]
        consumed = need_node & (leaf | operator)
        open_nodes[consumed] -= 1

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
                if arity not in self.legal_arities(pending_operator):
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

    def valid_next_token_masks(
        self,
        prefixes: torch.Tensor,
        *,
        remaining_tokens: int | torch.Tensor | None = None,
        allow_infeasible_prefixes: bool = False,
    ) -> torch.Tensor:
        """Compute every prefix mask with one vectorized recurrence over time.

        With a scalar budget, the value applies to the final prefix and earlier
        positions receive the positions between them and the final prefix.
        Omitting the budget is exactly the historical teacher-forcing behavior.
        ``allow_infeasible_prefixes`` returns an empty bounded mask when a valid
        prefix can no longer finish within its budget.  This is useful when
        scoring sampled training prefixes; bounded decoding remains strict by
        default.
        """

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
        # 0 = none; otherwise the actual pending operator vocabulary ID.
        pending = torch.zeros(batch, dtype=torch.long, device=prefixes.device)
        activity_ids = torch.tensor(
            [self.token_to_id[token] for token in self.activity_tokens],
            device=prefixes.device,
        )
        operator_ids = torch.tensor(
            [self.token_to_id[token] for token in self.operator_tokens],
            device=prefixes.device,
        )
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
            valid_arity = torch.zeros_like(awaiting_arity)
            for operator in self.operator_tokens:
                operator_rows = awaiting_arity & pending.eq(self.token_to_id[operator])
                legal = torch.zeros_like(operator_rows)
                for value in self.legal_arities(operator):
                    legal |= arity.eq(value)
                valid_arity |= operator_rows & is_arity & legal
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
            pending = torch.where(need_node & is_operator, token, pending)

            invalid_state = invalid | ~started
            flat_masks[invalid_state, position, self.eos_id] = True
            flat_masks[ended & ~invalid, position, self.pad_id] = True
            grammar_rows = started & ~invalid & ~ended
            if grammar_rows.any():
                flat_masks[grammar_rows, position] = self.prefix_grammar_mask(
                    open_nodes[grammar_rows],
                    pending[grammar_rows],
                )

            if remaining_tokens is not None:
                if invalid.any():
                    raise ValueError("bounded completion requires valid grammar prefixes")
                budget = torch.as_tensor(
                    remaining_tokens,
                    dtype=torch.long,
                    device=prefixes.device,
                )
                if budget.ndim == 0:
                    position_budget = budget + prefixes.shape[-1] - 1 - position
                    position_budget = position_budget.expand(batch)
                elif budget.shape == prefixes.shape:
                    position_budget = budget.reshape(-1, prefixes.shape[-1])[:, position]
                else:
                    try:
                        final_budget = torch.broadcast_to(
                            budget,
                            prefixes.shape[:-1],
                        ).reshape(-1)
                    except RuntimeError as exc:
                        raise ValueError("remaining token budget has incompatible shape") from exc
                    position_budget = final_budget + prefixes.shape[-1] - 1 - position
                if grammar_rows.any():
                    completion = self.completion_feasibility_mask(
                        open_nodes[grammar_rows],
                        pending[grammar_rows],
                        position_budget[grammar_rows],
                    )
                    flat_masks[grammar_rows, position] &= completion
                    all_have_feasible_token = flat_masks[
                        grammar_rows, position
                    ].any(dim=-1).all()
                    if not allow_infeasible_prefixes and not all_have_feasible_token:
                        raise ValueError(
                            "no grammar-legal completion fits the remaining token budget"
                        )
        return masks

    def valid_next_token_masks_reference(
        self,
        prefixes: torch.Tensor,
        *,
        remaining_tokens: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
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
                budget = None
                if remaining_tokens is not None:
                    supplied = torch.as_tensor(remaining_tokens)
                    if supplied.ndim == 0:
                        budget = int(supplied) + prefixes.shape[-1] - 1 - pos
                    elif supplied.shape == prefixes.shape:
                        budget = int(supplied.reshape(-1, prefixes.shape[-1])[row_idx, pos])
                    else:
                        budget = int(
                            torch.broadcast_to(supplied, prefixes.shape[:-1]).reshape(-1)[row_idx]
                        ) + prefixes.shape[-1] - 1 - pos
                flat_masks[row_idx, pos] = self.next_token_mask(
                    row[: pos + 1].tolist(),
                    device=prefixes.device,
                    remaining_tokens=budget,
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
