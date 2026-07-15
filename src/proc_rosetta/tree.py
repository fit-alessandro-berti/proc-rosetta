from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


class NodeKind(str, Enum):
    ACTIVITY = "activity"
    TAU = "tau"
    SEQ = "seq"
    XOR = "xor"
    AND = "and"
    LOOP = "loop"


COMMUTATIVE_KINDS = {NodeKind.XOR, NodeKind.AND}


@dataclass(frozen=True)
class ProcessTreeNode:
    """Small immutable process-tree representation used before pm4py conversion."""

    kind: NodeKind
    label: str | None = None
    children: tuple["ProcessTreeNode", ...] = ()

    def __post_init__(self) -> None:
        kind = NodeKind(self.kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "children", tuple(self.children))

        if kind is NodeKind.ACTIVITY:
            if not self.label:
                raise ValueError("activity nodes require a non-empty label")
            if self.children:
                raise ValueError("activity nodes cannot have children")
            return

        if kind is NodeKind.TAU:
            if self.label is not None:
                raise ValueError("tau nodes cannot have a label")
            if self.children:
                raise ValueError("tau nodes cannot have children")
            return

        if self.label is not None:
            raise ValueError("operator nodes cannot have labels")

        if kind in {NodeKind.SEQ, NodeKind.XOR, NodeKind.AND}:
            if len(self.children) < 2:
                raise ValueError(f"{kind.value} nodes require at least two children")
        elif kind is NodeKind.LOOP:
            if len(self.children) not in {2, 3}:
                raise ValueError("loop nodes require two or three children")

        if kind in COMMUTATIVE_KINDS:
            ordered = tuple(sorted(self.children, key=lambda child: child.canonical_key()))
            object.__setattr__(self, "children", ordered)

    @staticmethod
    def activity(label: str) -> "ProcessTreeNode":
        return ProcessTreeNode(NodeKind.ACTIVITY, label=label)

    @staticmethod
    def tau() -> "ProcessTreeNode":
        return ProcessTreeNode(NodeKind.TAU)

    @staticmethod
    def seq(*children: "ProcessTreeNode") -> "ProcessTreeNode":
        return ProcessTreeNode(NodeKind.SEQ, children=children)

    @staticmethod
    def xor(*children: "ProcessTreeNode") -> "ProcessTreeNode":
        return ProcessTreeNode(NodeKind.XOR, children=children)

    @staticmethod
    def and_(*children: "ProcessTreeNode") -> "ProcessTreeNode":
        return ProcessTreeNode(NodeKind.AND, children=children)

    @staticmethod
    def loop(
        body: "ProcessTreeNode",
        redo: "ProcessTreeNode",
        exit_: "ProcessTreeNode | None" = None,
    ) -> "ProcessTreeNode":
        children = (body, redo) if exit_ is None else (body, redo, exit_)
        return ProcessTreeNode(NodeKind.LOOP, children=children)

    @property
    def is_leaf(self) -> bool:
        return self.kind in {NodeKind.ACTIVITY, NodeKind.TAU}

    def activity_labels(self) -> tuple[str, ...]:
        labels: list[str] = []
        self._collect_activity_labels(labels)
        return tuple(labels)

    def _collect_activity_labels(self, labels: list[str]) -> None:
        if self.kind is NodeKind.ACTIVITY:
            assert self.label is not None
            labels.append(self.label)
        for child in self.children:
            child._collect_activity_labels(labels)

    def unique_activity_labels(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.activity_labels()))

    def size(self) -> int:
        return 1 + sum(child.size() for child in self.children)

    def max_depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(child.max_depth() for child in self.children)

    def canonical_key(self) -> tuple[Any, ...]:
        if self.kind is NodeKind.ACTIVITY:
            return (self.kind.value, self.label)
        return (self.kind.value, tuple(child.canonical_key() for child in self.children))

    def canonicalize_activity_labels(self, prefix: str = "A") -> "ProcessTreeNode":
        mapping: dict[str, str] = {}

        def visit(node: ProcessTreeNode) -> ProcessTreeNode:
            if node.kind is NodeKind.ACTIVITY:
                assert node.label is not None
                if node.label not in mapping:
                    mapping[node.label] = f"{prefix}{len(mapping)}"
                return ProcessTreeNode.activity(mapping[node.label])
            return ProcessTreeNode(node.kind, children=tuple(visit(child) for child in node.children))

        return visit(self)

    def relabel(self, mapping: dict[str, str]) -> "ProcessTreeNode":
        if self.kind is NodeKind.ACTIVITY:
            assert self.label is not None
            return ProcessTreeNode.activity(mapping.get(self.label, self.label))
        return ProcessTreeNode(self.kind, children=tuple(child.relabel(mapping) for child in self.children))

    def reassociate_operators(self, maximum_arity: int) -> "ProcessTreeNode":
        """Nest associative operators without dropping or reordering their behavior."""

        if maximum_arity < 2:
            raise ValueError("maximum operator arity must be at least two")
        if not self.children:
            return self
        children = tuple(
            child.reassociate_operators(maximum_arity) for child in self.children
        )
        if len(children) <= maximum_arity:
            return ProcessTreeNode(self.kind, children=children)
        if self.kind is NodeKind.LOOP:
            raise ValueError("loop operator arity cannot be reassociated safely")
        nested_tail = ProcessTreeNode(
            self.kind,
            children=children[maximum_arity - 1 :],
        ).reassociate_operators(maximum_arity)
        return ProcessTreeNode(
            self.kind,
            children=(*children[: maximum_arity - 1], nested_tail),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind.value}
        if self.label is not None:
            data["label"] = self.label
        if self.children:
            data["children"] = [child.to_dict() for child in self.children]
        return data

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ProcessTreeNode":
        return ProcessTreeNode(
            kind=NodeKind(data["kind"]),
            label=data.get("label"),
            children=tuple(ProcessTreeNode.from_dict(child) for child in data.get("children", ())),
        )

    def to_prefix_tokens(self) -> list[str]:
        if self.kind is NodeKind.ACTIVITY:
            assert self.label is not None
            return [self.label]
        if self.kind is NodeKind.TAU:
            return ["TAU"]

        tokens = [self.kind.name, f"ARITY_{len(self.children)}"]
        for child in self.children:
            tokens.extend(child.to_prefix_tokens())
        return tokens

    def __str__(self) -> str:
        if self.kind is NodeKind.ACTIVITY:
            return str(self.label)
        if self.kind is NodeKind.TAU:
            return "tau"
        inner = ", ".join(str(child) for child in self.children)
        return f"{self.kind.name}({inner})"


def activities(labels: Iterable[str]) -> tuple[ProcessTreeNode, ...]:
    return tuple(ProcessTreeNode.activity(label) for label in labels)
