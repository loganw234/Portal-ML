"""Behavior Tree Domain-Specific Language (DSL) AST, parser, and formatter.

Grammar (EBNF):
    node      := selector | sequence | condition | action ;
    selector  := "SELECTOR" "(" node ( "," node )* ")" ;
    sequence  := "SEQUENCE" "(" node ( "," node )* ")" ;
    condition := "CONDITION" "(" [ "NOT" ] ident [ "(" arg ( "," arg )* ")" ] ")" ;
    action    := "ACTION" "(" ident [ "(" arg ( "," arg )* ")" ] ")" ;
    arg       := ident ;
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class NodeStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


class BTNode:
    """Base class for all Behavior Tree AST nodes."""

    def clone(self) -> BTNode:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError

    def depth(self) -> int:
        raise NotImplementedError

    def to_dsl(self, indent: int = 0) -> str:
        raise NotImplementedError


@dataclass(slots=True)
class SelectorNode(BTNode):
    """Fallback composite: ticks children left-to-right until one succeeds/runs."""

    children: list[BTNode] = field(default_factory=list)

    def clone(self) -> SelectorNode:
        return SelectorNode([c.clone() for c in self.children])

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        return 1 + max((c.depth() for c in self.children), default=0)

    def to_dsl(self, indent: int = 0) -> str:
        pad = "  " * indent
        if not self.children:
            return f"{pad}SELECTOR()"
        inner = ",\n".join(c.to_dsl(indent + 1) for c in self.children)
        return f"{pad}SELECTOR(\n{inner}\n{pad})"


@dataclass(slots=True)
class SequenceNode(BTNode):
    """Sequence composite: ticks children left-to-right until one fails/runs."""

    children: list[BTNode] = field(default_factory=list)

    def clone(self) -> SequenceNode:
        return SequenceNode([c.clone() for c in self.children])

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        return 1 + max((c.depth() for c in self.children), default=0)

    def to_dsl(self, indent: int = 0) -> str:
        pad = "  " * indent
        if not self.children:
            return f"{pad}SEQUENCE()"
        inner = ",\n".join(c.to_dsl(indent + 1) for c in self.children)
        return f"{pad}SEQUENCE(\n{inner}\n{pad})"


@dataclass(slots=True)
class ConditionNode(BTNode):
    """Leaf predicate evaluated against the symbolic Blackboard."""

    predicate: str
    args: tuple[str, ...] = ()
    negated: bool = False

    def clone(self) -> ConditionNode:
        return ConditionNode(self.predicate, tuple(self.args), self.negated)

    def size(self) -> int:
        return 1

    def depth(self) -> int:
        return 1

    def to_dsl(self, indent: int = 0) -> str:
        pad = "  " * indent
        prefix = "NOT " if self.negated else ""
        if self.args:
            args_str = ", ".join(self.args)
            return f"{pad}CONDITION({prefix}{self.predicate}({args_str}))"
        return f"{pad}CONDITION({prefix}{self.predicate})"


@dataclass(slots=True)
class ActionNode(BTNode):
    """Leaf action: either a Blackboard update (SetTarget) or a Layer-3 primitive."""

    name: str
    args: tuple[str, ...] = ()

    def clone(self) -> ActionNode:
        return ActionNode(self.name, tuple(self.args))

    def size(self) -> int:
        return 1

    def depth(self) -> int:
        return 1

    def to_dsl(self, indent: int = 0) -> str:
        pad = "  " * indent
        if self.args:
            args_str = ", ".join(self.args)
            return f"{pad}ACTION({self.name}({args_str}))"
        return f"{pad}ACTION({self.name}())"


# --- Recursive-descent DSL parser ---

_TOKEN_RE = re.compile(r"\s*(?:(#[^\n]*)|([(),])|([A-Za-z_][A-Za-z0-9_:\-]*))")


class DSLParseError(ValueError):
    """Raised when a Behavior Tree DSL string has invalid syntax."""


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected: str | None = None) -> str:
        tok = self.peek()
        if tok is None:
            raise DSLParseError(f"Unexpected end of input (expected {expected!r})")
        if expected is not None and tok != expected:
            raise DSLParseError(
                f"Expected {expected!r} at token {self.pos}, got {tok!r}"
            )
        self.pos += 1
        return tok

    def parse_node(self) -> BTNode:
        kw = self.consume()
        if kw == "SELECTOR":
            return SelectorNode(self._parse_node_list())
        if kw == "SEQUENCE":
            return SequenceNode(self._parse_node_list())
        if kw == "CONDITION":
            return self._parse_condition()
        if kw == "ACTION":
            return self._parse_action()
        raise DSLParseError(f"Unknown node type {kw!r}")

    def _parse_node_list(self) -> list[BTNode]:
        self.consume("(")
        children: list[BTNode] = []
        if self.peek() != ")":
            children.append(self.parse_node())
            while self.peek() == ",":
                self.consume(",")
                if self.peek() == ")":
                    break
                children.append(self.parse_node())
        self.consume(")")
        return children

    def _parse_args(self) -> tuple[str, ...]:
        if self.peek() != "(":
            return ()
        self.consume("(")
        args: list[str] = []
        if self.peek() != ")":
            args.append(self.consume())
            while self.peek() == ",":
                self.consume(",")
                args.append(self.consume())
        self.consume(")")
        return tuple(args)

    def _parse_condition(self) -> ConditionNode:
        self.consume("(")
        negated = False
        if self.peek() == "NOT":
            self.consume("NOT")
            negated = True
        pred = self.consume()
        args = self._parse_args()
        self.consume(")")
        return ConditionNode(predicate=pred, args=args, negated=negated)

    def _parse_action(self) -> ActionNode:
        self.consume("(")
        name = self.consume()
        args = self._parse_args()
        self.consume(")")
        return ActionNode(name=name, args=args)


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    idx = 0
    n = len(text)
    while idx < n:
        match = _TOKEN_RE.match(text, idx)
        if not match:
            snippet = text[idx : idx + 20]
            raise DSLParseError(f"Unexpected character near {snippet!r}")
        idx = match.end()
        if match.group(1):
            continue  # comment
        tok = match.group(2) or match.group(3)
        if tok is not None:
            tokens.append(tok)
    return tokens


def parse_bt(text: str) -> BTNode:
    """Parse a Behavior Tree DSL string into a BTNode AST."""
    tokens = _tokenize(text)
    if not tokens:
        raise DSLParseError("Empty Behavior Tree DSL")
    parser = _Parser(tokens)
    node = parser.parse_node()
    if parser.peek() is not None:
        raise DSLParseError(f"Trailing tokens starting at {parser.peek()!r}")
    return node


def format_bt(node: BTNode) -> str:
    """Format a BTNode AST into clean, human-readable DSL text."""
    return node.to_dsl(indent=0)
