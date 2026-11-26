"""Behavior Tree genome generation, structural/semantic mutations, and crossover."""

from __future__ import annotations

import random
from dataclasses import dataclass

from portal_ml.bt.ast import (
    ActionNode,
    BTNode,
    ConditionNode,
    SelectorNode,
    SequenceNode,
)

# --- Typed DSL vocabularies ---

GOAL_CONDITIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("is_open", ("exit",)),
    ("exists", ("pedestal_button",)),
    ("exists", ("cube",)),
    ("exists", ("floor_button",)),
    ("is_holding", ("cube",)),
    ("is_pressed", ("floor_button",)),
    ("is_pressed", ("pedestal_button",)),
    ("cube_on_button", ()),
)

GOAL_ACTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("SetTarget", ("exit",)),
    ("SetTarget", ("pedestal_button",)),
    ("SetTarget", ("cube",)),
    ("SetTarget", ("floor_button",)),
)

EXEC_CONDITIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("is_reachable", ("target",)),
    ("in_range", ("target",)),
    ("portals_bridge", ("target",)),
    ("portal_in_zone", ("blue", "here")),
    ("portal_in_zone", ("orange", "here")),
    ("portal_in_zone", ("blue", "target")),
    ("portal_in_zone", ("orange", "target")),
    ("portals_active", ()),
    ("target_is", ("exit",)),
    ("target_is", ("pedestal_button",)),
    ("target_is", ("cube",)),
    ("target_is", ("floor_button",)),
    ("is_holding", ("cube",)),
)

EXEC_ACTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("MoveTo", ("target",)),
    ("MoveTo", ("portal_here",)),
    ("Grab", ("target",)),
    ("Drop", ()),
    ("Press", ("target",)),
    ("ShootPortal", ("blue", "wall_here")),
    ("ShootPortal", ("orange", "wall_here")),
    ("ShootPortal", ("blue", "wall_near_target")),
    ("ShootPortal", ("orange", "wall_near_target")),
)

ALL_CONDITIONS = GOAL_CONDITIONS + EXEC_CONDITIONS
ALL_ACTIONS = GOAL_ACTIONS + EXEC_ACTIONS


@dataclass(slots=True)
class SubtreeVocab:
    conditions: tuple[tuple[str, tuple[str, ...]], ...]
    actions: tuple[tuple[str, tuple[str, ...]], ...]


GOAL_VOCAB = SubtreeVocab(GOAL_CONDITIONS, GOAL_ACTIONS)
EXEC_VOCAB = SubtreeVocab(EXEC_CONDITIONS, EXEC_ACTIONS)
FULL_VOCAB = SubtreeVocab(ALL_CONDITIONS, ALL_ACTIONS)


class GenomeFactory:
    """Generates random Behavior Tree genomes for initialization and mutation."""

    def __init__(self, mode: str = "decoupled", max_nodes: int = 50) -> None:
        if mode not in ("decoupled", "unconstrained"):
            raise ValueError(f"Unsupported mode {mode!r}; use 'decoupled' or 'unconstrained'")
        self.mode = mode
        self.max_nodes = max_nodes

    def random_genome(self, rng: random.Random) -> BTNode:
        if self.mode == "decoupled":
            goal = self.random_subtree(rng, GOAL_VOCAB, max_depth=3)
            exec_bt = self.random_subtree(rng, EXEC_VOCAB, max_depth=3)
            return SequenceNode([goal, exec_bt])
        return self.random_subtree(rng, FULL_VOCAB, max_depth=4)

    def random_condition(
        self, rng: random.Random, vocab: SubtreeVocab
    ) -> ConditionNode:
        pred, args = rng.choice(vocab.conditions)
        return ConditionNode(pred, args, negated=rng.random() < 0.45)

    def random_action(self, rng: random.Random, vocab: SubtreeVocab) -> ActionNode:
        name, args = rng.choice(vocab.actions)
        return ActionNode(name, args)

    def random_guarded_action(
        self, rng: random.Random, vocab: SubtreeVocab
    ) -> BTNode:
        r = rng.random()
        if r < 0.28:
            return self.random_action(rng, vocab)
        if r < 0.44:
            # Sequential two-step action chain (e.g., MoveTo(target) -> Drop())
            return SequenceNode(
                [self.random_action(rng, vocab), self.random_action(rng, vocab)]
            )
        n_conds = 1 if rng.random() < 0.80 else 2
        children: list[BTNode] = [
            self.random_condition(rng, vocab) for _ in range(n_conds)
        ]
        children.append(self.random_action(rng, vocab))
        return SequenceNode(children)

    def random_subtree(
        self, rng: random.Random, vocab: SubtreeVocab, max_depth: int = 3
    ) -> BTNode:
        if max_depth <= 1:
            return self.random_action(rng, vocab)
        if max_depth == 2:
            if rng.random() < 0.55:
                return self.random_guarded_action(rng, vocab)
            k = rng.randint(2, 3)
            return SelectorNode(
                [self.random_guarded_action(rng, vocab) for _ in range(k)]
            )
        k = rng.randint(2, 4)
        children = [
            self.random_subtree(rng, vocab, max_depth=max_depth - 1)
            for _ in range(k)
        ]
        return SelectorNode(children)


def _collect_nodes(
    node: BTNode,
    parent: SelectorNode | SequenceNode | None = None,
    idx: int = -1,
    out: list[tuple[BTNode, SelectorNode | SequenceNode | None, int]] | None = None,
) -> list[tuple[BTNode, SelectorNode | SequenceNode | None, int]]:
    if out is None:
        out = []
    out.append((node, parent, idx))
    if isinstance(node, (SelectorNode, SequenceNode)):
        for i, child in enumerate(node.children):
            _collect_nodes(child, node, i, out)
    return out


def _mutate_subtree(
    root: BTNode,
    rng: random.Random,
    factory: GenomeFactory,
    vocab: SubtreeVocab,
    max_nodes: int,
) -> BTNode:
    tree = root.clone()
    nodes = _collect_nodes(tree)
    op = rng.choices(
        population=[
            "point",
            "insert_branch",
            "delete_node",
            "swap_children",
            "wrap",
            "replace_subtree",
            "flip_composite",
        ],
        weights=[0.26, 0.22, 0.14, 0.14, 0.10, 0.10, 0.04],
        k=1,
    )[0]

    if op == "point":
        leaves = [n for n, _, _ in nodes if isinstance(n, (ConditionNode, ActionNode))]
        if leaves:
            leaf = rng.choice(leaves)
            if isinstance(leaf, ConditionNode):
                r = rng.random()
                if r < 0.45:
                    leaf.negated = not leaf.negated
                else:
                    pred, args = rng.choice(vocab.conditions)
                    leaf.predicate = pred
                    leaf.args = args
                    if rng.random() < 0.35:
                        leaf.negated = not leaf.negated
            elif isinstance(leaf, ActionNode):
                # Semantic portal-color / wall swap or action swap
                if leaf.name == "ShootPortal" and len(leaf.args) == 2 and rng.random() < 0.5:
                    color, wall = leaf.args
                    if rng.random() < 0.5:
                        color = "orange" if color == "blue" else "blue"
                    else:
                        wall = "wall_near_target" if wall == "wall_here" else "wall_here"
                    leaf.args = (color, wall)
                else:
                    name, args = rng.choice(vocab.actions)
                    leaf.name = name
                    leaf.args = args

    elif op == "insert_branch":
        composites = [
            n for n, _, _ in nodes if isinstance(n, (SelectorNode, SequenceNode))
        ]
        if composites and tree.size() < max_nodes - 3:
            comp = rng.choice(composites)
            if len(comp.children) < 6:
                if isinstance(comp, SelectorNode):
                    new_child = factory.random_guarded_action(rng, vocab)
                else:
                    new_child = (
                        factory.random_condition(rng, vocab)
                        if rng.random() < 0.7
                        else factory.random_action(rng, vocab)
                    )
                pos = rng.randint(0, len(comp.children))
                comp.children.insert(pos, new_child)

    elif op == "delete_node":
        removable = [
            (n, p, i)
            for n, p, i in nodes
            if p is not None and len(p.children) >= 2
        ]
        if removable:
            _, parent, idx = rng.choice(removable)
            assert parent is not None
            parent.children.pop(idx)

    elif op == "swap_children":
        swappable = [
            n
            for n, _, _ in nodes
            if isinstance(n, (SelectorNode, SequenceNode)) and len(n.children) >= 2
        ]
        if swappable:
            comp = rng.choice(swappable)
            i, j = rng.sample(range(len(comp.children)), 2)
            comp.children[i], comp.children[j] = comp.children[j], comp.children[i]

    elif op == "wrap":
        if tree.size() < max_nodes - 3:
            target, parent, idx = rng.choice(nodes)
            if rng.random() < 0.5:
                wrapped: BTNode = SequenceNode(
                    [factory.random_condition(rng, vocab), target.clone()]
                )
            else:
                sibling = factory.random_guarded_action(rng, vocab)
                children = (
                    [target.clone(), sibling]
                    if rng.random() < 0.5
                    else [sibling, target.clone()]
                )
                wrapped = SelectorNode(children)
            if parent is None:
                tree = wrapped
            else:
                parent.children[idx] = wrapped

    elif op == "replace_subtree":
        target, parent, idx = rng.choice(nodes)
        replacement = factory.random_subtree(rng, vocab, max_depth=2)
        if parent is None:
            tree = replacement
        else:
            parent.children[idx] = replacement

    elif op == "flip_composite":
        composites = [
            (n, p, i)
            for n, p, i in nodes
            if isinstance(n, (SelectorNode, SequenceNode))
        ]
        if composites:
            comp, parent, idx = rng.choice(composites)
            flipped: BTNode = (
                SequenceNode(comp.children)
                if isinstance(comp, SelectorNode)
                else SelectorNode(comp.children)
            )
            if parent is None:
                tree = flipped
            else:
                parent.children[idx] = flipped

    tree = simplify_tree(tree)
    if tree.size() > max_nodes:
        return root.clone()
    return tree


def mutate(
    genome: BTNode,
    rng: random.Random,
    factory: GenomeFactory,
) -> BTNode:
    """Apply a structural or semantic mutation to a Behavior Tree genome."""
    if (
        factory.mode == "decoupled"
        and isinstance(genome, SequenceNode)
        and len(genome.children) == 2
    ):
        goal_tree, exec_tree = genome.children[0], genome.children[1]
        half_budget = max(12, factory.max_nodes // 2)
        choice = rng.random()
        if choice < 0.45:
            new_goal = _mutate_subtree(goal_tree, rng, factory, GOAL_VOCAB, half_budget)
            return SequenceNode([new_goal, exec_tree.clone()])
        if choice < 0.90:
            new_exec = _mutate_subtree(exec_tree, rng, factory, EXEC_VOCAB, half_budget)
            return SequenceNode([goal_tree.clone(), new_exec])
        # Occasionally mutate both modules in one step
        new_goal = _mutate_subtree(goal_tree, rng, factory, GOAL_VOCAB, half_budget)
        new_exec = _mutate_subtree(exec_tree, rng, factory, EXEC_VOCAB, half_budget)
        return SequenceNode([new_goal, new_exec])

    return _mutate_subtree(genome, rng, factory, FULL_VOCAB, factory.max_nodes)


def _crossover_subtrees(
    a: BTNode, b: BTNode, rng: random.Random, max_nodes: int
) -> BTNode:
    child = a.clone()
    nodes_a = _collect_nodes(child)
    nodes_b = _collect_nodes(b)
    _, parent_a, idx_a = rng.choice(nodes_a)
    donor, _, _ = rng.choice(nodes_b)
    if parent_a is None:
        candidate = donor.clone()
    else:
        parent_a.children[idx_a] = donor.clone()
        candidate = child
    candidate = simplify_tree(candidate)
    if candidate.size() > max_nodes:
        return a.clone()
    return candidate


def crossover(
    parent_a: BTNode,
    parent_b: BTNode,
    rng: random.Random,
    factory: GenomeFactory,
) -> BTNode:
    """Subtree crossover between two parent Behavior Tree genomes."""
    if (
        factory.mode == "decoupled"
        and isinstance(parent_a, SequenceNode)
        and isinstance(parent_b, SequenceNode)
        and len(parent_a.children) == 2
        and len(parent_b.children) == 2
    ):
        half_budget = max(12, factory.max_nodes // 2)
        r = rng.random()
        if r < 0.35:
            # Module-level swap: combine A's goal policy with B's execution policy
            return SequenceNode([parent_a.children[0].clone(), parent_b.children[1].clone()])
        if r < 0.68:
            goal = _crossover_subtrees(
                parent_a.children[0], parent_b.children[0], rng, half_budget
            )
            return SequenceNode([goal, parent_a.children[1].clone()])
        exec_bt = _crossover_subtrees(
            parent_a.children[1], parent_b.children[1], rng, half_budget
        )
        return SequenceNode([parent_a.children[0].clone(), exec_bt])

    return _crossover_subtrees(parent_a, parent_b, rng, factory.max_nodes)


def simplify_tree(node: BTNode) -> BTNode:
    """Remove redundant 1-child composites and flatten identical nested composites."""
    if isinstance(node, (ConditionNode, ActionNode)):
        return node

    if isinstance(node, SelectorNode):
        flat: list[BTNode] = []
        for child in node.children:
            s = simplify_tree(child)
            if isinstance(s, SelectorNode):
                flat.extend(s.children)
            else:
                flat.append(s)
        if len(flat) == 1:
            return flat[0]
        return SelectorNode(flat)

    if isinstance(node, SequenceNode):
        flat = []
        for child in node.children:
            s = simplify_tree(child)
            if isinstance(s, SequenceNode):
                flat.extend(s.children)
            else:
                flat.append(s)
        if len(flat) == 1:
            return flat[0]
        return SequenceNode(flat)

    return node
