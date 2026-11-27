"""Tests for Layer 2 Behavior Tree DSL parser, formatter, and evaluator."""

from __future__ import annotations

import unittest

from portal_ml.bt import (
    ActionNode,
    BehaviorTreeEvaluator,
    ConditionNode,
    NodeStatus,
    REFERENCE_SOLVER_DSL,
    SelectorNode,
    SequenceNode,
    format_bt,
    get_reference_tree,
    parse_bt,
)
from portal_ml.bt.ast import DSLParseError
from portal_ml.motor import MoveTo, ShootPortal
from portal_ml.schema import PortalColor
from portal_ml.sim import MockPortalEnv


class TestBehaviorTreeDSL(unittest.TestCase):
    def test_parse_and_format_roundtrip(self) -> None:
        tree = get_reference_tree()
        formatted = format_bt(tree)
        reparsed = parse_bt(formatted)
        self.assertEqual(format_bt(reparsed), formatted)
        self.assertGreater(tree.size(), 10)
        self.assertGreaterEqual(tree.depth(), 4)

    def test_parse_invalid_raises(self) -> None:
        with self.assertRaises(DSLParseError):
            parse_bt("UNKNOWN_NODE(foo)")
        with self.assertRaises(DSLParseError):
            parse_bt("")
        with self.assertRaises(DSLParseError):
            parse_bt("ACTION(Drop()) EXTRA")

    def test_evaluator_emits_concrete_primitives(self) -> None:
        env = MockPortalEnv()
        snap = env.reset(1)
        tree = SequenceNode(
            [
                ConditionNode("is_open", ("exit",)),
                ActionNode("SetTarget", ("exit",)),
                ActionNode("MoveTo", ("target",)),
            ]
        )
        ev = BehaviorTreeEvaluator(tree)
        res = ev.tick(snap)
        self.assertIs(res.status, NodeStatus.RUNNING)
        self.assertEqual(res.command, MoveTo(entity_id="c1_exit_0"))

    def test_evaluator_portal_shooting_on_gap(self) -> None:
        env = MockPortalEnv()
        snap = env.reset(4)
        ev = BehaviorTreeEvaluator(parse_bt(REFERENCE_SOLVER_DSL))
        res1 = ev.tick(snap)
        self.assertEqual(
            res1.command, ShootPortal(color=PortalColor.BLUE, wall_id="c4_wall_a")
        )
        snap = env.step(res1.command).snapshot
        res2 = ev.tick(snap)
        self.assertEqual(
            res2.command, ShootPortal(color=PortalColor.ORANGE, wall_id="c4_wall_b")
        )

    def test_drop_enforces_target_locality_symmetry(self) -> None:
        env = MockPortalEnv()
        snap = env.reset(3)
        # Artificially hold cube while far from floor_button
        snap.player.held_entity_id = "c3_cube_0"
        cube = snap.entity_by_id("c3_cube_0")
        assert cube is not None
        cube.held = True
        tree = SequenceNode(
            [
                ActionNode("SetTarget", ("floor_button",)),
                ActionNode("Drop", ()),
            ]
        )
        ev = BehaviorTreeEvaluator(tree)
        # Far from floor_button -> Drop must fail (symmetric with Grab/Press)
        res_far = ev.tick(snap)
        self.assertIs(res_far.status, NodeStatus.FAILURE)
        # Move player onto floor_button -> Drop succeeds
        btn = snap.entity_by_id("c3_floor_btn_0")
        assert btn is not None
        snap.player.pos = btn.pos
        res_near = ev.tick(snap)
        self.assertIs(res_near.status, NodeStatus.RUNNING)
        self.assertEqual(res_near.command.kind.value, "drop")


if __name__ == "__main__":
    unittest.main()
