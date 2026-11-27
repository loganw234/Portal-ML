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

    def test_single_portal_gun_remaps_orange_to_blue_and_protects_orange_zone(self) -> None:
        env = MockPortalEnv()
        snap = env.reset(4)
        # Valve pre-places Orange portal in zone 1; player is in zone 0 with blue-only gun
        snap.orange_portal.active = True
        snap.orange_portal.zone_id = 1
        snap.orange_portal.wall_id = "c4_wall_b"
        tree = SequenceNode(
            [
                ActionNode("SetTarget", ("exit",)),
                ActionNode("ShootPortal", ("orange", "wall_here")),
            ]
        )
        ev = BehaviorTreeEvaluator(tree, allow_orange_portal=False)
        res = ev.tick(snap)
        self.assertIs(res.status, NodeStatus.RUNNING)
        self.assertEqual(res.command, ShootPortal(color=PortalColor.BLUE, wall_id="c4_wall_a"))

        # Must refuse to waste Blue portal on the zone Valve's Orange portal already covers
        tree_same_zone = SequenceNode(
            [
                ActionNode("SetTarget", ("exit",)),
                ActionNode("ShootPortal", ("blue", "wall_at_target")),
            ]
        )
        ev2 = BehaviorTreeEvaluator(tree_same_zone, allow_orange_portal=False)
        res2 = ev2.tick(snap)
        self.assertIs(res2.status, NodeStatus.FAILURE)


if __name__ == "__main__":
    unittest.main()
