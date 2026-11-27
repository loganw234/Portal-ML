"""Tests for the 3D mock simulator across Chambers 1-5."""

from __future__ import annotations

import unittest

from portal_ml.bt import BehaviorTreeEvaluator, get_reference_tree
from portal_ml.motor import MoveTo, NoOp, Press
from portal_ml.schema import EntityKind
from portal_ml.sim import MockPortalEnv


class TestMockPortalEnv(unittest.TestCase):
    def test_reference_solver_passes_all_chambers(self) -> None:
        env = MockPortalEnv()
        tree = get_reference_tree()
        for cid in (1, 2, 3, 4, 5, 6, 7):
            ev = BehaviorTreeEvaluator(tree)
            snap = env.reset(cid)
            steps = 0
            while not snap.done:
                res = env.step(ev.tick(snap).command)
                snap = res.snapshot
                steps += 1
            self.assertTrue(snap.success, f"Chamber {cid} should be solved")
            self.assertLess(steps, env.spec.max_steps)

    def test_training_causal_augmentations_solvable_by_reference(self) -> None:
        env = MockPortalEnv()
        tree = get_reference_tree()
        for cid, aug in ((3, "ch3_distractor_walls"), (4, "ch4_stale_portals")):
            ev = BehaviorTreeEvaluator(tree)
            env.reset(cid)
            snap = env.apply_training_augmentation(aug)
            if aug == "ch3_distractor_walls":
                self.assertGreaterEqual(len(snap.portal_walls), 2)
            else:
                self.assertTrue(snap.blue_portal.active and snap.orange_portal.active)
                self.assertEqual(snap.blue_portal.zone_id, 0)
                self.assertEqual(snap.orange_portal.zone_id, 0)
            while not snap.done:
                snap = env.step(ev.tick(snap).command).snapshot
            self.assertTrue(snap.success, f"Reference solver must solve {aug}")

    def test_chamber_2_pedestal_timer_closes_door(self) -> None:
        env = MockPortalEnv(dt=0.5)
        snap = env.reset(2)
        # Walk to pedestal and press it
        for _ in range(15):
            snap = env.step(MoveTo(entity_id="c2_pedestal_0")).snapshot
        snap = env.step(Press(entity_id="c2_pedestal_0")).snapshot
        exit_ent = snap.first_of_kind(EntityKind.EXIT_DOOR)
        self.assertIsNotNone(exit_ent)
        self.assertTrue(exit_ent.active)
        # Wait past the 8.0s timer without entering the exit
        for _ in range(20):
            snap = env.step(NoOp()).snapshot
        exit_ent = snap.first_of_kind(EntityKind.EXIT_DOOR)
        self.assertFalse(exit_ent.active, "Timed pedestal door must re-close after timer expires")

    def test_chamber_4_cannot_walk_across_pit_without_portals(self) -> None:
        env = MockPortalEnv()
        snap = env.reset(4)
        for _ in range(50):
            snap = env.step(MoveTo(entity_id="c4_exit_0")).snapshot
        self.assertEqual(snap.player.zone_id, 0)
        self.assertFalse(snap.success)

    def test_in_place_reset_clears_portals_and_held_cube(self) -> None:
        env = MockPortalEnv()
        ev = BehaviorTreeEvaluator(get_reference_tree())
        snap = env.reset(5)
        for _ in range(15):
            snap = env.step(ev.tick(snap).command).snapshot
        # Reset immediately to Chamber 1
        snap1 = env.reset(1)
        self.assertEqual(snap1.chamber_id, 1)
        self.assertFalse(snap1.blue_portal.active)
        self.assertFalse(snap1.orange_portal.active)
        self.assertIsNone(snap1.player.held_entity_id)


if __name__ == "__main__":
    unittest.main()
