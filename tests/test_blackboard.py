"""Tests for Layer 1 Blackboard schema and JSON serialization."""

from __future__ import annotations

import unittest

from portal_ml.schema import (
    ActivePortal,
    BlackboardSnapshot,
    EntityKind,
    EntityState,
    PlayerState,
    PortalColor,
    PortalWall,
    Vec3,
)


class TestBlackboardSchema(unittest.TestCase):
    def test_json_roundtrip(self) -> None:
        snap = BlackboardSnapshot(
            tick=17,
            dt=0.05,
            chamber_id=5,
            player=PlayerState(
                pos=Vec3(192.0, 8256.0, 0.0),
                vel=Vec3(10.0, 20.0, 0.0),
                yaw=90.0,
                pitch=-5.0,
                zone_id=0,
                held_entity_id="c5_cube_0",
            ),
            entities=[
                EntityState(
                    id="c5_cube_0",
                    kind=EntityKind.CUBE,
                    pos=Vec3(192.0, 8296.0, 40.0),
                    zone_id=0,
                    held=True,
                ),
                EntityState(
                    id="c5_exit_0",
                    kind=EntityKind.EXIT_DOOR,
                    pos=Vec3(96.0, 8224.0, 0.0),
                    zone_id=0,
                    active=False,
                ),
            ],
            portal_walls=[
                PortalWall(
                    id="c5_wall_a",
                    pos=Vec3(0.0, 8320.0, 64.0),
                    normal=Vec3(1.0, 0.0, 0.0),
                    zone_id=0,
                ),
                PortalWall(
                    id="c5_wall_b",
                    pos=Vec3(384.0, 8704.0, 64.0),
                    normal=Vec3(-1.0, 0.0, 0.0),
                    zone_id=1,
                ),
            ],
            blue_portal=ActivePortal(
                color=PortalColor.BLUE,
                active=True,
                wall_id="c5_wall_a",
                pos=Vec3(0.0, 8320.0, 64.0),
                normal=Vec3(1.0, 0.0, 0.0),
                zone_id=0,
            ),
            orange_portal=ActivePortal(
                color=PortalColor.ORANGE,
                active=True,
                wall_id="c5_wall_b",
                pos=Vec3(384.0, 8704.0, 64.0),
                normal=Vec3(-1.0, 0.0, 0.0),
                zone_id=1,
            ),
        )

        raw = snap.to_json()
        restored = BlackboardSnapshot.from_json(raw)
        self.assertEqual(restored.tick, 17)
        self.assertEqual(restored.chamber_id, 5)
        self.assertEqual(restored.player.held_entity_id, "c5_cube_0")
        self.assertEqual(len(restored.entities), 2)
        self.assertTrue(restored.portals_bridge_zones(0, 1))
        self.assertTrue(restored.portals_bridge_zones(1, 0))
        self.assertFalse(restored.portals_bridge_zones(0, 0))
        self.assertEqual(restored.wall_in_zone(1).id, "c5_wall_b")


if __name__ == "__main__":
    unittest.main()
