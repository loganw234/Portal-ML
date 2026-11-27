"""5-Chamber Micro-Curriculum geometry shared by the mock 3D simulator and .vmf generator.

All five chambers are packed into a single coordinate space separated by 2048-unit
void gaps along Y, enabling sub-millisecond in-place resets in the Source Engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from portal_ml.schema.blackboard import EntityKind, PortalWall, Vec3


@dataclass(frozen=True, slots=True)
class WalkableZone:
    """Axis-aligned walkable floor region representing a single navmesh island."""

    zone_id: int
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    floor_z: float = 0.0

    def contains_xy(self, pos: Vec3) -> bool:
        return self.min_x <= pos.x <= self.max_x and self.min_y <= pos.y <= self.max_y

    def clamp_xy(self, pos: Vec3) -> Vec3:
        return Vec3(
            min(max(pos.x, self.min_x), self.max_x),
            min(max(pos.y, self.min_y), self.max_y),
            self.floor_z,
        )


@dataclass(frozen=True, slots=True)
class PitBox:
    """Hazard pit separating walkable zones (for .vmf brush carving)."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    depth: float = 128.0


@dataclass(frozen=True, slots=True)
class EntitySpawn:
    id: str
    kind: EntityKind
    pos: Vec3
    zone_id: int
    initially_active: bool = False
    pedestal_duration: float = 8.0


@dataclass(frozen=True, slots=True)
class ChamberSpec:
    chamber_id: int
    name: str
    description: str
    held_out: bool
    room_min: Vec3
    room_max: Vec3
    player_spawn: Vec3
    player_spawn_yaw: float
    player_spawn_zone: int
    zones: tuple[WalkableZone, ...]
    pits: tuple[PitBox, ...] = ()
    entities: tuple[EntitySpawn, ...] = ()
    portal_walls: tuple[PortalWall, ...] = ()
    max_steps: int = 200
    exit_starts_open: bool = False


def _build_chambers() -> dict[int, ChamberSpec]:
    # Chamber 1: The Hallway (Y offset = 0)
    c1 = ChamberSpec(
        chamber_id=1,
        name="The Hallway",
        description="Straight room with open exit door; validates bridge, stepping, and navigation.",
        held_out=False,
        room_min=Vec3(0.0, 0.0, 0.0),
        room_max=Vec3(512.0, 256.0, 256.0),
        player_spawn=Vec3(64.0, 128.0, 0.0),
        player_spawn_yaw=0.0,
        player_spawn_zone=0,
        zones=(WalkableZone(0, 16.0, 496.0, 16.0, 240.0, 0.0),),
        entities=(
            EntitySpawn(
                id="c1_exit_0",
                kind=EntityKind.EXIT_DOOR,
                pos=Vec3(480.0, 128.0, 0.0),
                zone_id=0,
                initially_active=True,
            ),
        ),
        max_steps=120,
        exit_starts_open=True,
    )

    # Chamber 2: The Pedestal (Y offset = 2048)
    y2 = 2048.0
    c2 = ChamberSpec(
        chamber_id=2,
        name="The Pedestal",
        description="Closed exit + timed pedestal button; validates 2-step causality.",
        held_out=False,
        room_min=Vec3(0.0, y2, 0.0),
        room_max=Vec3(512.0, y2 + 256.0, 256.0),
        player_spawn=Vec3(64.0, y2 + 128.0, 0.0),
        player_spawn_yaw=0.0,
        player_spawn_zone=0,
        zones=(WalkableZone(0, 16.0, 496.0, y2 + 16.0, y2 + 240.0, 0.0),),
        entities=(
            EntitySpawn(
                id="c2_pedestal_0",
                kind=EntityKind.PEDESTAL_BUTTON,
                pos=Vec3(224.0, y2 + 64.0, 0.0),
                zone_id=0,
                initially_active=False,
                pedestal_duration=8.0,
            ),
            EntitySpawn(
                id="c2_exit_0",
                kind=EntityKind.EXIT_DOOR,
                pos=Vec3(480.0, y2 + 128.0, 0.0),
                zone_id=0,
                initially_active=False,
            ),
        ),
        max_steps=160,
        exit_starts_open=False,
    )

    # Chamber 3: Box on Button (Y offset = 4096)
    y3 = 4096.0
    c3 = ChamberSpec(
        chamber_id=3,
        name="Box on Button",
        description="Weighted floor button + loose cube + closed exit; validates object permanence and state binding.",
        held_out=False,
        room_min=Vec3(0.0, y3, 0.0),
        room_max=Vec3(512.0, y3 + 384.0, 256.0),
        player_spawn=Vec3(64.0, y3 + 192.0, 0.0),
        player_spawn_yaw=0.0,
        player_spawn_zone=0,
        zones=(WalkableZone(0, 16.0, 496.0, y3 + 16.0, y3 + 368.0, 0.0),),
        entities=(
            EntitySpawn(
                id="c3_cube_0",
                kind=EntityKind.CUBE,
                pos=Vec3(192.0, y3 + 64.0, 0.0),
                zone_id=0,
            ),
            EntitySpawn(
                id="c3_floor_btn_0",
                kind=EntityKind.FLOOR_BUTTON,
                pos=Vec3(320.0, y3 + 304.0, 0.0),
                zone_id=0,
            ),
            EntitySpawn(
                id="c3_exit_0",
                kind=EntityKind.EXIT_DOOR,
                pos=Vec3(480.0, y3 + 192.0, 0.0),
                zone_id=0,
                initially_active=False,
            ),
        ),
        max_steps=200,
        exit_starts_open=False,
    )

    # Chamber 4: The Simple Gap (Y offset = 6144)
    y4 = 6144.0
    c4 = ChamberSpec(
        chamber_id=4,
        name="The Simple Gap",
        description="Pit separating player from exit with one portalable wall per side; validates non-Euclidean traversal.",
        held_out=False,
        room_min=Vec3(0.0, y4, 0.0),
        room_max=Vec3(640.0, y4 + 256.0, 256.0),
        player_spawn=Vec3(96.0, y4 + 128.0, 0.0),
        player_spawn_yaw=0.0,
        player_spawn_zone=0,
        zones=(
            WalkableZone(0, 16.0, 240.0, y4 + 16.0, y4 + 240.0, 0.0),
            WalkableZone(1, 400.0, 624.0, y4 + 16.0, y4 + 240.0, 0.0),
        ),
        pits=(PitBox(256.0, 384.0, y4, y4 + 256.0, depth=128.0),),
        entities=(
            EntitySpawn(
                id="c4_exit_0",
                kind=EntityKind.EXIT_DOOR,
                pos=Vec3(608.0, y4 + 128.0, 0.0),
                zone_id=1,
                initially_active=True,
            ),
        ),
        portal_walls=(
            PortalWall(
                id="c4_wall_a",
                pos=Vec3(128.0, y4, 64.0),
                normal=Vec3(0.0, 1.0, 0.0),
                zone_id=0,
                size=(128.0, 128.0),
            ),
            PortalWall(
                id="c4_wall_b",
                pos=Vec3(512.0, y4, 64.0),
                normal=Vec3(0.0, 1.0, 0.0),
                zone_id=1,
                size=(128.0, 128.0),
            ),
        ),
        max_steps=180,
        exit_starts_open=True,
    )

    # Chamber 5: Held-Out Generalization Test (Y offset = 8192, rotated 90 deg)
    # Spawn + floor button + closed exit on South platform (zone 0);
    # weighted cube across the East-West pit on North platform (zone 1).
    y5 = 8192.0
    c5 = ChamberSpec(
        chamber_id=5,
        name="Held-Out Generalization Test",
        description=(
            "Never seen during training. Rotated 90-degree combo of Chambers 3 and 4: "
            "cube across pit, floor button and closed exit at spawn."
        ),
        held_out=True,
        room_min=Vec3(0.0, y5, 0.0),
        room_max=Vec3(384.0, y5 + 640.0, 256.0),
        player_spawn=Vec3(192.0, y5 + 80.0, 0.0),
        player_spawn_yaw=90.0,
        player_spawn_zone=0,
        zones=(
            WalkableZone(0, 16.0, 368.0, y5 + 16.0, y5 + 240.0, 0.0),
            WalkableZone(1, 16.0, 368.0, y5 + 400.0, y5 + 624.0, 0.0),
        ),
        pits=(PitBox(0.0, 384.0, y5 + 256.0, y5 + 384.0, depth=128.0),),
        entities=(
            EntitySpawn(
                id="c5_cube_0",
                kind=EntityKind.CUBE,
                pos=Vec3(192.0, y5 + 528.0, 0.0),
                zone_id=1,
            ),
            EntitySpawn(
                id="c5_floor_btn_0",
                kind=EntityKind.FLOOR_BUTTON,
                pos=Vec3(304.0, y5 + 160.0, 0.0),
                zone_id=0,
            ),
            EntitySpawn(
                id="c5_exit_0",
                kind=EntityKind.EXIT_DOOR,
                pos=Vec3(96.0, y5 + 32.0, 0.0),
                zone_id=0,
                initially_active=False,
            ),
        ),
        portal_walls=(
            PortalWall(
                id="c5_wall_a",
                pos=Vec3(0.0, y5 + 128.0, 64.0),
                normal=Vec3(1.0, 0.0, 0.0),
                zone_id=0,
                size=(128.0, 128.0),
            ),
            PortalWall(
                id="c5_wall_b",
                pos=Vec3(384.0, y5 + 512.0, 64.0),
                normal=Vec3(-1.0, 0.0, 0.0),
                zone_id=1,
                size=(128.0, 128.0),
            ),
        ),
        max_steps=260,
        exit_starts_open=False,
    )

    # Chamber 6: Held-Out Generalization Test — Button Across Gap (Y offset = 10240)
    # Opposite asymmetry from Chamber 5: cube + spawn + closed exit on Zone 0;
    # weighted floor button across the pit on Zone 1 with opposite-facing portal walls.
    y6 = 10240.0
    c6 = ChamberSpec(
        chamber_id=6,
        name="Held-Out: Button Across Gap",
        description=(
            "Never seen during training. Cube and closed exit at spawn (zone 0); "
            "floor button across pit (zone 1). Must carry cube across, drop, and return."
        ),
        held_out=True,
        room_min=Vec3(0.0, y6, 0.0),
        room_max=Vec3(640.0, y6 + 384.0, 256.0),
        player_spawn=Vec3(96.0, y6 + 192.0, 0.0),
        player_spawn_yaw=0.0,
        player_spawn_zone=0,
        zones=(
            WalkableZone(0, 16.0, 240.0, y6 + 16.0, y6 + 368.0, 0.0),
            WalkableZone(1, 400.0, 624.0, y6 + 16.0, y6 + 368.0, 0.0),
        ),
        pits=(PitBox(256.0, 384.0, y6, y6 + 384.0, depth=128.0),),
        entities=(
            EntitySpawn(
                id="c6_cube_0",
                kind=EntityKind.CUBE,
                pos=Vec3(176.0, y6 + 64.0, 0.0),
                zone_id=0,
            ),
            EntitySpawn(
                id="c6_floor_btn_0",
                kind=EntityKind.FLOOR_BUTTON,
                pos=Vec3(528.0, y6 + 192.0, 0.0),
                zone_id=1,
            ),
            EntitySpawn(
                id="c6_exit_0",
                kind=EntityKind.EXIT_DOOR,
                pos=Vec3(48.0, y6 + 320.0, 0.0),
                zone_id=0,
                initially_active=False,
            ),
        ),
        portal_walls=(
            PortalWall(
                id="c6_wall_a",
                pos=Vec3(128.0, y6, 64.0),
                normal=Vec3(0.0, 1.0, 0.0),
                zone_id=0,
                size=(128.0, 128.0),
            ),
            PortalWall(
                id="c6_wall_b",
                pos=Vec3(512.0, y6 + 384.0, 64.0),
                normal=Vec3(0.0, -1.0, 0.0),
                zone_id=1,
                size=(128.0, 128.0),
            ),
        ),
        max_steps=260,
        exit_starts_open=False,
    )

    # Chamber 7: Held-Out Generalization Test — 3-Zone Two-Gap Relay (Y offset = 12288)
    # 3 disconnected islands separated by 2 pits:
    # Zone 0: spawn + closed exit; Zone 1: floor button; Zone 2: weighted cube.
    # Requires dynamically re-placing portals across 3 distinct bridges (0->2, 2->1, 1->0).
    y7 = 12288.0
    c7 = ChamberSpec(
        chamber_id=7,
        name="Held-Out: Three-Zone Relay",
        description=(
            "Never seen during training. Three islands separated by two pits: "
            "exit on zone 0, floor button on zone 1, cube on zone 2."
        ),
        held_out=True,
        room_min=Vec3(0.0, y7, 0.0),
        room_max=Vec3(1024.0, y7 + 256.0, 256.0),
        player_spawn=Vec3(128.0, y7 + 128.0, 0.0),
        player_spawn_yaw=0.0,
        player_spawn_zone=0,
        zones=(
            WalkableZone(0, 16.0, 240.0, y7 + 16.0, y7 + 240.0, 0.0),
            WalkableZone(1, 400.0, 624.0, y7 + 16.0, y7 + 240.0, 0.0),
            WalkableZone(2, 784.0, 1008.0, y7 + 16.0, y7 + 240.0, 0.0),
        ),
        pits=(
            PitBox(256.0, 384.0, y7, y7 + 256.0, depth=128.0),
            PitBox(640.0, 768.0, y7, y7 + 256.0, depth=128.0),
        ),
        entities=(
            EntitySpawn(
                id="c7_cube_0",
                kind=EntityKind.CUBE,
                pos=Vec3(912.0, y7 + 128.0, 0.0),
                zone_id=2,
            ),
            EntitySpawn(
                id="c7_floor_btn_0",
                kind=EntityKind.FLOOR_BUTTON,
                pos=Vec3(512.0, y7 + 128.0, 0.0),
                zone_id=1,
            ),
            EntitySpawn(
                id="c7_exit_0",
                kind=EntityKind.EXIT_DOOR,
                pos=Vec3(48.0, y7 + 128.0, 0.0),
                zone_id=0,
                initially_active=False,
            ),
        ),
        portal_walls=(
            PortalWall(
                id="c7_wall_a",
                pos=Vec3(128.0, y7, 64.0),
                normal=Vec3(0.0, 1.0, 0.0),
                zone_id=0,
                size=(128.0, 128.0),
            ),
            PortalWall(
                id="c7_wall_b",
                pos=Vec3(512.0, y7, 64.0),
                normal=Vec3(0.0, 1.0, 0.0),
                zone_id=1,
                size=(128.0, 128.0),
            ),
            PortalWall(
                id="c7_wall_c",
                pos=Vec3(896.0, y7, 64.0),
                normal=Vec3(0.0, 1.0, 0.0),
                zone_id=2,
                size=(128.0, 128.0),
            ),
        ),
        max_steps=320,
        exit_starts_open=False,
    )

    return {1: c1, 2: c2, 3: c3, 4: c4, 5: c5, 6: c6, 7: c7}


CHAMBERS: dict[int, ChamberSpec] = _build_chambers()


def get_chamber(chamber_id: int) -> ChamberSpec:
    if chamber_id not in CHAMBERS:
        raise ValueError(f"Unknown chamber_id={chamber_id}; expected 1..{len(CHAMBERS)}")
    return CHAMBERS[chamber_id]
