"""Multi-stage benchmark chambers (M1-M3) for controlled Layer-4 ablations.

Kept strictly outside ``portal_ml.sim.chambers.CHAMBERS[1..7]`` so the frozen
training set (Chambers 1-4) and held-out generalization suite (Chambers 5-7)
remain completely isolated and unchanged.
"""

from __future__ import annotations

from portal_ml.schema.blackboard import EntityKind, PortalWall, Vec3
from portal_ml.sim.chambers import (
    ChamberSpec,
    EntitySpawn,
    FizzlerBarrier,
    PitBox,
    WalkableZone,
)


def _build_multistage_benchmarks() -> dict[str, ChamberSpec]:
    # M1: Dual-Button Relay (modeled on Valve testchmb_a_02 Room 2 / testchmb_a_05)
    # Two weighted floor buttons (require_all_buttons=True); one cube in Zone 0,
    # second cube across a pit in Zone 1. Stateless Layer 2 stops after placing
    # cube 0 on button 0 and never fetches cube 1 for button 1.
    m1 = ChamberSpec(
        chamber_id=101,
        name="M1: Dual-Button Relay",
        description=(
            "Two weighted floor buttons both required to open exit; one cube local "
            "(zone 0) and a second cube across a portalable pit (zone 1)."
        ),
        held_out=True,
        room_min=Vec3(0.0, 0.0, 0.0),
        room_max=Vec3(768.0, 384.0, 256.0),
        player_spawn=Vec3(96.0, 192.0, 0.0),
        player_spawn_yaw=0.0,
        player_spawn_zone=0,
        zones=(
            WalkableZone(0, 16.0, 368.0, 16.0, 368.0, 0.0),
            WalkableZone(1, 528.0, 752.0, 16.0, 368.0, 0.0),
        ),
        pits=(PitBox(384.0, 512.0, 0.0, 384.0, depth=128.0),),
        entities=(
            EntitySpawn(
                id="m1_cube_0",
                kind=EntityKind.CUBE,
                pos=Vec3(224.0, 96.0, 0.0),
                zone_id=0,
            ),
            EntitySpawn(
                id="m1_cube_1",
                kind=EntityKind.CUBE,
                pos=Vec3(640.0, 192.0, 0.0),
                zone_id=1,
            ),
            EntitySpawn(
                id="m1_floor_btn_0",
                kind=EntityKind.FLOOR_BUTTON,
                pos=Vec3(288.0, 288.0, 0.0),
                zone_id=0,
            ),
            EntitySpawn(
                id="m1_floor_btn_1",
                kind=EntityKind.FLOOR_BUTTON,
                pos=Vec3(96.0, 320.0, 0.0),
                zone_id=0,
            ),
            EntitySpawn(
                id="m1_exit_0",
                kind=EntityKind.EXIT_DOOR,
                pos=Vec3(48.0, 192.0, 0.0),
                zone_id=0,
                initially_active=False,
            ),
        ),
        portal_walls=(
            PortalWall(
                id="m1_wall_a",
                pos=Vec3(192.0, 0.0, 64.0),
                normal=Vec3(0.0, 1.0, 0.0),
                zone_id=0,
                size=(128.0, 128.0),
            ),
            PortalWall(
                id="m1_wall_b",
                pos=Vec3(640.0, 0.0, 64.0),
                normal=Vec3(0.0, 1.0, 0.0),
                zone_id=1,
                size=(128.0, 128.0),
            ),
        ),
        require_all_buttons=True,
        max_steps=380,
        exit_starts_open=False,
    )

    # M2: Fizzler-Gated Decoy Cube
    # Zone 0 contains a decoy cube (m2_cube_0) behind a trigger_portal_cleanser
    # fizzler at x=336. Carrying m2_cube_0 back toward m2_floor_btn_0 vaporizes it.
    # Stateless Layer 2 loops on m2_cube_0 forever; Layer 4 records cube_fizzled
    # in EpisodicMemory and redirects Layer 2 to retrieve m2_cube_1 across the pit.
    m2 = ChamberSpec(
        chamber_id=102,
        name="M2: Fizzler-Gated Decoy Cube",
        description=(
            "Decoy cube behind an emancipation fizzler in zone 0; solvable cube "
            "across a portalable pit in zone 1. Tests episodic failure recovery."
        ),
        held_out=True,
        room_min=Vec3(0.0, 0.0, 0.0),
        room_max=Vec3(896.0, 384.0, 256.0),
        player_spawn=Vec3(96.0, 192.0, 0.0),
        player_spawn_yaw=0.0,
        player_spawn_zone=0,
        zones=(
            WalkableZone(0, 16.0, 496.0, 16.0, 368.0, 0.0),
            WalkableZone(1, 656.0, 880.0, 16.0, 368.0, 0.0),
        ),
        pits=(PitBox(512.0, 640.0, 0.0, 384.0, depth=128.0),),
        fizzlers=(
            FizzlerBarrier(
                id="m2_fizzler_0",
                zone_id=0,
                axis="x",
                coord=336.0,
                min_span=0.0,
                max_span=384.0,
            ),
        ),
        entities=(
            EntitySpawn(
                id="m2_cube_0",
                kind=EntityKind.CUBE,
                pos=Vec3(432.0, 192.0, 0.0),
                zone_id=0,
            ),
            EntitySpawn(
                id="m2_cube_1",
                kind=EntityKind.CUBE,
                pos=Vec3(768.0, 192.0, 0.0),
                zone_id=1,
            ),
            EntitySpawn(
                id="m2_floor_btn_0",
                kind=EntityKind.FLOOR_BUTTON,
                pos=Vec3(224.0, 288.0, 0.0),
                zone_id=0,
            ),
            EntitySpawn(
                id="m2_exit_0",
                kind=EntityKind.EXIT_DOOR,
                pos=Vec3(48.0, 192.0, 0.0),
                zone_id=0,
                initially_active=False,
            ),
        ),
        portal_walls=(
            PortalWall(
                id="m2_wall_a",
                pos=Vec3(160.0, 0.0, 64.0),
                normal=Vec3(0.0, 1.0, 0.0),
                zone_id=0,
                size=(128.0, 128.0),
            ),
            PortalWall(
                id="m2_wall_b",
                pos=Vec3(768.0, 0.0, 64.0),
                normal=Vec3(0.0, 1.0, 0.0),
                zone_id=1,
                size=(128.0, 128.0),
            ),
        ),
        require_all_buttons=True,
        max_steps=380,
        exit_starts_open=False,
    )

    # M3: Three-Stage Compound Vault
    # 3 disconnected zones separated by 2 pits; requires pressing BOTH weighted
    # floor buttons (zones 0 and 1) with two cubes (zones 1 and 2) AND then
    # pressing a timed pedestal button in zone 0 before sprinting to the exit.
    m3 = ChamberSpec(
        chamber_id=103,
        name="M3: Three-Stage Compound Vault",
        description=(
            "Three islands, two pits, two weighted floor buttons plus a timed pedestal "
            "button all required to open the exit. Tests multi-stage subgoal sequencing."
        ),
        held_out=True,
        room_min=Vec3(0.0, 0.0, 0.0),
        room_max=Vec3(1152.0, 384.0, 256.0),
        player_spawn=Vec3(96.0, 192.0, 0.0),
        player_spawn_yaw=0.0,
        player_spawn_zone=0,
        zones=(
            WalkableZone(0, 16.0, 368.0, 16.0, 368.0, 0.0),
            WalkableZone(1, 528.0, 752.0, 16.0, 368.0, 0.0),
            WalkableZone(2, 912.0, 1136.0, 16.0, 368.0, 0.0),
        ),
        pits=(
            PitBox(384.0, 512.0, 0.0, 384.0, depth=128.0),
            PitBox(768.0, 896.0, 0.0, 384.0, depth=128.0),
        ),
        entities=(
            EntitySpawn(
                id="m3_cube_0",
                kind=EntityKind.CUBE,
                pos=Vec3(640.0, 288.0, 0.0),
                zone_id=1,
            ),
            EntitySpawn(
                id="m3_cube_1",
                kind=EntityKind.CUBE,
                pos=Vec3(1024.0, 192.0, 0.0),
                zone_id=2,
            ),
            EntitySpawn(
                id="m3_floor_btn_0",
                kind=EntityKind.FLOOR_BUTTON,
                pos=Vec3(272.0, 304.0, 0.0),
                zone_id=0,
            ),
            EntitySpawn(
                id="m3_floor_btn_1",
                kind=EntityKind.FLOOR_BUTTON,
                pos=Vec3(640.0, 112.0, 0.0),
                zone_id=1,
            ),
            EntitySpawn(
                id="m3_pedestal_0",
                kind=EntityKind.PEDESTAL_BUTTON,
                pos=Vec3(224.0, 96.0, 0.0),
                zone_id=0,
                initially_active=False,
                pedestal_duration=14.0,
            ),
            EntitySpawn(
                id="m3_exit_0",
                kind=EntityKind.EXIT_DOOR,
                pos=Vec3(48.0, 192.0, 0.0),
                zone_id=0,
                initially_active=False,
            ),
        ),
        portal_walls=(
            PortalWall(
                id="m3_wall_a",
                pos=Vec3(160.0, 0.0, 64.0),
                normal=Vec3(0.0, 1.0, 0.0),
                zone_id=0,
                size=(128.0, 128.0),
            ),
            PortalWall(
                id="m3_wall_b",
                pos=Vec3(640.0, 0.0, 64.0),
                normal=Vec3(0.0, 1.0, 0.0),
                zone_id=1,
                size=(128.0, 128.0),
            ),
            PortalWall(
                id="m3_wall_c",
                pos=Vec3(1024.0, 0.0, 64.0),
                normal=Vec3(0.0, 1.0, 0.0),
                zone_id=2,
                size=(128.0, 128.0),
            ),
        ),
        require_all_buttons=True,
        max_steps=520,
        exit_starts_open=False,
    )

    return {"M1": m1, "M2": m2, "M3": m3}


MULTI_STAGE_BENCHMARKS: dict[str, ChamberSpec] = _build_multistage_benchmarks()


def get_multistage_benchmark(key: str) -> ChamberSpec:
    norm = key.upper()
    if norm not in MULTI_STAGE_BENCHMARKS:
        raise ValueError(
            f"Unknown multi-stage benchmark {key!r}; expected one of {tuple(MULTI_STAGE_BENCHMARKS)}"
        )
    return MULTI_STAGE_BENCHMARKS[norm]
