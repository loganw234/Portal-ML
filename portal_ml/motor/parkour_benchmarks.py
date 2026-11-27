"""Physical-skill benchmark chambers (P1-P3) for sensor-equipped neural navigation & jumping.

Kept strictly outside ``portal_ml.sim.chambers.CHAMBERS[1..7]`` so the frozen
training set (1-4) and held-out generalization suite (5-7) remain completely isolated.
"""

from __future__ import annotations

from portal_ml.schema.blackboard import EntityKind, HazardVolume, Vec3
from portal_ml.sim.chambers import (
    ChamberSpec,
    EntitySpawn,
    ObstacleBox,
    PitBox,
    WalkableZone,
)


def _build_parkour_benchmarks() -> dict[str, ChamberSpec]:
    # P1: Partition Slalom
    # A solid tall glass/concrete partition blocks the direct line from spawn to exit.
    # Blind MoveTo / NeuralMoveTo walks straight into the wall and stalls;
    # NeuralParkour reads LocalSensors.wall_rays and steers smoothly around it.
    p1 = ChamberSpec(
        chamber_id=201,
        name="P1: Partition Slalom",
        description=(
            "Solid tall partition blocking the straight-line path to the exit. "
            "Tests local wall-ray obstacle avoidance without hardcoded coordinates."
        ),
        held_out=True,
        room_min=Vec3(0.0, 0.0, 0.0),
        room_max=Vec3(512.0, 320.0, 256.0),
        player_spawn=Vec3(64.0, 160.0, 0.0),
        player_spawn_yaw=0.0,
        player_spawn_zone=0,
        zones=(WalkableZone(0, 16.0, 496.0, 16.0, 304.0, 0.0),),
        obstacles=(
            ObstacleBox(
                id="p1_partition_0",
                zone_id=0,
                min_x=232.0,
                max_x=264.0,
                min_y=88.0,
                max_y=232.0,
                min_z=0.0,
                max_z=128.0,
            ),
        ),
        entities=(
            EntitySpawn(
                id="p1_exit_0",
                kind=EntityKind.EXIT_DOOR,
                pos=Vec3(464.0, 160.0, 0.0),
                zone_id=0,
                initially_active=True,
            ),
        ),
        max_steps=140,
        exit_starts_open=True,
    )

    # P2: Hurdle & Acid Trench
    # Full-width jumpable 28u hurdle followed by a full-width 76u toxic acid trench
    # (trigger_hurt, 250 dps). Blind MoveTo / NeuralMoveTo stalls at the hurdle or dies
    # in the acid; NeuralParkour vaults the hurdle and jumps cleanly over the acid.
    p2 = ChamberSpec(
        chamber_id=202,
        name="P2: Hurdle & Acid Trench",
        description=(
            "Full-width low hurdle (28u) plus toxic acid trench (76u wide) separating "
            "spawn from exit. Tests step_clearance + hazard_ahead timed jumping."
        ),
        held_out=True,
        room_min=Vec3(0.0, 0.0, 0.0),
        room_max=Vec3(640.0, 256.0, 256.0),
        player_spawn=Vec3(64.0, 128.0, 0.0),
        player_spawn_yaw=0.0,
        player_spawn_zone=0,
        zones=(WalkableZone(0, 16.0, 624.0, 16.0, 240.0, 0.0),),
        obstacles=(
            ObstacleBox(
                id="p2_hurdle_0",
                zone_id=0,
                min_x=196.0,
                max_x=220.0,
                min_y=0.0,
                max_y=256.0,
                min_z=0.0,
                max_z=28.0,
            ),
        ),
        hazards=(
            HazardVolume(
                id="p2_acid_0",
                kind="acid",
                min_pos=Vec3(368.0, 0.0, -32.0),
                max_pos=Vec3(444.0, 256.0, 8.0),
                damage_per_sec=250.0,
            ),
        ),
        entities=(
            EntitySpawn(
                id="p2_exit_0",
                kind=EntityKind.EXIT_DOOR,
                pos=Vec3(592.0, 128.0, 0.0),
                zone_id=0,
                initially_active=True,
            ),
        ),
        max_steps=160,
        exit_starts_open=True,
    )

    # P3: Gap Jump & Timed Acid Vault
    # Two platforms separated by a 72u jumpable pit (no portals!), followed by an acid
    # strip between a timed pedestal button and the closed exit door. Tests combining
    # floor_probes gap-jumping + hazard jumping underneath frozen seed7_decoupled.bt!
    p3 = ChamberSpec(
        chamber_id=203,
        name="P3: Gap Jump & Timed Vault",
        description=(
            "72u jumpable pit gap + timed pedestal button + toxic acid trench before "
            "closed exit. Tests Layer-2 causal BT + Layer-3 NeuralParkour composition."
        ),
        held_out=True,
        room_min=Vec3(0.0, 0.0, 0.0),
        room_max=Vec3(768.0, 256.0, 256.0),
        player_spawn=Vec3(64.0, 128.0, 0.0),
        player_spawn_yaw=0.0,
        player_spawn_zone=0,
        zones=(
            WalkableZone(0, 16.0, 224.0, 16.0, 240.0, 0.0),
            WalkableZone(1, 296.0, 752.0, 16.0, 240.0, 0.0),
        ),
        pits=(PitBox(224.0, 296.0, 0.0, 256.0, depth=128.0),),
        hazards=(
            HazardVolume(
                id="p3_acid_0",
                kind="acid",
                min_pos=Vec3(516.0, 0.0, -32.0),
                max_pos=Vec3(588.0, 256.0, 8.0),
                damage_per_sec=250.0,
            ),
        ),
        entities=(
            EntitySpawn(
                id="p3_pedestal_0",
                kind=EntityKind.PEDESTAL_BUTTON,
                pos=Vec3(424.0, 128.0, 0.0),
                zone_id=1,
                initially_active=False,
                pedestal_duration=10.0,
            ),
            EntitySpawn(
                id="p3_exit_0",
                kind=EntityKind.EXIT_DOOR,
                pos=Vec3(720.0, 128.0, 0.0),
                zone_id=1,
                initially_active=False,
            ),
        ),
        max_steps=200,
        exit_starts_open=False,
    )

    return {"P1": p1, "P2": p2, "P3": p3}


PARKOUR_BENCHMARKS: dict[str, ChamberSpec] = _build_parkour_benchmarks()


def get_parkour_benchmark(key: str) -> ChamberSpec:
    norm = key.upper()
    if norm not in PARKOUR_BENCHMARKS:
        raise ValueError(
            f"Unknown parkour benchmark {key!r}; expected one of {tuple(PARKOUR_BENCHMARKS)}"
        )
    return PARKOUR_BENCHMARKS[norm]
