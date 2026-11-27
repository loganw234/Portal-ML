"""Programmatic generator for the single-map 5-chamber Portal 1 testbed (.vmf).

Reads exact chamber coordinates from `portal_ml.sim.chambers` so the compiled
Source Engine .bsp map matches the Python mock 3D simulator 1:1.
Uses verified Portal 1 (Source 2007/2013 SP) materials and entities.
"""

from __future__ import annotations

import math
from pathlib import Path

from portal_ml.schema.blackboard import EntityKind, PortalWall
from portal_ml.sim.chambers import CHAMBERS, ChamberSpec
from portal_ml.vmf.writer import VMFDocument, make_box_brush

MAT_METAL_WALL = "METAL/METALWALL048B"
MAT_METAL_FLOOR = "METAL/METAL_MODULAR_FLOOR001"
MAT_PORTAL_CONCRETE = "CONCRETE/CONCRETE_MODULAR_WALL001A"
MAT_HAZARD_FLOOR = "NATURE/HAZARD_LIQUID_BENEATH"
WALL_THICKNESS = 16.0


def _add_chamber_geometry(doc: VMFDocument, spec: ChamberSpec) -> None:
    x0, y0, z0 = spec.room_min.x, spec.room_min.y, spec.room_min.z
    x1, y1, z1 = spec.room_max.x, spec.room_max.y, spec.room_max.z
    t = WALL_THICKNESS
    pit_bottom = z0 - 144.0 if spec.pits else z0 - t

    # Ceiling
    doc.add_box(x0 - t, y0 - t, z1, x1 + t, y1 + t, z1 + t, material=MAT_METAL_WALL)

    # Outer sealed shell walls (non-portalable metal)
    doc.add_box(x0 - t, y0 - t, pit_bottom, x0, y1 + t, z1, material=MAT_METAL_WALL)  # West
    doc.add_box(x1, y0 - t, pit_bottom, x1 + t, y1 + t, z1, material=MAT_METAL_WALL)  # East
    doc.add_box(x0, y0 - t, pit_bottom, x1, y0, z1, material=MAT_METAL_WALL)  # South
    doc.add_box(x0, y1, pit_bottom, x1, y1 + t, z1, material=MAT_METAL_WALL)  # North

    # Sealed solid bottom floor beneath any pit (water shaders do not seal the void)
    doc.add_box(
        x0 - t,
        y0 - t,
        pit_bottom - t,
        x1 + t,
        y1 + t,
        pit_bottom,
        material=MAT_METAL_FLOOR,
    )

    # Walkable floor slabs per zone
    for zone in spec.zones:
        doc.add_box(
            zone.min_x - 16.0,
            zone.min_y - 16.0,
            zone.floor_z - t,
            zone.max_x + 16.0,
            zone.max_y + 16.0,
            zone.floor_z,
            material=MAT_METAL_FLOOR,
        )
        # Navigable zone trigger volume (exact AABB for C++ plugin zone lookup)
        doc.add_trigger_box(
            classname="trigger_multiple",
            min_x=zone.min_x,
            min_y=zone.min_y,
            min_z=zone.floor_z,
            max_x=zone.max_x,
            max_y=zone.max_y,
            max_z=z1 - 16.0,
            properties={
                "targetname": f"c{spec.chamber_id}_zone_{zone.zone_id}",
                "spawnflags": "1",
                "wait": "0.1",
            },
        )

    # Hazard trigger inside pits
    for idx, pit in enumerate(spec.pits):
        doc.add_trigger_box(
            classname="trigger_hurt",
            min_x=pit.min_x,
            min_y=pit.min_y,
            min_z=z0 - pit.depth,
            max_x=pit.max_x,
            max_y=pit.max_y,
            max_z=z0 - 16.0,
            properties={
                "targetname": f"c{spec.chamber_id}_pit_{idx}",
                "damage": "1000",
                "damagetype": "1048576",
                "spawnflags": "1",
            },
        )

    # Portalable concrete panels inlaid on walls + info_target markers for plugin
    for wall in spec.portal_walls:
        _add_portal_panel(doc, wall)

    # Spawn marker (info_player_start for Chamber 1, info_target for all chambers)
    sx, sy, sz = spec.player_spawn.x, spec.player_spawn.y, spec.player_spawn.z + 8.0
    if spec.chamber_id == 1:
        doc.add_entity(
            classname="info_player_start",
            properties={
                "origin": f"{sx:g} {sy:g} {sz:g}",
                "angles": f"0 {spec.player_spawn_yaw:g} 0",
            },
        )
    doc.add_entity(
        classname="info_target",
        properties={
            "targetname": f"c{spec.chamber_id}_spawn",
            "origin": f"{sx:g} {sy:g} {sz:g}",
            "angles": f"0 {spec.player_spawn_yaw:g} 0",
        },
    )

    # Chamber interior lighting (prevents dark/fullbright compile)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    doc.add_entity(
        classname="light",
        properties={
            "origin": f"{cx:g} {cy:g} {z1 - 32.0:g}",
            "_light": "230 240 255 320",
            "_quadratic_attn": "1",
        },
    )

    # Gameplay entities (authentic Portal 1 classes, targetnames matching BlackboardSnapshot IDs)
    exit_id = next(
        (e.id for e in spec.entities if e.kind is EntityKind.EXIT_DOOR),
        f"c{spec.chamber_id}_exit_0",
    )
    for ent in spec.entities:
        ex, ey, ez = ent.pos.x, ent.pos.y, ent.pos.z
        if ent.kind is EntityKind.CUBE:
            doc.add_entity(
                classname="prop_physics",
                properties={
                    "targetname": ent.id,
                    "origin": f"{ex:g} {ey:g} {ez + 24.0:g}",
                    "model": "models/props/metal_box.mdl",
                    "spawnflags": "256",
                },
            )
            # Dynamic drop teleporter (moved to drop position before firing Teleport)
            doc.add_entity(
                classname="point_teleport",
                properties={
                    "targetname": f"{ent.id}_tp",
                    "target": ent.id,
                    "origin": f"{ex:g} {ey:g} {ez + 24.0:g}",
                    "angles": "0 0 0",
                    "spawnflags": "0",
                },
            )
            # Dedicated Teleport-Home entity for ResetChamberInPlace()
            doc.add_entity(
                classname="point_teleport",
                properties={
                    "targetname": f"{ent.id}_reset_tp",
                    "target": ent.id,
                    "origin": f"{ex:g} {ey:g} {ez + 24.0:g}",
                    "angles": "0 0 0",
                    "spawnflags": "1",
                },
            )
        elif ent.kind is EntityKind.FLOOR_BUTTON:
            # Portal 1 weighted floor button: func_weight_button brush + base prop
            btn_brush = make_box_brush(
                doc.ids,
                ex - 40.0,
                ey - 40.0,
                ez,
                ex + 40.0,
                ey + 40.0,
                ez + 12.0,
                material=MAT_METAL_FLOOR,
            )
            doc.add_entity(
                classname="func_weight_button",
                properties={
                    "targetname": ent.id,
                    "origin": f"{ex:g} {ey:g} {ez + 6.0:g}",
                    "WeightToActivate": "35",
                },
                connections=[
                    ("OnPressed", f"{exit_id},Open,,0,-1"),
                    ("OnReleased", f"{exit_id},Close,,0,-1"),
                ],
                solids=[btn_brush],
            )
            doc.add_entity(
                classname="prop_dynamic",
                properties={
                    "targetname": f"{ent.id}_model",
                    "origin": f"{ex:g} {ey:g} {ez:g}",
                    "model": "models/props/button_base_reference.mdl",
                    "solid": "0",
                },
            )
        elif ent.kind is EntityKind.PEDESTAL_BUTTON:
            # Portal 1 timed pedestal button: func_button + switch prop
            ped_brush = make_box_brush(
                doc.ids,
                ex - 16.0,
                ey - 16.0,
                ez,
                ex + 16.0,
                ey + 16.0,
                ez + 48.0,
                material=MAT_METAL_WALL,
            )
            doc.add_entity(
                classname="func_button",
                properties={
                    "targetname": ent.id,
                    "origin": f"{ex:g} {ey:g} {ez + 24.0:g}",
                    "wait": f"{ent.pedestal_duration:g}",
                    "spawnflags": "1025",  # Don't Move + Use Activates
                },
                connections=[
                    ("OnPressed", f"{exit_id},Open,,0,-1"),
                    ("OnOut", f"{exit_id},Close,,0,-1"),
                ],
                solids=[ped_brush],
            )
            doc.add_entity(
                classname="prop_dynamic",
                properties={
                    "targetname": f"{ent.id}_model",
                    "origin": f"{ex:g} {ey:g} {ez:g}",
                    "model": "models/props/switch001.mdl",
                    "solid": "0",
                },
            )
        elif ent.kind is EntityKind.EXIT_DOOR:
            # Portal 1 sliding exit door (func_door slides vertically into ceiling)
            door_brush = make_box_brush(
                doc.ids,
                ex - 8.0,
                ey - 40.0,
                ez,
                ex + 8.0,
                ey + 40.0,
                ez + 112.0,
                material=MAT_METAL_WALL,
            )
            doc.add_entity(
                classname="func_door",
                properties={
                    "targetname": ent.id,
                    "origin": f"{ex:g} {ey:g} {ez + 56.0:g}",
                    "movedir": "-90 0 0",
                    "speed": "200",
                    "wait": "-1",
                    "spawnpos": "1" if ent.initially_active else "0",
                },
                solids=[door_brush],
            )
            doc.add_trigger_box(
                classname="trigger_multiple",
                min_x=ex - 36.0,
                min_y=ey - 36.0,
                min_z=ez,
                max_x=ex + 36.0,
                max_y=ey + 36.0,
                max_z=ez + 112.0,
                properties={
                    "targetname": f"c{spec.chamber_id}_exit_trigger",
                    "spawnflags": "1",
                },
            )


def _add_portal_panel(doc: VMFDocument, wall: PortalWall) -> None:
    """Emit a flush 128x128 portalable concrete panel brush + named info_target."""
    hw = wall.size[0] * 0.5
    hh = wall.size[1] * 0.5
    px, py, pz = wall.pos.x, wall.pos.y, wall.pos.z
    nx, ny = wall.normal.x, wall.normal.y
    panel_depth = 1.0

    if abs(ny) > 0.5:
        y_min = py if ny > 0 else py - panel_depth
        y_max = py + panel_depth if ny > 0 else py
        doc.add_box(
            px - hw,
            y_min,
            pz - hh,
            px + hw,
            y_max,
            pz + hh,
            material=MAT_PORTAL_CONCRETE,
        )
    else:
        x_min = px if nx > 0 else px - panel_depth
        x_max = px + panel_depth if nx > 0 else px
        doc.add_box(
            x_min,
            py - hw,
            pz - hh,
            x_max,
            py + hw,
            pz + hh,
            material=MAT_PORTAL_CONCRETE,
        )

    # Named marker so the C++ plugin reads exact portalable surface normal & zone
    doc.add_entity(
        classname="info_target",
        properties={
            "targetname": wall.id,
            "origin": f"{px:g} {py:g} {pz:g}",
            "angles": _normal_to_angles(wall.normal.x, wall.normal.y, wall.normal.z),
        },
    )


def _normal_to_angles(nx: float, ny: float, nz: float) -> str:
    yaw = math.degrees(math.atan2(ny, nx))
    pitch = math.degrees(-math.asin(max(-1.0, min(1.0, nz))))
    return f"{pitch:g} {yaw:g} 0"


def build_five_chamber_vmf() -> VMFDocument:
    """Build a single VMFDocument containing Chambers 1-5 separated by void."""
    doc = VMFDocument(mapversion=1)
    c1 = CHAMBERS[1]
    # Dual-portal device on spawn
    doc.add_entity(
        classname="weapon_portalgun",
        properties={
            "targetname": "portal_ml_gun",
            "origin": f"{c1.player_spawn.x:g} {c1.player_spawn.y:g} {c1.player_spawn.z + 16.0:g}",
            "CanFirePortal1": "1",
            "CanFirePortal2": "1",
        },
    )
    # Linked blue/orange prop_portal pair controlled deterministically by the plugin
    doc.add_entity(
        classname="prop_portal",
        properties={
            "targetname": "portal_ml_blue",
            "origin": f"{c1.player_spawn.x:g} {c1.player_spawn.y:g} {c1.player_spawn.z + 128.0:g}",
            "angles": "0 0 0",
            "Activated": "0",
            "PortalTwo": "0",
            "LinkageGroupID": "0",
        },
    )
    doc.add_entity(
        classname="prop_portal",
        properties={
            "targetname": "portal_ml_orange",
            "origin": f"{c1.player_spawn.x:g} {c1.player_spawn.y:g} {c1.player_spawn.z + 128.0:g}",
            "angles": "0 180 0",
            "Activated": "0",
            "PortalTwo": "1",
            "LinkageGroupID": "0",
        },
    )
    for cid in sorted(CHAMBERS):
        _add_chamber_geometry(doc, CHAMBERS[cid])
    return doc


def write_five_chamber_map(output_path: str | Path) -> Path:
    """Write the generated 5-chamber .vmf file to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = build_five_chamber_vmf()
    out.write_text(doc.to_vmf(), encoding="utf-8")
    return out


def write_compile_scripts(maps_dir: str | Path, map_stem: str = "portal_ml_5chambers") -> tuple[Path, Path]:
    """Emit Windows (.bat) and POSIX (.sh) vbsp/vvis/vrad batch compile scripts."""
    out_dir = Path(maps_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bat = out_dir / "compile_map.bat"
    bat.write_text(
        f"""@echo off
REM Batch-compile {map_stem}.vmf for Portal 1 (Source 2007/2013 SP)
REM Override BIN_DIR and GAME_DIR if installed outside default Steam path.

if "%BIN_DIR%"=="" set "BIN_DIR=C:\\Program Files (x86)\\Steam\\steamapps\\common\\Portal\\bin"
if "%GAME_DIR%"=="" set "GAME_DIR=C:\\Program Files (x86)\\Steam\\steamapps\\common\\Portal\\portal"

set "MAP_VMF=%~dp0{map_stem}.vmf"
set "MAP_BSP=%~dp0{map_stem}.bsp"

echo [1/3] Running vbsp...
"%BIN_DIR%\\vbsp.exe" -game "%GAME_DIR%" "%MAP_VMF%" || exit /b 1

echo [2/3] Running vvis (fast)...
"%BIN_DIR%\\vvis.exe" -fast -game "%GAME_DIR%" "%MAP_BSP%" || exit /b 1

echo [3/3] Running vrad (fast)...
"%BIN_DIR%\\vrad.exe" -fast -game "%GAME_DIR%" "%MAP_BSP%" || exit /b 1

echo Copying compiled BSP to %GAME_DIR%\\maps\\...
copy /Y "%MAP_BSP%" "%GAME_DIR%\\maps\\{map_stem}.bsp"
echo Done!
""",
        encoding="utf-8",
    )

    sh = out_dir / "compile_map.sh"
    sh.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

BIN_DIR="${{BIN_DIR:-$HOME/.steam/steam/steamapps/common/Portal/bin}}"
GAME_DIR="${{GAME_DIR:-$HOME/.steam/steam/steamapps/common/Portal/portal}}"
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
MAP_VMF="$SCRIPT_DIR/{map_stem}.vmf"
MAP_BSP="$SCRIPT_DIR/{map_stem}.bsp"

echo "[1/3] Running vbsp..."
"$BIN_DIR/vbsp" -game "$GAME_DIR" "$MAP_VMF"

echo "[2/3] Running vvis (fast)..."
"$BIN_DIR/vvis" -fast -game "$GAME_DIR" "$MAP_BSP"

echo "[3/3] Running vrad (fast)..."
"$BIN_DIR/vrad" -fast -game "$GAME_DIR" "$MAP_BSP"

cp -f "$MAP_BSP" "$GAME_DIR/maps/{map_stem}.bsp"
echo "Done!"
""",
        encoding="utf-8",
    )
    return bat, sh
