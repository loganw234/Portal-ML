"""Pure-Python SVG path & trajectory visualizer for Portal-ML chambers and live traces.

Generates crisp, scalable vector graphics (no external dependencies) for:
- Single-chamber comparisons (e.g. Scripted vs Blind Neural vs Sensor Parkour, or 3-Layer vs 4-Layer)
- Combined multi-panel hero dashboards for README / experiment reports
- Live unmodified Valve campaign traces (top-down XY + Z-elevation profile)
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from portal_ml.bt.ast import BTNode
from portal_ml.bt.evaluator import BehaviorTreeEvaluator
from portal_ml.motor.hybrid import HybridMotorController
from portal_ml.planner import FourLayerAgent, StrategicPlanner
from portal_ml.schema.blackboard import EntityKind, Vec3
from portal_ml.sim.chambers import ChamberSpec
from portal_ml.sim.mock_env import MockPortalEnv

# Aperture-inspired dark blueprint palette
BG_OUTER = "#070b12"
BG_PANEL = "#0d1421"
BG_ZONE = "#141f33"
STROKE_ZONE = "#2a3c5c"
BG_PIT = "#05080f"
GRID_COLOR = "#162238"
TEXT_PRIMARY = "#f1f5f9"
TEXT_MUTED = "#94a3b8"
TEXT_DIM = "#64748b"
COLOR_BLUE_PORTAL = "#38bdf8"
COLOR_ORANGE_PORTAL = "#fb923c"
COLOR_ACID_FILL = "#365314"
COLOR_ACID_STROKE = "#84cc16"
COLOR_FIZZLER = "#22d3ee"
COLOR_HURDLE_FILL = "#78350f"
COLOR_HURDLE_STROKE = "#fbbf24"
COLOR_WALL_FILL = "#334155"
COLOR_WALL_STROKE = "#64748b"
COLOR_CONCRETE = "#cbd5e1"
COLOR_EXIT = "#10b981"
COLOR_BUTTON = "#f43f5e"
COLOR_PEDESTAL = "#f59e0b"
COLOR_CUBE = "#e2e8f0"


@dataclass(slots=True)
class TracePoint:
    step: int
    pos: tuple[float, float, float]
    zone_id: int = 0
    held: str | None = None
    on_ground: bool = True
    health: float = 100.0
    target: str | None = None
    subgoal: str | None = None


@dataclass(slots=True)
class PortalMarker:
    color: str  # "blue" | "orange"
    pos: tuple[float, float, float]
    normal: tuple[float, float, float] = (0.0, 1.0, 0.0)


@dataclass(slots=True)
class PathSeries:
    label: str
    color: str
    points: list[TracePoint] = field(default_factory=list)
    portals: list[PortalMarker] = field(default_factory=list)
    events: list[tuple[float, float, str, str]] = field(default_factory=list)
    dashed: bool = False
    success: bool = False
    steps: int = 0
    final_health: float = 100.0
    extra_badge: str = ""


@dataclass(slots=True)
class PanelSpec:
    title: str
    subtitle: str
    chamber: ChamberSpec
    series: list[PathSeries]
    row_tag: str = ""


def _esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def load_jsonl_series(
    trace_path: str | Path,
    chamber_id: int,
    *,
    label: str,
    color: str,
    dashed: bool = False,
    spawn_pos: Vec3 | None = None,
    extra_badge: str = "",
) -> PathSeries:
    """Load one chamber's trajectory from a replay_live.py JSONL trace."""
    points: list[TracePoint] = []
    seen_portals: dict[tuple[str, int, int], PortalMarker] = {}
    success = False
    final_health = 100.0

    if spawn_pos is not None:
        points.append(
            TracePoint(
                step=0,
                pos=(spawn_pos.x, spawn_pos.y, spawn_pos.z),
                zone_id=0,
            )
        )

    with Path(trace_path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if int(row.get("chamber_id", -1)) != chamber_id:
                continue
            p = row["player"]
            pos = (float(p["pos"][0]), float(p["pos"][1]), float(p["pos"][2]))
            final_health = float(p.get("health", 100.0))
            points.append(
                TracePoint(
                    step=int(row.get("step", len(points))),
                    pos=pos,
                    zone_id=int(p.get("zone_id", 0)),
                    held=p.get("held_entity_id"),
                    on_ground=bool(p.get("on_ground", True)),
                    health=final_health,
                    target=row.get("target"),
                    subgoal=row.get("subgoal"),
                )
            )
            portals = row.get("portals", {})
            for col in ("blue", "orange"):
                pr = portals.get(col)
                if pr and pr.get("active"):
                    ppos = (
                        float(pr["pos"][0]),
                        float(pr["pos"][1]),
                        float(pr["pos"][2]),
                    )
                    pnorm = (
                        float(pr.get("normal", [0.0, 1.0, 0.0])[0]),
                        float(pr.get("normal", [0.0, 1.0, 0.0])[1]),
                        float(pr.get("normal", [0.0, 1.0, 0.0])[2]),
                    )
                    key = (col, int(round(ppos[0] / 8.0)), int(round(ppos[1] / 8.0)))
                    seen_portals[key] = PortalMarker(color=col, pos=ppos, normal=pnorm)
            if row.get("success"):
                success = True

    steps = points[-1].step if points else 0
    return PathSeries(
        label=label,
        color=color,
        points=points,
        portals=list(seen_portals.values()),
        dashed=dashed,
        success=success,
        steps=steps,
        final_health=final_health,
        extra_badge=extra_badge,
    )


def record_mock_series(
    spec: ChamberSpec,
    tree: BTNode,
    *,
    planner: StrategicPlanner | None = None,
    hybrid_motor: HybridMotorController | None = None,
    label: str,
    color: str,
    dashed: bool = False,
    extra_badge: str = "",
) -> PathSeries:
    """Run one deterministic mock episode and capture full trajectory + events."""
    env = MockPortalEnv()
    snap = env.reset(chamber_id=spec.chamber_id, spec=spec)
    ev = BehaviorTreeEvaluator(tree)
    agent = FourLayerAgent(ev, planner=planner, hybrid_motor=hybrid_motor)
    agent.reset(snap)

    points: list[TracePoint] = [
        TracePoint(
            step=0,
            pos=(snap.player.pos.x, snap.player.pos.y, snap.player.pos.z),
            zone_id=snap.player.zone_id,
            held=snap.player.held_entity_id,
            on_ground=snap.player.on_ground,
            health=snap.player.health,
        )
    ]
    seen_portals: dict[tuple[str, int, int], PortalMarker] = {}
    events: list[tuple[float, float, str, str]] = []
    seen_event_labels: set[str] = set()
    last_milestones: list[str] = []
    steps = 0

    while not snap.done and steps < spec.max_steps:
        cmd, tick_res = agent.act(snap, last_milestones)
        res = env.step(cmd)
        snap = res.snapshot
        last_milestones = list(res.milestones)
        steps += 1

        points.append(
            TracePoint(
                step=steps,
                pos=(snap.player.pos.x, snap.player.pos.y, snap.player.pos.z),
                zone_id=snap.player.zone_id,
                held=snap.player.held_entity_id,
                on_ground=snap.player.on_ground,
                health=snap.player.health,
                target=tick_res.target_role,
                subgoal=(
                    agent.current_directive.subgoal
                    if agent.current_directive is not None
                    else None
                ),
            )
        )

        for pr in (snap.blue_portal, snap.orange_portal):
            if pr.active:
                key = (
                    pr.color.value,
                    int(round(pr.pos.x / 8.0)),
                    int(round(pr.pos.y / 8.0)),
                )
                seen_portals[key] = PortalMarker(
                    color=pr.color.value,
                    pos=(pr.pos.x, pr.pos.y, pr.pos.z),
                    normal=(pr.normal.x, pr.normal.y, pr.normal.z),
                )

        for ms in last_milestones:
            if ms.startswith("cube_fizzled:") and "FIZZLED" not in seen_event_labels:
                seen_event_labels.add("FIZZLED")
                events.append((snap.player.pos.x, snap.player.pos.y, "FIZZLED", COLOR_FIZZLER))
            elif ms.startswith("player_died:") and "DIED" not in seen_event_labels:
                seen_event_labels.add("DIED")
                events.append((snap.player.pos.x, snap.player.pos.y, "DIED", "#ef4444"))

    badge = extra_badge
    if not badge and agent.planner_calls > 0:
        badge = f"{agent.planner_calls} L4 calls"
    return PathSeries(
        label=label,
        color=color,
        points=points,
        portals=list(seen_portals.values()),
        events=events,
        dashed=dashed,
        success=snap.success,
        steps=steps,
        final_health=snap.player.health,
        extra_badge=badge,
    )


class _ViewportTransform:
    """1:1 aspect-preserving world (XY) -> SVG screen transform (Y flipped up)."""

    def __init__(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        box_x: float,
        box_y: float,
        box_w: float,
        box_h: float,
        pad: float = 16.0,
    ) -> None:
        world_w = max(1.0, max_x - min_x)
        world_h = max(1.0, max_y - min_y)
        avail_w = max(10.0, box_w - 2.0 * pad)
        avail_h = max(10.0, box_h - 2.0 * pad)
        self.scale = min(avail_w / world_w, avail_h / world_h)
        drawn_w = world_w * self.scale
        drawn_h = world_h * self.scale
        self.off_x = box_x + (box_w - drawn_w) * 0.5
        self.off_y = box_y + (box_h - drawn_h) * 0.5
        self.min_x = min_x
        self.max_y = max_y

    def pt(self, wx: float, wy: float) -> tuple[float, float]:
        sx = self.off_x + (wx - self.min_x) * self.scale
        sy = self.off_y + (self.max_y - wy) * self.scale
        return sx, sy

    def rect(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> tuple[float, float, float, float]:
        sx0, sy0 = self.pt(x0, y1)
        sx1, sy1 = self.pt(x1, y0)
        return sx0, sy0, max(1.0, sx1 - sx0), max(1.0, sy1 - sy0)


def _svg_defs() -> str:
    return (
        "<defs>"
        '<pattern id="pitHatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        '<rect width="8" height="8" fill="#05080f"/>'
        '<line x1="0" y1="0" x2="0" y2="8" stroke="#172033" stroke-width="2"/>'
        "</pattern>"
        '<pattern id="acidHatch" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(-35)">'
        '<rect width="10" height="10" fill="#2c4410"/>'
        '<line x1="0" y1="0" x2="0" y2="10" stroke="#4d7c0f" stroke-width="2.5"/>'
        "</pattern>"
        '<filter id="softGlow" x="-30%" y="-30%" width="160%" height="160%">'
        '<feGaussianBlur stdDeviation="2.2" result="blur"/>'
        '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>'
        "</filter>"
        "</defs>"
    )


def _draw_arrowhead(
    x1: float, y1: float, x2: float, y2: float, color: str, size: float = 5.5
) -> str:
    ang = math.atan2(y2 - y1, x2 - x1)
    lx = x2 - size * math.cos(ang - 0.5)
    ly = y2 - size * math.sin(ang - 0.5)
    rx = x2 - size * math.cos(ang + 0.5)
    ry = y2 - size * math.sin(ang + 0.5)
    return (
        f'<polygon points="{x2:.1f},{y2:.1f} {lx:.1f},{ly:.1f} {rx:.1f},{ry:.1f}" '
        f'fill="{color}" opacity="0.92"/>'
    )


def _render_panel_elements(
    panel: PanelSpec,
    x: float,
    y: float,
    w: float,
    h: float,
) -> str:
    ch = panel.chamber
    parts: list[str] = []

    # Card frame
    parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="10" '
        f'fill="{BG_PANEL}" stroke="#1e2d47" stroke-width="1.2"/>'
    )

    # Header bar
    parts.append(
        f'<text x="{x + 14:.1f}" y="{y + 23:.1f}" fill="{TEXT_PRIMARY}" '
        f'font-family="Segoe UI, Inter, system-ui, sans-serif" font-size="13.5" font-weight="700">'
        f"{_esc(panel.title)}</text>"
    )
    parts.append(
        f'<text x="{x + 14:.1f}" y="{y + 39:.1f}" fill="{TEXT_MUTED}" '
        f'font-family="Segoe UI, Inter, system-ui, sans-serif" font-size="10.5">'
        f"{_esc(panel.subtitle)}</text>"
    )

    # World viewport region inside card (leave space for header + bottom badges)
    map_x = x + 10.0
    map_y = y + 48.0
    map_w = w - 20.0
    map_h = h - 84.0
    tf = _ViewportTransform(
        ch.room_min.x,
        ch.room_max.x,
        ch.room_min.y,
        ch.room_max.y,
        map_x,
        map_y,
        map_w,
        map_h,
        pad=14.0,
    )

    # Room outer shell
    rx, ry, rw, rh = tf.rect(ch.room_min.x, ch.room_min.y, ch.room_max.x, ch.room_max.y)
    parts.append(
        f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{rw:.1f}" height="{rh:.1f}" '
        f'fill="{BG_PIT}" stroke="#33486b" stroke-width="1.4"/>'
    )

    # Subtle 64u Hammer grid
    gx = math.ceil(ch.room_min.x / 64.0) * 64.0
    while gx < ch.room_max.x:
        sx, _ = tf.pt(gx, ch.room_min.y)
        parts.append(
            f'<line x1="{sx:.1f}" y1="{ry:.1f}" x2="{sx:.1f}" y2="{ry + rh:.1f}" '
            f'stroke="{GRID_COLOR}" stroke-width="0.6"/>'
        )
        gx += 64.0
    gy = math.ceil(ch.room_min.y / 64.0) * 64.0
    while gy < ch.room_max.y:
        _, sy = tf.pt(ch.room_min.x, gy)
        parts.append(
            f'<line x1="{rx:.1f}" y1="{sy:.1f}" x2="{rx + rw:.1f}" y2="{sy:.1f}" '
            f'stroke="{GRID_COLOR}" stroke-width="0.6"/>'
        )
        gy += 64.0

    # Pits (void hatches)
    for pit in ch.pits:
        px, py, pw, ph = tf.rect(pit.min_x, pit.min_y, pit.max_x, pit.max_y)
        parts.append(
            f'<rect x="{px:.1f}" y="{py:.1f}" width="{pw:.1f}" height="{ph:.1f}" '
            f'fill="url(#pitHatch)" stroke="#1e293b" stroke-width="0.9"/>'
        )
        if pw > 24 and ph > 18:
            parts.append(
                f'<text x="{px + pw * 0.5:.1f}" y="{py + ph * 0.5 + 3:.1f}" text-anchor="middle" '
                f'fill="#475569" font-family="Consolas, monospace" font-size="8.5" font-weight="600">PIT</text>'
            )

    # Walkable zones
    for z in ch.zones:
        zx, zy, zw, zh = tf.rect(z.min_x, z.min_y, z.max_x, z.max_y)
        parts.append(
            f'<rect x="{zx:.1f}" y="{zy:.1f}" width="{zw:.1f}" height="{zh:.1f}" '
            f'fill="{BG_ZONE}" stroke="{STROKE_ZONE}" stroke-width="1.0"/>'
        )
        if len(ch.zones) > 1:
            parts.append(
                f'<text x="{zx + 5:.1f}" y="{zy + 11:.1f}" fill="#475569" '
                f'font-family="Consolas, monospace" font-size="8.5">Z{z.zone_id}</text>'
            )

    # Toxic acid / trigger_hurt hazards
    for hz in ch.hazards:
        hx, hy, hw, hh = tf.rect(hz.min_pos.x, hz.min_pos.y, hz.max_pos.x, hz.max_pos.y)
        parts.append(
            f'<rect x="{hx:.1f}" y="{hy:.1f}" width="{hw:.1f}" height="{hh:.1f}" '
            f'fill="url(#acidHatch)" stroke="{COLOR_ACID_STROKE}" stroke-width="1.4" stroke-dasharray="4,2"/>'
        )
        parts.append(
            f'<text x="{hx + hw * 0.5:.1f}" y="{hy + hh * 0.5 + 3:.1f}" text-anchor="middle" '
            f'fill="#bef264" font-family="Consolas, monospace" font-size="8.5" font-weight="700">ACID</text>'
        )

    # Solid obstacles & jumpable hurdles
    for obs in ch.obstacles:
        ox, oy, ow, oh = tf.rect(obs.min_x, obs.min_y, obs.max_x, obs.max_y)
        is_hurdle = obs.max_z <= 42.0
        fill = COLOR_HURDLE_FILL if is_hurdle else COLOR_WALL_FILL
        stroke = COLOR_HURDLE_STROKE if is_hurdle else COLOR_WALL_STROKE
        parts.append(
            f'<rect x="{ox:.1f}" y="{oy:.1f}" width="{ow:.1f}" height="{oh:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>'
        )
        lbl = f"{int(obs.max_z)}u" if is_hurdle else "WALL"
        parts.append(
            f'<text x="{ox + ow * 0.5:.1f}" y="{oy + oh * 0.5 + 3:.1f}" text-anchor="middle" '
            f'fill="{TEXT_PRIMARY}" font-family="Consolas, monospace" font-size="8" font-weight="700">{lbl}</text>'
        )

    # Fizzlers
    for fz in ch.fizzlers:
        if fz.axis == "x":
            fx1, fy1 = tf.pt(fz.coord, fz.min_span)
            fx2, fy2 = tf.pt(fz.coord, fz.max_span)
        else:
            fx1, fy1 = tf.pt(fz.min_span, fz.coord)
            fx2, fy2 = tf.pt(fz.max_span, fz.coord)
        parts.append(
            f'<line x1="{fx1:.1f}" y1="{fy1:.1f}" x2="{fx2:.1f}" y2="{fy2:.1f}" '
            f'stroke="{COLOR_FIZZLER}" stroke-width="2.4" stroke-dasharray="5,3" filter="url(#softGlow)"/>'
        )
        parts.append(
            f'<text x="{(fx1 + fx2) * 0.5 + 4:.1f}" y="{min(fy1, fy2) + 12:.1f}" '
            f'fill="{COLOR_FIZZLER}" font-family="Consolas, monospace" font-size="8" font-weight="700">FIZZLER</text>'
        )

    # Portalable concrete wall panels
    for pw in ch.portal_walls:
        # Tangent along wall face (perpendicular to normal)
        tx, ty = -pw.normal.y, pw.normal.x
        half = pw.size[0] * 0.45
        p1x, p1y = tf.pt(pw.pos.x - tx * half, pw.pos.y - ty * half)
        p2x, p2y = tf.pt(pw.pos.x + tx * half, pw.pos.y + ty * half)
        parts.append(
            f'<line x1="{p1x:.1f}" y1="{p1y:.1f}" x2="{p2x:.1f}" y2="{p2y:.1f}" '
            f'stroke="{COLOR_CONCRETE}" stroke-width="4.0" stroke-linecap="round"/>'
        )

    # Static entity markers
    for ent in ch.entities:
        ex, ey = tf.pt(ent.pos.x, ent.pos.y)
        if ent.kind is EntityKind.EXIT_DOOR:
            parts.append(
                f'<rect x="{ex - 6:.1f}" y="{ey - 6:.1f}" width="12" height="12" rx="2.5" '
                f'fill="#064e3b" stroke="{COLOR_EXIT}" stroke-width="1.8"/>'
            )
            parts.append(
                f'<text x="{ex:.1f}" y="{ey + 3:.1f}" text-anchor="middle" fill="#a7f3d0" '
                f'font-family="Consolas, monospace" font-size="8" font-weight="700">E</text>'
            )
        elif ent.kind is EntityKind.FLOOR_BUTTON:
            parts.append(
                f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="6.5" fill="#4c0519" '
                f'stroke="{COLOR_BUTTON}" stroke-width="1.7"/>'
            )
            parts.append(
                f'<text x="{ex:.1f}" y="{ey + 2.8:.1f}" text-anchor="middle" fill="#fecdd3" '
                f'font-family="Consolas, monospace" font-size="7.5" font-weight="700">B</text>'
            )
        elif ent.kind is EntityKind.PEDESTAL_BUTTON:
            parts.append(
                f'<polygon points="{ex:.1f},{ey - 6.5:.1f} {ex + 6.5:.1f},{ey:.1f} '
                f'{ex:.1f},{ey + 6.5:.1f} {ex - 6.5:.1f},{ey:.1f}" '
                f'fill="#451a03" stroke="{COLOR_PEDESTAL}" stroke-width="1.6"/>'
            )
            parts.append(
                f'<text x="{ex:.1f}" y="{ey + 2.8:.1f}" text-anchor="middle" fill="#fde68a" '
                f'font-family="Consolas, monospace" font-size="7.5" font-weight="700">P</text>'
            )
        elif ent.kind is EntityKind.CUBE:
            parts.append(
                f'<rect x="{ex - 5:.1f}" y="{ey - 5:.1f}" width="10" height="10" rx="1.5" '
                f'fill="#334155" stroke="{COLOR_CUBE}" stroke-width="1.5"/>'
            )
            parts.append(
                f'<text x="{ex:.1f}" y="{ey + 2.8:.1f}" text-anchor="middle" fill="{TEXT_PRIMARY}" '
                f'font-family="Consolas, monospace" font-size="7.5" font-weight="700">C</text>'
            )

    # Spawn marker
    spx, spy = tf.pt(ch.player_spawn.x, ch.player_spawn.y)
    parts.append(
        f'<circle cx="{spx:.1f}" cy="{spy:.1f}" r="5.5" fill="#0f172a" '
        f'stroke="#a855f7" stroke-width="1.8"/>'
    )
    parts.append(
        f'<text x="{spx:.1f}" y="{spy + 2.8:.1f}" text-anchor="middle" fill="#e9d5ff" '
        f'font-family="Consolas, monospace" font-size="7.5" font-weight="700">S</text>'
    )

    # Draw active portals from the primary (last/winning) series so walls aren't double-drawn
    drawn_portals: set[tuple[str, int, int]] = set()
    for ser in panel.series:
        for pm in ser.portals:
            key = (pm.color, int(round(pm.pos[0] / 12.0)), int(round(pm.pos[1] / 12.0)))
            if key in drawn_portals:
                continue
            drawn_portals.add(key)
            px, py = tf.pt(pm.pos[0], pm.pos[1])
            pcol = COLOR_BLUE_PORTAL if pm.color == "blue" else COLOR_ORANGE_PORTAL
            parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.2" fill="{pcol}" '
                f'stroke="#0f172a" stroke-width="1.2" filter="url(#softGlow)"/>'
            )

    # Draw trajectory paths (draw FAIL/baseline series first, PASS/hero series on top)
    for ser in panel.series:
        if len(ser.points) < 2:
            continue
        dash_attr = ' stroke-dasharray="5,3.5"' if ser.dashed else ""
        opacity = "0.72" if ser.dashed else "0.96"
        stroke_w = "1.9" if ser.dashed else "2.4"

        segments: list[list[tuple[float, float, TracePoint]]] = [[]]
        teleports: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for idx, pt in enumerate(ser.points):
            sx, sy = tf.pt(pt.pos[0], pt.pos[1])
            if idx > 0:
                prev = ser.points[idx - 1]
                jump_dist = math.hypot(pt.pos[0] - prev.pos[0], pt.pos[1] - prev.pos[1])
                if jump_dist > 96.0:
                    psx, psy = tf.pt(prev.pos[0], prev.pos[1])
                    teleports.append(((psx, psy), (sx, sy)))
                    segments.append([])
            segments[-1].append((sx, sy, pt))

        # Portal teleport arcs
        for (ax, ay), (bx, by) in teleports:
            mx = (ax + bx) * 0.5
            my = (ay + by) * 0.5 - 18.0
            parts.append(
                f'<path d="M {ax:.1f} {ay:.1f} Q {mx:.1f} {my:.1f} {bx:.1f} {by:.1f}" '
                f'fill="none" stroke="{ser.color}" stroke-width="1.3" '
                f'stroke-dasharray="3,3" opacity="0.55"/>'
            )

        # Walking / jumping segments
        for seg in segments:
            if len(seg) < 2:
                continue
            pts_str = " ".join(f"{sx:.1f},{sy:.1f}" for sx, sy, _ in seg)
            parts.append(
                f'<polyline points="{pts_str}" fill="none" stroke="{ser.color}" '
                f'stroke-width="{stroke_w}" stroke-linecap="round" stroke-linejoin="round" '
                f'opacity="{opacity}"{dash_attr}/>'
            )
            # Cube-carry halo
            carry_run: list[tuple[float, float]] = []
            for sx, sy, pt in seg:
                if pt.held is not None:
                    carry_run.append((sx, sy))
                else:
                    if len(carry_run) >= 2:
                        cstr = " ".join(f"{cx:.1f},{cy:.1f}" for cx, cy in carry_run)
                        parts.append(
                            f'<polyline points="{cstr}" fill="none" stroke="{ser.color}" '
                            f'stroke-width="5.0" stroke-linecap="round" opacity="0.22"/>'
                        )
                    carry_run = []
            if len(carry_run) >= 2:
                cstr = " ".join(f"{cx:.1f},{cy:.1f}" for cx, cy in carry_run)
                parts.append(
                    f'<polyline points="{cstr}" fill="none" stroke="{ser.color}" '
                    f'stroke-width="5.0" stroke-linecap="round" opacity="0.22"/>'
                )

            # Directional arrowheads along path
            arrow_stride = max(6, len(seg) // 3)
            for k in range(arrow_stride, len(seg), arrow_stride):
                x1, y1, _ = seg[k - 1]
                x2, y2, _ = seg[k]
                if math.hypot(x2 - x1, y2 - y1) > 2.5:
                    parts.append(_draw_arrowhead(x1, y1, x2, y2, ser.color, size=5.0))

        # Airborne jump-arc rings (visualizing 3D Z-elevation in top-down view!)
        for pt in ser.points:
            if not pt.on_ground and pt.pos[2] > 6.0:
                jx, jy = tf.pt(pt.pos[0], pt.pos[1])
                rad = 2.4 + min(4.2, pt.pos[2] / 12.0)
                parts.append(
                    f'<circle cx="{jx:.1f}" cy="{jy:.1f}" r="{rad:.1f}" fill="none" '
                    f'stroke="#a7f3d0" stroke-width="1.2" opacity="0.85"/>'
                )

        # Event callouts (FIZZLED / DIED)
        for ex, ey, lbl, ecol in ser.events:
            sx, sy = tf.pt(ex, ey)
            parts.append(
                f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="4.0" fill="{ecol}" stroke="#0f172a" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{sx + 6:.1f}" y="{sy - 5:.1f}" fill="{ecol}" '
                f'font-family="Consolas, monospace" font-size="8" font-weight="700">{_esc(lbl)}</text>'
            )

        # Endpoint marker (X for fail, filled dot for pass)
        last_pt = ser.points[-1]
        lx, ly = tf.pt(last_pt.pos[0], last_pt.pos[1])
        if ser.success:
            parts.append(
                f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3.6" fill="{ser.color}" '
                f'stroke="#0f172a" stroke-width="1.2"/>'
            )
        else:
            parts.append(
                f'<path d="M {lx - 3.5:.1f} {ly - 3.5:.1f} L {lx + 3.5:.1f} {ly + 3.5:.1f} '
                f'M {lx + 3.5:.1f} {ly - 3.5:.1f} L {lx - 3.5:.1f} {ly + 3.5:.1f}" '
                f'stroke="{ser.color}" stroke-width="2.0" stroke-linecap="round"/>'
            )

    # Bottom series legend / status badges
    badge_y = y + h - 14.0
    cur_x = x + 14.0
    for ser in panel.series:
        status = "PASS" if ser.success else "FAIL"
        detail = f"{ser.steps}st"
        if ser.extra_badge:
            detail += f", {ser.extra_badge}"
        text = f"{ser.label}: {status} ({detail})"
        dash_attr = ' stroke-dasharray="4,2"' if ser.dashed else ""
        parts.append(
            f'<line x1="{cur_x:.1f}" y1="{badge_y - 3.5:.1f}" x2="{cur_x + 14:.1f}" y2="{badge_y - 3.5:.1f}" '
            f'stroke="{ser.color}" stroke-width="2.5"{dash_attr}/>'
        )
        parts.append(
            f'<text x="{cur_x + 18:.1f}" y="{badge_y:.1f}" fill="{TEXT_PRIMARY if ser.success else TEXT_MUTED}" '
            f'font-family="Consolas, monospace" font-size="9.5">{_esc(text)}</text>'
        )
        cur_x += max(175.0, len(text) * 5.8 + 28.0)

    return "\n".join(parts)


def render_single_panel_svg(
    panel: PanelSpec, *, width: int = 720, height: int = 440
) -> str:
    """Render one standalone chamber comparison panel as a complete SVG document."""
    body = _render_panel_elements(panel, 12.0, 12.0, width - 24.0, height - 24.0)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f"{_svg_defs()}\n"
        f'<rect width="{width}" height="{height}" fill="{BG_OUTER}"/>\n'
        f"{body}\n"
        "</svg>\n"
    )


def render_combined_grid_svg(
    panels: list[PanelSpec],
    *,
    title: str,
    subtitle: str,
    row_headers: list[str] | None = None,
    cols: int = 3,
    panel_w: int = 490,
    panel_h: int = 330,
) -> str:
    """Render a multi-row combined hero dashboard SVG across all suites."""
    rows = math.ceil(len(panels) / cols)
    gap_x = 16
    gap_y = 18
    row_hdr_h = 26 if row_headers else 0
    pad_x = 22
    header_h = 84
    footer_h = 40

    total_w = pad_x * 2 + cols * panel_w + (cols - 1) * gap_x
    total_h = (
        header_h
        + rows * (panel_h + row_hdr_h)
        + max(0, rows - 1) * gap_y
        + footer_h
    )

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" '
        f'viewBox="0 0 {total_w} {total_h}">',
        _svg_defs(),
        f'<rect width="{total_w}" height="{total_h}" fill="{BG_OUTER}"/>',
        # Top title banner
        f'<text x="{pad_x}" y="34" fill="{TEXT_PRIMARY}" '
        f'font-family="Segoe UI, Inter, system-ui, sans-serif" font-size="22" font-weight="800">'
        f"{_esc(title)}</text>",
        f'<text x="{pad_x}" y="56" fill="{TEXT_MUTED}" '
        f'font-family="Segoe UI, Inter, system-ui, sans-serif" font-size="12.5">'
        f"{_esc(subtitle)}</text>",
        f'<line x1="{pad_x}" y1="70" x2="{total_w - pad_x}" y2="70" stroke="#1e2d47" stroke-width="1.2"/>',
    ]

    for r in range(rows):
        row_y = header_h + r * (panel_h + row_hdr_h + gap_y)
        if row_headers and r < len(row_headers):
            parts.append(
                f'<text x="{pad_x}" y="{row_y + 16}" fill="{COLOR_BLUE_PORTAL}" '
                f'font-family="Consolas, monospace" font-size="12" font-weight="700">'
                f"{_esc(row_headers[r])}</text>"
            )
        card_y = row_y + row_hdr_h
        for c in range(cols):
            idx = r * cols + c
            if idx >= len(panels):
                break
            card_x = pad_x + c * (panel_w + gap_x)
            parts.append(
                _render_panel_elements(panels[idx], card_x, card_y, panel_w, panel_h)
            )

    # Bottom symbol key
    fy = total_h - 14
    legend = (
        "KEY:  [S] Spawn   [C] Cube   [B] Floor Button   [P] Timed Pedestal   [E] Exit   "
        "|  Blue/Orange Dots = Active Portals   |  Curved Dash = Portal Teleport   "
        "|  Bright Halo = Carrying Cube   |  Green Rings = Airborne Jump Arc"
    )
    parts.append(
        f'<text x="{pad_x}" y="{fy}" fill="{TEXT_DIM}" '
        f'font-family="Consolas, monospace" font-size="10.5">{_esc(legend)}</text>'
    )
    parts.append("</svg>\n")
    return "\n".join(parts)


def render_generic_campaign_svg(
    trace_path: str | Path,
    *,
    title: str = "Live Portal 1 Zero-Shot Solve — Unmodified Valve testchmb_a_02 (Room 1)",
    subtitle: str = "Frozen 24-node seed7_decoupled.bt (trained strictly on synthetic Chambers 1-4) | Top-down XY path + Z-elevation profile",
    width: int = 1120,
    height: int = 560,
) -> str:
    """Render any arbitrary live Valve campaign JSONL trace (top-down XY + Z-elevation strip)."""
    rows: list[dict[str, Any]] = []
    with Path(trace_path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Trace {trace_path} is empty")

    # Collect bounds across player path, entities, and portals
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    ent_positions: dict[str, tuple[str, float, float, float]] = {}
    portals: dict[str, tuple[float, float, float]] = {}

    for r in rows:
        p = r["player"]["pos"]
        xs.append(float(p[0]))
        ys.append(float(p[1]))
        zs.append(float(p[2]))
        for e in r.get("entities", []):
            ep = e["pos"]
            if e["id"] not in ent_positions or e.get("active"):
                ent_positions[e["id"]] = (e["kind"], float(ep[0]), float(ep[1]), float(ep[2]))
            xs.append(float(ep[0]))
            ys.append(float(ep[1]))
        for col in ("blue", "orange"):
            pr = r.get("portals", {}).get(col)
            if pr and pr.get("active"):
                pp = pr["pos"]
                portals[col] = (float(pp[0]), float(pp[1]), float(pp[2]))
                xs.append(float(pp[0]))
                ys.append(float(pp[1]))

    pad_world = 72.0
    min_x, max_x = min(xs) - pad_world, max(xs) + pad_world
    min_y, max_y = min(ys) - pad_world, max(ys) + pad_world

    map_x, map_y, map_w, map_h = 24.0, 78.0, width - 48.0, 326.0
    elev_x, elev_y, elev_w, elev_h = 24.0, 422.0, width - 48.0, 108.0
    tf = _ViewportTransform(min_x, max_x, min_y, max_y, map_x, map_y, map_w, map_h, pad=18.0)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        _svg_defs(),
        f'<rect width="{width}" height="{height}" fill="{BG_OUTER}"/>',
        f'<text x="24" y="34" fill="{TEXT_PRIMARY}" font-family="Segoe UI, Inter, system-ui, sans-serif" '
        f'font-size="20" font-weight="800">{_esc(title)}</text>',
        f'<text x="24" y="56" fill="{TEXT_MUTED}" font-family="Segoe UI, Inter, system-ui, sans-serif" '
        f'font-size="12">{_esc(subtitle)}</text>',
        # Top-down map panel
        f'<rect x="{map_x}" y="{map_y}" width="{map_w}" height="{map_h}" rx="10" '
        f'fill="{BG_PANEL}" stroke="#1e2d47" stroke-width="1.2"/>',
    ]

    # Blueprint grid inside map box
    gx = math.ceil(min_x / 128.0) * 128.0
    while gx < max_x:
        sx, _ = tf.pt(gx, min_y)
        if map_x + 8 < sx < map_x + map_w - 8:
            parts.append(
                f'<line x1="{sx:.1f}" y1="{map_y + 8:.1f}" x2="{sx:.1f}" y2="{map_y + map_h - 8:.1f}" '
                f'stroke="{GRID_COLOR}" stroke-width="0.7"/>'
            )
        gx += 128.0
    gy = math.ceil(min_y / 128.0) * 128.0
    while gy < max_y:
        _, sy = tf.pt(min_x, gy)
        if map_y + 8 < sy < map_y + map_h - 8:
            parts.append(
                f'<line x1="{map_x + 8:.1f}" y1="{sy:.1f}" x2="{map_x + map_w - 8:.1f}" y2="{sy:.1f}" '
                f'stroke="{GRID_COLOR}" stroke-width="0.7"/>'
            )
        gy += 128.0

    # Highlight sunken pit footprint from pit-zone samples (z < -100)
    pit_pts = [r["player"]["pos"] for r in rows if float(r["player"]["pos"][2]) < -100.0]
    if pit_pts:
        px0 = min(float(p[0]) for p in pit_pts) - 36.0
        px1 = max(float(p[0]) for p in pit_pts) + 36.0
        py0 = min(float(p[1]) for p in pit_pts) - 36.0
        py1 = max(float(p[1]) for p in pit_pts) + 36.0
        rx, ry, rw, rh = tf.rect(px0, py0, px1, py1)
        parts.append(
            f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{rw:.1f}" height="{rh:.1f}" rx="6" '
            f'fill="url(#pitHatch)" stroke="#f59e0b" stroke-width="1.2" stroke-dasharray="4,3"/>'
        )
        parts.append(
            f'<text x="{rx + 6:.1f}" y="{ry + 14:.1f}" fill="#fbbf24" '
            f'font-family="Consolas, monospace" font-size="9.5" font-weight="700">SUNKEN PIT (Zone 1, z=-256u)</text>'
        )

    # Path segments colored by phase: main floor cyan, sunken pit amber, carrying cube bright emerald
    for i in range(1, len(rows)):
        p1 = rows[i - 1]["player"]
        p2 = rows[i]["player"]
        x1, y1 = tf.pt(float(p1["pos"][0]), float(p1["pos"][1]))
        x2, y2 = tf.pt(float(p2["pos"][0]), float(p2["pos"][1]))
        jump_d = math.hypot(float(p2["pos"][0]) - float(p1["pos"][0]), float(p2["pos"][1]) - float(p1["pos"][1]))
        if jump_d > 120.0 or int(p2.get("zone_id", 0)) != int(p1.get("zone_id", 0)) and abs(float(p2["pos"][2]) - float(p1["pos"][2])) > 120.0:
            mx = (x1 + x2) * 0.5
            my = (y1 + y2) * 0.5 - 28.0
            parts.append(
                f'<path d="M {x1:.1f} {y1:.1f} Q {mx:.1f} {my:.1f} {x2:.1f} {y2:.1f}" '
                f'fill="none" stroke="{COLOR_BLUE_PORTAL}" stroke-width="2.0" stroke-dasharray="5,3" filter="url(#softGlow)"/>'
            )
            continue
        in_pit = float(p2["pos"][2]) < -100.0
        holding = p2.get("held_entity_id") is not None
        seg_col = "#34d399" if holding else ("#fbbf24" if in_pit else "#38bdf8")
        seg_w = "3.0" if holding else "2.1"
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{seg_col}" stroke-width="{seg_w}" stroke-linecap="round" opacity="0.92"/>'
        )
        if i % 24 == 0 and math.hypot(x2 - x1, y2 - y1) > 2.5:
            parts.append(_draw_arrowhead(x1, y1, x2, y2, seg_col, size=6.0))

    # Portals
    for col, (px, py, _pz) in portals.items():
        sx, sy = tf.pt(px, py)
        pcol = COLOR_BLUE_PORTAL if col == "blue" else COLOR_ORANGE_PORTAL
        parts.append(
            f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="7.5" fill="{pcol}" '
            f'stroke="#0f172a" stroke-width="1.6" filter="url(#softGlow)"/>'
        )
        parts.append(
            f'<text x="{sx + 10:.1f}" y="{sy + 4:.1f}" fill="{pcol}" '
            f'font-family="Consolas, monospace" font-size="10" font-weight="700">{col.upper()} PORTAL</text>'
        )

    # Entities & Spawn / Finish
    p_start = rows[0]["player"]["pos"]
    s0x, s0y = tf.pt(float(p_start[0]), float(p_start[1]))
    parts.append(f'<circle cx="{s0x:.1f}" cy="{s0y:.1f}" r="7" fill="#0f172a" stroke="#a855f7" stroke-width="2.2"/>')
    parts.append(f'<text x="{s0x - 8:.1f}" y="{s0y - 12:.1f}" fill="#e9d5ff" font-family="Consolas, monospace" font-size="10" font-weight="700">SPAWN</text>')

    p_end = rows[-1]["player"]["pos"]
    e0x, e0y = tf.pt(float(p_end[0]), float(p_end[1]))
    parts.append(f'<circle cx="{e0x:.1f}" cy="{e0y:.1f}" r="7.5" fill="{COLOR_EXIT}" stroke="#0f172a" stroke-width="2"/>')
    parts.append(f'<text x="{e0x - 60:.1f}" y="{e0y - 12:.1f}" fill="#a7f3d0" font-family="Consolas, monospace" font-size="10" font-weight="700">ELEVATOR (PASS)</text>')

    for eid, (ekind, ex, ey, _ez) in ent_positions.items():
        sx, sy = tf.pt(ex, ey)
        if ekind == "floor_button":
            parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="7.5" fill="#4c0519" stroke="{COLOR_BUTTON}" stroke-width="2"/>')
            parts.append(f'<text x="{sx - 28:.1f}" y="{sy + 20:.1f}" fill="#fecdd3" font-family="Consolas, monospace" font-size="9.5">{_esc(eid)}</text>')
        elif ekind == "exit_door":
            parts.append(f'<rect x="{sx - 6:.1f}" y="{sy - 6:.1f}" width="12" height="12" rx="2" fill="#064e3b" stroke="{COLOR_EXIT}" stroke-width="2"/>')
            parts.append(f'<text x="{sx - 24:.1f}" y="{sy + 20:.1f}" fill="#a7f3d0" font-family="Consolas, monospace" font-size="9.5">{_esc(eid)}</text>')

    # Bottom Z-elevation profile panel
    parts.append(
        f'<rect x="{elev_x}" y="{elev_y}" width="{elev_w}" height="{elev_h}" rx="10" '
        f'fill="{BG_PANEL}" stroke="#1e2d47" stroke-width="1.2"/>'
    )
    parts.append(
        f'<text x="{elev_x + 14}" y="{elev_y + 19}" fill="{TEXT_MUTED}" '
        f'font-family="Consolas, monospace" font-size="10.5" font-weight="700">'
        f"Z-ELEVATION PROFILE ({len(rows)} live steps, Blue=Main Floor, Amber=Sunken Pit, Emerald=Carrying Cube)</text>"
    )
    z_min, z_max = min(zs) - 24.0, max(zs) + 32.0
    plot_x0, plot_x1 = elev_x + 48.0, elev_x + elev_w - 20.0
    plot_y0, plot_y1 = elev_y + 28.0, elev_y + elev_h - 18.0
    n_steps = max(1, len(rows) - 1)
    for i in range(1, len(rows)):
        x1 = plot_x0 + ((i - 1) / n_steps) * (plot_x1 - plot_x0)
        x2 = plot_x0 + (i / n_steps) * (plot_x1 - plot_x0)
        z1 = float(rows[i - 1]["player"]["pos"][2])
        z2 = float(rows[i]["player"]["pos"][2])
        y1 = plot_y1 - ((z1 - z_min) / max(1.0, z_max - z_min)) * (plot_y1 - plot_y0)
        y2 = plot_y1 - ((z2 - z_min) / max(1.0, z_max - z_min)) * (plot_y1 - plot_y0)
        holding = rows[i]["player"].get("held_entity_id") is not None
        col = "#34d399" if holding else ("#fbbf24" if z2 < -100.0 else "#38bdf8")
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" stroke-width="2.2"/>')

    parts.append(
        f'<text x="{elev_x + 12}" y="{plot_y0 + 6:.1f}" fill="{TEXT_DIM}" font-family="Consolas, monospace" font-size="9">0u</text>'
    )
    parts.append(
        f'<text x="{elev_x + 6}" y="{plot_y1:.1f}" fill="{TEXT_DIM}" font-family="Consolas, monospace" font-size="9">{int(min(zs))}u</text>'
    )
    parts.append(
        f'<text x="24" y="{height - 10}" fill="{TEXT_DIM}" font-family="Consolas, monospace" font-size="10">'
        f"Source trace: {_esc(Path(trace_path).name)} | Result: {'PASS' if rows[-1].get('success') else 'FAIL'} ({len(rows)} steps)</text>"
    )
    parts.append("</svg>\n")
    return "\n".join(parts)
