"""Exact JSON Blackboard schema shared by the mock 3D simulator and Source Engine plugin.

All positions, velocities, and dimensions use Source Engine Hammer units
(1 unit ~= 0.75 in; standard test-chamber panel = 128x128 units).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EntityKind(str, Enum):
    CUBE = "cube"
    FLOOR_BUTTON = "floor_button"
    PEDESTAL_BUTTON = "pedestal_button"
    EXIT_DOOR = "exit_door"


class PortalColor(str, Enum):
    BLUE = "blue"
    ORANGE = "orange"

    def other(self) -> PortalColor:
        return PortalColor.ORANGE if self is PortalColor.BLUE else PortalColor.BLUE


@dataclass(frozen=True, slots=True)
class Vec3:
    x: float
    y: float
    z: float = 0.0

    def to_list(self) -> list[float]:
        return [round(self.x, 3), round(self.y, 3), round(self.z, 3)]

    @classmethod
    def from_list(cls, data: list[float] | tuple[float, ...]) -> Vec3:
        return cls(float(data[0]), float(data[1]), float(data[2]))

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, s: float) -> Vec3:
        return Vec3(self.x * s, self.y * s, self.z * s)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def length_xy(self) -> float:
        return math.hypot(self.x, self.y)

    def distance_to(self, other: Vec3) -> float:
        return (self - other).length()

    def distance_xy(self, other: Vec3) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def normalized(self) -> Vec3:
        mag = self.length()
        if mag < 1e-9:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(self.x / mag, self.y / mag, self.z / mag)

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z


@dataclass(slots=True)
class LocalSensors:
    """Egocentric non-visual physical geometry & hazard probes ("touch / LiDAR").

    Populated by both MockPortalEnv (analytic raycasts) and the C++ Source Engine
    plugin (native IEngineTrace::TraceRay). All distances are normalized to [0, 1].
    """

    # 8 horizontal rays at [0, +45, +90, +135, 180, -135, -90, -45] deg rel yaw (1.0 = clear >= 256u)
    wall_rays: list[float] = field(default_factory=lambda: [1.0] * 8)
    # 3 downward floor-drop probes at [0, +35, -35] deg, 48u ahead (0.28 = flat floor, 1.0 = drop-off)
    floor_probes: list[float] = field(default_factory=lambda: [0.28, 0.28, 0.28])
    # Forward hurdle probe: +1.0 = jumpable low step/hurdle, 0.0 = clear, -1.0 = tall unjumpable wall
    step_clearance: float = 0.0
    # Forward hazard proximity within 192u (0.0 = clear, 1.0 = touching acid/hurt volume)
    hazard_ahead: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_rays": [round(float(v), 3) for v in self.wall_rays],
            "floor_probes": [round(float(v), 3) for v in self.floor_probes],
            "step_clearance": round(float(self.step_clearance), 3),
            "hazard_ahead": round(float(self.hazard_ahead), 3),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LocalSensors:
        if not data:
            return cls()
        wr = [float(v) for v in data.get("wall_rays", [1.0] * 8)]
        fp = [float(v) for v in data.get("floor_probes", [0.28, 0.28, 0.28])]
        return cls(
            wall_rays=wr if len(wr) == 8 else ([1.0] * 8),
            floor_probes=fp if len(fp) == 3 else ([0.28, 0.28, 0.28]),
            step_clearance=float(data.get("step_clearance", 0.0)),
            hazard_ahead=float(data.get("hazard_ahead", 0.0)),
        )


@dataclass(slots=True)
class HazardVolume:
    """Axis-aligned lethal/damaging volume (e.g. Source trigger_hurt / toxic acid goo)."""

    id: str
    kind: str  # "acid" | "trigger_hurt" | "fizzler"
    min_pos: Vec3
    max_pos: Vec3
    damage_per_sec: float = 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "min": self.min_pos.to_list(),
            "max": self.max_pos.to_list(),
            "dps": round(self.damage_per_sec, 1),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HazardVolume:
        return cls(
            id=str(data["id"]),
            kind=str(data.get("kind", "trigger_hurt")),
            min_pos=Vec3.from_list(data["min"]),
            max_pos=Vec3.from_list(data["max"]),
            damage_per_sec=float(data.get("dps", 100.0)),
        )


@dataclass(slots=True)
class PlayerState:
    pos: Vec3
    vel: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    yaw: float = 0.0
    pitch: float = 0.0
    zone_id: int = 0
    held_entity_id: str | None = None
    health: float = 100.0
    on_ground: bool = True
    ducking: bool = False
    water_level: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pos": self.pos.to_list(),
            "vel": self.vel.to_list(),
            "yaw": round(self.yaw, 2),
            "pitch": round(self.pitch, 2),
            "zone_id": self.zone_id,
            "held_entity_id": self.held_entity_id,
            "health": round(self.health, 1),
            "on_ground": self.on_ground,
            "ducking": self.ducking,
            "water_level": self.water_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlayerState:
        return cls(
            pos=Vec3.from_list(data["pos"]),
            vel=Vec3.from_list(data.get("vel", [0.0, 0.0, 0.0])),
            yaw=float(data.get("yaw", 0.0)),
            pitch=float(data.get("pitch", 0.0)),
            zone_id=int(data.get("zone_id", 0)),
            held_entity_id=data.get("held_entity_id"),
            health=float(data.get("health", 100.0)),
            on_ground=bool(data.get("on_ground", True)),
            ducking=bool(data.get("ducking", False)),
            water_level=int(data.get("water_level", 0)),
        )


@dataclass(slots=True)
class EntityState:
    id: str
    kind: EntityKind
    pos: Vec3
    vel: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    zone_id: int = 0
    active: bool = False  # exit_door: is_open; button: is_pressed; cube: on_button
    held: bool = False
    timer: float = 0.0  # remaining seconds for timed pedestal buttons

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "pos": self.pos.to_list(),
            "vel": self.vel.to_list(),
            "zone_id": self.zone_id,
            "active": self.active,
            "held": self.held,
            "timer": round(self.timer, 3),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EntityState:
        return cls(
            id=str(data["id"]),
            kind=EntityKind(data["kind"]),
            pos=Vec3.from_list(data["pos"]),
            vel=Vec3.from_list(data.get("vel", [0.0, 0.0, 0.0])),
            zone_id=int(data.get("zone_id", 0)),
            active=bool(data.get("active", False)),
            held=bool(data.get("held", False)),
            timer=float(data.get("timer", 0.0)),
        )


@dataclass(slots=True)
class PortalWall:
    id: str
    pos: Vec3
    normal: Vec3
    zone_id: int = 0
    size: tuple[float, float] = (128.0, 128.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "pos": self.pos.to_list(),
            "normal": self.normal.to_list(),
            "zone_id": self.zone_id,
            "size": [round(self.size[0], 2), round(self.size[1], 2)],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PortalWall:
        sz = data.get("size", [128.0, 128.0])
        return cls(
            id=str(data["id"]),
            pos=Vec3.from_list(data["pos"]),
            normal=Vec3.from_list(data["normal"]),
            zone_id=int(data.get("zone_id", 0)),
            size=(float(sz[0]), float(sz[1])),
        )


@dataclass(slots=True)
class ActivePortal:
    color: PortalColor
    active: bool = False
    wall_id: str | None = None
    pos: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    normal: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    zone_id: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "color": self.color.value,
            "active": self.active,
            "wall_id": self.wall_id,
            "pos": self.pos.to_list(),
            "normal": self.normal.to_list(),
            "zone_id": self.zone_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActivePortal:
        return cls(
            color=PortalColor(data["color"]),
            active=bool(data.get("active", False)),
            wall_id=data.get("wall_id"),
            pos=Vec3.from_list(data.get("pos", [0.0, 0.0, 0.0])),
            normal=Vec3.from_list(data.get("normal", [0.0, 0.0, 0.0])),
            zone_id=int(data.get("zone_id", -1)),
        )


@dataclass(slots=True)
class WorkingMemory:
    """Mutable per-episode Blackboard scratchpad owned by the reasoning layer.

    ``role_overrides`` is written optionally by Layer 4 (Strategic Planner) to
    bind symbolic roles (``cube``, ``floor_button``, ``pedestal_button``, ``exit``)
    to specific entity IDs — or to ``None`` to temporarily mask a secondary role
    during the current subgoal. When empty (3-layer mode), resolution is
    100% identical to ``snapshot.first_of_kind(kind)``.
    """

    target_role: str | None = None
    target_entity_id: str | None = None
    role_overrides: dict[str, str | None] = field(default_factory=dict)

    def clear(self) -> None:
        self.target_role = None
        self.target_entity_id = None
        self.role_overrides.clear()


@dataclass(slots=True)
class BlackboardSnapshot:
    """Exact ground-truth symbolic world state emitted every tick."""

    tick: int
    dt: float
    chamber_id: int
    player: PlayerState
    entities: list[EntityState] = field(default_factory=list)
    portal_walls: list[PortalWall] = field(default_factory=list)
    blue_portal: ActivePortal = field(
        default_factory=lambda: ActivePortal(color=PortalColor.BLUE)
    )
    orange_portal: ActivePortal = field(
        default_factory=lambda: ActivePortal(color=PortalColor.ORANGE)
    )
    sensors: LocalSensors = field(default_factory=LocalSensors)
    hazards: list[HazardVolume] = field(default_factory=list)
    done: bool = False
    success: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "dt": round(self.dt, 5),
            "chamber_id": self.chamber_id,
            "player": self.player.to_dict(),
            "entities": [e.to_dict() for e in self.entities],
            "portal_walls": [w.to_dict() for w in self.portal_walls],
            "portals": {
                "blue": self.blue_portal.to_dict(),
                "orange": self.orange_portal.to_dict(),
            },
            "sensors": self.sensors.to_dict(),
            "hazards": [h.to_dict() for h in self.hazards],
            "done": self.done,
            "success": self.success,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BlackboardSnapshot:
        portals = data.get("portals", {})
        blue_data = portals.get("blue", {"color": "blue", "active": False})
        orange_data = portals.get("orange", {"color": "orange", "active": False})
        return cls(
            tick=int(data["tick"]),
            dt=float(data["dt"]),
            chamber_id=int(data["chamber_id"]),
            player=PlayerState.from_dict(data["player"]),
            entities=[EntityState.from_dict(e) for e in data.get("entities", [])],
            portal_walls=[
                PortalWall.from_dict(w) for w in data.get("portal_walls", [])
            ],
            blue_portal=ActivePortal.from_dict(blue_data),
            orange_portal=ActivePortal.from_dict(orange_data),
            sensors=LocalSensors.from_dict(data.get("sensors")),
            hazards=[HazardVolume.from_dict(h) for h in data.get("hazards", [])],
            done=bool(data.get("done", False)),
            success=bool(data.get("success", False)),
        )

    @classmethod
    def from_json(cls, raw: str) -> BlackboardSnapshot:
        return cls.from_dict(json.loads(raw))

    # --- Symbolic query helpers used by the BT evaluator ---

    def entity_by_id(self, entity_id: str) -> EntityState | None:
        for ent in self.entities:
            if ent.id == entity_id:
                return ent
        return None

    def entities_of_kind(self, kind: EntityKind) -> list[EntityState]:
        return [e for e in self.entities if e.kind is kind]

    def first_of_kind(self, kind: EntityKind) -> EntityState | None:
        for ent in self.entities:
            if ent.kind is kind:
                return ent
        return None

    def wall_by_id(self, wall_id: str) -> PortalWall | None:
        for w in self.portal_walls:
            if w.id == wall_id:
                return w
        return None

    def wall_in_zone(self, zone_id: int) -> PortalWall | None:
        candidates = [w for w in self.portal_walls if w.zone_id == zone_id]
        if not candidates:
            return None
        return min(candidates, key=lambda w: self.player.pos.distance_xy(w.pos))

    def portal(self, color: PortalColor) -> ActivePortal:
        return self.blue_portal if color is PortalColor.BLUE else self.orange_portal

    def portal_in_zone(self, zone_id: int) -> ActivePortal | None:
        """Return an active portal located in `zone_id` when both portals are linked."""
        if not (self.blue_portal.active and self.orange_portal.active):
            return None
        if self.blue_portal.zone_id == zone_id:
            return self.blue_portal
        if self.orange_portal.zone_id == zone_id:
            return self.orange_portal
        return None

    def portals_bridge_zones(self, zone_a: int, zone_b: int) -> bool:
        if zone_a == zone_b:
            return False
        if not (self.blue_portal.active and self.orange_portal.active):
            return False
        return {self.blue_portal.zone_id, self.orange_portal.zone_id} == {
            zone_a,
            zone_b,
        }
