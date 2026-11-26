"""Layer 3: Motor Execution Primitives.

Phase 1 executes these deterministically inside either the Python mock 3D
simulator or the C++ Source Engine server plugin.
Phase 2 can swap any primitive for a pre-trained neuroevolved neural network
implementing `MotorPolicy` without changing Layer 1 or Layer 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from portal_ml.schema.blackboard import BlackboardSnapshot, PortalColor


class MotorCommandKind(str, Enum):
    NOOP = "noop"
    MOVE_TO = "move_to"
    GRAB = "grab"
    DROP = "drop"
    PRESS = "press"
    SHOOT_PORTAL = "shoot_portal"


@dataclass(frozen=True, slots=True)
class MotorCommand:
    """Base wire-serializable motor command sent to the simulator or Source plugin."""

    kind: MotorCommandKind

    def to_dict(self) -> dict[str, Any]:
        return {"cmd": self.kind.value}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> MotorCommand:
        cmd = MotorCommandKind(data["cmd"])
        if cmd is MotorCommandKind.MOVE_TO:
            return MoveTo(entity_id=str(data["entity_id"]))
        if cmd is MotorCommandKind.GRAB:
            return Grab(entity_id=str(data["entity_id"]))
        if cmd is MotorCommandKind.DROP:
            return Drop()
        if cmd is MotorCommandKind.PRESS:
            return Press(entity_id=str(data["entity_id"]))
        if cmd is MotorCommandKind.SHOOT_PORTAL:
            return ShootPortal(
                color=PortalColor(data["color"]),
                wall_id=str(data["wall_id"]),
            )
        return NoOp()


@dataclass(frozen=True, slots=True)
class NoOp(MotorCommand):
    kind: MotorCommandKind = MotorCommandKind.NOOP


@dataclass(frozen=True, slots=True)
class MoveTo(MotorCommand):
    entity_id: str = ""
    kind: MotorCommandKind = MotorCommandKind.MOVE_TO

    def to_dict(self) -> dict[str, Any]:
        return {"cmd": self.kind.value, "entity_id": self.entity_id}


@dataclass(frozen=True, slots=True)
class Grab(MotorCommand):
    entity_id: str = ""
    kind: MotorCommandKind = MotorCommandKind.GRAB

    def to_dict(self) -> dict[str, Any]:
        return {"cmd": self.kind.value, "entity_id": self.entity_id}


@dataclass(frozen=True, slots=True)
class Drop(MotorCommand):
    kind: MotorCommandKind = MotorCommandKind.DROP


@dataclass(frozen=True, slots=True)
class Press(MotorCommand):
    entity_id: str = ""
    kind: MotorCommandKind = MotorCommandKind.PRESS

    def to_dict(self) -> dict[str, Any]:
        return {"cmd": self.kind.value, "entity_id": self.entity_id}


@dataclass(frozen=True, slots=True)
class ShootPortal(MotorCommand):
    color: PortalColor = PortalColor.BLUE
    wall_id: str = ""
    kind: MotorCommandKind = MotorCommandKind.SHOOT_PORTAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "cmd": self.kind.value,
            "color": self.color.value,
            "wall_id": self.wall_id,
        }


@dataclass(frozen=True, slots=True)
class LowLevelAction:
    """Continuous engine control vector (Phase 2 neural controller output)."""

    forward: float = 0.0  # [-1, 1]
    strafe: float = 0.0  # [-1, 1]
    yaw_delta: float = 0.0  # degrees/tick
    pitch_delta: float = 0.0  # degrees/tick
    use: bool = False
    fire_blue: bool = False
    fire_orange: bool = False


class MotorPolicy(Protocol):
    """Phase 2 hook: maps (BlackboardSnapshot, MotorCommand) -> LowLevelAction."""

    def compute_action(
        self, snapshot: BlackboardSnapshot, command: MotorCommand
    ) -> LowLevelAction: ...
