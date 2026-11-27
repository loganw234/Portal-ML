"""Phase 2: Small, inspectable neuroevolved neural network motor policies.

Pure Python (zero external dependencies). Flat row-major weight layout per layer:
    [out][in] weights followed by [out] biases, with tanh hidden & output activations.
Checkpoints carry explicit input/output schema names and refuse mismatched loads by code.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from portal_ml.motor.primitives import (
    LowLevelAction,
    MotorCommand,
    MoveTo,
    MoveVector,
    NoOp,
    ShootPortal,
)
from portal_ml.schema.blackboard import BlackboardSnapshot, PortalColor, Vec3

MOVETO_INPUT_NAMES: tuple[str, ...] = (
    "dx_body",
    "dy_body",
    "dir_fwd",
    "dir_left",
    "dist_norm",
    "is_portal",
)
MOVETO_OUTPUT_NAMES: tuple[str, ...] = ("forward", "strafe", "yaw_delta")

AIM_INPUT_NAMES: tuple[str, ...] = (
    "sin_yaw_err",
    "cos_yaw_err",
    "yaw_err_norm",
    "dist_norm",
)
AIM_OUTPUT_NAMES: tuple[str, ...] = ("yaw_delta", "fire_gate")

AIM_CONE_DEG = 18.0
FIRE_GATE_THRESHOLD = 0.25


def _wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass(slots=True)
class MLP:
    """Single-hidden-layer tanh MLP with flat row-major genome vector."""

    input_size: int
    hidden_size: int
    output_size: int
    weights: list[float]
    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def param_count(input_size: int, hidden_size: int, output_size: int) -> int:
        return (input_size * hidden_size + hidden_size) + (
            hidden_size * output_size + output_size
        )

    @classmethod
    def random(
        cls,
        input_size: int,
        hidden_size: int,
        output_size: int,
        rng: random.Random,
        scale: float = 0.5,
        input_names: tuple[str, ...] = (),
        output_names: tuple[str, ...] = (),
    ) -> MLP:
        n = cls.param_count(input_size, hidden_size, output_size)
        w = [rng.gauss(0.0, scale) for _ in range(n)]
        return cls(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            weights=w,
            input_names=input_names,
            output_names=output_names,
        )

    def clone(self) -> MLP:
        return MLP(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            weights=list(self.weights),
            input_names=self.input_names,
            output_names=self.output_names,
            metadata=dict(self.metadata),
        )

    def forward(self, inputs: list[float]) -> list[float]:
        w = self.weights
        n_in = self.input_size
        n_hid = self.hidden_size
        n_out = self.output_size

        # Layer 1: [hidden][in] + [hidden] biases
        w1_end = n_hid * n_in
        b1_start = w1_end
        b1_end = b1_start + n_hid

        hidden = [0.0] * n_hid
        for h in range(n_hid):
            row = h * n_in
            acc = w[b1_start + h]
            for i in range(n_in):
                acc += w[row + i] * inputs[i]
            hidden[h] = math.tanh(acc)

        # Layer 2: [out][hidden] + [out] biases
        w2_start = b1_end
        b2_start = w2_start + n_out * n_hid

        outputs = [0.0] * n_out
        for o in range(n_out):
            row = w2_start + o * n_hid
            acc = w[b2_start + o]
            for h in range(n_hid):
                acc += w[row + h] * hidden[h]
            outputs[o] = math.tanh(acc)

        return outputs

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "portal-ml-mlp-v1",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "weights": [round(x, 6) for x in self.weights],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        expected_inputs: tuple[str, ...] | None = None,
        expected_outputs: tuple[str, ...] | None = None,
    ) -> MLP:
        in_names = tuple(data.get("input_names", []))
        out_names = tuple(data.get("output_names", []))
        if expected_inputs is not None and in_names != expected_inputs:
            raise ValueError(
                f"CHECKPOINT-INPUTS: expected {expected_inputs}, got {in_names}"
            )
        if expected_outputs is not None and out_names != expected_outputs:
            raise ValueError(
                f"CHECKPOINT-OUTPUTS: expected {expected_outputs}, got {out_names}"
            )
        n_in = int(data["input_size"])
        n_hid = int(data["hidden_size"])
        n_out = int(data["output_size"])
        weights = [float(x) for x in data["weights"]]
        expected_n = cls.param_count(n_in, n_hid, n_out)
        if len(weights) != expected_n:
            raise ValueError(
                f"CHECKPOINT-WEIGHTS: expected {expected_n} weights, got {len(weights)}"
            )
        return cls(
            input_size=n_in,
            hidden_size=n_hid,
            output_size=n_out,
            weights=weights,
            input_names=in_names,
            output_names=out_names,
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return p

    @classmethod
    def load(
        cls,
        path: str | Path,
        expected_inputs: tuple[str, ...] | None = None,
        expected_outputs: tuple[str, ...] | None = None,
    ) -> MLP:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(
            data, expected_inputs=expected_inputs, expected_outputs=expected_outputs
        )


def resolve_move_target(
    snapshot: BlackboardSnapshot, entity_id: str
) -> tuple[Vec3, bool] | None:
    """Resolve MoveTo target world position and whether it is a portal."""
    if entity_id == "portal:blue":
        p = snapshot.blue_portal
        if not (p.active and snapshot.orange_portal.active):
            return None
        return _portal_approach_point(snapshot.player.pos, p.pos, p.normal), True
    if entity_id == "portal:orange":
        p = snapshot.orange_portal
        if not (p.active and snapshot.blue_portal.active):
            return None
        return _portal_approach_point(snapshot.player.pos, p.pos, p.normal), True
    ent = snapshot.entity_by_id(entity_id)
    if ent is None or ent.zone_id != snapshot.player.zone_id:
        return None
    return ent.pos, False


def _portal_approach_point(player_pos: Vec3, portal_pos: Vec3, normal: Vec3) -> Vec3:
    rx = player_pos.x - portal_pos.x
    ry = player_pos.y - portal_pos.y
    along_n = rx * normal.x + ry * normal.y
    lateral = abs(-rx * normal.y + ry * normal.x)
    if lateral > 28.0 and along_n < 44.0:
        return Vec3(
            portal_pos.x + normal.x * 56.0,
            portal_pos.y + normal.y * 56.0,
            portal_pos.z,
        )
    return Vec3(
        portal_pos.x + normal.x * 8.0,
        portal_pos.y + normal.y * 8.0,
        portal_pos.z,
    )


def moveto_egocentric_features(
    player_pos: Vec3, player_yaw_deg: float, target_pos: Vec3, is_portal: bool
) -> list[float]:
    """Compute the 6 rotation-invariant body-frame inputs for NeuralMoveToPolicy."""
    dx = target_pos.x - player_pos.x
    dy = target_pos.y - player_pos.y
    dist = math.hypot(dx, dy)
    rad = math.radians(player_yaw_deg)
    cos_y = math.cos(rad)
    sin_y = math.sin(rad)

    # Rotate world delta into player's body frame (forward=+x_b, left=+y_b)
    x_body = dx * cos_y + dy * sin_y
    y_body = -dx * sin_y + dy * cos_y

    if dist > 1e-6:
        dir_fwd = x_body / dist
        dir_left = y_body / dist
    else:
        dir_fwd = 0.0
        dir_left = 0.0

    return [
        _clamp(x_body / 256.0, -1.0, 1.0),
        _clamp(y_body / 256.0, -1.0, 1.0),
        dir_fwd,
        dir_left,
        min(dist / 512.0, 1.0),
        1.0 if is_portal else 0.0,
    ]


class NeuralMoveToPolicy:
    """Phase 2 neural replacement for scripted MoveTo(entity_id) (83 weights)."""

    def __init__(self, net: MLP) -> None:
        if net.input_size != len(MOVETO_INPUT_NAMES) or net.output_size != len(
            MOVETO_OUTPUT_NAMES
        ):
            raise ValueError("Invalid MLP shape for NeuralMoveToPolicy")
        self.net = net

    @classmethod
    def load(cls, path: str | Path) -> NeuralMoveToPolicy:
        net = MLP.load(
            path,
            expected_inputs=MOVETO_INPUT_NAMES,
            expected_outputs=MOVETO_OUTPUT_NAMES,
        )
        return cls(net)

    def compute_action(
        self, snapshot: BlackboardSnapshot, command: MotorCommand
    ) -> LowLevelAction:
        if not isinstance(command, MoveTo):
            return LowLevelAction()
        resolved = resolve_move_target(snapshot, command.entity_id)
        if resolved is None:
            return LowLevelAction()
        target_pos, is_portal = resolved
        feats = moveto_egocentric_features(
            snapshot.player.pos, snapshot.player.yaw, target_pos, is_portal
        )
        fwd, strafe, yaw_delta = self.net.forward(feats)
        return LowLevelAction(forward=fwd, strafe=strafe, yaw_delta=yaw_delta)

    def to_command(
        self, snapshot: BlackboardSnapshot, command: MoveTo
    ) -> MotorCommand:
        act = self.compute_action(snapshot, command)
        return MoveVector(
            forward=act.forward, strafe=act.strafe, yaw_delta=act.yaw_delta
        )


def aim_egocentric_features(
    player_pos: Vec3, player_yaw_deg: float, wall_pos: Vec3
) -> tuple[list[float], float]:
    """Compute the 4 egocentric aiming inputs and signed yaw error in degrees."""
    dx = wall_pos.x - player_pos.x
    dy = wall_pos.y - player_pos.y
    dist = math.hypot(dx, dy)
    target_yaw = math.degrees(math.atan2(dy, dx))
    err_deg = _wrap_deg(target_yaw - player_yaw_deg)
    err_rad = math.radians(err_deg)
    feats = [
        math.sin(err_rad),
        math.cos(err_rad),
        err_deg / 180.0,
        min(dist / 512.0, 1.0),
    ]
    return feats, err_deg


class NeuralAimPolicy:
    """Phase 2 neural aiming controller for ShootPortal(color, wall_id) (44 weights)."""

    def __init__(self, net: MLP) -> None:
        if net.input_size != len(AIM_INPUT_NAMES) or net.output_size != len(
            AIM_OUTPUT_NAMES
        ):
            raise ValueError("Invalid MLP shape for NeuralAimPolicy")
        self.net = net

    @classmethod
    def load(cls, path: str | Path) -> NeuralAimPolicy:
        net = MLP.load(
            path,
            expected_inputs=AIM_INPUT_NAMES,
            expected_outputs=AIM_OUTPUT_NAMES,
        )
        return cls(net)

    def compute_action(
        self, snapshot: BlackboardSnapshot, command: MotorCommand
    ) -> LowLevelAction:
        if not isinstance(command, ShootPortal):
            return LowLevelAction()
        wall = snapshot.wall_by_id(command.wall_id)
        if wall is None:
            return LowLevelAction()
        feats, err_deg = aim_egocentric_features(
            snapshot.player.pos, snapshot.player.yaw, wall.pos
        )
        yaw_delta, fire_gate = self.net.forward(feats)
        should_fire = fire_gate > FIRE_GATE_THRESHOLD and abs(err_deg) <= AIM_CONE_DEG
        return LowLevelAction(
            yaw_delta=yaw_delta,
            fire_blue=should_fire and command.color is PortalColor.BLUE,
            fire_orange=should_fire and command.color is PortalColor.ORANGE,
        )

    def to_command(
        self, snapshot: BlackboardSnapshot, command: ShootPortal
    ) -> MotorCommand:
        wall = snapshot.wall_by_id(command.wall_id)
        if wall is None:
            return NoOp()
        feats, err_deg = aim_egocentric_features(
            snapshot.player.pos, snapshot.player.yaw, wall.pos
        )
        yaw_delta, fire_gate = self.net.forward(feats)
        if fire_gate > FIRE_GATE_THRESHOLD and abs(err_deg) <= AIM_CONE_DEG:
            return command
        return MoveVector(forward=0.0, strafe=0.0, yaw_delta=yaw_delta)
