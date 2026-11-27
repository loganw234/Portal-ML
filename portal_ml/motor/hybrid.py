"""Hybrid Multi-Skill Motor Controller: routes symbolic commands to specialized neural MLPs.

Layer 1 (Blackboard + LocalSensors) and Layer 2 (frozen evolved Behavior Tree) remain
untouched. When both a precision MoveTo policy (83w) and a sensor-equipped Parkour
policy (208w) are attached, HybridMotorController routes dynamically based on
`skill_mode` ("auto" | "precision" | "parkour") and live Layer-1 physical sensors
(hurdles, walls, toxic acid hazards, and jumpable gaps) with maneuver hysteresis.
"""

from __future__ import annotations

from portal_ml.motor.neural import (
    NeuralAimPolicy,
    NeuralMoveToPolicy,
    NeuralParkourPolicy,
    resolve_move_target,
)
from portal_ml.motor.primitives import MotorCommand, MoveTo, ShootPortal
from portal_ml.schema.blackboard import BlackboardSnapshot


class HybridMotorController:
    """Translates symbolic Layer-2 MotorCommands using optional Phase-2/M7/M8 neural policies."""

    def __init__(
        self,
        moveto_policy: NeuralMoveToPolicy | NeuralParkourPolicy | None = None,
        aim_policy: NeuralAimPolicy | None = None,
        *,
        parkour_policy: NeuralParkourPolicy | None = None,
        skill_mode: str = "auto",
        parkour_hysteresis_ticks: int = 8,
    ) -> None:
        self.moveto_policy = moveto_policy
        self.aim_policy = aim_policy
        self.parkour_policy = parkour_policy
        self.skill_mode = skill_mode
        self.parkour_hysteresis_ticks = parkour_hysteresis_ticks
        self._parkour_hold_ticks: int = 0
        self.precision_ticks: int = 0
        self.parkour_ticks: int = 0
        self.active_skill: str = "scripted"

    def reset_stats(self) -> None:
        self._parkour_hold_ticks = 0
        self.precision_ticks = 0
        self.parkour_ticks = 0
        self.active_skill = "scripted"

    def _should_use_parkour(self, snapshot: BlackboardSnapshot, command: MoveTo) -> bool:
        if self.parkour_policy is None:
            return False
        if self.skill_mode == "parkour":
            return True
        if self.skill_mode == "precision":
            return False

        # Auto sensor-driven routing (stay in parkour while mid-jump, but do not enter
        # parkour solely because a chamber reset or portal exit spawns slightly off-floor)
        if not snapshot.player.on_ground and self._parkour_hold_ticks > 0:
            self._parkour_hold_ticks = self.parkour_hysteresis_ticks
            return True

        s = snapshot.sensors
        is_portal_cmd = command.entity_id.startswith("portal:")
        resolved = resolve_move_target(snapshot, command.entity_id)
        target_dist = (
            snapshot.player.pos.distance_xy(resolved[0])
            if resolved is not None
            else 999.0
        )
        trigger = False
        if s.hazard_ahead > 0.05 or bool(snapshot.hazards):
            trigger = True
        elif not is_portal_cmd and s.step_clearance > 0.0 and target_dist > 72.0:
            # Low jumpable hurdle genuinely ahead of player (not button/pedestal rim at destination)
            trigger = True
        elif not is_portal_cmd and s.step_clearance < 0.0 and target_dist > 96.0:
            # Tall solid wall genuinely BETWEEN player and distant target (not outer wall behind exit)
            trigger = True
        elif (
            not is_portal_cmd
            and not snapshot.portal_walls
            and len(s.floor_probes) >= 1
            and s.floor_probes[0] > 0.55
        ):
            trigger = True

        if trigger:
            self._parkour_hold_ticks = self.parkour_hysteresis_ticks
            return True
        if self._parkour_hold_ticks > 0:
            self._parkour_hold_ticks -= 1
            return True
        return False

    def lower(
        self, snapshot: BlackboardSnapshot, command: MotorCommand
    ) -> MotorCommand:
        if isinstance(command, MoveTo):
            if self._should_use_parkour(snapshot, command) and self.parkour_policy is not None:
                self.parkour_ticks += 1
                self.active_skill = "parkour"
                return self.parkour_policy.to_command(snapshot, command)
            if self.moveto_policy is not None:
                if isinstance(self.moveto_policy, NeuralParkourPolicy):
                    self.parkour_ticks += 1
                    self.active_skill = "parkour"
                else:
                    self.precision_ticks += 1
                    self.active_skill = "precision"
                return self.moveto_policy.to_command(snapshot, command)
        if isinstance(command, ShootPortal) and self.aim_policy is not None:
            return self.aim_policy.to_command(snapshot, command)
        return command
