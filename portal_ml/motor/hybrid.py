"""Hybrid Motor Controller: swaps individual Phase-1 primitives for Phase-2 neural policies.

Layer 1 (Blackboard) and Layer 2 (frozen evolved Behavior Tree) remain completely
untouched; HybridMotorController intercepts high-level MoveTo / ShootPortal commands
and lowers them through pre-trained neuroevolved MLPs into continuous MoveVector steps.
"""

from __future__ import annotations

from portal_ml.motor.neural import (
    NeuralAimPolicy,
    NeuralMoveToPolicy,
    NeuralParkourPolicy,
)
from portal_ml.motor.primitives import MotorCommand, MoveTo, ShootPortal
from portal_ml.schema.blackboard import BlackboardSnapshot


class HybridMotorController:
    """Translates symbolic Layer-2 MotorCommands using optional Phase-2/M7 neural policies."""

    def __init__(
        self,
        moveto_policy: NeuralMoveToPolicy | NeuralParkourPolicy | None = None,
        aim_policy: NeuralAimPolicy | None = None,
    ) -> None:
        self.moveto_policy = moveto_policy
        self.aim_policy = aim_policy

    def lower(
        self, snapshot: BlackboardSnapshot, command: MotorCommand
    ) -> MotorCommand:
        if isinstance(command, MoveTo) and self.moveto_policy is not None:
            return self.moveto_policy.to_command(snapshot, command)
        if isinstance(command, ShootPortal) and self.aim_policy is not None:
            return self.aim_policy.to_command(snapshot, command)
        return command
