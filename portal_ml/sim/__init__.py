"""3D mock simulator and 5-chamber curriculum specifications."""

from portal_ml.sim.chambers import (
    CHAMBERS,
    ChamberSpec,
    EntitySpawn,
    PitBox,
    WalkableZone,
    get_chamber,
)
from portal_ml.sim.mock_env import MockPortalEnv, PortalEnv, StepResult

__all__ = [
    "CHAMBERS",
    "ChamberSpec",
    "EntitySpawn",
    "MockPortalEnv",
    "PitBox",
    "PortalEnv",
    "StepResult",
    "WalkableZone",
    "get_chamber",
]
