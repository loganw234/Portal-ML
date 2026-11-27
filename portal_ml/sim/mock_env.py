"""Fast deterministic 3D physics/state mock simulator for Chambers 1-5.

Executes thousands of episodes per second in pure Python while maintaining
identical coordinates, Blackboard snapshots, and MotorCommand semantics as the
C++ Source Engine server plugin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from portal_ml.bt.predicates import INTERACT_RADIUS
from portal_ml.motor.primitives import (
    Drop,
    Grab,
    MotorCommand,
    MoveTo,
    MoveVector,
    Press,
    ShootPortal,
)
from portal_ml.schema.blackboard import (
    ActivePortal,
    BlackboardSnapshot,
    EntityKind,
    EntityState,
    HazardVolume,
    LocalSensors,
    PlayerState,
    PortalColor,
    PortalWall,
    Vec3,
)
from portal_ml.sim.chambers import ChamberSpec, ObstacleBox, WalkableZone, get_chamber

# Physics constants (Source Engine Hammer units)
FIXED_DT = 0.1  # 10 Hz high-level stepping (matches host_framerate 0.05 x 2 sub-ticks)
PLAYER_SPEED = 240.0  # units/sec (~24 u per step)
MAX_YAW_RATE = 360.0  # deg/sec (up to 36 deg per step for continuous Phase-2 control)
MAX_PITCH_RATE = 240.0  # deg/sec (up to 24 deg per step)
GRAVITY = 600.0  # Source Engine default sv_gravity = 600 u/s^2
JUMP_SPEED = 240.0  # Apex ~48u, airtime ~0.8s (~192u horizontal jump span at full run)
PLAYER_RADIUS = 16.0  # Source Engine player hull half-width
SENSOR_RAY_RANGE = 256.0
SENSOR_HAZARD_RANGE = 192.0
PORTAL_ENTER_RADIUS = 32.0
PORTAL_EXIT_OFFSET = 44.0
BUTTON_TRIGGER_RADIUS = 52.0
EXIT_TRIGGER_RADIUS = 36.0
CUBE_HOLD_OFFSET = 40.0
CUBE_HOLD_HEIGHT = 40.0


@dataclass(slots=True)
class StepResult:
    snapshot: BlackboardSnapshot
    reward: float
    done: bool
    success: bool
    milestones: list[str] = field(default_factory=list)


class PortalEnv(Protocol):
    """Common environment interface shared by MockPortalEnv and SourcePortalEnv."""

    def reset(self, chamber_id: int) -> BlackboardSnapshot: ...

    def step(self, command: MotorCommand) -> StepResult: ...


class MockPortalEnv:
    """In-memory 3D state simulator for fast BT evolution and debugging."""

    def __init__(self, dt: float = FIXED_DT, reuse_snapshot: bool = False) -> None:
        self.dt = dt
        self.reuse_snapshot = reuse_snapshot
        self._spec: ChamberSpec = get_chamber(1)
        self._tick: int = 0
        self._player: PlayerState = PlayerState(pos=Vec3(0.0, 0.0, 0.0))
        self._entities: dict[str, EntityState] = {}
        self._entity_list: list[EntityState] = []
        self._walls: dict[str, PortalWall] = {}
        self._wall_list: list[PortalWall] = []
        self._blue: ActivePortal = ActivePortal(color=PortalColor.BLUE)
        self._orange: ActivePortal = ActivePortal(color=PortalColor.ORANGE)
        self._done: bool = False
        self._success: bool = False
        self._achieved_milestones: set[str] = set()
        self._portal_cooldown: int = 0
        self._sensors: LocalSensors = LocalSensors()
        self._hazards: list[HazardVolume] = []
        self._shared_snapshot: BlackboardSnapshot | None = None
        self.reset(1)

    @property
    def spec(self) -> ChamberSpec:
        return self._spec

    def reset(
        self, chamber_id: int = 1, *, spec: ChamberSpec | None = None
    ) -> BlackboardSnapshot:
        """Sub-millisecond in-place reset to any chamber spec."""
        self._spec = spec if spec is not None else get_chamber(chamber_id)
        self._spawn_by_id = {s.id: s for s in self._spec.entities}
        self._tick = 0
        self._done = False
        self._success = False
        self._achieved_milestones.clear()
        self._portal_cooldown = 0

        self._player = PlayerState(
            pos=self._spec.player_spawn,
            vel=Vec3(0.0, 0.0, 0.0),
            yaw=self._spec.player_spawn_yaw,
            pitch=0.0,
            zone_id=self._spec.player_spawn_zone,
            held_entity_id=None,
        )

        self._entities = {}
        for spawn in self._spec.entities:
            self._entities[spawn.id] = EntityState(
                id=spawn.id,
                kind=spawn.kind,
                pos=spawn.pos,
                vel=Vec3(0.0, 0.0, 0.0),
                zone_id=spawn.zone_id,
                active=spawn.initially_active,
                held=False,
                timer=0.0,
            )

        self._entity_list = list(self._entities.values())
        self._walls = {w.id: w for w in self._spec.portal_walls}
        self._wall_list = list(self._spec.portal_walls)
        self._blue = ActivePortal(color=PortalColor.BLUE)
        self._orange = ActivePortal(color=PortalColor.ORANGE)
        self._hazards = list(self._spec.hazards)
        self._sensors = (
            self._compute_local_sensors()
            if (not self.reuse_snapshot or self._spec.obstacles or self._spec.hazards)
            else LocalSensors()
        )
        self._shared_snapshot = BlackboardSnapshot(
            tick=0,
            dt=self.dt,
            chamber_id=self._spec.chamber_id,
            player=self._player,
            entities=self._entity_list,
            portal_walls=self._wall_list,
            blue_portal=self._blue,
            orange_portal=self._orange,
            sensors=self._sensors,
            hazards=self._hazards,
            done=False,
            success=False,
        )
        return self._make_snapshot()

    def apply_training_augmentation(self, kind: str) -> BlackboardSnapshot:
        """Apply a training-only causal invariance perturbation to Chambers 3 or 4.

        Breaks two spurious correlations present in the minimal 4-chamber training set
        without leaking any held-out chamber geometry:
        - ``ch3_distractor_walls``: adds non-essential portalable walls to single-zone
          Chamber 3 so evolution cannot equate ``wall_exists`` with ``target_unreachable``.
        - ``ch4_stale_portals``: pre-places both portals on zone-0 walls in Chamber 4 so
          evolution cannot equate ``portals_active`` with ``portals_bridge(target)``.
        """
        if kind == "ch3_distractor_walls":
            if self._spec.chamber_id != 3:
                raise ValueError("ch3_distractor_walls only applies to Chamber 3")
            extra_walls = [
                PortalWall(
                    id="c3_distractor_a",
                    pos=Vec3(256.0, 4096.0, 64.0),
                    normal=Vec3(0.0, 1.0, 0.0),
                    zone_id=0,
                ),
                PortalWall(
                    id="c3_distractor_b",
                    pos=Vec3(256.0, 4480.0, 64.0),
                    normal=Vec3(0.0, -1.0, 0.0),
                    zone_id=0,
                ),
            ]
            for w in extra_walls:
                self._walls[w.id] = w
                self._wall_list.append(w)
        elif kind == "ch4_stale_portals":
            if self._spec.chamber_id != 4:
                raise ValueError("ch4_stale_portals only applies to Chamber 4")
            wall_a = self._walls["c4_wall_a"]
            stale_wall = PortalWall(
                id="c4_wall_a_stale",
                pos=Vec3(128.0, 6400.0, 64.0),
                normal=Vec3(0.0, -1.0, 0.0),
                zone_id=0,
            )
            self._walls[stale_wall.id] = stale_wall
            self._wall_list.append(stale_wall)
            self._blue.active = True
            self._blue.wall_id = wall_a.id
            self._blue.pos = wall_a.pos
            self._blue.normal = wall_a.normal
            self._blue.zone_id = wall_a.zone_id
            self._orange.active = True
            self._orange.wall_id = stale_wall.id
            self._orange.pos = stale_wall.pos
            self._orange.normal = stale_wall.normal
            self._orange.zone_id = stale_wall.zone_id
        else:
            raise ValueError(f"Unknown training augmentation: {kind!r}")
        return self._make_snapshot()

    def step(self, command: MotorCommand) -> StepResult:
        if self._done:
            return StepResult(
                snapshot=self._make_snapshot(),
                reward=0.0,
                done=True,
                success=self._success,
            )

        self._tick += 1
        if self._portal_cooldown > 0:
            self._portal_cooldown -= 1

        step_milestones: list[str] = []
        reward = -0.05  # step penalty encourages concise, efficient plans

        # 1. Execute motor primitive
        prev_pos = self._player.pos
        prev_zone = self._player.zone_id
        prev_vz = self._player.vel.z
        self._player.vel = Vec3(0.0, 0.0, prev_vz)
        if isinstance(command, MoveTo):
            self._exec_move_to(command.entity_id, step_milestones)
        elif isinstance(command, MoveVector):
            self._exec_move_vec(command, step_milestones)
        elif isinstance(command, Grab):
            self._exec_grab(command.entity_id, step_milestones)
        elif isinstance(command, Drop):
            self._exec_drop()
        elif isinstance(command, Press):
            self._exec_press(command.entity_id, step_milestones)
        elif isinstance(command, ShootPortal):
            self._exec_shoot_portal(command.color, command.wall_id, step_milestones)

        # 1a. Advance vertical jump/gravity physics if airborne
        if not self._player.on_ground or self._player.vel.z != 0.0 or self._player.pos.z > self._zone(self._player.zone_id).floor_z:
            reward += self._step_vertical_physics(step_milestones)

        # 1b. Check fizzler barriers (trigger_portal_cleanser) if carrying a cube
        if self._spec.fizzlers and self._player.held_entity_id is not None:
            self._check_fizzler_crossing(prev_pos, prev_zone, step_milestones)

        # 1c. Check damaging hazards (acid goo / trigger_hurt)
        if self._spec.hazards and not self._done:
            reward += self._update_hazards(step_milestones)

        # 2. Keep held cube attached in front of player
        self._update_held_cube()

        # 3. Update timers, floor buttons, and door logic
        self._update_world_logic(step_milestones)

        # 4. Check milestone rewards (shaping for Chambers 1-4; Chamber 5 ignores shaping in evaluator)
        reward += self._score_milestones(step_milestones)

        # 5. Check exit condition
        exit_ent = self._first_entity(EntityKind.EXIT_DOOR)
        if (
            not self._done
            and exit_ent is not None
            and exit_ent.active
            and exit_ent.zone_id == self._player.zone_id
            and self._player.pos.distance_xy(exit_ent.pos) <= EXIT_TRIGGER_RADIUS
        ):
            self._done = True
            self._success = True
            if "exit_reached" not in self._achieved_milestones:
                self._achieved_milestones.add("exit_reached")
                step_milestones.append("exit_reached")
                reward += 100.0
        elif not self._done and self._tick >= self._spec.max_steps:
            self._done = True
            self._success = False

        if not self.reuse_snapshot or self._spec.obstacles or self._spec.hazards:
            self._sensors = self._compute_local_sensors()

        return StepResult(
            snapshot=self._make_snapshot(),
            reward=reward,
            done=self._done,
            success=self._success,
            milestones=step_milestones,
        )

    # --- Motor primitive implementations ---

    def _zone(self, zone_id: int) -> WalkableZone:
        for z in self._spec.zones:
            if z.zone_id == zone_id:
                return z
        return self._spec.zones[0]

    def _exec_move_to(self, target_id: str, milestones: list[str]) -> None:
        target_pos: Vec3 | None = None
        target_portal: ActivePortal | None = None

        if target_id == "portal:blue":
            if self._blue.active and self._orange.active and self._blue.zone_id == self._player.zone_id:
                target_portal = self._blue
                target_pos = self._blue.pos + self._blue.normal.scale(16.0)
        elif target_id == "portal:orange":
            if self._orange.active and self._blue.active and self._orange.zone_id == self._player.zone_id:
                target_portal = self._orange
                target_pos = self._orange.pos + self._orange.normal.scale(16.0)
        else:
            ent = self._entities.get(target_id)
            if ent is not None and ent.zone_id == self._player.zone_id:
                target_pos = ent.pos

        if target_pos is None:
            return

        dx = target_pos.x - self._player.pos.x
        dy = target_pos.y - self._player.pos.y
        dist = math.hypot(dx, dy)
        max_step = PLAYER_SPEED * self.dt

        if dist > 1e-6:
            self._player.yaw = math.degrees(math.atan2(dy, dx))
            step_dist = min(dist, max_step)
            nx = dx / dist
            ny = dy / dist
            clamped = self._resolve_horizontal_motion(nx * step_dist, ny * step_dist)
            self._player.vel = Vec3(
                (clamped.x - self._player.pos.x) / self.dt,
                (clamped.y - self._player.pos.y) / self.dt,
                self._player.vel.z,
            )
            self._player.pos = clamped

        # Check portal traversal when moving into an active linked portal
        if target_portal is not None and self._portal_cooldown == 0:
            entry_point = target_portal.pos + target_portal.normal.scale(16.0)
            if self._player.pos.distance_xy(entry_point) <= PORTAL_ENTER_RADIUS:
                dest = (
                    self._orange
                    if target_portal.color is PortalColor.BLUE
                    else self._blue
                )
                dest_zone = self._zone(dest.zone_id)
                exit_pos = dest_zone.clamp_xy(
                    Vec3(
                        dest.pos.x + dest.normal.x * PORTAL_EXIT_OFFSET,
                        dest.pos.y + dest.normal.y * PORTAL_EXIT_OFFSET,
                        dest_zone.floor_z,
                    )
                )
                old_zone = self._player.zone_id
                self._player.pos = exit_pos
                self._player.zone_id = dest.zone_id
                self._player.yaw = math.degrees(
                    math.atan2(dest.normal.y, dest.normal.x)
                )
                self._portal_cooldown = 2
                if dest.zone_id != old_zone:
                    self._record_milestone("portal_traversed", milestones)

        # Record cube contact milestone when walking close to a loose cube
        cube = self._first_entity(EntityKind.CUBE)
        if (
            cube is not None
            and not cube.held
            and cube.zone_id == self._player.zone_id
            and self._player.pos.distance_xy(cube.pos) <= INTERACT_RADIUS
        ):
            self._record_milestone("cube_contact", milestones)

    def _exec_move_vec(self, cmd: MoveVector, milestones: list[str]) -> None:
        """Execute full human-equivalent body-frame control (Phase 2 / Milestone 7)."""
        fwd = max(-1.0, min(1.0, float(cmd.forward)))
        stf = max(-1.0, min(1.0, float(cmd.strafe)))
        yd = max(-1.0, min(1.0, float(cmd.yaw_delta)))
        pd = max(-1.0, min(1.0, float(cmd.pitch_delta)))

        # Normalize (forward, strafe) to unit disk so diagonal motion cannot exceed PLAYER_SPEED
        mag = math.hypot(fwd, stf)
        if mag > 1.0:
            fwd /= mag
            stf /= mag

        self._player.ducking = bool(cmd.crouch)
        speed_scale = 0.55 if self._player.ducking else 1.0

        # Update heading (yaw & pitch) in degrees
        new_yaw = (self._player.yaw + yd * MAX_YAW_RATE * self.dt + 180.0) % 360.0 - 180.0
        new_pitch = max(-85.0, min(85.0, self._player.pitch + pd * MAX_PITCH_RATE * self.dt))
        self._player.yaw = new_yaw
        self._player.pitch = new_pitch

        # Jump impulse when grounded
        if cmd.jump and self._player.on_ground:
            self._player.on_ground = False
            self._player.vel = Vec3(self._player.vel.x, self._player.vel.y, JUMP_SPEED)

        rad = math.radians(new_yaw)
        cos_y = math.cos(rad)
        sin_y = math.sin(rad)

        # Body-frame forward=(cos, sin), left=(-sin, cos)
        wx = fwd * cos_y - stf * sin_y
        wy = fwd * sin_y + stf * cos_y
        step_dist = PLAYER_SPEED * speed_scale * self.dt
        clamped = self._resolve_horizontal_motion(wx * step_dist, wy * step_dist)
        self._player.vel = Vec3(
            (clamped.x - self._player.pos.x) / self.dt,
            (clamped.y - self._player.pos.y) / self.dt,
            self._player.vel.z,
        )
        self._player.pos = clamped

        # Optional human-equivalent discrete interactions (use, fire_blue, fire_orange)
        if cmd.use:
            if self._player.held_entity_id is not None:
                self._exec_drop()
            else:
                acted = False
                for e in self._entities.values():
                    if (
                        e.kind is EntityKind.CUBE
                        and e.zone_id == self._player.zone_id
                        and self._player.pos.distance_xy(e.pos) <= INTERACT_RADIUS
                    ):
                        self._exec_grab(e.id, milestones)
                        acted = True
                        break
                if not acted:
                    for e in self._entities.values():
                        if (
                            e.kind is EntityKind.PEDESTAL_BUTTON
                            and e.zone_id == self._player.zone_id
                            and self._player.pos.distance_xy(e.pos) <= INTERACT_RADIUS
                        ):
                            self._exec_press(e.id, milestones)
                            break
        if cmd.fire_blue:
            self._fire_portal_along_aim(PortalColor.BLUE, milestones)
        if cmd.fire_orange:
            self._fire_portal_along_aim(PortalColor.ORANGE, milestones)

        # Continuous physics portal traversal: entering ANY active linked portal in current zone
        if self._portal_cooldown == 0 and self._blue.active and self._orange.active:
            for portal, dest in ((self._blue, self._orange), (self._orange, self._blue)):
                if portal.zone_id != self._player.zone_id:
                    continue
                entry_point = portal.pos + portal.normal.scale(16.0)
                if self._player.pos.distance_xy(entry_point) <= PORTAL_ENTER_RADIUS:
                    dest_zone = self._zone(dest.zone_id)
                    exit_pos = dest_zone.clamp_xy(
                        Vec3(
                            dest.pos.x + dest.normal.x * PORTAL_EXIT_OFFSET,
                            dest.pos.y + dest.normal.y * PORTAL_EXIT_OFFSET,
                            dest_zone.floor_z,
                        )
                    )
                    old_zone = self._player.zone_id
                    self._player.pos = exit_pos
                    self._player.zone_id = dest.zone_id
                    self._player.yaw = math.degrees(
                        math.atan2(dest.normal.y, dest.normal.x)
                    )
                    self._portal_cooldown = 2
                    if dest.zone_id != old_zone:
                        self._record_milestone("portal_traversed", milestones)
                    break

        cube = self._first_entity(EntityKind.CUBE)
        if (
            cube is not None
            and not cube.held
            and cube.zone_id == self._player.zone_id
            and self._player.pos.distance_xy(cube.pos) <= INTERACT_RADIUS
        ):
            self._record_milestone("cube_contact", milestones)

    def _fire_portal_along_aim(self, color: PortalColor, milestones: list[str]) -> None:
        best_wall: PortalWall | None = None
        best_err = 22.0
        for w in self._wall_list:
            dx = w.pos.x - self._player.pos.x
            dy = w.pos.y - self._player.pos.y
            ang = math.degrees(math.atan2(dy, dx))
            err = abs((ang - self._player.yaw + 180.0) % 360.0 - 180.0)
            if err <= best_err:
                best_err = err
                best_wall = w
        if best_wall is not None:
            self._exec_shoot_portal(color, best_wall.id, milestones)

    def _obstacle_blocks_xy(self, x: float, y: float, z: float) -> bool:
        for obs in self._spec.obstacles:
            if z >= obs.max_z - 4.0:
                continue  # Airborne cleanly over the top of a jumpable hurdle
            if (
                obs.min_x - PLAYER_RADIUS < x < obs.max_x + PLAYER_RADIUS
                and obs.min_y - PLAYER_RADIUS < y < obs.max_y + PLAYER_RADIUS
            ):
                return True
        return False

    def _resolve_horizontal_motion(self, dx: float, dy: float) -> Vec3:
        cur = self._player.pos
        nx = cur.x + dx
        ny = cur.y + dy
        if self._player.on_ground:
            zone = self._zone(self._player.zone_id)
            cx = min(max(nx, zone.min_x), zone.max_x)
            cy = min(max(ny, zone.min_y), zone.max_y)
        else:
            cx = min(
                max(nx, self._spec.room_min.x + PLAYER_RADIUS),
                self._spec.room_max.x - PLAYER_RADIUS,
            )
            cy = min(
                max(ny, self._spec.room_min.y + PLAYER_RADIUS),
                self._spec.room_max.y - PLAYER_RADIUS,
            )

        if self._spec.obstacles:
            if self._obstacle_blocks_xy(cx, cy, cur.z):
                # Slide independently along X or Y axis
                can_x = not self._obstacle_blocks_xy(cx, cur.y, cur.z)
                can_y = not self._obstacle_blocks_xy(cur.x, cy, cur.z)
                if can_x and not can_y:
                    cy = cur.y
                elif can_y and not can_x:
                    cx = cur.x
                elif can_x and can_y:
                    if abs(dx) >= abs(dy):
                        cy = cur.y
                    else:
                        cx = cur.x
                else:
                    cx, cy = cur.x, cur.y

        return Vec3(cx, cy, cur.z)

    def _step_vertical_physics(self, milestones: list[str]) -> float:
        """Integrate vertical jump/gravity arc, hurdle landings, and cross-gap landings."""
        vz = self._player.vel.z
        new_z = self._player.pos.z + vz * self.dt - 0.5 * GRAVITY * self.dt * self.dt
        new_vz = vz - GRAVITY * self.dt

        # Determine highest solid support surface under player XY
        support_zone: WalkableZone | None = None
        for z in self._spec.zones:
            if z.contains_xy(self._player.pos):
                support_zone = z
                break

        support_z = support_zone.floor_z if support_zone is not None else -128.0
        for obs in self._spec.obstacles:
            if (
                obs.min_x - PLAYER_RADIUS <= self._player.pos.x <= obs.max_x + PLAYER_RADIUS
                and obs.min_y - PLAYER_RADIUS <= self._player.pos.y <= obs.max_y + PLAYER_RADIUS
                and self._player.pos.z >= obs.max_z - 6.0
            ):
                support_z = max(support_z, obs.max_z)

        if new_z <= support_z:
            if support_zone is None and support_z < -64.0:
                # Fell into an open pit without clearing the gap
                self._player.pos = Vec3(self._player.pos.x, self._player.pos.y, support_z)
                self._player.vel = Vec3(0.0, 0.0, 0.0)
                self._player.health = 0.0
                self._player.on_ground = True
                self._done = True
                self._success = False
                milestones.append("player_died:pit")
                return -50.0
            self._player.pos = Vec3(self._player.pos.x, self._player.pos.y, support_z)
            self._player.vel = Vec3(self._player.vel.x, self._player.vel.y, 0.0)
            self._player.on_ground = True
            if support_zone is not None:
                self._player.zone_id = support_zone.zone_id
            return 0.0

        self._player.pos = Vec3(self._player.pos.x, self._player.pos.y, new_z)
        self._player.vel = Vec3(self._player.vel.x, self._player.vel.y, new_vz)
        self._player.on_ground = False
        return 0.0

    def _update_hazards(self, milestones: list[str]) -> float:
        self._player.water_level = 0
        p = self._player.pos
        penalty = 0.0
        for h in self._spec.hazards:
            if (
                h.min_pos.x <= p.x <= h.max_pos.x
                and h.min_pos.y <= p.y <= h.max_pos.y
                and p.z <= h.max_pos.z
            ):
                self._player.water_level = 2
                dmg = h.damage_per_sec * self.dt
                self._player.health = max(0.0, self._player.health - dmg)
                milestones.append(f"player_damaged:{h.id}")
                penalty -= dmg * 0.25
                if self._player.health <= 0.0:
                    milestones.append(f"player_died:{h.id}")
                    self._done = True
                    self._success = False
                    penalty -= 50.0
                    break
        return penalty

    def _cast_horizontal_ray(
        self, ox: float, oy: float, dx: float, dy: float, z_height: float, max_dist: float
    ) -> float:
        """Analytic raycast against room walls and solid ObstacleBoxes at height z_height."""
        best = max_dist
        rmin = self._spec.room_min
        rmax = self._spec.room_max
        if dx > 1e-6:
            best = min(best, (rmax.x - ox) / dx)
        elif dx < -1e-6:
            best = min(best, (rmin.x - ox) / dx)
        if dy > 1e-6:
            best = min(best, (rmax.y - oy) / dy)
        elif dy < -1e-6:
            best = min(best, (rmin.y - oy) / dy)

        for obs in self._spec.obstacles:
            if z_height < obs.min_z or z_height > obs.max_z:
                continue
            t = self._ray_aabb_2d(ox, oy, dx, dy, obs.min_x, obs.max_x, obs.min_y, obs.max_y)
            if t is not None and 0.0 <= t < best:
                best = t
        return max(0.0, best)

    @staticmethod
    def _ray_aabb_2d(
        ox: float,
        oy: float,
        dx: float,
        dy: float,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ) -> float | None:
        tmin = 0.0
        tmax = 1e9
        if abs(dx) < 1e-9:
            if ox < min_x or ox > max_x:
                return None
        else:
            tx1 = (min_x - ox) / dx
            tx2 = (max_x - ox) / dx
            if tx1 > tx2:
                tx1, tx2 = tx2, tx1
            tmin = max(tmin, tx1)
            tmax = min(tmax, tx2)
            if tmin > tmax:
                return None

        if abs(dy) < 1e-9:
            if oy < min_y or oy > max_y:
                return None
        else:
            ty1 = (min_y - oy) / dy
            ty2 = (max_y - oy) / dy
            if ty1 > ty2:
                ty1, ty2 = ty2, ty1
            tmin = max(tmin, ty1)
            tmax = min(tmax, ty2)
            if tmin > tmax:
                return None
        return tmin

    def _compute_local_sensors(self) -> LocalSensors:
        p = self._player.pos
        yaw_deg = self._player.yaw
        chest_z = p.z + 36.0
        knee_z = p.z + 18.0
        high_z = p.z + 52.0

        # 1. 8 horizontal obstacle rays at chest height
        wall_rays: list[float] = []
        for rel_deg in (0.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0, -45.0):
            ang = math.radians(yaw_deg + rel_deg)
            d = self._cast_horizontal_ray(
                p.x, p.y, math.cos(ang), math.sin(ang), chest_z, SENSOR_RAY_RANGE
            )
            wall_rays.append(min(1.0, d / SENSOR_RAY_RANGE))

        # 2. 3 downward floor-drop probes at 48u ahead ([0, +35, -35] deg)
        floor_probes: list[float] = []
        for rel_deg in (0.0, 35.0, -35.0):
            ang = math.radians(yaw_deg + rel_deg)
            px = p.x + math.cos(ang) * 48.0
            py = p.y + math.sin(ang) * 48.0
            probe_pt = Vec3(px, py, p.z)
            supported = any(z.contains_xy(probe_pt) for z in self._spec.zones)
            if supported:
                floor_probes.append(0.28)
            else:
                floor_probes.append(1.0)

        # 3. Forward knee vs high step-clearance probe (72u ahead)
        fwd_rad = math.radians(yaw_deg)
        fdx = math.cos(fwd_rad)
        fdy = math.sin(fwd_rad)
        d_knee = self._cast_horizontal_ray(p.x, p.y, fdx, fdy, knee_z, 72.0)
        d_high = self._cast_horizontal_ray(p.x, p.y, fdx, fdy, high_z, 72.0)
        if d_knee < 68.0 and d_high >= 68.0:
            step_clearance = 1.0  # Jumpable hurdle ahead
        elif d_knee < 68.0 and d_high < 68.0:
            step_clearance = -1.0  # Tall unjumpable wall ahead
        else:
            step_clearance = 0.0  # Clear ahead

        # 4. Forward hazard proximity probe (0.0 clear -> 1.0 at acid edge)
        hazard_ahead = 0.0
        if self._spec.hazards:
            best_hd = SENSOR_HAZARD_RANGE
            for h in self._spec.hazards:
                if (
                    h.min_pos.x <= p.x <= h.max_pos.x
                    and h.min_pos.y <= p.y <= h.max_pos.y
                ):
                    best_hd = 0.0
                    break
                t = self._ray_aabb_2d(
                    p.x, p.y, fdx, fdy, h.min_pos.x, h.max_pos.x, h.min_pos.y, h.max_pos.y
                )
                if t is not None and 0.0 <= t < best_hd:
                    best_hd = t
            hazard_ahead = max(0.0, 1.0 - best_hd / SENSOR_HAZARD_RANGE)

        return LocalSensors(
            wall_rays=wall_rays,
            floor_probes=floor_probes,
            step_clearance=step_clearance,
            hazard_ahead=hazard_ahead,
        )

    def _exec_grab(self, entity_id: str, milestones: list[str]) -> None:
        if self._player.held_entity_id is not None:
            return
        ent = self._entities.get(entity_id)
        if ent is None or ent.kind is not EntityKind.CUBE:
            return
        if ent.zone_id != self._player.zone_id:
            return
        if self._player.pos.distance_xy(ent.pos) > INTERACT_RADIUS:
            return
        self._player.held_entity_id = ent.id
        ent.held = True
        ent.active = False
        self._record_milestone("cube_contact", milestones)
        self._record_milestone("cube_grabbed", milestones)

    def _exec_drop(self) -> None:
        held_id = self._player.held_entity_id
        if held_id is None:
            return
        ent = self._entities.get(held_id)
        self._player.held_entity_id = None
        if ent is None:
            return
        ent.held = False
        zone = self._zone(self._player.zone_id)
        # Drop gently onto the floor at player's feet / slightly ahead
        ent.zone_id = self._player.zone_id
        ent.pos = zone.clamp_xy(
            Vec3(self._player.pos.x, self._player.pos.y, zone.floor_z)
        )
        ent.vel = Vec3(0.0, 0.0, 0.0)

    def _exec_press(self, entity_id: str, milestones: list[str]) -> None:
        ent = self._entities.get(entity_id)
        if ent is None or ent.kind is not EntityKind.PEDESTAL_BUTTON:
            return
        if ent.zone_id != self._player.zone_id:
            return
        if self._player.pos.distance_xy(ent.pos) > INTERACT_RADIUS:
            return
        spawn = next(s for s in self._spec.entities if s.id == ent.id)
        ent.active = True
        ent.timer = spawn.pedestal_duration
        self._record_milestone("button_pressed", milestones)

    def _exec_shoot_portal(
        self, color: PortalColor, wall_id: str, milestones: list[str]
    ) -> None:
        wall = self._walls.get(wall_id)
        if wall is None:
            return
        # Ensure we don't overlap blue & orange on the exact same wall
        other = self._orange if color is PortalColor.BLUE else self._blue
        if other.active and other.wall_id == wall.id:
            other.active = False
            other.wall_id = None
            other.zone_id = -1

        portal = self._blue if color is PortalColor.BLUE else self._orange
        portal.active = True
        portal.wall_id = wall.id
        portal.pos = wall.pos
        portal.normal = wall.normal
        portal.zone_id = wall.zone_id

        self._record_milestone(f"portal_{color.value}_placed", milestones)
        if (
            self._blue.active
            and self._orange.active
            and self._blue.zone_id != self._orange.zone_id
        ):
            self._record_milestone("portals_bridged", milestones)

    # --- World physics / trigger updates ---

    def _check_fizzler_crossing(
        self, prev_pos: Vec3, prev_zone: int, milestones: list[str]
    ) -> None:
        held_id = self._player.held_entity_id
        if held_id is None or self._player.zone_id != prev_zone:
            return
        cur_pos = self._player.pos
        for fz in self._spec.fizzlers:
            if fz.zone_id != prev_zone:
                continue
            crossed = False
            if fz.axis == "x":
                span_mid = 0.5 * (prev_pos.y + cur_pos.y)
                if fz.min_span <= span_mid <= fz.max_span:
                    if (prev_pos.x - fz.coord) * (cur_pos.x - fz.coord) <= 0.0 and (
                        prev_pos.x != cur_pos.x
                    ):
                        crossed = True
            elif fz.axis == "y":
                span_mid = 0.5 * (prev_pos.x + cur_pos.x)
                if fz.min_span <= span_mid <= fz.max_span:
                    if (prev_pos.y - fz.coord) * (cur_pos.y - fz.coord) <= 0.0 and (
                        prev_pos.y != cur_pos.y
                    ):
                        crossed = True
            if crossed:
                cube = self._entities.get(held_id)
                spawn = self._spawn_by_id.get(held_id)
                if cube is not None and spawn is not None:
                    cube.held = False
                    cube.pos = spawn.pos
                    cube.zone_id = spawn.zone_id
                    cube.active = False
                self._player.held_entity_id = None
                # Fizzler always emits event (even on repeated attempts) so Layer 4 sees it
                milestones.append(f"cube_fizzled:{held_id}")
                return

    def _update_held_cube(self) -> None:
        held_id = self._player.held_entity_id
        if held_id is None:
            return
        cube = self._entities.get(held_id)
        if cube is None:
            self._player.held_entity_id = None
            return
        rad = math.radians(self._player.yaw)
        cube.zone_id = self._player.zone_id
        cube.pos = Vec3(
            self._player.pos.x + math.cos(rad) * CUBE_HOLD_OFFSET,
            self._player.pos.y + math.sin(rad) * CUBE_HOLD_OFFSET,
            self._player.pos.z + CUBE_HOLD_HEIGHT,
        )

    def _update_world_logic(self, milestones: list[str]) -> None:
        # 1. Tick pedestal timers
        pedestal_active = False
        for ent in self._entities.values():
            if ent.kind is EntityKind.PEDESTAL_BUTTON:
                if ent.timer > 0.0:
                    ent.timer = max(0.0, ent.timer - self.dt)
                    ent.active = ent.timer > 0.0
                else:
                    ent.active = False
                if ent.active:
                    pedestal_active = True

        # 2. Check weighted floor buttons against loose cubes
        floor_buttons: list[EntityState] = []
        pedestals: list[EntityState] = []
        cubes = [e for e in self._entities.values() if e.kind is EntityKind.CUBE]
        for c in cubes:
            c.active = False

        for ent in self._entities.values():
            if ent.kind is EntityKind.PEDESTAL_BUTTON:
                pedestals.append(ent)
            elif ent.kind is EntityKind.FLOOR_BUTTON:
                floor_buttons.append(ent)
                pressed_by_cube = False
                for c in cubes:
                    if (
                        not c.held
                        and c.zone_id == ent.zone_id
                        and c.pos.distance_xy(ent.pos) <= BUTTON_TRIGGER_RADIUS
                    ):
                        pressed_by_cube = True
                        c.active = True
                ent.active = pressed_by_cube
                if ent.active:
                    self._record_milestone("button_pressed", milestones)

        # 3. Update exit door open state
        exit_ent = self._first_entity(EntityKind.EXIT_DOOR)
        if exit_ent is not None:
            if self._spec.exit_starts_open:
                exit_ent.active = True
            elif self._spec.require_all_buttons:
                all_floor = all(b.active for b in floor_buttons) if floor_buttons else True
                all_ped = all(p.active for p in pedestals) if pedestals else True
                exit_ent.active = (bool(floor_buttons) or bool(pedestals)) and all_floor and all_ped
            else:
                any_floor = any(b.active for b in floor_buttons)
                exit_ent.active = pedestal_active or any_floor

    def _record_milestone(self, name: str, step_milestones: list[str]) -> None:
        if name not in self._achieved_milestones:
            self._achieved_milestones.add(name)
            step_milestones.append(name)

    def _score_milestones(self, step_milestones: list[str]) -> float:
        weights = {
            "cube_contact": 10.0,
            "cube_grabbed": 15.0,
            "button_pressed": 25.0,
            "portal_blue_placed": 8.0,
            "portal_orange_placed": 8.0,
            "portals_bridged": 15.0,
            "portal_traversed": 30.0,
        }
        return sum(weights.get(m, 0.0) for m in step_milestones)

    def _first_entity(self, kind: EntityKind) -> EntityState | None:
        for ent in self._entities.values():
            if ent.kind is kind:
                return ent
        return None

    def _make_snapshot(self) -> BlackboardSnapshot:
        if self.reuse_snapshot and self._shared_snapshot is not None:
            snap = self._shared_snapshot
            snap.tick = self._tick
            snap.sensors = self._sensors
            snap.done = self._done
            snap.success = self._success
            return snap
        return BlackboardSnapshot(
            tick=self._tick,
            dt=self.dt,
            chamber_id=self._spec.chamber_id,
            player=PlayerState(
                pos=self._player.pos,
                vel=self._player.vel,
                yaw=self._player.yaw,
                pitch=self._player.pitch,
                zone_id=self._player.zone_id,
                held_entity_id=self._player.held_entity_id,
                health=self._player.health,
                on_ground=self._player.on_ground,
                ducking=self._player.ducking,
                water_level=self._player.water_level,
            ),
            entities=[
                EntityState(
                    id=e.id,
                    kind=e.kind,
                    pos=e.pos,
                    vel=e.vel,
                    zone_id=e.zone_id,
                    active=e.active,
                    held=e.held,
                    timer=e.timer,
                )
                for e in self._entities.values()
            ],
            portal_walls=list(self._walls.values()),
            blue_portal=ActivePortal(
                color=PortalColor.BLUE,
                active=self._blue.active,
                wall_id=self._blue.wall_id,
                pos=self._blue.pos,
                normal=self._blue.normal,
                zone_id=self._blue.zone_id,
            ),
            orange_portal=ActivePortal(
                color=PortalColor.ORANGE,
                active=self._orange.active,
                wall_id=self._orange.wall_id,
                pos=self._orange.pos,
                normal=self._orange.normal,
                zone_id=self._orange.zone_id,
            ),
            sensors=LocalSensors(
                wall_rays=list(self._sensors.wall_rays),
                floor_probes=list(self._sensors.floor_probes),
                step_clearance=self._sensors.step_clearance,
                hazard_ahead=self._sensors.hazard_ahead,
            ),
            hazards=list(self._hazards),
            done=self._done,
            success=self._success,
        )
