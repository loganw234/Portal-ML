"""Four-layer event-driven controller uniting Layer 4, Layer 2, and Layer 3."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from portal_ml.bt.evaluator import BehaviorTreeEvaluator, TickResult
from portal_ml.motor.hybrid import HybridMotorController
from portal_ml.motor.primitives import MotorCommand
from portal_ml.planner.memory import EpisodicMemory
from portal_ml.planner.planners import NullStrategicPlanner, OpenAICompatibleLLMPlanner
from portal_ml.planner.protocol import (
    StrategicDirective,
    StrategicPlanner,
    TriggerEvent,
)
from portal_ml.schema.blackboard import BlackboardSnapshot, EntityKind, Vec3
from portal_ml.sim.chambers import ChamberSpec
from portal_ml.sim.mock_env import MockPortalEnv, PortalEnv


@dataclass(slots=True)
class EpisodeMetrics:
    """Structured telemetry for controlled 3-layer vs 4-layer ablations."""

    chamber_id: int
    chamber_name: str
    success: bool
    steps: int
    reward: float
    milestones: list[str]
    planner_name: str
    planner_calls: int
    planner_triggers: dict[str, int]
    total_planner_ms: float
    fallback_count: int
    validation_warnings: int
    role_switches: int
    memory_entries: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chamber_id": self.chamber_id,
            "chamber_name": self.chamber_name,
            "success": self.success,
            "steps": self.steps,
            "reward": round(self.reward, 3),
            "milestones": list(self.milestones),
            "planner_name": self.planner_name,
            "planner_calls": self.planner_calls,
            "planner_triggers": dict(self.planner_triggers),
            "total_planner_ms": round(self.total_planner_ms, 3),
            "fallback_count": self.fallback_count,
            "validation_warnings": self.validation_warnings,
            "role_switches": self.role_switches,
            "memory_entries": list(self.memory_entries),
        }


class FourLayerAgent:
    """Event-driven Layer-4 coordinator wrapping a frozen Layer-2 BT + Layer-3 motor.

    When ``planner`` is ``NullStrategicPlanner``, Layer 4 stays completely passive
    (zero overrides, zero planner calls) so 3-layer execution is bit-for-bit
    identical to running ``BehaviorTreeEvaluator`` directly.
    """

    def __init__(
        self,
        evaluator: BehaviorTreeEvaluator,
        planner: StrategicPlanner | None = None,
        *,
        hybrid_motor: HybridMotorController | None = None,
        max_memory_entries: int = 8,
        stall_window: int = 45,
        progress_distance: float = 32.0,
    ) -> None:
        self.evaluator = evaluator
        self.planner: StrategicPlanner = (
            planner if planner is not None else NullStrategicPlanner()
        )
        self.hybrid_motor = hybrid_motor
        self.memory = EpisodicMemory(max_entries=max_memory_entries)
        self.stall_window = stall_window
        self.progress_distance = progress_distance

        # Per-episode tracking state
        self._current_directive: StrategicDirective | None = None
        self._prev_active_map: dict[str, bool] = {}
        self._prev_zone: int = -1
        self._prev_held: str | None = None
        self._progress_anchor: Vec3 | None = None
        self._last_progress_tick: int = 0
        self._last_plan_tick: int = -999
        self._initialized: bool = False

        # Episode telemetry counters
        self.planner_calls: int = 0
        self.planner_triggers: dict[str, int] = {}
        self.total_planner_ms: float = 0.0
        self.fallback_count: int = 0
        self.validation_warnings: int = 0
        self.role_switches: int = 0

    @property
    def current_directive(self) -> StrategicDirective | None:
        return self._current_directive

    @property
    def is_passthrough(self) -> bool:
        return isinstance(self.planner, NullStrategicPlanner)

    def reset(self, initial_snapshot: BlackboardSnapshot | None = None) -> None:
        self.evaluator.reset()
        self.memory.clear()
        if self.hybrid_motor is not None:
            self.hybrid_motor.reset_stats()
        self._current_directive = None
        self._prev_active_map.clear()
        self._prev_zone = -1
        self._prev_held = None
        self._progress_anchor = None
        self._last_progress_tick = 0
        self._last_plan_tick = -999
        self._initialized = False

        self.planner_calls = 0
        self.planner_triggers = {}
        self.total_planner_ms = 0.0
        self.fallback_count = 0
        self.validation_warnings = 0
        self.role_switches = 0

        if initial_snapshot is not None:
            self._init_episode_state(initial_snapshot)

    def act(
        self,
        snapshot: BlackboardSnapshot,
        last_milestones: list[str] | None = None,
    ) -> tuple[MotorCommand, TickResult]:
        """Advance Layer 4 (only on trigger events), tick Layer 2, and lower via Layer 3."""
        if not self._initialized:
            self._init_episode_state(snapshot)
        elif not self.is_passthrough:
            trigger = self._detect_trigger(snapshot, last_milestones or [])
            if trigger is not None:
                self._invoke_planner(snapshot, trigger)
            self._update_progress_tracking(snapshot)

        tick_res = self.evaluator.tick(snapshot)
        cmd = tick_res.command
        if self.hybrid_motor is not None:
            cmd = self.hybrid_motor.lower(snapshot, cmd)
        return cmd, tick_res

    def run_episode(
        self,
        env: PortalEnv,
        chamber_id: int = 1,
        *,
        spec: ChamberSpec | None = None,
        max_steps: int | None = None,
    ) -> EpisodeMetrics:
        """Run one complete episode and return structured ablation metrics."""
        if spec is not None and isinstance(env, MockPortalEnv):
            snap = env.reset(chamber_id=spec.chamber_id, spec=spec)
            chamber_name = spec.name
            step_cap = max_steps if max_steps is not None else spec.max_steps
        else:
            snap = env.reset(chamber_id)
            chamber_name = (
                env.spec.name if isinstance(env, MockPortalEnv) else f"Chamber {chamber_id}"
            )
            step_cap = (
                max_steps
                if max_steps is not None
                else (env.spec.max_steps if isinstance(env, MockPortalEnv) else 260)
            )

        self.reset(snap)
        steps = 0
        total_reward = 0.0
        all_milestones: list[str] = []
        last_milestones: list[str] = []

        while not snap.done and steps < step_cap:
            cmd, _tick = self.act(snap, last_milestones)
            step_res = env.step(cmd)
            snap = step_res.snapshot
            total_reward += step_res.reward
            last_milestones = list(step_res.milestones)
            if last_milestones:
                all_milestones.extend(last_milestones)
            steps += 1

        return EpisodeMetrics(
            chamber_id=snap.chamber_id,
            chamber_name=chamber_name,
            success=snap.success,
            steps=steps,
            reward=total_reward,
            milestones=all_milestones,
            planner_name=self.planner.name,
            planner_calls=self.planner_calls,
            planner_triggers=dict(self.planner_triggers),
            total_planner_ms=self.total_planner_ms,
            fallback_count=self.fallback_count,
            validation_warnings=self.validation_warnings,
            role_switches=self.role_switches,
            memory_entries=self.memory.to_list(),
        )

    # --- Event detection & Layer-4 invocation ---

    def _init_episode_state(self, snapshot: BlackboardSnapshot) -> None:
        self._initialized = True
        self._prev_active_map = {e.id: e.active for e in snapshot.entities}
        self._prev_zone = snapshot.player.zone_id
        self._prev_held = snapshot.player.held_entity_id
        self._progress_anchor = snapshot.player.pos
        self._last_progress_tick = snapshot.tick
        if not self.is_passthrough:
            self._invoke_planner(snapshot, TriggerEvent.EPISODE_START)

    def _detect_trigger(
        self, snapshot: BlackboardSnapshot, milestones: list[str]
    ) -> TriggerEvent | None:
        # 1. Urgent: cube vaporized by a fizzler (bypasses dwell hysteresis)
        fizzled_ids = [
            m.split(":", 1)[1]
            for m in milestones
            if m.startswith("cube_fizzled:") and ":" in m
        ]
        if fizzled_ids:
            for cid in fizzled_ids:
                self.memory.record(
                    snapshot.tick,
                    "cube_fizzled",
                    (cid,),
                    f"cube {cid} vaporized by fizzler; avoid reusing this cube",
                )
            return TriggerEvent.CUBE_FIZZLED

        # 2. Urgent: exit opened or button state transitioned
        exit_opened = False
        subgoal_changed = False
        for ent in snapshot.entities:
            prev_active = self._prev_active_map.get(ent.id, False)
            if ent.active == prev_active:
                continue
            if ent.kind is EntityKind.EXIT_DOOR and ent.active:
                exit_opened = True
                self.memory.record(
                    snapshot.tick,
                    "exit_opened",
                    (ent.id,),
                    f"exit door {ent.id} is now open",
                )
            elif ent.kind in (EntityKind.FLOOR_BUTTON, EntityKind.PEDESTAL_BUTTON):
                subgoal_changed = True
                ev_name = "button_activated" if ent.active else "button_deactivated"
                note = (
                    f"{ent.kind.value} {ent.id} activated"
                    if ent.active
                    else f"{ent.kind.value} {ent.id} deactivated"
                )
                self.memory.record(snapshot.tick, ev_name, (ent.id,), note)

        if exit_opened:
            return TriggerEvent.EXIT_OPENED
        if subgoal_changed:
            return TriggerEvent.SUBGOAL_COMPLETED

        # 3. Non-urgent: stall detection (respects commitment dwell hysteresis)
        dwell = (
            self._current_directive.min_dwell_steps
            if self._current_directive is not None
            else 10
        )
        if (
            snapshot.tick - self._last_progress_tick >= self.stall_window
            and snapshot.tick - self._last_plan_tick >= dwell
        ):
            focused_cube = self.evaluator.memory.role_overrides.get("cube")
            if isinstance(focused_cube, str):
                self.memory.record(
                    snapshot.tick,
                    "cube_stalled",
                    (focused_cube,),
                    f"no progress for {self.stall_window} ticks while targeting {focused_cube}",
                )
            else:
                self.memory.record(
                    snapshot.tick,
                    "stall_detected",
                    (),
                    f"no progress for {self.stall_window} ticks",
                )
            self._last_progress_tick = snapshot.tick
            return TriggerEvent.STALL_DETECTED

        return None

    def _update_progress_tracking(self, snapshot: BlackboardSnapshot) -> None:
        progressed = False
        cur_active = {e.id: e.active for e in snapshot.entities}
        if cur_active != self._prev_active_map:
            progressed = True
        if snapshot.player.zone_id != self._prev_zone:
            progressed = True
        if snapshot.player.held_entity_id != self._prev_held:
            progressed = True
        if (
            self._progress_anchor is None
            or snapshot.player.pos.distance_xy(self._progress_anchor)
            >= self.progress_distance
        ):
            progressed = True

        self._prev_active_map = cur_active
        self._prev_zone = snapshot.player.zone_id
        self._prev_held = snapshot.player.held_entity_id
        if progressed:
            self._progress_anchor = snapshot.player.pos
            self._last_progress_tick = snapshot.tick

    def _invoke_planner(
        self, snapshot: BlackboardSnapshot, trigger: TriggerEvent
    ) -> None:
        prev_fallbacks = (
            self.planner.fallbacks
            if isinstance(self.planner, OpenAICompatibleLLMPlanner)
            else 0
        )
        t0 = time.perf_counter()
        try:
            raw_directive = self.planner.plan(
                snapshot, self.memory, trigger, self._current_directive
            )
        except Exception:
            self.fallback_count += 1
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.planner_calls += 1
        self.total_planner_ms += elapsed_ms
        self.planner_triggers[trigger.value] = (
            self.planner_triggers.get(trigger.value, 0) + 1
        )
        self._last_plan_tick = snapshot.tick

        if isinstance(self.planner, OpenAICompatibleLLMPlanner):
            if self.planner.fallbacks > prev_fallbacks:
                self.fallback_count += self.planner.fallbacks - prev_fallbacks

        sanitized, warnings = raw_directive.validate_against_snapshot(snapshot)
        self.validation_warnings += len(warnings)

        if sanitized.memory_note:
            self.memory.record(
                snapshot.tick, "planner_note", (), sanitized.memory_note
            )

        prev_overrides = dict(self.evaluator.memory.role_overrides)
        self.evaluator.memory.role_overrides = dict(sanitized.role_overrides)
        if self.evaluator.memory.role_overrides != prev_overrides:
            self.role_switches += 1
            # Clear cached target so Layer 2 immediately rebinds on the next tick.
            self.evaluator.memory.target_role = None
            self.evaluator.memory.target_entity_id = None

        self._current_directive = sanitized
