"""Layer-4 strategic planner protocol, directive schema, and symbolic digest."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from portal_ml.bt.predicates import ENTITY_ROLES, ROLE_TO_KIND
from portal_ml.planner.memory import EpisodicMemory
from portal_ml.schema.blackboard import BlackboardSnapshot, EntityKind


class TriggerEvent(str, Enum):
    """Discrete world events that wake the event-driven Layer-4 planner."""

    EPISODE_START = "episode_start"
    SUBGOAL_COMPLETED = "subgoal_completed"
    CUBE_FIZZLED = "cube_fizzled"
    EXIT_OPENED = "exit_opened"
    STALL_DETECTED = "stall_detected"


@dataclass(slots=True)
class StrategicDirective:
    """Structured Layer-4 output written into Layer-2 WorkingMemory.

    ``role_overrides`` maps symbolic BT roles (``cube``, ``floor_button``,
    ``pedestal_button``, ``exit``) to either:
    - a live entity ID of the matching ``EntityKind`` (focuses Layer 2 on it), or
    - ``None`` (explicitly masks that secondary role during the current subgoal).
    Omitted keys leave Layer-2 resolution on its default ``first_of_kind`` rule.
    """

    subgoal: str = "default"
    role_overrides: dict[str, str | None] = field(default_factory=dict)
    memory_note: str | None = None
    min_dwell_steps: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "subgoal": self.subgoal,
            "role_overrides": dict(self.role_overrides),
            "memory_note": self.memory_note,
            "min_dwell_steps": self.min_dwell_steps,
        }

    def validate_against_snapshot(
        self, snapshot: BlackboardSnapshot
    ) -> tuple[StrategicDirective, list[str]]:
        """Sanitize directive against live Blackboard entities to prevent hallucinations."""
        warnings: list[str] = []
        clean_overrides: dict[str, str | None] = {}

        for role, ent_id in self.role_overrides.items():
            if role not in ROLE_TO_KIND:
                warnings.append(f"unknown_role:{role}")
                continue
            if ent_id is None:
                clean_overrides[role] = None
                continue
            if not isinstance(ent_id, str) or not ent_id:
                warnings.append(f"invalid_id:{role}={ent_id!r}")
                continue
            ent = snapshot.entity_by_id(ent_id)
            expected_kind = ROLE_TO_KIND[role]
            if ent is None:
                warnings.append(f"hallucinated_entity:{role}={ent_id}")
                continue
            if ent.kind is not expected_kind:
                warnings.append(
                    f"kind_mismatch:{role}={ent_id}(got={ent.kind.value},want={expected_kind.value})"
                )
                continue
            clean_overrides[role] = ent.id

        dwell = max(1, min(60, int(self.min_dwell_steps)))
        clean_subgoal = " ".join(str(self.subgoal).split())[:100] or "default"
        clean_note: str | None = None
        if self.memory_note is not None:
            stripped = " ".join(str(self.memory_note).split())[:120]
            clean_note = stripped if stripped else None

        return (
            StrategicDirective(
                subgoal=clean_subgoal,
                role_overrides=clean_overrides,
                memory_note=clean_note,
                min_dwell_steps=dwell,
            ),
            warnings,
        )


def compile_semantic_action(
    snapshot: BlackboardSnapshot,
    *,
    action: str,
    cube_id: str | None = None,
    button_id: str | None = None,
    pedestal_id: str | None = None,
    exit_id: str | None = None,
) -> StrategicDirective:
    """Translate a high-level Layer-4 semantic subgoal into Layer-2 role_overrides.

    The strategic planner chooses *which* stage (`place_cube_on_button`,
    `press_pedestal`, or `go_to_exit`) and *which* concrete entity IDs to target.
    This compiler maps that semantic choice onto `WorkingMemory.role_overrides`
    so the frozen Layer-2 Behavior Tree executes the chosen stage cleanly without
    requiring small LLMs to memorize internal BT selector branch ordering.
    """
    exits = snapshot.entities_of_kind(EntityKind.EXIT_DOOR)
    resolved_exit = exit_id if exit_id is not None else (exits[0].id if exits else None)
    active_cube = next(
        (c.id for c in snapshot.entities_of_kind(EntityKind.CUBE) if c.active and not c.held),
        None,
    )

    if action == "place_cube_on_button":
        return StrategicDirective(
            subgoal=f"place {cube_id} on {button_id}",
            role_overrides={
                "cube": cube_id,
                "floor_button": button_id,
                "pedestal_button": None,
                "exit": resolved_exit,
            },
            min_dwell_steps=12,
        )

    if action == "press_pedestal":
        return StrategicDirective(
            subgoal=f"press {pedestal_id}",
            role_overrides={
                "cube": active_cube,
                "floor_button": None,
                "pedestal_button": pedestal_id,
                "exit": resolved_exit,
            },
            min_dwell_steps=8,
        )

    if action == "go_to_exit":
        return StrategicDirective(
            subgoal=f"exit via {resolved_exit}",
            role_overrides={
                "cube": active_cube,
                "floor_button": None,
                "pedestal_button": None,
                "exit": resolved_exit,
            },
            min_dwell_steps=8,
        )

    raise ValueError(f"Unknown Layer-4 semantic action: {action!r}")


def build_directive_json_schema(snapshot: BlackboardSnapshot) -> dict[str, Any]:
    """Build a dynamic compact JSON Schema restricting actions & IDs to live entities.

    Compatible with OpenAI ``response_format: {"type": "json_schema", ...}``
    (supported natively by LM Studio, llama.cpp server, and vLLM via GBNF).
    """
    cube_ids: list[str | None] = [
        e.id for e in snapshot.entities if e.kind is EntityKind.CUBE
    ]
    btn_ids: list[str | None] = [
        e.id for e in snapshot.entities if e.kind is EntityKind.FLOOR_BUTTON
    ]
    ped_ids: list[str | None] = [
        e.id for e in snapshot.entities if e.kind is EntityKind.PEDESTAL_BUTTON
    ]
    exit_ids: list[str | None] = [
        e.id for e in snapshot.entities if e.kind is EntityKind.EXIT_DOOR
    ]

    valid_actions: list[str] = []
    any_exit_open = any(e.active for e in snapshot.entities if e.kind is EntityKind.EXIT_DOOR)
    any_floor_unpressed = any(
        not e.active for e in snapshot.entities if e.kind is EntityKind.FLOOR_BUTTON
    )
    any_ped_unpressed = any(
        not e.active for e in snapshot.entities if e.kind is EntityKind.PEDESTAL_BUTTON
    )

    if any_exit_open:
        valid_actions.append("go_to_exit")
    if cube_ids and btn_ids and any_floor_unpressed:
        valid_actions.append("place_cube_on_button")
    if ped_ids and any_ped_unpressed:
        valid_actions.append("press_pedestal")
    if not valid_actions:
        valid_actions.append("go_to_exit")

    return {
        "name": "portal_strategic_directive",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "cube_id", "button_id", "pedestal_id", "exit_id"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": valid_actions,
                },
                "cube_id": {
                    "enum": [*cube_ids, None],
                },
                "button_id": {
                    "enum": [*btn_ids, None],
                },
                "pedestal_id": {
                    "enum": [*ped_ids, None],
                },
                "exit_id": {
                    "enum": exit_ids if exit_ids else [None],
                },
            },
        },
    }


def build_symbolic_digest(
    snapshot: BlackboardSnapshot,
    memory: EpisodicMemory,
    trigger: TriggerEvent,
    current_directive: StrategicDirective | None = None,
) -> str:
    """Format a compact (< 300 token) deterministic symbolic summary for Layer 4."""
    p = snapshot.player
    fizzled_cubes = memory.entities_with_event("cube_fizzled")
    stalled_cubes = memory.entities_with_event("cube_stalled")
    lines = [
        f"TRIGGER: {trigger.value} | tick={snapshot.tick} | chamber={snapshot.chamber_id}",
        (
            f"PLAYER: zone={p.zone_id} pos=({p.pos.x:.0f},{p.pos.y:.0f}) "
            f"held={p.held_entity_id or 'none'}"
        ),
    ]

    for label, kind in (
        ("CUBES", EntityKind.CUBE),
        ("FLOOR_BUTTONS", EntityKind.FLOOR_BUTTON),
        ("PEDESTAL_BUTTONS", EntityKind.PEDESTAL_BUTTON),
        ("EXITS", EntityKind.EXIT_DOOR),
    ):
        ents = snapshot.entities_of_kind(kind)
        if not ents:
            lines.append(f"{label}: none")
            continue
        parts = []
        for e in ents:
            if e.kind is EntityKind.EXIT_DOOR:
                state_str = "OPEN" if e.active else "CLOSED"
            elif e.kind in (EntityKind.FLOOR_BUTTON, EntityKind.PEDESTAL_BUTTON):
                state_str = "PRESSED" if e.active else "UNPRESSED"
            else:
                state_str = "ON_BUTTON" if e.active else "LOOSE"
            flags = [state_str, f"z={e.zone_id}"]
            if e.held:
                flags.append("HELD")
            if e.timer > 0.0:
                flags.append(f"t={e.timer:.1f}s")
            if e.kind is EntityKind.CUBE:
                if e.id in fizzled_cubes:
                    flags.append("FIZZLED_DO_NOT_USE")
                elif e.id in stalled_cubes:
                    flags.append("STALLED_AVOID")
            parts.append(f"{e.id}[{','.join(flags)}]")
        lines.append(f"{label}: " + "; ".join(parts))

    wall_zones = sorted({w.zone_id for w in snapshot.portal_walls})
    b = snapshot.blue_portal
    o = snapshot.orange_portal
    lines.append(
        f"PORTALS: blue={'z'+str(b.zone_id) if b.active else 'off'} "
        f"orange={'z'+str(o.zone_id) if o.active else 'off'} "
        f"wall_zones={wall_zones}"
    )

    if current_directive is not None:
        lines.append(f"PREV_SUBGOAL: {current_directive.subgoal}")

    lines.append("EPISODIC_MEMORY:")
    lines.append(memory.format_for_prompt())
    return "\n".join(lines)


class StrategicPlanner(Protocol):
    """Interface implemented by all Layer-4 planners (null, rule, local LLM)."""

    name: str

    def plan(
        self,
        snapshot: BlackboardSnapshot,
        memory: EpisodicMemory,
        trigger: TriggerEvent,
        current_directive: StrategicDirective | None = None,
    ) -> StrategicDirective: ...
