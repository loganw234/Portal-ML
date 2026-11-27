"""Tick-based Behavior Tree evaluator producing Layer-3 MotorCommands."""

from __future__ import annotations

from dataclasses import dataclass

from portal_ml.bt.ast import (
    ActionNode,
    BTNode,
    ConditionNode,
    NodeStatus,
    SelectorNode,
    SequenceNode,
)
from portal_ml.bt.predicates import (
    INTERACT_RADIUS,
    evaluate_predicate,
    resolve_entity,
    resolve_wall,
)
from portal_ml.motor.primitives import (
    Drop,
    Grab,
    MotorCommand,
    MoveTo,
    NoOp,
    Press,
    ShootPortal,
)
from portal_ml.schema.blackboard import (
    BlackboardSnapshot,
    EntityKind,
    PortalColor,
    WorkingMemory,
)


@dataclass(slots=True)
class TickResult:
    status: NodeStatus
    command: MotorCommand
    nodes_visited: int
    target_role: str | None = None
    target_entity_id: str | None = None


class BehaviorTreeEvaluator:
    """Evaluates a Behavior Tree AST against a BlackboardSnapshot each tick.

    Tick semantics:
    - CONDITION returns SUCCESS or FAILURE (no motor command emitted).
    - ACTION(SetTarget(role)) writes to WorkingMemory and returns SUCCESS/FAILURE
      without terminating the tick, allowing downstream navigation/action nodes
      in a SEQUENCE to act on `target` in the same tick.
    - Motor ACTIONs (MoveTo, Grab, Drop, Press, ShootPortal) validate basic
      entity resolution / local preconditions:
        * If impossible (e.g., Grab when not holding/no cube in range, MoveTo
          on a missing entity or unlinked portal), return FAILURE so a parent
          SELECTOR can try fallback branches.
        * Otherwise emit the concrete Layer-3 MotorCommand and return RUNNING,
          ending the tick immediately.
    """

    def __init__(
        self,
        tree: BTNode,
        solid_arrival_radius: float = 18.0,
        floor_arrival_radius: float = 18.0,
        interact_radius: float = INTERACT_RADIUS,
    ) -> None:
        self.tree = tree
        self.solid_arrival_radius = solid_arrival_radius
        self.floor_arrival_radius = floor_arrival_radius
        self.interact_radius = interact_radius
        self.memory = WorkingMemory()

    def reset(self) -> None:
        self.memory.clear()

    def tick(self, snapshot: BlackboardSnapshot) -> TickResult:
        state = _TickContext(snapshot=snapshot, memory=self.memory)
        status = self._eval(self.tree, state)
        return TickResult(
            status=status,
            command=state.command if state.command is not None else NoOp(),
            nodes_visited=state.nodes_visited,
            target_role=self.memory.target_role,
            target_entity_id=self.memory.target_entity_id,
        )

    def _eval(self, node: BTNode, ctx: _TickContext) -> NodeStatus:
        ctx.nodes_visited += 1

        if isinstance(node, SelectorNode):
            for child in node.children:
                status = self._eval(child, ctx)
                if status is not NodeStatus.FAILURE:
                    return status
            return NodeStatus.FAILURE

        if isinstance(node, SequenceNode):
            if not node.children:
                return NodeStatus.FAILURE
            for child in node.children:
                status = self._eval(child, ctx)
                if status is not NodeStatus.SUCCESS:
                    return status
            return NodeStatus.SUCCESS

        if isinstance(node, ConditionNode):
            val = evaluate_predicate(
                node.predicate, node.args, ctx.snapshot, ctx.memory
            )
            if node.negated:
                val = not val
            return NodeStatus.SUCCESS if val else NodeStatus.FAILURE

        if isinstance(node, ActionNode):
            return self._eval_action(node, ctx)

        return NodeStatus.FAILURE

    def _eval_action(self, node: ActionNode, ctx: _TickContext) -> NodeStatus:
        name = node.name
        args = node.args
        snap = ctx.snapshot
        mem = ctx.memory

        if name == "SetTarget":
            if not args:
                return NodeStatus.FAILURE
            role = args[0]
            ent = resolve_entity(snap, mem, role)
            if ent is None:
                return NodeStatus.FAILURE
            mem.target_role = role
            mem.target_entity_id = ent.id
            return NodeStatus.SUCCESS

        if name == "MoveTo":
            ref = args[0] if args else "target"
            if ref == "portal_here":
                portal = snap.portal_in_zone(snap.player.zone_id)
                if portal is None:
                    return NodeStatus.FAILURE
                ctx.command = MoveTo(entity_id=f"portal:{portal.color.value}")
                return NodeStatus.RUNNING
            if ref in ("blue_portal", "orange_portal"):
                color = PortalColor.BLUE if ref == "blue_portal" else PortalColor.ORANGE
                portal = snap.portal(color)
                other = snap.portal(color.other())
                if not (portal.active and other.active):
                    return NodeStatus.FAILURE
                if portal.zone_id != snap.player.zone_id:
                    return NodeStatus.FAILURE
                ctx.command = MoveTo(entity_id=f"portal:{color.value}")
                return NodeStatus.RUNNING

            ent = resolve_entity(snap, mem, ref)
            if ent is None or ent.zone_id != snap.player.zone_id:
                return NodeStatus.FAILURE
            # Textbook BT arrival semantics: non-exit targets succeed once reached.
            # Default 18u preserves the frozen mock yardstick; Source Engine passes 42u/28u
            # to account for its 32u player + 32u solid prop collision hulls.
            if ent.kind in (EntityKind.CUBE, EntityKind.PEDESTAL_BUTTON):
                if snap.player.pos.distance_xy(ent.pos) <= self.solid_arrival_radius:
                    return NodeStatus.SUCCESS
            elif ent.kind is EntityKind.FLOOR_BUTTON:
                if snap.player.pos.distance_xy(ent.pos) <= self.floor_arrival_radius:
                    return NodeStatus.SUCCESS
            ctx.command = MoveTo(entity_id=ent.id)
            return NodeStatus.RUNNING

        if name == "Grab":
            if snap.player.held_entity_id is not None:
                return NodeStatus.FAILURE
            ref = args[0] if args else "target"
            ent = resolve_entity(snap, mem, ref)
            if ent is None or ent.kind is not EntityKind.CUBE:
                return NodeStatus.FAILURE
            if ent.zone_id != snap.player.zone_id:
                return NodeStatus.FAILURE
            if snap.player.pos.distance_xy(ent.pos) > self.interact_radius:
                return NodeStatus.FAILURE
            ctx.command = Grab(entity_id=ent.id)
            return NodeStatus.RUNNING

        if name == "Drop":
            if snap.player.held_entity_id is None:
                return NodeStatus.FAILURE
            # Enforce local-target symmetry with Grab/Press when a drop destination is bound.
            ref = args[0] if args else "target"
            ent = resolve_entity(snap, mem, ref)
            if ent is not None and ent.id != snap.player.held_entity_id:
                if ent.zone_id != snap.player.zone_id:
                    return NodeStatus.FAILURE
                if snap.player.pos.distance_xy(ent.pos) > self.interact_radius:
                    return NodeStatus.FAILURE
            ctx.command = Drop()
            return NodeStatus.RUNNING

        if name == "Press":
            ref = args[0] if args else "target"
            ent = resolve_entity(snap, mem, ref)
            if ent is None or ent.kind is not EntityKind.PEDESTAL_BUTTON:
                return NodeStatus.FAILURE
            if ent.zone_id != snap.player.zone_id:
                return NodeStatus.FAILURE
            if snap.player.pos.distance_xy(ent.pos) > self.interact_radius:
                return NodeStatus.FAILURE
            ctx.command = Press(entity_id=ent.id)
            return NodeStatus.RUNNING

        if name == "ShootPortal":
            if len(args) < 2:
                return NodeStatus.FAILURE
            try:
                color = PortalColor(args[0])
            except ValueError:
                return NodeStatus.FAILURE
            wall = resolve_wall(snap, mem, args[1])
            if wall is None:
                return NodeStatus.FAILURE
            # Avoid futilely re-shooting the exact same color onto the exact same wall
            current = snap.portal(color)
            if current.active and current.wall_id == wall.id:
                return NodeStatus.FAILURE
            ctx.command = ShootPortal(color=color, wall_id=wall.id)
            return NodeStatus.RUNNING

        return NodeStatus.FAILURE


@dataclass(slots=True)
class _TickContext:
    snapshot: BlackboardSnapshot
    memory: WorkingMemory
    command: MotorCommand | None = None
    nodes_visited: int = 0
