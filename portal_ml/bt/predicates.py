"""Symbolic target resolution and predicate evaluation over BlackboardSnapshot."""

from __future__ import annotations

from portal_ml.schema.blackboard import (
    BlackboardSnapshot,
    EntityKind,
    EntityState,
    PortalColor,
    PortalWall,
    WorkingMemory,
)

# Source Hammer units: interaction radius (strictly <= BUTTON_TRIGGER_RADIUS=52.0)
INTERACT_RADIUS = 44.0

ROLE_TO_KIND: dict[str, EntityKind] = {
    "exit": EntityKind.EXIT_DOOR,
    "cube": EntityKind.CUBE,
    "floor_button": EntityKind.FLOOR_BUTTON,
    "pedestal_button": EntityKind.PEDESTAL_BUTTON,
}

ENTITY_ROLES: tuple[str, ...] = ("exit", "cube", "floor_button", "pedestal_button")


def resolve_entity(
    snapshot: BlackboardSnapshot, memory: WorkingMemory, ref: str
) -> EntityState | None:
    """Resolve a symbolic role ('target', 'cube', 'exit', ...) or concrete ID."""
    if ref == "target":
        if memory.target_entity_id is None:
            return None
        return snapshot.entity_by_id(memory.target_entity_id)
    kind = ROLE_TO_KIND.get(ref)
    if kind is not None:
        return snapshot.first_of_kind(kind)
    return snapshot.entity_by_id(ref)


def resolve_zone(
    snapshot: BlackboardSnapshot, memory: WorkingMemory, zone_ref: str
) -> int | None:
    """Resolve a zone reference ('here', 'target', 'exit', 'cube', 'floor_button')."""
    if zone_ref == "here":
        return snapshot.player.zone_id
    ent = resolve_entity(snapshot, memory, zone_ref)
    return ent.zone_id if ent is not None else None


def resolve_wall(
    snapshot: BlackboardSnapshot, memory: WorkingMemory, wall_ref: str
) -> PortalWall | None:
    """Resolve a symbolic wall selector or concrete wall ID."""
    if wall_ref == "wall_here":
        return snapshot.wall_in_zone(snapshot.player.zone_id)
    if wall_ref == "wall_other_zone":
        for w in snapshot.portal_walls:
            if w.zone_id != snapshot.player.zone_id:
                return w
        return None
    if wall_ref.startswith("wall_near_"):
        role = wall_ref[len("wall_near_") :]
        ent = resolve_entity(snapshot, memory, role)
        if ent is None:
            return None
        return snapshot.wall_in_zone(ent.zone_id)
    return snapshot.wall_by_id(wall_ref)


def evaluate_predicate(
    predicate: str,
    args: tuple[str, ...],
    snapshot: BlackboardSnapshot,
    memory: WorkingMemory,
) -> bool:
    """Evaluate a named symbolic predicate on the current Blackboard snapshot."""
    if predicate == "exists":
        if not args:
            return False
        return resolve_entity(snapshot, memory, args[0]) is not None

    if predicate == "is_open":
        ref = args[0] if args else "exit"
        ent = resolve_entity(snapshot, memory, ref)
        return ent is not None and ent.kind is EntityKind.EXIT_DOOR and ent.active

    if predicate == "is_pressed":
        if not args:
            return False
        ent = resolve_entity(snapshot, memory, args[0])
        return (
            ent is not None
            and ent.kind in (EntityKind.FLOOR_BUTTON, EntityKind.PEDESTAL_BUTTON)
            and ent.active
        )

    if predicate == "is_holding":
        held_id = snapshot.player.held_entity_id
        if held_id is None:
            return False
        if not args or args[0] == "any":
            return True
        ref = args[0]
        ent = resolve_entity(snapshot, memory, ref)
        return ent is not None and ent.id == held_id

    if predicate == "is_reachable":
        ref = args[0] if args else "target"
        ent = resolve_entity(snapshot, memory, ref)
        return ent is not None and ent.zone_id == snapshot.player.zone_id

    if predicate == "in_range":
        ref = args[0] if args else "target"
        ent = resolve_entity(snapshot, memory, ref)
        if ent is None or ent.zone_id != snapshot.player.zone_id:
            return False
        return snapshot.player.pos.distance_xy(ent.pos) <= INTERACT_RADIUS

    if predicate == "target_is":
        if not args or memory.target_role is None:
            return False
        return memory.target_role == args[0]

    if predicate == "portals_active":
        return snapshot.blue_portal.active and snapshot.orange_portal.active

    if predicate == "portals_bridge":
        ref = args[0] if args else "target"
        ent = resolve_entity(snapshot, memory, ref)
        if ent is None:
            return False
        return snapshot.portals_bridge_zones(snapshot.player.zone_id, ent.zone_id)

    if predicate == "portal_in_zone":
        if len(args) < 2:
            return False
        try:
            color = PortalColor(args[0])
        except ValueError:
            return False
        zone_id = resolve_zone(snapshot, memory, args[1])
        if zone_id is None:
            return False
        portal = snapshot.portal(color)
        return portal.active and portal.zone_id == zone_id

    if predicate == "cube_on_button":
        cube = snapshot.first_of_kind(EntityKind.CUBE)
        return cube is not None and cube.active

    return False
