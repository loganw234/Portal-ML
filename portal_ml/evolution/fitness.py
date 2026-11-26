"""Fitness evaluation on Chambers 1-4 and strict held-out testing on Chamber 5."""

from __future__ import annotations

from dataclasses import dataclass, field

from portal_ml.bt.ast import BTNode
from portal_ml.bt.evaluator import BehaviorTreeEvaluator
from portal_ml.schema.blackboard import BlackboardSnapshot, EntityKind
from portal_ml.sim.mock_env import MockPortalEnv, PortalEnv

TRAIN_CHAMBERS: tuple[int, ...] = (1, 2, 3, 4)
HELD_OUT_CHAMBER: int = 5


@dataclass(slots=True)
class ChamberEval:
    chamber_id: int
    passed: bool
    steps: int
    reward: float
    milestones: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FitnessReport:
    total_fitness: float
    raw_reward: float
    parsimony_penalty: float
    chambers_solved: int
    tree_size: int
    per_chamber: list[ChamberEval] = field(default_factory=list)


@dataclass(slots=True)
class HeldOutReport:
    chamber_id: int
    passed: bool
    steps: int
    milestones: list[str] = field(default_factory=list)


def _closest_approach_bonus(
    chamber_id: int,
    initial_snap: BlackboardSnapshot,
    min_dist_to_goal: float,
    initial_dist: float,
) -> float:
    """Smooth evolutionary gradient (0..5 pts) for approaching the next objective."""
    if initial_dist <= 1e-3:
        return 0.0
    progress = max(0.0, min(1.0, (initial_dist - min_dist_to_goal) / initial_dist))
    return 5.0 * progress


def _primary_approach_target(
    chamber_id: int, snap: BlackboardSnapshot
):
    if chamber_id == 1:
        return snap.first_of_kind(EntityKind.EXIT_DOOR)
    if chamber_id == 2:
        return snap.first_of_kind(EntityKind.PEDESTAL_BUTTON)
    if chamber_id == 3:
        return snap.first_of_kind(EntityKind.CUBE)
    if chamber_id == 4:
        return snap.first_of_kind(EntityKind.EXIT_DOOR)
    return None


def _state_signature(snap: BlackboardSnapshot) -> tuple:
    p = snap.player
    return (
        round(p.pos.x, 1),
        round(p.pos.y, 1),
        p.zone_id,
        p.held_entity_id,
        snap.blue_portal.wall_id if snap.blue_portal.active else None,
        snap.orange_portal.wall_id if snap.orange_portal.active else None,
        tuple((e.id, round(e.pos.x, 1), round(e.pos.y, 1), e.active, e.held) for e in snap.entities),
    )


def evaluate_chamber(
    tree: BTNode,
    env: PortalEnv,
    chamber_id: int,
    include_approach_shaping: bool = True,
    stall_limit: int = 8,
) -> ChamberEval:
    """Run a single deterministic episode on `chamber_id`."""
    evaluator = BehaviorTreeEvaluator(tree)
    snap = env.reset(chamber_id)
    total_reward = 0.0
    milestones: list[str] = []

    target_ent = _primary_approach_target(chamber_id, snap) if include_approach_shaping else None
    init_dist = (
        snap.player.pos.distance_xy(target_ent.pos)
        if target_ent is not None and target_ent.zone_id == snap.player.zone_id
        else 0.0
    )
    min_dist = init_dist

    # Secondary carrying shaping for Chamber 3: reward bringing held cube closer to floor button
    floor_btn = snap.first_of_kind(EntityKind.FLOOR_BUTTON) if include_approach_shaping else None
    cube_init_btn_dist = 0.0
    cube_min_btn_dist = 0.0
    if floor_btn is not None:
        cube0 = snap.first_of_kind(EntityKind.CUBE)
        if cube0 is not None and cube0.zone_id == floor_btn.zone_id:
            cube_init_btn_dist = cube0.pos.distance_xy(floor_btn.pos)
            cube_min_btn_dist = cube_init_btn_dist

    steps = 0
    done = False
    success = False
    prev_sig = _state_signature(snap)
    stalled_ticks = 0
    max_steps = env.spec.max_steps if isinstance(env, MockPortalEnv) else 260

    while not done:
        tick_res = evaluator.tick(snap)
        step_res = env.step(tick_res.command)
        snap = step_res.snapshot
        total_reward += step_res.reward
        if step_res.milestones:
            milestones.extend(step_res.milestones)
        steps += 1
        done = step_res.done
        success = step_res.success

        if target_ent is not None and init_dist > 0.0:
            cur_ent = snap.entity_by_id(target_ent.id)
            if cur_ent is not None and cur_ent.zone_id == snap.player.zone_id:
                d = snap.player.pos.distance_xy(cur_ent.pos)
                if d < min_dist:
                    min_dist = d

        if floor_btn is not None and cube_init_btn_dist > 0.0 and snap.player.held_entity_id is not None:
            d_btn = snap.player.pos.distance_xy(floor_btn.pos)
            if d_btn < cube_min_btn_dist:
                cube_min_btn_dist = d_btn

        if not done and stall_limit > 0:
            sig = _state_signature(snap)
            if sig == prev_sig:
                stalled_ticks += 1
                if stalled_ticks >= stall_limit:
                    # Deterministic stall: charge remaining step penalties and exit early
                    remaining = max(0, max_steps - steps)
                    total_reward -= 0.05 * remaining
                    steps = max_steps
                    done = True
                    success = False
            else:
                prev_sig = sig
                stalled_ticks = 0

    if include_approach_shaping and not success:
        if init_dist > 0.0:
            total_reward += _closest_approach_bonus(chamber_id, snap, min_dist, init_dist)
        if cube_init_btn_dist > 0.0 and cube_min_btn_dist < cube_init_btn_dist:
            carry_prog = max(
                0.0,
                min(1.0, (cube_init_btn_dist - cube_min_btn_dist) / cube_init_btn_dist),
            )
            total_reward += 10.0 * carry_prog

    return ChamberEval(
        chamber_id=chamber_id,
        passed=success,
        steps=steps,
        reward=total_reward,
        milestones=milestones,
    )


def evaluate_genome(
    tree: BTNode,
    env: PortalEnv | None = None,
    parsimony_weight: float = 0.15,
    train_chambers: tuple[int, ...] = TRAIN_CHAMBERS,
) -> FitnessReport:
    """Evaluate a genome strictly on Training Chambers 1-4."""
    sim = env if env is not None else MockPortalEnv()
    evals: list[ChamberEval] = []
    raw_reward = 0.0
    solved = 0

    for cid in train_chambers:
        res = evaluate_chamber(tree, sim, cid, include_approach_shaping=True)
        evals.append(res)
        raw_reward += res.reward
        if res.passed:
            solved += 1

    # Bonus for solving all training chambers simultaneously (prevents specializing on 3 of 4)
    all_solved_bonus = 60.0 if solved == len(train_chambers) else 0.0
    size = tree.size()
    parsimony_penalty = parsimony_weight * size
    total_fitness = raw_reward + all_solved_bonus - parsimony_penalty

    return FitnessReport(
        total_fitness=total_fitness,
        raw_reward=raw_reward,
        parsimony_penalty=parsimony_penalty,
        chambers_solved=solved,
        tree_size=size,
        per_chamber=evals,
    )


def evaluate_held_out(
    tree: BTNode,
    env: PortalEnv | None = None,
    chamber_id: int = HELD_OUT_CHAMBER,
) -> HeldOutReport:
    """Strictly evaluate on held-out Chamber 5 (pass/fail + step count only)."""
    sim = env if env is not None else MockPortalEnv()
    res = evaluate_chamber(tree, sim, chamber_id, include_approach_shaping=False)
    return HeldOutReport(
        chamber_id=chamber_id,
        passed=res.passed,
        steps=res.steps,
        milestones=res.milestones,
    )
