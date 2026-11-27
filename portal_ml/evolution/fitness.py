"""Fitness evaluation on Chambers 1-4 and strict held-out testing on Chamber 5."""

from __future__ import annotations

from dataclasses import dataclass, field

from portal_ml.bt.ast import BTNode
from portal_ml.bt.evaluator import BehaviorTreeEvaluator
from portal_ml.schema.blackboard import BlackboardSnapshot, EntityKind
from portal_ml.sim.mock_env import MockPortalEnv, PortalEnv

TRAIN_CHAMBERS: tuple[int, ...] = (1, 2, 3, 4)
HELD_OUT_CHAMBER: int = 5
HELD_OUT_CHAMBERS: tuple[int, ...] = (5, 6, 7)


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
    augmentations_solved: int = 0
    augmentations_total: int = 0

    @property
    def all_train_checks_passed(self) -> bool:
        return (
            self.chambers_solved == len(self.per_chamber)
            and self.augmentations_solved == self.augmentations_total
        )


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
    augmentation: str | None = None,
) -> ChamberEval:
    """Run a single deterministic episode on `chamber_id`."""
    evaluator = BehaviorTreeEvaluator(tree)
    snap = env.reset(chamber_id)
    if augmentation is not None and isinstance(env, MockPortalEnv):
        snap = env.apply_training_augmentation(augmentation)
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
    causal_augmentation: bool = True,
) -> FitnessReport:
    """Evaluate a genome strictly on Training Chambers 1-4.

    When ``causal_augmentation`` is True (and ``env`` is a ``MockPortalEnv``),
    adds two lightweight causal-invariance rollouts strictly derived from
    Training Chambers 3 and 4 (distractor walls on Ch3; stale non-bridging
    zone-0 portals on Ch4) to break training-set spurious correlations without
    using any held-out chamber.
    """
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

    aug_bonus = 0.0
    aug_solved = 0
    aug_total = 0
    if causal_augmentation and isinstance(sim, MockPortalEnv):
        if 3 in train_chambers:
            aug_total += 1
            aug3 = evaluate_chamber(
                tree,
                sim,
                3,
                include_approach_shaping=False,
                augmentation="ch3_distractor_walls",
            )
            aug_bonus += 0.5 * aug3.reward + (25.0 if aug3.passed else 0.0)
            if aug3.passed:
                aug_solved += 1
        if 4 in train_chambers:
            aug_total += 1
            aug4 = evaluate_chamber(
                tree,
                sim,
                4,
                include_approach_shaping=False,
                augmentation="ch4_stale_portals",
            )
            aug_bonus += 0.5 * aug4.reward + (25.0 if aug4.passed else 0.0)
            if aug4.passed:
                aug_solved += 1

    # Bonus for solving all training chambers (and any causal augmentations) simultaneously
    all_solved_bonus = (
        60.0
        if (solved == len(train_chambers) and aug_solved == aug_total)
        else 0.0
    )
    size = tree.size()
    parsimony_penalty = parsimony_weight * size
    total_fitness = raw_reward + aug_bonus + all_solved_bonus - parsimony_penalty

    return FitnessReport(
        total_fitness=total_fitness,
        raw_reward=raw_reward,
        parsimony_penalty=parsimony_penalty,
        chambers_solved=solved,
        tree_size=size,
        per_chamber=evals,
        augmentations_solved=aug_solved,
        augmentations_total=aug_total,
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
