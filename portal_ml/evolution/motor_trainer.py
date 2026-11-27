"""Phase 2 Neuroevolution trainer for small MLP motor primitives.

Evolves flat weight vectors from random Gaussian initialization via elitist
tournament neuroevolution with adaptive Gaussian mutation (zero backprop,
zero external dependencies, completely chamber-agnostic synthetic training tasks).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from portal_ml.motor.neural import (
    AIM_CONE_DEG,
    AIM_INPUT_NAMES,
    AIM_OUTPUT_NAMES,
    FIRE_GATE_THRESHOLD,
    MLP,
    MOVETO_INPUT_NAMES,
    MOVETO_OUTPUT_NAMES,
    NeuralAimPolicy,
    NeuralMoveToPolicy,
    aim_egocentric_features,
    moveto_egocentric_features,
)
from portal_ml.schema.blackboard import Vec3
from portal_ml.sim.mock_env import FIXED_DT, MAX_YAW_RATE, PLAYER_SPEED


@dataclass(slots=True)
class MotorEvoConfig:
    population_size: int = 80
    generations: int = 120
    elitism: int = 6
    tournament_size: int = 4
    mutation_sigma: float = 0.22
    mutation_prob: float = 0.35
    crossover_rate: float = 0.4
    episodes_per_eval: int = 16
    seed: int = 7


@dataclass(slots=True)
class MotorTrainResult:
    net: MLP
    best_fitness: float
    success_rate: float
    generations: int
    elapsed_sec: float


def _mutate_weights(
    parent: MLP, rng: random.Random, sigma: float, prob: float
) -> MLP:
    child = parent.clone()
    w = child.weights
    for i in range(len(w)):
        if rng.random() < prob:
            w[i] += rng.gauss(0.0, sigma)
    return child


def _crossover_weights(a: MLP, b: MLP, rng: random.Random) -> MLP:
    child = a.clone()
    wb = b.weights
    wc = child.weights
    for i in range(len(wc)):
        if rng.random() < 0.5:
            wc[i] = wb[i]
    return child


def _make_moveto_episodes(
    rng: random.Random, count: int
) -> list[tuple[float, Vec3, bool]]:
    """Generate chamber-agnostic synthetic target-reaching episodes."""
    eps: list[tuple[float, Vec3, bool]] = []
    for i in range(count):
        init_yaw = rng.uniform(-180.0, 180.0)
        bearing = rng.uniform(-math.pi, math.pi)
        dist = rng.uniform(96.0, 420.0)
        target = Vec3(dist * math.cos(bearing), dist * math.sin(bearing), 0.0)
        is_portal = (i % 4 == 0)
        eps.append((init_yaw, target, is_portal))
    return eps


def _eval_moveto_mlp(
    net: MLP, episodes: list[tuple[float, Vec3, bool]]
) -> tuple[float, float]:
    total_fit = 0.0
    successes = 0
    step_dist = PLAYER_SPEED * FIXED_DT
    max_ticks = 26

    for init_yaw, target, is_portal in episodes:
        px, py = 0.0, 0.0
        yaw = init_yaw
        d0 = math.hypot(target.x, target.y)
        reached = False
        end_tick = max_ticks
        min_d = d0

        for tick in range(1, max_ticks + 1):
            feats = moveto_egocentric_features(Vec3(px, py, 0.0), yaw, target, is_portal)
            fwd, strafe, yd = net.forward(feats)

            mag = math.hypot(fwd, strafe)
            if mag > 1.0:
                fwd /= mag
                strafe /= mag

            yaw = (yaw + yd * MAX_YAW_RATE * FIXED_DT + 180.0) % 360.0 - 180.0
            rad = math.radians(yaw)
            cos_y = math.cos(rad)
            sin_y = math.sin(rad)
            px += (fwd * cos_y - strafe * sin_y) * step_dist
            py += (fwd * sin_y + strafe * cos_y) * step_dist

            d = math.hypot(target.x - px, target.y - py)
            if d < min_d:
                min_d = d
            if d <= 16.0:
                reached = True
                end_tick = tick
                break

        d_end = math.hypot(target.x - px, target.y - py)
        progress = max(0.0, (d0 - d_end) / max(d0, 1.0))
        fit = 40.0 * progress + 10.0 * max(0.0, (d0 - min_d) / max(d0, 1.0))
        if reached:
            successes += 1
            # Reward fast, direct arrival + facing generally toward target
            target_bearing = math.degrees(math.atan2(target.y - py, target.x - px))
            err = abs(((target_bearing - yaw) + 180.0) % 360.0 - 180.0)
            fit += 60.0 + 1.5 * (max_ticks - end_tick) + 4.0 * max(0.0, 1.0 - err / 180.0)
        total_fit += fit

    n = max(1, len(episodes))
    return total_fit / n, successes / n


def evolve_moveto_policy(config: MotorEvoConfig | None = None) -> MotorTrainResult:
    """Neuroevolve an 83-weight MLP (6 -> 8 -> 3) for continuous egocentric MoveTo."""
    cfg = config or MotorEvoConfig()
    rng = random.Random(cfg.seed)
    t0 = time.perf_counter()

    # Fixed validation panel (disjoint seed) for unbiased champion selection
    val_rng = random.Random(cfg.seed + 10_000)
    val_episodes = _make_moveto_episodes(val_rng, 24)

    pop = [
        MLP.random(
            len(MOVETO_INPUT_NAMES),
            8,
            len(MOVETO_OUTPUT_NAMES),
            rng,
            scale=0.6,
            input_names=MOVETO_INPUT_NAMES,
            output_names=MOVETO_OUTPUT_NAMES,
        )
        for _ in range(cfg.population_size)
    ]

    best_net = pop[0].clone()
    best_val_fit = -1e9
    best_val_rate = 0.0

    for gen in range(cfg.generations):
        train_eps = _make_moveto_episodes(rng, cfg.episodes_per_eval)
        scored: list[tuple[MLP, float]] = []
        for net in pop:
            fit, _ = _eval_moveto_mlp(net, train_eps)
            scored.append((net, fit))
        scored.sort(key=lambda item: item[1], reverse=True)

        # Re-evaluate top 4 candidates on the fixed validation panel
        for cand, _ in scored[:4]:
            vfit, vrate = _eval_moveto_mlp(cand, val_episodes)
            if vfit > best_val_fit:
                best_val_fit = vfit
                best_val_rate = vrate
                best_net = cand.clone()

        sigma = cfg.mutation_sigma * (0.4 + 0.6 * (1.0 - gen / max(1, cfg.generations)))
        next_pop = [n.clone() for n, _ in scored[: cfg.elitism]]
        while len(next_pop) < cfg.population_size:
            p1 = max(rng.sample(scored, cfg.tournament_size), key=lambda x: x[1])[0]
            if rng.random() < cfg.crossover_rate:
                p2 = max(rng.sample(scored, cfg.tournament_size), key=lambda x: x[1])[0]
                child = _crossover_weights(p1, p2, rng)
            else:
                child = p1.clone()
            child = _mutate_weights(child, rng, sigma, cfg.mutation_prob)
            next_pop.append(child)
        pop = next_pop

    elapsed = time.perf_counter() - t0
    best_net.metadata = {
        "primitive": "MoveTo",
        "seed": cfg.seed,
        "generations": cfg.generations,
        "param_count": len(best_net.weights),
        "val_fitness": round(best_val_fit, 3),
        "val_success_rate": round(best_val_rate, 4),
    }
    return MotorTrainResult(
        net=best_net,
        best_fitness=best_val_fit,
        success_rate=best_val_rate,
        generations=cfg.generations,
        elapsed_sec=elapsed,
    )


def _make_aim_episodes(rng: random.Random, count: int) -> list[tuple[float, Vec3]]:
    eps: list[tuple[float, Vec3]] = []
    for _ in range(count):
        init_yaw = rng.uniform(-180.0, 180.0)
        bearing = rng.uniform(-math.pi, math.pi)
        dist = rng.uniform(128.0, 512.0)
        wall = Vec3(dist * math.cos(bearing), dist * math.sin(bearing), 64.0)
        eps.append((init_yaw, wall))
    return eps


def _eval_aim_mlp(
    net: MLP, episodes: list[tuple[float, Vec3]]
) -> tuple[float, float]:
    total_fit = 0.0
    successes = 0
    max_ticks = 12
    origin = Vec3(0.0, 0.0, 0.0)

    for init_yaw, wall in episodes:
        yaw = init_yaw
        misfires = 0
        fired_on_target = False
        end_tick = max_ticks
        min_abs_err = 180.0

        for tick in range(1, max_ticks + 1):
            feats, err_deg = aim_egocentric_features(origin, yaw, wall)
            if abs(err_deg) < min_abs_err:
                min_abs_err = abs(err_deg)
            yd, fire_gate = net.forward(feats)

            if fire_gate > FIRE_GATE_THRESHOLD:
                if abs(err_deg) <= AIM_CONE_DEG:
                    fired_on_target = True
                    end_tick = tick
                    break
                misfires += 1

            yaw = (yaw + yd * MAX_YAW_RATE * FIXED_DT + 180.0) % 360.0 - 180.0

        fit = 30.0 * max(0.0, 1.0 - min_abs_err / 180.0) - 8.0 * misfires
        if fired_on_target:
            successes += 1
            fit += 100.0 + 3.0 * (max_ticks - end_tick)
        total_fit += fit

    n = max(1, len(episodes))
    return total_fit / n, successes / n


def evolve_aim_policy(config: MotorEvoConfig | None = None) -> MotorTrainResult:
    """Neuroevolve a 44-weight MLP (4 -> 6 -> 2) for ShootPortal yaw aiming & firing."""
    cfg = config or MotorEvoConfig(generations=100)
    rng = random.Random(cfg.seed)
    t0 = time.perf_counter()

    val_rng = random.Random(cfg.seed + 20_000)
    val_episodes = _make_aim_episodes(val_rng, 32)

    pop = [
        MLP.random(
            len(AIM_INPUT_NAMES),
            6,
            len(AIM_OUTPUT_NAMES),
            rng,
            scale=0.6,
            input_names=AIM_INPUT_NAMES,
            output_names=AIM_OUTPUT_NAMES,
        )
        for _ in range(cfg.population_size)
    ]

    best_net = pop[0].clone()
    best_val_fit = -1e9
    best_val_rate = 0.0

    for gen in range(cfg.generations):
        train_eps = _make_aim_episodes(rng, cfg.episodes_per_eval)
        scored: list[tuple[MLP, float]] = []
        for net in pop:
            fit, _ = _eval_aim_mlp(net, train_eps)
            scored.append((net, fit))
        scored.sort(key=lambda item: item[1], reverse=True)

        for cand, _ in scored[:4]:
            vfit, vrate = _eval_aim_mlp(cand, val_episodes)
            if vfit > best_val_fit:
                best_val_fit = vfit
                best_val_rate = vrate
                best_net = cand.clone()

        sigma = cfg.mutation_sigma * (0.35 + 0.65 * (1.0 - gen / max(1, cfg.generations)))
        next_pop = [n.clone() for n, _ in scored[: cfg.elitism]]
        while len(next_pop) < cfg.population_size:
            p1 = max(rng.sample(scored, cfg.tournament_size), key=lambda x: x[1])[0]
            if rng.random() < cfg.crossover_rate:
                p2 = max(rng.sample(scored, cfg.tournament_size), key=lambda x: x[1])[0]
                child = _crossover_weights(p1, p2, rng)
            else:
                child = p1.clone()
            child = _mutate_weights(child, rng, sigma, cfg.mutation_prob)
            next_pop.append(child)
        pop = next_pop

    elapsed = time.perf_counter() - t0
    best_net.metadata = {
        "primitive": "ShootPortalAim",
        "seed": cfg.seed,
        "generations": cfg.generations,
        "param_count": len(best_net.weights),
        "val_fitness": round(best_val_fit, 3),
        "val_success_rate": round(best_val_rate, 4),
    }
    return MotorTrainResult(
        net=best_net,
        best_fitness=best_val_fit,
        success_rate=best_val_rate,
        generations=cfg.generations,
        elapsed_sec=elapsed,
    )
