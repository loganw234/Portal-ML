"""Evolutionary training loop for Behavior Tree AST genomes."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable

from portal_ml.bt.ast import BTNode
from portal_ml.evolution.fitness import (
    FitnessReport,
    HELD_OUT_CHAMBERS,
    HeldOutReport,
    TRAIN_CHAMBERS,
    evaluate_genome,
    evaluate_held_out,
)
from portal_ml.evolution.genome import GenomeFactory, crossover, mutate
from portal_ml.sim.mock_env import MockPortalEnv, PortalEnv


@dataclass(slots=True)
class EvolutionConfig:
    population_size: int = 120
    generations: int = 120
    elitism: int = 6
    tournament_size: int = 5
    crossover_rate: float = 0.45
    mutation_rate: float = 0.85
    immigrant_fraction: float = 0.10
    parsimony_weight: float = 0.15
    max_nodes: int = 48
    mode: str = "decoupled"  # "decoupled" | "unconstrained"
    seed: int = 42
    polish_generations_after_solve: int = 12
    causal_augmentation: bool = True


@dataclass(slots=True)
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    chambers_solved: int
    best_tree_size: int
    episodes_total: int


@dataclass(slots=True)
class EvolutionResult:
    champion: BTNode
    train_report: FitnessReport
    held_out_report: HeldOutReport
    held_out_reports: list[HeldOutReport] = field(default_factory=list)
    history: list[GenerationStats] = field(default_factory=list)
    episodes_evaluated: int = 0
    elapsed_sec: float = 0.0

    @property
    def episodes_per_sec(self) -> float:
        return self.episodes_evaluated / max(self.elapsed_sec, 1e-9)


class EvolutionTrainer:
    """Runs tournament-selection + elitism GP over Behavior Tree ASTs."""

    def __init__(
        self,
        config: EvolutionConfig | None = None,
        env: PortalEnv | None = None,
    ) -> None:
        self.config = config or EvolutionConfig()
        self.env: PortalEnv = (
            env if env is not None else MockPortalEnv(reuse_snapshot=True)
        )
        self.factory = GenomeFactory(
            mode=self.config.mode, max_nodes=self.config.max_nodes
        )

    def run(
        self,
        on_generation: Callable[[GenerationStats], None] | None = None,
    ) -> EvolutionResult:
        cfg = self.config
        rng = random.Random(cfg.seed)
        t0 = time.perf_counter()

        population: list[BTNode] = [
            self.factory.random_genome(rng) for _ in range(cfg.population_size)
        ]

        history: list[GenerationStats] = []
        episodes_total = 0
        best_genome: BTNode = population[0]
        best_report: FitnessReport | None = None
        first_all_solved_gen: int | None = None
        stagnant_gens = 0

        for gen in range(cfg.generations):
            scored: list[tuple[BTNode, FitnessReport]] = []
            for genome in population:
                report = evaluate_genome(
                    genome,
                    env=self.env,
                    parsimony_weight=cfg.parsimony_weight,
                    causal_augmentation=cfg.causal_augmentation,
                )
                episodes_total += len(TRAIN_CHAMBERS) + (
                    2 if cfg.causal_augmentation and isinstance(self.env, MockPortalEnv) else 0
                )
                scored.append((genome, report))

            scored.sort(key=lambda item: item[1].total_fitness, reverse=True)
            gen_best_genome, gen_best_report = scored[0]

            if (
                best_report is None
                or gen_best_report.total_fitness > best_report.total_fitness + 1e-6
            ):
                best_genome = gen_best_genome.clone()
                best_report = gen_best_report
                stagnant_gens = 0
            else:
                stagnant_gens += 1

            mean_fit = sum(r.total_fitness for _, r in scored) / len(scored)
            stats = GenerationStats(
                generation=gen,
                best_fitness=gen_best_report.total_fitness,
                mean_fitness=mean_fit,
                chambers_solved=gen_best_report.chambers_solved,
                best_tree_size=gen_best_report.tree_size,
                episodes_total=episodes_total,
            )
            history.append(stats)
            if on_generation is not None:
                on_generation(stats)

            if gen_best_report.all_train_checks_passed:
                if first_all_solved_gen is None:
                    first_all_solved_gen = gen
                elif (
                    gen - first_all_solved_gen
                    >= cfg.polish_generations_after_solve
                ):
                    break

            # Build next generation
            next_pop: list[BTNode] = [
                g.clone() for g, _ in scored[: cfg.elitism]
            ]

            num_immigrants = max(1, int(cfg.population_size * cfg.immigrant_fraction))
            for _ in range(num_immigrants):
                next_pop.append(self.factory.random_genome(rng))

            while len(next_pop) < cfg.population_size:
                p1 = self._tournament(scored, rng, cfg.tournament_size)
                if rng.random() < cfg.crossover_rate:
                    p2 = self._tournament(scored, rng, cfg.tournament_size)
                    child = crossover(p1, p2, rng, self.factory)
                else:
                    child = p1.clone()
                if rng.random() < cfg.mutation_rate:
                    child = mutate(child, rng, self.factory)
                    if stagnant_gens >= 5 and rng.random() < 0.55:
                        child = mutate(child, rng, self.factory)
                next_pop.append(child)

            population = next_pop

        assert best_report is not None
        # Strictly evaluate champion zero-shot on held-out Chambers 5, 6, and 7
        held_out_all = [
            evaluate_held_out(best_genome, env=self.env, chamber_id=cid)
            for cid in HELD_OUT_CHAMBERS
        ]
        held_out = held_out_all[0]
        elapsed = time.perf_counter() - t0

        return EvolutionResult(
            champion=best_genome,
            train_report=best_report,
            held_out_report=held_out,
            held_out_reports=held_out_all,
            history=history,
            episodes_evaluated=episodes_total,
            elapsed_sec=elapsed,
        )

    @staticmethod
    def _tournament(
        scored: list[tuple[BTNode, FitnessReport]],
        rng: random.Random,
        k: int,
    ) -> BTNode:
        contenders = rng.sample(scored, min(k, len(scored)))
        winner = max(contenders, key=lambda item: item[1].total_fitness)
        return winner[0]
