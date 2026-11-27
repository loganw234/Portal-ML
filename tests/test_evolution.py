"""Tests for evolutionary BT mutations, crossover, and Chamber 5 generalization."""

from __future__ import annotations

import random
import unittest

from portal_ml.bt import format_bt, parse_bt
from portal_ml.evolution import (
    EvolutionConfig,
    EvolutionTrainer,
    GenomeFactory,
    crossover,
    mutate,
)


class TestEvolution(unittest.TestCase):
    def test_mutations_and_crossover_preserve_valid_dsl(self) -> None:
        rng = random.Random(123)
        factory = GenomeFactory(mode="decoupled", max_nodes=48)
        g1 = factory.random_genome(rng)
        g2 = factory.random_genome(rng)
        for _ in range(50):
            g1 = mutate(g1, rng, factory)
            child = crossover(g1, g2, rng, factory)
            reparsed = parse_bt(format_bt(child))
            self.assertEqual(reparsed.size(), child.size())

    def test_evolution_solves_train_and_generalizes_to_held_out_suite(self) -> None:
        cfg = EvolutionConfig(
            population_size=120,
            generations=90,
            seed=7,
            mode="decoupled",
        )
        res = EvolutionTrainer(cfg).run()
        self.assertEqual(
            res.train_report.chambers_solved,
            4,
            "Evolved champion should solve all 4 training chambers",
        )
        self.assertTrue(
            res.held_out_report.passed,
            "Champion trained strictly on Chambers 1-4 should generalize zero-shot to Chamber 5",
        )
        for ho in res.held_out_reports:
            self.assertTrue(
                ho.passed,
                f"Champion trained strictly on Chambers 1-4 should generalize zero-shot to Chamber {ho.chamber_id}",
            )
        self.assertLess(res.held_out_report.steps, 100)


if __name__ == "__main__":
    unittest.main()
