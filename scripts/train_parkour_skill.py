"""Neuroevolve the Milestone-7 sensor-equipped NeuralParkourPolicy (208 weights).

Trains strictly on randomized synthetic 3D physics scenarios (open field, tall
obstacle slalom, jumpable hurdle, toxic acid trench, platform gap jump) with zero
exposure to Chambers 1-7 or held-out physical benchmarks P1-P3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portal_ml.evolution.motor_trainer import MotorEvoConfig, evolve_parkour_policy


def main() -> int:
    parser = argparse.ArgumentParser(description="Evolve NeuralParkourPolicy MLP checkpoint.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pop", type=int, default=72)
    parser.add_argument("--gens", type=int, default=120)
    parser.add_argument("--hidden", type=int, default=12)
    parser.add_argument("--out", default="champions/parkour_mlp.json")
    args = parser.parse_args()

    cfg = MotorEvoConfig(
        population_size=args.pop,
        generations=args.gens,
        elitism=6,
        tournament_size=4,
        mutation_sigma=0.25,
        mutation_prob=0.30,
        crossover_rate=0.45,
        seed=args.seed,
    )
    print(
        f"Evolving NeuralParkourPolicy (12->{args.hidden}->4) "
        f"seed={args.seed} pop={args.pop} gens={args.gens}..."
    )
    res = evolve_parkour_policy(cfg, hidden_size=args.hidden)
    saved = res.net.save(args.out)
    print(
        f"Saved {saved} | params={len(res.net.weights)} | "
        f"val_success={res.success_rate * 100.0:.1f}% | "
        f"val_fitness={res.best_fitness:.2f} | elapsed={res.elapsed_sec:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
