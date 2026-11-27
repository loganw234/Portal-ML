"""CLI runner for evolving Behavior Tree agents on Chambers 1-4 and testing Chamber 5."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portal_ml.bridge.source_env import SourcePortalEnv
from portal_ml.bt.ast import format_bt
from portal_ml.evolution.trainer import EvolutionConfig, EvolutionTrainer, GenerationStats
from portal_ml.sim.mock_env import MockPortalEnv, PortalEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evolve a Behavior Tree DSL agent on Chambers 1-4 and evaluate zero-shot on Chamber 5."
    )
    parser.add_argument("--backend", choices=["mock", "source"], default="mock")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=27500)
    parser.add_argument("--mode", choices=["decoupled", "unconstrained"], default="decoupled")
    parser.add_argument("--pop", type=int, default=120, help="Population size")
    parser.add_argument("--gens", type=int, default=100, help="Max generations")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed")
    parser.add_argument(
        "--no-causal-aug",
        action="store_true",
        help="Disable training-only causal invariance augmentations on Chambers 3 & 4.",
    )
    parser.add_argument("--save-champion", type=str, default="", help="Optional path to save champion DSL")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = EvolutionConfig(
        population_size=args.pop,
        generations=args.gens,
        mode=args.mode,
        seed=args.seed,
        causal_augmentation=not args.no_causal_aug,
    )

    source_env: SourcePortalEnv | None = None
    env: PortalEnv
    if args.backend == "source":
        source_env = SourcePortalEnv(host=args.host, port=args.port)
        source_env.connect()
        env = source_env
    else:
        env = MockPortalEnv(reuse_snapshot=True)

    def on_gen(stats: GenerationStats) -> None:
        if not args.quiet and (stats.generation % 10 == 0 or stats.chambers_solved == 4):
            print(
                f"gen={stats.generation:3d} | solved={stats.chambers_solved}/4 "
                f"| best_fit={stats.best_fitness:7.2f} | mean_fit={stats.mean_fitness:7.2f} "
                f"| size={stats.best_tree_size:2d} | episodes={stats.episodes_total}"
            )

    try:
        trainer = EvolutionTrainer(config=cfg, env=env)
        result = trainer.run(on_generation=on_gen)
    finally:
        if source_env is not None:
            source_env.close()

    print("\n=== Evolution Summary ===")
    print(f"Elapsed:       {result.elapsed_sec:.2f}s ({result.episodes_per_sec:,.0f} episodes/sec)")
    print(f"Episodes:      {result.episodes_evaluated:,}")
    print(f"Train solved:  {result.train_report.chambers_solved}/4 (size={result.train_report.tree_size})")
    for c in result.train_report.per_chamber:
        status = "PASS" if c.passed else "FAIL"
        print(f"  Chamber {c.chamber_id}: {status:4s} | steps={c.steps:3d} | reward={c.reward:7.2f} | milestones={c.milestones}")

    print("\n=== Held-Out Generalization Suite (Chambers 5, 6, 7) ===")
    for ho in result.held_out_reports:
        ho_status = "PASS" if ho.passed else "FAIL"
        print(f"  Chamber {ho.chamber_id}: {ho_status:4s} | steps={ho.steps:3d} | milestones={ho.milestones}")

    dsl = format_bt(result.champion)
    print("\n=== Evolved Champion Behavior Tree (DSL) ===")
    print(dsl)

    if args.save_champion:
        out = Path(args.save_champion)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(dsl + "\n", encoding="utf-8")
        print(f"\nSaved champion DSL to {out}")


if __name__ == "__main__":
    main()
