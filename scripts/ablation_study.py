"""Multi-seed ablation study comparing `decoupled` vs `unconstrained` BT evolution.

Trains strictly on Chambers 1-4 and evaluates zero-shot generalization on held-out Chamber 5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portal_ml.evolution.trainer import EvolutionConfig, EvolutionTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run decoupled vs unconstrained BT evolution ablation.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 13, 21, 42])
    parser.add_argument("--pop", type=int, default=120)
    parser.add_argument("--gens", type=int, default=120)
    args = parser.parse_args()

    print(f"Running ablation across seeds={args.seeds} (pop={args.pop}, gens={args.gens})\n")
    for mode in ("decoupled", "unconstrained"):
        train_4_of_4 = 0
        ch5_passes = 0
        ch5_steps: list[int] = []
        sizes: list[int] = []
        eps_rates: list[float] = []
        print(f"=== Mode: {mode} ===")
        for s in args.seeds:
            cfg = EvolutionConfig(
                population_size=args.pop,
                generations=args.gens,
                mode=mode,
                seed=s,
            )
            res = EvolutionTrainer(cfg).run()
            solved = res.train_report.chambers_solved
            if solved == 4:
                train_4_of_4 += 1
            if res.held_out_report.passed:
                ch5_passes += 1
                ch5_steps.append(res.held_out_report.steps)
            sizes.append(res.train_report.tree_size)
            eps_rates.append(res.episodes_per_sec)
            print(
                f"  seed={s:2d} | train={solved}/4 | ch5={'PASS' if res.held_out_report.passed else 'FAIL'} "
                f"| ch5_steps={res.held_out_report.steps:3d} | gens={len(res.history):3d} "
                f"| size={res.train_report.tree_size:2d} | eps/s={res.episodes_per_sec:6.0f}"
            )
        avg_steps = f"{sum(ch5_steps) / len(ch5_steps):.1f}" if ch5_steps else "n/a"
        avg_size = sum(sizes) / len(sizes)
        avg_eps = sum(eps_rates) / len(eps_rates)
        print(
            f"  SUMMARY ({mode}): train_4/4={train_4_of_4}/{len(args.seeds)} | "
            f"ch5_zero_shot={ch5_passes}/{len(args.seeds)} | avg_ch5_steps={avg_steps} | "
            f"avg_size={avg_size:.1f} | avg_eps/s={avg_eps:.0f}\n"
        )


if __name__ == "__main__":
    main()
