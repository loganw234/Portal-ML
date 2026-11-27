"""Multi-seed ablation study comparing `decoupled` vs `unconstrained` BT evolution.

Trains strictly on Chambers 1-4 and evaluates zero-shot generalization on held-out
Chambers 5, 6, and 7.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portal_ml.evolution.fitness import HELD_OUT_CHAMBERS
from portal_ml.evolution.trainer import EvolutionConfig, EvolutionTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run decoupled vs unconstrained BT evolution ablation.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 13, 21, 42])
    parser.add_argument("--pop", type=int, default=120)
    parser.add_argument("--gens", type=int, default=120)
    parser.add_argument(
        "--no-causal-aug",
        action="store_true",
        help="Disable training-only causal invariance augmentations on Chambers 3 & 4.",
    )
    args = parser.parse_args()
    use_aug = not args.no_causal_aug

    print(
        f"Running ablation across seeds={args.seeds} "
        f"(pop={args.pop}, gens={args.gens}, causal_aug={use_aug})\n"
    )
    for mode in ("decoupled", "unconstrained"):
        train_4_of_4 = 0
        ho_passes = {cid: 0 for cid in HELD_OUT_CHAMBERS}
        ho_all_pass = 0
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
                causal_augmentation=use_aug,
            )
            res = EvolutionTrainer(cfg).run()
            solved = res.train_report.chambers_solved
            if solved == 4:
                train_4_of_4 += 1
            by_cid = {r.chamber_id: r for r in res.held_out_reports}
            for cid in HELD_OUT_CHAMBERS:
                if by_cid[cid].passed:
                    ho_passes[cid] += 1
            if all(by_cid[cid].passed for cid in HELD_OUT_CHAMBERS):
                ho_all_pass += 1
            if res.held_out_report.passed:
                ch5_steps.append(res.held_out_report.steps)
            sizes.append(res.train_report.tree_size)
            eps_rates.append(res.episodes_per_sec)
            ho_str = " ".join(
                f"ch{cid}={'PASS' if by_cid[cid].passed else 'FAIL'}({by_cid[cid].steps:3d})"
                for cid in HELD_OUT_CHAMBERS
            )
            print(
                f"  seed={s:2d} | train={solved}/4 | {ho_str} "
                f"| gens={len(res.history):3d} | size={res.train_report.tree_size:2d} "
                f"| eps/s={res.episodes_per_sec:6.0f}"
            )
        avg_steps = f"{sum(ch5_steps) / len(ch5_steps):.1f}" if ch5_steps else "n/a"
        avg_size = sum(sizes) / len(sizes)
        avg_eps = sum(eps_rates) / len(eps_rates)
        ho_sum = " ".join(
            f"ch{cid}={ho_passes[cid]}/{len(args.seeds)}" for cid in HELD_OUT_CHAMBERS
        )
        print(
            f"  SUMMARY ({mode}): train_4/4={train_4_of_4}/{len(args.seeds)} | "
            f"{ho_sum} | all_ho={ho_all_pass}/{len(args.seeds)} | "
            f"avg_ch5_steps={avg_steps} | avg_size={avg_size:.1f} | avg_eps/s={avg_eps:.0f}\n"
        )


if __name__ == "__main__":
    main()
