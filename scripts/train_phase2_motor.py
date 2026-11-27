"""Train Phase-2 neuroevolved MLP motor primitives and benchmark under a FROZEN BT.

Evolves:
  1. NeuralMoveToPolicy (6 -> 8 -> 3, 83 weights) -> champions/moveto_mlp.json
  2. NeuralAimPolicy    (4 -> 6 -> 2, 44 weights) -> champions/aim_mlp.json

Then evaluates the frozen evolved Behavior Tree (`champions/seed7_decoupled.bt`) across
Chambers 1-5 in three progressive swap stages:
  - Phase 1:  All deterministic scripted primitives
  - Phase 2A: Neural MoveTo + scripted ShootPortal/Grab/Drop/Press
  - Phase 2B: Neural MoveTo + Neural ShootPortal Aim + scripted Grab/Drop/Press
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portal_ml.bt.ast import parse_bt
from portal_ml.bt.evaluator import BehaviorTreeEvaluator
from portal_ml.evolution.motor_trainer import (
    MotorEvoConfig,
    evolve_aim_policy,
    evolve_moveto_policy,
)
from portal_ml.motor.hybrid import HybridMotorController
from portal_ml.motor.neural import NeuralAimPolicy, NeuralMoveToPolicy
from portal_ml.sim.mock_env import MockPortalEnv


def eval_under_frozen_bt(
    bt_path: Path, hybrid: HybridMotorController, label: str
) -> int:
    tree = parse_bt(bt_path.read_text(encoding="utf-8"))
    env = MockPortalEnv()
    passed = 0
    details: list[str] = []
    for cid in (1, 2, 3, 4, 5):
        ev = BehaviorTreeEvaluator(tree)
        snap = env.reset(cid)
        steps = 0
        while not snap.done:
            sym_cmd = ev.tick(snap).command
            low_cmd = hybrid.lower(snap, sym_cmd)
            snap = env.step(low_cmd).snapshot
            steps += 1
        if snap.success:
            passed += 1
        details.append(f"Ch{cid}:{'PASS' if snap.success else 'FAIL'}({steps})")
    print(f"  {label:38s} | {passed}/5 | " + "  ".join(details))
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Neuroevolve Phase-2 motor MLPs.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gens", type=int, default=120)
    parser.add_argument("--pop", type=int, default=80)
    parser.add_argument("--bt", default="champions/seed7_decoupled.bt")
    parser.add_argument("--moveto-out", default="champions/moveto_mlp.json")
    parser.add_argument("--aim-out", default="champions/aim_mlp.json")
    args = parser.parse_args()

    print(f"=== Phase 2 Neuroevolution (seed={args.seed}, pop={args.pop}, gens={args.gens}) ===")
    mt_res = evolve_moveto_policy(
        MotorEvoConfig(population_size=args.pop, generations=args.gens, seed=args.seed)
    )
    mt_path = mt_res.net.save(args.moveto_out)
    print(
        f"MoveTo MLP ({len(mt_res.net.weights)} weights): val_success={mt_res.success_rate*100:.1f}% "
        f"| val_fit={mt_res.best_fitness:.2f} | elapsed={mt_res.elapsed_sec:.2f}s -> {mt_path}"
    )

    aim_res = evolve_aim_policy(
        MotorEvoConfig(
            population_size=args.pop, generations=max(80, args.gens), seed=args.seed + 100
        )
    )
    aim_path = aim_res.net.save(args.aim_out)
    print(
        f"Aim    MLP ({len(aim_res.net.weights)} weights): val_success={aim_res.success_rate*100:.1f}% "
        f"| val_fit={aim_res.best_fitness:.2f} | elapsed={aim_res.elapsed_sec:.2f}s -> {aim_path}"
    )

    bt_path = Path(args.bt)
    moveto_pol = NeuralMoveToPolicy(mt_res.net)
    aim_pol = NeuralAimPolicy(aim_res.net)

    print(f"\n=== Progressive Primitive Swap Under FROZEN BT ({bt_path}) ===")
    p1 = eval_under_frozen_bt(bt_path, HybridMotorController(), "Phase 1  (all scripted)")
    p2a = eval_under_frozen_bt(
        bt_path, HybridMotorController(moveto_policy=moveto_pol), "Phase 2A (neural MoveTo)"
    )
    p2b = eval_under_frozen_bt(
        bt_path,
        HybridMotorController(moveto_policy=moveto_pol, aim_policy=aim_pol),
        "Phase 2B (neural MoveTo + neural Aim)",
    )
    return 0 if (p1 == 5 and p2a == 5 and p2b == 5) else 1


if __name__ == "__main__":
    raise SystemExit(main())
