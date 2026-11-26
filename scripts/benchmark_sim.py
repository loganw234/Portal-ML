"""Benchmark mock 3D simulator throughput (episodes/sec and ticks/sec)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portal_ml.bt.evaluator import BehaviorTreeEvaluator
from portal_ml.bt.reference_trees import get_reference_tree
from portal_ml.sim.mock_env import MockPortalEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MockPortalEnv throughput.")
    parser.add_argument("--episodes", type=int, default=2000, help="Total episodes across Chambers 1-5")
    args = parser.parse_args()

    tree = get_reference_tree()
    evaluator = BehaviorTreeEvaluator(tree)
    env = MockPortalEnv(reuse_snapshot=True)

    total_ticks = 0
    passed = 0
    t0 = time.perf_counter()
    for i in range(args.episodes):
        cid = (i % 5) + 1
        evaluator.reset()
        snap = env.reset(cid)
        while not snap.done:
            cmd = evaluator.tick(snap).command
            res = env.step(cmd)
            snap = res.snapshot
            total_ticks += 1
        if snap.success:
            passed += 1
    elapsed = time.perf_counter() - t0

    print(f"Episodes:   {args.episodes:,} (passed={passed:,}/{args.episodes:,})")
    print(f"Ticks:      {total_ticks:,}")
    print(f"Elapsed:    {elapsed:.3f}s")
    print(f"Throughput: {args.episodes / elapsed:,.0f} episodes/sec | {total_ticks / elapsed:,.0f} ticks/sec")


if __name__ == "__main__":
    main()
