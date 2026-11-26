"""Live smoke test against the C++ Source Engine plugin inside Portal 1.

Usage:
    1. Start Portal 1 via scripts/launch_portal.ps1
    2. Wait until portal_ml_5chambers finishes loading
    3. Run: python scripts/test_live_portal.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portal_ml.bridge.source_env import SourcePortalEnv
from portal_ml.bt.evaluator import BehaviorTreeEvaluator
from portal_ml.bt.reference_trees import get_reference_tree


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test live Portal 1 IPC bridge.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=27500)
    parser.add_argument("--chambers", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--max-steps", type=int, default=80)
    args = parser.parse_args()

    with SourcePortalEnv(host=args.host, port=args.port, timeout=15.0) as env:
        assert env.ping(), "Bridge ping failed"
        print(f"Connected to Portal-ML plugin at {args.host}:{args.port} (pong OK)")

        tree = get_reference_tree()
        for cid in args.chambers:
            t0 = time.perf_counter()
            snap = env.reset(cid)
            reset_ms = (time.perf_counter() - t0) * 1000.0
            ev = BehaviorTreeEvaluator(tree)
            steps = 0
            while not snap.done and steps < args.max_steps:
                cmd = ev.tick(snap).command
                res = env.step(cmd)
                snap = res.snapshot
                steps += 1
            status = "PASS" if snap.success else "INCOMPLETE"
            print(
                f"  Chamber {cid}: {status:10s} | reset={reset_ms:5.2f}ms "
                f"| steps={steps:3d} | player={snap.player.pos.to_list()} zone={snap.player.zone_id}"
            )


if __name__ == "__main__":
    main()
