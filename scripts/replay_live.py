"""Replay any saved Behavior Tree (.bt) DSL file across Chambers 1-5 on mock or live Portal 1.

Logs structured per-tick traces (optional --trace-out JSONL) and prints a concise
chamber-by-chamber report including reset latency, step count, final position, and
stagnation diagnostics for sim-to-Source physics calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portal_ml.bridge.source_env import SourcePortalEnv
from portal_ml.bt.ast import BTNode, parse_bt
from portal_ml.bt.evaluator import BehaviorTreeEvaluator
from portal_ml.bt.reference_trees import get_reference_tree
from portal_ml.sim.mock_env import MockPortalEnv, PortalEnv


def load_tree(bt_path: str) -> tuple[BTNode, str]:
    if not bt_path or bt_path == "reference":
        return get_reference_tree(), "reference"
    p = Path(bt_path)
    return parse_bt(p.read_text(encoding="utf-8")), str(p)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a Behavior Tree DSL across Chambers 1-5.")
    parser.add_argument("--bt", default="champions/seed7_decoupled.bt", help="Path to .bt file or 'reference'")
    parser.add_argument("--backend", choices=["mock", "source"], default="source")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=27500)
    parser.add_argument("--chambers", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--verbose", action="store_true", help="Print every tick")
    parser.add_argument("--trace-out", default="", help="Optional JSONL file to record tick traces")
    args = parser.parse_args()

    tree, label = load_tree(args.bt)
    print(f"Replaying BT: {label} (nodes={tree.size()}) on backend={args.backend}")

    trace_fp = None
    if args.trace_out:
        out_path = Path(args.trace_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        trace_fp = out_path.open("w", encoding="utf-8")

    source_env: SourcePortalEnv | None = None
    env: PortalEnv
    if args.backend == "source":
        source_env = SourcePortalEnv(host=args.host, port=args.port, timeout=15.0)
        source_env.connect()
        assert source_env.ping(), "Bridge ping failed"
        env = source_env
    else:
        env = MockPortalEnv()

    solid_r = 42.0 if args.backend == "source" else 18.0
    floor_r = 28.0 if args.backend == "source" else 18.0
    passed_count = 0
    try:
        for cid in args.chambers:
            ev = BehaviorTreeEvaluator(
                tree, solid_arrival_radius=solid_r, floor_arrival_radius=floor_r
            )
            t0 = time.perf_counter()
            snap = env.reset(cid)
            reset_ms = (time.perf_counter() - t0) * 1000.0

            ep_start = time.perf_counter()
            steps = 0
            last_cmd_str = "none"
            while not snap.done and steps < args.max_steps:
                tick_res = ev.tick(snap)
                cmd = tick_res.command
                last_cmd_str = f"{cmd.kind.value}({cmd.to_dict()})"
                step_res = env.step(cmd)
                snap = step_res.snapshot
                steps += 1

                if trace_fp is not None:
                    trace_fp.write(
                        json.dumps(
                            {
                                "chamber_id": cid,
                                "step": steps,
                                "target": tick_res.target_role,
                                "command": cmd.to_dict(),
                                "player": snap.player.to_dict(),
                                "entities": [e.to_dict() for e in snap.entities],
                                "portals": {
                                    "blue": snap.blue_portal.to_dict(),
                                    "orange": snap.orange_portal.to_dict(),
                                },
                                "done": snap.done,
                                "success": snap.success,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )

                if args.verbose and (steps <= 5 or steps % 10 == 0 or snap.done):
                    ents = ", ".join(
                        f"{e.id}[z={e.zone_id},act={int(e.active)},held={int(e.held)},pos={[round(v,1) for v in e.pos.to_list()]}]"
                        for e in snap.entities
                    )
                    print(
                        f"    t={steps:3d} tgt={str(tick_res.target_role):15s} cmd={cmd.kind.value:12s} "
                        f"p={[round(v,1) for v in snap.player.pos.to_list()]} z={snap.player.zone_id} "
                        f"held={snap.player.held_entity_id} | {ents}"
                    )

            ep_sec = time.perf_counter() - ep_start
            status = "PASS" if snap.success else "FAIL"
            if snap.success:
                passed_count += 1
            print(
                f"  Chamber {cid}: {status:4s} | reset={reset_ms:6.2f}ms | steps={steps:3d} "
                f"| elapsed={ep_sec:5.2f}s | final_pos={[round(v,1) for v in snap.player.pos.to_list()]} "
                f"z={snap.player.zone_id} held={snap.player.held_entity_id} last={last_cmd_str}"
            )
    finally:
        if source_env is not None:
            source_env.close()
        if trace_fp is not None:
            trace_fp.close()

    print(f"SUMMARY: {passed_count}/{len(args.chambers)} chambers passed ({args.backend})")
    return 0 if passed_count == len(args.chambers) else 1


if __name__ == "__main__":
    raise SystemExit(main())
