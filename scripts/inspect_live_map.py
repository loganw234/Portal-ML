"""Connect to a live Portal 1 instance and inspect generic BSP/edict Blackboard extraction.

Works on both `portal_ml_5chambers` and unmodified Valve Portal 1 campaign maps
such as `testchmb_a_02` or `testchmb_a_00`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portal_ml.bridge.source_env import SourcePortalEnv
from portal_ml.motor.primitives import MoveVector, NoOp


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect live Blackboard extraction on any Portal 1 map.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=27500)
    parser.add_argument("--chamber", type=int, default=0, help="Chamber ID (0 for generic campaign maps)")
    parser.add_argument("--step-forward", type=int, default=0, help="Optional MoveVector forward steps to test motion")
    parser.add_argument("--dump-json", default="", help="Optional path to save snapshot JSON")
    args = parser.parse_args()

    with SourcePortalEnv(host=args.host, port=args.port, timeout=15.0) as env:
        assert env.ping(), "Bridge ping failed"
        info = env.map_info()
        print("MAP_INFO:", json.dumps(info))
        snap = env.reset(args.chamber)
        print(
            f"INITIAL SNAPSHOT: tick={snap.tick} chamber={snap.chamber_id} "
            f"player_pos={snap.player.pos.to_list()} yaw={snap.player.yaw:.1f} "
            f"entities={len(snap.entities)} portal_walls={len(snap.portal_walls)}"
        )
        for e in snap.entities:
            print(f"  ENT  {e.kind:15s} id={e.id:34s} pos={e.pos.to_list()} active={e.active}")
        print(
            f"  PORTALS: blue(active={snap.blue_portal.active}, pos={snap.blue_portal.pos.to_list()}), "
            f"orange(active={snap.orange_portal.active}, pos={snap.orange_portal.pos.to_list()})"
        )
        print(f"  NEAREST PORTALABLE SURFACES (showing first 8 of {len(snap.portal_walls)}):")
        for w in snap.portal_walls[:8]:
            print(f"    {w.id:12s} pos={w.pos.to_list()} n={w.normal.to_list()} size={list(w.size)}")

        for i in range(args.step_forward):
            res = env.step(MoveVector(forward=1.0, strafe=0.0, yaw_delta=0.0))
            snap = res.snapshot
            print(
                f"  STEP {i + 1}: pos={snap.player.pos.to_list()} vel={snap.player.vel.to_list()} "
                f"reward={res.reward:.2f} milestones={res.milestones}"
            )
        if args.step_forward > 0:
            env.step(NoOp())

        if args.dump_json:
            out = Path(args.dump_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"map_info": info, "snapshot": snap.to_dict()}, indent=2) + "\n", encoding="utf-8")
            print(f"Saved snapshot to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
