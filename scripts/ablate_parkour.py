"""Controlled motor-skill ablation: Scripted vs Blind NeuralMoveTo vs Sensor NeuralParkour.

Evaluates all three Layer-3 motor controllers underneath the exact frozen Layer-2
champion BT (`champions/seed7_decoupled.bt`) across:
- Standard Chambers 1-7 (regression check)
- Multi-Stage Chambers M1-M3 with Layer-4 symbolic planner (composition check)
- Held-out Physical-Skill Chambers P1-P3 (partition slalom, hurdle & acid trench, gap jump)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portal_ml.bt.ast import parse_bt
from portal_ml.bt.evaluator import BehaviorTreeEvaluator
from portal_ml.motor.hybrid import HybridMotorController
from portal_ml.motor.neural import (
    NeuralAimPolicy,
    NeuralMoveToPolicy,
    NeuralParkourPolicy,
)
from portal_ml.motor.parkour_benchmarks import PARKOUR_BENCHMARKS
from portal_ml.planner import (
    MULTI_STAGE_BENCHMARKS,
    FourLayerAgent,
    NullStrategicPlanner,
    RuleBasedStrategicPlanner,
)
from portal_ml.sim.chambers import CHAMBERS
from portal_ml.sim.mock_env import MockPortalEnv


def _eval_motor(
    label: str,
    tree: Any,
    hybrid: HybridMotorController | None,
) -> dict[str, Any]:
    env = MockPortalEnv()
    std_rows = []
    for cid in sorted(CHAMBERS.keys()):
        agent = FourLayerAgent(
            BehaviorTreeEvaluator(tree), NullStrategicPlanner(), hybrid_motor=hybrid
        )
        m = agent.run_episode(env, chamber_id=cid)
        std_rows.append(m.to_dict())

    ms_rows = []
    for key, spec in MULTI_STAGE_BENCHMARKS.items():
        agent = FourLayerAgent(
            BehaviorTreeEvaluator(tree), RuleBasedStrategicPlanner(), hybrid_motor=hybrid
        )
        m = agent.run_episode(env, spec=spec)
        ms_rows.append(m.to_dict())

    pk_rows = []
    for key, spec in PARKOUR_BENCHMARKS.items():
        agent = FourLayerAgent(
            BehaviorTreeEvaluator(tree), NullStrategicPlanner(), hybrid_motor=hybrid
        )
        m = agent.run_episode(env, spec=spec)
        row = m.to_dict()
        row["final_health"] = round(env._player.health, 1)
        pk_rows.append(row)

    summary = {
        "label": label,
        "standard_passed": sum(1 for r in std_rows if r["success"]),
        "standard_total": len(std_rows),
        "multistage_passed": sum(1 for r in ms_rows if r["success"]),
        "multistage_total": len(ms_rows),
        "parkour_passed": sum(1 for r in pk_rows if r["success"]),
        "parkour_total": len(pk_rows),
        "standard": std_rows,
        "multistage": ms_rows,
        "parkour": pk_rows,
    }
    print(
        f"\n=== {label} | Standard: {summary['standard_passed']}/{summary['standard_total']} "
        f"| Multi-Stage (L4): {summary['multistage_passed']}/{summary['multistage_total']} "
        f"| Parkour (P1-P3): {summary['parkour_passed']}/{summary['parkour_total']} ==="
    )
    for row in pk_rows:
        st = "PASS" if row["success"] else "FAIL"
        print(
            f"  [parkour   ] ch={row['chamber_id']} ({row['chamber_name'][:28]:28s}) "
            f"{st:4s} | steps={row['steps']:3d} | hp={row['final_health']:5.1f} | ms={row['milestones']}"
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Ablate Scripted vs NeuralMoveTo vs NeuralParkour.")
    parser.add_argument("--bt", default="champions/seed7_decoupled.bt")
    parser.add_argument("--moveto", default="champions/moveto_mlp.json")
    parser.add_argument("--aim", default="champions/aim_mlp.json")
    parser.add_argument("--parkour", default="champions/parkour_mlp.json")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    tree = parse_bt(Path(args.bt).read_text(encoding="utf-8"))
    aim = NeuralAimPolicy.load(args.aim)
    moveto = NeuralMoveToPolicy.load(args.moveto)
    parkour = NeuralParkourPolicy.load(args.parkour)

    configs: list[tuple[str, HybridMotorController | None]] = [
        ("Phase 1 Scripted MoveTo", None),
        (
            f"Phase 2B Blind NeuralMoveTo ({len(moveto.net.weights)}w+{len(aim.net.weights)}w)",
            HybridMotorController(moveto_policy=moveto, aim_policy=aim),
        ),
        (
            f"Milestone 7 Sensor NeuralParkour ({len(parkour.net.weights)}w+{len(aim.net.weights)}w)",
            HybridMotorController(moveto_policy=parkour, aim_policy=aim),
        ),
    ]

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bt_path": args.bt,
        "results": [],
    }
    for label, hybrid in configs:
        report["results"].append(_eval_motor(label, tree, hybrid))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved parkour ablation -> {out_path}")

    pk = report["results"][-1]
    ok = (
        pk["standard_passed"] == pk["standard_total"]
        and pk["multistage_passed"] == pk["multistage_total"]
        and pk["parkour_passed"] == pk["parkour_total"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
