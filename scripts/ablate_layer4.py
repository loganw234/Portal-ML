"""Controlled 3-layer vs 4-layer ablation across standard (1-7) and multi-stage (M1-M3) chambers.

Evaluates the exact frozen Layer-2 champion BT (`champions/seed7_decoupled.bt`) under:
- 3-layer baseline (`NullStrategicPlanner`)
- 4-layer symbolic baseline (`RuleBasedStrategicPlanner`)
- Optional 4-layer local LLM (`OpenAICompatibleLLMPlanner` against LM Studio / Ollama / llama-server)

across both Phase 1 (scripted motor) and Phase 2B (neuroevolved MoveTo + Aim MLPs).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portal_ml.bt.ast import BTNode, parse_bt
from portal_ml.bt.evaluator import BehaviorTreeEvaluator
from portal_ml.motor.hybrid import HybridMotorController
from portal_ml.motor.neural import NeuralAimPolicy, NeuralMoveToPolicy
from portal_ml.planner import (
    MULTI_STAGE_BENCHMARKS,
    EpisodeMetrics,
    FourLayerAgent,
    NullStrategicPlanner,
    OpenAICompatibleLLMPlanner,
    RuleBasedStrategicPlanner,
    StrategicPlanner,
)
from portal_ml.sim.chambers import CHAMBERS, ChamberSpec
from portal_ml.sim.mock_env import MockPortalEnv


def _run_suite(
    tree: BTNode,
    planner_factory: Any,
    hybrid: HybridMotorController | None,
    standard_ids: list[int],
    multistage_keys: list[str],
) -> dict[str, Any]:
    env = MockPortalEnv()
    std_results: list[EpisodeMetrics] = []
    for cid in standard_ids:
        ev = BehaviorTreeEvaluator(tree)
        agent = FourLayerAgent(ev, planner_factory(), hybrid_motor=hybrid)
        std_results.append(agent.run_episode(env, chamber_id=cid))

    ms_results: list[EpisodeMetrics] = []
    for key in multistage_keys:
        spec: ChamberSpec = MULTI_STAGE_BENCHMARKS[key]
        ev = BehaviorTreeEvaluator(tree)
        agent = FourLayerAgent(ev, planner_factory(), hybrid_motor=hybrid)
        ms_results.append(agent.run_episode(env, chamber_id=spec.chamber_id, spec=spec))

    return {
        "standard": [r.to_dict() for r in std_results],
        "multistage": [r.to_dict() for r in ms_results],
        "standard_passed": sum(1 for r in std_results if r.success),
        "standard_total": len(std_results),
        "multistage_passed": sum(1 for r in ms_results if r.success),
        "multistage_total": len(ms_results),
        "total_planner_calls": sum(r.planner_calls for r in std_results + ms_results),
        "total_fallbacks": sum(r.fallback_count for r in std_results + ms_results),
        "total_validation_warnings": sum(
            r.validation_warnings for r in std_results + ms_results
        ),
    }


def _print_suite_table(label: str, data: dict[str, Any]) -> None:
    print(
        f"\n=== {label} | Standard: {data['standard_passed']}/{data['standard_total']} "
        f"| Multi-Stage: {data['multistage_passed']}/{data['multistage_total']} "
        f"| Calls: {data['total_planner_calls']} | Fallbacks: {data['total_fallbacks']} "
        f"| Warnings: {data['total_validation_warnings']} ==="
    )
    for bucket_name in ("standard", "multistage"):
        for row in data[bucket_name]:
            status = "PASS" if row["success"] else "FAIL"
            print(
                f"  [{bucket_name:10s}] ch={row['chamber_id']:3d} ({row['chamber_name'][:28]:28s}) "
                f"{status:4s} | steps={row['steps']:3d} | rew={row['reward']:6.1f} "
                f"| calls={row['planner_calls']:2d} ({row['total_planner_ms']:5.2f}ms) "
                f"| switches={row['role_switches']:2d} | triggers={row['planner_triggers']}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run controlled 3-layer vs 4-layer ablations on standard + multi-stage suites."
    )
    parser.add_argument("--bt", default="champions/seed7_decoupled.bt")
    parser.add_argument("--neural-moveto", default="champions/moveto_mlp.json")
    parser.add_argument("--neural-aim", default="champions/aim_mlp.json")
    parser.add_argument(
        "--skip-neural",
        action="store_true",
        help="Run only Phase-1 scripted motor control",
    )
    parser.add_argument(
        "--llm-url",
        default="",
        help="Optional OpenAI-compatible chat completions URL (e.g. http://127.0.0.1:1234/v1/chat/completions)",
    )
    parser.add_argument("--llm-model", default="phi-4-mini-instruct")
    parser.add_argument("--llm-timeout", type=float, default=15.0)
    parser.add_argument("--out", default="", help="Optional JSON output report path")
    args = parser.parse_args()

    tree = parse_bt(Path(args.bt).read_text(encoding="utf-8"))
    standard_ids = sorted(CHAMBERS.keys())
    multistage_keys = list(MULTI_STAGE_BENCHMARKS.keys())

    motor_configs: list[tuple[str, HybridMotorController | None]] = [("scripted", None)]
    if not args.skip_neural:
        moveto_path = Path(args.neural_moveto)
        aim_path = Path(args.neural_aim)
        if moveto_path.exists() and aim_path.exists():
            hybrid = HybridMotorController(
                moveto_policy=NeuralMoveToPolicy.load(moveto_path),
                aim_policy=NeuralAimPolicy.load(aim_path),
            )
            motor_configs.append(("neural_2b", hybrid))

    planner_configs: list[tuple[str, Any]] = [
        ("3-layer (none)", lambda: NullStrategicPlanner()),
        ("4-layer (rule)", lambda: RuleBasedStrategicPlanner()),
    ]
    if args.llm_url:
        planner_configs.append(
            (
                f"4-layer (llm:{args.llm_model})",
                lambda: OpenAICompatibleLLMPlanner(
                    base_url=args.llm_url,
                    model=args.llm_model,
                    timeout_s=args.llm_timeout,
                ),
            )
        )

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bt_path": args.bt,
        "bt_nodes": tree.size(),
        "results": {},
    }

    all_rule_ok = True
    for motor_name, hybrid in motor_configs:
        report["results"][motor_name] = {}
        for planner_label, factory in planner_configs:
            res = _run_suite(tree, factory, hybrid, standard_ids, multistage_keys)
            report["results"][motor_name][planner_label] = res
            _print_suite_table(f"{planner_label} [{motor_name}]", res)
            if planner_label == "4-layer (rule)":
                if (
                    res["standard_passed"] != res["standard_total"]
                    or res["multistage_passed"] != res["multistage_total"]
                ):
                    all_rule_ok = False

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved ablation report -> {out_path}")

    return 0 if all_rule_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
