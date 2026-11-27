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
from portal_ml.motor.hybrid import HybridMotorController
from portal_ml.motor.neural import (
    NeuralAimPolicy,
    NeuralMoveToPolicy,
    NeuralParkourPolicy,
)
from portal_ml.planner import (
    FourLayerAgent,
    NullStrategicPlanner,
    OpenAICompatibleLLMPlanner,
    RuleBasedStrategicPlanner,
    StrategicPlanner,
)
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
    parser.add_argument("--chambers", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--verbose", action="store_true", help="Print every tick")
    parser.add_argument("--trace-out", default="", help="Optional JSONL file to record tick traces")
    parser.add_argument("--neural-moveto", default="", help="Optional Phase-2 precision MoveTo MLP JSON checkpoint")
    parser.add_argument("--neural-parkour", default="", help="Optional Milestone-7 sensor-equipped Parkour MLP JSON checkpoint")
    parser.add_argument(
        "--skill-mode",
        choices=["auto", "precision", "parkour"],
        default="auto",
        help="Multi-skill routing mode when both --neural-moveto and --neural-parkour are provided",
    )
    parser.add_argument("--neural-aim", default="", help="Optional Phase-2 ShootPortal Aim MLP JSON checkpoint")
    parser.add_argument(
        "--planner",
        choices=["none", "rule", "llm"],
        default="none",
        help="Layer-4 strategic planner mode (none=3-layer, rule=symbolic, llm=OpenAI-compatible endpoint)",
    )
    parser.add_argument(
        "--llm-url",
        default="http://127.0.0.1:1234/v1/chat/completions",
        help="OpenAI-compatible chat completions URL (LM Studio default: port 1234)",
    )
    parser.add_argument(
        "--llm-model",
        default="phi-4-mini-instruct",
        help="Model identifier served by LM Studio / Ollama / llama-server",
    )
    parser.add_argument("--llm-timeout", type=float, default=8.0, help="Layer-4 LLM HTTP timeout in seconds")
    args = parser.parse_args()

    tree, label = load_tree(args.bt)
    moveto_pol: NeuralMoveToPolicy | NeuralParkourPolicy | None = None
    parkour_pol: NeuralParkourPolicy | None = None
    if args.neural_moveto and args.neural_parkour:
        moveto_pol = NeuralMoveToPolicy.load(args.neural_moveto)
        parkour_pol = NeuralParkourPolicy.load(args.neural_parkour)
    elif args.neural_parkour:
        moveto_pol = NeuralParkourPolicy.load(args.neural_parkour)
    elif args.neural_moveto:
        moveto_pol = NeuralMoveToPolicy.load(args.neural_moveto)
    aim_pol = NeuralAimPolicy.load(args.neural_aim) if args.neural_aim else None
    hybrid = HybridMotorController(
        moveto_policy=moveto_pol,
        aim_policy=aim_pol,
        parkour_policy=parkour_pol,
        skill_mode=args.skill_mode,
    )
    motor_desc = []
    if moveto_pol is not None:
        mtag = "neural_parkour" if isinstance(moveto_pol, NeuralParkourPolicy) else "neural_moveto"
        motor_desc.append(f"{mtag}({len(moveto_pol.net.weights)}w)")
    if parkour_pol is not None:
        motor_desc.append(f"parkour({len(parkour_pol.net.weights)}w,mode={args.skill_mode})")
    if aim_pol is not None:
        motor_desc.append(f"neural_aim({len(aim_pol.net.weights)}w)")
    motor_str = "+".join(motor_desc) if motor_desc else "scripted"
    planner_desc = args.planner if args.planner != "llm" else f"llm:{args.llm_model}"
    print(
        f"Replaying BT: {label} (nodes={tree.size()}) on backend={args.backend} "
        f"motor={motor_str} planner={planner_desc}"
    )

    trace_fp = None
    if args.trace_out:
        out_path = Path(args.trace_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        trace_fp = out_path.open("w", encoding="utf-8")

    source_env: SourcePortalEnv | None = None
    env: PortalEnv
    allow_orange = True
    chambers_to_run = list(args.chambers)
    if args.backend == "source":
        source_env = SourcePortalEnv(host=args.host, port=args.port, timeout=15.0)
        source_env.connect()
        assert source_env.ping(), "Bridge ping failed"
        info = source_env.map_info()
        if info.get("generic_mode", False):
            allow_orange = bool(info.get("can_fire_orange", True))
            chambers_to_run = [0]
            print(
                f"Generic campaign map detected: {info.get('map_name')} "
                f"(entities={info.get('discovered_entities')}, "
                f"surfaces={info.get('portal_surfaces')}, "
                f"can_fire_orange={allow_orange})"
            )
        env = source_env
    else:
        env = MockPortalEnv()

    solid_r = 46.0 if args.backend == "source" else 18.0
    floor_r = 28.0 if args.backend == "source" else 18.0
    interact_r = 48.0 if args.backend == "source" else 44.0
    passed_count = 0
    try:
        for cid in chambers_to_run:
            ev = BehaviorTreeEvaluator(
                tree,
                solid_arrival_radius=solid_r,
                floor_arrival_radius=floor_r,
                interact_radius=interact_r,
                allow_orange_portal=allow_orange,
            )
            planner: StrategicPlanner
            if args.planner == "rule":
                planner = RuleBasedStrategicPlanner()
            elif args.planner == "llm":
                planner = OpenAICompatibleLLMPlanner(
                    base_url=args.llm_url,
                    model=args.llm_model,
                    timeout_s=args.llm_timeout,
                )
            else:
                planner = NullStrategicPlanner()
            agent = FourLayerAgent(ev, planner=planner, hybrid_motor=hybrid)

            t0 = time.perf_counter()
            snap = env.reset(cid)
            reset_ms = (time.perf_counter() - t0) * 1000.0
            agent.reset(snap)

            ep_start = time.perf_counter()
            steps = 0
            ep_reward = 0.0
            ep_milestones: list[str] = []
            last_milestones: list[str] = []
            while not snap.done and steps < args.max_steps:
                cmd, tick_res = agent.act(snap, last_milestones)
                step_res = env.step(cmd)
                snap = step_res.snapshot
                ep_reward += step_res.reward
                last_milestones = list(step_res.milestones)
                if last_milestones:
                    ep_milestones.extend(last_milestones)
                steps += 1

                if trace_fp is not None:
                    trace_fp.write(
                        json.dumps(
                            {
                                "chamber_id": cid,
                                "step": steps,
                                "target": tick_res.target_role,
                                "subgoal": (
                                    agent.current_directive.subgoal
                                    if agent.current_directive is not None
                                    else None
                                ),
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
            l4_info = (
                f" | l4_calls={agent.planner_calls}({agent.total_planner_ms:.1f}ms)"
                if args.planner != "none"
                else ""
            )
            print(
                f"  Chamber {cid}: {status:4s} | reset={reset_ms:6.2f}ms | steps={steps:3d} "
                f"| elapsed={ep_sec:5.2f}s | reward={ep_reward:6.2f}{l4_info} | ms={ep_milestones} "
                f"| final_pos={[round(v,1) for v in snap.player.pos.to_list()]} z={snap.player.zone_id}"
            )
    finally:
        if source_env is not None:
            source_env.close()
        if trace_fp is not None:
            trace_fp.close()

    print(f"SUMMARY: {passed_count}/{len(chambers_to_run)} chambers passed ({args.backend})")
    return 0 if passed_count == len(chambers_to_run) else 1


if __name__ == "__main__":
    raise SystemExit(main())
