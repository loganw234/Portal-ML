"""Unit and loopback-HTTP integration tests for the Layer-4 Strategic Planner."""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from portal_ml.bt.ast import parse_bt
from portal_ml.bt.evaluator import BehaviorTreeEvaluator
from portal_ml.motor.hybrid import HybridMotorController
from portal_ml.motor.neural import NeuralAimPolicy, NeuralMoveToPolicy
from portal_ml.planner import (
    MULTI_STAGE_BENCHMARKS,
    EpisodicMemory,
    FourLayerAgent,
    NullStrategicPlanner,
    OpenAICompatibleLLMPlanner,
    RuleBasedStrategicPlanner,
    StrategicDirective,
    TriggerEvent,
    build_directive_json_schema,
    build_symbolic_digest,
    strip_reasoning_and_extract_json,
)
from portal_ml.sim.chambers import CHAMBERS
from portal_ml.sim.mock_env import MockPortalEnv

ROOT = Path(__file__).resolve().parents[1]
CHAMPION_BT_PATH = ROOT / "champions" / "seed7_decoupled.bt"
MOVETO_MLP_PATH = ROOT / "champions" / "moveto_mlp.json"
AIM_MLP_PATH = ROOT / "champions" / "aim_mlp.json"


class _LoopbackOpenAIServer:
    """In-process mock OpenAI /v1/chat/completions server for deterministic testing."""

    def __init__(self, handler_fn: Any) -> None:
        self._handler_fn = handler_fn
        self.received_bodies: list[dict[str, Any]] = []
        parent = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8")
                body = json.loads(raw)
                parent.received_bodies.append(body)
                status, payload = parent._handler_fn(body, len(parent.received_bodies))
                out = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

        self._httpd = HTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self._httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/v1/chat/completions"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> _LoopbackOpenAIServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2.0)


class TestEpisodicMemoryAndSchema(unittest.TestCase):
    def test_memory_capacity_and_deduplication(self) -> None:
        mem = EpisodicMemory(max_entries=8)
        for i in range(12):
            mem.record(tick=i, event="step_note", entity_ids=(f"e{i}",), note=f"n{i}")
        self.assertEqual(len(mem), 8)
        self.assertEqual(mem.entries[0].entity_ids, ("e4",))
        self.assertEqual(mem.entries[-1].entity_ids, ("e11",))

        # Deduplicate existing (event, entity_ids) by refreshing in place
        mem.record(tick=99, event="step_note", entity_ids=("e5",), note="updated")
        self.assertEqual(len(mem), 8)
        self.assertEqual(mem.entries[-1].tick, 99)
        self.assertEqual(mem.entries[-1].note, "updated")

        mem.record(tick=100, event="cube_fizzled", entity_ids=("m2_cube_0",), note="fizzled")
        self.assertTrue(mem.has_event("cube_fizzled", "m2_cube_0"))
        self.assertEqual(mem.entities_with_event("cube_fizzled"), {"m2_cube_0"})
        self.assertIn("cube_fizzled (m2_cube_0)", mem.format_for_prompt())

    def test_directive_validation_rejects_hallucinations_and_kind_mismatch(self) -> None:
        env = MockPortalEnv()
        snap = env.reset(spec=MULTI_STAGE_BENCHMARKS["M1"])

        raw = StrategicDirective(
            subgoal="  test   directive  ",
            role_overrides={
                "cube": "m1_cube_0",  # valid
                "floor_button": "m1_cube_1",  # kind mismatch!
                "pedestal_button": None,  # valid explicit mask
                "exit": "nonexistent_door_99",  # hallucinated ID!
                "bogus_role": "m1_cube_0",  # unknown role!
            },
            min_dwell_steps=999,
        )
        clean, warnings = raw.validate_against_snapshot(snap)
        self.assertEqual(clean.subgoal, "test directive")
        self.assertEqual(clean.min_dwell_steps, 60)
        self.assertEqual(clean.role_overrides, {"cube": "m1_cube_0", "pedestal_button": None})
        self.assertEqual(len(warnings), 3)
        self.assertTrue(any("kind_mismatch" in w for w in warnings))
        self.assertTrue(any("hallucinated_entity" in w for w in warnings))
        self.assertTrue(any("unknown_role" in w for w in warnings))

    def test_dynamic_json_schema_and_compact_digest(self) -> None:
        env = MockPortalEnv()
        snap = env.reset(spec=MULTI_STAGE_BENCHMARKS["M3"])
        schema = build_directive_json_schema(snap)
        props = schema["schema"]["properties"]

        self.assertEqual(
            set(props["action"]["enum"]),
            {"place_cube_on_button", "press_pedestal"},
        )
        self.assertEqual(set(props["cube_id"]["enum"]), {"m3_cube_0", "m3_cube_1", None})
        self.assertEqual(
            set(props["button_id"]["enum"]),
            {"m3_floor_btn_0", "m3_floor_btn_1", None},
        )
        self.assertEqual(
            set(props["pedestal_id"]["enum"]), {"m3_pedestal_0", None}
        )
        self.assertEqual(set(props["exit_id"]["enum"]), {"m3_exit_0"})

        mem = EpisodicMemory()
        mem.record(12, "button_activated", ("m3_floor_btn_0",), "first button done")
        digest = build_symbolic_digest(snap, mem, TriggerEvent.SUBGOAL_COMPLETED)
        self.assertLess(len(digest), 1400)
        self.assertIn("m3_cube_1", digest)
        self.assertIn("button_activated (m3_floor_btn_0)", digest)

    def test_reasoning_tag_and_fence_stripping(self) -> None:
        raw = (
            "<think>Cube 0 fizzled behind barrier, so switch to m2_cube_1.</think>\n"
            "```json\n"
            '{"subgoal": "use m2_cube_1", "role_overrides": {"cube": "m2_cube_1"}, '
            '"memory_note": null, "min_dwell_steps": 12}\n'
            "```"
        )
        parsed = strip_reasoning_and_extract_json(raw)
        self.assertEqual(parsed["subgoal"], "use m2_cube_1")
        self.assertEqual(parsed["role_overrides"]["cube"], "m2_cube_1")


class TestOpenAICompatibleLLMPlanner(unittest.TestCase):
    def test_loopback_http_end_to_end_solves_m2_with_think_tags(self) -> None:
        # Mock an LFM2.5-Thinking / Phi-4-mini-reasoning server that switches from
        # m2_cube_0 to m2_cube_1 only after seeing `cube_fizzled (m2_cube_0)` in the digest!
        def handle(body: dict[str, Any], _call_idx: int) -> tuple[int, dict[str, Any]]:
            user_msg = body["messages"][-1]["content"]
            if "EXIT_OPENED" in user_msg or "exit_opened" in user_msg:
                directive = {
                    "subgoal": "walk out open exit",
                    "role_overrides": {
                        "cube": "m2_cube_1",
                        "floor_button": "m2_floor_btn_0",
                        "pedestal_button": None,
                        "exit": "m2_exit_0",
                    },
                    "memory_note": None,
                    "min_dwell_steps": 8,
                }
            elif "cube_fizzled (m2_cube_0)" in user_msg:
                directive = {
                    "subgoal": "retrieve m2_cube_1 across pit",
                    "role_overrides": {
                        "cube": "m2_cube_1",
                        "floor_button": "m2_floor_btn_0",
                        "pedestal_button": None,
                        "exit": "m2_exit_0",
                    },
                    "memory_note": "avoid m2_cube_0 due to fizzler",
                    "min_dwell_steps": 12,
                }
            else:
                directive = {
                    "subgoal": "try local m2_cube_0 first",
                    "role_overrides": {
                        "cube": "m2_cube_0",
                        "floor_button": "m2_floor_btn_0",
                        "pedestal_button": None,
                        "exit": "m2_exit_0",
                    },
                    "memory_note": None,
                    "min_dwell_steps": 12,
                }
            content = f"<think>Selecting subgoal: {directive['subgoal']}</think>\n" + json.dumps(
                directive
            )
            return 200, {"choices": [{"message": {"content": content}}]}

        tree = parse_bt(CHAMPION_BT_PATH.read_text(encoding="utf-8"))
        env = MockPortalEnv()
        with _LoopbackOpenAIServer(handle) as server:
            planner = OpenAICompatibleLLMPlanner(
                base_url=server.url, model="lfm2.5-1.2b-thinking", timeout_s=3.0
            )
            agent = FourLayerAgent(BehaviorTreeEvaluator(tree), planner=planner)
            metrics = agent.run_episode(env, spec=MULTI_STAGE_BENCHMARKS["M2"])

            self.assertTrue(metrics.success)
            self.assertEqual(metrics.fallback_count, 0)
            self.assertEqual(metrics.validation_warnings, 0)
            self.assertEqual(metrics.planner_calls, 3)
            self.assertEqual(metrics.planner_triggers.get("cube_fizzled"), 1)
            # Confirm dynamic json_schema was actually sent to the server
            self.assertEqual(
                server.received_bodies[0]["response_format"]["type"], "json_schema"
            )

    def test_http_400_retries_without_json_schema(self) -> None:
        def handle(body: dict[str, Any], _call_idx: int) -> tuple[int, dict[str, Any]]:
            if "response_format" in body:
                return 400, {"error": "json_schema unsupported"}
            payload = {
                "subgoal": "retry_plain",
                "role_overrides": {"cube": "m1_cube_0"},
                "memory_note": None,
                "min_dwell_steps": 10,
            }
            return 200, {"choices": [{"message": {"content": json.dumps(payload)}}]}

        env = MockPortalEnv()
        snap = env.reset(spec=MULTI_STAGE_BENCHMARKS["M1"])
        with _LoopbackOpenAIServer(handle) as server:
            planner = OpenAICompatibleLLMPlanner(base_url=server.url, timeout_s=2.0)
            directive = planner.plan(snap, EpisodicMemory(), TriggerEvent.EPISODE_START)
            self.assertEqual(directive.subgoal, "retry_plain")
            self.assertEqual(planner.fallbacks, 0)
            self.assertEqual(len(server.received_bodies), 2)

    def test_unreachable_endpoint_falls_back_gracefully(self) -> None:
        env = MockPortalEnv()
        snap = env.reset(1)
        planner = OpenAICompatibleLLMPlanner(
            base_url="http://127.0.0.1:1/v1/chat/completions",
            timeout_s=0.3,
        )
        directive = planner.plan(snap, EpisodicMemory(), TriggerEvent.EPISODE_START)
        self.assertEqual(directive.subgoal, "reactive_passthrough")
        self.assertEqual(planner.fallbacks, 1)
        self.assertIsNotNone(planner.last_error)


class TestControlledLayer4Ablation(unittest.TestCase):
    def test_standard_suite_isolated_and_regression_free(self) -> None:
        # Multi-stage benchmarks must stay strictly outside CHAMBERS[1..7]
        self.assertEqual(sorted(CHAMBERS.keys()), [1, 2, 3, 4, 5, 6, 7])
        for spec in MULTI_STAGE_BENCHMARKS.values():
            self.assertNotIn(spec.chamber_id, CHAMBERS)

        tree = parse_bt(CHAMPION_BT_PATH.read_text(encoding="utf-8"))
        env = MockPortalEnv()
        for cid in sorted(CHAMBERS.keys()):
            m_3layer = FourLayerAgent(
                BehaviorTreeEvaluator(tree), NullStrategicPlanner()
            ).run_episode(env, chamber_id=cid)
            m_4layer = FourLayerAgent(
                BehaviorTreeEvaluator(tree), RuleBasedStrategicPlanner()
            ).run_episode(env, chamber_id=cid)
            self.assertTrue(m_3layer.success, f"3-layer failed chamber {cid}")
            self.assertTrue(m_4layer.success, f"4-layer failed chamber {cid}")
            self.assertEqual(m_3layer.steps, m_4layer.steps)
            self.assertEqual(m_3layer.planner_calls, 0)
            self.assertLessEqual(m_4layer.planner_calls, 2)

    def test_multistage_ablation_scripted_and_neural_2b(self) -> None:
        tree = parse_bt(CHAMPION_BT_PATH.read_text(encoding="utf-8"))
        hybrid = HybridMotorController(
            moveto_policy=NeuralMoveToPolicy.load(MOVETO_MLP_PATH),
            aim_policy=NeuralAimPolicy.load(AIM_MLP_PATH),
        )
        env = MockPortalEnv()

        for motor_label, motor in (("scripted", None), ("neural_2b", hybrid)):
            for key, spec in MULTI_STAGE_BENCHMARKS.items():
                m_3layer = FourLayerAgent(
                    BehaviorTreeEvaluator(tree),
                    NullStrategicPlanner(),
                    hybrid_motor=motor,
                ).run_episode(env, spec=spec)
                m_4layer = FourLayerAgent(
                    BehaviorTreeEvaluator(tree),
                    RuleBasedStrategicPlanner(),
                    hybrid_motor=motor,
                ).run_episode(env, spec=spec)

                self.assertFalse(
                    m_3layer.success,
                    f"Expected 3-layer baseline to fail {key} ({motor_label})",
                )
                self.assertTrue(
                    m_4layer.success,
                    f"Expected 4-layer agent to solve {key} ({motor_label})",
                )
                self.assertLessEqual(m_4layer.planner_calls, 5)
                self.assertEqual(m_4layer.fallback_count, 0)
                self.assertEqual(m_4layer.validation_warnings, 0)


if __name__ == "__main__":
    unittest.main()
