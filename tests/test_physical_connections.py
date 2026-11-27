"""Unit and integration tests for Milestone-7 physical connections & NeuralParkour skill."""

from __future__ import annotations

import unittest
from pathlib import Path

from portal_ml.bt.ast import parse_bt
from portal_ml.bt.evaluator import BehaviorTreeEvaluator
from portal_ml.motor.hybrid import HybridMotorController
from portal_ml.motor.neural import (
    MLP,
    PARKOUR_INPUT_NAMES,
    PARKOUR_OUTPUT_NAMES,
    NeuralAimPolicy,
    NeuralMoveToPolicy,
    NeuralParkourPolicy,
)
from portal_ml.motor.parkour_benchmarks import PARKOUR_BENCHMARKS
from portal_ml.motor.primitives import MotorCommand, MoveVector
from portal_ml.planner import FourLayerAgent, NullStrategicPlanner
from portal_ml.schema.blackboard import (
    BlackboardSnapshot,
    HazardVolume,
    LocalSensors,
    PlayerState,
    Vec3,
)
from portal_ml.sim.chambers import CHAMBERS
from portal_ml.sim.mock_env import MockPortalEnv

ROOT = Path(__file__).resolve().parents[1]
CHAMPION_BT_PATH = ROOT / "champions" / "seed7_decoupled.bt"
MOVETO_MLP_PATH = ROOT / "champions" / "moveto_mlp.json"
AIM_MLP_PATH = ROOT / "champions" / "aim_mlp.json"
PARKOUR_MLP_PATH = ROOT / "champions" / "parkour_mlp.json"


class TestPhysicalConnectionsSchemaAndPhysics(unittest.TestCase):
    def test_blackboard_proprioception_sensors_and_hazards_roundtrip(self) -> None:
        snap = BlackboardSnapshot(
            tick=7,
            dt=0.1,
            chamber_id=202,
            player=PlayerState(
                pos=Vec3(100.0, 128.0, 24.0),
                vel=Vec3(120.0, 0.0, 180.0),
                yaw=15.0,
                pitch=-6.0,
                zone_id=0,
                health=75.0,
                on_ground=False,
                ducking=True,
                water_level=2,
            ),
            sensors=LocalSensors(
                wall_rays=[0.5, 0.7, 1.0, 0.8, 0.25, 0.8, 1.0, 0.7],
                floor_probes=[1.0, 0.28, 0.28],
                step_clearance=1.0,
                hazard_ahead=0.65,
            ),
            hazards=[
                HazardVolume(
                    id="acid_0",
                    kind="acid",
                    min_pos=Vec3(300.0, 0.0, -32.0),
                    max_pos=Vec3(380.0, 256.0, 8.0),
                    damage_per_sec=250.0,
                )
            ],
        )
        restored = BlackboardSnapshot.from_json(snap.to_json())
        self.assertEqual(restored.player.health, 75.0)
        self.assertFalse(restored.player.on_ground)
        self.assertTrue(restored.player.ducking)
        self.assertEqual(restored.player.water_level, 2)
        self.assertEqual(len(restored.sensors.wall_rays), 8)
        self.assertAlmostEqual(restored.sensors.step_clearance, 1.0, places=3)
        self.assertAlmostEqual(restored.sensors.hazard_ahead, 0.65, places=3)
        self.assertEqual(len(restored.hazards), 1)
        self.assertEqual(restored.hazards[0].id, "acid_0")

    def test_full_move_vector_wire_roundtrip(self) -> None:
        cmd = MoveVector(
            forward=0.8,
            strafe=-0.25,
            yaw_delta=0.4,
            pitch_delta=-0.3,
            jump=True,
            crouch=True,
            use=True,
            fire_blue=True,
            fire_orange=False,
        )
        restored = MotorCommand.from_dict(cmd.to_dict())
        self.assertIsInstance(restored, MoveVector)
        assert isinstance(restored, MoveVector)
        self.assertAlmostEqual(restored.pitch_delta, -0.3, places=4)
        self.assertTrue(restored.jump)
        self.assertTrue(restored.crouch)
        self.assertTrue(restored.use)
        self.assertTrue(restored.fire_blue)
        self.assertFalse(restored.fire_orange)

    def test_mock_env_jump_arc_and_acid_damage_physics(self) -> None:
        env = MockPortalEnv()
        snap = env.reset(spec=PARKOUR_BENCHMARKS["P2"])
        self.assertEqual(snap.player.health, 100.0)
        self.assertTrue(snap.player.on_ground)
        self.assertEqual(len(snap.sensors.wall_rays), 8)

        # Jump from grounded state -> goes airborne and peaks near ~48u
        max_z = 0.0
        saw_airborne = False
        for i in range(10):
            res = env.step(MoveVector(forward=0.0, strafe=0.0, jump=(i == 0)))
            max_z = max(max_z, res.snapshot.player.pos.z)
            if not res.snapshot.player.on_ground:
                saw_airborne = True
        self.assertTrue(saw_airborne)
        self.assertGreater(max_z, 35.0)
        self.assertTrue(res.snapshot.player.on_ground)

        # Walking straight into the P2 hurdle blocks horizontal progress without jumping
        env.reset(spec=PARKOUR_BENCHMARKS["P2"])
        for _ in range(15):
            res = env.step(MoveVector(forward=1.0))
        self.assertLess(res.snapshot.player.pos.x, 196.0)
        self.assertAlmostEqual(res.snapshot.sensors.step_clearance, 1.0, places=2)

    def test_parkour_checkpoint_schema_and_ablation(self) -> None:
        self.assertTrue(PARKOUR_MLP_PATH.exists(), "champions/parkour_mlp.json missing")
        parkour = NeuralParkourPolicy.load(PARKOUR_MLP_PATH)
        self.assertEqual(parkour.net.input_names, PARKOUR_INPUT_NAMES)
        self.assertEqual(parkour.net.output_names, PARKOUR_OUTPUT_NAMES)
        self.assertEqual(len(parkour.net.weights), MLP.param_count(12, 12, 4))

        tree = parse_bt(CHAMPION_BT_PATH.read_text(encoding="utf-8"))
        aim = NeuralAimPolicy.load(AIM_MLP_PATH)
        blind = NeuralMoveToPolicy.load(MOVETO_MLP_PATH)
        env = MockPortalEnv()

        # Controlled ablation on P1-P3: blind NeuralMoveTo fails 0/3; sensor NeuralParkour solves 3/3
        blind_hybrid = HybridMotorController(moveto_policy=blind, aim_policy=aim)
        pk_hybrid = HybridMotorController(moveto_policy=parkour, aim_policy=aim)
        for key, spec in PARKOUR_BENCHMARKS.items():
            m_blind = FourLayerAgent(
                BehaviorTreeEvaluator(tree), NullStrategicPlanner(), hybrid_motor=blind_hybrid
            ).run_episode(env, spec=spec)
            m_pk = FourLayerAgent(
                BehaviorTreeEvaluator(tree), NullStrategicPlanner(), hybrid_motor=pk_hybrid
            ).run_episode(env, spec=spec)
            self.assertFalse(m_blind.success, f"Blind MoveTo should fail {key}")
            self.assertTrue(m_pk.success, f"NeuralParkour should solve {key}")

        # Zero regression on standard Chambers 1-7
        for cid in sorted(CHAMBERS.keys()):
            m = FourLayerAgent(
                BehaviorTreeEvaluator(tree), NullStrategicPlanner(), hybrid_motor=pk_hybrid
            ).run_episode(env, chamber_id=cid)
            self.assertTrue(m.success, f"NeuralParkour failed standard chamber {cid}")

    def test_multi_skill_auto_routing_and_distill_dataset(self) -> None:
        tree = parse_bt(CHAMPION_BT_PATH.read_text(encoding="utf-8"))
        aim = NeuralAimPolicy.load(AIM_MLP_PATH)
        moveto = NeuralMoveToPolicy.load(MOVETO_MLP_PATH)
        parkour = NeuralParkourPolicy.load(PARKOUR_MLP_PATH)
        multi = HybridMotorController(
            moveto_policy=moveto,
            parkour_policy=parkour,
            aim_policy=aim,
            skill_mode="auto",
        )
        env = MockPortalEnv()

        # Standard Chamber 1 stays in precision MoveTo (zero unnecessary parkour jumps)
        m_c1 = FourLayerAgent(
            BehaviorTreeEvaluator(tree), NullStrategicPlanner(), hybrid_motor=multi
        ).run_episode(env, chamber_id=1)
        self.assertTrue(m_c1.success)
        self.assertGreater(multi.precision_ticks, 0)
        self.assertEqual(multi.parkour_ticks, 0)

        # Held-out Parkour P1-P3 dynamically engage NeuralParkour and pass 3/3
        for key, spec in PARKOUR_BENCHMARKS.items():
            m_pk = FourLayerAgent(
                BehaviorTreeEvaluator(tree), NullStrategicPlanner(), hybrid_motor=multi
            ).run_episode(env, spec=spec)
            self.assertTrue(m_pk.success, f"Multi-skill auto failed {key}")
            self.assertGreater(multi.parkour_ticks, 0, f"Parkour did not engage on {key}")

        # Verify generated Layer-4 distillation JSONL has zero benchmark leakage
        import json

        train_path = ROOT / "records" / "layer4_distill_train.jsonl"
        self.assertTrue(train_path.exists())
        lines = train_path.read_text(encoding="utf-8").splitlines()
        self.assertGreaterEqual(len(lines), 100)
        for line in lines[:25]:
            obj = json.loads(line)
            msgs = obj["messages"]
            self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant"])
            user_txt = msgs[1]["content"]
            for forbidden in ("m1_", "m2_", "m3_", "c5_", "c6_", "c7_"):
                self.assertNotIn(forbidden, user_txt)
            act = json.loads(msgs[2]["content"])
            self.assertIn(act["action"], ("place_cube_on_button", "press_pedestal", "go_to_exit"))


if __name__ == "__main__":
    unittest.main()
