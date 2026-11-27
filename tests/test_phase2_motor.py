"""Tests for Phase-2 neuroevolved neural motor primitives and hybrid BT execution."""

from __future__ import annotations

import math
import random
import tempfile
import unittest
from pathlib import Path

from portal_ml.bt import BehaviorTreeEvaluator, parse_bt
from portal_ml.motor import (
    AIM_INPUT_NAMES,
    AIM_OUTPUT_NAMES,
    HybridMotorController,
    MLP,
    MOVETO_INPUT_NAMES,
    MOVETO_OUTPUT_NAMES,
    MotorCommand,
    MoveVector,
    NeuralAimPolicy,
    NeuralMoveToPolicy,
)
from portal_ml.sim import MockPortalEnv

ROOT = Path(__file__).resolve().parents[1]


def _ref_forward(net: MLP, x: list[float]) -> list[float]:
    """Independent reference forward pass (checks flat row-major weight layout)."""
    n_in, n_hid, n_out = net.input_size, net.hidden_size, net.output_size
    w = net.weights
    h = [
        math.tanh(
            w[n_hid * n_in + j] + sum(w[j * n_in + i] * x[i] for i in range(n_in))
        )
        for j in range(n_hid)
    ]
    base2 = n_hid * n_in + n_hid
    return [
        math.tanh(
            w[base2 + n_out * n_hid + k]
            + sum(w[base2 + k * n_hid + j] * h[j] for j in range(n_hid))
        )
        for k in range(n_out)
    ]


class TestPhase2Motor(unittest.TestCase):
    def test_mlp_forward_matches_independent_reference(self) -> None:
        rng = random.Random(99)
        net = MLP.random(
            6, 8, 3, rng, input_names=MOVETO_INPUT_NAMES, output_names=MOVETO_OUTPUT_NAMES
        )
        for _ in range(10):
            x = [rng.uniform(-1.0, 1.0) for _ in range(6)]
            got = net.forward(x)
            ref = _ref_forward(net, x)
            for a, b in zip(got, ref):
                self.assertAlmostEqual(a, b, places=12)

    def test_checkpoint_roundtrip_and_schema_rejection(self) -> None:
        rng = random.Random(12)
        net = MLP.random(
            6, 8, 3, rng, input_names=MOVETO_INPUT_NAMES, output_names=MOVETO_OUTPUT_NAMES
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = net.save(Path(tmp) / "m.json")
            loaded = MLP.load(
                p,
                expected_inputs=MOVETO_INPUT_NAMES,
                expected_outputs=MOVETO_OUTPUT_NAMES,
            )
            self.assertEqual(len(loaded.weights), 83)
            # Refuses wrong input schema by name
            with self.assertRaisesRegex(ValueError, "CHECKPOINT-INPUTS"):
                MLP.load(p, expected_inputs=AIM_INPUT_NAMES)
            # Refuses wrong output schema by name
            with self.assertRaisesRegex(ValueError, "CHECKPOINT-OUTPUTS"):
                MLP.load(p, expected_outputs=AIM_OUTPUT_NAMES)

    def test_move_vector_wire_roundtrip(self) -> None:
        cmd = MoveVector(forward=0.85, strafe=-0.25, yaw_delta=0.4)
        restored = MotorCommand.from_dict(cmd.to_dict())
        self.assertEqual(restored, cmd)

    def test_frozen_bt_passes_all_chambers_with_phase2_neural_primitives(self) -> None:
        bt_path = ROOT / "champions" / "seed7_decoupled.bt"
        mt_path = ROOT / "champions" / "moveto_mlp.json"
        aim_path = ROOT / "champions" / "aim_mlp.json"
        tree = parse_bt(bt_path.read_text(encoding="utf-8"))
        moveto = NeuralMoveToPolicy.load(mt_path)
        aim = NeuralAimPolicy.load(aim_path)

        env = MockPortalEnv()
        for hybrid in (
            HybridMotorController(moveto_policy=moveto),
            HybridMotorController(moveto_policy=moveto, aim_policy=aim),
        ):
            for cid in (1, 2, 3, 4, 5, 6, 7):
                ev = BehaviorTreeEvaluator(tree)
                snap = env.reset(cid)
                while not snap.done:
                    sym_cmd = ev.tick(snap).command
                    low_cmd = hybrid.lower(snap, sym_cmd)
                    snap = env.step(low_cmd).snapshot
                self.assertTrue(snap.success, f"Phase 2 failed on Chamber {cid}")


if __name__ == "__main__":
    unittest.main()
