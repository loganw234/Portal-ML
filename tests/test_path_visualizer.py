"""Unit and integration tests for the zero-dependency SVG path visualizer."""

from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from portal_ml.bt.ast import parse_bt
from portal_ml.motor.hybrid import HybridMotorController
from portal_ml.motor.neural import NeuralAimPolicy, NeuralParkourPolicy
from portal_ml.motor.parkour_benchmarks import PARKOUR_BENCHMARKS
from portal_ml.planner import MULTI_STAGE_BENCHMARKS, NullStrategicPlanner, RuleBasedStrategicPlanner
from portal_ml.sim.chambers import CHAMBERS
from portal_ml.viz import (
    PanelSpec,
    load_jsonl_series,
    record_mock_series,
    render_combined_grid_svg,
    render_generic_campaign_svg,
    render_single_panel_svg,
)

ROOT = Path(__file__).resolve().parents[1]
CHAMPION_BT_PATH = ROOT / "champions" / "seed7_decoupled.bt"


class TestPathVisualizer(unittest.TestCase):
    def test_record_mock_and_render_single_and_combined_svg(self) -> None:
        tree = parse_bt(CHAMPION_BT_PATH.read_text(encoding="utf-8"))
        parkour = NeuralParkourPolicy.load(ROOT / "champions" / "parkour_mlp.json")
        aim = NeuralAimPolicy.load(ROOT / "champions" / "aim_mlp.json")
        pk_hybrid = HybridMotorController(moveto_policy=parkour, aim_policy=aim)

        s_m2_3l = record_mock_series(
            MULTI_STAGE_BENCHMARKS["M2"],
            tree,
            planner=NullStrategicPlanner(),
            label="3-Layer",
            color="#f43f5e",
            dashed=True,
        )
        s_m2_4l = record_mock_series(
            MULTI_STAGE_BENCHMARKS["M2"],
            tree,
            planner=RuleBasedStrategicPlanner(),
            label="4-Layer",
            color="#22d3ee",
        )
        self.assertFalse(s_m2_3l.success)
        self.assertTrue(s_m2_4l.success)
        self.assertTrue(any(ev[2] == "FIZZLED" for ev in s_m2_4l.events))

        s_p2 = record_mock_series(
            PARKOUR_BENCHMARKS["P2"],
            tree,
            hybrid_motor=pk_hybrid,
            label="Parkour",
            color="#34d399",
        )
        self.assertTrue(s_p2.success)
        self.assertTrue(any(not pt.on_ground for pt in s_p2.points))

        p_m2 = PanelSpec(
            title="M2 Test",
            subtitle="Fizzler recovery",
            chamber=MULTI_STAGE_BENCHMARKS["M2"],
            series=[s_m2_3l, s_m2_4l],
        )
        p_p2 = PanelSpec(
            title="P2 Test",
            subtitle="Hurdle & Acid",
            chamber=PARKOUR_BENCHMARKS["P2"],
            series=[s_p2],
        )

        single_svg = render_single_panel_svg(p_m2)
        root_single = ET.fromstring(single_svg)
        self.assertTrue(root_single.tag.endswith("svg"))
        self.assertIn("FIZZLED", single_svg)

        combined_svg = render_combined_grid_svg(
            [p_m2, p_p2],
            title="Combined Test",
            subtitle="Two panels",
            cols=2,
        )
        root_combined = ET.fromstring(combined_svg)
        self.assertTrue(root_combined.tag.endswith("svg"))
        self.assertIn("ACID", combined_svg)

    def test_load_live_jsonl_and_generic_campaign_svg(self) -> None:
        live_7ch = ROOT / "records" / "live_7ch_phase1_2026-11-26.jsonl"
        if live_7ch.exists():
            ser = load_jsonl_series(
                live_7ch,
                5,
                label="Live Ch5",
                color="#38bdf8",
                spawn_pos=CHAMBERS[5].player_spawn,
            )
            self.assertTrue(ser.success)
            self.assertGreater(len(ser.points), 20)
            self.assertGreaterEqual(len(ser.portals), 2)

        live_a02 = ROOT / "records" / "live_testchmb_a02_solve_2026-11-26.jsonl"
        if live_a02.exists():
            svg = render_generic_campaign_svg(live_a02)
            root = ET.fromstring(svg)
            self.assertTrue(root.tag.endswith("svg"))
            self.assertIn("SUNKEN PIT", svg)
            self.assertIn("ELEVATOR (PASS)", svg)


if __name__ == "__main__":
    unittest.main()
