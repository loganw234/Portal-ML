"""Generate individual and combined SVG path visualizations for Portal-ML.

Examples:
    # Generate README combined 3x3 showcase + live testchmb_a_02 SVG + individual SVGs:
    python scripts/render_paths.py

    # Render any custom live or mock JSONL trace:
    python scripts/render_paths.py --trace records/live_testchmb_a02_solve_2026-11-26.jsonl --out custom.svg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portal_ml.bt.ast import parse_bt
from portal_ml.motor.hybrid import HybridMotorController
from portal_ml.motor.neural import (
    NeuralAimPolicy,
    NeuralMoveToPolicy,
    NeuralParkourPolicy,
)
from portal_ml.motor.parkour_benchmarks import PARKOUR_BENCHMARKS
from portal_ml.planner import (
    MULTI_STAGE_BENCHMARKS,
    NullStrategicPlanner,
    RuleBasedStrategicPlanner,
)
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


def build_showcase_panels() -> list[PanelSpec]:
    tree = parse_bt((ROOT / "champions" / "seed7_decoupled.bt").read_text(encoding="utf-8"))
    moveto = NeuralMoveToPolicy.load(ROOT / "champions" / "moveto_mlp.json")
    aim = NeuralAimPolicy.load(ROOT / "champions" / "aim_mlp.json")
    parkour = NeuralParkourPolicy.load(ROOT / "champions" / "parkour_mlp.json")

    blind_hybrid = HybridMotorController(moveto_policy=moveto, aim_policy=aim)
    parkour_hybrid = HybridMotorController(moveto_policy=parkour, aim_policy=aim)

    live_p1_trace = ROOT / "records" / "live_7ch_phase1_2026-11-26.jsonl"
    live_p2b_trace = ROOT / "records" / "live_7ch_phase2b_2026-11-26.jsonl"

    panels: list[PanelSpec] = []

    # Row 1: Live Portal 1 Held-Out Generalization (Chambers 5, 6, 7)
    row1_meta = [
        (5, "Ch 5: Held-Out 90 deg Rotated", "Live Portal 1 | Cube across pit, button & exit at spawn"),
        (6, "Ch 6: Held-Out Button Across Gap", "Live Portal 1 | Opposite asymmetry + opposite walls"),
        (7, "Ch 7: Held-Out 3-Zone Relay", "Live Portal 1 | 3 islands, 2 pits, dynamic portal re-link"),
    ]
    for cid, title, sub in row1_meta:
        spec = CHAMBERS[cid]
        if live_p1_trace.exists() and live_p2b_trace.exists():
            s_p1 = load_jsonl_series(
                live_p1_trace,
                cid,
                label="Phase 1 Scripted",
                color="#38bdf8",
                spawn_pos=spec.player_spawn,
            )
            s_p2b = load_jsonl_series(
                live_p2b_trace,
                cid,
                label="Phase 2B Neural",
                color="#f59e0b",
                spawn_pos=spec.player_spawn,
            )
        else:
            s_p1 = record_mock_series(spec, tree, label="Phase 1 Scripted", color="#38bdf8")
            s_p2b = record_mock_series(
                spec, tree, hybrid_motor=blind_hybrid, label="Phase 2B Neural", color="#f59e0b"
            )
        panels.append(PanelSpec(title=title, subtitle=sub, chamber=spec, series=[s_p1, s_p2b]))

    # Row 2: Layer-4 Event-Driven Strategic Memory (M1, M2, M3)
    row2_meta = [
        ("M1", "M1: Dual-Button Relay", "2 cubes + 2 required buttons across pit | 3-Layer vs 4-Layer"),
        ("M2", "M2: Fizzler-Gated Decoy Cube", "Decoy cube behind fizzler -> episodic recovery to cube 1"),
        ("M3", "M3: 3-Stage Compound Vault", "2 cubes + 2 floor buttons + timed pedestal across 2 pits"),
    ]
    for key, title, sub in row2_meta:
        spec = MULTI_STAGE_BENCHMARKS[key]
        s_3l = record_mock_series(
            spec,
            tree,
            planner=NullStrategicPlanner(),
            label="3-Layer",
            color="#f43f5e",
            dashed=True,
        )
        s_4l = record_mock_series(
            spec,
            tree,
            planner=RuleBasedStrategicPlanner(),
            label="4-Layer",
            color="#22d3ee",
        )
        panels.append(PanelSpec(title=title, subtitle=sub, chamber=spec, series=[s_3l, s_4l]))

    # Row 3: Milestone 7 Sensor-Equipped Neural Physical Skills (P1, P2, P3)
    row3_meta = [
        ("P1", "P1: Partition Slalom", "Tall wall blocks direct path | Local wall-ray steering"),
        ("P2", "P2: Hurdle & Acid Trench", "28u hurdle + lethal acid trench | Timed jumping (green rings)"),
        ("P3", "P3: Gap Jump & Timed Vault", "72u pit jump + timed pedestal + acid jump | BT + NeuralParkour"),
    ]
    for key, title, sub in row3_meta:
        spec = PARKOUR_BENCHMARKS[key]
        s_blind = record_mock_series(
            spec,
            tree,
            hybrid_motor=blind_hybrid,
            label="Blind 83w",
            color="#f43f5e",
            dashed=True,
        )
        s_pk = record_mock_series(
            spec,
            tree,
            hybrid_motor=parkour_hybrid,
            label="Parkour 208w",
            color="#34d399",
        )
        panels.append(PanelSpec(title=title, subtitle=sub, chamber=spec, series=[s_blind, s_pk]))

    return panels


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Portal-ML SVG path visualizations.")
    parser.add_argument(
        "--trace",
        default="",
        help="Optional path to a specific JSONL trace file to visualize",
    )
    parser.add_argument(
        "--chamber",
        type=int,
        default=0,
        help="Chamber ID inside --trace (0 for generic campaign maps like testchmb_a_02)",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output SVG path (for --trace mode) or output directory (default: docs/images)",
    )
    args = parser.parse_args()

    if args.trace:
        trace_path = Path(args.trace)
        out_path = Path(args.out) if args.out else Path("docs/images/custom_trace.svg")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.chamber in CHAMBERS:
            spec = CHAMBERS[args.chamber]
            ser = load_jsonl_series(
                trace_path,
                args.chamber,
                label=trace_path.stem,
                color="#38bdf8",
                spawn_pos=spec.player_spawn,
            )
            svg = render_single_panel_svg(
                PanelSpec(
                    title=f"Chamber {args.chamber}: {spec.name}",
                    subtitle=f"Trace: {trace_path.name}",
                    chamber=spec,
                    series=[ser],
                )
            )
        else:
            svg = render_generic_campaign_svg(
                trace_path,
                title=f"Portal-ML Trajectory — {trace_path.stem}",
                subtitle=f"Source trace: {trace_path.name} | Top-down XY + Z-elevation profile",
            )
        out_path.write_text(svg, encoding="utf-8")
        print(f"Wrote {out_path} ({len(svg):,} bytes)")
        return 0

    out_dir = Path(args.out) if args.out else (ROOT / "docs" / "images")
    ch_dir = out_dir / "chambers"
    ch_dir.mkdir(parents=True, exist_ok=True)

    panels = build_showcase_panels()

    # 1. Combined 3x3 Hero Overview SVG
    combined_svg = render_combined_grid_svg(
        panels,
        title="Portal-ML — Trajectory & Approach Comparisons Across All 4 Layers",
        subtitle=(
            "Frozen 24-node Behavior Tree (seed7_decoupled.bt, evolved strictly on mock Ch 1-4) "
            "compared across Live Portal 1 transfer, Layer-4 Strategic Memory, and Sensor-Equipped Neural Parkour"
        ),
        row_headers=[
            "ROW 1 — ZERO-SHOT LIVE PORTAL 1 HELD-OUT TRANSFER (Phase 1 Scripted vs Phase 2B Neural 83w+44w)",
            "ROW 2 — LAYER-4 STRATEGIC MEMORY & MULTI-STAGE PLANNING (Stateless 3-Layer FAIL vs 4-Layer PASS)",
            "ROW 3 — MILESTONE 7 PHYSICAL CONNECTIONS & NEURAL SKILLS (Blind MoveTo 83w FAIL vs Sensor Parkour 208w PASS)",
        ],
        cols=3,
        panel_w=490,
        panel_h=330,
    )
    combined_path = out_dir / "portal_ml_paths_combined.svg"
    combined_path.write_text(combined_svg, encoding="utf-8")
    print(f"Wrote {combined_path} ({len(combined_svg):,} bytes)")

    # 2. Individual per-chamber SVGs
    slugs = ["ch5", "ch6", "ch7", "m1", "m2", "m3", "p1", "p2", "p3"]
    for slug, panel in zip(slugs, panels):
        svg = render_single_panel_svg(panel, width=760, height=460)
        p = ch_dir / f"{slug}.svg"
        p.write_text(svg, encoding="utf-8")
    print(f"Wrote {len(slugs)} individual chamber SVGs -> {ch_dir}")

    # 3. Live Valve testchmb_a_02 zero-shot solve SVG
    a02_trace = ROOT / "records" / "live_testchmb_a02_solve_2026-11-26.jsonl"
    if a02_trace.exists():
        a02_svg = render_generic_campaign_svg(a02_trace)
        a02_path = out_dir / "live_testchmb_a02_solve.svg"
        a02_path.write_text(a02_svg, encoding="utf-8")
        print(f"Wrote {a02_path} ({len(a02_svg):,} bytes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
