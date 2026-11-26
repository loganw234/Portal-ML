"""Tests for the programmatic 5-chamber .vmf generator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from portal_ml.vmf import (
    build_five_chamber_vmf,
    write_compile_scripts,
    write_five_chamber_map,
)


class TestVMFGenerator(unittest.TestCase):
    def test_vmf_contains_all_five_chambers_and_balanced_braces(self) -> None:
        doc = build_five_chamber_vmf()
        text = doc.to_vmf()

        # Balanced KeyValues braces
        self.assertEqual(text.count("{"), text.count("}"))
        self.assertIn('"classname" "worldspawn"', text)
        self.assertIn('"classname" "info_player_start"', text)
        self.assertIn('"classname" "weapon_portalgun"', text)
        self.assertIn('"classname" "func_weight_button"', text)
        self.assertIn('"classname" "func_button"', text)
        self.assertIn('"classname" "func_door"', text)
        self.assertIn('"classname" "prop_portal"', text)

        # Verify named entities from all 5 chambers exist in the single map
        for expected in (
            "c1_spawn",
            "c1_exit_0",
            "c2_pedestal_0",
            "c2_exit_0",
            "c3_cube_0",
            "c3_floor_btn_0",
            "c4_wall_a",
            "c4_wall_b",
            "c4_pit_0",
            "c5_cube_0",
            "c5_floor_btn_0",
            "c5_wall_a",
            "c5_wall_b",
            "c5_pit_0",
            "CONCRETE/CONCRETE_MODULAR_WALL001A",
            "METAL/METALWALL048B",
        ):
            self.assertIn(expected, text, f"Missing {expected} in generated .vmf")

    def test_write_map_and_compile_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            vmf_path = write_five_chamber_map(tmp_dir / "test_5ch.vmf")
            bat, sh = write_compile_scripts(tmp_dir, map_stem="test_5ch")
            self.assertTrue(vmf_path.exists())
            self.assertGreater(vmf_path.stat().st_size, 10_000)
            self.assertIn("vbsp.exe", bat.read_text(encoding="utf-8"))
            self.assertIn("vrad", sh.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
