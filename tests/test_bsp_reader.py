"""Tests for Source BSP v20 entity and portalable-surface scanner."""

from __future__ import annotations

import struct
import unittest
from pathlib import Path

from portal_ml.vmf.bsp_reader import scan_bsp_bytes, scan_bsp_file

ROOT = Path(__file__).resolve().parents[1]


def _make_minimal_bsp(entities_text: str) -> bytes:
    """Build a valid minimal Source BSP v20 buffer with only Lump 0 populated."""
    ent_bytes = entities_text.encode("latin-1") + b"\x00"
    header = bytearray(1032)
    header[0:4] = b"VBSP"
    struct.pack_into("<i", header, 4, 20)
    # Lump 0 at offset 1032
    struct.pack_into("<iiii", header, 8, 1032, len(ent_bytes), 0, 0)
    return bytes(header) + ent_bytes


class TestBspReader(unittest.TestCase):
    def test_rejects_invalid_magic(self) -> None:
        with self.assertRaisesRegex(ValueError, "VBSP"):
            scan_bsp_bytes(b"NOT_A_BSP" * 200)

    def test_parses_minimal_bsp_entities(self) -> None:
        raw = _make_minimal_bsp(
            '{\n"classname" "info_player_start"\n"origin" "64 128 8"\n"angles" "0 90 0"\n}\n'
            '{\n"classname" "prop_physics"\n"targetname" "box"\n"origin" "192 256 24"\n"model" "models/props/metal_box.mdl"\n}\n'
        )
        res = scan_bsp_bytes(raw, map_name="synthetic")
        self.assertEqual(res.bsp_version, 20)
        self.assertEqual(len(res.entities), 2)
        self.assertEqual(res.entities[0].classname, "info_player_start")
        self.assertEqual(res.entities[1].targetname, "box")
        self.assertEqual(res.entities[1].model, "models/props/metal_box.mdl")

    def test_scans_compiled_seven_chamber_bsp_if_present(self) -> None:
        bsp_path = ROOT / "maps" / "portal_ml_5chambers.bsp"
        if not bsp_path.exists():
            self.skipTest("Compiled .bsp not present in CI")
        res = scan_bsp_file(bsp_path)
        self.assertEqual(res.bsp_version, 20)
        self.assertEqual(len(res.portal_surfaces), 9)
        # Verify exact opposite outward normals on Chamber 5 & Chamber 6 walls
        normals = {tuple(s.normal.to_list()) for s in res.portal_surfaces}
        self.assertIn((1.0, 0.0, 0.0), normals)
        self.assertIn((-1.0, 0.0, 0.0), normals)
        self.assertIn((0.0, 1.0, 0.0), normals)
        self.assertIn((0.0, -1.0, 0.0), normals)


if __name__ == "__main__":
    unittest.main()
