"""Inspect real Portal 1 campaign .bsp entity lumps (loose or inside portal_pak VPKs).

Audits gameplay entities across testchmb_a_00..a_15 and escape_00..02 to map out
which real campaign chambers can serve as advanced held-out test suites and what
new Blackboard/motor abstractions are required to progress through the full game.
"""

from __future__ import annotations

import argparse
import re
import struct
from collections import Counter
from pathlib import Path

KEY_CLASSES = (
    "prop_physics",
    "func_weight_button",
    "func_button",
    "prop_portal",
    "weapon_portalgun",
    "trigger_portal_cleanser",
    "point_energy_ball_launcher",
    "npc_portal_turret_floor",
    "npc_rocket_turret",
    "func_tracktrain",
    "func_movelinear",
    "func_door",
    "trigger_changelevel",
    "prop_glados_core",
    "trigger_once",
    "trigger_multiple",
)

CLASSNAME_RE = re.compile(r'"classname"\s+"([^"]+)"')


def _read_vpk_maps(portal_dir: Path) -> list[tuple[str, bytes]]:
    """Minimal Valve VPK v1/v2 reader for maps/*.bsp inside portal_pak_dir.vpk."""
    dir_vpk = portal_dir / "portal" / "portal_pak_dir.vpk"
    if not dir_vpk.exists():
        return []
    raw = dir_vpk.read_bytes()
    sig, ver, tree_size = struct.unpack_from("<III", raw, 0)
    if sig != 0x55AA1234:
        return []
    pos = 12 if ver == 1 else 28

    def read_cstr(buf: bytes, offset: int) -> tuple[str, int]:
        end = buf.index(0, offset)
        return buf[offset:end].decode("latin-1"), end + 1

    results: list[tuple[str, bytes]] = []
    tree_end = pos + tree_size
    while pos < tree_end:
        ext, pos = read_cstr(raw, pos)
        if not ext:
            break
        while True:
            directory, pos = read_cstr(raw, pos)
            if not directory:
                break
            while True:
                fname, pos = read_cstr(raw, pos)
                if not fname:
                    break
                crc, preload_bytes, archive_idx, entry_ofs, entry_len, term = struct.unpack_from(
                    "<IHHIIH", raw, pos
                )
                pos += 18
                preload = raw[pos : pos + preload_bytes]
                pos += preload_bytes
                if ext == "bsp" and directory == "maps" and fname.startswith(("testchmb_", "escape_")):
                    if archive_idx == 0x7FFF:
                        data = preload + raw[tree_end + entry_ofs : tree_end + entry_ofs + entry_len]
                    else:
                        arch = portal_dir / "portal" / f"portal_pak_{archive_idx:03d}.vpk"
                        with arch.open("rb") as fp:
                            fp.seek(entry_ofs)
                            data = preload + fp.read(entry_len)
                    results.append((fname, data))
    return sorted(results, key=lambda x: x[0])


def audit_bsp(name: str, data: bytes) -> dict[str, int]:
    ident, ver = struct.unpack_from("<4sI", data, 0)
    if ident != b"VBSP":
        return {}
    ent_ofs, ent_len = struct.unpack_from("<II", data, 8)
    ent_text = data[ent_ofs : ent_ofs + ent_len].decode("latin-1", errors="ignore")
    counts = Counter(CLASSNAME_RE.findall(ent_text))
    return {k: counts[k] for k in KEY_CLASSES if counts[k]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit real Portal 1 campaign BSP entity lumps.")
    parser.add_argument(
        "--portal-dir",
        default=r"C:\Program Files (x86)\Steam\steamapps\common\Portal",
        help="Path to Portal 1 install root",
    )
    args = parser.parse_args()
    pdir = Path(args.portal_dir)

    maps: list[tuple[str, bytes]] = []
    loose_dir = pdir / "portal" / "maps"
    if loose_dir.exists():
        for bsp in sorted(loose_dir.glob("*.bsp")):
            if bsp.stem.startswith(("testchmb_", "escape_")):
                maps.append((bsp.stem, bsp.read_bytes()))
    if not maps:
        maps = _read_vpk_maps(pdir)

    print(f"Found {len(maps)} Portal 1 campaign maps:")
    for name, data in maps:
        summary = audit_bsp(name, data)
        # Compact display of key puzzle mechanics
        tags = ", ".join(f"{k}={v}" for k, v in summary.items())
        print(f"  {name:16s} | {tags}")


if __name__ == "__main__":
    main()
