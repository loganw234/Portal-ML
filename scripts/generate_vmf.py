"""Generate the 5-chamber single-.vmf map and vbsp/vvis/vrad compile scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portal_ml.vmf.generator import write_compile_scripts, write_five_chamber_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate portal_ml_5chambers.vmf containing Chambers 1-5 separated by void."
    )
    parser.add_argument("--out-dir", default="maps", help="Output directory")
    parser.add_argument("--stem", default="portal_ml_5chambers", help="Map filename stem")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    vmf_path = write_five_chamber_map(out_dir / f"{args.stem}.vmf")
    bat_path, sh_path = write_compile_scripts(out_dir, map_stem=args.stem)
    print(f"Generated VMF:           {vmf_path} ({vmf_path.stat().st_size:,} bytes)")
    print(f"Generated Windows build: {bat_path}")
    print(f"Generated POSIX build:   {sh_path}")


if __name__ == "__main__":
    main()
