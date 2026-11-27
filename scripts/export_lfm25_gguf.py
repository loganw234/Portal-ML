"""Export a merged HuggingFace LFM2.5-1.2B Layer-4 checkpoint to Q8_0 GGUF for LM Studio.

Uses llama.cpp's sparse Python converter (zero C++ build required) to quantize
`champions/lfm25_layer4_lora/merged` directly into a ~1.25 GB `Q8_0` GGUF and
optionally imports/loads it in LM Studio via the `lms` CLI.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _ensure_converter(conv_dir: Path) -> Path:
    script = conv_dir / "convert_hf_to_gguf.py"
    if script.exists():
        return script
    conv_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "https://github.com/ggml-org/llama.cpp.git",
            str(conv_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(conv_dir),
            "sparse-checkout",
            "set",
            "--no-cone",
            "/convert_hf_to_gguf.py",
            "/conversion/",
            "/gguf-py/",
        ],
        check=True,
    )
    return script


def main() -> int:
    parser = argparse.ArgumentParser(description="Export merged LFM2.5-1.2B LoRA to GGUF.")
    parser.add_argument("--merged-dir", default="champions/lfm25_layer4_lora/merged")
    parser.add_argument(
        "--out-gguf",
        default="champions/lfm25_layer4_lora/lfm25-1.2b-portal-ml-layer4-Q8_0.gguf",
    )
    parser.add_argument("--outtype", default="q8_0", choices=("f16", "bf16", "q8_0"))
    parser.add_argument("--converter-dir", default="champions/lfm25_layer4_lora/llama_conv")
    parser.add_argument(
        "--lms-import",
        action="store_true",
        help="Import the exported GGUF into LM Studio via `lms import`",
    )
    parser.add_argument("--user-repo", default="portal-ml/lfm25-1.2b-layer4")
    args = parser.parse_args()

    merged_dir = Path(args.merged_dir)
    if not (merged_dir / "config.json").exists():
        print(f"ERROR: Merged HF checkpoint not found at {merged_dir}")
        print("Run scripts/train_lfm25_lora.py --merge first.")
        return 2

    out_gguf = Path(args.out_gguf)
    out_gguf.parent.mkdir(parents=True, exist_ok=True)

    conv_dir = Path(args.converter_dir)
    script = _ensure_converter(conv_dir)

    env = dict(os.environ)
    gguf_py = str((conv_dir / "gguf-py").resolve())
    env["PYTHONPATH"] = gguf_py + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    cmd = [
        sys.executable,
        str(script),
        str(merged_dir),
        "--outtype",
        args.outtype,
        "--outfile",
        str(out_gguf),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)
    print(f"Exported GGUF -> {out_gguf} ({out_gguf.stat().st_size / (1024**3):.2f} GiB)")

    if args.lms_import:
        lms = shutil.which("lms")
        if not lms:
            print("WARNING: `lms` CLI not found on PATH; import the GGUF manually in LM Studio.")
            return 0
        subprocess.run(
            [lms, "import", "-c", "-y", "--user-repo", args.user_repo, str(out_gguf)],
            check=True,
        )
        print("Imported into LM Studio. Load with:")
        print("  lms load lfm25-1.2b-layer4 --context-length 4096 --gpu max -y")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
